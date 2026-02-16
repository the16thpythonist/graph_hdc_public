"""
Experiment helper utilities for FlowEdgeDecoder training and testing.

Helper functions and callbacks extracted from experiment files to enable
code reuse across base and child experiments via PyComex inheritance.
"""

from __future__ import annotations

import math
import signal
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from PIL import Image, ImageDraw
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from torch import Tensor
from torch_geometric.data import Data

from graph_hdc.hypernet.configs import (
    DSHDCConfig,
    FeatureConfig,
    Features,
    IndexRange,
)
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.hypernet.feature_encoders import CombinatoricIntegerEncoder
from graph_hdc.hypernet.types import VSAModel
from graph_hdc.datasets.zinc_smiles import mol_to_data as zinc_mol_to_data
from graph_hdc.models.flow_edge_decoder import (
    ZINC_ATOM_TYPES,
    NUM_ATOM_CLASSES,
    NODE_FEATURE_DIM,
    FlowEdgeDecoder,
    onehot_to_raw_features,
)
from graph_hdc.utils.helpers import scatter_hd

if TYPE_CHECKING:
    from pycomex.functional.experiment import Experiment


# ========= HDC Configuration =========


def create_hdc_config(
    dataset: str,
    hv_dim: int,
    depth: int,
    device: str = "cpu",
) -> DSHDCConfig:
    """
    Create a HyperNet configuration from parameters.

    Args:
        dataset: Dataset name ("qm9" or "zinc")
        hv_dim: Hypervector dimension
        depth: Message passing depth
        device: Device string

    Returns:
        DSHDCConfig for HyperNet initialization
    """
    dataset = dataset.lower()

    if dataset == "qm9":
        # QM9: 4 atom types, 5 degrees, 3 charges, 5 hydrogens
        node_feature_config = FeatureConfig(
            count=math.prod([4, 5, 3, 5]),  # 300 combinations
            encoder_cls=CombinatoricIntegerEncoder,
            index_range=IndexRange((0, 4)),
            bins=[4, 5, 3, 5],
        )
        base_dataset = "qm9"
    elif dataset == "zinc":
        # ZINC: 9 atom types, 6 degrees, 3 charges, 4 hydrogens, 2 ring flags
        node_feature_config = FeatureConfig(
            count=math.prod([9, 6, 3, 4, 2]),  # 1296 combinations
            encoder_cls=CombinatoricIntegerEncoder,
            index_range=IndexRange((0, 5)),
            bins=[9, 6, 3, 4, 2],
        )
        base_dataset = "zinc"
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: qm9, zinc")

    return DSHDCConfig(
        name=f"{dataset.upper()}_HRR_{hv_dim}_depth{depth}",
        hv_dim=hv_dim,
        vsa=VSAModel.HRR,
        base_dataset=base_dataset,
        hypernet_depth=depth,
        device=device,
        seed=42,
        normalize=True,
        dtype="float64",
        node_feature_configs=OrderedDict([
            (Features.NODE_FEATURES, node_feature_config),
        ]),
    )


def load_or_create_encoder(
    config_path: str,
    dataset: str,
    hv_dim: int,
    depth: int,
    device: torch.device,
    rw_config=None,
    prune_codebook: bool = True,
) -> HyperNet:
    """
    Load encoder from checkpoint or create a new one.

    Args:
        config_path: Path to saved encoder checkpoint (empty string to create new)
        dataset: Dataset name
        hv_dim: Hypervector dimension
        depth: Message passing depth
        device: Device to load to
        rw_config: Optional RWConfig for random walk features.
                   When enabled and creating a new encoder, the dataset
                   is scanned for observed RW-augmented feature tuples.
        prune_codebook: Whether to prune the codebook to observed tuples.

    Returns:
        HyperNet encoder instance (in eval mode)
    """
    if config_path and Path(config_path).exists():
        hypernet = HyperNet.load(config_path, device=str(device))
    elif rw_config is not None and rw_config.enabled:
        from graph_hdc.datasets.utils import scan_node_features_with_rw
        from graph_hdc.hypernet.configs import create_config_with_rw

        observed = scan_node_features_with_rw(dataset.lower(), rw_config) if prune_codebook else None
        config = create_config_with_rw(
            dataset.lower(), hv_dim, rw_config=rw_config, hypernet_depth=depth,
            prune_codebook=prune_codebook,
        )
        hypernet = HyperNet(config, observed_node_features=observed)
        hypernet = hypernet.to(device)
    else:
        config = create_hdc_config(dataset, hv_dim, depth, device=str(device))
        config.prune_codebook = prune_codebook
        hypernet = HyperNet(config)
        hypernet = hypernet.to(device)

    hypernet.eval()
    return hypernet


# ========= HDC Decoding =========


def decode_nodes_from_hdc(
    hypernet: HyperNet,
    hdc_vector: Tensor,
    base_hdc_dim: int,
    debug: bool = False,
) -> Tuple[List[Tuple[int, ...]], int] | Tuple[List[Tuple[int, ...]], int, List[float], List[float]]:
    """
    Decode node tuples from HDC embedding.

    The hdc_vector is a concatenation of [order_0 | order_N], where:
    - order_0: Bundled node hypervectors (first base_hdc_dim dimensions)
    - order_N: Graph embedding after message passing (last base_hdc_dim dimensions)

    Node decoding uses the order_0 part since it contains only node information
    without structural noise from message passing.

    Args:
        hypernet: HyperNet encoder instance
        hdc_vector: Concatenated HDC vector (2 * base_hdc_dim,) or (batch, 2 * base_hdc_dim)
        base_hdc_dim: Base hypervector dimension (hdc_vector has 2x this)
        debug: If True, also return residual norms and cosine similarities
            from each iteration of the decoding loop.

    Returns:
        Tuple of (list of node tuples, number of nodes)
        Each tuple is (atom_idx, degree, charge, num_hs, is_ring)
        If debug=True, additionally returns (norms_history, similarities)
    """
    # Extract order_0 from the first half of concatenated embedding
    if hdc_vector.dim() == 1:
        order_zero = hdc_vector[:base_hdc_dim].unsqueeze(0)
    else:
        order_zero = hdc_vector[:, :base_hdc_dim]

    # Ensure order_zero matches the codebook's VSATensor subclass and dtype
    # so torchhd.cos() works.  Plain float32 tensors from the flow model
    # would otherwise cause type/dtype mismatches with the float64 HRRTensor
    # codebook.
    cb = hypernet.nodes_codebook
    if order_zero.dtype != cb.dtype:
        order_zero = order_zero.to(dtype=cb.dtype)
    if type(order_zero) is not type(cb):
        order_zero = order_zero.as_subclass(type(cb))

    if debug:
        # Call iterative decoder directly with debug=True for a single embedding
        result = hypernet.decode_order_zero_iterative(order_zero[0], debug=True)
        decoded_nodes, norms_history, similarities = result

        node_tuples = list(decoded_nodes)
        return node_tuples, len(node_tuples), norms_history, similarities

    # Decode node counts from order-0 embedding using iterative unbinding
    # Returns dict: {batch_idx: Counter({node_tuple: count, ...})}
    node_counter_dict = hypernet.decode_order_zero_counter_iterative(order_zero)

    # Get counter for batch 0 (single embedding)
    node_counter = node_counter_dict.get(0, Counter())

    # Convert to list of node tuples (expand by count)
    node_tuples = []
    for node_tuple, count in node_counter.items():
        # node_tuple is (atom_idx, degree, charge, num_hs, is_ring)
        node_tuples.extend([node_tuple] * count)

    return node_tuples, len(node_tuples)


# ========= Molecule Conversion =========

# ZINC atom type mapping (index -> symbol)
ZINC_IDX_TO_ATOM = {0: "Br", 1: "C", 2: "Cl", 3: "F", 4: "I", 5: "N", 6: "O", 7: "P", 8: "S"}


def pyg_to_mol(data: Data) -> Optional[Chem.Mol]:
    """
    Convert FlowEdgeDecoder output (PyG Data) to RDKit Mol.

    Expects 24-dim one-hot node features where first 9 dims are atom type.
    Edge classes: 0=no-edge, 1=single, 2=double, 3=triple, 4=aromatic

    Args:
        data: PyG Data from FlowEdgeDecoder.sample()

    Returns:
        RDKit Mol or None if conversion fails
    """
    try:
        mol = Chem.RWMol()

        # Get atom types from first 9 dimensions of 24-dim one-hot
        if data.x.dim() > 1:
            atom_type_probs = data.x[:, :NUM_ATOM_CLASSES]
            node_types = atom_type_probs.argmax(dim=-1).cpu().numpy()
        else:
            node_types = data.x.cpu().numpy()

        # Add atoms
        atom_map = {}
        for i, atom_type_idx in enumerate(node_types):
            atom_symbol = ZINC_IDX_TO_ATOM[int(atom_type_idx)]
            atom = Chem.Atom(atom_symbol)
            rdkit_idx = mol.AddAtom(atom)
            atom_map[i] = rdkit_idx

        # Get edge info
        if data.edge_index is None or data.edge_index.numel() == 0:
            return mol.GetMol()

        edge_index = data.edge_index.cpu().numpy()
        if data.edge_attr is not None:
            if data.edge_attr.dim() > 1:
                edge_types = data.edge_attr.argmax(dim=-1).cpu().numpy()
            else:
                edge_types = data.edge_attr.cpu().numpy()
        else:
            edge_types = [1] * edge_index.shape[1]

        bond_type_map = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
            4: Chem.BondType.AROMATIC,
        }

        added_bonds = set()
        for k in range(edge_index.shape[1]):
            i, j = int(edge_index[0, k]), int(edge_index[1, k])
            edge_type = int(edge_types[k])

            if edge_type == 0 or i == j:
                continue

            bond_key = (min(i, j), max(i, j))
            if bond_key in added_bonds:
                continue
            added_bonds.add(bond_key)

            if edge_type in bond_type_map:
                mol.AddBond(atom_map[i], atom_map[j], bond_type_map[edge_type])

        mol = mol.GetMol()

        # Try full sanitization first
        try:
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            pass

        # If full sanitization fails, try partial sanitization
        try:
            Chem.FastFindRings(mol)
            mol.UpdatePropertyCache(strict=False)
            return mol
        except Exception:
            pass

        # If even partial init fails, return None
        return None

    except Exception:
        return None


def scrub_smiles(smiles: str) -> Optional[str]:
    """
    Clean a SMILES string by neutralizing formal charges, removing
    stereochemistry, and stripping explicit hydrogens.

    Args:
        smiles: Input SMILES string.

    Returns:
        Canonical scrubbed SMILES, or None on failure.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Remove stereochemistry (E/Z, R/S)
        Chem.RemoveStereochemistry(mol)

        # Strip explicit hydrogens
        mol = Chem.RemoveAllHs(mol)

        # Neutralize formal charges
        from rdkit.Chem.MolStandardize import rdMolStandardize
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def is_valid_mol(mol: Optional[Chem.Mol]) -> bool:
    """
    Check if molecule is valid (not None and has valid SMILES).

    Args:
        mol: RDKit molecule or None

    Returns:
        True if valid, False otherwise
    """
    if mol is None:
        return False
    try:
        smiles = Chem.MolToSmiles(mol)
        return smiles is not None and len(smiles) > 0
    except Exception:
        return False


def get_canonical_smiles(mol: Optional[Chem.Mol]) -> Optional[str]:
    """
    Get canonical SMILES from molecule.

    Args:
        mol: RDKit molecule or None

    Returns:
        Canonical SMILES string or None if conversion fails
    """
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


# ========= Visualization =========


def draw_mol_or_error(
    mol: Optional[Chem.Mol],
    size: Tuple[int, int] = (200, 200),
) -> Image.Image:
    """
    Draw molecule image or red X if drawing fails.

    Args:
        mol: RDKit molecule or None
        size: Image size (width, height)

    Returns:
        PIL Image
    """
    if mol is not None:
        try:
            # Try to compute 2D coords if not present
            if mol.GetNumConformers() == 0:
                AllChem.Compute2DCoords(mol)
            return Draw.MolToImage(mol, size=size)
        except Exception:
            # Try drawing without 2D coords computation
            try:
                return Draw.MolToImage(mol, size=size)
            except Exception:
                pass

    # Draw red X for invalid/failed molecules
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    margin = 20
    draw.line([(margin, margin), (size[0] - margin, size[1] - margin)], fill="red", width=5)
    draw.line([(size[0] - margin, margin), (margin, size[1] - margin)], fill="red", width=5)
    return img


def create_reconstruction_plot(
    original_mol: Chem.Mol,
    generated_mol: Optional[Chem.Mol],
    original_smiles: str,
    generated_smiles: Optional[str],
    is_valid: bool,
    is_match: bool,
    sample_idx: int,
    save_path: Path,
) -> None:
    """
    Create side-by-side plot of original vs generated molecule.

    Args:
        original_mol: Original RDKit molecule
        generated_mol: Generated RDKit molecule (can be None)
        original_smiles: Original SMILES string
        generated_smiles: Generated SMILES string (can be None)
        is_valid: Whether generated molecule is valid
        is_match: Whether SMILES match
        sample_idx: Sample index for labeling
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original molecule
    try:
        img_original = Draw.MolToImage(original_mol, size=(400, 400))
        axes[0].imshow(img_original)
    except Exception:
        axes[0].text(0.5, 0.5, "Failed to draw", ha="center", va="center", fontsize=14)
    axes[0].set_title(f"Original\n{original_smiles[:50]}{'...' if len(original_smiles) > 50 else ''}", fontsize=10)
    axes[0].axis("off")

    # Plot generated molecule
    if generated_mol is not None:
        try:
            img_generated = Draw.MolToImage(generated_mol, size=(400, 400))
            axes[1].imshow(img_generated)
        except Exception:
            axes[1].text(0.5, 0.5, "Failed to draw", ha="center", va="center", fontsize=14)
    else:
        axes[1].text(0.5, 0.5, "Generation failed", ha="center", va="center", fontsize=14)

    gen_smiles_display = generated_smiles or "N/A"
    if len(gen_smiles_display) > 50:
        gen_smiles_display = gen_smiles_display[:50] + "..."
    axes[1].set_title(f"Generated\n{gen_smiles_display}", fontsize=10)
    axes[1].axis("off")

    # Add status information
    status_color = "green" if is_match else ("orange" if is_valid else "red")
    status_text = "MATCH" if is_match else ("Valid" if is_valid else "Invalid")
    fig.suptitle(f"Sample {sample_idx + 1}: {status_text}", fontsize=14, color=status_color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_tanimoto_similarity(
    mol1: Optional[Chem.Mol],
    mol2: Optional[Chem.Mol],
    radius: int = 2,
    n_bits: int = 2048,
) -> float:
    """
    Compute Tanimoto similarity between two molecules.

    Args:
        mol1: First RDKit molecule
        mol2: Second RDKit molecule
        radius: Morgan fingerprint radius
        n_bits: Fingerprint bit length

    Returns:
        Tanimoto similarity (0.0 if either molecule is invalid)
    """
    if mol1 is None or mol2 is None:
        return 0.0

    # Check that molecules have atoms
    if mol1.GetNumAtoms() == 0 or mol2.GetNumAtoms() == 0:
        return 0.0

    try:
        # Use the newer MorganGenerator API to avoid deprecation warnings
        from rdkit.Chem import rdFingerprintGenerator

        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp1 = generator.GetFingerprint(mol1)
        fp2 = generator.GetFingerprint(mol2)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        # Fallback: return 0.0 for any error
        return 0.0


# ========= Callbacks =========


class LossTrackingCallback(Callback):
    """
    PyTorch Lightning callback to track training and validation loss per epoch.

    Logs losses using PyComex experiment tracking (e.track()) for time-series
    visualization and analysis.
    """

    def __init__(self, experiment: "Experiment"):
        super().__init__()
        self.experiment = experiment

    def on_train_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        """Track training loss at the end of each training epoch."""
        train_loss = trainer.callback_metrics.get("train/loss")
        if train_loss is not None:
            # Use underscore instead of slash - PyComex has a bug where slash-separated
            # keys in track() get overwritten each call instead of appended
            self.experiment.track("loss_train", float(train_loss))

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        """Track validation loss."""
        val_loss = trainer.callback_metrics.get("val/loss")
        if val_loss is not None:
            self.experiment.track("loss_val", float(val_loss))


def compute_hdc_distance(
    generated_data: Data,
    original_hdc_vectors: Tensor,
    base_hdc_dim: int,
    hypernet: HyperNet,
    device: torch.device,
    dataset: str = "zinc",
    original_x: Optional[Tensor] = None,
) -> float:
    """
    Compute HDC cosine distance between a generated graph and the original.

    Encodes the generated graph directly using its ``edge_index`` and node
    features, bypassing the lossy SMILES round-trip.  When ``original_x``
    (raw integer features, e.g. shape ``(n, 5)`` for ZINC) is provided it is
    used as-is; otherwise the 24-dim one-hot ``generated_data.x`` is reversed
    via ``onehot_to_raw_features``.

    Delegates to ``hypernet.calculate_distance`` which handles both single
    HyperNet (one order-N comparison) and MultiHyperNet (per-sub-HyperNet
    average) transparently.

    Args:
        generated_data: Generated PyG Data object (from FlowEdgeDecoder.sample)
        original_hdc_vectors: Full concatenated [order_0 | order_N] vector
        base_hdc_dim: Base HDC dimension (unused, kept for API compat)
        hypernet: HyperNet or MultiHyperNet encoder
        device: Unused, kept for API compatibility
        dataset: Dataset type for feature encoding ("zinc" or "qm9")
        original_x: Optional raw node features tensor (n, num_raw_features).
                     When provided, used directly instead of reversing one-hot.

    Returns:
        Cosine distance (float). Lower is better. Returns inf on failure.
    """
    try:
        # Get raw node features for HyperNet encoding
        if original_x is not None:
            raw_x = original_x.clone()
        else:
            raw_x = onehot_to_raw_features(generated_data.x)

        # Build a clean Data with raw features and generated edges
        hdc_device = hypernet.nodes_codebook.device
        gen_data = Data(
            x=raw_x.to(hdc_device),
            edge_index=generated_data.edge_index.to(hdc_device),
        )
        gen_data.batch = torch.zeros(
            gen_data.x.size(0), dtype=torch.long, device=hdc_device
        )

        # Augment with RW features if the hypernet expects them
        if hasattr(hypernet, "rw_config") and hypernet.rw_config.enabled:
            from graph_hdc.utils.rw_features import augment_data_with_rw
            gen_data = augment_data_with_rw(
                gen_data,
                k_values=hypernet.rw_config.k_values,
                num_bins=hypernet.rw_config.num_bins,
                bin_boundaries=hypernet.rw_config.bin_boundaries,
            )

        with torch.no_grad():
            gen_data = hypernet.encode_properties(gen_data)
            if gen_data.node_hv.device != hdc_device:
                gen_data.node_hv = gen_data.node_hv.to(hdc_device)

            gen_order_zero = scatter_hd(
                src=gen_data.node_hv, index=gen_data.batch, op="bundle",
            )
            gen_output = hypernet.forward(gen_data, normalize=True)
            gen_order_n = gen_output["graph_embedding"]
            gen_hdc_vector = torch.cat([gen_order_zero, gen_order_n], dim=-1)

        return hypernet.calculate_distance(
            original_hdc_vectors.to(hdc_device), gen_hdc_vector,
        ).item()
    except Exception:
        import traceback
        traceback.print_exc()
        return float("inf")


class ReconstructionVisualizationCallback(Callback):
    """
    Visualize molecule reconstructions at each validation epoch.

    Takes fixed validation samples, generates reconstructions using the model,
    and creates a 2xN figure comparing originals with reconstructions.

    When ``num_repetitions > 1``, each molecule is decoded multiple times in
    a single batched ``sample()`` call and the result with the lowest HDC
    cosine distance to the original is kept (best-of-N).
    """

    def __init__(
        self,
        experiment: "Experiment",
        vis_samples: List[Data],
        sample_steps: int,
        eta: float,
        omega: float,
        time_distortion: str,
        hypernet: Optional[HyperNet] = None,
        num_repetitions: int = 1,
        dataset: str = "zinc",
    ):
        """
        Initialize the visualization callback.

        Args:
            experiment: PyComex experiment for tracking
            vis_samples: List of preprocessed Data objects for visualization
            sample_steps: Number of sampling steps
            eta: Stochasticity parameter
            omega: Target guidance strength
            time_distortion: Time distortion type
            hypernet: Optional HyperNet encoder for computing HDC cosine distance
            num_repetitions: Number of parallel decodings per molecule (best-of-N)
            dataset: Dataset type for HDC distance computation ("zinc" or "qm9")
        """
        super().__init__()
        self.experiment = experiment
        self.vis_samples = vis_samples
        self.sample_steps = sample_steps
        self.eta = eta
        self.omega = omega
        self.time_distortion = time_distortion
        self.hypernet = hypernet
        self.num_repetitions = num_repetitions
        self.dataset = dataset

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Generate reconstructions and visualize."""
        if len(self.vis_samples) == 0:
            return

        device = pl_module.device
        num_reps = self.num_repetitions
        n_samples = len(self.vis_samples)

        # Collect per-sample inputs
        scrubbed_smiles_list = []
        original_mols = []
        hdc_vectors_list = []
        node_features_list = []

        for data in self.vis_samples:
            scrubbed = scrub_smiles(data.smiles) or data.smiles
            scrubbed_smiles_list.append(scrubbed)
            original_mols.append(Chem.MolFromSmiles(scrubbed))

            hdc_vec = data.hdc_vector
            if hdc_vec.dim() == 2:
                hdc_vec = hdc_vec.squeeze(0)
            hdc_vectors_list.append(hdc_vec)
            node_features_list.append(data.x)

        # Generate reconstructions: process each molecule sequentially,
        # but batch num_repetitions parallel decodings per molecule.
        pl_module.eval()
        self.experiment.log(
            f"\nGenerating {n_samples} reconstructions "
            f"(best-of-{num_reps}, {self.sample_steps} steps)..."
        )

        reconstructed_samples = []
        best_distances = []

        sample_kwargs = dict(
            sample_steps=self.sample_steps,
            eta=self.eta,
            omega=self.omega,
            time_distortion=self.time_distortion,
            device=device,
            show_progress=False,
        )

        with torch.no_grad():
            for i in range(n_samples):
                hdc_vec = hdc_vectors_list[i].to(device).float().unsqueeze(0)
                nf = node_features_list[i].to(device).unsqueeze(0)
                n_nodes = node_features_list[i].size(0)
                mask = torch.ones(1, n_nodes, dtype=torch.bool, device=device)

                if num_reps > 1 and self.hypernet is not None:
                    orig_hdc = hdc_vectors_list[i]
                    base_hdc_dim = orig_hdc.size(-1) // 2

                    raw_x = getattr(self.vis_samples[i], "original_x", None)

                    def score_fn(s, _orig=orig_hdc, _dim=base_hdc_dim, _raw_x=raw_x):
                        return compute_hdc_distance(
                            s, _orig.unsqueeze(0), _dim,
                            self.hypernet, device, dataset=self.dataset,
                            original_x=_raw_x,
                        )

                    best_sample, best_dist = pl_module.sample_best_of_n(
                        hdc_vectors=hdc_vec,
                        node_features=nf,
                        node_mask=mask,
                        num_repetitions=num_reps,
                        score_fn=score_fn,
                        **sample_kwargs,
                    )
                else:
                    samples = pl_module.sample(
                        hdc_vectors=hdc_vec,
                        node_features=nf,
                        node_mask=mask,
                        **sample_kwargs,
                    )
                    best_sample = samples[0]
                    best_dist = None

                reconstructed_samples.append(best_sample)
                best_distances.append(best_dist)
                self.experiment.log(
                    f"  Molecule {i + 1}/{n_samples}"
                    + (f" (best HDC dist: {best_dist:.6f})" if best_dist is not None else "")
                )

        # Move generated samples to CPU and release all GPU references
        # from the sampling loop so empty_cache() can fully reclaim memory.
        reconstructed_samples = [s.cpu() for s in reconstructed_samples]
        # Break references to GPU tensors left over from the last loop iteration
        # (hdc_vec, nf, mask are .to(device) copies; best_sample holds GPU Data;
        # samples/score_fn may exist depending on which branch ran).
        hdc_vec = nf = mask = best_sample = best_dist = None  # noqa: F841
        samples = score_fn = None  # noqa: F841
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Convert reconstructions to RDKit molecules
        reconstructed_mols = [pyg_to_mol(data) for data in reconstructed_samples]

        # Compute Tanimoto similarities
        similarities = []
        for orig_mol, recon_mol in zip(original_mols, reconstructed_mols):
            sim = compute_tanimoto_similarity(orig_mol, recon_mol)
            similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # Compute HDC cosine distances if hypernet is available
        hdc_cosine_distances = []
        if self.hypernet is not None:
            hdc_device = torch.device(self.hypernet.device)
            with torch.no_grad():
                for i, (orig_data, recon_mol) in enumerate(zip(self.vis_samples, reconstructed_mols)):
                    if recon_mol is None:
                        hdc_cosine_distances.append(0.0)
                        continue
                    recon_data = recon_out = recon_emb = orig_order_n = None
                    try:
                        recon_data = zinc_mol_to_data(recon_mol).to(hdc_device)
                        recon_data.batch = torch.zeros(
                            recon_data.x.size(0), dtype=torch.long, device=hdc_device
                        )
                        if hasattr(self.hypernet, "rw_config") and self.hypernet.rw_config.enabled:
                            from graph_hdc.utils.rw_features import augment_data_with_rw
                            recon_data = augment_data_with_rw(
                                recon_data,
                                k_values=self.hypernet.rw_config.k_values,
                                num_bins=self.hypernet.rw_config.num_bins,
                                bin_boundaries=self.hypernet.rw_config.bin_boundaries,
                            )
                        recon_out = self.hypernet.forward(recon_data, normalize=True)
                        recon_emb = recon_out["graph_embedding"]

                        hdc_vec = orig_data.hdc_vector
                        if hdc_vec.dim() == 2:
                            hdc_vec = hdc_vec.squeeze(0)
                        hdc_dim = hdc_vec.size(-1) // 2
                        orig_order_n = hdc_vec[hdc_dim:].unsqueeze(0).to(hdc_device)

                        cos_sim = torch.nn.functional.cosine_similarity(
                            orig_order_n.float(), recon_emb.float(), dim=-1
                        ).item()
                        hdc_cosine_distances.append(cos_sim)
                    except Exception:
                        hdc_cosine_distances.append(0.0)
                    finally:
                        del recon_data, recon_out, recon_emb, orig_order_n
            # Release remaining references
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            avg_hdc_cosine = (
                sum(hdc_cosine_distances) / len(hdc_cosine_distances)
                if hdc_cosine_distances
                else 0.0
            )
        else:
            avg_hdc_cosine = None

        # Create 2xN visualization figure
        n_samples = len(self.vis_samples)
        fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))

        # Handle single sample case (axes not 2D)
        if n_samples == 1:
            axes = axes.reshape(2, 1)

        for i in range(n_samples):
            # Top row: original molecules
            orig_img = draw_mol_or_error(original_mols[i], size=(200, 200))
            axes[0, i].imshow(orig_img)
            axes[0, i].axis("off")
            orig_smiles = scrubbed_smiles_list[i]
            truncated_orig = orig_smiles[:20] + ("..." if len(orig_smiles) > 20 else "")
            axes[0, i].set_title(f"Original\n{truncated_orig}", fontsize=8)

            # Bottom row: reconstructions
            recon_img = draw_mol_or_error(reconstructed_mols[i], size=(200, 200))
            axes[1, i].imshow(recon_img)
            axes[1, i].axis("off")

            # Get reconstruction SMILES if valid
            recon_smiles = "Invalid"
            if reconstructed_mols[i] is not None:
                try:
                    recon_smiles = Chem.MolToSmiles(reconstructed_mols[i], canonical=True)
                except Exception:
                    recon_smiles = "Invalid"

            truncated_recon = recon_smiles[:20] + ("..." if len(recon_smiles) > 20 else "")
            label = f"Recon (Tan={similarities[i]:.3f}"
            if hdc_cosine_distances:
                label += f", HDC={hdc_cosine_distances[i]:.3f}"
            label += f")\n{truncated_recon}"
            axes[1, i].set_title(label, fontsize=8)

        title = f"Validation Reconstructions (Epoch {trainer.current_epoch}, Avg Tanimoto: {avg_similarity:.3f}"
        if avg_hdc_cosine is not None:
            title += f", Avg HDC Cos: {avg_hdc_cosine:.3f}"
        title += ")"
        fig.suptitle(title)
        plt.tight_layout()

        # Track figure and metric with PyComex
        self.experiment.track("validation_reconstructions", fig)
        self.experiment.track("tanimoto_similarity", avg_similarity)
        if avg_hdc_cosine is not None:
            self.experiment.track("hdc_cosine_similarity", avg_hdc_cosine)

        plt.close(fig)


class GracefulInterruptHandler:
    """
    Context manager for handling CTRL+C gracefully during training.

    First CTRL+C: Sets trainer.should_stop = True for graceful shutdown
    Second CTRL+C: Forces immediate exit
    """

    def __init__(self, trainer: Optional[Trainer] = None):
        """
        Initialize the interrupt handler.

        Args:
            trainer: Optional PyTorch Lightning Trainer instance
        """
        self.trainer = trainer
        self.original_handler = None
        self.interrupt_count = 0

    def set_trainer(self, trainer: Trainer) -> None:
        """
        Set the trainer after initialization.

        Args:
            trainer: PyTorch Lightning Trainer instance
        """
        self.trainer = trainer

    def _handler(self, signum, frame):
        """Handle SIGINT signal."""
        self.interrupt_count += 1

        if self.interrupt_count == 1:
            print("\n" + "=" * 60)
            print("CTRL+C received - Gracefully stopping training...")
            print("(Press CTRL+C again to force quit)")
            print("=" * 60 + "\n")

            if self.trainer is not None:
                self.trainer.should_stop = True
        else:
            print("\n" + "=" * 60)
            print("Second CTRL+C received - Forcing exit...")
            print("=" * 60 + "\n")
            # Restore original handler and re-raise
            signal.signal(signal.SIGINT, self.original_handler)
            raise KeyboardInterrupt

    def __enter__(self):
        self.original_handler = signal.signal(signal.SIGINT, self._handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self.original_handler)
        # Don't suppress exceptions
        return False


# ========= Training Metrics Callback =========


# Edge class labels for plotting
EDGE_CLASS_LABELS = ["None", "Single", "Double", "Triple", "Aromatic"]

# Module group prefixes for gradient tracking
MODULE_GROUPS = {
    "HDC Cond.": "condition_mlp",
    "Time Emb.": "time_mlp",
    "Transformer": "model",
}


class TrainingMetricsCallback(Callback):
    """
    Track and visualize comprehensive training metrics in a 4x4 grid.

    Generates plots for:
    - Row 1: Loss curves, per-class loss, learning rate, timestep-binned loss
    - Row 2: Edge distribution, per-class accuracy, confusion matrix, loss ratio
    - Row 3: Gradient norms (global, by module, HDC, variance)
    - Row 4: Parameter changes, weight norms, convergence metrics

    The callback reads batch metrics from the model's `batch_metrics` attribute,
    which should be populated by the model's training_step.
    """

    def __init__(
        self,
        experiment: "Experiment",
        num_timestep_bins: int = 10,
        num_edge_classes: int = 5,
        smoothing_window: int = 10,
    ):
        """
        Initialize the training metrics callback.

        Args:
            experiment: PyComex Experiment instance for tracking
            num_timestep_bins: Number of bins for timestep analysis
            num_edge_classes: Number of edge classes (default 5)
            smoothing_window: Window size for loss smoothing
        """
        super().__init__()
        self.experiment = experiment
        self.num_timestep_bins = num_timestep_bins
        self.num_edge_classes = num_edge_classes
        self.smoothing_window = smoothing_window

        # Initialize storage for metrics
        self._init_metric_storage()

    def _init_metric_storage(self):
        """Initialize all metric storage containers."""
        # Row 1: Loss metrics
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.class_losses: List[List[float]] = []  # Per-epoch, per-class
        self.timestep_losses: List[List[float]] = []  # Per-epoch, per-bin

        # Row 2: Edge prediction metrics
        self.pred_edge_counts: List[np.ndarray] = []
        self.gt_edge_counts: List[np.ndarray] = []
        self.class_accuracies: List[List[float]] = []
        self.confusion_matrices: List[np.ndarray] = []

        # Row 3: Gradient metrics
        self.global_grad_norms: List[float] = []
        self.module_grad_norms: List[Dict[str, float]] = []
        self.hdc_grad_norms: List[float] = []
        self.grad_norm_variances: List[float] = []

        # Row 4: Convergence metrics
        self.param_deltas: List[float] = []
        self.weight_norms: List[Dict[str, float]] = []
        self.module_param_deltas: List[Dict[str, float]] = []
        self.tanimoto_similarities: List[float] = []

        # Within-epoch accumulators (reset each epoch)
        self._reset_epoch_accumulators()

        # Parameter snapshot for delta computation
        self._param_snapshot: Optional[Dict[str, Tensor]] = None

    def _reset_epoch_accumulators(self):
        """Reset accumulators at start of each epoch."""
        self._epoch_grad_norms: List[float] = []
        self._epoch_module_grad_norms: Dict[str, List[float]] = defaultdict(list)
        self._epoch_timestep_losses: Dict[int, List[float]] = defaultdict(list)
        self._epoch_confusion = np.zeros(
            (self.num_edge_classes, self.num_edge_classes), dtype=np.int64
        )
        self._epoch_pred_counts = np.zeros(self.num_edge_classes, dtype=np.int64)
        self._epoch_gt_counts = np.zeros(self.num_edge_classes, dtype=np.int64)
        self._epoch_class_correct = np.zeros(self.num_edge_classes, dtype=np.int64)
        self._epoch_class_total = np.zeros(self.num_edge_classes, dtype=np.int64)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _compute_gradient_norm(
        self, model: pl.LightningModule, prefix: Optional[str] = None
    ) -> float:
        """Compute gradient L2 norm for parameters matching prefix."""
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                if prefix is None or name.startswith(prefix):
                    total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm**0.5

    def _compute_weight_norm(
        self, model: pl.LightningModule, prefix: Optional[str] = None
    ) -> float:
        """Compute weight L2 norm for parameters matching prefix."""
        total_norm = 0.0
        for name, param in model.named_parameters():
            if prefix is None or name.startswith(prefix):
                total_norm += param.data.norm(2).item() ** 2
        return total_norm**0.5

    def _compute_param_delta(
        self,
        model: pl.LightningModule,
        snapshot: Dict[str, Tensor],
        prefix: Optional[str] = None,
    ) -> float:
        """Compute parameter change from snapshot."""
        total_delta = 0.0
        for name, param in model.named_parameters():
            if prefix is None or name.startswith(prefix):
                if name in snapshot:
                    delta = (param.data.cpu() - snapshot[name]).norm(2).item() ** 2
                    total_delta += delta
        return total_delta**0.5

    def _take_param_snapshot(self, model: pl.LightningModule) -> Dict[str, Tensor]:
        """Take snapshot of all parameters (detached, CPU)."""
        return {
            name: param.data.clone().cpu() for name, param in model.named_parameters()
        }

    def _smooth(self, values: List[float], window: int) -> List[float]:
        """Apply simple moving average smoothing."""
        if len(values) < window:
            return values
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            smoothed.append(sum(values[start : i + 1]) / (i - start + 1))
        return smoothed

    def _bin_timestep(self, t: float) -> int:
        """Bin timestep t in [0,1] to bin index."""
        return min(int(t * self.num_timestep_bins), self.num_timestep_bins - 1)

    # =========================================================================
    # PyTorch Lightning Hooks
    # =========================================================================

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Take parameter snapshot and reset accumulators."""
        self._param_snapshot = self._take_param_snapshot(pl_module)
        self._reset_epoch_accumulators()

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
        optimizer,
    ) -> None:
        """Capture gradient norms before optimizer step (while gradients exist)."""
        # Global gradient norm
        grad_norm = self._compute_gradient_norm(pl_module)
        self._epoch_grad_norms.append(grad_norm)

        # Module-specific gradient norms (must be captured here, before zero_grad)
        for name, prefix in MODULE_GROUPS.items():
            module_grad = self._compute_gradient_norm(pl_module, prefix)
            self._epoch_module_grad_norms[name].append(module_grad)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Accumulate batch-level metrics."""
        # Check if model has batch_metrics
        if not hasattr(pl_module, "batch_metrics") or not pl_module.batch_metrics:
            return

        metrics = pl_module.batch_metrics

        # Extract data
        t = metrics.get("t")
        pred_classes = metrics.get("pred_classes")
        true_classes = metrics.get("true_classes")
        node_mask = metrics.get("node_mask")
        loss = metrics.get("loss")

        if t is None or pred_classes is None or true_classes is None:
            return

        # Bin timestep and accumulate loss
        if loss is not None:
            # t might be (bs,) or (bs, 1)
            t_flat = t.flatten()
            for t_val in t_flat:
                bin_idx = self._bin_timestep(t_val.item())
                self._epoch_timestep_losses[bin_idx].append(loss.item())

        # Update confusion matrix and counts (all tensors are on CPU)
        if node_mask is not None:
            bs, n = node_mask.shape
            edge_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
            diag_mask = ~torch.eye(n, dtype=torch.bool)
            edge_mask = edge_mask & diag_mask.unsqueeze(0)
            triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            edge_mask = edge_mask & triu_mask.unsqueeze(0)

            pred_flat = pred_classes[edge_mask].numpy()
            true_flat = true_classes[edge_mask].numpy()

            # Update confusion matrix
            for p, t in zip(pred_flat, true_flat):
                self._epoch_confusion[t, p] += 1

            # Update counts
            for c in range(self.num_edge_classes):
                self._epoch_pred_counts[c] += (pred_flat == c).sum()
                self._epoch_gt_counts[c] += (true_flat == c).sum()
                correct = ((pred_flat == c) & (true_flat == c)).sum()
                total = (true_flat == c).sum()
                self._epoch_class_correct[c] += correct
                self._epoch_class_total[c] += total

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Aggregate epoch metrics."""
        # Skip during sanity check
        if trainer.sanity_checking:
            return

        # Store train loss
        train_loss = trainer.callback_metrics.get("train/loss")
        if train_loss is not None:
            self.train_losses.append(float(train_loss))

        # Store gradient metrics
        if self._epoch_grad_norms:
            mean_grad = np.mean(self._epoch_grad_norms)
            var_grad = np.var(self._epoch_grad_norms)
            self.global_grad_norms.append(mean_grad)
            self.grad_norm_variances.append(var_grad)

            # Module-specific gradients (averaged from accumulated values)
            module_grads = {}
            for name in MODULE_GROUPS.keys():
                if self._epoch_module_grad_norms[name]:
                    module_grads[name] = np.mean(self._epoch_module_grad_norms[name])
                else:
                    module_grads[name] = 0.0
            self.module_grad_norms.append(module_grads)
            self.hdc_grad_norms.append(module_grads.get("HDC Cond.", 0.0))

        # Store timestep-binned losses
        bin_losses = []
        for bin_idx in range(self.num_timestep_bins):
            losses = self._epoch_timestep_losses.get(bin_idx, [])
            bin_losses.append(np.mean(losses) if losses else 0.0)
        self.timestep_losses.append(bin_losses)

        # Store edge counts and confusion matrix
        self.pred_edge_counts.append(self._epoch_pred_counts.copy())
        self.gt_edge_counts.append(self._epoch_gt_counts.copy())
        self.confusion_matrices.append(self._epoch_confusion.copy())

        # Compute per-class accuracy
        class_acc = []
        for c in range(self.num_edge_classes):
            if self._epoch_class_total[c] > 0:
                acc = self._epoch_class_correct[c] / self._epoch_class_total[c]
            else:
                acc = 0.0
            class_acc.append(acc)
        self.class_accuracies.append(class_acc)

        # Compute parameter delta
        if self._param_snapshot is not None:
            total_delta = self._compute_param_delta(pl_module, self._param_snapshot)
            self.param_deltas.append(total_delta)

            # Module-specific deltas
            module_deltas = {}
            for name, prefix in MODULE_GROUPS.items():
                module_deltas[name] = self._compute_param_delta(
                    pl_module, self._param_snapshot, prefix
                )
            self.module_param_deltas.append(module_deltas)

        # Compute weight norms
        weight_norms = {}
        for name, prefix in MODULE_GROUPS.items():
            weight_norms[name] = self._compute_weight_norm(pl_module, prefix)
        self.weight_norms.append(weight_norms)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Store validation loss, compute improvement, and generate plots."""
        # Skip during sanity check (no training has happened yet)
        if trainer.sanity_checking:
            return

        # Store validation loss
        val_loss = trainer.callback_metrics.get("val/loss")
        if val_loss is not None:
            self.val_losses.append(float(val_loss))

        # Read Tanimoto similarity from ReconstructionVisualizationCallback
        # (must run after it in callback order)
        tracked = self.experiment.data.get("tanimoto_similarity", [])
        if len(tracked) > len(self.tanimoto_similarities):
            self.tanimoto_similarities.append(float(tracked[-1]))

        # Generate plots (only if we have at least one epoch of data)
        epoch = trainer.current_epoch + 1
        if len(self.train_losses) > 0:
            try:
                fig = self._create_metrics_plot(epoch)
                self.experiment.track("training_metrics", fig)
                plt.close(fig)
            except Exception as e:
                # Log error but don't crash training
                self.experiment.log(f"Warning: Failed to create metrics plot: {e}")

    # =========================================================================
    # Plot Generation
    # =========================================================================

    def _create_metrics_plot(self, epoch: int) -> plt.Figure:
        """Create 4x4 grid of training metrics."""
        fig, axes = plt.subplots(4, 4, figsize=(16, 14))

        epochs = list(range(1, len(self.train_losses) + 1))

        # Row 1: Loss curves
        self._plot_loss_curves(axes[0, 0], epochs)
        self._plot_class_losses(axes[0, 1], epochs)
        self._plot_loss_log_scale(axes[0, 2], epochs)
        self._plot_timestep_loss(axes[0, 3])

        # Row 2: Edge Type Analysis
        self._plot_edge_distribution(axes[1, 0])
        self._plot_class_accuracy(axes[1, 1], epochs)
        self._plot_confusion_matrix(axes[1, 2])
        self._plot_class_loss_ratio(axes[1, 3], epochs)

        # Row 3: Gradients
        self._plot_global_grad_norm(axes[2, 0], epochs)
        self._plot_module_grad_norms(axes[2, 1], epochs)
        self._plot_hdc_grad_norm(axes[2, 2], epochs)
        self._plot_grad_variance(axes[2, 3], epochs)

        # Row 4: Convergence
        self._plot_param_delta(axes[3, 0], epochs)
        self._plot_weight_norms(axes[3, 1], epochs)
        self._plot_module_param_deltas(axes[3, 2], epochs)
        self._plot_tanimoto_similarity(axes[3, 3], epochs)

        fig.suptitle(f"Training Metrics - Epoch {epoch}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def _plot_loss_curves(self, ax, epochs):
        """Plot train/val loss with smoothing."""
        if not self.train_losses:
            ax.set_title("Train/Val Loss")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        # Ensure we only plot data we have
        n_train = min(len(epochs), len(self.train_losses))
        train_epochs = epochs[:n_train]
        train_losses = self.train_losses[:n_train]

        ax.plot(train_epochs, train_losses, alpha=0.3, color="blue", label="Train (raw)")
        ax.plot(
            train_epochs,
            self._smooth(train_losses, self.smoothing_window),
            color="blue",
            linewidth=2,
            label="Train (smooth)",
        )
        if self.val_losses:
            # Only plot as many val losses as we have epochs
            n_val = min(len(epochs), len(self.val_losses))
            val_epochs = epochs[:n_val]
            val_losses = self.val_losses[:n_val]
            ax.plot(
                val_epochs, val_losses, alpha=0.3, color="orange", label="Val (raw)"
            )
            ax.plot(
                val_epochs,
                self._smooth(val_losses, self.smoothing_window),
                color="orange",
                linewidth=2,
                label="Val (smooth)",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Train/Val Loss")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_class_losses(self, ax, epochs):
        """Plot loss broken down by class (from confusion matrix errors)."""
        if not self.confusion_matrices:
            ax.set_title("Per-Class Error Rate")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        # Compute error rate per class over epochs
        error_rates = []
        for cm in self.confusion_matrices:
            row_sums = cm.sum(axis=1)
            diag = np.diag(cm)
            # Error rate = 1 - accuracy
            rates = []
            for c in range(self.num_edge_classes):
                if row_sums[c] > 0:
                    rates.append(1.0 - diag[c] / row_sums[c])
                else:
                    rates.append(0.0)
            error_rates.append(rates)

        error_rates = np.array(error_rates)
        n_data = min(len(epochs), len(error_rates))
        plot_epochs = epochs[:n_data]
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_edge_classes))

        for c in range(self.num_edge_classes):
            ax.plot(
                plot_epochs,
                error_rates[:n_data, c],
                color=colors[c],
                label=EDGE_CLASS_LABELS[c],
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error Rate")
        ax.set_title("Per-Class Error Rate")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_loss_log_scale(self, ax, epochs):
        """Plot train/val loss on a log scale."""
        if not self.train_losses and not self.val_losses:
            ax.set_title("Loss (log scale)")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        if self.train_losses:
            n_train = min(len(epochs), len(self.train_losses))
            ax.plot(epochs[:n_train], self.train_losses[:n_train], label="Train", linewidth=1.5)
        if self.val_losses:
            val_epochs = list(range(1, len(self.val_losses) + 1))
            ax.plot(val_epochs, self.val_losses, label="Val", linewidth=1.5)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title("Loss (log scale)")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls="--", alpha=0.3)

    def _plot_timestep_loss(self, ax):
        """Plot loss vs timestep bin."""
        if not self.timestep_losses:
            ax.set_title("Loss vs Timestep")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        # Use latest epoch's timestep losses
        latest = self.timestep_losses[-1]
        bins = np.linspace(0, 1, self.num_timestep_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        ax.bar(bin_centers, latest, width=1.0 / self.num_timestep_bins, alpha=0.7)
        ax.set_xlabel("Timestep t")
        ax.set_ylabel("Loss")
        ax.set_title("Loss vs Timestep (Latest)")
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

    def _plot_edge_distribution(self, ax):
        """Plot predicted vs ground truth edge distribution."""
        if not self.pred_edge_counts or not self.gt_edge_counts:
            ax.set_title("Edge Distribution")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        # Use latest epoch
        pred = self.pred_edge_counts[-1].astype(float)
        gt = self.gt_edge_counts[-1].astype(float)

        # Normalize
        pred = pred / (pred.sum() + 1e-8)
        gt = gt / (gt.sum() + 1e-8)

        x = np.arange(self.num_edge_classes)
        width = 0.35

        ax.bar(x - width / 2, gt, width, label="Ground Truth", alpha=0.7)
        ax.bar(x + width / 2, pred, width, label="Predicted", alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(EDGE_CLASS_LABELS, fontsize=8, rotation=45)
        ax.set_ylabel("Frequency")
        ax.set_title("Edge Distribution")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_class_accuracy(self, ax, epochs):
        """Plot per-class accuracy over epochs."""
        if not self.class_accuracies:
            ax.set_title("Per-Class Accuracy")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        accs = np.array(self.class_accuracies)
        n_data = min(len(epochs), len(accs))
        plot_epochs = epochs[:n_data]
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_edge_classes))

        for c in range(self.num_edge_classes):
            ax.plot(
                plot_epochs,
                accs[:n_data, c],
                color=colors[c],
                label=EDGE_CLASS_LABELS[c],
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Per-Class Accuracy")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    def _plot_confusion_matrix(self, ax):
        """Plot edge type confusion matrix (normalized by row)."""
        if not self.confusion_matrices:
            ax.set_title("Confusion Matrix")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        cm = self.confusion_matrices[-1]
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = cm / (row_sums + 1e-8)

        im = ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(self.num_edge_classes))
        ax.set_yticks(range(self.num_edge_classes))
        ax.set_xticklabels(EDGE_CLASS_LABELS, fontsize=7, rotation=45)
        ax.set_yticklabels(EDGE_CLASS_LABELS, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title("Confusion Matrix")
        plt.colorbar(im, ax=ax, fraction=0.046)

    def _plot_class_loss_ratio(self, ax, epochs):
        """Plot relative contribution of each class to total error."""
        if not self.confusion_matrices:
            ax.set_title("Class Error Contribution")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        # Compute error contribution per class
        contributions = []
        for cm in self.confusion_matrices:
            row_sums = cm.sum(axis=1)
            diag = np.diag(cm)
            errors_per_class = row_sums - diag
            total_errors = errors_per_class.sum()
            if total_errors > 0:
                contrib = errors_per_class / total_errors
            else:
                contrib = np.zeros(self.num_edge_classes)
            contributions.append(contrib)

        contributions = np.array(contributions)
        n_data = min(len(epochs), len(contributions))
        plot_epochs = epochs[:n_data]

        # Stacked area plot
        ax.stackplot(
            plot_epochs,
            contributions[:n_data].T,
            labels=EDGE_CLASS_LABELS,
            alpha=0.7,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error Contribution")
        ax.set_title("Class Error Contribution")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    def _plot_global_grad_norm(self, ax, epochs):
        """Plot global gradient norm."""
        if not self.global_grad_norms:
            ax.set_title("Global Gradient Norm")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        n_data = min(len(epochs), len(self.global_grad_norms))
        ax.plot(
            epochs[:n_data],
            self.global_grad_norms[:n_data],
            color="purple",
            linewidth=2,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Global Gradient Norm")
        ax.grid(True, alpha=0.3)

    def _plot_module_grad_norms(self, ax, epochs):
        """Plot gradient norms by module group."""
        if not self.module_grad_norms:
            ax.set_title("Gradient by Module")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        colors = {"HDC Cond.": "red", "Time Emb.": "green", "Transformer": "blue"}

        for name in MODULE_GROUPS.keys():
            values = [d.get(name, 0.0) for d in self.module_grad_norms]
            n_data = min(len(epochs), len(values))
            ax.plot(
                epochs[:n_data],
                values[:n_data],
                color=colors.get(name, "gray"),
                label=name,
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient by Module")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_hdc_grad_norm(self, ax, epochs):
        """Plot HDC conditioning gradient norm."""
        if not self.hdc_grad_norms:
            ax.set_title("HDC Conditioning Gradient")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        n_data = min(len(epochs), len(self.hdc_grad_norms))
        ax.plot(
            epochs[:n_data],
            self.hdc_grad_norms[:n_data],
            color="red",
            linewidth=2,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("HDC Conditioning Gradient")
        ax.grid(True, alpha=0.3)

    def _plot_grad_variance(self, ax, epochs):
        """Plot gradient norm variance within epochs."""
        if not self.grad_norm_variances:
            ax.set_title("Gradient Variance")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        n_data = min(len(epochs), len(self.grad_norm_variances))
        ax.plot(
            epochs[:n_data],
            self.grad_norm_variances[:n_data],
            color="orange",
            linewidth=2,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Variance")
        ax.set_title("Gradient Norm Variance")
        ax.grid(True, alpha=0.3)

    def _plot_param_delta(self, ax, epochs):
        """Plot total parameter change per epoch."""
        if not self.param_deltas:
            ax.set_title("Parameter Change")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        n_data = min(len(epochs), len(self.param_deltas))
        ax.plot(
            epochs[:n_data],
            self.param_deltas[:n_data],
            color="teal",
            linewidth=2,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("||Δθ||")
        ax.set_title("Parameter Change (Total)")
        ax.grid(True, alpha=0.3)

    def _plot_weight_norms(self, ax, epochs):
        """Plot weight norms by module group."""
        if not self.weight_norms:
            ax.set_title("Weight Norms")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        colors = {"HDC Cond.": "red", "Time Emb.": "green", "Transformer": "blue"}

        for name in MODULE_GROUPS.keys():
            values = [d.get(name, 0.0) for d in self.weight_norms]
            n_data = min(len(epochs), len(values))
            ax.plot(
                epochs[:n_data],
                values[:n_data],
                color=colors.get(name, "gray"),
                label=name,
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("||θ||")
        ax.set_title("Weight Norms by Module")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_module_param_deltas(self, ax, epochs):
        """Plot parameter change by module group."""
        if not self.module_param_deltas:
            ax.set_title("Parameter Change by Module")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        colors = {"HDC Cond.": "red", "Time Emb.": "green", "Transformer": "blue"}

        for name in MODULE_GROUPS.keys():
            values = [d.get(name, 0.0) for d in self.module_param_deltas]
            n_data = min(len(epochs), len(values))
            ax.plot(
                epochs[:n_data],
                values[:n_data],
                color=colors.get(name, "gray"),
                label=name,
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("||Δθ||")
        ax.set_title("Parameter Change by Module")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_tanimoto_similarity(self, ax, epochs):
        """Plot average Tanimoto similarity over validation samples per epoch."""
        if not self.tanimoto_similarities:
            ax.set_title("Tanimoto Similarity")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        n_data = min(len(epochs), len(self.tanimoto_similarities))
        ep = epochs[:n_data]
        raw = self.tanimoto_similarities[:n_data]

        # Raw values (semi-transparent)
        ax.plot(ep, raw, color="green", alpha=0.3, linewidth=1, label="Raw")

        # Smoothed values
        smoothed = self._smooth(raw, window=10)
        ax.plot(ep, smoothed, color="green", linewidth=2, label="Smoothed")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Avg Tanimoto")
        ax.set_title("Tanimoto Similarity")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)


# ========= Test Experiment Helpers =========


def load_smiles_from_csv(csv_path: str) -> list[str]:
    """
    Load SMILES strings from a CSV file.

    Args:
        csv_path: Path to CSV file. Must contain a "smiles" column.

    Returns:
        List of SMILES strings (NaN values dropped).

    Raises:
        ValueError: If CSV file does not contain a "smiles" column.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "smiles" not in df.columns:
        raise ValueError(f"CSV file must have 'smiles' column. Found: {list(df.columns)}")

    return df["smiles"].dropna().tolist()


def smiles_to_pyg_data(smiles: str, dataset: str = "zinc") -> Optional[Data]:
    """
    Convert SMILES string to PyG Data object with node features.

    Supports both QM9 (4-dim features) and ZINC (5-dim features) datasets.

    Args:
        smiles: SMILES string to convert.
        dataset: Dataset name for feature encoding ("qm9" or "zinc").

    Returns:
        PyG Data object with ``x``, ``edge_index``, and ``smiles`` attributes,
        or ``None`` if the SMILES cannot be parsed or contains unsupported atoms.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Get atom type mapping based on dataset
    if dataset.lower() == "qm9":
        from graph_hdc.datasets.qm9_smiles import QM9_ATOM_TO_IDX
        atom_to_idx = QM9_ATOM_TO_IDX
        num_features = 4
    else:
        from graph_hdc.datasets.zinc_smiles import ZINC_ATOM_TO_IDX
        atom_to_idx = ZINC_ATOM_TO_IDX
        num_features = 5

    # Build node features
    x = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in atom_to_idx:
            return None  # Unsupported atom type

        atom_type = atom_to_idx[sym]
        degree = max(0, min(5, atom.GetDegree() - 1))
        charge = atom.GetFormalCharge()
        charge_idx = 0 if charge == 0 else (1 if charge > 0 else 2)
        explicit_hs = min(3, atom.GetTotalNumHs())

        if num_features == 4:
            x.append([float(atom_type), float(degree), float(charge_idx), float(explicit_hs)])
        else:
            is_in_ring = float(atom.IsInRing())
            x.append([float(atom_type), float(degree), float(charge_idx), float(explicit_hs), is_in_ring])

    x = torch.tensor(x, dtype=torch.float)

    # Build edge index
    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, smiles=smiles)


def create_summary_bar_chart(
    match_count: int,
    valid_count: int,
    invalid_count: int,
    total_count: int,
    save_path: Path,
    title_prefix: str = "",
) -> None:
    """
    Create summary bar chart showing match/valid/invalid counts.

    Args:
        match_count: Number of exact SMILES matches.
        valid_count: Number of valid molecules (includes matches).
        invalid_count: Number of invalid molecules.
        total_count: Total number of processed molecules.
        save_path: Path to save the chart image.
        title_prefix: Optional prefix for the chart title (e.g. "HDC-Guided").
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ["Match", "Valid (no match)", "Invalid"]
    valid_no_match = valid_count - match_count
    counts = [match_count, valid_no_match, invalid_count]
    colors = ["green", "orange", "red"]

    bars = ax.bar(categories, counts, color=colors, edgecolor="black", linewidth=1.2)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percentage = 100 * count / total_count if total_count > 0 else 0
        ax.annotate(
            f"{count}\n({percentage:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Count", fontsize=12)
    prefix = f"{title_prefix} " if title_prefix else ""
    ax.set_title(f"{prefix}Reconstruction Results (n={total_count})", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.2 if max(counts) > 0 else 1)

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_accuracy_by_size_chart(
    results: List[Dict],
    save_path: Path,
    title_prefix: str = "",
) -> None:
    """
    Create bar chart showing exact match accuracy grouped by molecule size.

    Each bar represents molecules with the same number of heavy atoms.
    The y-axis shows the fraction of exact SMILES matches for that size.
    Sizes with no processed molecules are omitted (no bar).

    Args:
        results: List of result dicts, each containing at least
            ``num_atoms`` (int) and ``is_match`` (bool). Entries without
            these keys (e.g. skipped molecules) are ignored.
        save_path: Path to save the chart image.
        title_prefix: Optional prefix for the chart title.
    """
    # Group results by molecule size (skip entries without num_atoms)
    size_groups: Dict[int, List[bool]] = defaultdict(list)
    for r in results:
        if "num_atoms" not in r or "is_match" not in r:
            continue
        size_groups[r["num_atoms"]].append(r["is_match"])

    if not size_groups:
        return

    sizes = sorted(size_groups.keys())
    accuracies = []
    counts = []
    for s in sizes:
        matches = size_groups[s]
        counts.append(len(matches))
        accuracies.append(sum(matches) / len(matches))

    fig, ax = plt.subplots(figsize=(max(8, len(sizes) * 0.6), 6))

    bars = ax.bar(
        [str(s) for s in sizes],
        accuracies,
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.8,
    )

    # Add count labels above each bar
    for bar, acc, cnt in zip(bars, accuracies, counts):
        height = bar.get_height()
        ax.annotate(
            f"{acc:.0%}\n(n={cnt})",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Number of Heavy Atoms", fontsize=12)
    ax.set_ylabel("Exact Match Accuracy", fontsize=12)
    prefix = f"{title_prefix} " if title_prefix else ""
    total = sum(counts)
    ax.set_title(
        f"{prefix}Reconstruction Accuracy by Molecule Size (n={total})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, min(1.0, max(accuracies) * 1.3) if accuracies else 1.0)

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_test_dummy_models(
    device: torch.device,
) -> Tuple[HyperNet, FlowEdgeDecoder, int]:
    """
    Create dummy HyperNet and FlowEdgeDecoder models for testing mode.

    Creates minimal ZINC-compatible models with small dimensions that can
    run quickly without pretrained checkpoints.

    Args:
        device: Device to create models on.

    Returns:
        Tuple of (hypernet, decoder, base_hdc_dim) where base_hdc_dim is 256.
    """
    base_hdc_dim = 256

    # Create minimal HyperNet (ZINC config with 5 features)
    node_feature_config = FeatureConfig(
        count=math.prod([9, 6, 3, 4, 2]),  # ZINC: 9*6*3*4*2 = 1296
        encoder_cls=CombinatoricIntegerEncoder,
        index_range=IndexRange((0, 5)),
        bins=[9, 6, 3, 4, 2],
    )
    config = DSHDCConfig(
        name="TEST_ZINC",
        hv_dim=base_hdc_dim,
        vsa=VSAModel.HRR,
        base_dataset="zinc",
        hypernet_depth=3,
        device=str(device),
        seed=42,
        normalize=True,
        dtype="float64",
        node_feature_configs=OrderedDict([(Features.NODE_FEATURES, node_feature_config)]),
    )
    hypernet = HyperNet(config)

    # Create minimal FlowEdgeDecoder (24-dim node features)
    decoder = FlowEdgeDecoder(
        num_node_classes=NODE_FEATURE_DIM,
        hdc_dim=2 * base_hdc_dim,  # Concatenated [order_0 | order_N]
        condition_dim=64,
        n_layers=2,
        hidden_dim=64,
        hidden_mlp_dim=128,
    )

    return hypernet, decoder, base_hdc_dim
