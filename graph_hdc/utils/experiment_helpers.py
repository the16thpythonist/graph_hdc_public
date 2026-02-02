"""
Experiment helper utilities for FlowEdgeDecoder training.

Helper functions and callbacks extracted from experiment files to enable
code reuse across base and child experiments via PyComex inheritance.
"""

from __future__ import annotations

import math
import signal
from collections import Counter, OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

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
from graph_hdc.models.flow_edge_decoder import (
    ZINC_ATOM_TYPES,
    NUM_ATOM_CLASSES,
)

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
) -> HyperNet:
    """
    Load encoder from checkpoint or create a new one.

    Args:
        config_path: Path to saved encoder checkpoint (empty string to create new)
        dataset: Dataset name
        hv_dim: Hypervector dimension
        depth: Message passing depth
        device: Device to load to

    Returns:
        HyperNet encoder instance (in eval mode)
    """
    if config_path and Path(config_path).exists():
        hypernet = HyperNet.load(config_path, device=str(device))
    else:
        config = create_hdc_config(dataset, hv_dim, depth, device=str(device))
        hypernet = HyperNet(config)
        hypernet = hypernet.to(device)

    hypernet.eval()
    return hypernet


# ========= HDC Decoding =========


def decode_nodes_from_hdc(
    hypernet: HyperNet,
    hdc_vector: Tensor,
    base_hdc_dim: int,
) -> Tuple[List[Tuple[int, ...]], int]:
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

    Returns:
        Tuple of (list of node tuples, number of nodes)
        Each tuple is (atom_idx, degree, charge, num_hs, is_ring)
    """
    # Extract order_0 from the first half of concatenated embedding
    if hdc_vector.dim() == 1:
        order_zero = hdc_vector[:base_hdc_dim].unsqueeze(0)
    else:
        order_zero = hdc_vector[:, :base_hdc_dim]

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
        """
        Initialize the callback with a PyComex experiment.

        Args:
            experiment: PyComex Experiment instance for tracking metrics
        """
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
        """Track validation loss at the end of each validation epoch."""
        val_loss = trainer.callback_metrics.get("val/loss")
        if val_loss is not None:
            self.experiment.track("loss_val", float(val_loss))


class ReconstructionVisualizationCallback(Callback):
    """
    Visualize molecule reconstructions at each validation epoch.

    Takes fixed validation samples, generates reconstructions using the model,
    and creates a 2xN figure comparing originals with reconstructions.
    """

    def __init__(
        self,
        experiment: "Experiment",
        vis_samples: List[Data],
        sample_steps: int,
        eta: float,
        omega: float,
        time_distortion: str,
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
        """
        super().__init__()
        self.experiment = experiment
        self.vis_samples = vis_samples
        self.sample_steps = sample_steps
        self.eta = eta
        self.omega = omega
        self.time_distortion = time_distortion

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Generate reconstructions and visualize."""
        if len(self.vis_samples) == 0:
            return

        # Collect inputs for batch sampling
        hdc_vectors_list = []
        node_features_list = []
        original_mols = []
        max_nodes = 0

        for data in self.vis_samples:
            # Parse original molecule from SMILES
            original_mol = Chem.MolFromSmiles(data.smiles)
            original_mols.append(original_mol)

            # Get preprocessed HDC vector (already concatenated [order_0 | order_N])
            hdc_vec = data.hdc_vector
            if hdc_vec.dim() == 2:
                hdc_vec = hdc_vec.squeeze(0)
            hdc_vectors_list.append(hdc_vec)

            # Get 24-dim one-hot node features
            node_features_list.append(data.x)
            max_nodes = max(max_nodes, data.x.size(0))

        # Pad to common size and stack
        device = pl_module.device
        batch_size = len(self.vis_samples)
        hdc_dim = hdc_vectors_list[0].size(-1)
        node_feature_dim = node_features_list[0].size(-1)

        hdc_vectors = torch.stack(hdc_vectors_list, dim=0).to(device).float()

        # Pad node features and create masks
        node_features = torch.zeros(batch_size, max_nodes, node_feature_dim, device=device)
        node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)

        for i, nf in enumerate(node_features_list):
            n = nf.size(0)
            node_features[i, :n] = nf.to(device)
            node_mask[i, :n] = True

        # Generate reconstructions one at a time with logging
        pl_module.eval()
        reconstructed_samples = []
        self.experiment.log(f"\nGenerating {batch_size} reconstructions ({self.sample_steps} steps each)...")

        with torch.no_grad():
            for i in range(batch_size):
                sample = pl_module.sample(
                    hdc_vectors=hdc_vectors[i : i + 1],
                    node_features=node_features[i : i + 1],
                    node_mask=node_mask[i : i + 1],
                    sample_steps=self.sample_steps,
                    eta=self.eta,
                    omega=self.omega,
                    time_distortion=self.time_distortion,
                    device=device,
                    show_progress=False,
                )
                reconstructed_samples.extend(sample)
                self.experiment.log(f"  Reconstructed molecule {i + 1}/{batch_size}")

        # Convert reconstructions to RDKit molecules
        reconstructed_mols = [pyg_to_mol(data) for data in reconstructed_samples]

        # Compute Tanimoto similarities
        similarities = []
        for orig_mol, recon_mol in zip(original_mols, reconstructed_mols):
            sim = compute_tanimoto_similarity(orig_mol, recon_mol)
            similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

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
            orig_smiles = self.vis_samples[i].smiles
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
            axes[1, i].set_title(
                f"Recon (Tan={similarities[i]:.3f})\n{truncated_recon}",
                fontsize=8,
            )

        fig.suptitle(
            f"Validation Reconstructions (Epoch {trainer.current_epoch}, Avg Tanimoto: {avg_similarity:.3f})"
        )
        plt.tight_layout()

        # Track figure and metric with PyComex
        self.experiment.track("validation_reconstructions", fig)
        self.experiment.track("tanimoto_similarity", avg_similarity)

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
