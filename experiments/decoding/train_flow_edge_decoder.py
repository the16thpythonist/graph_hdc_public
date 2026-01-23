#!/usr/bin/env python
"""
Train FlowEdgeDecoder - Edge-only DeFoG decoder conditioned on HDC vectors.

This experiment trains a discrete flow matching model that generates molecular
edges given:
1. Pre-computed HDC vectors as conditioning
2. Fixed node types (7 atom classes: C, N, O, F, S, Cl, Br)

The model learns to predict edges (5 classes: no-edge, single, double, triple,
aromatic) through a denoising process while keeping nodes fixed.

Usage:
    # Quick test
    python train_flow_edge_decoder.py --__TESTING__ True

    # Full training on QM9 with new encoder
    python train_flow_edge_decoder.py --__DEBUG__ False --EPOCHS 100

    # Train with existing encoder checkpoint
    python train_flow_edge_decoder.py --HDC_CONFIG_PATH /path/to/encoder.ckpt

    # Train on ZINC
    python train_flow_edge_decoder.py --DATASET zinc --__DEBUG__ False
"""

from __future__ import annotations

import math
import os
import random
import tempfile
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from defog.core import to_dense
from graph_hdc.datasets.utils import get_split
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
    FLOW_ATOM_TYPES,
    NUM_ATOM_CLASSES,
    NUM_EDGE_CLASSES,
    QM9_TO_7CLASS,
    ZINC_TO_7CLASS,
    FlowEdgeDecoder,
    compute_edge_marginals,
    compute_node_counts,
    preprocess_dataset,
)

# =============================================================================
# PARAMETERS
# =============================================================================

# Dataset configuration
DATASET: str = "qm9"  # "qm9" or "zinc"

# HDC Encoder configuration
# If HDC_CONFIG_PATH is provided, load encoder from checkpoint
# Otherwise, create a new encoder with the parameters below
HDC_CONFIG_PATH: str = ""  # Path to saved HyperNet checkpoint (.ckpt)
HDC_DIM: int = 512  # Hypervector dimension (used if HDC_CONFIG_PATH is empty)
HDC_DEPTH: int = 3  # Message passing depth (used if HDC_CONFIG_PATH is empty)

# Model architecture
N_LAYERS: int = 6
HIDDEN_DIM: int = 256
HIDDEN_MLP_DIM: int = 512
N_HEADS: int = 8
DROPOUT: float = 0.1

# Training hyperparameters
EPOCHS: int = 100
BATCH_SIZE: int = 32
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 1e-5
TRAIN_TIME_DISTORTION: str = "identity"
GRADIENT_CLIP_VAL: float = 1.0

# Sampling configuration
SAMPLE_STEPS: int = 100
ETA: float = 0.0
OMEGA: float = 0.0
SAMPLE_TIME_DISTORTION: str = "polydec"

# Extra features
EXTRA_FEATURES_TYPE: str = "rrwp"
RRWP_STEPS: int = 10

# Noise distribution
NOISE_TYPE: str = "marginal"

# System configuration
SEED: int = 42
NUM_WORKERS: int = 0  # Set to 0 to avoid AF_UNIX path too long errors with multiprocessing
PRECISION: str = "32"

# Reconstruction evaluation
NUM_RECONSTRUCTION_SAMPLES: int = 10  # Number of molecules to reconstruct and visualize

# Debug/Testing modes
__DEBUG__: bool = True
__TESTING__: bool = False


# =============================================================================
# Helper Functions
# =============================================================================


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
        HyperNet encoder instance
    """
    if config_path and Path(config_path).exists():
        # Load from checkpoint
        return HyperNet.load(config_path, device=str(device))
    else:
        # Create new encoder from parameters
        config = create_hdc_config(dataset, hv_dim, depth, device=str(device))
        return HyperNet(config)


def decode_nodes_from_hdc(
    hypernet: HyperNet,
    graph_embedding: torch.Tensor,
    dataset: str,
) -> tuple[list[int], int]:
    """
    Decode node types from HDC graph embedding.

    Args:
        hypernet: HyperNet encoder instance
        graph_embedding: HDC graph embedding tensor (hv_dim,)
        dataset: Dataset name ("qm9" or "zinc")

    Returns:
        Tuple of (list of 7-class atom type indices, number of nodes)
    """
    # Get the atom type mapping for this dataset
    if dataset.lower() == "qm9":
        dataset_to_7class = QM9_TO_7CLASS
    else:
        dataset_to_7class = ZINC_TO_7CLASS

    # Decode node counts from HDC embedding
    # Returns dict: {batch_idx: Counter({node_tuple: count, ...})}
    node_counter_dict = hypernet.decode_order_zero_counter(graph_embedding)

    # Get counter for batch 0 (single embedding)
    node_counter = node_counter_dict.get(0, Counter())

    # Convert to list of 7-class atom types
    atom_types_7class = []
    for node_tuple, count in node_counter.items():
        # node_tuple[0] is the atom type index in the dataset's encoding
        dataset_atom_idx = node_tuple[0]

        # Map to 7-class
        if dataset_atom_idx in dataset_to_7class:
            atom_7class = dataset_to_7class[dataset_atom_idx]
            atom_types_7class.extend([atom_7class] * count)

    return atom_types_7class, len(atom_types_7class)


def pyg_to_mol(data: Data) -> Optional[Chem.Mol]:
    """
    Convert FlowEdgeDecoder output (PyG Data) to RDKit Mol.

    The Data object has:
    - x: node types as class indices (n,) in 7-class format
    - edge_index: edge connectivity (2, num_edges)
    - edge_attr: edge types as class indices (num_edges,) in 5-class format

    Edge classes: 0=no-edge, 1=single, 2=double, 3=triple, 4=aromatic

    Args:
        data: PyG Data from FlowEdgeDecoder.sample()

    Returns:
        RDKit Mol or None if conversion fails
    """
    try:
        # Create editable mol
        mol = Chem.RWMol()

        # Get node types (class indices)
        if data.x.dim() > 1:
            node_types = data.x.argmax(dim=-1).cpu().numpy()
        else:
            node_types = data.x.cpu().numpy()

        # Add atoms
        atom_map = {}  # PyG node idx -> RDKit atom idx
        for i, atom_type_idx in enumerate(node_types):
            atom_symbol = FLOW_ATOM_TYPES[int(atom_type_idx)]
            atom = Chem.Atom(atom_symbol)
            rdkit_idx = mol.AddAtom(atom)
            atom_map[i] = rdkit_idx

        # Get edge info
        if data.edge_index is None or data.edge_index.numel() == 0:
            # No edges - return mol with just atoms
            mol = mol.GetMol()
            return mol

        edge_index = data.edge_index.cpu().numpy()
        if data.edge_attr is not None:
            if data.edge_attr.dim() > 1:
                edge_types = data.edge_attr.argmax(dim=-1).cpu().numpy()
            else:
                edge_types = data.edge_attr.cpu().numpy()
        else:
            # Default to single bonds
            edge_types = [1] * edge_index.shape[1]

        # Bond type mapping: 0=no-edge, 1=single, 2=double, 3=triple, 4=aromatic
        bond_type_map = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
            4: Chem.BondType.AROMATIC,
        }

        # Add bonds (only process upper triangle to avoid duplicates)
        added_bonds = set()
        for k in range(edge_index.shape[1]):
            i, j = int(edge_index[0, k]), int(edge_index[1, k])
            edge_type = int(edge_types[k])

            # Skip no-edge and self-loops
            if edge_type == 0 or i == j:
                continue

            # Skip if already added (since edges are bidirectional)
            bond_key = (min(i, j), max(i, j))
            if bond_key in added_bonds:
                continue
            added_bonds.add(bond_key)

            # Add bond
            if edge_type in bond_type_map:
                mol.AddBond(atom_map[i], atom_map[j], bond_type_map[edge_type])

        # Convert to regular mol and try to sanitize
        mol = mol.GetMol()

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            # Try without sanitization
            pass

        return mol

    except Exception:
        return None


def is_valid_mol(mol: Optional[Chem.Mol]) -> bool:
    """Check if molecule is valid."""
    if mol is None:
        return False
    try:
        smiles = Chem.MolToSmiles(mol)
        return smiles is not None and len(smiles) > 0
    except Exception:
        return False


def get_canonical_smiles(mol: Optional[Chem.Mol]) -> Optional[str]:
    """Get canonical SMILES from molecule."""
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


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


# =============================================================================
# EXPERIMENT
# =============================================================================


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    """Train FlowEdgeDecoder model."""

    # =========================================================================
    # Setup
    # =========================================================================

    # Fix for PyTorch Lightning's _atomic_save on systems with limited tmpfs
    custom_tmpdir = Path(e.path) / ".tmp_checkpoints"
    custom_tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(custom_tmpdir)
    tempfile.tempdir = str(custom_tmpdir)

    # Set seed
    pl.seed_everything(e.SEED)

    e.log("=" * 60)
    e.log("FlowEdgeDecoder Training")
    e.log("=" * 60)
    e.log(f"Dataset: {e.DATASET.upper()}")
    e.log(f"HDC config path: {e.HDC_CONFIG_PATH or '(creating new)'}")
    e.log(f"HDC dim: {e.HDC_DIM}, depth: {e.HDC_DEPTH}")
    e.log(f"Architecture: {e.N_LAYERS} layers, {e.HIDDEN_DIM} hidden dim")
    e.log(f"Training: {e.EPOCHS} epochs, batch size {e.BATCH_SIZE}")
    e.log(f"Debug mode: {e.__DEBUG__}")
    e.log("=" * 60)

    # Store config
    e["config/dataset"] = e.DATASET
    e["config/hdc_config_path"] = e.HDC_CONFIG_PATH
    e["config/hdc_dim"] = e.HDC_DIM
    e["config/hdc_depth"] = e.HDC_DEPTH
    e["config/n_layers"] = e.N_LAYERS
    e["config/hidden_dim"] = e.HIDDEN_DIM
    e["config/epochs"] = e.EPOCHS
    e["config/batch_size"] = e.BATCH_SIZE
    e["config/lr"] = e.LEARNING_RATE

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e.log(f"Using device: {device}")
    e["config/device"] = str(device)

    # =========================================================================
    # Load or Create HyperNet Encoder
    # =========================================================================

    e.log("\nLoading/Creating HyperNet encoder...")

    hypernet = load_or_create_encoder(
        config_path=e.HDC_CONFIG_PATH,
        dataset=e.DATASET,
        hv_dim=e.HDC_DIM,
        depth=e.HDC_DEPTH,
        device=device,
    )
    hypernet.eval()

    actual_hdc_dim = hypernet.hv_dim
    e.log(f"HyperNet initialized: hv_dim={actual_hdc_dim}, depth={hypernet.depth}")
    e["config/actual_hdc_dim"] = actual_hdc_dim
    e["config/actual_hdc_depth"] = hypernet.depth

    # Save the encoder if we created a new one
    if not e.HDC_CONFIG_PATH:
        encoder_path = Path(e.path) / "hypernet_encoder.ckpt"
        hypernet.save(encoder_path)
        e.log(f"Saved new encoder to: {encoder_path}")
        e["results/encoder_path"] = str(encoder_path)

    # =========================================================================
    # Load and Preprocess Datasets
    # =========================================================================

    e.log("\nLoading datasets...")
    train_ds = get_split("train", dataset=e.DATASET.lower())
    valid_ds = get_split("valid", dataset=e.DATASET.lower())

    e.log(f"Raw train: {len(train_ds)}, valid: {len(valid_ds)}")

    e.log("\nPreprocessing datasets for FlowEdgeDecoder...")
    e.log("(Computing HDC embeddings and converting features...)")

    train_data = preprocess_dataset(
        train_ds,
        hypernet,
        e.DATASET.lower(),
        device=device,
        show_progress=True,
    )
    valid_data = preprocess_dataset(
        valid_ds,
        hypernet,
        e.DATASET.lower(),
        device=device,
        show_progress=True,
    )

    e.log(f"Processed train: {len(train_data)}, valid: {len(valid_data)}")
    e["data/train_size"] = len(train_data)
    e["data/valid_size"] = len(valid_data)

    if len(train_data) == 0:
        e.log("ERROR: No training data after preprocessing!")
        return

    # =========================================================================
    # Compute Statistics
    # =========================================================================

    e.log("\nComputing edge marginals...")
    edge_marginals = compute_edge_marginals(train_data)
    e.log(f"Edge marginals: {edge_marginals.tolist()}")
    e["data/edge_marginals"] = edge_marginals.tolist()

    node_counts = compute_node_counts(train_data)
    max_nodes = int(node_counts.nonzero()[-1].item()) if node_counts.sum() > 0 else 50
    e.log(f"Max nodes: {max_nodes}")
    e["data/max_nodes"] = max_nodes

    # =========================================================================
    # Create Data Loaders
    # =========================================================================

    train_loader = DataLoader(
        train_data,
        batch_size=e.BATCH_SIZE,
        shuffle=True,
        num_workers=e.NUM_WORKERS,
        pin_memory=(e.NUM_WORKERS > 0),  # Only pin memory if using workers
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=e.BATCH_SIZE,
        shuffle=False,
        num_workers=e.NUM_WORKERS,
        pin_memory=(e.NUM_WORKERS > 0),
    )

    e.log(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

    # =========================================================================
    # Create Model
    # =========================================================================

    e.log("\nCreating FlowEdgeDecoder model...")

    model = FlowEdgeDecoder(
        num_node_classes=7,
        num_edge_classes=5,
        hdc_dim=actual_hdc_dim,
        n_layers=e.N_LAYERS,
        hidden_dim=e.HIDDEN_DIM,
        hidden_mlp_dim=e.HIDDEN_MLP_DIM,
        n_heads=e.N_HEADS,
        dropout=e.DROPOUT,
        noise_type=e.NOISE_TYPE,
        edge_marginals=edge_marginals,
        node_counts=node_counts,
        max_nodes=max_nodes + 10,  # Buffer
        extra_features_type=e.EXTRA_FEATURES_TYPE,
        rrwp_steps=e.RRWP_STEPS,
        lr=e.LEARNING_RATE,
        weight_decay=e.WEIGHT_DECAY,
        train_time_distortion=e.TRAIN_TIME_DISTORTION,
        sample_steps=e.SAMPLE_STEPS,
        eta=e.ETA,
        omega=e.OMEGA,
        sample_time_distortion=e.SAMPLE_TIME_DISTORTION,
    )

    e.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    e["model/num_parameters"] = sum(p.numel() for p in model.parameters())

    # =========================================================================
    # Setup Training
    # =========================================================================

    e.log("\nSetting up training...")

    # Callbacks (no EarlyStopping)
    callbacks = [
        ModelCheckpoint(
            dirpath=e.path,
            filename="best-{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Logger
    logger = CSVLogger(e.path, name="logs")

    # Trainer
    trainer = Trainer(
        max_epochs=e.EPOCHS,
        accelerator="auto",
        devices=1,
        precision=e.PRECISION,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=e.path,
        log_every_n_steps=10,
        gradient_clip_val=e.GRADIENT_CLIP_VAL,
        enable_progress_bar=True,
    )

    # =========================================================================
    # Train
    # =========================================================================

    e.log("\nStarting training...")
    e.log("-" * 40)

    trainer.fit(model, train_loader, valid_loader)

    e.log("-" * 40)
    e.log("Training complete!")

    # =========================================================================
    # Save Results
    # =========================================================================

    # Get best metrics
    best_val_loss = trainer.callback_metrics.get("val/loss")
    if best_val_loss is not None:
        e["results/best_val_loss"] = float(best_val_loss)
        e.log(f"Best validation loss: {best_val_loss:.4f}")

    # Save final model
    final_path = Path(e.path) / "final_model.ckpt"
    model.save(str(final_path))
    e.log(f"Model saved to: {final_path}")
    e["results/model_path"] = str(final_path)

    # Find best checkpoint
    best_ckpts = list(Path(e.path).glob("best-*.ckpt"))
    if best_ckpts:
        best_ckpt = best_ckpts[0]
        e.log(f"Best checkpoint: {best_ckpt}")
        e["results/best_checkpoint"] = str(best_ckpt)

    # =========================================================================
    # Verification
    # =========================================================================

    e.log("\nRunning verification...")

    try:
        # Load best model
        if best_ckpts:
            loaded_model = FlowEdgeDecoder.load(str(best_ckpts[0]), device=device)
        else:
            loaded_model = model

        loaded_model.eval()
        loaded_model.to(device)

        # Get a sample batch for verification
        sample_batch = next(iter(valid_loader))
        sample_batch = sample_batch.to(device)

        # Extract HDC vectors and node features for sampling
        with torch.no_grad():
            # Get dense representation
            dense_data, node_mask = to_dense(
                sample_batch.x,
                sample_batch.edge_index,
                sample_batch.edge_attr,
                sample_batch.batch,
            )

            # Get HDC vectors
            hdc_vectors = loaded_model._get_hdc_vectors_from_batch(sample_batch)

            # Sample edges
            num_test = min(5, hdc_vectors.size(0))
            samples = loaded_model.sample(
                hdc_vectors=hdc_vectors[:num_test],
                node_features=dense_data.X[:num_test],
                node_mask=node_mask[:num_test],
                sample_steps=20,  # Quick sampling for verification
                show_progress=False,
            )

            e.log(f"Generated {len(samples)} samples successfully")
            e["verification/num_samples"] = len(samples)
            e["verification/status"] = "passed"

    except Exception as ex:
        e.log(f"Verification failed: {ex}")
        e["verification/status"] = "failed"
        e["verification/error"] = str(ex)

    # =========================================================================
    # Reconstruction Evaluation
    # =========================================================================

    e.log("\n" + "=" * 60)
    e.log("Reconstruction Evaluation")
    e.log("=" * 60)

    try:
        # Load best model for reconstruction
        if best_ckpts:
            recon_model = FlowEdgeDecoder.load(str(best_ckpts[0]), device=device)
        else:
            recon_model = model
        recon_model.eval()
        recon_model.to(device)

        # Get raw validation dataset (with SMILES)
        raw_valid_ds = get_split("valid", dataset=e.DATASET.lower())

        # Select random samples
        num_samples = min(e.NUM_RECONSTRUCTION_SAMPLES, len(raw_valid_ds))
        sample_indices = random.sample(range(len(raw_valid_ds)), num_samples)

        e.log(f"Evaluating reconstruction on {num_samples} samples...")

        # Create output directory for plots
        recon_dir = Path(e.path) / "reconstruction_plots"
        recon_dir.mkdir(exist_ok=True)

        # Track metrics
        valid_count = 0
        match_count = 0
        results = []

        for idx, sample_idx in enumerate(sample_indices):
            e.log(f"\nSample {idx + 1}/{num_samples}:")

            # Get original data
            original_data = raw_valid_ds[sample_idx]
            original_smiles = original_data.smiles if hasattr(original_data, "smiles") else "N/A"
            original_mol = Chem.MolFromSmiles(original_smiles) if original_smiles != "N/A" else None

            e.log(f"  Original SMILES: {original_smiles}")

            # Encode with HyperNet
            # Need to add batch attribute for single graph
            data_for_encoding = original_data.clone()
            data_for_encoding = data_for_encoding.to(device)
            if not hasattr(data_for_encoding, "batch") or data_for_encoding.batch is None:
                data_for_encoding.batch = torch.zeros(
                    data_for_encoding.x.size(0), dtype=torch.long, device=device
                )

            with torch.no_grad():
                encoder_output = hypernet.forward(data_for_encoding, normalize=True)
                graph_embedding = encoder_output["graph_embedding"].squeeze(0)  # (hv_dim,)

            # Decode nodes from HDC embedding
            atom_types_7class, num_nodes = decode_nodes_from_hdc(
                hypernet, graph_embedding.unsqueeze(0), e.DATASET
            )

            e.log(f"  Decoded {num_nodes} nodes: {atom_types_7class}")

            # Handle case where no nodes were decoded
            if num_nodes == 0:
                e.log("  WARNING: No nodes decoded from HDC embedding, skipping...")
                results.append({
                    "sample_idx": sample_idx,
                    "original_smiles": original_smiles,
                    "generated_smiles": None,
                    "is_valid": False,
                    "is_match": False,
                    "error": "No nodes decoded",
                })
                continue

            # Prepare inputs for FlowEdgeDecoder.sample()
            # node_features: (1, n, 7) one-hot
            node_features = F.one_hot(
                torch.tensor(atom_types_7class, dtype=torch.long, device=device),
                num_classes=NUM_ATOM_CLASSES,
            ).float().unsqueeze(0)  # (1, n, 7)

            # node_mask: (1, n)
            node_mask = torch.ones(1, num_nodes, dtype=torch.bool, device=device)

            # hdc_vectors: (1, hdc_dim)
            hdc_vectors = graph_embedding.unsqueeze(0)

            # Generate edges
            with torch.no_grad():
                generated_samples = recon_model.sample(
                    hdc_vectors=hdc_vectors,
                    node_features=node_features,
                    node_mask=node_mask,
                    sample_steps=e.SAMPLE_STEPS,
                    show_progress=False,
                )

            # Convert to RDKit molecule
            generated_data = generated_samples[0]
            generated_mol = pyg_to_mol(generated_data)
            generated_smiles = get_canonical_smiles(generated_mol)

            e.log(f"  Generated SMILES: {generated_smiles or 'N/A'}")

            # Check validity and match
            is_valid = is_valid_mol(generated_mol)
            original_canonical = get_canonical_smiles(original_mol) if original_mol else None
            is_match = (
                is_valid
                and generated_smiles is not None
                and original_canonical is not None
                and generated_smiles == original_canonical
            )

            if is_valid:
                valid_count += 1
            if is_match:
                match_count += 1

            status = "MATCH" if is_match else ("Valid" if is_valid else "Invalid")
            e.log(f"  Status: {status}")

            # Create visualization
            if original_mol is not None:
                plot_path = recon_dir / f"reconstruction_{idx + 1:03d}.png"
                create_reconstruction_plot(
                    original_mol=original_mol,
                    generated_mol=generated_mol,
                    original_smiles=original_smiles,
                    generated_smiles=generated_smiles or "N/A",
                    is_valid=is_valid,
                    is_match=is_match,
                    sample_idx=idx,
                    save_path=plot_path,
                )

            # Store result
            results.append({
                "sample_idx": sample_idx,
                "original_smiles": original_smiles,
                "generated_smiles": generated_smiles,
                "is_valid": is_valid,
                "is_match": is_match,
            })

        # Log summary
        e.log("\n" + "-" * 40)
        e.log("Reconstruction Summary:")
        e.log(f"  Total samples: {num_samples}")
        e.log(f"  Valid molecules: {valid_count} ({100 * valid_count / num_samples:.1f}%)")
        e.log(f"  Exact matches: {match_count} ({100 * match_count / num_samples:.1f}%)")
        e.log("-" * 40)

        # Store metrics
        e["reconstruction/num_samples"] = num_samples
        e["reconstruction/valid_count"] = valid_count
        e["reconstruction/match_count"] = match_count
        e["reconstruction/valid_rate"] = valid_count / num_samples if num_samples > 0 else 0
        e["reconstruction/match_rate"] = match_count / num_samples if num_samples > 0 else 0
        e["reconstruction/results"] = results

        # Save results as JSON
        e.commit_json("reconstruction_results.json", {
            "num_samples": num_samples,
            "valid_count": valid_count,
            "match_count": match_count,
            "valid_rate": valid_count / num_samples if num_samples > 0 else 0,
            "match_rate": match_count / num_samples if num_samples > 0 else 0,
            "results": results,
        })

    except Exception as ex:
        e.log(f"Reconstruction evaluation failed: {ex}")
        import traceback
        e.log(traceback.format_exc())
        e["reconstruction/status"] = "failed"
        e["reconstruction/error"] = str(ex)

    e.log("\n" + "=" * 60)
    e.log("Experiment completed!")
    e.log("=" * 60)


# =============================================================================
# Testing Mode
# =============================================================================


@experiment.testing
def testing(e: Experiment):
    """Quick test mode with reduced parameters."""
    e.EPOCHS = 2
    e.BATCH_SIZE = 4
    e.SAMPLE_STEPS = 10


# =============================================================================
# Entry Point
# =============================================================================

experiment.run_if_main()
