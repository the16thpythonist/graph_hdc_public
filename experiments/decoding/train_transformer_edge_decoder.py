#!/usr/bin/env python
"""
Train TransformerEdgeDecoder with Streaming Fragment Dataset.

This experiment trains a single-shot edge prediction model using a streaming dataset
that generates infinite molecular data by combining BRICS fragments on-the-fly.

Key differences from FlowEdgeDecoder training:
1. No iterative denoising (single forward pass)
2. No time embedding in global features
3. No edge marginals or noise distribution needed
4. No sampling parameters (eta, omega, sample_steps)

Usage:
    # Quick test
    python train_transformer_edge_decoder.py --__TESTING__ True

    # Full training
    python train_transformer_edge_decoder.py --__DEBUG__ False --EPOCHS 50

    # Custom configuration
    python train_transformer_edge_decoder.py --NUM_FRAGMENT_WORKERS 8 --BUFFER_SIZE 20000
"""

from __future__ import annotations

import math
import os
import random
import signal
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph_hdc.datasets.streaming_fragments import (
    FragmentLibrary,
    StreamingFragmentDataLoader,
    StreamingFragmentDataset,
)
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
from graph_hdc.models.flow_edge_decoder import FLOW_ATOM_TYPES, preprocess_dataset
from graph_hdc.models.transformer_edge_decoder import TransformerEdgeDecoder

# =============================================================================
# PARAMETERS
# =============================================================================

# Dataset configuration
DATASET: str = "zinc"  # Base dataset for fragments and validation

# HDC Encoder configuration
HDC_CONFIG_PATH: str = ""  # Path to saved HyperNet checkpoint (.ckpt)
HDC_DIM: int = 1024  # Hypervector dimension (used if HDC_CONFIG_PATH is empty)
HDC_DEPTH: int = 4  # Message passing depth (used if HDC_CONFIG_PATH is empty)

# Streaming dataset configuration
BUFFER_SIZE: int = 10000  # Number of samples in buffer
NUM_FRAGMENT_WORKERS: int = 4  # Number of worker processes generating samples
FRAGMENTS_RANGE_MIN: int = 2  # Minimum fragments per molecule
FRAGMENTS_RANGE_MAX: int = 4  # Maximum fragments per molecule
MAX_GENERATED_NODES: int = 50  # Maximum atoms in generated molecules
PREFILL_FRACTION: float = 0.1  # Wait for buffer to reach this fraction before training
STEPS_PER_EPOCH: int = 1000  # Training steps per epoch

# Validation configuration
NUM_VALID_SAMPLES: int = 1000  # Number of ZINC samples for validation

# Model architecture
N_LAYERS: int = 6
HIDDEN_DIM: int = 512
HIDDEN_MLP_DIM: int = 512
N_HEADS: int = 8
DROPOUT: float = 0.0
CONDITION_DIM: int = 512  # Reduced dimension for HDC conditioning after MLP

# Training hyperparameters
EPOCHS: int = 500
BATCH_SIZE: int = 32
LEARNING_RATE: float = 0.5e-4
WEIGHT_DECAY: float = 1e-5
GRADIENT_CLIP_VAL: float = 1.0

# Extra features
EXTRA_FEATURES_TYPE: str = "rrwp"
RRWP_STEPS: int = 10

# System configuration
SEED: int = 1
PRECISION: str = "32"

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
    """Create a HyperNet configuration from parameters."""
    dataset = dataset.lower()

    if dataset == "qm9":
        node_feature_config = FeatureConfig(
            count=math.prod([4, 5, 3, 5]),
            encoder_cls=CombinatoricIntegerEncoder,
            index_range=IndexRange((0, 4)),
            bins=[4, 5, 3, 5],
        )
        base_dataset = "qm9"
    elif dataset == "zinc":
        node_feature_config = FeatureConfig(
            count=math.prod([9, 6, 3, 4, 2]),
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
    """Load encoder from checkpoint or create a new one."""
    if config_path and Path(config_path).exists():
        hypernet = HyperNet.load(config_path, device=str(device))
    else:
        config = create_hdc_config(dataset, hv_dim, depth, device=str(device))
        hypernet = HyperNet(config)
        hypernet = hypernet.to(device)

    hypernet.eval()
    return hypernet


# =============================================================================
# Callbacks
# =============================================================================


class LossTrackingCallback(Callback):
    """Track training and validation loss per epoch."""

    def __init__(self, experiment: Experiment):
        super().__init__()
        self.experiment = experiment

    def on_train_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        train_loss = trainer.callback_metrics.get("train/loss")
        if train_loss is not None:
            self.experiment.track("loss_train", float(train_loss))

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        val_loss = trainer.callback_metrics.get("val/loss")
        if val_loss is not None:
            self.experiment.track("loss_val", float(val_loss))


class StreamingDatasetCleanupCallback(Callback):
    """Cleanup streaming dataset workers on training end."""

    def __init__(self, streaming_loader: StreamingFragmentDataLoader):
        super().__init__()
        self.streaming_loader = streaming_loader

    def on_train_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        self.streaming_loader.stop()


class GracefulInterruptHandler:
    """
    Context manager for handling CTRL+C gracefully during training.

    First CTRL+C: Sets trainer.should_stop = True for graceful shutdown
    Second CTRL+C: Forces immediate exit
    """

    def __init__(self, trainer: Optional[Trainer] = None):
        self.trainer = trainer
        self.original_handler = None
        self.interrupt_count = 0

    def set_trainer(self, trainer: Trainer) -> None:
        """Set the trainer after initialization."""
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


# =============================================================================
# Reconstruction Helpers
# =============================================================================


def pyg_to_mol(data: Data, skip_sanitization: bool = True) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    """
    Convert PyG Data to RDKit Mol.

    Args:
        data: PyG Data with x (node features) and edge_index, edge_attr
        skip_sanitization: If True, skip sanitization to allow drawing invalid molecules

    Returns:
        Tuple of (RDKit Mol or None, error message or None)
    """
    try:
        mol = Chem.RWMol()

        # Get node types
        if data.x.dim() > 1:
            node_types = data.x.argmax(dim=-1).cpu().numpy()
        else:
            node_types = data.x.cpu().numpy()

        # Add atoms
        atom_map = {}
        for i, atom_type_idx in enumerate(node_types):
            atom_symbol = FLOW_ATOM_TYPES[int(atom_type_idx)]
            atom = Chem.Atom(atom_symbol)
            rdkit_idx = mol.AddAtom(atom)
            atom_map[i] = rdkit_idx

        # Get edge info
        if data.edge_index is None or data.edge_index.numel() == 0:
            mol = mol.GetMol()
            return mol, None

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

        # Optionally try sanitization but don't fail if it doesn't work
        sanitization_error = None
        if not skip_sanitization:
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                sanitization_error = f"Sanitization: {str(e)[:50]}"

        return mol, sanitization_error

    except Exception as e:
        return None, f"PyG->Mol: {str(e)[:50]}"


def create_placeholder_image(size: Tuple[int, int] = (300, 300)) -> Image.Image:
    """Create a placeholder image for failed reconstructions."""
    img = Image.new("RGB", size, color=(240, 240, 240))
    return img


class ReconstructionVisualizationCallback(Callback):
    """
    Visualize reconstruction quality during validation.

    Creates a 10x2 grid showing original vs reconstructed molecules
    for a fixed set of validation samples.
    """

    def __init__(
        self,
        experiment: Experiment,
        validation_samples: List[Data],
        original_smiles: List[str],
        image_size: Tuple[int, int] = (300, 300),
    ):
        super().__init__()
        self.experiment = experiment
        self.validation_samples = validation_samples
        self.original_smiles = original_smiles
        self.image_size = image_size

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        """Generate reconstruction plots at the end of each validation epoch."""
        fig = self._create_comparison_plot(pl_module)
        epoch = trainer.current_epoch
        self.experiment.commit_fig(f"reconstruction_epoch_{epoch:03d}.png", fig)
        plt.close(fig)

    def _create_comparison_plot(self, model: TransformerEdgeDecoder) -> plt.Figure:
        """Create a 2xN grid comparing original (top) vs reconstructed (bottom) molecules."""
        n_samples = len(self.validation_samples)
        fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 7))

        # Handle single sample case
        if n_samples == 1:
            axes = axes.reshape(2, 1)

        for idx in range(n_samples):
            sample = self.validation_samples[idx]
            smiles = self.original_smiles[idx]

            # Draw original molecule (top row)
            orig_img = self._draw_molecule_from_smiles(smiles)
            axes[0, idx].imshow(orig_img)
            axes[0, idx].axis("off")
            truncated_smiles = smiles[:20] + "..." if len(smiles) > 20 else smiles
            axes[0, idx].set_title(f"Original\n{truncated_smiles}", fontsize=7)

            # Reconstruct and draw (bottom row)
            reconstructed_mol, error_msg = self._reconstruct_molecule(model, sample)
            recon_img = self._draw_molecule(reconstructed_mol)
            axes[1, idx].imshow(recon_img)
            axes[1, idx].axis("off")

            if reconstructed_mol is not None:
                try:
                    recon_smiles = Chem.MolToSmiles(reconstructed_mol)
                    truncated_recon = recon_smiles[:20] + "..." if len(recon_smiles) > 20 else recon_smiles
                    title = f"Reconstructed\n{truncated_recon}"
                    if error_msg:
                        title += f"\n({error_msg[:30]})"
                    axes[1, idx].set_title(title, fontsize=7)
                except Exception:
                    axes[1, idx].set_title("Reconstructed\n(SMILES failed)", fontsize=7)
            else:
                # Show error message if available
                error_display = error_msg[:35] if error_msg else "Unknown error"
                axes[1, idx].set_title(f"Failed\n{error_display}", fontsize=6, color="red")

        fig.suptitle("Validation Reconstructions", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig

    def _reconstruct_molecule(
        self, model: TransformerEdgeDecoder, sample: Data
    ) -> Tuple[Optional[Chem.Mol], Optional[str]]:
        """
        Reconstruct a molecule from HDC vector + ground truth nodes.

        Returns:
            Tuple of (molecule or None, error message or None)
        """
        try:
            device = next(model.parameters()).device

            # Prepare inputs
            # HDC vector might be [hdc_dim] or [1, hdc_dim] - ensure it's [1, hdc_dim]
            hdc_vector = sample.hdc_vector.to(device).float().view(1, -1)
            num_nodes = sample.x.size(0)

            # Create node features tensor (already one-hot from preprocessing)
            node_features = sample.x.unsqueeze(0).to(device).float()

            # Pad or truncate to max_nodes
            max_nodes = model.max_nodes
            if num_nodes < max_nodes:
                padding = torch.zeros(1, max_nodes - num_nodes, node_features.size(-1), device=device)
                node_features = torch.cat([node_features, padding], dim=1)
            elif num_nodes > max_nodes:
                # Truncate if sample has more nodes than model supports
                node_features = node_features[:, :max_nodes, :]
                num_nodes = max_nodes

            # Create node mask
            node_mask = torch.zeros(1, max_nodes, dtype=torch.bool, device=device)
            node_mask[0, :num_nodes] = True

            # Predict edges using model.sample()
            samples = model.sample(
                hdc_vectors=hdc_vector,
                node_features=node_features,
                node_mask=node_mask,
                device=device,
            )

            if samples and len(samples) > 0:
                mol, mol_error = pyg_to_mol(samples[0])
                return mol, mol_error
            return None, "model.sample() returned empty"

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)[:80]}"
            return None, error_msg

    def _draw_molecule_from_smiles(self, smiles: str) -> Image.Image:
        """Draw molecule from SMILES string."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Draw.MolToImage(mol, size=self.image_size)
        except Exception:
            pass
        return create_placeholder_image(self.image_size)

    def _draw_molecule(self, mol: Optional[Chem.Mol]) -> Image.Image:
        """Draw molecule or return placeholder. Forces drawing even for invalid molecules."""
        if mol is None:
            return create_placeholder_image(self.image_size)
        try:
            # Try normal drawing first
            return Draw.MolToImage(mol, size=self.image_size)
        except Exception:
            # If normal drawing fails, try with kekulize=False to handle aromatic issues
            try:
                return Draw.MolToImage(mol, size=self.image_size, kekulize=False)
            except Exception:
                # Last resort: create a copy and try to force 2D coords
                try:
                    from rdkit.Chem import AllChem
                    mol_copy = Chem.RWMol(mol)
                    AllChem.Compute2DCoords(mol_copy)
                    return Draw.MolToImage(mol_copy.GetMol(), size=self.image_size, kekulize=False)
                except Exception:
                    return create_placeholder_image(self.image_size)


# =============================================================================
# EXPERIMENT
# =============================================================================


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    """Train TransformerEdgeDecoder with streaming fragment dataset."""

    # =========================================================================
    # Setup
    # =========================================================================

    # Fix for PyTorch Lightning's _atomic_save
    custom_tmpdir = Path(e.path) / ".tmp_checkpoints"
    custom_tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(custom_tmpdir)
    tempfile.tempdir = str(custom_tmpdir)

    pl.seed_everything(e.SEED)

    e.log("=" * 60)
    e.log("TransformerEdgeDecoder Training (Streaming Fragment Dataset)")
    e.log("=" * 60)
    e.log(f"Base dataset: {e.DATASET.upper()}")
    e.log(f"HDC config: dim={e.HDC_DIM}, depth={e.HDC_DEPTH}")
    e.log(f"Streaming: {e.NUM_FRAGMENT_WORKERS} workers, buffer={e.BUFFER_SIZE}")
    e.log(f"Fragments per molecule: {e.FRAGMENTS_RANGE_MIN}-{e.FRAGMENTS_RANGE_MAX}")
    e.log(f"Steps per epoch: {e.STEPS_PER_EPOCH}")
    e.log(f"Training: {e.EPOCHS} epochs, batch size {e.BATCH_SIZE}")
    e.log(f"Debug mode: {e.__DEBUG__}")
    e.log("=" * 60)

    # Store config
    e["config/dataset"] = e.DATASET
    e["config/hdc_dim"] = e.HDC_DIM
    e["config/hdc_depth"] = e.HDC_DEPTH
    e["config/buffer_size"] = e.BUFFER_SIZE
    e["config/num_fragment_workers"] = e.NUM_FRAGMENT_WORKERS
    e["config/steps_per_epoch"] = e.STEPS_PER_EPOCH
    e["config/epochs"] = e.EPOCHS
    e["config/batch_size"] = e.BATCH_SIZE

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

    actual_hdc_dim = hypernet.hv_dim
    e.log(f"HyperNet initialized: hv_dim={actual_hdc_dim}, depth={hypernet.depth}")

    # Save encoder checkpoint for workers
    encoder_path = Path(e.path) / "hypernet_encoder.ckpt"
    hypernet.save(encoder_path)
    e.log(f"Saved encoder to: {encoder_path}")
    e["results/encoder_path"] = str(encoder_path)

    # =========================================================================
    # Build Fragment Library
    # =========================================================================

    e.log("\nBuilding fragment library from ZINC...")

    # Load ZINC training data
    zinc_train = get_split("train", dataset="zinc")
    e.log(f"ZINC train size: {len(zinc_train)}")

    # Build fragment library
    fragment_library = FragmentLibrary(min_atoms=2, max_atoms=30)

    if e.__DEBUG__:
        # Use subset for debugging
        fragment_library.build_from_dataset(
            zinc_train,
            show_progress=True,
            max_molecules=1000,
        )
    else:
        fragment_library.build_from_dataset(
            zinc_train,
            show_progress=True,
        )

    e.log(f"Fragment library built: {fragment_library.num_fragments} fragments")
    e["data/num_fragments"] = fragment_library.num_fragments

    # Save fragment library
    library_path = Path(e.path) / "fragment_library.pkl"
    fragment_library.save(library_path)
    e.log(f"Saved fragment library to: {library_path}")

    # =========================================================================
    # Create Validation Data from ZINC
    # =========================================================================

    e.log("\nPreparing validation data from ZINC...")

    zinc_valid = get_split("valid", dataset="zinc")
    num_valid = min(e.NUM_VALID_SAMPLES, len(zinc_valid))
    zinc_valid_subset = list(zinc_valid)[:num_valid]

    valid_data = preprocess_dataset(
        zinc_valid_subset,
        hypernet,
        "zinc",
        device=device,
        show_progress=True,
    )

    e.log(f"Validation data: {len(valid_data)} samples")
    e["data/valid_size"] = len(valid_data)

    # Get max_nodes from validation data
    max_nodes = max(d.x.size(0) for d in valid_data) if valid_data else 50
    max_nodes = max(max_nodes, e.MAX_GENERATED_NODES)  # Account for larger generated molecules
    e.log(f"Max nodes: {max_nodes}")
    e["data/max_nodes"] = max_nodes

    # Use smaller batch size in debug mode for limited GPU memory
    batch_size = 16 if e.__DEBUG__ else e.BATCH_SIZE
    if e.__DEBUG__:
        e.log(f"DEBUG: Using smaller batch size: {batch_size}")

    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # =========================================================================
    # Select Fixed Samples for Reconstruction Visualization
    # =========================================================================

    e.log("\nSelecting samples for reconstruction visualization...")

    # Use fixed seed for reproducibility
    random.seed(e.SEED)
    num_recon_samples = min(10, len(valid_data))
    reconstruction_indices = random.sample(range(len(valid_data)), num_recon_samples)
    reconstruction_samples = [valid_data[i] for i in reconstruction_indices]
    reconstruction_smiles = [valid_data[i].smiles for i in reconstruction_indices]

    e.log(f"Selected {num_recon_samples} samples for reconstruction visualization")
    e["config/reconstruction_indices"] = reconstruction_indices

    # =========================================================================
    # Create Streaming Training Dataset
    # =========================================================================

    e.log("\nCreating streaming fragment dataset...")

    streaming_dataset = StreamingFragmentDataset(
        fragment_library=fragment_library,
        hypernet_checkpoint_path=encoder_path,
        buffer_size=e.BUFFER_SIZE,
        num_workers=e.NUM_FRAGMENT_WORKERS,
        fragments_range=(e.FRAGMENTS_RANGE_MIN, e.FRAGMENTS_RANGE_MAX),
        max_nodes=e.MAX_GENERATED_NODES,
        dataset_name="zinc",
        prefill_fraction=e.PREFILL_FRACTION,
    )

    train_loader = StreamingFragmentDataLoader(
        dataset=streaming_dataset,
        batch_size=batch_size,
        steps_per_epoch=e.STEPS_PER_EPOCH,
    )

    e.log(f"Streaming loader: {len(train_loader)} steps per epoch")
    e.log(f"Validation loader: {len(valid_loader)} batches")

    # =========================================================================
    # Test Streaming DataLoader
    # =========================================================================

    e.log("\nTesting streaming dataloader (3 batches)...")
    try:
        train_loader.test_iteration(num_batches=3)
        e.log("Streaming dataloader test passed!")
    except Exception as ex:
        e.log(f"ERROR: Streaming dataloader test failed: {ex}")
        train_loader.stop()
        raise

    # =========================================================================
    # Create Model
    # =========================================================================

    e.log("\nCreating TransformerEdgeDecoder model...")

    concat_hdc_dim = 2 * actual_hdc_dim
    e.log(f"Concatenated HDC dim: {concat_hdc_dim}")
    e["config/concat_hdc_dim"] = concat_hdc_dim

    # Cap max_nodes to avoid OOM
    model_max_nodes = min(max_nodes + 5, 55)
    e.log(f"Model max_nodes: {model_max_nodes}")

    model = TransformerEdgeDecoder(
        num_node_classes=7,
        num_edge_classes=5,
        hdc_dim=concat_hdc_dim,
        condition_dim=e.CONDITION_DIM,
        n_layers=e.N_LAYERS,
        hidden_dim=e.HIDDEN_DIM,
        hidden_mlp_dim=e.HIDDEN_MLP_DIM,
        n_heads=e.N_HEADS,
        dropout=e.DROPOUT,
        max_nodes=model_max_nodes,
        extra_features_type=e.EXTRA_FEATURES_TYPE,
        rrwp_steps=e.RRWP_STEPS,
        lr=e.LEARNING_RATE,
        weight_decay=e.WEIGHT_DECAY,
    )

    e.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    e["model/num_parameters"] = sum(p.numel() for p in model.parameters())

    # =========================================================================
    # Setup Training
    # =========================================================================

    e.log("\nSetting up training...")

    # Create checkpoint directory
    checkpoint_dir = Path(e.path) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    e.log(f"Checkpoints will be saved to: {checkpoint_dir}")

    callbacks = [
        # Best model checkpoint (based on validation loss)
        ModelCheckpoint(
            dirpath=e.path,
            filename="best-{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        # Save checkpoint after every epoch (overwrites previous)
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="latest",
            save_top_k=1,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            enable_version_counter=False,  # Prevents adding version suffix
        ),
        LearningRateMonitor(logging_interval="epoch"),
        LossTrackingCallback(experiment=e),
        StreamingDatasetCleanupCallback(train_loader),
        ReconstructionVisualizationCallback(
            experiment=e,
            validation_samples=reconstruction_samples,
            original_smiles=reconstruction_smiles,
        ),
    ]

    logger = CSVLogger(e.path, name="logs")

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
    e.log("(Press CTRL+C to gracefully stop training)")
    e.log("-" * 40)

    interrupted = False
    with GracefulInterruptHandler() as interrupt_handler:
        interrupt_handler.set_trainer(trainer)
        try:
            trainer.fit(model, train_loader, valid_loader)
        except KeyboardInterrupt:
            interrupted = True
            e.log("\nTraining interrupted by user (force quit)")
        finally:
            # Ensure workers are stopped
            train_loader.stop()

    if trainer.should_stop and not interrupted:
        e.log("-" * 40)
        e.log("Training gracefully stopped by user")
    elif not interrupted:
        e.log("-" * 40)
        e.log("Training complete!")

    # =========================================================================
    # Save Results
    # =========================================================================

    best_val_loss = trainer.callback_metrics.get("val/loss")
    if best_val_loss is not None:
        e["results/best_val_loss"] = float(best_val_loss)
        e.log(f"Best validation loss: {best_val_loss:.4f}")

    final_path = Path(e.path) / "final_model.ckpt"
    model.save(str(final_path))
    e.log(f"Saved final model to: {final_path}")
    e["results/final_model_path"] = str(final_path)

    e.log("\n" + "=" * 60)
    e.log("Experiment complete!")
    e.log("=" * 60)


experiment.run_if_main()
