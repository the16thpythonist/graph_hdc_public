#!/usr/bin/env python
"""
Train FlowEdgeDecoder - Edge-only DeFoG decoder conditioned on HDC vectors.

This experiment trains a discrete flow matching model that generates molecular
edges given:
1. Pre-computed HDC vectors as conditioning
2. Fixed node features (24-dim one-hot: atom type, degree, charge, Hs, ring)

The model learns to predict edges (5 classes: no-edge, single, double, triple,
aromatic) through a denoising process while keeping nodes fixed.

This is the BASE EXPERIMENT. Child experiments can inherit via Experiment.extend()
and override hooks for custom data loading behavior.

Usage:
    # Quick test
    python train_flow_edge_decoder.py --__TESTING__ True

    # Full training with new encoder
    python train_flow_edge_decoder.py --__DEBUG__ False --EPOCHS 100

    # Train with existing encoder checkpoint
    python train_flow_edge_decoder.py --HDC_CONFIG_PATH /path/to/encoder.ckpt
"""

from __future__ import annotations

import os
import random
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from rdkit import Chem
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph_hdc.datasets.utils import get_split
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.models.flow_edge_decoder import (
    NODE_FEATURE_DIM,
    FlowEdgeDecoder,
    compute_edge_marginals,
    compute_node_counts,
    preprocess_dataset,
    node_tuples_to_onehot,
)
from graph_hdc.utils.experiment_helpers import (
    GracefulInterruptHandler,
    LossTrackingCallback,
    ReconstructionVisualizationCallback,
    create_reconstruction_plot,
    decode_nodes_from_hdc,
    get_canonical_smiles,
    is_valid_mol,
    load_or_create_encoder,
    pyg_to_mol,
)
from graph_hdc.utils.helpers import scatter_hd

# =============================================================================
# PARAMETERS
# =============================================================================

# :param DATASET:
#     Dataset name for training. Currently only "zinc" is supported.
DATASET: str = "zinc"

# -----------------------------------------------------------------------------
# HDC Encoder Configuration
# -----------------------------------------------------------------------------

# :param HDC_CONFIG_PATH:
#     Path to saved HyperNet encoder checkpoint (.ckpt). If empty, creates a new
#     encoder with HDC_DIM and HDC_DEPTH parameters.
HDC_CONFIG_PATH: str = ""

# :param HDC_DIM:
#     Hypervector dimension for the HyperNet encoder. Only used if HDC_CONFIG_PATH
#     is empty. Typical values: 256, 512, 1024.
HDC_DIM: int = 1024

# :param HDC_DEPTH:
#     Message passing depth for the HyperNet encoder. Only used if HDC_CONFIG_PATH
#     is empty. Higher values capture longer-range structural information.
HDC_DEPTH: int = 6

# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------

# :param N_LAYERS:
#     Number of transformer layers in the FlowEdgeDecoder.
N_LAYERS: int = 6

# :param HIDDEN_DIM:
#     Hidden dimension for transformer layers.
HIDDEN_DIM: int = 256

# :param HIDDEN_MLP_DIM:
#     Hidden dimension for MLP blocks in transformer.
HIDDEN_MLP_DIM: int = 256

# :param N_HEADS:
#     Number of attention heads in transformer layers.
N_HEADS: int = 8

# :param DROPOUT:
#     Dropout probability in transformer layers.
DROPOUT: float = 0.0

# :param CONDITION_DIM:
#     Dimension for HDC conditioning after MLP projection.
CONDITION_DIM: int = 256

# :param TIME_EMBED_DIM:
#     Dimension for sinusoidal time embedding.
TIME_EMBED_DIM: int = 256

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------

# :param EPOCHS:
#     Number of training epochs.
EPOCHS: int = 50

# :param BATCH_SIZE:
#     Batch size for training and validation.
BATCH_SIZE: int = 32

# :param LEARNING_RATE:
#     Initial learning rate for Adam optimizer.
LEARNING_RATE: float = 1e-4

# :param WEIGHT_DECAY:
#     Weight decay (L2 regularization) for optimizer.
WEIGHT_DECAY: float = 1e-5

# :param TRAIN_TIME_DISTORTION:
#     Time distortion type during training. Options: "identity", "polydec".
TRAIN_TIME_DISTORTION: str = "identity"

# :param GRADIENT_CLIP_VAL:
#     Gradient clipping value. Set to 0.0 to disable.
GRADIENT_CLIP_VAL: float = 1.0

# -----------------------------------------------------------------------------
# Sampling Configuration
# -----------------------------------------------------------------------------

# :param SAMPLE_STEPS:
#     Number of denoising steps during sampling.
SAMPLE_STEPS: int = 1000

# :param ETA:
#     Stochasticity parameter for sampling. 0.0 = deterministic.
ETA: float = 0.0

# :param OMEGA:
#     Target guidance strength parameter.
OMEGA: float = 0.0

# :param SAMPLE_TIME_DISTORTION:
#     Time distortion type during sampling. Options: "identity", "polydec".
SAMPLE_TIME_DISTORTION: str = "polydec"

# -----------------------------------------------------------------------------
# Extra Features
# -----------------------------------------------------------------------------

# :param EXTRA_FEATURES_TYPE:
#     Type of extra positional features. Options: "rrwp", "none".
EXTRA_FEATURES_TYPE: str = "rrwp"

# :param RRWP_STEPS:
#     Number of random walk steps for RRWP positional encoding.
RRWP_STEPS: int = 10

# -----------------------------------------------------------------------------
# Noise Distribution
# -----------------------------------------------------------------------------

# :param NOISE_TYPE:
#     Noise distribution type. Options: "uniform", "marginal".
NOISE_TYPE: str = "uniform"

# -----------------------------------------------------------------------------
# System Configuration
# -----------------------------------------------------------------------------

# :param SEED:
#     Random seed for reproducibility.
SEED: int = 1

# :param NUM_WORKERS:
#     Number of data loader workers. Set to 0 to avoid multiprocessing issues.
NUM_WORKERS: int = 0

# :param PRECISION:
#     Training precision. Options: "32", "16", "bf16".
PRECISION: str = "32"

# -----------------------------------------------------------------------------
# Reconstruction Evaluation
# -----------------------------------------------------------------------------

# :param NUM_RECONSTRUCTION_SAMPLES:
#     Number of molecules to reconstruct and visualize after training.
NUM_RECONSTRUCTION_SAMPLES: int = 100

# :param NUM_VALIDATION_VISUALIZATIONS:
#     Number of molecules to visualize during each validation epoch.
NUM_VALIDATION_VISUALIZATIONS: int = 5

# -----------------------------------------------------------------------------
# Resume Training
# -----------------------------------------------------------------------------

# :param RESUME_CHECKPOINT_PATH:
#     Path to PyTorch Lightning checkpoint (.ckpt) to resume training from.
#     If set, RESUME_ENCODER_PATH must also be provided.
RESUME_CHECKPOINT_PATH: Optional[str] = None

# :param RESUME_ENCODER_PATH:
#     Path to HyperNet encoder checkpoint when resuming training.
#     Required if RESUME_CHECKPOINT_PATH is set.
RESUME_ENCODER_PATH: Optional[str] = None

# -----------------------------------------------------------------------------
# Debug/Testing Modes
# -----------------------------------------------------------------------------

# :param __DEBUG__:
#     Debug mode - reuses same output folder during development.
__DEBUG__: bool = True

# :param __TESTING__:
#     Testing mode - runs with minimal iterations for validation.
__TESTING__: bool = False


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

    # Validate resume configuration
    resuming = bool(e.RESUME_CHECKPOINT_PATH)
    if resuming:
        resume_ckpt_path = Path(e.RESUME_CHECKPOINT_PATH)
        if not resume_ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt_path}")

        if not e.RESUME_ENCODER_PATH:
            raise ValueError("RESUME_ENCODER_PATH is required when RESUME_CHECKPOINT_PATH is set")

        resume_encoder_path = Path(e.RESUME_ENCODER_PATH)
        if not resume_encoder_path.exists():
            raise FileNotFoundError(f"Resume encoder not found: {resume_encoder_path}")

    e.log("=" * 60)
    if resuming:
        e.log("FlowEdgeDecoder Training (RESUMING from checkpoint)")
    else:
        e.log("FlowEdgeDecoder Training")
    e.log("=" * 60)
    e.log(f"Dataset: {e.DATASET.upper()}")
    e.log(f"HDC config path: {e.HDC_CONFIG_PATH or '(creating new)'}")
    e.log(f"HDC dim: {e.HDC_DIM}, depth: {e.HDC_DEPTH}")
    e.log(f"Architecture: {e.N_LAYERS} layers, {e.HIDDEN_DIM} hidden dim")
    e.log(f"Training: {e.EPOCHS} epochs, batch size {e.BATCH_SIZE}")
    e.log(f"Debug mode: {e.__DEBUG__}")
    if resuming:
        e.log(f"Resuming from: {e.RESUME_CHECKPOINT_PATH}")
        e.log(f"Using encoder: {e.RESUME_ENCODER_PATH}")
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
    e["config/resuming"] = resuming

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e.log(f"Using device: {device}")
    e["config/device"] = str(device)

    # =========================================================================
    # Load or Create HyperNet Encoder
    # =========================================================================

    e.log("\nLoading/Creating HyperNet encoder...")

    if resuming:
        hypernet = HyperNet.load(str(resume_encoder_path), device=str(device))
        hypernet.eval()
    else:
        hypernet = load_or_create_encoder(
            config_path=e.HDC_CONFIG_PATH,
            dataset=e.DATASET,
            hv_dim=e.HDC_DIM,
            depth=e.HDC_DEPTH,
            device=device,
        )

    actual_hdc_dim = hypernet.hv_dim
    e.log(f"HyperNet initialized: hv_dim={actual_hdc_dim}, depth={hypernet.depth}")
    e["config/actual_hdc_dim"] = actual_hdc_dim
    e["config/actual_hdc_depth"] = hypernet.depth

    # Save the encoder if we created a new one
    if not e.HDC_CONFIG_PATH and not resuming:
        encoder_path = Path(e.path) / "hypernet_encoder.ckpt"
        hypernet.save(encoder_path)
        e.log(f"Saved new encoder to: {encoder_path}")
        e["results/encoder_path"] = str(encoder_path)

    # =========================================================================
    # Apply Hooks for Data Loading
    # =========================================================================

    train_loader, train_data = e.apply_hook(
        "load_train_data",
        hypernet=hypernet,
        device=device,
    )

    valid_data, valid_loader, vis_samples = e.apply_hook(
        "load_valid_data",
        hypernet=hypernet,
        device=device,
    )

    edge_marginals, node_counts, max_nodes = e.apply_hook(
        "compute_statistics",
        train_data=train_data,
        hypernet=hypernet,
        device=device,
    )

    # Store statistics
    e["data/edge_marginals"] = edge_marginals.tolist()
    e["data/max_nodes"] = max_nodes
    e["data/train_size"] = len(train_data) if train_data else "streaming"
    e["data/valid_size"] = len(valid_data)

    # =========================================================================
    # Create Model
    # =========================================================================

    e.log("\nCreating FlowEdgeDecoder model...")

    # HDC vector dimension is 2 * base_dim (concatenation of order_0 and order_N)
    concat_hdc_dim = 2 * actual_hdc_dim
    e.log(f"Concatenated HDC dim: {concat_hdc_dim} (order_0 + order_N)")
    e.log(f"Condition dim after MLP: {e.CONDITION_DIM}")
    e.log(f"Time embedding dim: {e.TIME_EMBED_DIM}")
    e["config/concat_hdc_dim"] = concat_hdc_dim
    e["config/condition_dim"] = e.CONDITION_DIM
    e["config/time_embed_dim"] = e.TIME_EMBED_DIM

    model = FlowEdgeDecoder(
        num_node_classes=NODE_FEATURE_DIM,  # 24-dim: atom + degree + charge + Hs + ring
        num_edge_classes=5,
        hdc_dim=concat_hdc_dim,  # Input dim is 2x base (concatenated)
        condition_dim=e.CONDITION_DIM,  # Reduced dim after MLP
        time_embed_dim=e.TIME_EMBED_DIM,  # Dimension for time embedding
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

    # Load weights if resuming
    if resuming:
        e.log(f"\nLoading model weights from checkpoint...")
        checkpoint = torch.load(resume_ckpt_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        e.log("Model weights loaded successfully!")
        if "epoch" in checkpoint:
            e.log(f"  (Checkpoint was from epoch {checkpoint['epoch']})")
            e["config/resume_from_epoch"] = checkpoint["epoch"]

    e.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    e["model/num_parameters"] = sum(p.numel() for p in model.parameters())

    # =========================================================================
    # Create Standard Callbacks
    # =========================================================================

    # Checkpoint directory for latest checkpoint
    checkpoint_dir = Path(e.path) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
            enable_version_counter=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        LossTrackingCallback(experiment=e),
        ReconstructionVisualizationCallback(
            experiment=e,
            vis_samples=vis_samples,
            sample_steps=e.SAMPLE_STEPS,
            eta=e.ETA,
            omega=e.OMEGA,
            time_distortion=e.SAMPLE_TIME_DISTORTION,
        ),
    ]

    # Allow child experiments to modify callbacks (add, remove, or change)
    callbacks = e.apply_hook(
        "modify_callbacks",
        callbacks=callbacks,
        train_loader=train_loader,
    )

    # =========================================================================
    # Setup Training
    # =========================================================================

    e.log("\nSetting up training...")

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
    # Train with Graceful Interrupt Handling
    # =========================================================================

    e.log("\nStarting training...")
    e.log("(Press CTRL+C to gracefully stop)")
    e.log("-" * 40)

    interrupted = False
    with GracefulInterruptHandler() as handler:
        handler.set_trainer(trainer)
        try:
            trainer.fit(model, train_loader, valid_loader)
        except KeyboardInterrupt:
            interrupted = True
            e.log("\nTraining interrupted by user (force quit)")

    if trainer.should_stop and not interrupted:
        e.log("-" * 40)
        e.log("Training gracefully stopped by user")
    elif not interrupted:
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
    # Reconstruction Evaluation
    # =========================================================================

    e.log("\n" + "=" * 60)
    e.log("Reconstruction Evaluation")
    e.log("=" * 60)

    try:
        # Use final model for reconstruction
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

        # Start timing
        sampling_start_time = time.time()

        for idx, sample_idx in enumerate(sample_indices):
            e.log(f"\nSample {idx + 1}/{num_samples}:")

            # Get original data
            original_data = raw_valid_ds[sample_idx]
            original_smiles = original_data.smiles if hasattr(original_data, "smiles") else "N/A"
            original_mol = Chem.MolFromSmiles(original_smiles) if original_smiles != "N/A" else None

            e.log(f"  Original SMILES: {original_smiles}")

            # Encode with HyperNet - compute concatenated [order_0 | order_N]
            data_for_encoding = original_data.clone()
            data_for_encoding = data_for_encoding.to(device)
            if not hasattr(data_for_encoding, "batch") or data_for_encoding.batch is None:
                data_for_encoding.batch = torch.zeros(
                    data_for_encoding.x.size(0), dtype=torch.long, device=device
                )

            with torch.no_grad():
                # Encode node properties
                data_for_encoding = hypernet.encode_properties(data_for_encoding)

                # Order-0: Bundle node hypervectors (no message passing)
                order_zero = scatter_hd(
                    src=data_for_encoding.node_hv,
                    index=data_for_encoding.batch,
                    op="bundle"
                )

                # Order-N: Full graph embedding with message passing
                encoder_output = hypernet.forward(data_for_encoding, normalize=True)
                order_n = encoder_output["graph_embedding"]

                # Concatenate [order_0 | order_N]
                graph_embedding = torch.cat([order_zero, order_n], dim=-1).squeeze(0)

            # Decode nodes from HDC embedding (using order_0 part)
            node_tuples, num_nodes = decode_nodes_from_hdc(
                hypernet, graph_embedding.unsqueeze(0), actual_hdc_dim
            )

            e.log(f"  Decoded {num_nodes} nodes: {node_tuples[:5]}{'...' if num_nodes > 5 else ''}")

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
            node_features = node_tuples_to_onehot(node_tuples, device=device)
            node_features = node_features.unsqueeze(0)  # (1, n, 24)
            node_mask = torch.ones(1, num_nodes, dtype=torch.bool, device=device)
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
            original_canonical = None
            if original_mol is not None:
                try:
                    original_mol_no_h = Chem.RemoveAllHs(original_mol)
                    original_canonical = Chem.MolToSmiles(original_mol_no_h, canonical=True)
                except Exception:
                    original_canonical = get_canonical_smiles(original_mol)

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

        # End timing
        total_sampling_time = time.time() - sampling_start_time

        # Log summary
        e.log("\n" + "-" * 40)
        e.log("Reconstruction Summary:")
        e.log(f"  Total samples: {num_samples}")
        e.log(f"  Valid molecules: {valid_count} ({100 * valid_count / num_samples:.1f}%)")
        e.log(f"  Exact matches: {match_count} ({100 * match_count / num_samples:.1f}%)")
        e.log(f"  Total sampling time: {total_sampling_time:.2f} seconds")
        e.log(f"  Average time per sample: {total_sampling_time / num_samples:.2f} seconds")
        e.log("-" * 40)

        # Store metrics
        e["reconstruction/num_samples"] = num_samples
        e["reconstruction/valid_count"] = valid_count
        e["reconstruction/match_count"] = match_count
        e["reconstruction/valid_rate"] = valid_count / num_samples if num_samples > 0 else 0
        e["reconstruction/match_rate"] = match_count / num_samples if num_samples > 0 else 0
        e["reconstruction/total_sampling_time_seconds"] = total_sampling_time
        e["reconstruction/avg_time_per_sample_seconds"] = total_sampling_time / num_samples if num_samples > 0 else 0
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
# HOOKS
# =============================================================================


@experiment.hook("load_train_data", default=True)
def load_train_data(
    e: Experiment,
    hypernet: HyperNet,
    device: torch.device,
) -> Tuple[DataLoader, List[Data]]:
    """
    Load and preprocess training data.

    Default implementation loads ZINC training split and preprocesses with HyperNet.
    Child experiments can override this hook to provide custom data sources.

    Args:
        e: Experiment instance for accessing parameters
        hypernet: HyperNet encoder for preprocessing
        device: Device for HDC computation

    Returns:
        Tuple of (train_loader, train_data list)
    """
    e.log("\nLoading training data...")
    train_ds = get_split("train", dataset=e.DATASET.lower())
    e.log(f"Raw train: {len(train_ds)}")

    e.log("Preprocessing training data...")
    e.log("(Computing HDC embeddings and converting features...)")
    train_data = preprocess_dataset(
        train_ds,
        hypernet,
        device=device,
        show_progress=True,
    )
    e.log(f"Processed train: {len(train_data)}")

    if len(train_data) == 0:
        raise ValueError("No training data after preprocessing!")

    train_loader = DataLoader(
        train_data,
        batch_size=e.BATCH_SIZE,
        shuffle=True,
        num_workers=e.NUM_WORKERS,
        pin_memory=(e.NUM_WORKERS > 0),
    )

    return train_loader, train_data


@experiment.hook("load_valid_data", default=True)
def load_valid_data(
    e: Experiment,
    hypernet: HyperNet,
    device: torch.device,
) -> Tuple[List[Data], DataLoader, List[Data]]:
    """
    Load and preprocess validation data.

    Default implementation loads ZINC validation split and preprocesses with HyperNet.

    Args:
        e: Experiment instance
        hypernet: HyperNet encoder for preprocessing
        device: Device for HDC computation

    Returns:
        Tuple of (valid_data list, valid_loader, vis_samples for visualization)
    """
    e.log("\nLoading validation data...")
    valid_ds = get_split("valid", dataset=e.DATASET.lower())
    e.log(f"Raw valid: {len(valid_ds)}")

    e.log("Preprocessing validation data...")
    valid_data = preprocess_dataset(
        valid_ds,
        hypernet,
        device=device,
        show_progress=True,
    )
    e.log(f"Processed valid: {len(valid_data)}")

    valid_loader = DataLoader(
        valid_data,
        batch_size=e.BATCH_SIZE,
        shuffle=False,
        num_workers=e.NUM_WORKERS,
        pin_memory=(e.NUM_WORKERS > 0),
    )

    # Select visualization samples
    num_vis = min(e.NUM_VALIDATION_VISUALIZATIONS, len(valid_data))
    vis_samples = valid_data[:num_vis]
    e.log(f"Selected {num_vis} samples for validation visualization")

    return valid_data, valid_loader, vis_samples


@experiment.hook("compute_statistics", default=True)
def compute_statistics(
    e: Experiment,
    train_data: Optional[List[Data]],
    hypernet: HyperNet,
    device: torch.device,
) -> Tuple[Tensor, Tensor, int]:
    """
    Compute edge marginals and node counts for model initialization.

    Default implementation computes statistics from the training data.
    Child experiments can override to compute from different data sources.

    Args:
        e: Experiment instance
        train_data: Training data (None for streaming)
        hypernet: HyperNet encoder (needed if train_data is None)
        device: Device for computation

    Returns:
        Tuple of (edge_marginals, node_counts, max_nodes)
    """
    if train_data is None:
        raise ValueError("train_data is required for base experiment statistics computation")

    e.log("\nComputing edge marginals...")
    edge_marginals = compute_edge_marginals(train_data)
    e.log(f"Edge marginals: {edge_marginals.tolist()}")

    node_counts = compute_node_counts(train_data)
    max_nodes = int(node_counts.nonzero()[-1].item()) if node_counts.sum() > 0 else 50
    e.log(f"Max nodes: {max_nodes}")

    return edge_marginals, node_counts, max_nodes


@experiment.hook("modify_callbacks", default=True)
def modify_callbacks(
    e: Experiment,
    callbacks: List[Callback],
    train_loader: DataLoader,
) -> List[Callback]:
    """
    Modify the standard callbacks list.

    Override this hook to add, remove, or modify callbacks. The standard
    callbacks (ModelCheckpoint, LearningRateMonitor, LossTrackingCallback,
    ReconstructionVisualizationCallback) are created in the main function.

    The default implementation returns the callbacks unchanged.

    Args:
        e: Experiment instance
        callbacks: List of standard callbacks to modify
        train_loader: Training data loader (useful for cleanup callbacks)

    Returns:
        Modified list of callbacks
    """
    return callbacks


# =============================================================================
# Entry Point
# =============================================================================

experiment.run_if_main()
