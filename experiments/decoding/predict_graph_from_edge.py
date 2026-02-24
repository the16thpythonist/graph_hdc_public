#!/usr/bin/env python
"""
Predict graph_embedding (order-N) from node_terms (order-0) using a Residual MLP.

This experiment trains a simple but configurable MLP with residual connections that
takes the node_terms of a loaded HyperNet encoder as input and predicts the
graph_embedding. This tests how much structural information can be recovered from
node composition alone.

The HDC vectors are precomputed once at startup for efficiency.

Usage:
    # Quick test
    python predict_graph_from_edge.py --__TESTING__ True --HDC_CHECKPOINT_PATH /path/to/encoder.ckpt

    # Full training
    python predict_graph_from_edge.py --HDC_CHECKPOINT_PATH /path/to/encoder.ckpt --EPOCHS 100

    # Custom architecture
    python predict_graph_from_edge.py --HDC_CHECKPOINT_PATH /path/to/encoder.ckpt \
        --HIDDEN_DIM 1024 --NUM_LAYERS 6 --DROPOUT 0.2
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from graph_hdc.datasets.utils import get_split
from graph_hdc.hypernet import load_hypernet
from graph_hdc.hypernet.multi_hypernet import MultiHyperNet

if TYPE_CHECKING:
    from pycomex.functional.experiment import Experiment as ExperimentType

# =============================================================================
# PARAMETERS
# =============================================================================

# -----------------------------------------------------------------------------
# HDC Encoder Configuration
# -----------------------------------------------------------------------------

# :param HDC_CHECKPOINT_PATH:
#     Path to saved HyperNet encoder checkpoint (.ckpt). Required.
#     Supports HyperNet and RRWPHyperNet. MultiHyperNet is not supported.
HDC_CHECKPOINT_PATH: str = ""

# :param DATASET:
#     Dataset name for training. Currently supports "zinc" and "qm9".
DATASET: str = "zinc"

# -----------------------------------------------------------------------------
# MLP Architecture
# -----------------------------------------------------------------------------

# :param HIDDEN_DIM:
#     Hidden dimension for the residual MLP layers.
HIDDEN_DIM: int = 512

# :param NUM_LAYERS:
#     Number of residual blocks in the MLP.
NUM_LAYERS: int = 4

# :param DROPOUT:
#     Dropout probability in residual blocks.
DROPOUT: float = 0.1

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------

# :param EPOCHS:
#     Number of training epochs.
EPOCHS: int = 100

# :param BATCH_SIZE:
#     Batch size for training and validation.
BATCH_SIZE: int = 256

# :param LEARNING_RATE:
#     Initial learning rate for Adam optimizer.
LEARNING_RATE: float = 1e-3

# :param WEIGHT_DECAY:
#     Weight decay (L2 regularization) for optimizer.
WEIGHT_DECAY: float = 1e-5

# :param GRADIENT_CLIP_VAL:
#     Gradient clipping value. Set to 0.0 to disable.
GRADIENT_CLIP_VAL: float = 1.0

# -----------------------------------------------------------------------------
# System Configuration
# -----------------------------------------------------------------------------

# :param SEED:
#     Random seed for reproducibility.
SEED: int = 42

# :param NUM_WORKERS:
#     Number of data loader workers.
NUM_WORKERS: int = 4

# :param ACCELERATOR:
#     PyTorch Lightning accelerator. Options: "auto", "gpu", "cpu".
ACCELERATOR: str = "auto"

# :param PRECISION:
#     Training precision. Options: "32", "16", "bf16".
PRECISION: str = "32"

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
# MODEL
# =============================================================================


class ResidualBlock(nn.Module):
    """Single residual block: Linear -> ReLU -> Dropout -> Linear + skip."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + residual
        x = F.relu(x)
        return x


class ResidualMLP(nn.Module):
    """MLP with residual connections for predicting graph_embedding from node_terms."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


class NodeToGraphPredictor(pl.LightningModule):
    """Lightning module wrapping ResidualMLP for node_terms -> graph_embedding prediction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.mlp = ResidualMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        node_terms = batch.x  # [B, hv_dim]
        graph_embedding = batch.y  # [B, hv_dim]
        pred = self(node_terms)
        loss = F.mse_loss(pred, graph_embedding)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> None:
        node_terms = batch.x
        graph_embedding = batch.y
        pred = self(node_terms)

        loss = F.mse_loss(pred, graph_embedding)
        cos_sim = F.cosine_similarity(pred, graph_embedding, dim=-1)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/cos_sim_mean", cos_sim.mean(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/cos_sim_median", cos_sim.median(), on_step=False, on_epoch=True)
        self.log("val/cos_sim_min", cos_sim.min(), on_step=False, on_epoch=True)
        self.log("val/cos_sim_max", cos_sim.max(), on_step=False, on_epoch=True)
        self.log("val/pct_above_0.90", (cos_sim > 0.90).float().mean() * 100, on_step=False, on_epoch=True)
        self.log("val/pct_above_0.95", (cos_sim > 0.95).float().mean() * 100, on_step=False, on_epoch=True)
        self.log("val/pct_above_0.99", (cos_sim > 0.99).float().mean() * 100, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }


# =============================================================================
# CALLBACK
# =============================================================================


class GraphPredictionMetricsCallback(Callback):
    """Tracks training metrics and generates visualization figures.

    Tracks per epoch:
    - Training and validation MSE loss
    - Validation cosine similarity statistics (mean, median, min, max)
    - Threshold accuracy (% above 0.9, 0.95, 0.99)
    - Global gradient L2 norm
    - Learning rate

    Generates a 2x2 figure at the end of each validation epoch.
    """

    def __init__(self, experiment: "ExperimentType"):
        super().__init__()
        self.experiment = experiment

        # Accumulated metrics
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.cos_sim_means: list[float] = []
        self.cos_sim_medians: list[float] = []
        self.cos_sim_mins: list[float] = []
        self.cos_sim_maxs: list[float] = []
        self.pct_above_090: list[float] = []
        self.pct_above_095: list[float] = []
        self.pct_above_099: list[float] = []
        self.grad_norms: list[float] = []
        self.learning_rates: list[float] = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics

        # Track training loss
        train_loss = metrics.get("train/loss")
        if train_loss is not None:
            val = float(train_loss)
            self.train_losses.append(val)
            self.experiment.track("loss_train", val)

        # Track gradient norm
        grad_norm = self._compute_gradient_norm(pl_module)
        self.grad_norms.append(grad_norm)
        self.experiment.track("grad_norm", grad_norm)

        # Track learning rate
        optimizer = trainer.optimizers[0]
        lr = optimizer.param_groups[0]["lr"]
        self.learning_rates.append(lr)
        self.experiment.track("learning_rate", lr)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics

        # Track validation loss
        val_loss = metrics.get("val/loss")
        if val_loss is not None:
            val = float(val_loss)
            self.val_losses.append(val)
            self.experiment.track("loss_val", val)

        # Track cosine similarity metrics
        for attr, key in [
            ("cos_sim_means", "val/cos_sim_mean"),
            ("cos_sim_medians", "val/cos_sim_median"),
            ("cos_sim_mins", "val/cos_sim_min"),
            ("cos_sim_maxs", "val/cos_sim_max"),
            ("pct_above_090", "val/pct_above_0.90"),
            ("pct_above_095", "val/pct_above_0.95"),
            ("pct_above_099", "val/pct_above_0.99"),
        ]:
            metric_val = metrics.get(key)
            if metric_val is not None:
                val = float(metric_val)
                getattr(self, attr).append(val)
                # Convert key for PyComex tracking (underscores, not slashes)
                track_key = key.replace("/", "_").replace(".", "")
                self.experiment.track(track_key, val)

        # Generate figure
        if len(self.val_losses) > 0:
            self._generate_figure(trainer.current_epoch)

    def _compute_gradient_norm(self, model: pl.LightningModule) -> float:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def _generate_figure(self, epoch: int) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Graph Prediction Metrics (Epoch {epoch})", fontsize=14)

        # Panel 1: Loss curves
        ax = axes[0, 0]
        if self.train_losses:
            ax.plot(self.train_losses, label="Train MSE", alpha=0.8)
        if self.val_losses:
            ax.plot(self.val_losses, label="Val MSE", alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Loss Curves")
        ax.legend()
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # Panel 2: Cosine similarity over time
        ax = axes[0, 1]
        if self.cos_sim_means:
            epochs_range = range(len(self.cos_sim_means))
            ax.plot(epochs_range, self.cos_sim_means, label="Mean", linewidth=2)
            ax.plot(epochs_range, self.cos_sim_medians, label="Median", linewidth=1, alpha=0.7)
            ax.fill_between(
                epochs_range,
                self.cos_sim_mins,
                self.cos_sim_maxs,
                alpha=0.15,
                label="Min-Max range",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Validation Cosine Similarity")
        ax.legend()
        ax.set_ylim(-0.1, 1.05)
        ax.grid(True, alpha=0.3)

        # Panel 3: Threshold accuracy bars
        ax = axes[1, 0]
        if self.pct_above_090:
            epochs_range = range(len(self.pct_above_090))
            ax.plot(epochs_range, self.pct_above_090, label="> 0.90", linewidth=2)
            ax.plot(epochs_range, self.pct_above_095, label="> 0.95", linewidth=2)
            ax.plot(epochs_range, self.pct_above_099, label="> 0.99", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("% of Samples")
        ax.set_title("Threshold Accuracy")
        ax.legend()
        ax.set_ylim(-2, 102)
        ax.grid(True, alpha=0.3)

        # Panel 4: Gradient norms and learning rate
        ax = axes[1, 1]
        if self.grad_norms:
            ax.plot(self.grad_norms, label="Grad Norm", alpha=0.8, color="tab:blue")
            ax.set_ylabel("Gradient Norm", color="tab:blue")
            ax.tick_params(axis="y", labelcolor="tab:blue")
        if self.learning_rates:
            ax2 = ax.twinx()
            ax2.plot(self.learning_rates, label="LR", alpha=0.8, color="tab:orange", linestyle="--")
            ax2.set_ylabel("Learning Rate", color="tab:orange")
            ax2.tick_params(axis="y", labelcolor="tab:orange")
            ax2.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title("Gradient Norms & Learning Rate")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.experiment.commit_fig("metrics_overview.png", fig)
        plt.close(fig)


# =============================================================================
# HELPERS
# =============================================================================


def precompute_hdc_vectors(
    dataset,
    hypernet,
    device: torch.device,
    show_progress: bool = True,
) -> list[Data]:
    """Precompute node_terms and graph_embedding for each molecule.

    Returns a list of Data objects with:
    - x: node_terms [hv_dim] (input to MLP)
    - y: graph_embedding [hv_dim] (target)
    """
    results = []
    iterator = tqdm(range(len(dataset)), desc="Precomputing HDC vectors") if show_progress else range(len(dataset))

    for idx in iterator:
        mol_data = dataset[idx]
        try:
            data = mol_data.clone().to(device)
            if not hasattr(data, "batch") or data.batch is None:
                data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

            # Augment with RW features if encoder expects them
            if hasattr(hypernet, "rw_config") and hypernet.rw_config.enabled:
                from graph_hdc.utils.rw_features import augment_data_with_rw

                data = augment_data_with_rw(
                    data,
                    k_values=hypernet.rw_config.k_values,
                    num_bins=hypernet.rw_config.num_bins,
                    bin_boundaries=getattr(hypernet.rw_config, "bin_boundaries", None),
                    clip_range=getattr(hypernet.rw_config, "clip_range", None),
                )

            with torch.no_grad():
                out = hypernet.forward(data, normalize=True)
                node_terms = out["node_terms"].squeeze(0).cpu()  # [hv_dim]
                graph_embedding = out["graph_embedding"].squeeze(0).cpu()  # [hv_dim]

            results.append(Data(x=node_terms, y=graph_embedding))
        except Exception:
            # Skip molecules that fail encoding
            continue

    return results


# =============================================================================
# EXPERIMENT
# =============================================================================


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    """Train MLP to predict graph_embedding from node_terms."""

    # =========================================================================
    # Setup
    # =========================================================================

    pl.seed_everything(e.SEED)

    e.log("=" * 60)
    e.log("Predict graph_embedding from node_terms (Residual MLP)")
    e.log("=" * 60)

    # Validate checkpoint path
    if not e.HDC_CHECKPOINT_PATH:
        raise ValueError(
            "HDC_CHECKPOINT_PATH is required. Provide a path to a saved HyperNet checkpoint."
        )

    ckpt_path = Path(e.HDC_CHECKPOINT_PATH)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {ckpt_path}")

    # Device
    if e.ACCELERATOR == "gpu":
        device = torch.device("cuda")
    elif e.ACCELERATOR == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    e.log(f"Dataset: {e.DATASET.upper()}")
    e.log(f"Encoder checkpoint: {e.HDC_CHECKPOINT_PATH}")
    e.log(f"Architecture: {e.NUM_LAYERS} residual blocks, {e.HIDDEN_DIM} hidden dim, dropout={e.DROPOUT}")
    e.log(f"Training: {e.EPOCHS} epochs, batch size {e.BATCH_SIZE}, lr={e.LEARNING_RATE}")
    e.log(f"Device: {device}")
    e.log("=" * 60)

    # =========================================================================
    # Load Encoder
    # =========================================================================

    e.log("\nLoading HyperNet encoder...")
    hdc_device = torch.device("cpu")
    hypernet = load_hypernet(str(ckpt_path), device=str(hdc_device))

    # Reject MultiHyperNet
    if isinstance(hypernet, MultiHyperNet):
        raise ValueError(
            "MultiHyperNet is not supported for this experiment. "
            "Please provide a single HyperNet or RRWPHyperNet checkpoint."
        )

    hypernet.eval()
    hv_dim = hypernet.hv_dim

    e.log(f"Encoder type: {type(hypernet).__name__}")
    e.log(f"Hypervector dim: {hv_dim}")
    e.log(f"Message passing depth: {hypernet.depth}")
    if hasattr(hypernet, "rw_config") and hypernet.rw_config.enabled:
        e.log(f"RW features: k={hypernet.rw_config.k_values}, bins={hypernet.rw_config.num_bins}")

    # Store config
    e["config/dataset"] = e.DATASET
    e["config/encoder_type"] = type(hypernet).__name__
    e["config/hv_dim"] = hv_dim
    e["config/depth"] = hypernet.depth
    e["config/hidden_dim"] = e.HIDDEN_DIM
    e["config/num_layers"] = e.NUM_LAYERS
    e["config/dropout"] = e.DROPOUT
    e["config/epochs"] = e.EPOCHS
    e["config/batch_size"] = e.BATCH_SIZE
    e["config/lr"] = e.LEARNING_RATE
    e["config/weight_decay"] = e.WEIGHT_DECAY
    e["config/device"] = str(device)

    # =========================================================================
    # Load and Precompute Data
    # =========================================================================

    e.log("\nLoading dataset...")
    train_ds = get_split("train", dataset=e.DATASET.lower())
    valid_ds = get_split("valid", dataset=e.DATASET.lower())
    e.log(f"Raw train: {len(train_ds)}, Raw valid: {len(valid_ds)}")

    e.log("\nPrecomputing HDC vectors for training set...")
    train_data = precompute_hdc_vectors(train_ds, hypernet, hdc_device, show_progress=True)
    e.log(f"Precomputed train: {len(train_data)} samples")

    e.log("Precomputing HDC vectors for validation set...")
    valid_data = precompute_hdc_vectors(valid_ds, hypernet, hdc_device, show_progress=True)
    e.log(f"Precomputed valid: {len(valid_data)} samples")

    if len(train_data) == 0:
        raise ValueError("No training data after preprocessing!")
    if len(valid_data) == 0:
        raise ValueError("No validation data after preprocessing!")

    e["data/train_size"] = len(train_data)
    e["data/valid_size"] = len(valid_data)

    # Free encoder memory
    del hypernet
    gc.collect()
    torch.cuda.empty_cache()

    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=e.BATCH_SIZE,
        shuffle=True,
        num_workers=e.NUM_WORKERS,
        pin_memory=(e.NUM_WORKERS > 0),
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=e.BATCH_SIZE,
        shuffle=False,
        num_workers=e.NUM_WORKERS,
        pin_memory=(e.NUM_WORKERS > 0),
    )

    # =========================================================================
    # Create Model
    # =========================================================================

    e.log("\nCreating NodeToGraphPredictor model...")
    model = NodeToGraphPredictor(
        input_dim=hv_dim,
        output_dim=hv_dim,
        hidden_dim=e.HIDDEN_DIM,
        num_layers=e.NUM_LAYERS,
        dropout=e.DROPOUT,
        lr=e.LEARNING_RATE,
        weight_decay=e.WEIGHT_DECAY,
    )

    num_params = sum(p.numel() for p in model.parameters())
    e.log(f"Model parameters: {num_params:,}")
    e["model/num_parameters"] = num_params

    # =========================================================================
    # Callbacks and Trainer
    # =========================================================================

    callbacks = [
        ModelCheckpoint(
            dirpath=e.path,
            filename="best-{epoch:03d}-{val/loss:.6f}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        GraphPredictionMetricsCallback(experiment=e),
    ]

    logger = CSVLogger(e.path, name="logs")

    trainer = Trainer(
        max_epochs=e.EPOCHS,
        accelerator=e.ACCELERATOR,
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
    # Results
    # =========================================================================

    best_val_loss = trainer.callback_metrics.get("val/loss")
    best_cos_sim = trainer.callback_metrics.get("val/cos_sim_mean")

    if best_val_loss is not None:
        e["results/final_val_loss"] = float(best_val_loss)
        e.log(f"Final validation loss: {float(best_val_loss):.6f}")
    if best_cos_sim is not None:
        e["results/final_cos_sim_mean"] = float(best_cos_sim)
        e.log(f"Final validation cosine similarity (mean): {float(best_cos_sim):.4f}")

    for key in ["val/pct_above_0.90", "val/pct_above_0.95", "val/pct_above_0.99"]:
        val = trainer.callback_metrics.get(key)
        if val is not None:
            clean_key = key.replace("/", "_").replace(".", "")
            e[f"results/{clean_key}"] = float(val)
            e.log(f"  {key}: {float(val):.1f}%")

    # Save final model
    final_path = Path(e.path) / "final_model.ckpt"
    trainer.save_checkpoint(str(final_path))
    e.log(f"Model saved to: {final_path}")
    e["results/model_path"] = str(final_path)

    e.log("\n" + "=" * 60)
    e.log("Experiment completed!")
    e.log("=" * 60)


# =============================================================================
# TESTING
# =============================================================================


@experiment.testing
def testing(e: Experiment) -> None:
    """Reduce iterations for quick test runs."""
    e.EPOCHS = 2
    e.BATCH_SIZE = 16


# =============================================================================
# Entry Point
# =============================================================================

experiment.run_if_main()
