#!/usr/bin/env python
"""
Train Flow Matching model for HDC vector generation.

Learns a continuous normalizing flow from standard Gaussian to the distribution
of HDC graph encodings using conditional flow matching with optional minibatch
OT coupling.

The model treats concatenated [node_terms | graph_terms] vectors as flat data
and is agnostic to the internal HDC structure.

Usage:
    # Quick test
    python train_flow_matching.py --__TESTING__ True

    # Full training on ZINC (requires pre-saved encoder)
    python train_flow_matching.py --__DEBUG__ False --ENCODER_PATH /path/to/encoder.ckpt
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EMAWeightAveraging, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.loader import DataLoader

from graph_hdc.datasets.utils import (
    get_split,
    post_compute_encodings,
)
from graph_hdc.hypernet import load_hypernet
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.hypernet.multi_hypernet import MultiHyperNet
from graph_hdc.models.flow_matching import FlowMatchingModel, MultiCondition, build_condition
from graph_hdc.utils.experiment_helpers import GracefulInterruptHandler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flow_matching.solver import ODESolver
from torch import Tensor


# =============================================================================
# PARAMETERS
# =============================================================================

# -----------------------------------------------------------------------------
# Dataset & Encoder
# -----------------------------------------------------------------------------

# :param DATASET:
#     Dataset name for training.
DATASET: str = "zinc"

# :param ENCODER_PATH:
#     Path to a saved HyperNet encoder checkpoint (.ckpt). Required.
ENCODER_PATH: str = "/media/ssd2/Programming/graph_hdc_public/experiments/decoding/results/train_flow_edge_decoder_streaming/GOOD/hypernet_encoder.ckpt"

# :param DEVICE:
#     Device for model training and evaluation. Options: "auto" (prefer GPU),
#     "cpu", "cuda".
DEVICE: str = "cuda"

# :param HDC_DEVICE:
#     Device for the HyperNet HDC encoder (encoding step). Options: "auto"
#     (prefer GPU), "cpu", "cuda".
HDC_DEVICE: str = "cpu"

# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------

# :param HIDDEN_DIM:
#     Hidden dimension for the velocity MLP.
HIDDEN_DIM: int = 512

# :param NUM_BLOCKS:
#     Number of residual FiLM blocks in the velocity MLP.
NUM_BLOCKS: int = 12

# :param TIME_EMBED_DIM:
#     Dimension of sinusoidal time embedding.
TIME_EMBED_DIM: int = 64

# :param CONDITIONS:
#     List of condition names from the registry. None or empty = unconditional.
#     Available: "logp" (Crippen LogP), "heavy_atoms" (heavy atom count).
#     Example: ["logp"] for single, ["logp", "heavy_atoms"] for multi.
CONDITIONS: Optional[List[str]] = None

# :param CONDITION_EMBED_DIM:
#     Dimension to project raw condition values into via a learned MLP.
#     Only used when CONDITIONS is set.
CONDITION_EMBED_DIM: int = 64

# :param DROPOUT:
#     Dropout rate in residual blocks.
DROPOUT: float = 0.0

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------

# :param EPOCHS:
#     Number of training epochs.
EPOCHS: int = 200

# :param BATCH_SIZE:
#     Batch size for training.
BATCH_SIZE: int = 256

# :param LEARNING_RATE:
#     Learning rate for AdamW optimizer.
LEARNING_RATE: float = 2e-4

# :param WEIGHT_DECAY:
#     Weight decay for AdamW.
WEIGHT_DECAY: float = 0e-5

# :param WARMUP_EPOCHS:
#     Number of linear warmup epochs.
WARMUP_EPOCHS: int = 5

# :param ENCODER_BATCH_SIZE:
#     Batch size for HDC encoding. Lower values reduce GPU memory pressure
#     during FFT operations (relevant for MultiHyperNet).
ENCODER_BATCH_SIZE: int = 256

# :param GRADIENT_CLIP_VAL:
#     Gradient clipping value.
GRADIENT_CLIP_VAL: float = 1.0

# :param EVAL_TRACKING_EVERY_N:
#     How often (in epochs) to compute expensive tracking metrics
#     (sample quality, reconstruction, velocity norm, etc.).
EVAL_TRACKING_EVERY_N: int = 5

# :param USE_OT_COUPLING:
#     Whether to use minibatch optimal transport coupling.
USE_OT_COUPLING: bool = True

# :param USE_EMA:
#     Whether to use Exponential Moving Average of model weights.
USE_EMA: bool = True

# :param EMA_DECAY:
#     EMA decay rate. 0.9999 is standard (DiT, SD3).
EMA_DECAY: float = 0.9999

# :param STANDARDIZE:
#     Whether to standardize input vectors (zero mean, unit variance).
STANDARDIZE: bool = True

# -----------------------------------------------------------------------------
# Sampling & Evaluation
# -----------------------------------------------------------------------------

# :param SOLVER_METHOD:
#     ODE solver method for sampling (euler, midpoint, dopri5).
SOLVER_METHOD: str = "midpoint"

# :param SAMPLE_STEPS:
#     Number of ODE steps for sampling.
SAMPLE_STEPS: int = 1_000

# :param NLL_EVAL_STEPS:
#     Number of ODE steps for NLL computation.
NLL_EVAL_STEPS: int = 2_000

# :param NLL_EVAL_SAMPLES:
#     Number of validation samples to evaluate NLL on.
NLL_EVAL_SAMPLES: int = 1000

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------

# :param SEED:
#     Random seed for reproducibility.
SEED: int = 42

# :param PRECISION:
#     Training precision.
PRECISION: str = "32"

# -----------------------------------------------------------------------------
# Debug/Testing
# -----------------------------------------------------------------------------

# :param __DEBUG__:
#     Debug mode - reuses same output folder during development.
__DEBUG__: bool = True

# :param __TESTING__:
#     Testing mode - runs with minimal iterations for validation.
__TESTING__: bool = False


# =============================================================================
# TRAINING TRACKER CALLBACK
# =============================================================================


class TrainingTrackerCallback(pl.Callback):
    """
    Tracks training progress and produces a 4x4 panel visualization.

    Layout:
        Row 1: Train+Val Overlay | Train Loss | OT Cost Ratio | Learning Rate
        Row 2: Sample Mean MAE | Sample Std MAE | Per-Dim Mean R² | Per-Dim Std R²
        Row 3: Recon MSE | Recon Cos Sim | Per-Dim Mean Scatter | Velocity Norm
        Row 4: Sample L2 Norm | Inter-Sample Cos Sim | Centroid Cos Sim | Grad Norm

    Uses PyComex ``experiment.track()`` for time-series metrics and figure
    tracking, following the same pattern as ``TrainingMetricsCallback``.

    Row 1 and gradient norms are updated every epoch (cheap).
    Rows 2-4 are updated every ``eval_every_n_epochs`` epochs (expensive).

    The OT Cost Ratio panel shows ``mean(||x0_ot - x1||²) / mean(||x0_rand - x1||²)``
    per epoch, confirming that minibatch OT coupling is actively reducing
    transport cost.  Values near 0.3-0.7 are typical; 1.0 means OT is
    not helping (e.g. batch_size=1 or coupling disabled).
    """

    def __init__(
        self,
        experiment: "Experiment",
        train_mean_per_dim: Tensor,
        train_std_per_dim: Tensor,
        train_norm_mean: float,
        recon_vectors: Tensor,
        eval_every_n_epochs: int = 5,
        num_eval_samples: int = 500,
        num_recon_steps: int = 50,
        smoothing_window: int = 5,
    ):
        super().__init__()
        self.experiment = experiment
        self.train_mean_per_dim = train_mean_per_dim.cpu()
        self.train_std_per_dim = train_std_per_dim.cpu()
        self.train_norm_mean = train_norm_mean
        self.train_centroid = F.normalize(
            self.train_mean_per_dim.unsqueeze(0), dim=-1,
        )
        self.recon_vectors = recon_vectors.cpu()
        self.eval_every_n_epochs = eval_every_n_epochs
        self.num_eval_samples = num_eval_samples
        self.num_recon_steps = num_recon_steps
        self.smoothing_window = smoothing_window

        # Per-epoch histories
        self.epochs: list[int] = []
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.lrs: list[float] = []
        self.grad_norms: list[float] = []
        self.ot_cost_ratios: list[float] = []

        # Per-eval histories (every N epochs)
        self.eval_epochs: list[int] = []
        self.sample_mean_maes: list[float] = []
        self.sample_std_maes: list[float] = []
        self.per_dim_mean_r2s: list[float] = []
        self.per_dim_std_r2s: list[float] = []
        self.recon_mses: list[float] = []
        self.recon_cos_sims: list[float] = []
        self.sample_norm_means: list[float] = []
        self.inter_sample_cos_sims: list[float] = []
        self.centroid_cos_sims: list[float] = []
        self.velocity_norms: list[float] = []

        # Latest per-dim means for scatter plot
        self.latest_gen_mean_per_dim: Optional[Tensor] = None

        # Gradient norm accumulator (reset each epoch)
        self._grad_norm_sum = 0.0
        self._grad_norm_count = 0

        # OT cost ratio accumulator (reset each epoch)
        self._ot_ratio_sum = 0.0
        self._ot_ratio_count = 0

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _smooth(self, values: list[float], window: Optional[int] = None) -> list[float]:
        """Apply simple moving average smoothing."""
        w = window or self.smoothing_window
        if len(values) < w:
            return values
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - w + 1)
            smoothed.append(sum(values[start : i + 1]) / (i - start + 1))
        return smoothed

    def _filter_nan(self, x: list, y: list[float]):
        """Filter out NaN values, returning paired lists."""
        pairs = [(xi, yi) for xi, yi in zip(x[: len(y)], y) if yi == yi]
        if not pairs:
            return [], []
        return zip(*pairs)

    # -----------------------------------------------------------------
    # Cheap per-epoch hooks
    # -----------------------------------------------------------------

    def on_train_epoch_start(self, trainer, pl_module):
        self._grad_norm_sum = 0.0
        self._grad_norm_count = 0
        self._ot_ratio_sum = 0.0
        self._ot_ratio_count = 0

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        total_norm_sq = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.norm(2).item() ** 2
        self._grad_norm_sum += total_norm_sq ** 0.5
        self._grad_norm_count += 1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Collect OT cost ratio stored by compute_loss
        ratio = getattr(pl_module, "_ot_cost_ratio", None)
        if ratio is not None:
            self._ot_ratio_sum += ratio
            self._ot_ratio_count += 1

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch
        self.epochs.append(epoch)

        # Loss
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train/loss_epoch", metrics.get("train/loss"))
        val_loss = metrics.get("val/loss")

        tl = float(train_loss) if train_loss is not None else float("nan")
        vl = float(val_loss) if val_loss is not None else float("nan")
        self.train_losses.append(tl)
        self.val_losses.append(vl)

        # Track scalars via PyComex
        if train_loss is not None:
            self.experiment.track("fm_loss_train", tl)
        if val_loss is not None:
            self.experiment.track("fm_loss_val", vl)

        # Learning rate
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0]["lr"]
            self.lrs.append(lr)
            self.experiment.track("fm_learning_rate", lr)
        else:
            self.lrs.append(float("nan"))

        # Average gradient norm over the training epoch
        if self._grad_norm_count > 0:
            gn = self._grad_norm_sum / self._grad_norm_count
            self.grad_norms.append(gn)
            self.experiment.track("fm_grad_norm", gn)
        else:
            self.grad_norms.append(float("nan"))

        # Average OT cost ratio over the training epoch
        if self._ot_ratio_count > 0:
            ot = self._ot_ratio_sum / self._ot_ratio_count
            self.ot_cost_ratios.append(ot)
            self.experiment.track("fm_ot_cost_ratio", ot)
        else:
            self.ot_cost_ratios.append(float("nan"))

        # Expensive evaluation every N epochs
        if epoch % self.eval_every_n_epochs == 0:
            self._compute_expensive_metrics(pl_module)
            self.eval_epochs.append(epoch)

        # Generate and track the 4x4 panel figure
        try:
            fig = self._create_metrics_plot(epoch)
            self.experiment.track("flow_training_metrics", fig)
            plt.close(fig)
        except Exception as ex:
            self.experiment.log(f"Warning: Failed to create metrics plot: {ex}")

    # -----------------------------------------------------------------
    # Expensive evaluation (every N epochs)
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _compute_expensive_metrics(self, model: FlowMatchingModel):
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        try:
            # --- Generate samples ---
            generated = model.sample(
                self.num_eval_samples, device=device,
            ).cpu()
            gen_mean = generated.mean(dim=0)
            gen_std = generated.std(dim=0)

            # Sample mean MAE
            v = (gen_mean - self.train_mean_per_dim).abs().mean().item()
            self.sample_mean_maes.append(v)
            self.experiment.track("fm_sample_mean_mae", v)

            # Sample std MAE
            v = (gen_std - self.train_std_per_dim).abs().mean().item()
            self.sample_std_maes.append(v)
            self.experiment.track("fm_sample_std_mae", v)

            # Per-dim mean R²
            ss_res = ((gen_mean - self.train_mean_per_dim) ** 2).sum().item()
            ss_tot = (
                (self.train_mean_per_dim - self.train_mean_per_dim.mean()) ** 2
            ).sum().item()
            v = 1.0 - ss_res / max(ss_tot, 1e-8)
            self.per_dim_mean_r2s.append(v)
            self.experiment.track("fm_per_dim_mean_r2", v)

            # Per-dim std R²
            ss_res_s = ((gen_std - self.train_std_per_dim) ** 2).sum().item()
            ss_tot_s = (
                (self.train_std_per_dim - self.train_std_per_dim.mean()) ** 2
            ).sum().item()
            v = 1.0 - ss_res_s / max(ss_tot_s, 1e-8)
            self.per_dim_std_r2s.append(v)
            self.experiment.track("fm_per_dim_std_r2", v)

            # Store for scatter plot
            self.latest_gen_mean_per_dim = gen_mean

            # --- Reconstruction (encode -> decode roundtrip) ---
            recon_sub = self.recon_vectors.to(device)
            encoded = model.encode(recon_sub, num_steps=self.num_recon_steps)

            step_size = 1.0 / self.num_recon_steps
            solver = ODESolver(velocity_model=model.wrapper)
            x1_std_hat = solver.sample(
                x_init=encoded,
                step_size=step_size,
                method=model.solver_method,
                time_grid=torch.tensor([0.0, 1.0], device=device),
                return_intermediates=False,
            )
            reconstructed = model.destandardize(x1_std_hat).cpu()

            v = F.mse_loss(reconstructed, self.recon_vectors).item()
            self.recon_mses.append(v)
            self.experiment.track("fm_recon_mse", v)

            v = F.cosine_similarity(
                reconstructed, self.recon_vectors, dim=-1,
            ).mean().item()
            self.recon_cos_sims.append(v)
            self.experiment.track("fm_recon_cos_sim", v)

            # --- Sample norms ---
            v = generated.norm(dim=-1).mean().item()
            self.sample_norm_means.append(v)
            self.experiment.track("fm_sample_norm_mean", v)

            # --- Inter-sample cosine similarity ---
            sub = generated[: min(200, len(generated))]
            normed = F.normalize(sub, dim=-1)
            cos_mat = normed @ normed.T
            cos_mat.fill_diagonal_(0)
            triu = torch.triu(torch.ones_like(cos_mat), diagonal=1).bool()
            v = cos_mat[triu].mean().item()
            self.inter_sample_cos_sims.append(v)
            self.experiment.track("fm_inter_sample_cos_sim", v)

            # --- Centroid cosine similarity ---
            gen_centroid = F.normalize(gen_mean.unsqueeze(0), dim=-1)
            v = F.cosine_similarity(
                gen_centroid, self.train_centroid, dim=-1,
            ).item()
            self.centroid_cos_sims.append(v)
            self.experiment.track("fm_centroid_cos_sim", v)

            # --- Velocity norm at t=0.5 ---
            val_sub = self.recon_vectors[:200].to(device)
            val_std = model.standardize(val_sub)
            t = torch.full((val_sub.shape[0],), 0.5, device=device)
            vel = model.velocity_net(val_std, t)
            v = vel.norm(dim=-1).mean().item()
            self.velocity_norms.append(v)
            self.experiment.track("fm_velocity_norm", v)

        finally:
            if was_training:
                model.train()

    # -----------------------------------------------------------------
    # 4x4 panel plot
    # -----------------------------------------------------------------

    def _create_metrics_plot(self, epoch: int) -> plt.Figure:
        """Create 4x4 grid of training metrics."""
        fig, axes = plt.subplots(4, 4, figsize=(16, 14))

        ep = self.epochs
        ev = self.eval_epochs

        # Row 1: Loss, OT & LR
        self._plot_loss_overlay(axes[0, 0], ep)
        self._plot_loss_overlay(axes[0, 1], ep, log_scale=True)
        self._plot_ot_cost_ratio(axes[0, 2], ep)
        self._plot_learning_rate(axes[0, 3], ep)

        # Row 2: Distribution Match
        self._plot_line(
            axes[1, 0], ev, self.sample_mean_maes,
            "Sample Mean MAE", "MAE", "tab:blue",
        )
        self._plot_line(
            axes[1, 1], ev, self.sample_std_maes,
            "Sample Std MAE", "MAE", "tab:blue",
        )
        self._plot_line(
            axes[1, 2], ev, self.per_dim_mean_r2s,
            "Per-Dim Mean R\u00b2", "R\u00b2", "tab:green",
        )
        self._plot_line(
            axes[1, 3], ev, self.per_dim_std_r2s,
            "Per-Dim Std R\u00b2", "R\u00b2", "tab:green",
        )

        # Row 3: Flow Quality
        self._plot_line(
            axes[2, 0], ev, self.recon_mses,
            "Reconstruction MSE", "MSE", "tab:red",
        )
        self._plot_line(
            axes[2, 1], ev, self.recon_cos_sims,
            "Reconstruction Cos Sim", "Cos Sim", "tab:purple",
        )
        self._plot_per_dim_scatter(axes[2, 2])
        self._plot_line(
            axes[2, 3], ev, self.velocity_norms,
            "Velocity Norm (t=0.5)", "L2 Norm", "tab:brown",
        )

        # Row 4: Sample Health
        self._plot_line(
            axes[3, 0], ev, self.sample_norm_means,
            "Sample L2 Norm", "Mean Norm", "tab:blue",
            ref_line=self.train_norm_mean,
        )
        self._plot_line(
            axes[3, 1], ev, self.inter_sample_cos_sims,
            "Inter-Sample Cos Sim", "Cos Sim", "tab:olive",
        )
        self._plot_line(
            axes[3, 2], ev, self.centroid_cos_sims,
            "Centroid Cos Sim", "Cos Sim", "tab:cyan",
        )
        self._plot_grad_norm(axes[3, 3], ep)

        fig.suptitle(
            f"Flow Matching Training Progress \u2014 Epoch {epoch}",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # Individual panel plot methods
    # -----------------------------------------------------------------

    def _plot_line(
        self, ax, x, y, title: str, ylabel: str, color: str,
        ref_line: Optional[float] = None,
    ):
        """Plot a line chart with raw + smoothed, or 'No data' if empty."""
        xs, ys = self._filter_nan(x, y)
        if not xs:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            return

        xs, ys = list(xs), list(ys)
        ax.plot(xs, ys, color=color, alpha=0.3, linewidth=1, label="Raw")
        ax.plot(xs, self._smooth(ys), color=color, linewidth=2, label="Smooth")
        if ref_line is not None:
            ax.axhline(ref_line, color="tab:red", ls="--", alpha=0.7, label="Train ref")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_loss_overlay(self, ax, ep, log_scale: bool = False):
        """Train vs Val loss with raw + smoothed lines."""
        has_data = False
        for data, label, color in [
            (self.train_losses, "Train", "blue"),
            (self.val_losses, "Val", "orange"),
        ]:
            xs, ys = self._filter_nan(ep, data)
            if xs:
                xs, ys = list(xs), list(ys)
                ax.plot(xs, ys, color=color, alpha=0.3, linewidth=1)
                ax.plot(xs, self._smooth(ys), color=color, linewidth=2,
                        label=f"{label} (smooth)")
                has_data = True

        if not has_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")

        title = "Train/Val Loss (log)" if log_scale else "Train/Val Loss"
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_ot_cost_ratio(self, ax, ep):
        """OT cost ratio: OT_cost / random_cost per epoch.

        Values < 1 mean OT coupling is reducing transport cost.
        A flat line near 0.3-0.7 is typical and confirms OT is active.
        """
        xs, ys = self._filter_nan(ep, self.ot_cost_ratios)
        if not xs:
            ax.set_title("OT Cost Ratio")
            ax.text(0.5, 0.5, "No data\n(OT disabled?)", ha="center",
                    va="center", transform=ax.transAxes, fontsize=12,
                    color="gray")
            return

        xs, ys = list(xs), list(ys)
        ax.plot(xs, ys, color="tab:purple", alpha=0.3, linewidth=1,
                label="Raw")
        ax.plot(xs, self._smooth(ys), color="tab:purple", linewidth=2,
                label="Smooth")
        ax.axhline(1.0, color="tab:red", ls="--", alpha=0.5,
                   label="No OT (ratio=1)")
        ax.set_title("OT Cost Ratio")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("OT / Random")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_learning_rate(self, ax, ep):
        """Plot learning rate schedule."""
        xs, ys = self._filter_nan(ep, self.lrs)
        if not xs:
            ax.set_title("Learning Rate")
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            return

        ax.plot(list(xs), list(ys), color="green", linewidth=2)
        ax.set_title("Learning Rate")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    def _plot_per_dim_scatter(self, ax):
        """Per-dim mean scatter: generated vs training."""
        if self.latest_gen_mean_per_dim is None:
            ax.set_title("Per-Dim Mean: Gen vs Train")
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            return

        train_m = self.train_mean_per_dim.numpy()
        gen_m = self.latest_gen_mean_per_dim.numpy()
        ax.scatter(train_m, gen_m, alpha=0.3, s=4, color="tab:blue",
                   edgecolors="none")
        lo = min(train_m.min(), gen_m.min())
        hi = max(train_m.max(), gen_m.max())
        ax.plot([lo, hi], [lo, hi], "r--", alpha=0.6, linewidth=1.5,
                label="y=x")
        ax.set_title("Per-Dim Mean: Gen vs Train")
        ax.set_xlabel("Train")
        ax.set_ylabel("Generated")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_grad_norm(self, ax, ep):
        """Gradient norm with raw + smoothed."""
        xs, ys = self._filter_nan(ep, self.grad_norms)
        if not xs:
            ax.set_title("Gradient Norm")
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            return

        xs, ys = list(xs), list(ys)
        ax.plot(xs, ys, color="tab:red", alpha=0.3, linewidth=1, label="Raw")
        ax.plot(xs, self._smooth(ys), color="tab:red", linewidth=2, label="Smooth")
        ax.set_title("Gradient Norm")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("L2 Norm")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)


# =============================================================================
# EXPERIMENT
# =============================================================================


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    """Train Flow Matching model for HDC vector generation."""

    # =========================================================================
    # Setup
    # =========================================================================

    custom_tmpdir = Path(e.path) / ".tmp_checkpoints"
    custom_tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(custom_tmpdir)
    tempfile.tempdir = str(custom_tmpdir)

    pl.seed_everything(e.SEED)

    # Build conditioning interface
    conditioning: Optional[MultiCondition] = None
    condition_dim = 0
    if e.CONDITIONS:
        conditioning = build_condition(e.CONDITIONS)
        condition_dim = conditioning.condition_dim

    e.log("=" * 60)
    e.log("Flow Matching Training")
    e.log("=" * 60)
    e.log(f"Dataset: {e.DATASET}")
    e.log(f"Encoder: {e.ENCODER_PATH or '(auto-created for testing)'}")
    e.log(f"Architecture: {e.NUM_BLOCKS} blocks, {e.HIDDEN_DIM} hidden dim")
    e.log(f"OT coupling: {e.USE_OT_COUPLING}")
    if conditioning:
        e.log(f"Conditioning: {conditioning.name} (dim={condition_dim})")
        e.log(f"  {conditioning.description}")
    else:
        e.log("Conditioning: unconditional")
    e.log(f"Training: {e.EPOCHS} epochs, batch size {e.BATCH_SIZE}")
    e.log("=" * 60)

    if e.DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(e.DEVICE)

    if e.HDC_DEVICE == "auto":
        hdc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        hdc_device = torch.device(e.HDC_DEVICE)

    e.log(f"Device: {device}, HDC Device: {hdc_device}")

    # =========================================================================
    # Load encoder
    # =========================================================================

    hypernet = e.apply_hook("load_encoder", device=hdc_device)
    e.log(str(hypernet))

    hv_dim = hypernet.hv_dim
    if isinstance(hypernet, MultiHyperNet):
        graph_dim = hypernet.ensemble_graph_dim
        data_dim = hv_dim + graph_dim
        e.log(
            f"MultiHyperNet: K={hypernet.num_hypernets}, "
            f"hv_dim={hv_dim}, ensemble_graph_dim={graph_dim}, "
            f"data_dim={data_dim}"
        )
    else:
        data_dim = 2 * hv_dim
        e.log(f"HyperNet: hv_dim={hv_dim}, depth={hypernet.depth}, data_dim={data_dim}")

    e["config/hv_dim"] = hv_dim
    e["config/data_dim"] = data_dim

    # =========================================================================
    # Load and encode data
    # =========================================================================

    train_encoded, valid_encoded = e.apply_hook(
        "load_and_encode_data",
        hypernet=hypernet,
        device=hdc_device,
        conditioning=conditioning,
    )

    # num_workers=0 because data is already pre-computed in memory (no I/O
    # benefit) and forked workers conflict with CUDA initialization.
    train_loader = DataLoader(
        train_encoded,
        batch_size=e.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_encoded,
        batch_size=e.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # =========================================================================
    # Create model
    # =========================================================================

    model = FlowMatchingModel(
        data_dim=data_dim,
        hidden_dim=e.HIDDEN_DIM,
        num_blocks=e.NUM_BLOCKS,
        time_embed_dim=e.TIME_EMBED_DIM,
        condition_dim=condition_dim,
        condition_embed_dim=e.CONDITION_EMBED_DIM,
        dropout=e.DROPOUT,
        use_ot_coupling=e.USE_OT_COUPLING,
        solver_method=e.SOLVER_METHOD,
        default_sample_steps=e.SAMPLE_STEPS,
        lr=e.LEARNING_RATE,
        weight_decay=e.WEIGHT_DECAY,
        warmup_epochs=e.WARMUP_EPOCHS,
    )

    # Fit standardization
    if e.STANDARDIZE:
        model = e.apply_hook(
            "fit_standardization",
            model=model,
            train_encoded=train_encoded,
        )

    num_params = sum(p.numel() for p in model.parameters())
    e.log(f"Model parameters: {num_params:,}")
    e["config/num_parameters"] = num_params

    # =========================================================================
    # Prepare training tracker
    # =========================================================================

    # Extract training vectors for statistics (subsample for speed)
    train_vectors = []
    for d in train_encoded[: min(5000, len(train_encoded))]:
        node = d.node_terms.view(-1)
        graph = d.graph_terms.view(-1)
        train_vectors.append(torch.cat([node, graph]).float())
    train_vectors = torch.stack(train_vectors).as_subclass(torch.Tensor)

    # Extract validation vectors for reconstruction evaluation
    recon_vectors = []
    for d in valid_encoded[: min(200, len(valid_encoded))]:
        node = d.node_terms.view(-1)
        graph = d.graph_terms.view(-1)
        recon_vectors.append(torch.cat([node, graph]).float())
    recon_vectors = torch.stack(recon_vectors).as_subclass(torch.Tensor)

    tracking_callback = TrainingTrackerCallback(
        experiment=e,
        train_mean_per_dim=train_vectors.mean(dim=0),
        train_std_per_dim=train_vectors.std(dim=0),
        train_norm_mean=train_vectors.norm(dim=-1).mean().item(),
        recon_vectors=recon_vectors,
        eval_every_n_epochs=e.EVAL_TRACKING_EVERY_N,
    )

    e.log(
        f"Tracking callback: eval every {e.EVAL_TRACKING_EVERY_N} epochs, "
        f"{tracking_callback.num_eval_samples} samples, "
        f"{len(recon_vectors)} recon vectors"
    )

    # =========================================================================
    # Train
    # =========================================================================

    callbacks = [
        tracking_callback,
    ]
    if e.USE_EMA:
        callbacks.append(EMAWeightAveraging(decay=e.EMA_DECAY))
    callbacks.extend([
        ModelCheckpoint(
            dirpath=e.path,
            filename="best-{epoch:03d}-{val/loss:.6f}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ])

    logger = CSVLogger(e.path, name="logs")
    trainer = Trainer(
        max_epochs=e.EPOCHS,
        accelerator=device.type,
        devices=1,
        precision=e.PRECISION,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=e.path,
        log_every_n_steps=10,
        gradient_clip_val=e.GRADIENT_CLIP_VAL,
        enable_progress_bar=True,
    )

    e.log("\nStarting training...")
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
        e.log("Training stopped gracefully (CTRL+C)")
        e.log("-" * 40)

    # =========================================================================
    # Evaluate
    # =========================================================================

    e.log("\nComputing validation NLL...")
    e.apply_hook(
        "compute_validation_nll",
        model=model,
        valid_encoded=valid_encoded,
        device=device,
    )

    e.log("\nGenerating samples...")
    e.apply_hook(
        "evaluate_samples",
        model=model,
        train_encoded=train_encoded,
        device=device,
    )

    e.log("\n" + "=" * 60)
    e.log("Experiment completed!")
    e.log("=" * 60)


# =============================================================================
# HOOKS
# =============================================================================


@experiment.hook("load_encoder", default=True)
def load_encoder(e: Experiment, device: torch.device):
    """Load HyperNet or MultiHyperNet encoder from checkpoint."""
    if e.ENCODER_PATH and Path(e.ENCODER_PATH).exists():
        hypernet = load_hypernet(e.ENCODER_PATH, device=str(device))
    elif e.__TESTING__:
        # For testing: create a small encoder on-the-fly
        from graph_hdc.hypernet.configs import get_config

        config = get_config("ZINC_SMILES_HRR_256_F64_5G1NG4")
        config.device = str(device)
        config.dtype = "float32"
        hypernet = HyperNet(config)
        hypernet = hypernet.to(device)
    else:
        raise ValueError(
            "ENCODER_PATH must be set to a valid HyperNet checkpoint path. "
            "Use --__TESTING__ True for quick tests without an encoder."
        )

    hypernet.eval()
    return hypernet


@experiment.hook("load_and_encode_data", default=True)
def load_and_encode_data(
    e: Experiment,
    hypernet: HyperNet,
    device: torch.device,
    conditioning: Optional[MultiCondition] = None,
):
    """Load dataset and compute HDC encodings."""
    e.log("Loading dataset...")
    train_ds = get_split("train", dataset=e.DATASET.lower())
    valid_ds = get_split("valid", dataset=e.DATASET.lower())

    if e.__TESTING__:
        train_ds = train_ds[:64]
        valid_ds = valid_ds[:16]

    e.log(f"Train: {len(train_ds)}, Valid: {len(valid_ds)}")

    e.log("Computing HDC encodings...")
    train_encoded = post_compute_encodings(
        train_ds, hypernet, device=device, batch_size=e.ENCODER_BATCH_SIZE,
    )
    torch.cuda.empty_cache()
    valid_encoded = post_compute_encodings(
        valid_ds, hypernet, device=device, batch_size=e.ENCODER_BATCH_SIZE,
    )
    torch.cuda.empty_cache()
    e.log(f"Encoded: {len(train_encoded)} train, {len(valid_encoded)} valid")

    # Attach condition vectors if conditioning is active
    if conditioning is not None:
        e.log("Computing condition values...")
        for data_list in [train_encoded, valid_encoded]:
            for d in data_list:
                cond = conditioning.evaluate_multi(d.smiles)
                d.condition = cond
        e.log(f"  Condition dim: {conditioning.condition_dim}")

    return train_encoded, valid_encoded


@experiment.hook("fit_standardization", default=True)
def fit_standardization(e: Experiment, model: FlowMatchingModel, train_encoded):
    """Compute and set mean/std standardization on training vectors."""
    node_terms = torch.stack([d.node_terms for d in train_encoded]).float()
    graph_terms = torch.stack([d.graph_terms for d in train_encoded]).float()
    vectors = torch.cat([node_terms, graph_terms], dim=-1)

    mean = vectors.mean(dim=0)
    std = vectors.std(dim=0).clamp(min=1e-8)
    model.set_standardization(mean, std)
    e.log(
        f"Standardization: mean [{mean.min().item():.4f}, {mean.max().item():.4f}], "
        f"std [{std.min().item():.4f}, {std.max().item():.4f}]"
    )

    return model


@experiment.hook("compute_validation_nll", default=True)
def compute_validation_nll(
    e: Experiment,
    model: FlowMatchingModel,
    valid_encoded,
    device: torch.device,
):
    """Compute NLL on a subset of validation data."""
    model.eval()
    model.to(device)

    num_eval = min(e.NLL_EVAL_SAMPLES, len(valid_encoded))
    eval_loader = DataLoader(valid_encoded[:num_eval], batch_size=64, shuffle=False)

    all_nll = []
    for batch in eval_loader:
        batch = batch.to(device)
        x1 = model._extract_vectors(batch)
        condition = model._extract_condition(batch)
        nll = model.compute_nll(x1, condition=condition, num_steps=e.NLL_EVAL_STEPS)
        all_nll.append(nll.cpu())

    all_nll = torch.cat(all_nll)
    mean_nll = all_nll.mean().item()
    std_nll = all_nll.std().item()

    e.log(f"Validation NLL: {mean_nll:.4f} +/- {std_nll:.4f}")
    e["results/val_nll_mean"] = mean_nll
    e["results/val_nll_std"] = std_nll

    bpd = mean_nll / (model.data_dim * math.log(2))
    e.log(f"Bits per dimension: {bpd:.4f}")
    e["results/val_bpd"] = bpd


@experiment.hook("evaluate_samples", default=True)
def evaluate_samples(
    e: Experiment,
    model: FlowMatchingModel,
    train_encoded,
    device: torch.device,
):
    """Generate samples and compare distribution statistics to training data."""
    model.eval()
    model.to(device)

    num_gen = 1000
    with torch.no_grad():
        generated = model.sample(num_gen, device=device)

    gen_mean = generated.mean(dim=0)
    gen_std = generated.std(dim=0)

    # Training data statistics (subsample for speed)
    train_vectors = []
    for d in train_encoded[: min(5000, len(train_encoded))]:
        node = d.node_terms.view(-1)
        graph = d.graph_terms.view(-1)
        train_vectors.append(torch.cat([node, graph]))
    train_vectors = torch.stack(train_vectors)

    train_mean = train_vectors.mean(dim=0)
    train_std = train_vectors.std(dim=0)

    mean_mae = (gen_mean.cpu() - train_mean).abs().mean().item()
    std_mae = (gen_std.cpu() - train_std).abs().mean().item()

    e.log(f"Generated {num_gen} samples")
    e.log(f"Mean MAE (gen vs train): {mean_mae:.6f}")
    e.log(f"Std MAE (gen vs train): {std_mae:.6f}")

    e["results/sample_mean_mae"] = mean_mae
    e["results/sample_std_mae"] = std_mae

    # Inter-sample cosine similarity
    gen_normed = F.normalize(generated[:500].cpu(), dim=-1)
    cos_sim = gen_normed @ gen_normed.T
    cos_sim.fill_diagonal_(0)
    triu_mask = torch.triu(torch.ones_like(cos_sim), diagonal=1).bool()
    cos_values = cos_sim[triu_mask]

    e.log(
        f"Inter-sample cosine sim: {cos_values.mean():.4f} +/- {cos_values.std():.4f}"
    )
    e["results/inter_sample_cos_mean"] = cos_values.mean().item()
    e["results/inter_sample_cos_std"] = cos_values.std().item()


experiment.run_if_main()
