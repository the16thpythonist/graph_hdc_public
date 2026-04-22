#!/usr/bin/env python
"""
Train a (Variational) Autoencoder on HDC [edge_terms | graph_terms] vectors.

Learns a compressed latent representation of HDC graph encodings.  Supports
both standard AE and VAE modes with configurable architecture.

The encoder must be created with RRWP features using
``experiments/scripts/create_rrwp_encoder.py``.

Usage:
    # Quick test
    python train_autoencoder.py --__TESTING__ True

    # Full training (AE)
    python train_autoencoder.py --__DEBUG__ False --ENCODER_PATH /path/to/encoder.ckpt

    # Full training (VAE)
    python train_autoencoder.py --__DEBUG__ False --VARIATIONAL True --ENCODER_PATH /path/to/encoder.ckpt
"""
from __future__ import annotations

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

from graph_hdc.datasets.utils import get_split, post_compute_encodings
from graph_hdc.hypernet import load_hypernet
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.models.autoencoder import HDCAutoencoder
from graph_hdc.utils.experiment_helpers import GracefulInterruptHandler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
ENCODER_PATH: str = "/media/ssd2/Programming/_branch/graph_hdc_public/experiments/encoders/zinc_d1024_depth3_k6_10_14_b8.ckpt"

# :param DEVICE:
#     Device for model training. Options: "auto", "cpu", "cuda".
DEVICE: str = "cuda"

# :param HDC_DEVICE:
#     Device for HDC encoding. Options: "auto", "cpu", "cuda".
HDC_DEVICE: str = "cpu"

# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------

# :param LATENT_DIM:
#     Bottleneck / latent space dimension.
LATENT_DIM: int = 256

# :param ENCODER_HIDDEN_DIMS:
#     Hidden layer sizes for the encoder network.
ENCODER_HIDDEN_DIMS: List[int] = [3000, 2000, 1000, 500]

# :param DECODER_HIDDEN_DIMS:
#     Hidden layer sizes for the decoder network.
#     None means use reversed ENCODER_HIDDEN_DIMS.
DECODER_HIDDEN_DIMS: Optional[List[int]] = None

# :param RESBLOCK_DEPTH:
#     Number of SwiGLU residual blocks per projection layer in the
#     encoder and decoder networks.
RESBLOCK_DEPTH: int = 2

# :param TRAINING_TARGET:
#     Which HDC components to reconstruct. Options: "both" (edge + graph),
#     "edge" (edge terms only), "graph" (graph terms only).
TRAINING_TARGET: str = "edge"

# :param VARIATIONAL:
#     Whether to use a variational autoencoder (VAE) with reparameterization
#     trick and KL divergence regularization.
VARIATIONAL: bool = False

# -----------------------------------------------------------------------------
# Loss Weights
# -----------------------------------------------------------------------------

# :param RECON_LOSS_TYPE:
#     Reconstruction loss type: "mse" or "mae".
RECON_LOSS_TYPE: str = "berhu"

# :param MSE_WEIGHT:
#     Weight for the reconstruction loss component (MSE or MAE).
MSE_WEIGHT: float = 1.0

# :param COSINE_WEIGHT:
#     Weight for cosine similarity loss component (1 - cos_sim).
COSINE_WEIGHT: float = 0.0

# :param KL_WEIGHT:
#     Maximum beta for KL divergence loss (VAE only). This is the final
#     value after warmup.
KL_WEIGHT: float = 1e-3

# :param KL_WARMUP_EPOCHS:
#     Number of epochs to linearly ramp beta from 0 to KL_WEIGHT (VAE only).
KL_WARMUP_EPOCHS: int = 20

# :param LOSS_CLAMP:
#     If set, clamp per-sample reconstruction loss to this maximum before
#     averaging. Prevents outlier samples from causing gradient explosions.
#     None means no clamping.
LOSS_CLAMP: Optional[float] = None

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------

# :param EPOCHS:
#     Number of training epochs.
EPOCHS: int = 500

# :param BATCH_SIZE:
#     Batch size for training.
BATCH_SIZE: int = 128

# :param LEARNING_RATE:
#     Learning rate for AdamW optimizer.
LEARNING_RATE: float = 1e-4

# :param WEIGHT_DECAY:
#     Weight decay for AdamW.
WEIGHT_DECAY: float = 1e-5

# :param WARMUP_EPOCHS:
#     Number of linear warmup epochs for learning rate.
WARMUP_EPOCHS: int = 5

# :param GRADIENT_CLIP_VAL:
#     Gradient clipping value.
GRADIENT_CLIP_VAL: float = 1.0

# :param USE_EMA:
#     Whether to use Exponential Moving Average of model weights.
USE_EMA: bool = False

# :param EMA_DECAY:
#     EMA decay rate. Only used when USE_EMA is True.
EMA_DECAY: float = 0.999

# :param ENCODER_BATCH_SIZE:
#     Batch size for HDC encoding.
ENCODER_BATCH_SIZE: int = 256

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

# :param RECON_EVAL_EVERY_N_EPOCHS:
#     How often (in epochs) to run molecular-level reconstruction evaluation.
RECON_EVAL_EVERY_N_EPOCHS: int = 10

# :param RECON_EVAL_N_SAMPLES:
#     Number of validation molecules to reconstruct per evaluation.
RECON_EVAL_N_SAMPLES: int = 50

# :param SCATTER_EVERY_N_EPOCHS:
#     How often (in epochs) to compute input-vs-output per-dim scatter plot.
SCATTER_EVERY_N_EPOCHS: int = 5

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------

# :param NUM_SUBSAMPLE:
#     Optional subsample size. None = full dataset.
NUM_SUBSAMPLE: Optional[int] = None

# :param SEED:
#     Random seed.
SEED: int = 42

# :param PRECISION:
#     Training precision.
PRECISION: str = "32"

# -----------------------------------------------------------------------------
# Debug/Testing
# -----------------------------------------------------------------------------

# :param __DEBUG__:
#     Debug mode - reuses same output folder.
__DEBUG__: bool = True

# :param __TESTING__:
#     Testing mode - minimal iterations for validation.
__TESTING__: bool = False


# =============================================================================
# HELPERS
# =============================================================================


def _extract_vector_from_data(d, training_target: str = "both") -> torch.Tensor:
    """Extract HDC vectors from a single Data object.

    After the edge_terms -> node_terms swap, node_terms contains edge_terms.
    Returns edge_terms, graph_terms, or both concatenated depending on
    ``training_target``.
    """
    node = d.node_terms.view(-1)
    graph = d.graph_terms.view(-1)
    if training_target == "edge":
        return node.float().as_subclass(torch.Tensor)
    if training_target == "graph":
        return graph.float().as_subclass(torch.Tensor)
    return torch.cat([node, graph]).float().as_subclass(torch.Tensor)


# =============================================================================
# AUTOENCODER TRACKING CALLBACK
# =============================================================================


class AutoencoderTrackingCallback(pl.Callback):
    """Tracks training progress and produces a 4x4 panel visualization.

    Layout:
        Row 1: Train/Val Loss | Train/Val Loss (log) | Learning Rate | Loss Ratio
        Row 2: MSE (train+val) | Cos Sim (train+val) | KL Divergence | Beta Schedule
        Row 3: Latent Mean Norm | Latent Std per Dim | Grad Norm | Input vs Output Scatter
        Row 4: Mol Validity | Mol Exact Match | Mol Tanimoto | Mol HDC Cos Sim
        Row 5: Loss Histogram | Loss Percentiles | Cos Sim Histogram | Tail Ratio
        Row 6: Participation Ratio | Per-Dim Variance | Update/Weight Ratio | Edge vs Graph Cos Sim

    KL / Beta panels show "N/A" for non-variational mode.
    Row 4 is populated by ReconstructionEvalCallback via ``record_recon_metrics()``.
    """

    def __init__(
        self,
        experiment: Experiment,
        variational: bool = False,
        val_vectors: Optional[Tensor] = None,
        scatter_every_n_epochs: int = 5,
        smoothing_window: int = 5,
        hv_dim: Optional[int] = None,
    ):
        super().__init__()
        self.experiment = experiment
        self.variational = variational
        self.val_vectors = val_vectors
        self.scatter_every_n_epochs = scatter_every_n_epochs
        self.smoothing_window = smoothing_window
        self.hv_dim = hv_dim

        # Per-epoch histories
        self.epochs: list[int] = []
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_mses: list[float] = []
        self.val_mses: list[float] = []
        self.train_cos_sims: list[float] = []
        self.val_cos_sims: list[float] = []
        self.kl_divs: list[float] = []
        self.betas: list[float] = []
        self.lrs: list[float] = []
        self.grad_norms: list[float] = []
        self.latent_mean_norms: list[float] = []
        self.latent_std_means: list[float] = []

        # Molecular reconstruction metrics (fed by ReconstructionEvalCallback)
        self.recon_eval_epochs: list[int] = []
        self.recon_validities: list[float] = []
        self.recon_exact_matches: list[float] = []
        self.recon_tanimotos: list[float] = []
        self.recon_hdc_cos_sims: list[float] = []

        # Per-sample distribution data (Row 5)
        self.loss_percentiles_epochs: list[int] = []
        self.loss_p5: list[float] = []
        self.loss_p25: list[float] = []
        self.loss_p50: list[float] = []
        self.loss_p75: list[float] = []
        self.loss_p95: list[float] = []
        self.tail_ratios: list[float] = []  # fraction of samples > 2x median
        self.latest_per_sample_losses: Optional[Tensor] = None
        self.latest_per_sample_cos_sims: Optional[Tensor] = None

        # Scatter plot data
        self.latest_recon_mean_per_dim: Optional[Tensor] = None
        self.latest_input_mean_per_dim: Optional[Tensor] = None

        # Gradient norm accumulator
        self._grad_norm_sum = 0.0
        self._grad_norm_count = 0

        # Row 6: Latent space & training health
        self.participation_ratios: list[float] = []
        self.latest_per_dim_var: Optional[Tensor] = None
        self.edge_cos_sims: list[float] = []
        self.graph_cos_sims: list[float] = []

        # Update-to-weight ratio (per layer, per epoch)
        self._param_snapshot: dict[str, Tensor] = {}
        self._update_ratios_epoch: dict[str, list[float]] = {}
        self._layer_short_names: dict[str, str] = {}
        self.latest_uwr_means: dict[str, float] = {}
        self.latest_uwr_stds: dict[str, float] = {}

    def record_recon_metrics(
        self,
        epoch: int,
        validity: float,
        exact_match: float,
        tanimoto: float,
        hdc_cos_sim: float,
    ):
        """Called by ReconstructionEvalCallback to feed molecular metrics."""
        self.recon_eval_epochs.append(epoch)
        self.recon_validities.append(validity)
        self.recon_exact_matches.append(exact_match)
        self.recon_tanimotos.append(tanimoto)
        self.recon_hdc_cos_sims.append(hdc_cos_sim)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _smooth(self, values: list[float], window: Optional[int] = None) -> list[float]:
        w = window or self.smoothing_window
        if len(values) < w:
            return values
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - w + 1)
            smoothed.append(sum(values[start : i + 1]) / (i - start + 1))
        return smoothed

    def _filter_nan(self, x: list, y: list[float]):
        pairs = [(xi, yi) for xi, yi in zip(x[: len(y)], y) if yi == yi]
        if not pairs:
            return [], []
        return zip(*pairs)

    # -----------------------------------------------------------------
    # Per-epoch hooks
    # -----------------------------------------------------------------

    @staticmethod
    def _block_key(name: str) -> str:
        """Map a parameter name to its block-level group key.

        Groups all parameters within the same ProjectionBlock (or standalone
        linear layer) together so the update-to-weight bar chart shows one
        bar per logical layer.

        Examples:
            'encoder_backbone.0.proj.weight'        -> 'Enc 1'
            'encoder_backbone.0.blocks.1.w_gate.weight' -> 'Enc 1'
            'encoder_backbone.2.blocks.0.norm.weight'   -> 'Enc 3'
            'fc_latent.weight'                      -> 'Bottleneck'
            'decoder.0.proj.weight'                 -> 'Dec 1'
            'decoder.3.weight'                      -> 'Dec Out'
        """
        parts = name.split(".")
        if parts[0] == "encoder_backbone":
            return f"Enc {int(parts[1]) + 1}"
        if parts[0] == "fc_latent":
            return "Bottleneck"
        if parts[0] == "fc_mu":
            return "FC \u03bc"
        if parts[0] == "fc_logvar":
            return "FC log\u03c3\u00b2"
        if parts[0] == "decoder":
            # The final plain Linear has no sub-modules (e.g. 'decoder.3.weight')
            # while ProjectionBlocks have 'decoder.0.proj.weight' etc.
            if len(parts) == 3:
                return "Dec Out"
            return f"Dec {int(parts[1]) + 1}"
        return name

    def on_train_epoch_start(self, trainer, pl_module):
        self._grad_norm_sum = 0.0
        self._grad_norm_count = 0
        self._param_snapshot = {}
        self._update_ratios_epoch: dict[str, list[float]] = {}
        # Build block-key mapping once
        if not self._layer_short_names:
            for name, p in pl_module.named_parameters():
                if p.requires_grad:
                    self._layer_short_names[name] = self._block_key(name)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        total_norm_sq = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.norm(2).item() ** 2
        self._grad_norm_sum += total_norm_sq ** 0.5
        self._grad_norm_count += 1

        # Snapshot all parameters for update-to-weight ratio
        self._param_snapshot = {
            name: p.data.clone()
            for name, p in pl_module.named_parameters()
            if p.requires_grad
        }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._param_snapshot:
            return

        # Accumulate squared norms per block for this step
        block_update_sq: dict[str, float] = {}
        block_weight_sq: dict[str, float] = {}
        for name, p in pl_module.named_parameters():
            if name not in self._param_snapshot:
                continue
            key = self._layer_short_names[name]
            old = self._param_snapshot[name]
            block_update_sq[key] = block_update_sq.get(key, 0.0) + (
                (p.data - old).norm().item() ** 2
            )
            block_weight_sq[key] = block_weight_sq.get(key, 0.0) + (
                old.norm().item() ** 2
            )

        for key in block_update_sq:
            w_norm = block_weight_sq[key] ** 0.5
            if w_norm > 0:
                ratio = block_update_sq[key] ** 0.5 / w_norm
                self._update_ratios_epoch.setdefault(key, []).append(ratio)

        self._param_snapshot = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch
        self.epochs.append(epoch)
        metrics = trainer.callback_metrics

        # Losses
        tl = float(metrics.get("train/loss", float("nan")))
        vl = float(metrics.get("val/loss", float("nan")))
        self.train_losses.append(tl)
        self.val_losses.append(vl)

        # MSE and cosine similarity
        self.train_mses.append(float(metrics.get("train/recon", float("nan"))))
        self.val_mses.append(float(metrics.get("val/recon", float("nan"))))
        self.train_cos_sims.append(float(metrics.get("train/cos_sim", float("nan"))))
        self.val_cos_sims.append(float(metrics.get("val/cos_sim", float("nan"))))

        # VAE-specific
        if self.variational:
            self.kl_divs.append(float(metrics.get("train/kl", float("nan"))))
            self.betas.append(float(metrics.get("train/beta", float("nan"))))

        # Learning rate
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0]["lr"]
            self.lrs.append(lr)
        else:
            self.lrs.append(float("nan"))

        # Gradient norm
        if self._grad_norm_count > 0:
            gn = self._grad_norm_sum / self._grad_norm_count
            self.grad_norms.append(gn)
        else:
            self.grad_norms.append(float("nan"))

        # Latent stats (logged by model)
        self.latent_mean_norms.append(
            float(metrics.get("val/z_mean_norm", float("nan")))
        )
        self.latent_std_means.append(
            float(metrics.get("val/z_mean_std_per_dim", float("nan")))
        )

        # PyComex tracking
        self.experiment.track("ae_loss_train", tl)
        self.experiment.track("ae_loss_val", vl)
        self.experiment.track("ae_recon_train", self.train_mses[-1])
        self.experiment.track("ae_recon_val", self.val_mses[-1])
        self.experiment.track("ae_cos_sim_train", self.train_cos_sims[-1])
        self.experiment.track("ae_cos_sim_val", self.val_cos_sims[-1])
        if self.variational:
            self.experiment.track("ae_kl", self.kl_divs[-1])
            self.experiment.track("ae_beta", self.betas[-1])
        self.experiment.track("ae_lr", self.lrs[-1])
        self.experiment.track("ae_grad_norm", self.grad_norms[-1])
        self.experiment.track("ae_latent_mean_norm", self.latent_mean_norms[-1])
        self.experiment.track("ae_latent_std_per_dim", self.latent_std_means[-1])
        if self.participation_ratios:
            self.experiment.track("ae_participation_ratio", self.participation_ratios[-1])
        if self.edge_cos_sims:
            self.experiment.track("ae_edge_cos_sim", self.edge_cos_sims[-1])
            self.experiment.track("ae_graph_cos_sim", self.graph_cos_sims[-1])

        # Scatter (every N epochs)
        if (
            self.val_vectors is not None
            and epoch % self.scatter_every_n_epochs == 0
        ):
            self._compute_scatter(pl_module)

        # Per-sample loss distribution
        if self.val_vectors is not None:
            self._compute_per_sample_stats(pl_module, epoch)

        # Row 6: participation ratio, per-dim variance, edge/graph cos sim
        if self.val_vectors is not None:
            self._compute_latent_and_component_stats(pl_module)

        # Update-to-weight ratio: aggregate epoch stats
        self._aggregate_update_to_weight_ratios()

        # Generate plot
        try:
            fig = self._create_metrics_plot(epoch)
            self.experiment.track("ae_training_metrics", fig)
            plt.close(fig)
        except Exception as ex:
            self.experiment.log(f"Warning: Failed to create metrics plot: {ex}")

    @torch.no_grad()
    def _compute_scatter(self, model: HDCAutoencoder):
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        try:
            x = self.val_vectors.to(device)
            x_hat = model.reconstruct(x).cpu()
            self.latest_input_mean_per_dim = self.val_vectors.mean(dim=0)
            self.latest_recon_mean_per_dim = x_hat.mean(dim=0)
        finally:
            if was_training:
                model.train()

    @torch.no_grad()
    def _compute_per_sample_stats(self, model: HDCAutoencoder, epoch: int):
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        try:
            x = self.val_vectors.to(device)
            stats = model.per_sample_losses(x)
            losses = stats["loss"].cpu()
            cos_sims = stats["cos_sim"].cpu()

            self.latest_per_sample_losses = losses
            self.latest_per_sample_cos_sims = cos_sims

            # Percentiles
            self.loss_percentiles_epochs.append(epoch)
            self.loss_p5.append(float(torch.quantile(losses, 0.05)))
            self.loss_p25.append(float(torch.quantile(losses, 0.25)))
            self.loss_p50.append(float(torch.quantile(losses, 0.50)))
            self.loss_p75.append(float(torch.quantile(losses, 0.75)))
            self.loss_p95.append(float(torch.quantile(losses, 0.95)))

            # Tail ratio: fraction of samples with loss > 2x median
            median = torch.median(losses)
            self.tail_ratios.append(float((losses > 2.0 * median).float().mean()))
        finally:
            if was_training:
                model.train()

    @torch.no_grad()
    def _compute_latent_and_component_stats(self, model: HDCAutoencoder):
        """Compute participation ratio, per-dim variance, and edge/graph cos sim."""
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        try:
            x = self.val_vectors.to(device)
            enc_out = model.encode(x)
            z = (enc_out["mu"] if model.variational else enc_out["z"]).cpu()

            # Per-dimension variance and participation ratio
            per_dim_var = z.var(dim=0)
            self.latest_per_dim_var = per_dim_var
            var_sum = per_dim_var.sum().item()
            var_sq_sum = (per_dim_var ** 2).sum().item()
            pr = (var_sum ** 2 / var_sq_sum) if var_sq_sum > 0 else 0.0
            self.participation_ratios.append(pr)

            # Edge vs graph component cosine similarity
            if self.hv_dim is not None:
                x_hat = model.decode(z.to(device)).cpu()
                x_cpu = self.val_vectors
                edge_cos = F.cosine_similarity(
                    x_hat[:, : self.hv_dim], x_cpu[:, : self.hv_dim], dim=-1,
                ).mean().item()
                graph_cos = F.cosine_similarity(
                    x_hat[:, self.hv_dim :], x_cpu[:, self.hv_dim :], dim=-1,
                ).mean().item()
                self.edge_cos_sims.append(edge_cos)
                self.graph_cos_sims.append(graph_cos)
        finally:
            if was_training:
                model.train()

    def _aggregate_update_to_weight_ratios(self):
        """Aggregate per-step update-to-weight ratios into epoch summary."""
        self.latest_uwr_means = {}
        self.latest_uwr_stds = {}
        for key, ratios in self._update_ratios_epoch.items():
            if ratios:
                t = torch.tensor(ratios)
                self.latest_uwr_means[key] = t.mean().item()
                self.latest_uwr_stds[key] = t.std().item() if len(ratios) > 1 else 0.0

    # -----------------------------------------------------------------
    # 6x4 panel plot
    # -----------------------------------------------------------------

    def _create_metrics_plot(self, epoch: int) -> plt.Figure:
        fig, axes = plt.subplots(6, 4, figsize=(16, 21))
        ep = self.epochs

        # Row 1: Loss overview
        self._plot_loss_overlay(axes[0, 0], ep, log_scale=False)
        self._plot_loss_overlay(axes[0, 1], ep, log_scale=True)
        self._plot_lr(axes[0, 2], ep)
        self._plot_loss_ratio(axes[0, 3], ep)

        # Row 2: Loss components
        self._plot_train_val(
            axes[1, 0], ep,
            self.train_mses, self.val_mses,
            "Reconstruction Loss", "Loss",
        )
        self._plot_train_val(
            axes[1, 1], ep,
            self.train_cos_sims, self.val_cos_sims,
            "Cosine Similarity", "Cos Sim",
        )
        self._plot_kl(axes[1, 2], ep)
        self._plot_beta(axes[1, 3], ep)

        # Row 3: Latent space & training health
        self._plot_line(
            axes[2, 0], ep, self.latent_mean_norms,
            "Latent Mean Norm", "L2 Norm", "tab:blue",
        )
        self._plot_line(
            axes[2, 1], ep, self.latent_std_means,
            "Latent Std per Dim", "Std", "tab:green",
        )
        self._plot_line(
            axes[2, 2], ep, self.grad_norms,
            "Gradient Norm", "L2 Norm", "tab:red",
        )
        self._plot_scatter(axes[2, 3])

        # Row 4: Molecular reconstruction quality
        rev = self.recon_eval_epochs
        self._plot_line(
            axes[3, 0], rev, self.recon_validities,
            "Mol Validity", "%", "tab:green",
        )
        self._plot_line(
            axes[3, 1], rev, self.recon_exact_matches,
            "Mol Exact Match", "%", "tab:blue",
        )
        self._plot_line(
            axes[3, 2], rev, self.recon_tanimotos,
            "Mol Tanimoto Sim", "Similarity", "tab:orange",
        )
        self._plot_line(
            axes[3, 3], rev, self.recon_hdc_cos_sims,
            "Mol Recon HDC Cos Sim", "Cos Sim", "tab:purple",
        )

        # Row 5: Per-sample loss distribution
        self._plot_loss_histogram(axes[4, 0])
        self._plot_loss_percentiles(axes[4, 1])
        self._plot_cos_sim_histogram(axes[4, 2])
        self._plot_tail_ratio(axes[4, 3])

        # Row 6: Latent utilization & component analysis
        self._plot_participation_ratio(axes[5, 0], ep)
        self._plot_per_dim_variance(axes[5, 1])
        self._plot_update_to_weight(axes[5, 2])
        self._plot_edge_graph_cos_sim(axes[5, 3], ep)

        mode = "VAE" if self.variational else "AE"
        fig.suptitle(
            f"Autoencoder ({mode}) Training Progress \u2014 Epoch {epoch}",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # Individual panel methods
    # -----------------------------------------------------------------

    def _plot_line(
        self, ax, x, y, title: str, ylabel: str, color: str,
        ref_line: Optional[float] = None,
    ):
        xs, ys = self._filter_nan(x, y)
        if not xs:
            ax.set_title(title)
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
            )
            return

        xs, ys = list(xs), list(ys)
        ax.plot(xs, ys, color=color, alpha=0.3, linewidth=1, label="Raw")
        ax.plot(xs, self._smooth(ys), color=color, linewidth=2, label="Smooth")
        if ref_line is not None:
            ax.axhline(ref_line, color="tab:red", ls="--", alpha=0.7, label="Ref")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_loss_overlay(self, ax, ep, log_scale: bool = False):
        has_data = False
        for data, label, color in [
            (self.train_losses, "Train", "blue"),
            (self.val_losses, "Val", "orange"),
        ]:
            xs, ys = self._filter_nan(ep, data)
            if xs:
                xs, ys = list(xs), list(ys)
                ax.plot(xs, ys, color=color, alpha=0.3, linewidth=1)
                ax.plot(
                    xs, self._smooth(ys), color=color, linewidth=2,
                    label=f"{label} (smooth)",
                )
                has_data = True

        if not has_data:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
            )

        title = "Train/Val Loss (log)" if log_scale else "Train/Val Loss"
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        if log_scale and has_data:
            ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_train_val(
        self, ax, ep,
        train_data: list[float], val_data: list[float],
        title: str, ylabel: str,
    ):
        has_data = False
        for data, label, color in [
            (train_data, "Train", "tab:blue"),
            (val_data, "Val", "tab:orange"),
        ]:
            xs, ys = self._filter_nan(ep, data)
            if xs:
                xs, ys = list(xs), list(ys)
                ax.plot(xs, ys, color=color, alpha=0.3, linewidth=1)
                ax.plot(
                    xs, self._smooth(ys), color=color, linewidth=2,
                    label=f"{label} (smooth)",
                )
                has_data = True

        if not has_data:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
            )

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_lr(self, ax, ep):
        xs, ys = self._filter_nan(ep, self.lrs)
        if not xs:
            ax.set_title("Learning Rate")
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
            )
            return

        ax.plot(list(xs), list(ys), color="green", linewidth=2)
        ax.set_title("Learning Rate")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    def _plot_loss_ratio(self, ax, ep):
        ratios = []
        for tl, vl in zip(self.train_losses, self.val_losses):
            if tl == tl and vl == vl and vl > 0:
                ratios.append(tl / vl)
            else:
                ratios.append(float("nan"))

        xs, ys = self._filter_nan(ep, ratios)
        if not xs:
            ax.set_title("Train/Val Loss Ratio")
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
            )
            return

        xs, ys = list(xs), list(ys)
        ax.plot(xs, ys, color="tab:gray", alpha=0.3, linewidth=1, label="Raw")
        ax.plot(xs, self._smooth(ys), color="tab:gray", linewidth=2, label="Smooth")
        ax.axhline(1.0, color="tab:red", ls="--", alpha=0.5, label="ratio=1")
        ax.set_title("Train/Val Loss Ratio")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train / Val")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_kl(self, ax, ep):
        if not self.variational:
            ax.set_title("KL Divergence")
            ax.text(
                0.5, 0.5, "N/A\n(non-variational)", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
            )
            return

        self._plot_line(ax, ep, self.kl_divs, "KL Divergence", "KL", "tab:purple")

    def _plot_beta(self, ax, ep):
        if not self.variational:
            ax.set_title("Beta Schedule")
            ax.text(
                0.5, 0.5, "N/A\n(non-variational)", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
            )
            return

        xs, ys = self._filter_nan(ep, self.betas)
        if not xs:
            ax.set_title("Beta Schedule")
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
            )
            return

        ax.plot(list(xs), list(ys), color="tab:pink", linewidth=2)
        ax.set_title("Beta Schedule")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("\u03b2")
        ax.grid(True, alpha=0.3)
        if ys:
            ys_list = list(ys)
            xs_list = list(xs)
            ax.annotate(
                f"{ys_list[-1]:.4f}", xy=(xs_list[-1], ys_list[-1]),
                fontsize=9, color="tab:pink", fontweight="bold",
                ha="right", va="bottom",
            )

    def _plot_scatter(self, ax):
        if self.latest_input_mean_per_dim is None:
            ax.set_title("Per-Dim Mean: Input vs Recon")
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
            )
            return

        inp = self.latest_input_mean_per_dim.numpy()
        rec = self.latest_recon_mean_per_dim.numpy()
        ax.scatter(inp, rec, alpha=0.3, s=4, color="tab:blue", edgecolors="none")
        lo = min(inp.min(), rec.min())
        hi = max(inp.max(), rec.max())
        ax.plot([lo, hi], [lo, hi], "r--", alpha=0.6, linewidth=1.5, label="y=x")
        ax.set_title("Per-Dim Mean: Input vs Recon")
        ax.set_xlabel("Input")
        ax.set_ylabel("Reconstructed")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_loss_histogram(self, ax):
        if self.latest_per_sample_losses is None:
            ax.set_title("Per-Sample Loss Distribution")
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
            )
            return

        data = self.latest_per_sample_losses.numpy()
        ax.hist(data, bins=50, color="tab:blue", alpha=0.7, edgecolor="white", linewidth=0.5)
        median = float(np.median(data))
        ax.axvline(median, color="tab:red", ls="--", lw=1.5, label=f"median={median:.4f}")
        p95 = float(np.percentile(data, 95))
        ax.axvline(p95, color="tab:orange", ls=":", lw=1.5, label=f"p95={p95:.4f}")
        ax.set_title("Per-Sample Loss Distribution")
        ax.set_xlabel("Loss")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_loss_percentiles(self, ax):
        ep = self.loss_percentiles_epochs
        if not ep:
            ax.set_title("Loss Percentiles Over Time")
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
            )
            return

        ax.fill_between(ep, self.loss_p5, self.loss_p95, alpha=0.15, color="tab:blue", label="p5-p95")
        ax.fill_between(ep, self.loss_p25, self.loss_p75, alpha=0.3, color="tab:blue", label="p25-p75")
        ax.plot(ep, self.loss_p50, color="tab:blue", lw=2, label="median")
        ax.set_title("Loss Percentiles Over Time")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_cos_sim_histogram(self, ax):
        if self.latest_per_sample_cos_sims is None:
            ax.set_title("Per-Sample Cosine Similarity")
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
            )
            return

        data = self.latest_per_sample_cos_sims.numpy()
        ax.hist(data, bins=50, color="tab:green", alpha=0.7, edgecolor="white", linewidth=0.5)
        median = float(np.median(data))
        ax.axvline(median, color="tab:red", ls="--", lw=1.5, label=f"median={median:.4f}")
        ax.set_title("Per-Sample Cosine Similarity")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_tail_ratio(self, ax):
        ep = self.loss_percentiles_epochs
        if not ep or not self.tail_ratios:
            ax.set_title("Loss Tail Ratio (>2x median)")
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
            )
            return

        ax.plot(ep, self.tail_ratios, color="tab:red", lw=2, label="Tail ratio")
        ax.fill_between(ep, 0, self.tail_ratios, alpha=0.2, color="tab:red")
        ax.set_title("Loss Tail Ratio (>2x median)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Fraction")
        ax.set_ylim(0, max(0.05, max(self.tail_ratios) * 1.2))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # -----------------------------------------------------------------
    # Row 6 panels
    # -----------------------------------------------------------------

    def _plot_participation_ratio(self, ax, ep):
        self._plot_line(
            ax, ep, self.participation_ratios,
            "Participation Ratio", "PR", "teal",
        )

    def _plot_per_dim_variance(self, ax):
        if self.latest_per_dim_var is None:
            ax.set_title("Latent Per-Dim Variance")
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
            )
            return

        variances = self.latest_per_dim_var.numpy()
        sorted_var = np.sort(variances)[::-1]
        ax.bar(
            range(len(sorted_var)), sorted_var,
            color="tab:purple", alpha=0.7, width=1.0,
        )
        ax.set_title("Latent Per-Dim Variance (sorted)")
        ax.set_xlabel("Dimension (rank)")
        ax.set_ylabel("Variance")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

    def _plot_update_to_weight(self, ax):
        if not self.latest_uwr_means:
            ax.set_title("Update / Weight Ratio (per layer)")
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
            )
            return

        names = list(self.latest_uwr_means.keys())
        means = [self.latest_uwr_means[n] for n in names]
        stds = [self.latest_uwr_stds[n] for n in names]

        x_pos = np.arange(len(names))
        ax.bar(
            x_pos, means, yerr=stds,
            color="tab:orange", alpha=0.7, capsize=3, ecolor="tab:gray",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_title("Update / Weight Ratio (per layer)")
        ax.set_ylabel("||delta_w|| / ||w||")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

    def _plot_edge_graph_cos_sim(self, ax, ep):
        if not self.edge_cos_sims:
            ax.set_title("Edge vs Graph Recon Cos Sim")
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
            )
            return

        for data, label, color in [
            (self.edge_cos_sims, "Edge terms", "tab:blue"),
            (self.graph_cos_sims, "Graph terms", "tab:red"),
        ]:
            xs, ys = self._filter_nan(ep, data)
            if xs:
                xs, ys = list(xs), list(ys)
                ax.plot(xs, ys, color=color, alpha=0.3, linewidth=1)
                ax.plot(
                    xs, self._smooth(ys), color=color, linewidth=2,
                    label=f"{label} (smooth)",
                )

        ax.set_title("Edge vs Graph Recon Cos Sim")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cosine Similarity")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)


# =============================================================================
# RECONSTRUCTION EVALUATION CALLBACK
# =============================================================================


class ReconstructionEvalCallback(pl.Callback):
    """Molecular-level reconstruction evaluation.

    Every ``every_n_epochs`` epochs:
    1. Pass validation HDC vectors through the AE (deterministic).
    2. Decode reconstructed vectors back to molecules via HyperNet.
    3. Compute validity, exact match rate, and Tanimoto similarity.

    Args:
        experiment: PyComex Experiment instance.
        hypernet: HyperNet encoder/decoder.
        eval_data: List of encoded Data objects for evaluation.
        dataset: Dataset name (for ``reconstruct_for_eval``).
        hv_dim: Hypervector dimension (to split edge/graph terms).
        every_n_epochs: Evaluation frequency.
        n_eval: Number of molecules to evaluate.
    """

    def __init__(
        self,
        experiment: Experiment,
        hypernet: HyperNet,
        eval_data: list,
        tracking_callback: AutoencoderTrackingCallback,
        dataset: str = "zinc",
        hv_dim: int = 1024,
        training_target: str = "both",
        every_n_epochs: int = 10,
        n_eval: int = 50,
    ):
        super().__init__()
        self.experiment = experiment
        self.hypernet = hypernet
        self.eval_data = eval_data
        self.tracking_callback = tracking_callback
        self.dataset = dataset
        self.hv_dim = hv_dim
        self.training_target = training_target
        self.every_n_epochs = every_n_epochs
        self.n_eval = n_eval

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return
        self._evaluate(pl_module, epoch)

    @torch.no_grad()
    def _evaluate(self, model: HDCAutoencoder, epoch: int):
        import time

        from graph_hdc.hypernet.configs import FallbackDecoderSettings
        from graph_hdc.utils.chem import reconstruct_for_eval
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs, Draw

        self.experiment.log(
            f"[ReconEval] Epoch {epoch}: evaluating {self.n_eval} molecules..."
        )
        t0 = time.time()

        device = next(model.parameters()).device
        model.eval()

        subset = self.eval_data[: self.n_eval]
        original_smiles = [d.smiles for d in subset]

        # Build input tensor
        vectors = torch.stack([
            _extract_vector_from_data(d, self.training_target) for d in subset
        ])
        vectors = vectors.to(device)

        # Deterministic reconstruction
        reconstructed = model.reconstruct(vectors)

        # HDC-space metrics
        per_sample_cos = F.cosine_similarity(reconstructed, vectors, dim=-1)
        hdc_mse = F.mse_loss(reconstructed, vectors).item()
        hdc_cos_sim = per_sample_cos.mean().item()

        # Decode to molecules
        reconstructed = reconstructed.cpu()
        decode_device = self.hypernet.nodes_codebook.device
        vsa_cls = self.hypernet.vsa.tensor_class

        decoder_settings = FallbackDecoderSettings(
            beam_size=8,
            limit=1024,
            top_k=1,
        )

        n_valid = 0
        n_exact = 0
        tanimoto_sims: list[float] = []

        # Collect pairs for the molecule grid: (orig_mol, recon_mol, cos_sim, is_exact)
        mol_pairs: list[tuple] = []

        for i in range(len(subset)):
            d = subset[i]
            if self.training_target == "edge":
                edge_term = reconstructed[i].to(decode_device).as_subclass(vsa_cls)
                graph_term = d.graph_terms.view(-1).to(decode_device).as_subclass(vsa_cls)
            elif self.training_target == "graph":
                edge_term = d.node_terms.view(-1).to(decode_device).as_subclass(vsa_cls)
                graph_term = reconstructed[i].to(decode_device).as_subclass(vsa_cls)
            else:
                edge_term = (
                    reconstructed[i, : self.hv_dim]
                    .to(decode_device)
                    .as_subclass(vsa_cls)
                )
                graph_term = (
                    reconstructed[i, self.hv_dim :]
                    .to(decode_device)
                    .as_subclass(vsa_cls)
                )

            orig_mol = Chem.MolFromSmiles(original_smiles[i])
            recon_mol = None
            is_exact = False
            cs = per_sample_cos[i].item()

            try:
                result = self.hypernet.decode_graph_greedy(
                    edge_term=edge_term,
                    graph_term=graph_term,
                    decoder_settings=decoder_settings,
                )
                if result.nx_graphs:
                    g = result.nx_graphs[0]
                    recon_mol = reconstruct_for_eval(g, dataset=self.dataset)

                    if recon_mol is not None and orig_mol is not None:
                        n_valid += 1

                        # Compare without stereochemistry — HDC encoding
                        # does not preserve E/Z or chirality information.
                        orig_nosmi = Chem.RWMol(orig_mol)
                        Chem.RemoveStereochemistry(orig_nosmi)
                        recon_nosmi = Chem.RWMol(recon_mol)
                        Chem.RemoveStereochemistry(recon_nosmi)
                        orig_smi = Chem.MolToSmiles(orig_nosmi, canonical=True)
                        recon_smi = Chem.MolToSmiles(recon_nosmi, canonical=True)

                        if recon_smi == orig_smi:
                            n_exact += 1
                            is_exact = True

                        fp_orig = AllChem.GetMorganFingerprintAsBitVect(
                            orig_mol, 2, nBits=2048,
                        )
                        fp_recon = AllChem.GetMorganFingerprintAsBitVect(
                            recon_mol, 2, nBits=2048,
                        )
                        tanimoto_sims.append(
                            DataStructs.TanimotoSimilarity(fp_orig, fp_recon)
                        )
            except Exception:
                pass

            mol_pairs.append((orig_mol, recon_mol, cs, is_exact))

        n_total = len(subset)
        validity = 100.0 * n_valid / max(1, n_total)
        exact_match = 100.0 * n_exact / max(1, n_total)
        mean_tanimoto = (
            sum(tanimoto_sims) / len(tanimoto_sims) if tanimoto_sims else 0.0
        )

        self.experiment.track("recon_validity", validity)
        self.experiment.track("recon_exact_match", exact_match)
        self.experiment.track("recon_tanimoto", mean_tanimoto)
        self.experiment.track("recon_hdc_cos_sim", hdc_cos_sim)
        self.experiment.track("recon_hdc_mse", hdc_mse)

        # Feed into tracking callback for the 4x4 plot grid
        self.tracking_callback.record_recon_metrics(
            epoch=epoch,
            validity=validity,
            exact_match=exact_match,
            tanimoto=mean_tanimoto,
            hdc_cos_sim=hdc_cos_sim,
        )

        # ── Molecule comparison grid (original vs reconstructed) ──
        # Show up to 10 pairs as 2 columns (original | reconstructed)
        n_grid = min(10, len(mol_pairs))
        if n_grid > 0:
            fig, axes = plt.subplots(n_grid, 2, figsize=(8, 3 * n_grid))
            if n_grid == 1:
                axes = axes.reshape(1, 2)

            fig.suptitle(
                f"Epoch {epoch} — Validity: {validity:.1f}%, "
                f"Exact: {exact_match:.1f}%, Tanimoto: {mean_tanimoto:.3f}",
                fontsize=12, fontweight="bold",
            )

            for idx in range(n_grid):
                orig_mol, recon_mol, cs, is_exact = mol_pairs[idx]
                ax_orig = axes[idx, 0]
                ax_recon = axes[idx, 1]

                # Original molecule
                ax_orig.set_xticks([])
                ax_orig.set_yticks([])
                if orig_mol is not None:
                    try:
                        AllChem.Compute2DCoords(orig_mol)
                        img = Draw.MolToImage(orig_mol, size=(250, 250))
                        ax_orig.imshow(img)
                        smi = Chem.MolToSmiles(orig_mol, canonical=True)
                        ax_orig.set_title(
                            f"Original: {smi[:35]}", fontsize=7, color="black",
                        )
                    except Exception:
                        ax_orig.text(
                            0.5, 0.5, "Draw failed", ha="center", va="center",
                            fontsize=10, color="orange", transform=ax_orig.transAxes,
                        )
                else:
                    ax_orig.text(
                        0.5, 0.5, "No original", ha="center", va="center",
                        fontsize=10, color="gray", transform=ax_orig.transAxes,
                    )

                # Reconstructed molecule
                ax_recon.set_xticks([])
                ax_recon.set_yticks([])
                if recon_mol is not None:
                    try:
                        AllChem.Compute2DCoords(recon_mol)
                        img = Draw.MolToImage(recon_mol, size=(250, 250))
                        ax_recon.imshow(img)
                        smi = Chem.MolToSmiles(recon_mol, canonical=True)
                        color = "green" if is_exact else "blue"
                        label = "EXACT" if is_exact else f"cos={cs:.3f}"
                        ax_recon.set_title(
                            f"Recon ({label}): {smi[:35]}",
                            fontsize=7, color=color,
                        )
                    except Exception:
                        ax_recon.text(
                            0.5, 0.5, "Draw failed", ha="center", va="center",
                            fontsize=10, color="orange", transform=ax_recon.transAxes,
                        )
                        ax_recon.set_facecolor("#fff3e0")
                else:
                    ax_recon.plot(
                        [0, 1], [0, 1], "r-", lw=3, transform=ax_recon.transAxes,
                    )
                    ax_recon.plot(
                        [0, 1], [1, 0], "r-", lw=3, transform=ax_recon.transAxes,
                    )
                    ax_recon.set_title(
                        f"Failed (cos={cs:.3f})", fontsize=7, color="red",
                    )
                    ax_recon.set_facecolor("#ffebee")

                if idx == 0:
                    ax_orig.set_ylabel("Original", fontsize=9, fontweight="bold")
                    ax_recon.set_ylabel("Reconstructed", fontsize=9, fontweight="bold")

            plt.tight_layout()
            self.experiment.track("recon_molecule_grid", fig)
            plt.close(fig)

        elapsed = time.time() - t0
        self.experiment.log(
            f"[ReconEval] Epoch {epoch}: validity={validity:.1f}%, "
            f"exact_match={exact_match:.1f}%, tanimoto={mean_tanimoto:.3f}, "
            f"cos_sim={hdc_cos_sim:.4f}, mse={hdc_mse:.6f} ({elapsed:.1f}s)"
        )


# =============================================================================
# EXPERIMENT
# =============================================================================


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:

    # Ensure temp dir is inside the experiment folder
    custom_tmpdir = Path(e.path) / "tmp"
    custom_tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(custom_tmpdir)
    tempfile.tempdir = str(custom_tmpdir)

    pl.seed_everything(e.SEED)

    mode = "VAE" if e.VARIATIONAL else "AE"
    e.log("=" * 60)
    e.log(f"Autoencoder Training ({mode})")
    e.log("=" * 60)
    e.log(f"Dataset: {e.DATASET}")
    e.log(f"Encoder: {e.ENCODER_PATH or '(auto-created for testing)'}")
    e.log(f"Architecture: encoder={e.ENCODER_HIDDEN_DIMS}, latent={e.LATENT_DIM}")
    e.log(f"Variational: {e.VARIATIONAL}")
    if e.VARIATIONAL:
        e.log(f"  KL weight: {e.KL_WEIGHT}, warmup: {e.KL_WARMUP_EPOCHS} epochs")
    e.log(f"Loss weights: mse={e.MSE_WEIGHT}, cosine={e.COSINE_WEIGHT}")
    # In testing mode, reduce epochs and eval frequency for speed
    epochs = e.EPOCHS
    recon_eval_every = e.RECON_EVAL_EVERY_N_EPOCHS
    if e.__TESTING__:
        epochs = min(e.EPOCHS, 12)
        recon_eval_every = max(recon_eval_every, epochs)  # Only at last epoch

    e.log(f"Training: {epochs} epochs, batch size {e.BATCH_SIZE}")
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
    if e.TRAINING_TARGET == "both":
        data_dim = 2 * hv_dim
    else:
        data_dim = hv_dim
    e.log(f"HyperNet: hv_dim={hv_dim}, data_dim={data_dim}, target={e.TRAINING_TARGET}")

    e["config/hv_dim"] = hv_dim
    e["config/data_dim"] = data_dim

    # =========================================================================
    # Load and encode data
    # =========================================================================

    train_encoded, valid_encoded = e.apply_hook(
        "load_and_encode_data",
        hypernet=hypernet,
        device=hdc_device,
    )

    train_loader, valid_loader = e.apply_hook(
        "create_data_loaders",
        train_encoded=train_encoded,
        valid_encoded=valid_encoded,
    )

    # =========================================================================
    # Create model
    # =========================================================================

    model = HDCAutoencoder(
        data_dim=data_dim,
        latent_dim=e.LATENT_DIM,
        encoder_hidden_dims=e.ENCODER_HIDDEN_DIMS,
        decoder_hidden_dims=e.DECODER_HIDDEN_DIMS,
        resblock_depth=e.RESBLOCK_DEPTH,
        training_target=e.TRAINING_TARGET,
        recon_loss_type=e.RECON_LOSS_TYPE,
        variational=e.VARIATIONAL,
        mse_weight=e.MSE_WEIGHT,
        cosine_weight=e.COSINE_WEIGHT,
        kl_weight=e.KL_WEIGHT,
        kl_warmup_epochs=e.KL_WARMUP_EPOCHS,
        loss_clamp=e.LOSS_CLAMP,
        lr=e.LEARNING_RATE,
        weight_decay=e.WEIGHT_DECAY,
        warmup_epochs=e.WARMUP_EPOCHS,
    )

    num_params = sum(p.numel() for p in model.parameters())
    e.log(f"Model parameters: {num_params:,}")
    e["config/num_parameters"] = num_params
    e["config/latent_dim"] = e.LATENT_DIM
    e["config/variational"] = e.VARIATIONAL

    # =========================================================================
    # Prepare callbacks
    # =========================================================================

    # Validation vectors for scatter plot (subsample for speed)
    val_vectors = torch.stack([
        _extract_vector_from_data(d, e.TRAINING_TARGET)
        for d in valid_encoded[: min(500, len(valid_encoded))]
    ])

    tracking_callback = AutoencoderTrackingCallback(
        experiment=e,
        variational=e.VARIATIONAL,
        val_vectors=val_vectors,
        scatter_every_n_epochs=e.SCATTER_EVERY_N_EPOCHS,
        hv_dim=hv_dim if e.TRAINING_TARGET == "both" else None,
    )

    recon_callback = ReconstructionEvalCallback(
        experiment=e,
        hypernet=hypernet,
        eval_data=valid_encoded,
        tracking_callback=tracking_callback,
        dataset=e.DATASET,
        hv_dim=hv_dim,
        training_target=e.TRAINING_TARGET,
        every_n_epochs=recon_eval_every,
        n_eval=e.RECON_EVAL_N_SAMPLES,
    )

    callbacks = [
        tracking_callback,
        recon_callback,
        ModelCheckpoint(
            dirpath=e.path,
            filename="best-{epoch:03d}-{val/loss:.6f}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if e.USE_EMA:
        callbacks.append(EMAWeightAveraging(decay=e.EMA_DECAY))

    # Allow child experiments to add/modify callbacks
    callbacks = e.apply_hook(
        "modify_callbacks",
        callbacks=callbacks,
        train_loader=train_loader,
    )

    # =========================================================================
    # Train
    # =========================================================================

    logger = CSVLogger(e.path, name="logs")
    trainer = Trainer(
        max_epochs=epochs,
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
    # Final evaluation
    # =========================================================================

    e.log("\nFinal reconstruction evaluation...")
    model.eval()
    model.to(device)

    # HDC-space reconstruction on full validation set
    all_vectors = torch.stack([
        _extract_vector_from_data(d, e.TRAINING_TARGET) for d in valid_encoded
    ]).to(device)

    with torch.no_grad():
        all_recon = model.reconstruct(all_vectors)

    final_mse = F.mse_loss(all_recon, all_vectors).item()
    final_cos_sim = F.cosine_similarity(all_recon, all_vectors, dim=-1).mean().item()

    e.log(f"Final validation MSE: {final_mse:.6f}")
    e.log(f"Final validation Cos Sim: {final_cos_sim:.4f}")
    e["results/final_val_mse"] = final_mse
    e["results/final_val_cos_sim"] = final_cos_sim

    # Save final model
    model_path = Path(e.path) / "autoencoder.ckpt"
    model.save(model_path)
    e.log(f"Saved autoencoder to: {model_path}")
    e["results/model_path"] = str(model_path)

    e.log("\n" + "=" * 60)
    e.log("Experiment completed!")
    e.log("=" * 60)


# =============================================================================
# HOOKS
# =============================================================================


@experiment.hook("load_encoder", default=True)
def load_encoder(e: Experiment, device: torch.device):
    """Load HyperNet encoder and prune edge codebook."""
    if e.ENCODER_PATH and Path(e.ENCODER_PATH).exists():
        hypernet = load_hypernet(e.ENCODER_PATH, device=str(device))
    elif e.__TESTING__:
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

    # Prune edges codebook to observed edge pairs only
    needs_rw = hasattr(hypernet, "rw_config") and hypernet.rw_config.enabled
    if needs_rw:
        from graph_hdc.datasets.utils import scan_features_with_rw

        e.log("Scanning dataset for observed edge pairs (with RW augmentation)...")
        _, observed_edges = scan_features_with_rw(e.DATASET.lower(), hypernet.rw_config)
    else:
        from graph_hdc.datasets.utils import get_dataset_info

        e.log("Loading observed edge pairs from dataset info...")
        observed_edges = get_dataset_info(e.DATASET.lower()).edge_features
    hypernet.limit_edges_codebook(observed_edges)
    e.log(f"Pruned edges codebook: {hypernet.edges_codebook.shape[0]} entries")

    hypernet.eval()
    return hypernet


@experiment.hook("load_and_encode_data", default=True)
def load_and_encode_data(
    e: Experiment,
    hypernet: HyperNet,
    device: torch.device,
):
    """Load dataset, encode, then swap edge_terms into node_terms slot.

    After encoding, ``node_terms`` is overwritten with ``edge_terms`` so
    that ``_extract_vectors`` returns ``[edge_terms | graph_terms]``.
    """
    e.log("Loading dataset...")
    train_ds = get_split("train", dataset=e.DATASET.lower())
    valid_ds = get_split("valid", dataset=e.DATASET.lower())

    if e.__TESTING__:
        train_ds = train_ds[:64]
        valid_ds = valid_ds[:16]
    elif e.NUM_SUBSAMPLE is not None:
        n = e.NUM_SUBSAMPLE
        n_val = max(1, n // 5)
        train_ds = train_ds[:n]
        valid_ds = valid_ds[:n_val]

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

    # Swap edge_terms -> node_terms
    e.log("Swapping edge_terms into node_terms slot for [edge_terms | graph_terms] training")
    for d in train_encoded + valid_encoded:
        d.node_terms = d.edge_terms

    return train_encoded, valid_encoded


@experiment.hook("create_data_loaders", default=True)
def create_data_loaders(e: Experiment, train_encoded, valid_encoded):
    """Create DataLoaders from encoded data.

    Override this hook for streaming or custom data loading strategies.
    """
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
    return train_loader, valid_loader


@experiment.hook("modify_callbacks", default=True)
def modify_callbacks(e: Experiment, callbacks, train_loader):
    """Hook for child experiments to add or modify callbacks.

    Override this to inject streaming cleanup callbacks, etc.
    """
    return callbacks


experiment.run_if_main()
