#!/usr/bin/env python
"""
Train a Real NVP density estimator on HDC vectors.

Learns the joint distribution of ``[node_terms | graph_terms]`` encodings from
a molecular dataset.  The trained model provides exact log-likelihood scores
via a single forward pass, which can be used to rank candidate HDC vectors
during generation (higher log-prob = more plausible molecule).

Key features:
  * Per-term semantic masking (coupling layers alternate between node_terms
    and graph_terms halves).
  * Dequantization noise smooths the discrete HDC lattice during training.
  * Standalone ``save()`` / ``load()`` for easy checkpoint management.

Usage:
    # Quick test with auto-created encoder
    python train_density_nvp.py --__TESTING__ True

    # Full training on ZINC
    python train_density_nvp.py --__DEBUG__ False \\
        --ENCODER_PATH /path/to/encoder.ckpt
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.loader import DataLoader

from graph_hdc.datasets.utils import get_split, post_compute_encodings
from graph_hdc.hypernet import load_hypernet
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.hypernet.multi_hypernet import MultiHyperNet
from graph_hdc.models.flows.density_nvp import DensityConfig, DensityNVP
from graph_hdc.utils.experiment_helpers import GracefulInterruptHandler





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
#     Path to a saved HyperNet / MultiHyperNet encoder checkpoint (.ckpt).
#     Required for real runs. Leave empty + ``__TESTING__=True`` for smoke tests.
ENCODER_PATH: str = "/media/ssd2/Programming/graph_hdc_public/experiments/decoding/results/train_flow_edge_decoder_streaming/GOOD/hypernet_encoder.ckpt"

# :param DEVICE:
#     Device for model training. Options: "auto", "cpu", "cuda".
DEVICE: str = "cuda"

# :param HDC_DEVICE:
#     Device for the HyperNet encoder during the encoding step.
HDC_DEVICE: str = "cpu"

# :param ENCODER_BATCH_SIZE:
#     Batch size for HDC encoding.
ENCODER_BATCH_SIZE: int = 256

# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------

# :param NUM_FLOWS:
#     Number of Real NVP coupling layers.
NUM_FLOWS: int = 14

# :param HIDDEN_DIM:
#     Hidden dimension inside each coupling MLP.
HIDDEN_DIM: int = 1024

# :param NUM_HIDDEN_LAYERS:
#     Number of hidden layers per coupling MLP.
NUM_HIDDEN_LAYERS: int = 3

# :param SMAX:
#     Maximum scale magnitude (tanh * smax) at full warmup.
SMAX: float = 6.0

# :param SMAX_INITIAL:
#     Starting smax value during warmup.
SMAX_INITIAL: float = 2.0

# :param SMAX_WARMUP_EPOCHS:
#     Number of epochs to linearly ramp smax from SMAX_INITIAL to SMAX.
SMAX_WARMUP_EPOCHS: int = 15

# :param USE_ACT_NORM:
#     Whether to prepend an ActNorm layer.
USE_ACT_NORM: bool = True

# :param DEQUANT_SIGMA:
#     Dequantization noise std added during training only.
#     Smooths the discrete HDC lattice. 0.0 disables it.
DEQUANT_SIGMA: float = 0.05

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------

# :param EPOCHS:
#     Number of training epochs.
EPOCHS: int = 500

# :param BATCH_SIZE:
#     Training batch size.
BATCH_SIZE: int = 128

# :param LEARNING_RATE:
#     AdamW learning rate.
LEARNING_RATE: float = 2e-4

# :param WEIGHT_DECAY:
#     AdamW weight decay.
WEIGHT_DECAY: float = 0e-4

# :param GRADIENT_CLIP_VAL:
#     Gradient clipping value.
GRADIENT_CLIP_VAL: float = 1.0

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------

# :param NUM_SUBSAMPLE:
#     Optional: only use this many training samples (+ 20% for validation).
#     None = full dataset.
NUM_SUBSAMPLE: Optional[int] = None

# :param SEED:
#     Random seed.
SEED: int = 42

# :param PRECISION:
#     Training precision.
PRECISION: str = "32"

# :param __DEBUG__:
#     Debug mode — reuses same output folder during development.
__DEBUG__: bool = True

# :param __TESTING__:
#     Testing mode — runs with minimal iterations for validation.
__TESTING__: bool = False

# =============================================================================
# CALLBACK
# =============================================================================


class DensityNVPMetricsCallback(Callback):
    """Lightweight training metrics callback for DensityNVP.

    Tracks per-epoch metrics and produces a 2x4 summary figure at each
    validation epoch end:

        Row 1: Train/Val NLL | NLL (log scale) | Bits per dim | Scale magnitude
        Row 2: Learning rate | Gradient norm   | Param delta  | Weight norms
    """

    def __init__(self, experiment: Experiment, flat_dim: int):
        super().__init__()
        self.experiment = experiment
        self.flat_dim = flat_dim
        self.log2 = math.log(2)

        # Per-epoch history
        self.train_nll: list[float] = []
        self.val_nll: list[float] = []
        self.bpd: list[float] = []
        self.s_pre_absmax: list[float] = []
        self.lr: list[float] = []
        self.grad_norm: list[float] = []
        self.param_delta: list[float] = []
        self.weight_norm: list[float] = []

        # Snapshot of parameters at epoch start for delta computation
        self._param_snapshot: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_metric(self, trainer, key: str) -> Optional[float]:
        val = trainer.callback_metrics.get(key)
        if val is not None:
            return float(val)
        return None

    @staticmethod
    def _smooth(values: list[float], alpha: float = 0.3) -> list[float]:
        """Exponential moving average for smoothed curves."""
        if not values:
            return []
        s = [values[0]]
        for v in values[1:]:
            s.append(alpha * v + (1 - alpha) * s[-1])
        return s

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_train_epoch_start(self, trainer, pl_module):
        # Snapshot parameters for delta computation
        self._param_snapshot = {
            n: p.detach().clone()
            for n, p in pl_module.named_parameters()
            if p.requires_grad
        }

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Compute global gradient L2 norm
        total_norm_sq = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.detach().norm(2).item() ** 2
        self._last_grad_norm = math.sqrt(total_norm_sq)

    def on_train_epoch_end(self, trainer, pl_module):
        # Train NLL
        train_loss = self._get_metric(trainer, "train_loss_epoch")
        if train_loss is not None:
            self.train_nll.append(train_loss)
            self.experiment.track("nll_train", train_loss)

        # Scale magnitude
        s_max = self._get_metric(trainer, "s_pre_absmax")
        if s_max is not None:
            self.s_pre_absmax.append(s_max)
            self.experiment.track("s_pre_absmax", s_max)

        # Learning rate
        lr = None
        for opt in trainer.optimizers:
            for pg in opt.param_groups:
                lr = pg["lr"]
                break
        if lr is not None:
            self.lr.append(lr)
            self.experiment.track("learning_rate", lr)

        # Gradient norm (last batch of epoch)
        gn = getattr(self, "_last_grad_norm", None)
        if gn is not None:
            self.grad_norm.append(gn)
            self.experiment.track("grad_norm", gn)

        # Parameter delta
        delta_sq = 0.0
        total_norm_sq = 0.0
        for n, p in pl_module.named_parameters():
            if p.requires_grad:
                total_norm_sq += p.detach().norm(2).item() ** 2
                if n in self._param_snapshot:
                    delta_sq += (p.detach() - self._param_snapshot[n].to(p.device)).norm(2).item() ** 2
        self.param_delta.append(math.sqrt(delta_sq))
        self.weight_norm.append(math.sqrt(total_norm_sq))
        self.experiment.track("param_delta", self.param_delta[-1])
        self.experiment.track("weight_norm", self.weight_norm[-1])

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = self._get_metric(trainer, "val_loss")
        if val_loss is not None:
            self.val_nll.append(val_loss)
            self.bpd.append(val_loss / (self.flat_dim * self.log2))
            self.experiment.track("nll_val", val_loss)
            self.experiment.track("bpd_val", self.bpd[-1])

        # Generate figure (skip very first sanity-check validation)
        if len(self.val_nll) >= 2:
            self._plot(trainer)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot(self, trainer):
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        fig.suptitle(
            f"DensityNVP Training — Epoch {trainer.current_epoch}",
            fontsize=14, fontweight="bold",
        )

        epochs_t = list(range(1, len(self.train_nll) + 1))
        epochs_v = list(range(1, len(self.val_nll) + 1))

        # ---- Row 1, Col 1: Train/Val NLL ----
        ax = axes[0, 0]
        if self.train_nll:
            ax.plot(epochs_t, self.train_nll, "C0", alpha=0.3, linewidth=0.8)
            ax.plot(epochs_t, self._smooth(self.train_nll), "C0", label="train (smooth)")
        if self.val_nll:
            ax.plot(epochs_v, self.val_nll, "C1", alpha=0.3, linewidth=0.8)
            ax.plot(epochs_v, self._smooth(self.val_nll), "C1", label="val (smooth)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("NLL")
        ax.set_title("Train / Val NLL")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- Row 1, Col 2: NLL (log scale) ----
        ax = axes[0, 1]
        if self.train_nll:
            pos_train = [(e, v) for e, v in zip(epochs_t, self.train_nll) if v > 0]
            if pos_train:
                ax.semilogy([e for e, _ in pos_train], [v for _, v in pos_train], "C0", alpha=0.5, label="train")
        if self.val_nll:
            pos_val = [(e, v) for e, v in zip(epochs_v, self.val_nll) if v > 0]
            if pos_val:
                ax.semilogy([e for e, _ in pos_val], [v for _, v in pos_val], "C1", alpha=0.5, label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("NLL (log)")
        ax.set_title("NLL (log scale)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- Row 1, Col 3: Bits per dimension ----
        ax = axes[0, 2]
        if self.bpd:
            ax.plot(epochs_v, self.bpd, "C2")
            ax.plot(epochs_v, self._smooth(self.bpd), "C2", linestyle="--", alpha=0.7)
            if len(self.bpd) > 0:
                ax.axhline(min(self.bpd), color="gray", linestyle=":", alpha=0.5)
                ax.annotate(
                    f"best: {min(self.bpd):.3f}",
                    xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, color="gray",
                )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BPD")
        ax.set_title("Bits per Dimension (val)")
        ax.grid(True, alpha=0.3)

        # ---- Row 1, Col 4: Scale magnitude ----
        ax = axes[0, 3]
        if self.s_pre_absmax:
            ep_s = list(range(1, len(self.s_pre_absmax) + 1))
            ax.plot(ep_s, self.s_pre_absmax, "C3")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("|s| max")
        ax.set_title("Scale Pre-activation |s|max")
        ax.grid(True, alpha=0.3)

        # ---- Row 2, Col 1: Learning rate ----
        ax = axes[1, 0]
        if self.lr:
            ep_lr = list(range(1, len(self.lr) + 1))
            ax.plot(ep_lr, self.lr, "C4")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_title("Learning Rate")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.grid(True, alpha=0.3)

        # ---- Row 2, Col 2: Gradient norm ----
        ax = axes[1, 1]
        if self.grad_norm:
            ep_g = list(range(1, len(self.grad_norm) + 1))
            ax.plot(ep_g, self.grad_norm, "C5", alpha=0.4, linewidth=0.8)
            ax.plot(ep_g, self._smooth(self.grad_norm), "C5")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("||grad||")
        ax.set_title("Gradient L2 Norm")
        ax.grid(True, alpha=0.3)

        # ---- Row 2, Col 3: Parameter delta ----
        ax = axes[1, 2]
        if self.param_delta:
            ep_d = list(range(1, len(self.param_delta) + 1))
            ax.plot(ep_d, self.param_delta, "C6")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("||delta theta||")
        ax.set_title("Parameter Change / Epoch")
        ax.grid(True, alpha=0.3)

        # ---- Row 2, Col 4: Weight norms ----
        ax = axes[1, 3]
        if self.weight_norm:
            ep_w = list(range(1, len(self.weight_norm) + 1))
            ax.plot(ep_w, self.weight_norm, "C7")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("||theta||")
        ax.set_title("Total Weight Norm")
        ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.experiment.commit_fig("training_metrics.png", fig)
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
    """Train Real NVP density estimator on HDC vectors."""

    custom_tmpdir = Path(e.path) / ".tmp_checkpoints"
    custom_tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(custom_tmpdir)
    tempfile.tempdir = str(custom_tmpdir)

    pl.seed_everything(e.SEED)

    e.log("=" * 60)
    e.log("Density NVP Training")
    e.log("=" * 60)
    e.log(f"Dataset: {e.DATASET}")
    e.log(f"Encoder: {e.ENCODER_PATH or '(auto-created for testing)'}")
    e.log(f"Architecture: {e.NUM_FLOWS} coupling layers, "
          f"{e.HIDDEN_DIM} hidden dim, {e.NUM_HIDDEN_LAYERS} hidden layers")
    if e.DEQUANT_SIGMA > 0:
        e.log(f"Dequantization: sigma={e.DEQUANT_SIGMA}")
    e.log(f"Training: {e.EPOCHS} epochs, batch size {e.BATCH_SIZE}")
    e.log("=" * 60)

    # ----- Devices -----

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
    # Step 1: Load encoder
    # =========================================================================

    hypernet = e.apply_hook("load_encoder", device=hdc_device)

    hv_dim = hypernet.hv_dim
    if isinstance(hypernet, MultiHyperNet):
        graph_dim = hypernet.ensemble_graph_dim
        data_dim = hv_dim + graph_dim
        e.log(
            f"MultiHyperNet: K={hypernet.num_hypernets}, "
            f"hv_dim={hv_dim}, ensemble_graph_dim={graph_dim}, "
            f"data_dim={data_dim}"
        )
        # For MultiHyperNet, hv_dim passed to DensityConfig is half the flat_dim
        config_hv_dim = data_dim // 2
    else:
        data_dim = 2 * hv_dim
        config_hv_dim = hv_dim
        e.log(f"HyperNet: hv_dim={hv_dim}, depth={hypernet.depth}, data_dim={data_dim}")

    e["config/hv_dim"] = hv_dim
    e["config/data_dim"] = data_dim

    # =========================================================================
    # Step 2: Load and encode data
    # =========================================================================

    train_encoded, valid_encoded = e.apply_hook(
        "load_and_encode_data",
        hypernet=hypernet,
        device=hdc_device,
    )

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
    # Step 3: Create model
    # =========================================================================

    cfg = DensityConfig(
        hv_dim=config_hv_dim,
        num_flows=e.NUM_FLOWS,
        hidden_dim=e.HIDDEN_DIM,
        num_hidden_layers=e.NUM_HIDDEN_LAYERS,
        smax=e.SMAX,
        smax_initial=e.SMAX_INITIAL,
        smax_warmup_epochs=e.SMAX_WARMUP_EPOCHS,
        use_act_norm=e.USE_ACT_NORM,
        dequant_sigma=e.DEQUANT_SIGMA,
        lr=e.LEARNING_RATE,
        weight_decay=e.WEIGHT_DECAY,
        seed=e.SEED,
    )
    model = DensityNVP(cfg)

    num_params = sum(p.numel() for p in model.parameters())
    e.log(f"Model parameters: {num_params:,}")
    e["config/num_parameters"] = num_params

    # =========================================================================
    # Step 4: Fit standardization
    # =========================================================================

    model = e.apply_hook(
        "fit_standardization",
        model=model,
        train_encoded=train_encoded,
    )

    # =========================================================================
    # Step 5: Train
    # =========================================================================

    callbacks = [
        ModelCheckpoint(
            dirpath=e.path,
            filename="best-{epoch:03d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        DensityNVPMetricsCallback(experiment=e, flat_dim=model.flat_dim),
    ]

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
        e.log("Training stopped gracefully (CTRL+C or early stopping)")
        e.log("-" * 40)

    # =========================================================================
    # Step 6: Save standalone checkpoint + evaluate
    # =========================================================================

    save_path = Path(e.path) / "density_nvp.pt"
    model.save(save_path)
    e.log(f"\nSaved standalone model to: {save_path}")

    # Verify load roundtrip
    loaded = DensityNVP.load(save_path, device="cpu")
    test_input = torch.randn(4, model.flat_dim)
    with torch.no_grad():
        test_lp = loaded.log_prob(test_input)
    assert torch.isfinite(test_lp).all(), "Load roundtrip produced non-finite log_prob!"
    e.log(f"Load roundtrip OK — test log_prob: {test_lp.tolist()}")

    # Compute final validation NLL
    e.log("\nComputing final validation NLL...")
    model.eval()
    model.to(device)
    all_nll = []
    with torch.no_grad():
        for batch in valid_loader:
            batch = batch.to(device)
            flat = model._flat_from_batch(batch)
            nll = model.nll(flat)
            nll = nll[torch.isfinite(nll)]
            if nll.numel() > 0:
                all_nll.append(nll.cpu())

    if all_nll:
        all_nll = torch.cat(all_nll)
        mean_nll = all_nll.mean().item()
        std_nll = all_nll.std().item()
        bpd = mean_nll / (model.flat_dim * math.log(2))
        e.log(f"Validation NLL: {mean_nll:.4f} +/- {std_nll:.4f}")
        e.log(f"Bits per dimension: {bpd:.4f}")
        e["results/val_nll_mean"] = mean_nll
        e["results/val_nll_std"] = std_nll
        e["results/val_bpd"] = bpd
    else:
        e.log("WARNING: No finite NLL values computed.")

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
):
    """Load dataset and compute HDC encodings."""
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

    return train_encoded, valid_encoded


@experiment.hook("fit_standardization", default=True)
def fit_standardization(e: Experiment, model: DensityNVP, train_encoded):
    """Compute per-term mean/std from training data and set on model."""
    D = model.D
    vectors = []
    for d in train_encoded:
        node = d.node_terms.view(-1)
        graph = d.graph_terms.view(-1)
        vectors.append(torch.cat([node, graph]).float())
    vectors = torch.stack(vectors)

    mean = vectors.mean(dim=0)
    std = vectors.std(dim=0).clamp(min=1e-8)
    model.set_standardization(mean, std)

    e.log(
        f"Standardization: "
        f"node mean [{mean[:D].min().item():.4f}, {mean[:D].max().item():.4f}], "
        f"node std [{std[:D].min().item():.4f}, {std[:D].max().item():.4f}], "
        f"graph mean [{mean[D:].min().item():.4f}, {mean[D:].max().item():.4f}], "
        f"graph std [{std[D:].min().item():.4f}, {std[D:].max().item():.4f}]"
    )

    return model


# =============================================================================
# TESTING
# =============================================================================


@experiment.testing
def testing(e: Experiment) -> None:
    """Quick test mode with reduced parameters."""
    e.NUM_FLOWS = 2
    e.HIDDEN_DIM = 64
    e.NUM_HIDDEN_LAYERS = 1
    e.EPOCHS = 3
    e.BATCH_SIZE = 32
    e.DATASET = "zinc"


# =============================================================================
# ENTRY POINT
# =============================================================================

experiment.run_if_main()
