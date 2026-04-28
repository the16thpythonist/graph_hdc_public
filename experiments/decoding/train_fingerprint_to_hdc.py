#!/usr/bin/env python
"""
Train Fingerprint-to-HDC Translation Network.

Trains a SwiGLU residual MLP (TranslatorMLP) to map ECFP4 Morgan fingerprints
(2048-bit) to HDC hypervectors of the form ``[edge_terms | graph_embedding]``
(the same target format used by the autoencoder experiment).  After training,
evaluates end-to-end molecular reconstruction via the HyperNet's greedy
decoder:

    SMILES → ECFP4 → TranslatorMLP → predicted [edge_terms | graph_embedding]
           → hypernet.decode_graph_greedy → reconstruct_for_eval → RDKit mol
           → compare with original molecule.

This is the BASE EXPERIMENT.  Child experiments can inherit via
Experiment.extend() and override hooks to swap the model architecture
(e.g. Transformer), data loading strategy, or evaluation procedure.

Usage:
    # Quick smoke test
    python train_fingerprint_to_hdc.py --__TESTING__ True

    # Full training (uses data/qm9_smiles.csv by default)
    python train_fingerprint_to_hdc.py \\
        --HYPERNET_PATH /path/to/encoder.ckpt
"""

from __future__ import annotations

import csv
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch.utils.data import DataLoader, TensorDataset

from graph_hdc.datasets.zinc_smiles import mol_to_data as mol_to_zinc_data
from graph_hdc.hypernet import load_hypernet
from graph_hdc.hypernet.configs import FallbackDecoderSettings
from graph_hdc.hypernet.multi_hypernet import MultiHyperNet
from graph_hdc.models.translator_mlp import TranslatorMLP
from graph_hdc.utils.chem import reconstruct_for_eval
from graph_hdc.utils.experiment_helpers import (
    GracefulInterruptHandler,
    compute_tanimoto_similarity,
    create_reconstruction_plot,
    get_canonical_smiles,
    is_valid_mol,
)

# =============================================================================
# PARAMETERS
# =============================================================================

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

# :param CSV_PATH:
#     Path to a CSV file with a ``smiles`` column.  Each row is one molecule.
CSV_PATH: str = "data/qm9_smiles.csv"

# :param TRAIN_RATIO:
#     Fraction of molecules used for training.
TRAIN_RATIO: float = 0.8

# :param VAL_RATIO:
#     Fraction of molecules used for validation.
VAL_RATIO: float = 0.1

# :param TEST_RATIO:
#     Fraction of molecules used for testing / evaluation.
TEST_RATIO: float = 0.1

# -----------------------------------------------------------------------------
# HDC Encoder
# -----------------------------------------------------------------------------

# :param HYPERNET_PATH:
#     Path to a saved HyperNet / MultiHyperNet / RRWPHyperNet checkpoint.
#     This encoder is used both to compute the ground-truth HDC targets and
#     to greedily decode predicted HDC vectors back to molecular graphs.
#     Matches the encoder used in the autoencoder experiment.
HYPERNET_PATH: str = "/media/ssd2/Programming/_branch/graph_hdc_public/experiments/encoders/zinc_d1024_depth3_k6_10_14_b8.ckpt"

# :param DATASET:
#     Dataset key passed to ``reconstruct_for_eval`` during evaluation. Kept
#     as ``"zinc"`` even for QM9 CSVs because the loaded hypernet was trained
#     on ZINC and ``mol_to_zinc_data`` produces ZINC-schema node features.
DATASET: str = "zinc"

# -----------------------------------------------------------------------------
# Fingerprint
# -----------------------------------------------------------------------------

# :param FP_RADIUS:
#     Morgan fingerprint radius.  Radius 2 corresponds to ECFP4.
FP_RADIUS: int = 2

# :param FP_NBITS:
#     Number of bits in the Morgan fingerprint bit-vector.
FP_NBITS: int = 2048

# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------

# :param HIDDEN_DIMS:
#     Hidden-stage widths for the TranslatorMLP. Each entry produces a
#     ProjectionBlock: Linear(prev, h) followed by ``RESBLOCK_DEPTH`` SwiGLU
#     residual blocks at width h. A final plain Linear maps to the HDC dim.
HIDDEN_DIMS: tuple = (2048, 2048, 2048, 2048)

# :param RESBLOCK_DEPTH:
#     Number of SwiGLU residual blocks stacked inside each ProjectionBlock.
RESBLOCK_DEPTH: int = 2

# :param BERHU_C_FRACTION:
#     Fraction of the batch's max absolute residual used as the BerHu
#     threshold c (detached). Values < c are L1, values > c are L2-scaled.
BERHU_C_FRACTION: float = 0.1

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

# :param EPOCHS:
#     Number of training epochs.
EPOCHS: int = 500

# :param BATCH_SIZE:
#     Mini-batch size.
BATCH_SIZE: int = 64

# :param LEARNING_RATE:
#     Initial learning rate for AdamW.
LEARNING_RATE: float = 1e-4

# :param WEIGHT_DECAY:
#     Weight decay (L2 regularization).
WEIGHT_DECAY: float = 1e-4

# :param WARMUP_EPOCHS:
#     Number of epochs for linear LR warmup.  Set to 0 to disable warmup.
WARMUP_EPOCHS: int = 3

# :param USE_PROCRUSTES_INIT:
#     Whether to initialize the output layer via least-squares (Procrustes)
#     fit on the training data before training begins.
USE_PROCRUSTES_INIT: bool = False

# :param PROCRUSTES_NUM_SAMPLES:
#     Number of training samples used for the Procrustes fit.
#     0 = use all training samples.
PROCRUSTES_NUM_SAMPLES: int = 0

# :param GRADIENT_CLIP_VAL:
#     Gradient clipping value.  0.0 disables clipping.
GRADIENT_CLIP_VAL: float = 1.0

# -----------------------------------------------------------------------------
# Greedy Decoder (evaluation only)
# -----------------------------------------------------------------------------

# :param BEAM_SIZE:
#     Beam width for ``hypernet.decode_graph_greedy``. Larger values recover
#     more graphs but scale roughly linearly in decode cost.
BEAM_SIZE: int = 32

# :param LIMIT:
#     Population limit for the greedy beam search.
LIMIT: int = 2048

# :param TOP_K:
#     Top-k candidates considered at each greedy step.
TOP_K: int = 1

# -----------------------------------------------------------------------------
# Intermediate reconstruction eval (during training)
# -----------------------------------------------------------------------------

# :param RECON_EVAL_EVERY_N_EPOCHS:
#     Run an intermediate molecular-reconstruction eval every N validation
#     epochs. Set to 0 to disable.
RECON_EVAL_EVERY_N_EPOCHS: int = 10

# :param RECON_EVAL_N_SAMPLES:
#     Number of (test_data) molecules used for each intermediate eval.
RECON_EVAL_N_SAMPLES: int = 25

# :param RECON_EVAL_BEAM_SIZE:
#     Beam width for the *intermediate* greedy decode. Smaller than the final
#     eval's BEAM_SIZE to keep the periodic eval cheap.
RECON_EVAL_BEAM_SIZE: int = 8

# :param RECON_EVAL_LIMIT:
#     Population limit for the *intermediate* greedy decode.
RECON_EVAL_LIMIT: int = 1024

# -----------------------------------------------------------------------------
# Evaluation (end of training)
# -----------------------------------------------------------------------------

# :param NUM_TEST_SAMPLES:
#     Maximum number of test molecules to run through the final reconstruction
#     eval at the end of training.
NUM_TEST_SAMPLES: int = 100

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------

# :param SEED:
#     Random seed for reproducibility.
SEED: int = 42

# :param ACCELERATOR:
#     PyTorch Lightning accelerator.  Options: "auto", "gpu", "cpu".
ACCELERATOR: str = "auto"

# :param PRECISION:
#     Training precision.  Options: "32", "16", "bf16".
PRECISION: str = "32"

# :param NUM_WORKERS:
#     DataLoader workers.  Set to 0 to avoid multiprocessing issues.
NUM_WORKERS: int = 0

# :param __DEBUG__:
#     Debug mode — reuses the same output folder during development.
__DEBUG__: bool = True

# :param __TESTING__:
#     Testing mode — runs with minimal iterations for quick validation.
__TESTING__: bool = False


# =============================================================================
# TRAINING METRICS CALLBACK
# =============================================================================


class TrainingMetricsCallback(Callback):
    """
    Per-epoch training diagnostics for the BerHu fingerprint\u2192HDC translator.

    Produces a 2x5 grid:
        Row 1: train/val loss, per-sample val loss histogram, per-sample val
               cosine-sim histogram, mean\u00b1IQR cosine sim, mean\u00b1std L2.
        Row 2: learning rate, gradient norm, parameter delta, weight norm,
               train-val overfit gap.

    Per-sample val stats (loss, cosine, L2) are pulled directly off the
    LightningModule \u2014 TranslatorMLP stashes them during validation_step.
    """

    def __init__(self, experiment: Experiment):
        super().__init__()
        self.experiment = experiment

        # Per-epoch scalar curves.
        self.train_loss: list[float] = []
        self.val_loss: list[float] = []
        self.overfit_gap: list[float] = []
        self.lr: list[float] = []
        self.grad_norm: list[float] = []
        self.param_delta: list[float] = []
        self.weight_norm: list[float] = []

        # Per-epoch distribution summaries (train and val).
        self.cos_train_mean: list[float] = []
        self.cos_train_q25: list[float] = []
        self.cos_train_q75: list[float] = []
        self.cos_val_mean: list[float] = []
        self.cos_val_q25: list[float] = []
        self.cos_val_q75: list[float] = []
        self.l2_val_mean: list[float] = []
        self.l2_val_std: list[float] = []

        # Latest-epoch raw per-sample arrays for histograms.
        self._latest_val_loss: np.ndarray | None = None
        self._latest_val_cosine: np.ndarray | None = None

        # Internal state.
        self._param_snapshot: dict[str, torch.Tensor] = {}
        self._last_grad_norm: float | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_metric(trainer: Trainer, key: str) -> float | None:
        val = trainer.callback_metrics.get(key)
        return float(val) if val is not None else None

    @staticmethod
    def _smooth(values: list[float], alpha: float = 0.3) -> list[float]:
        if not values:
            return []
        s = [values[0]]
        for v in values[1:]:
            s.append(alpha * v + (1 - alpha) * s[-1])
        return s

    @staticmethod
    def _collect_tensor_list(tensors: list[torch.Tensor]) -> np.ndarray | None:
        if not tensors:
            return None
        return torch.cat(tensors, dim=0).numpy()

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_train_epoch_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        self._param_snapshot = {
            n: p.detach().clone()
            for n, p in pl_module.named_parameters()
            if p.requires_grad
        }

    def on_before_optimizer_step(self, trainer: Trainer, pl_module: pl.LightningModule, optimizer) -> None:
        total_norm_sq = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.detach().norm(2).item() ** 2
        self._last_grad_norm = math.sqrt(total_norm_sq)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        tl = self._get_metric(trainer, "train/loss")
        if tl is not None:
            self.train_loss.append(tl)
            self.experiment.track("loss_train", tl)

        tc = self._get_metric(trainer, "train/cosine_sim")
        if tc is not None:
            # Train-time per-sample cosine isn't stored (would cost memory);
            # use the epoch mean as both mean and IQR center.
            self.cos_train_mean.append(tc)
            self.cos_train_q25.append(tc)
            self.cos_train_q75.append(tc)
            self.experiment.track("cosine_sim_train", tc)

        # Learning rate.
        lr = None
        for opt in trainer.optimizers:
            for pg in opt.param_groups:
                lr = pg["lr"]
                break
        if lr is not None:
            self.lr.append(lr)
            self.experiment.track("learning_rate", lr)

        # Gradient norm.
        if self._last_grad_norm is not None:
            self.grad_norm.append(self._last_grad_norm)
            self.experiment.track("grad_norm", self._last_grad_norm)

        # Parameter delta + weight norm.
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

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        vl = self._get_metric(trainer, "val/loss")
        if vl is not None:
            self.val_loss.append(vl)
            self.experiment.track("loss_val", vl)

        # Per-sample val distributions (stashed by TranslatorMLP).
        losses = self._collect_tensor_list(getattr(pl_module, "_val_per_sample_loss", []))
        cosines = self._collect_tensor_list(getattr(pl_module, "_val_per_sample_cosine", []))
        l2s = self._collect_tensor_list(getattr(pl_module, "_val_per_sample_l2", []))

        self._latest_val_loss = losses
        self._latest_val_cosine = cosines

        if cosines is not None and cosines.size > 0:
            self.cos_val_mean.append(float(np.mean(cosines)))
            self.cos_val_q25.append(float(np.quantile(cosines, 0.25)))
            self.cos_val_q75.append(float(np.quantile(cosines, 0.75)))
            self.experiment.track("cosine_sim_val", self.cos_val_mean[-1])

        if l2s is not None and l2s.size > 0:
            self.l2_val_mean.append(float(np.mean(l2s)))
            self.l2_val_std.append(float(np.std(l2s)))
            self.experiment.track("l2_val_mean", self.l2_val_mean[-1])

        # Overfit gap.
        if self.train_loss and self.val_loss:
            gap = self.val_loss[-1] - self.train_loss[-1]
            self.overfit_gap.append(gap)
            self.experiment.track("overfit_gap", gap)

        if len(self.val_loss) >= 2:
            self._plot(trainer)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot(self, trainer: Trainer) -> None:
        fig, axes = plt.subplots(2, 5, figsize=(25, 8))
        fig.suptitle(
            f"FP\u2192HDC Training \u2014 Epoch {trainer.current_epoch}",
            fontsize=14, fontweight="bold",
        )

        # ---- Row 1, Col 1: Train/Val Loss (BerHu) ----
        ax = axes[0, 0]
        if self.train_loss:
            ep = list(range(1, len(self.train_loss) + 1))
            ax.plot(ep, self.train_loss, "C0", alpha=0.3, linewidth=0.8)
            ax.plot(ep, self._smooth(self.train_loss), "C0", label="train")
        if self.val_loss:
            ep = list(range(1, len(self.val_loss) + 1))
            ax.plot(ep, self.val_loss, "C1", alpha=0.3, linewidth=0.8)
            ax.plot(ep, self._smooth(self.val_loss), "C1", label="val")
            ax.axhline(min(self.val_loss), color="gray", linestyle=":", alpha=0.5)
            ax.annotate(
                f"best: {min(self.val_loss):.4f}",
                xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, color="gray",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BerHu Loss")
        ax.set_title("Train / Val Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- Row 1, Col 2: Per-sample Val Loss Histogram ----
        ax = axes[0, 1]
        if self._latest_val_loss is not None and self._latest_val_loss.size > 0:
            ax.hist(self._latest_val_loss, bins=40, color="C1", alpha=0.75)
            ax.axvline(float(np.mean(self._latest_val_loss)), color="black",
                       linestyle="--", alpha=0.7, label=f"mean {np.mean(self._latest_val_loss):.4f}")
            ax.axvline(float(np.median(self._latest_val_loss)), color="gray",
                       linestyle=":", alpha=0.7, label=f"median {np.median(self._latest_val_loss):.4f}")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "no val samples yet", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_xlabel("Per-sample BerHu loss")
        ax.set_ylabel("Count")
        ax.set_title("Val Loss Distribution (latest epoch)")
        ax.grid(True, alpha=0.3)

        # ---- Row 1, Col 3: Per-sample Val Cosine Similarity Histogram ----
        ax = axes[0, 2]
        if self._latest_val_cosine is not None and self._latest_val_cosine.size > 0:
            ax.hist(self._latest_val_cosine, bins=40, color="C2", alpha=0.75,
                    range=(-0.1, 1.0))
            ax.axvline(float(np.mean(self._latest_val_cosine)), color="black",
                       linestyle="--", alpha=0.7,
                       label=f"mean {np.mean(self._latest_val_cosine):.3f}")
            ax.axvline(float(np.median(self._latest_val_cosine)), color="gray",
                       linestyle=":", alpha=0.7,
                       label=f"median {np.median(self._latest_val_cosine):.3f}")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "no val samples yet", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_xlabel("cos(pred, target)")
        ax.set_ylabel("Count")
        ax.set_title("Val Cosine Sim Distribution (latest)")
        ax.grid(True, alpha=0.3)

        # ---- Row 1, Col 4: Cosine Sim Mean \u00b1 IQR over epochs ----
        ax = axes[0, 3]
        if self.cos_train_mean:
            ep = list(range(1, len(self.cos_train_mean) + 1))
            ax.plot(ep, self.cos_train_mean, "C0", label="train (mean)")
        if self.cos_val_mean:
            ep = list(range(1, len(self.cos_val_mean) + 1))
            ax.fill_between(ep, self.cos_val_q25, self.cos_val_q75,
                            color="C1", alpha=0.25, label="val IQR")
            ax.plot(ep, self.cos_val_mean, "C1", label="val (mean)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Cosine Sim \u2014 Mean / IQR")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- Row 1, Col 5: Val L2 distance mean \u00b1 std ----
        ax = axes[0, 4]
        if self.l2_val_mean:
            ep = list(range(1, len(self.l2_val_mean) + 1))
            lo = [m - s for m, s in zip(self.l2_val_mean, self.l2_val_std)]
            hi = [m + s for m, s in zip(self.l2_val_mean, self.l2_val_std)]
            ax.fill_between(ep, lo, hi, color="C3", alpha=0.25, label="\u00b1 std")
            ax.plot(ep, self.l2_val_mean, "C3", label="mean")
            ax.legend(fontsize=8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("||pred - target||_2")
        ax.set_title("Val L2 Distance")
        ax.grid(True, alpha=0.3)

        # ---- Row 2, Col 1: Learning Rate ----
        ax = axes[1, 0]
        if self.lr:
            ep = list(range(1, len(self.lr) + 1))
            ax.plot(ep, self.lr, "C4")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_title("Learning Rate")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.grid(True, alpha=0.3)

        # ---- Row 2, Col 2: Gradient L2 Norm ----
        ax = axes[1, 1]
        if self.grad_norm:
            ep = list(range(1, len(self.grad_norm) + 1))
            ax.plot(ep, self.grad_norm, "C5", alpha=0.4, linewidth=0.8)
            ax.plot(ep, self._smooth(self.grad_norm), "C5")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("||grad||")
        ax.set_title("Gradient L2 Norm")
        ax.grid(True, alpha=0.3)

        # ---- Row 2, Col 3: Parameter Change per Epoch ----
        ax = axes[1, 2]
        if self.param_delta:
            ep = list(range(1, len(self.param_delta) + 1))
            ax.plot(ep, self.param_delta, "C6")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("||delta theta||")
        ax.set_title("Parameter Change / Epoch")
        ax.grid(True, alpha=0.3)

        # ---- Row 2, Col 4: Total Weight Norm ----
        ax = axes[1, 3]
        if self.weight_norm:
            ep = list(range(1, len(self.weight_norm) + 1))
            ax.plot(ep, self.weight_norm, "C7")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("||theta||")
        ax.set_title("Total Weight Norm")
        ax.grid(True, alpha=0.3)

        # ---- Row 2, Col 5: Overfit Gap ----
        ax = axes[1, 4]
        if self.overfit_gap:
            ep = list(range(1, len(self.overfit_gap) + 1))
            ax.plot(ep, self.overfit_gap, "C3", alpha=0.4, linewidth=0.8)
            ax.plot(ep, self._smooth(self.overfit_gap), "C3")
            ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("val - train")
        ax.set_title("Overfitting Gap")
        ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.experiment.track("training_metrics", fig)
        plt.close(fig)


# =============================================================================
# RECONSTRUCTION EVAL CALLBACK
# =============================================================================


class ReconstructionEvalCallback(Callback):
    """
    Periodic molecular-reconstruction evaluation during training.

    Every ``every_n_epochs`` validation epochs, runs:
        fingerprint -> TranslatorMLP -> predicted [edge_terms | graph_embedding]
                    -> hypernet.decode_graph_greedy
                    -> reconstruct_for_eval -> RDKit mol
    on the first ``n_samples`` items of ``test_data``, and writes a
    side-by-side (original | reconstructed) grid figure with SMILES in titles.

    Tracks scalar metrics (validity, exact-match rate, mean Tanimoto, mean
    cos(pred, gt)) via ``experiment.track`` so they auto-plot under .track/.
    """

    def __init__(
        self,
        experiment: Experiment,
        hypernet,
        test_data: List[Dict[str, Any]],
        device: torch.device,
        actual_hdc_dim: int,
        dataset: str,
        n_samples: int = 50,
        every_n_epochs: int = 10,
        beam_size: int = 8,
        limit: int = 1024,
        top_k: int = 1,
        max_grid_pairs: int = 10,
    ):
        super().__init__()
        self.experiment = experiment
        self.hypernet = hypernet
        self.test_data = test_data
        self.device = device
        self.hv_dim = actual_hdc_dim
        self.dataset = dataset
        self.n_samples = n_samples
        self.every_n_epochs = every_n_epochs
        self.decoder_settings = FallbackDecoderSettings(
            beam_size=beam_size, limit=limit, top_k=top_k,
        )
        self.max_grid_pairs = max_grid_pairs

        self._decode_device = hypernet.nodes_codebook.device
        self._vsa_cls = hypernet.vsa.tensor_class

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking or self.every_n_epochs <= 0:
            return
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return
        self._evaluate(pl_module, epoch)

    @torch.no_grad()
    def _evaluate(self, model: pl.LightningModule, epoch: int) -> None:
        from rdkit.Chem import AllChem, Draw

        n = min(self.n_samples, len(self.test_data))
        if n == 0:
            return

        self.experiment.log(
            f"[ReconEval] Epoch {epoch}: decoding {n} molecules "
            f"(beam={self.decoder_settings.beam_size}, limit={self.decoder_settings.limit})..."
        )

        was_training = model.training
        model.eval()

        n_valid = 0
        n_exact = 0
        tanimoto_sims: list[float] = []
        cosines: list[float] = []
        mol_pairs: list[tuple] = []  # (orig_mol, recon_mol, cos_sim, is_exact, orig_smi, recon_smi)

        for idx in range(n):
            item = self.test_data[idx]
            original_smiles = item["smiles"]
            fp = item["fingerprint"].unsqueeze(0).to(self.device)
            gt_hdc = item["hdc_vector"].to(self.device)

            orig_mol = Chem.MolFromSmiles(original_smiles)

            pred_hdc = model(fp).squeeze(0)
            cos_sim = float(F.cosine_similarity(pred_hdc, gt_hdc, dim=-1))
            cosines.append(cos_sim)

            pred_cpu = pred_hdc.detach().cpu()
            edge_term = pred_cpu[:self.hv_dim].to(self._decode_device).as_subclass(self._vsa_cls)
            graph_term = pred_cpu[self.hv_dim:].to(self._decode_device).as_subclass(self._vsa_cls)

            recon_mol = None
            recon_smi = None
            try:
                result = self.hypernet.decode_graph_greedy(
                    edge_term=edge_term,
                    graph_term=graph_term,
                    decoder_settings=self.decoder_settings,
                )
                if result.nx_graphs:
                    recon_mol = reconstruct_for_eval(result.nx_graphs[0], dataset=self.dataset)
                    recon_smi = get_canonical_smiles(recon_mol)
            except Exception:
                pass

            is_valid = is_valid_mol(recon_mol)

            orig_canonical = None
            if orig_mol is not None:
                try:
                    orig_canonical = Chem.MolToSmiles(Chem.RemoveAllHs(orig_mol), canonical=True)
                except Exception:
                    orig_canonical = get_canonical_smiles(orig_mol)
            is_exact = (
                is_valid
                and recon_smi is not None
                and orig_canonical is not None
                and recon_smi == orig_canonical
            )

            tan = compute_tanimoto_similarity(orig_mol, recon_mol)
            if is_valid:
                n_valid += 1
            if is_exact:
                n_exact += 1
            tanimoto_sims.append(tan)

            mol_pairs.append((orig_mol, recon_mol, cos_sim, is_exact, original_smiles, recon_smi))

        validity = 100.0 * n_valid / n
        exact_match = 100.0 * n_exact / n
        mean_tan = float(np.mean(tanimoto_sims)) if tanimoto_sims else 0.0
        mean_cos = float(np.mean(cosines)) if cosines else 0.0

        self.experiment.log(
            f"[ReconEval] Epoch {epoch}: validity={validity:.1f}%, "
            f"exact={exact_match:.1f}%, tanimoto={mean_tan:.3f}, "
            f"cos(pred,gt)={mean_cos:.3f}"
        )

        self.experiment.track("recon_validity", validity)
        self.experiment.track("recon_exact_match", exact_match)
        self.experiment.track("recon_tanimoto", mean_tan)
        self.experiment.track("recon_pred_gt_cosine", mean_cos)

        # Side-by-side grid figure
        self._save_grid(epoch, mol_pairs, validity, exact_match, mean_tan, AllChem, Draw)

        if was_training:
            model.train()

    def _save_grid(
        self,
        epoch: int,
        mol_pairs: list,
        validity: float,
        exact_match: float,
        mean_tan: float,
        AllChem,
        Draw,
    ) -> None:
        n_grid = min(self.max_grid_pairs, len(mol_pairs))
        if n_grid == 0:
            return

        fig, axes = plt.subplots(n_grid, 2, figsize=(8, 3 * n_grid))
        if n_grid == 1:
            axes = axes.reshape(1, 2)

        fig.suptitle(
            f"Epoch {epoch} — Validity: {validity:.1f}%, "
            f"Exact: {exact_match:.1f}%, Tanimoto: {mean_tan:.3f}",
            fontsize=12, fontweight="bold",
        )

        for i in range(n_grid):
            orig_mol, recon_mol, cs, is_exact, orig_smi, recon_smi = mol_pairs[i]
            ax_o, ax_r = axes[i, 0], axes[i, 1]
            for ax in (ax_o, ax_r):
                ax.set_xticks([])
                ax.set_yticks([])

            # Original
            if orig_mol is not None:
                try:
                    AllChem.Compute2DCoords(orig_mol)
                    ax_o.imshow(Draw.MolToImage(orig_mol, size=(250, 250)))
                except Exception:
                    ax_o.text(0.5, 0.5, "Draw failed", ha="center", va="center",
                              transform=ax_o.transAxes, fontsize=10, color="orange")
            else:
                ax_o.text(0.5, 0.5, "No original", ha="center", va="center",
                          transform=ax_o.transAxes, fontsize=10, color="gray")
            ax_o.set_title(f"Original: {orig_smi[:40]}", fontsize=7, color="black")

            # Reconstructed
            if recon_mol is not None:
                try:
                    AllChem.Compute2DCoords(recon_mol)
                    ax_r.imshow(Draw.MolToImage(recon_mol, size=(250, 250)))
                except Exception:
                    ax_r.text(0.5, 0.5, "Draw failed", ha="center", va="center",
                              transform=ax_r.transAxes, fontsize=10, color="orange")
                color = "green" if is_exact else "blue"
                label = "EXACT" if is_exact else f"cos={cs:+.3f}"
                title = f"Recon ({label}): {(recon_smi or 'N/A')[:35]}"
            else:
                ax_r.text(0.5, 0.5, "Decode failed", ha="center", va="center",
                          transform=ax_r.transAxes, fontsize=10, color="red")
                color = "red"
                title = "Recon: N/A"
            ax_r.set_title(title, fontsize=7, color=color)

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        self.experiment.track("reconstructions_grid", fig)
        plt.close(fig)


# =============================================================================
# HELPERS
# =============================================================================


def clean_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Clean a molecule: remove stereo, radicals, charges, explicit Hs.

    Steps:
    1. Remove all stereochemistry (chiral centers + cis/trans bonds)
    2. Clear radical electrons (e.g. [NH]· → NH, [N]· → N)
    3. Neutralize formal charges (strip H from cations, add H to anions)
    4. Remove explicit hydrogens
    5. Re-sanitize and return canonical mol

    Returns None if any step fails.
    """
    try:
        mol = Chem.RWMol(mol)

        # 1. Remove stereochemistry
        Chem.RemoveStereochemistry(mol)

        # 2. Clear radical electrons (e.g. [NH]· → NH, [N]· → N)
        for atom in mol.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                atom.SetNumRadicalElectrons(0)
                atom.SetNoImplicit(False)

        # 3. Neutralize charges
        for atom in mol.GetAtoms():
            charge = atom.GetFormalCharge()
            if charge > 0:
                # Remove H from positively charged atoms (e.g. [NH3+] -> N)
                hs = atom.GetNumExplicitHs()
                remove = min(charge, hs)
                atom.SetNumExplicitHs(hs - remove)
                atom.SetFormalCharge(charge - remove)
            elif charge < 0:
                # Add H to negatively charged atoms (e.g. [O-] -> O)
                atom.SetNumExplicitHs(atom.GetNumExplicitHs() + abs(charge))
                atom.SetFormalCharge(0)

        mol = mol.GetMol()

        # 4. Remove explicit Hs
        mol = Chem.RemoveHs(mol)

        # 5. Re-sanitize
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def smiles_to_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[np.ndarray]:
    """Convert a SMILES string to a Morgan fingerprint numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = gen.GetFingerprint(mol)
        arr = np.zeros(n_bits, dtype=np.float32)
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None


# =============================================================================
# EXPERIMENT
# =============================================================================


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    """Train fingerprint-to-HDC translator and evaluate reconstruction."""

    # ── tmpdir fix for Lightning's _atomic_save ──
    custom_tmpdir = Path(e.path) / ".tmp_checkpoints"
    custom_tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(custom_tmpdir)
    tempfile.tempdir = str(custom_tmpdir)

    pl.seed_everything(e.SEED)

    e.log("=" * 60)
    e.log("Fingerprint → HDC Translation Experiment")
    e.log("=" * 60)
    e.log(f"CSV:            {e.CSV_PATH}")
    e.log(f"HDC encoder:    {e.HYPERNET_PATH}")
    e.log(f"Dataset key:    {e.DATASET}")
    e.log(f"FP radius={e.FP_RADIUS}, bits={e.FP_NBITS}")
    e.log(f"Hidden dims:    {e.HIDDEN_DIMS}")
    e.log(f"Epochs:         {e.EPOCHS}, batch={e.BATCH_SIZE}")
    e.log(f"Resblock depth: {e.RESBLOCK_DEPTH}, BerHu c_fraction={e.BERHU_C_FRACTION}")
    e.log("=" * 60)

    # ── device ──
    if e.ACCELERATOR == "gpu":
        device = torch.device("cuda")
    elif e.ACCELERATOR == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e.log(f"Device: {device}")

    # ── store config ──
    e["config/csv_path"] = e.CSV_PATH
    e["config/hdc_config_path"] = e.HYPERNET_PATH
    e["config/fp_radius"] = e.FP_RADIUS
    e["config/fp_nbits"] = e.FP_NBITS
    e["config/hidden_dims"] = list(e.HIDDEN_DIMS)
    e["config/epochs"] = e.EPOCHS
    e["config/batch_size"] = e.BATCH_SIZE
    e["config/lr"] = e.LEARNING_RATE
    e["config/resblock_depth"] = e.RESBLOCK_DEPTH
    e["config/berhu_c_fraction"] = e.BERHU_C_FRACTION

    # =====================================================================
    # Load HyperNet Encoder
    # =====================================================================

    if not e.HYPERNET_PATH:
        raise ValueError("HYPERNET_PATH is required — provide a HyperNet checkpoint.")

    e.log("\nLoading HyperNet encoder...")
    hypernet = load_hypernet(e.HYPERNET_PATH, device="cpu")
    hypernet.eval()

    if isinstance(hypernet, MultiHyperNet):
        actual_hdc_dim = hypernet.hv_dim
        ensemble_graph_dim = hypernet.ensemble_graph_dim
    else:
        actual_hdc_dim = hypernet.hv_dim
        ensemble_graph_dim = actual_hdc_dim

    concat_hdc_dim = actual_hdc_dim + ensemble_graph_dim
    e.log(f"HyperNet loaded: hdc_dim={actual_hdc_dim}, concat_dim={concat_hdc_dim}")
    e["config/actual_hdc_dim"] = actual_hdc_dim
    e["config/concat_hdc_dim"] = concat_hdc_dim

    # =====================================================================
    # Load Data (hook)
    # =====================================================================

    train_loader, val_loader, test_data = e.apply_hook(
        "load_data",
        hypernet=hypernet,
        device=torch.device("cpu"),
    )

    e["data/train_size"] = len(train_loader.dataset)
    e["data/val_size"] = len(val_loader.dataset)
    e["data/test_size"] = len(test_data)

    # =====================================================================
    # Create Model (hook)
    # =====================================================================

    model = e.apply_hook(
        "create_model",
        input_dim=e.FP_NBITS,
        output_dim=concat_hdc_dim,
    )

    num_params = sum(p.numel() for p in model.parameters())
    e.log(f"Model parameters: {num_params:,}")
    e["model/num_parameters"] = num_params

    # =====================================================================
    # Procrustes Initialization
    # =====================================================================

    if e.USE_PROCRUSTES_INIT:
        e.log("\n--- Procrustes (least-squares) Initialization ---")
        train_fp = train_loader.dataset.tensors[0]
        train_hdc = train_loader.dataset.tensors[1]

        n = e.PROCRUSTES_NUM_SAMPLES
        if n > 0 and n < len(train_fp):
            idx = torch.randperm(len(train_fp))[:n]
            train_fp_sub = train_fp[idx]
            train_hdc_sub = train_hdc[idx]
        else:
            train_fp_sub = train_fp
            train_hdc_sub = train_hdc

        hidden_dim = model.net[-1].in_features
        output_dim = model.net[-1].out_features
        e.log(f"  Running {len(train_fp_sub)} samples through backbone "
              f"to get hidden features (dim={hidden_dim})")
        e.log(f"  Solving least-squares: H_aug[{len(train_fp_sub)}, {hidden_dim + 1}] "
              f"@ W[{hidden_dim + 1}, {output_dim}] = Y[{len(train_fp_sub)}, {output_dim}]")

        t0 = time.time()
        residual_mse = model.initialize_procrustes(train_fp_sub, train_hdc_sub)
        elapsed = time.time() - t0

        e.log(f"  Initialized output layer weights via SVD/lstsq in {elapsed:.1f}s")
        e.log(f"  Residual MSE = {residual_mse:.6f}")
        e["model/procrustes_residual_mse"] = residual_mse

    # =====================================================================
    # Train (hook)
    # =====================================================================

    model = e.apply_hook(
        "train_model",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        hypernet=hypernet,
        test_data=test_data,
        device=device,
        actual_hdc_dim=actual_hdc_dim,
    )

    # =====================================================================
    # Evaluate (hook)
    # =====================================================================

    e.apply_hook(
        "evaluate",
        model=model,
        hypernet=hypernet,
        test_data=test_data,
        device=device,
        actual_hdc_dim=actual_hdc_dim,
    )

    e.log("\n" + "=" * 60)
    e.log("Experiment completed!")
    e.log("=" * 60)


# =============================================================================
# HOOKS
# =============================================================================


@experiment.hook("clean_data", default=True)
def clean_data(
    e: Experiment,
    smiles_list: List[str],
) -> List[str]:
    """
    Clean raw SMILES: remove stereochemistry, neutralize charges, strip
    explicit hydrogens.  Override this hook to change cleaning behaviour.

    Returns
    -------
    list[str]
        Cleaned, deduplicated canonical SMILES.
    """
    e.log("\nCleaning molecules...")
    cleaned: list[str] = []
    failed = 0
    seen: set[str] = set()

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed += 1
            continue

        mol = clean_mol(mol)
        if mol is None:
            failed += 1
            continue

        canon = Chem.MolToSmiles(mol, canonical=True)
        if canon in seen:
            continue
        seen.add(canon)
        cleaned.append(canon)

    e.log(f"Cleaning: {len(smiles_list)} raw -> {len(cleaned)} cleaned "
          f"({failed} failed, {len(smiles_list) - len(cleaned) - failed} duplicates)")
    return cleaned


@experiment.hook("encode_molecules", default=True)
def encode_molecules(
    e: Experiment,
    smiles_list: List[str],
    hypernet,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """
    Compute Morgan fingerprints and ground-truth HDC vectors for a list of
    cleaned canonical SMILES. Molecules that fail any step (fingerprinting,
    mol_to_data conversion, HDC preprocessing) are silently skipped.

    Override this hook to swap the fingerprint type or HDC encoding pipeline.

    Returns
    -------
    list[dict]
        Each dict has keys: smiles, fingerprint (Tensor), hdc_vector (Tensor).
    """
    from torch_geometric.data import Batch as PyGBatch

    needs_rw = (
        hasattr(hypernet, "rw_config")
        and hypernet.rw_config is not None
        and getattr(hypernet.rw_config, "enabled", False)
    )
    if needs_rw:
        from graph_hdc.utils.rw_features import augment_data_with_rw

    e.log("Computing fingerprints and HDC vectors...")
    out: list[dict] = []
    skipped = 0
    for i, smi in enumerate(smiles_list):
        if (i + 1) % 5000 == 0:
            e.log(f"  processed {i + 1}/{len(smiles_list)}...")

        fp_arr = smiles_to_fingerprint(smi, radius=e.FP_RADIUS, n_bits=e.FP_NBITS)
        if fp_arr is None:
            skipped += 1
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            skipped += 1
            continue

        try:
            pyg_data = mol_to_zinc_data(mol)
        except (ValueError, Exception):
            skipped += 1
            continue

        if pyg_data.edge_index.numel() == 0:
            # Greedy decoder requires edges; skip single-atom molecules.
            skipped += 1
            continue

        if needs_rw:
            pyg_data = augment_data_with_rw(
                pyg_data,
                k_values=hypernet.rw_config.k_values,
                num_bins=hypernet.rw_config.num_bins,
                bin_boundaries=hypernet.rw_config.bin_boundaries,
                clip_range=hypernet.rw_config.clip_range,
            )

        batch = PyGBatch.from_data_list([pyg_data]).to(device)
        with torch.no_grad():
            hdc_out = hypernet.forward(batch)
            edge_terms = hdc_out["edge_terms"].detach().cpu().float().view(-1)
            graph_terms = hdc_out["graph_embedding"].detach().cpu().float().view(-1)
        hdc_vec = torch.cat([edge_terms, graph_terms], dim=-1)

        out.append({
            "smiles": Chem.MolToSmiles(mol, canonical=True),
            "fingerprint": torch.from_numpy(fp_arr),
            "hdc_vector": hdc_vec,
        })

    e.log(f"Successfully processed {len(out)} molecules (skipped {skipped})")
    return out


@experiment.hook("load_data", default=True)
def load_data(
    e: Experiment,
    hypernet,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, List[Dict[str, Any]]]:
    """
    Load SMILES from CSV, compute fingerprints and HDC targets, split data.

    Returns
    -------
    train_loader : DataLoader
        TensorDataset(fingerprints, hdc_vectors)
    val_loader : DataLoader
        TensorDataset(fingerprints, hdc_vectors)
    test_data : list[dict]
        Each dict has keys: smiles, fingerprint (Tensor), hdc_vector (Tensor)
    """
    csv_path = e.CSV_PATH
    if not csv_path:
        raise ValueError("CSV_PATH is required.")

    # Resolve a relative CSV_PATH against the repo root (two levels up from
    # this file), so the experiment works regardless of CWD — in particular
    # when launched via a YAML config from another directory.
    p = Path(csv_path)
    if not p.is_absolute() and not p.exists():
        repo_root = Path(__file__).resolve().parents[2]
        candidate = repo_root / p
        if candidate.exists():
            csv_path = str(candidate)

    e.log(f"\nLoading SMILES from {csv_path}...")
    smiles_list: list[str] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row["smiles"].strip()
            if smi:
                smiles_list.append(smi)

    e.log(f"Read {len(smiles_list)} SMILES from CSV")

    if e.__TESTING__:
        smiles_list = smiles_list[:50]
        e.log(f"TESTING mode: using {len(smiles_list)} molecules")

    # ── cleaning + encoding (two-level caching) ──

    def _pack_records(records_list):
        """Pack records into compact numpy arrays for cache serialization.

        Streams row-by-row into preallocated arrays and nulls out each source
        record as it's copied, so the peak RSS during packing is ~one copy of
        the data, not two. For 1M mols (GDB13) this drops the peak from
        ~32 GB (torch.stack twice over a live list of 1M tensors) to ~16 GB.
        """
        n = len(records_list)
        if n == 0:
            return {
                "smiles": [],
                "fingerprints": np.zeros((0, 0), dtype=np.float32),
                "hdc_vectors": np.zeros((0, 0), dtype=np.float32),
            }
        fp_dim = records_list[0]["fingerprint"].shape[0]
        hdc_dim = records_list[0]["hdc_vector"].shape[0]
        fps = np.empty((n, fp_dim), dtype=np.float32)
        hdcs = np.empty((n, hdc_dim), dtype=np.float32)
        smiles = []
        for i in range(n):
            r = records_list[i]
            fps[i] = r["fingerprint"].numpy()
            hdcs[i] = r["hdc_vector"].numpy()
            smiles.append(r["smiles"])
            # Drop the per-record tensors immediately so GC can reclaim their
            # storage while we still have n-i entries to process.
            records_list[i] = None
        return {"smiles": smiles, "fingerprints": fps, "hdc_vectors": hdcs}

    def _unpack_records(packed):
        """Unpack cached arrays back into list of record dicts.

        The per-sample fingerprint / hdc_vector tensors are views into the
        shared packed arrays — they keep the arrays alive. Callers that hold
        these records long-term should clone the per-sample tensors (and
        drop the packed dict) to avoid keeping the full ~16 GB resident.
        """
        fps = torch.from_numpy(packed["fingerprints"])
        hdcs = torch.from_numpy(packed["hdc_vectors"])
        return [
            {"smiles": smi, "fingerprint": fps[i], "hdc_vector": hdcs[i]}
            for i, smi in enumerate(packed["smiles"])
        ]

    if not e.__TESTING__:
        hp = Path(e.HYPERNET_PATH)

        # Cache 1: cleaned SMILES — depends only on the raw CSV.
        @e.cache.cached(
            name="cleaned_smiles",
            scope=lambda _e: ("cleaned_smiles", Path(_e.CSV_PATH).stem),
        )
        def cleaned_smiles_cached():
            return e.apply_hook("clean_data", smiles_list=smiles_list)

        # Cache 2: fingerprint + HDC records — depends on the hypernet and
        # fingerprint parameters in addition to the cleaned SMILES.
        @e.cache.cached(
            name="records",
            scope=lambda _e: (
                "fingerprint_hdc",
                Path(_e.CSV_PATH).stem,
                f"{hp.parent.name}_{hp.stem}",
                f"fp_r{_e.FP_RADIUS}_b{_e.FP_NBITS}",
            ),
        )
        def records_cached():
            cleaned = cleaned_smiles_cached()
            records_list = e.apply_hook(
                "encode_molecules",
                smiles_list=cleaned,
                hypernet=hypernet,
                device=device,
            )
            return _pack_records(records_list)

        t0 = time.time()
        packed = records_cached()
        records = _unpack_records(packed)
        e.log(f"Data ready: {len(records)} records ({time.time() - t0:.1f}s)")
    else:
        cleaned = e.apply_hook("clean_data", smiles_list=smiles_list)
        records = e.apply_hook(
            "encode_molecules",
            smiles_list=cleaned,
            hypernet=hypernet,
            device=device,
        )

    if len(records) == 0:
        raise ValueError("No valid molecules after preprocessing!")

    # ── split ──
    rng = random.Random(e.SEED)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    n_train = int(len(records) * e.TRAIN_RATIO)
    n_val = int(len(records) * e.VAL_RATIO)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    e.log(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # ── build tensors ──
    def make_tensors(idx_list):
        fps = torch.stack([records[i]["fingerprint"] for i in idx_list])
        hdcs = torch.stack([records[i]["hdc_vector"] for i in idx_list])
        return fps, hdcs

    train_fp, train_hdc = make_tensors(train_idx)
    val_fp, val_hdc = make_tensors(val_idx)

    train_loader = DataLoader(
        TensorDataset(train_fp, train_hdc),
        batch_size=e.BATCH_SIZE,
        shuffle=True,
        num_workers=e.NUM_WORKERS,
        pin_memory=(e.NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        TensorDataset(val_fp, val_hdc),
        batch_size=e.BATCH_SIZE,
        shuffle=False,
        num_workers=e.NUM_WORKERS,
        pin_memory=(e.NUM_WORKERS > 0),
    )

    # Clone per-sample tensors for test_data so it doesn't hold views into
    # `packed`/`records` — then we can drop those and free ~16 GB of RSS.
    test_data = [
        {
            "smiles": records[i]["smiles"],
            "fingerprint": records[i]["fingerprint"].clone(),
            "hdc_vector": records[i]["hdc_vector"].clone(),
        }
        for i in test_idx
    ]

    # Explicitly release the shared packed arrays and the views list: train/val
    # tensors and test_data now own all their data.
    del records
    if not e.__TESTING__:
        del packed

    # ── export test SMILES for standalone evaluation ──
    test_csv_path = Path(e.path) / "test_smiles.csv"
    with open(test_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles"])
        for item in test_data:
            writer.writerow([item["smiles"]])
    e.log(f"Exported {len(test_data)} test SMILES to {test_csv_path}")

    return train_loader, val_loader, test_data


@experiment.hook("create_model", default=True)
def create_model(
    e: Experiment,
    input_dim: int,
    output_dim: int,
) -> pl.LightningModule:
    """
    Create the translation model.

    Default implementation returns a TranslatorMLP.  Override this hook to
    use a Transformer, attention-based model, or any other architecture.

    Returns
    -------
    pl.LightningModule
        Model with forward(fingerprint) → predicted_hdc_vector
    """
    e.log(f"\nCreating TranslatorMLP: {input_dim} → {e.HIDDEN_DIMS} "
          f"(resblock_depth={e.RESBLOCK_DEPTH}) → {output_dim}")
    return TranslatorMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=e.HIDDEN_DIMS,
        resblock_depth=e.RESBLOCK_DEPTH,
        berhu_c_fraction=e.BERHU_C_FRACTION,
        lr=e.LEARNING_RATE,
        weight_decay=e.WEIGHT_DECAY,
        warmup_epochs=e.WARMUP_EPOCHS,
        total_epochs=e.EPOCHS,
    )


@experiment.hook("train_model", default=True)
def train_model(
    e: Experiment,
    model: pl.LightningModule,
    train_loader: DataLoader,
    val_loader: DataLoader,
    hypernet,
    test_data: List[Dict[str, Any]],
    device: torch.device,
    actual_hdc_dim: int,
) -> pl.LightningModule:
    """
    Train the translation model with PyTorch Lightning.

    Returns the model with the best validation checkpoint loaded.
    """
    e.log("\nSetting up training...")

    checkpoint_dir = Path(e.path) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:03d}-{val/loss:.6f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    metrics_callback = TrainingMetricsCallback(experiment=e)
    callbacks = [best_ckpt_callback, metrics_callback]

    if e.RECON_EVAL_EVERY_N_EPOCHS > 0 and len(test_data) > 0:
        recon_callback = ReconstructionEvalCallback(
            experiment=e,
            hypernet=hypernet,
            test_data=test_data,
            device=device,
            actual_hdc_dim=actual_hdc_dim,
            dataset=e.DATASET.lower(),
            n_samples=e.RECON_EVAL_N_SAMPLES,
            every_n_epochs=e.RECON_EVAL_EVERY_N_EPOCHS,
            beam_size=e.RECON_EVAL_BEAM_SIZE,
            limit=e.RECON_EVAL_LIMIT,
            top_k=e.TOP_K,
        )
        callbacks.append(recon_callback)
        e.log(
            f"ReconEval: every {e.RECON_EVAL_EVERY_N_EPOCHS} epochs on "
            f"{min(e.RECON_EVAL_N_SAMPLES, len(test_data))} test molecules "
            f"(beam={e.RECON_EVAL_BEAM_SIZE}, limit={e.RECON_EVAL_LIMIT})"
        )

    log_dir = Path(e.path) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(e.path, name="logs", version=0)

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

    e.log("Starting training...")
    e.log("(Press CTRL+C to gracefully stop)")
    e.log("-" * 40)

    interrupted = False
    with GracefulInterruptHandler() as handler:
        handler.set_trainer(trainer)
        try:
            trainer.fit(model, train_loader, val_loader)
        except KeyboardInterrupt:
            interrupted = True
            e.log("\nTraining interrupted by user")
        except FileNotFoundError as exc:
            # CSVLogger directory can disappear if the debug folder is
            # cleaned by another run.  Recover using the best checkpoint.
            e.log(f"\nWarning: training aborted — log directory was deleted: {exc}")
            e.log("Recovering with best available checkpoint...")
            interrupted = True

    if not interrupted:
        e.log("-" * 40)
        e.log("Training complete!")

    # ── log final metrics ──
    best_val_loss = trainer.callback_metrics.get("val/loss")
    if best_val_loss is not None:
        e["results/best_val_loss"] = float(best_val_loss)
        e.log(f"Best val loss: {float(best_val_loss):.6f}")

    # ── load best checkpoint ──
    best_path = best_ckpt_callback.best_model_path
    if best_path and Path(best_path).exists():
        e.log(f"Loading best checkpoint: {best_path}")
        best_model = TranslatorMLP.load_from_checkpoint(best_path)
        best_model.eval()
        e["results/best_checkpoint"] = str(best_path)
        return best_model

    model.eval()
    return model


@experiment.hook("evaluate", default=True)
def evaluate(
    e: Experiment,
    model: pl.LightningModule,
    hypernet,
    test_data: List[Dict[str, Any]],
    device: torch.device,
    actual_hdc_dim: int,
) -> Dict[str, Any]:
    """
    End-to-end reconstruction evaluation using the HyperNet's greedy decoder.

    For each test molecule:
        fingerprint → TranslatorMLP → predicted [edge_terms | graph_embedding]
                    → hypernet.decode_graph_greedy
                    → reconstruct_for_eval(..., dataset=DATASET)
                    → RDKit mol for comparison against the original.
    """
    e.log("\n" + "=" * 60)
    e.log("Reconstruction Evaluation (greedy decoder)")
    e.log("=" * 60)
    e.log(f"Decoder settings: beam_size={e.BEAM_SIZE}, limit={e.LIMIT}, top_k={e.TOP_K}")

    decoder_settings = FallbackDecoderSettings(
        beam_size=e.BEAM_SIZE,
        limit=e.LIMIT,
        top_k=e.TOP_K,
    )

    hv_dim = actual_hdc_dim
    decode_device = hypernet.nodes_codebook.device
    vsa_cls = hypernet.vsa.tensor_class

    num_samples = min(e.NUM_TEST_SAMPLES, len(test_data))
    e.log(f"Evaluating {num_samples} test molecules...")

    model.to(device)
    model.eval()

    recon_dir = Path(e.path) / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    valid_count = 0
    match_count = 0
    tanimoto_scores: list[float] = []
    per_sample_cosines: list[float] = []

    decode_start = time.time()

    for idx in range(num_samples):
        item = test_data[idx]
        original_smiles = item["smiles"]
        fp_tensor = item["fingerprint"].unsqueeze(0).to(device)
        gt_hdc = item["hdc_vector"].to(device)

        original_mol = Chem.MolFromSmiles(original_smiles)
        if original_mol is None:
            results.append({
                "idx": idx,
                "original_smiles": original_smiles,
                "error": "Could not parse original SMILES",
            })
            continue

        # Predict HDC from fingerprint.
        with torch.no_grad():
            pred_hdc = model(fp_tensor).squeeze(0)
        cos_sim = float(F.cosine_similarity(pred_hdc, gt_hdc, dim=-1))
        per_sample_cosines.append(cos_sim)

        # Split [edge_terms | graph_embedding] and cast to the VSA tensor class.
        pred_hdc_cpu = pred_hdc.detach().cpu()
        edge_term = pred_hdc_cpu[:hv_dim].to(decode_device).as_subclass(vsa_cls)
        graph_term = pred_hdc_cpu[hv_dim:].to(decode_device).as_subclass(vsa_cls)

        # Greedy decode.
        generated_mol = None
        generated_smiles = None
        try:
            result = hypernet.decode_graph_greedy(
                edge_term=edge_term,
                graph_term=graph_term,
                decoder_settings=decoder_settings,
            )
            if result.nx_graphs:
                generated_mol = reconstruct_for_eval(
                    result.nx_graphs[0], dataset=e.DATASET.lower(),
                )
                generated_smiles = get_canonical_smiles(generated_mol)
        except Exception as ex:
            results.append({
                "idx": idx,
                "original_smiles": original_smiles,
                "error": f"Greedy decode failed: {ex}",
            })
            continue

        is_valid = is_valid_mol(generated_mol)

        original_canonical = None
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

        tanimoto = compute_tanimoto_similarity(original_mol, generated_mol)

        if is_valid:
            valid_count += 1
        if is_match:
            match_count += 1
        tanimoto_scores.append(tanimoto)

        status = "MATCH" if is_match else ("Valid" if is_valid else "Invalid")
        e.log(f"  * molecule {idx + 1}/{num_samples} - {status} - "
              f"cos(pred,gt): {cos_sim:+.3f} - tanimoto: {tanimoto:.3f} - "
              f"{original_smiles} -> {generated_smiles or 'N/A'}")

        results.append({
            "idx": idx,
            "original_smiles": original_smiles,
            "generated_smiles": generated_smiles,
            "is_valid": is_valid,
            "is_match": is_match,
            "tanimoto": tanimoto,
            "pred_gt_cosine": cos_sim,
        })

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

    total_decode_time = time.time() - decode_start
    mean_tanimoto = float(np.mean(tanimoto_scores)) if tanimoto_scores else 0.0
    median_tanimoto = float(np.median(tanimoto_scores)) if tanimoto_scores else 0.0
    mean_cos = float(np.mean(per_sample_cosines)) if per_sample_cosines else 0.0

    e.log("\n" + "-" * 40)
    e.log("Reconstruction Summary:")
    e.log(f"  Total samples:          {num_samples}")
    e.log(f"  Valid molecules:        {valid_count} ({100 * valid_count / num_samples:.1f}%)")
    e.log(f"  Exact matches:          {match_count} ({100 * match_count / num_samples:.1f}%)")
    e.log(f"  Mean Tanimoto:          {mean_tanimoto:.4f}")
    e.log(f"  Median Tanimoto:        {median_tanimoto:.4f}")
    e.log(f"  Mean cos(pred,gt):      {mean_cos:.4f}")
    e.log(f"  Decode time:            {total_decode_time:.2f}s")
    e.log("-" * 40)

    metrics = {
        "num_samples": num_samples,
        "valid_count": valid_count,
        "match_count": match_count,
        "valid_rate": valid_count / num_samples if num_samples > 0 else 0,
        "match_rate": match_count / num_samples if num_samples > 0 else 0,
        "mean_tanimoto": mean_tanimoto,
        "median_tanimoto": median_tanimoto,
        "mean_pred_gt_cosine": mean_cos,
        "total_decode_time_seconds": total_decode_time,
        "results": results,
    }

    e["evaluation/num_samples"] = num_samples
    e["evaluation/valid_count"] = valid_count
    e["evaluation/match_count"] = match_count
    e["evaluation/valid_rate"] = metrics["valid_rate"]
    e["evaluation/match_rate"] = metrics["match_rate"]
    e["evaluation/mean_tanimoto"] = mean_tanimoto
    e["evaluation/median_tanimoto"] = median_tanimoto
    e["evaluation/mean_pred_gt_cosine"] = mean_cos
    e["evaluation/total_decode_time_seconds"] = total_decode_time

    e.commit_json("evaluation_results.json", metrics)

    return metrics


# =============================================================================
# Entry Point
# =============================================================================

experiment.run_if_main()
