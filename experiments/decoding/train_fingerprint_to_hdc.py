#!/usr/bin/env python
"""
Train Fingerprint-to-HDC Translation Network.

Trains a deep residual MLP (TranslatorMLP) to map ECFP4 Morgan fingerprints
(2048-bit) to HDC hypervectors.  After training, evaluates end-to-end
molecular reconstruction:

    SMILES → ECFP4 → TranslatorMLP → predicted HDC
        → decode_nodes_from_hdc → FlowEdgeDecoder.sample() → graph
        → compare with original molecule

This is the BASE EXPERIMENT.  Child experiments can inherit via
Experiment.extend() and override hooks to swap the model architecture
(e.g. Transformer), data loading strategy, or evaluation procedure.

Usage:
    # Quick smoke test
    python train_fingerprint_to_hdc.py --__TESTING__ True

    # Full training (uses data/qm9_smiles.csv by default)
    python train_fingerprint_to_hdc.py \\
        --HYPERNET_PATH /path/to/encoder.ckpt \\
        --FLOW_EDGE_DECODER_PATH /path/to/decoder.ckpt
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
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data

from graph_hdc.datasets.zinc_smiles import mol_to_data as mol_to_zinc_data
from graph_hdc.hypernet import load_hypernet
from graph_hdc.hypernet.multi_hypernet import MultiHyperNet
from graph_hdc.models.flow_edge_decoder import (
    FlowEdgeDecoder,
    get_node_feature_bins,
    node_tuples_to_onehot,
    preprocess_for_flow_edge_decoder,
)
from graph_hdc.models.translator_mlp import TranslatorMLP
from graph_hdc.utils.experiment_helpers import (
    GracefulInterruptHandler,
    compute_hdc_distance,
    compute_tanimoto_similarity,
    create_reconstruction_plot,
    decode_nodes_from_hdc,
    get_canonical_smiles,
    is_valid_mol,
    pyg_to_mol,
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
#     This encoder is used to compute the ground-truth HDC targets.
HYPERNET_PATH: str = "/media/ssd2/Programming/graph_hdc_public/experiments/decoding/results/train_flow_edge_decoder_streaming/GOOD/hypernet_encoder.ckpt"

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
#     Hidden layer dimensions for the TranslatorMLP.  Consecutive equal dims
#     produce residual blocks; dimension changes use linear projections.
HIDDEN_DIMS: tuple = (1024 * 4, 1024 * 2, 1024 * 2, 1024, 1024, 1024, 1024, 1024 * 2, 1024 * 2, 1024 * 4)

# :param ACTIVATION:
#     Activation function.  One of "relu", "gelu", "silu", "leaky_relu", "tanh".
ACTIVATION: str = "relu"

# :param NORM:
#     Normalisation layer.  One of "lay_norm", "batch_norm", "none".
NORM: str = "lay_norm"

# :param DROPOUT:
#     Dropout probability inside residual / projection blocks.
DROPOUT: float = 0.25

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

# :param EPOCHS:
#     Number of training epochs.
EPOCHS: int = 500

# :param BATCH_SIZE:
#     Mini-batch size.
BATCH_SIZE: int = 256

# :param LEARNING_RATE:
#     Initial learning rate for AdamW.
LEARNING_RATE: float = 2e-4

# :param WEIGHT_DECAY:
#     Weight decay (L2 regularization).
WEIGHT_DECAY: float = 0e-5

# :param COSINE_LOSS_WEIGHT:
#     Weight of the cosine-similarity loss component.
COSINE_LOSS_WEIGHT: float = 1.0

# :param MSE_LOSS_WEIGHT:
#     Weight of the MSE loss component.
MSE_LOSS_WEIGHT: float = 1.0

# :param INFONCE_LOSS_WEIGHT:
#     Weight of the InfoNCE (NT-Xent) contrastive loss component.
#     Encourages the model to preserve neighborhood structure: similar
#     fingerprints map to similar HDC vectors.  Set to 0.0 to disable.
INFONCE_LOSS_WEIGHT: float = 0.1

# :param INFONCE_TEMPERATURE:
#     Temperature for InfoNCE softmax.  Lower = sharper distribution
#     (harder negatives).  Typical range: 0.05–0.1.
INFONCE_TEMPERATURE: float = 0.07

# :param WARMUP_EPOCHS:
#     Number of epochs for linear LR warmup.  Set to 0 to disable warmup.
WARMUP_EPOCHS: int = 10

# :param USE_PROCRUSTES_INIT:
#     Whether to initialize the output layer via least-squares (Procrustes)
#     fit on the training data before training begins.
USE_PROCRUSTES_INIT: bool = True

# :param PROCRUSTES_NUM_SAMPLES:
#     Number of training samples used for the Procrustes fit.
#     0 = use all training samples.
PROCRUSTES_NUM_SAMPLES: int = 0

# :param GRADIENT_CLIP_VAL:
#     Gradient clipping value.  0.0 disables clipping.
GRADIENT_CLIP_VAL: float = 1.0

# -----------------------------------------------------------------------------
# FlowEdgeDecoder (evaluation only)
# -----------------------------------------------------------------------------

# :param FLOW_EDGE_DECODER_PATH:
#     Path to a pre-trained FlowEdgeDecoder checkpoint (.ckpt).
#     Used only during the end-of-training reconstruction evaluation.
FLOW_EDGE_DECODER_PATH: str = "/media/ssd2/Programming/graph_hdc_public/experiments/decoding/results/train_flow_edge_decoder_streaming/GOOD/last.ckpt"

# :param SAMPLE_STEPS:
#     Number of denoising steps for FlowEdgeDecoder sampling.
SAMPLE_STEPS: int = 50

# :param ETA:
#     Stochasticity parameter for FlowEdgeDecoder sampling.
ETA: float = 0.0

# :param SAMPLE_TIME_DISTORTION:
#     Time distortion during sampling.  Options: "identity", "polydec".
SAMPLE_TIME_DISTORTION: str = "polydec"

# :param NUM_REPETITIONS:
#     Best-of-N repetitions per molecule during FlowEdgeDecoder sampling.
#     Each molecule is decoded this many times and the result with the
#     lowest HDC cosine distance is kept.
NUM_REPETITIONS: int = 128

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

# :param NUM_TEST_SAMPLES:
#     Maximum number of test molecules to run through reconstruction eval.
NUM_TEST_SAMPLES: int = 100

# :param RECONSTRUCTION_BATCH_SIZE:
#     Batch size for FlowEdgeDecoder sampling during evaluation.
RECONSTRUCTION_BATCH_SIZE: int = 1

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
    Track per-epoch training diagnostics and produce a 2x4 metrics grid.

    Tracks losses (combined, cosine, MSE), cosine similarity, learning rate,
    gradient norm, parameter delta, and weight norm.  After each validation
    epoch the plot is overwritten so the latest version is always on disk.
    """

    def __init__(self, experiment: Experiment):
        super().__init__()
        self.experiment = experiment

        # Loss curves
        self.train_loss: list[float] = []
        self.val_loss: list[float] = []
        self.train_cosine: list[float] = []
        self.val_cosine: list[float] = []
        self.train_mse: list[float] = []
        self.val_mse: list[float] = []

        # Derived: cosine similarity = 1 - cosine_loss
        self.cosine_sim_train: list[float] = []
        self.cosine_sim_val: list[float] = []

        # InfoNCE loss
        self.train_infonce: list[float] = []
        self.val_infonce: list[float] = []

        # Overfitting gap
        self.overfit_gap: list[float] = []

        # Training diagnostics
        self.lr: list[float] = []
        self.grad_norm: list[float] = []
        self.param_delta: list[float] = []
        self.weight_norm: list[float] = []

        # Internal state
        self._param_snapshot: dict[str, torch.Tensor] = {}
        self._last_grad_norm: float | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_metric(trainer: Trainer, key: str) -> float | None:
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
        # Combined loss
        tl = self._get_metric(trainer, "train/loss")
        if tl is not None:
            self.train_loss.append(tl)
            self.experiment.track("loss_train", tl)

        # Cosine loss
        tc = self._get_metric(trainer, "train/cosine")
        if tc is not None:
            self.train_cosine.append(tc)
            self.experiment.track("cosine_train", tc)
            self.cosine_sim_train.append(1.0 - tc)
            self.experiment.track("cosine_sim_train", 1.0 - tc)

        # MSE loss
        tm = self._get_metric(trainer, "train/mse")
        if tm is not None:
            self.train_mse.append(tm)
            self.experiment.track("mse_train", tm)

        # InfoNCE loss
        ti = self._get_metric(trainer, "train/infonce")
        if ti is not None:
            self.train_infonce.append(ti)
            self.experiment.track("infonce_train", ti)

        # Learning rate
        lr = None
        for opt in trainer.optimizers:
            for pg in opt.param_groups:
                lr = pg["lr"]
                break
        if lr is not None:
            self.lr.append(lr)
            self.experiment.track("learning_rate", lr)

        # Gradient norm
        if self._last_grad_norm is not None:
            self.grad_norm.append(self._last_grad_norm)
            self.experiment.track("grad_norm", self._last_grad_norm)

        # Parameter delta and weight norm
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

        vc = self._get_metric(trainer, "val/cosine")
        if vc is not None:
            self.val_cosine.append(vc)
            self.experiment.track("cosine_val", vc)
            self.cosine_sim_val.append(1.0 - vc)
            self.experiment.track("cosine_sim_val", 1.0 - vc)

        vm = self._get_metric(trainer, "val/mse")
        if vm is not None:
            self.val_mse.append(vm)
            self.experiment.track("mse_val", vm)

        # InfoNCE loss
        vi = self._get_metric(trainer, "val/infonce")
        if vi is not None:
            self.val_infonce.append(vi)
            self.experiment.track("infonce_val", vi)

        # Overfitting gap (val - train; positive = overfitting)
        if self.train_loss and self.val_loss:
            gap = self.val_loss[-1] - self.train_loss[-1]
            self.overfit_gap.append(gap)
            self.experiment.track("overfit_gap", gap)

        # Generate figure (skip very first sanity-check validation)
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

        epochs_t = list(range(1, len(self.train_loss) + 1))
        epochs_v = list(range(1, len(self.val_loss) + 1))

        # ---- Row 1, Col 1: Train/Val Combined Loss ----
        ax = axes[0, 0]
        if self.train_loss:
            ax.plot(epochs_t, self.train_loss, "C0", alpha=0.3, linewidth=0.8)
            ax.plot(epochs_t, self._smooth(self.train_loss), "C0", label="train")
        if self.val_loss:
            ax.plot(epochs_v, self.val_loss, "C1", alpha=0.3, linewidth=0.8)
            ax.plot(epochs_v, self._smooth(self.val_loss), "C1", label="val")
            ax.axhline(min(self.val_loss), color="gray", linestyle=":", alpha=0.5)
            ax.annotate(
                f"best: {min(self.val_loss):.4f}",
                xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, color="gray",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Combined Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- Row 1, Col 2: Train/Val Cosine Loss ----
        ax = axes[0, 1]
        if self.train_cosine:
            ep = list(range(1, len(self.train_cosine) + 1))
            ax.plot(ep, self.train_cosine, "C0", alpha=0.3, linewidth=0.8)
            ax.plot(ep, self._smooth(self.train_cosine), "C0", label="train")
        if self.val_cosine:
            ep = list(range(1, len(self.val_cosine) + 1))
            ax.plot(ep, self.val_cosine, "C1", alpha=0.3, linewidth=0.8)
            ax.plot(ep, self._smooth(self.val_cosine), "C1", label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cosine Loss")
        ax.set_title("Cosine Loss (1 - sim)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- Row 1, Col 3: Train/Val MSE Loss ----
        ax = axes[0, 2]
        if self.train_mse:
            ep = list(range(1, len(self.train_mse) + 1))
            ax.plot(ep, self.train_mse, "C0", alpha=0.3, linewidth=0.8)
            ax.plot(ep, self._smooth(self.train_mse), "C0", label="train")
        if self.val_mse:
            ep = list(range(1, len(self.val_mse) + 1))
            ax.plot(ep, self.val_mse, "C1", alpha=0.3, linewidth=0.8)
            ax.plot(ep, self._smooth(self.val_mse), "C1", label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title("MSE Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- Row 1, Col 4: Mean Cosine Similarity ----
        ax = axes[0, 3]
        if self.cosine_sim_train:
            ep = list(range(1, len(self.cosine_sim_train) + 1))
            ax.plot(ep, self.cosine_sim_train, "C0", alpha=0.3, linewidth=0.8)
            ax.plot(ep, self._smooth(self.cosine_sim_train), "C0", label="train")
        if self.cosine_sim_val:
            ep = list(range(1, len(self.cosine_sim_val) + 1))
            ax.plot(ep, self.cosine_sim_val, "C1", alpha=0.3, linewidth=0.8)
            ax.plot(ep, self._smooth(self.cosine_sim_val), "C1", label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Mean Cosine Similarity")
        ax.legend(fontsize=8)
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

        # ---- Row 1, Col 5: InfoNCE Loss ----
        ax = axes[0, 4]
        if self.train_infonce:
            ep = list(range(1, len(self.train_infonce) + 1))
            ax.plot(ep, self.train_infonce, "C0", alpha=0.3, linewidth=0.8)
            ax.plot(ep, self._smooth(self.train_infonce), "C0", label="train")
        if self.val_infonce:
            ep = list(range(1, len(self.val_infonce) + 1))
            ax.plot(ep, self.val_infonce, "C1", alpha=0.3, linewidth=0.8)
            ax.plot(ep, self._smooth(self.val_infonce), "C1", label="val")
        if not self.train_infonce and not self.val_infonce:
            ax.text(0.5, 0.5, "InfoNCE disabled\n(weight = 0)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="gray")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("InfoNCE")
        ax.set_title("InfoNCE Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- Row 2, Col 5: Train-Val Gap (overfitting indicator) ----
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
    e.log(f"Decoder ckpt:   {e.FLOW_EDGE_DECODER_PATH}")
    e.log(f"FP radius={e.FP_RADIUS}, bits={e.FP_NBITS}")
    e.log(f"Hidden dims:    {e.HIDDEN_DIMS}")
    e.log(f"Epochs:         {e.EPOCHS}, batch={e.BATCH_SIZE}")
    e.log(f"Loss weights:   cosine={e.COSINE_LOSS_WEIGHT}, mse={e.MSE_LOSS_WEIGHT}, "
          f"infonce={e.INFONCE_LOSS_WEIGHT} (tau={e.INFONCE_TEMPERATURE})")
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
    e["config/cosine_weight"] = e.COSINE_LOSS_WEIGHT
    e["config/mse_weight"] = e.MSE_LOSS_WEIGHT

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
    )

    # =====================================================================
    # Evaluate (hook)
    # =====================================================================

    if e.FLOW_EDGE_DECODER_PATH:
        metrics = e.apply_hook(
            "evaluate",
            model=model,
            hypernet=hypernet,
            test_data=test_data,
            device=device,
            actual_hdc_dim=actual_hdc_dim,
        )
    else:
        e.log("\nNo FLOW_EDGE_DECODER_PATH provided — skipping reconstruction evaluation.")

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

    # ── compute fingerprints + HDC vectors (with caching) ──

    def _encode_all(smiles_to_encode):
        """Clean, fingerprint, and HDC-encode a list of SMILES."""
        cleaned = e.apply_hook("clean_data", smiles_list=smiles_to_encode)

        e.log("Computing fingerprints and HDC vectors...")
        out: list[dict] = []
        skipped = 0
        for i, smi in enumerate(cleaned):
            if (i + 1) % 5000 == 0:
                e.log(f"  processed {i + 1}/{len(cleaned)}...")

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

            processed = preprocess_for_flow_edge_decoder(pyg_data, hypernet, device=device)
            if processed is None:
                skipped += 1
                continue

            hdc_vec = processed.hdc_vector.squeeze(0)  # (concat_hdc_dim,)

            out.append({
                "smiles": Chem.MolToSmiles(mol, canonical=True),
                "fingerprint": torch.from_numpy(fp_arr),
                "hdc_vector": hdc_vec,
            })

        e.log(f"Successfully processed {len(out)} molecules (skipped {skipped})")
        return out

    def _pack_records(records_list):
        """Pack records into compact arrays for efficient cache serialization."""
        return {
            "smiles": [r["smiles"] for r in records_list],
            "fingerprints": torch.stack([r["fingerprint"] for r in records_list]).numpy(),
            "hdc_vectors": torch.stack([r["hdc_vector"] for r in records_list]).numpy(),
        }

    def _unpack_records(packed):
        """Unpack cached arrays back into list of record dicts."""
        fps = torch.from_numpy(packed["fingerprints"])
        hdcs = torch.from_numpy(packed["hdc_vectors"])
        return [
            {"smiles": smi, "fingerprint": fps[i], "hdc_vector": hdcs[i]}
            for i, smi in enumerate(packed["smiles"])
        ]

    if not e.__TESTING__:
        hp = Path(e.HYPERNET_PATH)

        @e.cache.cached(
            name="records",
            scope=lambda _e: (
                "fingerprint_hdc",
                Path(_e.CSV_PATH).stem,
                f"{hp.parent.name}_{hp.stem}",
                f"fp_r{_e.FP_RADIUS}_b{_e.FP_NBITS}",
            ),
        )
        def compute_records_cached():
            return _pack_records(_encode_all(smiles_list))

        t0 = time.time()
        packed = compute_records_cached()
        records = _unpack_records(packed)
        e.log(f"Data ready: {len(records)} records ({time.time() - t0:.1f}s)")
    else:
        records = _encode_all(smiles_list)

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

    test_data = [records[i] for i in test_idx]

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
    e.log(f"\nCreating TranslatorMLP: {input_dim} → {e.HIDDEN_DIMS} → {output_dim}")
    return TranslatorMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=e.HIDDEN_DIMS,
        activation=e.ACTIVATION,
        norm=e.NORM,
        dropout=e.DROPOUT,
        lr=e.LEARNING_RATE,
        weight_decay=e.WEIGHT_DECAY,
        cosine_loss_weight=e.COSINE_LOSS_WEIGHT,
        mse_loss_weight=e.MSE_LOSS_WEIGHT,
        infonce_loss_weight=e.INFONCE_LOSS_WEIGHT,
        infonce_temperature=e.INFONCE_TEMPERATURE,
        warmup_epochs=e.WARMUP_EPOCHS,
        total_epochs=e.EPOCHS,
    )


@experiment.hook("train_model", default=True)
def train_model(
    e: Experiment,
    model: pl.LightningModule,
    train_loader: DataLoader,
    val_loader: DataLoader,
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
    End-to-end reconstruction evaluation.

    For each test molecule:
    1. Fingerprint → MLP → predicted HDC
    2. decode_nodes_from_hdc → node tuples
    3. FlowEdgeDecoder.sample() → generated graph
    4. Compare with original molecule
    """
    e.log("\n" + "=" * 60)
    e.log("Reconstruction Evaluation")
    e.log("=" * 60)

    # ── load decoder ──
    decoder_path = e.FLOW_EDGE_DECODER_PATH
    e.log(f"Loading FlowEdgeDecoder from {decoder_path}...")

    decoder = FlowEdgeDecoder.load(decoder_path, device=device)
    decoder.eval()
    e.log("FlowEdgeDecoder loaded.")

    # ── get feature bins for node decoding ──
    rw_config = getattr(hypernet, "rw_config", None)
    feature_bins = get_node_feature_bins(rw_config)

    # ── select test samples ──
    num_samples = min(e.NUM_TEST_SAMPLES, len(test_data))
    e.log(f"Evaluating {num_samples} test molecules...")

    model.to(device)
    model.eval()

    # ── phase 1: predict HDC + decode nodes ──
    e.log("Phase 1: predicting HDC vectors and decoding nodes...")
    sampling_start = time.time()
    prepared: list[dict] = []
    results: list[dict] = []
    node_decode_correct = 0

    for idx in range(num_samples):
        if (idx + 1) % 10 == 0 or idx == 0:
            e.log(f"  decoding nodes {idx + 1}/{num_samples}...")
        item = test_data[idx]
        original_smiles = item["smiles"]
        fp_tensor = item["fingerprint"].unsqueeze(0).to(device)
        gt_hdc = item["hdc_vector"]

        original_mol = Chem.MolFromSmiles(original_smiles)
        if original_mol is None:
            results.append({
                "idx": idx,
                "original_smiles": original_smiles,
                "error": "Could not parse original SMILES",
            })
            continue

        # predict HDC from fingerprint
        with torch.no_grad():
            pred_hdc = model(fp_tensor).squeeze(0).cpu()

        # decode nodes from predicted HDC
        try:
            node_tuples, num_nodes = decode_nodes_from_hdc(
                hypernet, pred_hdc.unsqueeze(0), actual_hdc_dim
            )
        except Exception as ex:
            results.append({
                "idx": idx,
                "original_smiles": original_smiles,
                "error": f"Node decode failed: {ex}",
            })
            continue

        if num_nodes == 0:
            results.append({
                "idx": idx,
                "original_smiles": original_smiles,
                "error": "No nodes decoded",
            })
            continue

        # check node decode accuracy
        original_num_atoms = original_mol.GetNumAtoms()
        if num_nodes == original_num_atoms:
            node_decode_correct += 1

        node_features = node_tuples_to_onehot(
            node_tuples, device=device, feature_bins=feature_bins
        )

        prepared.append({
            "idx": idx,
            "original_smiles": original_smiles,
            "original_mol": original_mol,
            "pred_hdc": pred_hdc,
            "node_features": node_features,
            "num_nodes": num_nodes,
        })

    # ── phase 2: generate, evaluate, and plot per molecule ──
    num_reps = e.NUM_REPETITIONS
    e.log(f"Generating edges for {len(prepared)} molecules "
          f"(best-of-{num_reps}, {e.SAMPLE_STEPS} steps)...")

    sample_kwargs = dict(
        sample_steps=e.SAMPLE_STEPS,
        eta=e.ETA,
        time_distortion=e.SAMPLE_TIME_DISTORTION,
        show_progress=False,
        device=device,
    )

    recon_dir = Path(e.path) / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)

    valid_count = 0
    match_count = 0
    plot_count = 0
    tanimoto_scores: list[float] = []

    for i, item in enumerate(prepared):
        idx = item["idx"]
        original_smiles = item["original_smiles"]
        original_mol = item["original_mol"]

        # ── generate edges (best-of-N) ──
        hdc_vec = item["pred_hdc"].unsqueeze(0).to(device)
        n = item["num_nodes"]
        nf = item["node_features"].unsqueeze(0).to(device)
        mask = torch.ones(1, n, dtype=torch.bool, device=device)

        def score_fn(s, _orig=hdc_vec, _dim=actual_hdc_dim):
            return compute_hdc_distance(
                s, _orig, _dim, hypernet, device,
            )

        with torch.no_grad():
            best_sample, best_dist, avg_dist = decoder.sample_best_of_n(
                hdc_vectors=hdc_vec,
                node_features=nf,
                node_mask=mask,
                num_repetitions=num_reps,
                score_fn=score_fn,
                **sample_kwargs,
            )

        # ── evaluate ──
        generated_mol = pyg_to_mol(best_sample)
        generated_smiles = get_canonical_smiles(generated_mol)
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

        tanimoto = compute_tanimoto_similarity(original_mol, generated_mol)

        if is_valid:
            valid_count += 1
        if is_match:
            match_count += 1
        tanimoto_scores.append(tanimoto)

        status = "MATCH" if is_match else ("Valid" if is_valid else "Invalid")
        e.log(f"  * molecule {i + 1}/{len(prepared)} - {status} - "
              f"dist: {best_dist:.4f} - tanimoto: {tanimoto:.3f} - "
              f"{original_smiles} -> {generated_smiles or 'N/A'}")

        results.append({
            "idx": idx,
            "original_smiles": original_smiles,
            "generated_smiles": generated_smiles,
            "is_valid": is_valid,
            "is_match": is_match,
            "tanimoto": tanimoto,
            "hdc_distance": best_dist,
        })

        # ── plot immediately ──
        plot_path = recon_dir / f"reconstruction_{plot_count + 1:03d}.png"
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
        plot_count += 1

    if plot_count > 0:
        e.log(f"Saved {plot_count} reconstruction plots to {recon_dir}")

    # ── summary ──
    total_sampling_time = time.time() - sampling_start
    mean_tanimoto = float(np.mean(tanimoto_scores)) if tanimoto_scores else 0.0
    median_tanimoto = float(np.median(tanimoto_scores)) if tanimoto_scores else 0.0

    e.log("\n" + "-" * 40)
    e.log("Reconstruction Summary:")
    e.log(f"  Total samples:          {num_samples}")
    e.log(f"  Valid molecules:        {valid_count} ({100 * valid_count / num_samples:.1f}%)")
    e.log(f"  Exact matches:          {match_count} ({100 * match_count / num_samples:.1f}%)")
    e.log(f"  Node count accuracy:    {node_decode_correct}/{num_samples} "
          f"({100 * node_decode_correct / num_samples:.1f}%)")
    e.log(f"  Mean Tanimoto:          {mean_tanimoto:.4f}")
    e.log(f"  Median Tanimoto:        {median_tanimoto:.4f}")
    e.log(f"  Sampling time:          {total_sampling_time:.2f}s")
    e.log("-" * 40)

    # ── store metrics ──
    metrics = {
        "num_samples": num_samples,
        "valid_count": valid_count,
        "match_count": match_count,
        "valid_rate": valid_count / num_samples if num_samples > 0 else 0,
        "match_rate": match_count / num_samples if num_samples > 0 else 0,
        "node_decode_correct": node_decode_correct,
        "node_decode_accuracy": node_decode_correct / num_samples if num_samples > 0 else 0,
        "mean_tanimoto": mean_tanimoto,
        "median_tanimoto": median_tanimoto,
        "total_sampling_time_seconds": total_sampling_time,
        "results": results,
    }

    e["evaluation/num_samples"] = num_samples
    e["evaluation/valid_count"] = valid_count
    e["evaluation/match_count"] = match_count
    e["evaluation/valid_rate"] = metrics["valid_rate"]
    e["evaluation/match_rate"] = metrics["match_rate"]
    e["evaluation/node_decode_accuracy"] = metrics["node_decode_accuracy"]
    e["evaluation/mean_tanimoto"] = mean_tanimoto
    e["evaluation/median_tanimoto"] = median_tanimoto
    e["evaluation/total_sampling_time_seconds"] = total_sampling_time

    e.commit_json("evaluation_results.json", metrics)

    return metrics


# =============================================================================
# Entry Point
# =============================================================================

experiment.run_if_main()
