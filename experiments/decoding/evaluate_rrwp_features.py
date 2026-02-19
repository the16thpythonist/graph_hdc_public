#!/usr/bin/env python
"""
Evaluate RRWP Feature Distributions by Molecule Size.

This experiment compares the distributions of raw random walk return
probabilities between small (<=10 atoms) and medium (20-30 atoms) molecules
from the ZINC dataset.  For each k in {2, 6, 16} it produces a KDE/histogram
overlay with the current 4-bin quantile boundaries drawn as vertical lines,
making it easy to see whether the bin boundaries that were calibrated on the
full (predominantly large) ZINC training set provide adequate discrimination
for small molecules.

Usage:
    # Quick test (100 molecules per group, fast)
    python evaluate_rrwp_features.py --__TESTING__ True

    # Full evaluation
    python evaluate_rrwp_features.py --__DEBUG__ False
"""

from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from tqdm import tqdm

from graph_hdc.datasets.utils import get_split
from graph_hdc.utils.rw_features import (
    compute_rw_return_probabilities,
    get_zinc_rw_boundaries,
)

# =============================================================================
# PARAMETERS
# =============================================================================

# :param K_VALUES:
#     Random walk step counts at which to evaluate return probabilities.
K_VALUES: tuple[int, ...] = (2, 6, 16)

# :param NUM_BINS:
#     Number of bins for discretising RW return probabilities.  When
#     BIN_MODE is "quantile", must be one of {3, 4, 5, 6} (precomputed).
NUM_BINS: int = 6

# :param BIN_MODE:
#     Binning strategy to visualise.  One of:
#       - "quantile": precomputed quantile boundaries (default).
#       - "clipped":  uniform bins over CLIP_RANGE.
#       - "uniform":  uniform bins over [0, 1].
BIN_MODE: str = "clipped"

# :param CLIP_RANGE:
#     (lo, hi) range for clipped uniform binning.  Only used when
#     BIN_MODE is "clipped".
CLIP_RANGE: tuple[float, float] = (0.0, 0.8)

# :param SMALL_MAX_ATOMS:
#     Maximum number of (heavy) atoms for the "small" molecule group.
SMALL_MAX_ATOMS: int = 8

# :param MEDIUM_MIN_ATOMS:
#     Minimum number of atoms for the "medium" molecule group.
MEDIUM_MIN_ATOMS: int = 25

# :param MEDIUM_MAX_ATOMS:
#     Maximum number of atoms for the "medium" molecule group.
MEDIUM_MAX_ATOMS: int = 30

# :param SAMPLES_PER_GROUP:
#     Number of molecules to sample from each size group.  Set to 0 to use
#     all molecules that match the filter.
SAMPLES_PER_GROUP: int = 5000

# :param SEED:
#     Random seed for reproducible molecule sampling.
SEED: int = 42

# :param HISTOGRAM_BINS:
#     Number of histogram bins for the raw probability distributions.
HISTOGRAM_BINS: int = 80

# :param KDE_BANDWIDTH:
#     Bandwidth for the Gaussian KDE overlay.  Set to 0 to disable KDE.
KDE_BANDWIDTH: float = 0.01

__DEBUG__ = True
__TESTING__ = False

# =============================================================================
# EXPERIMENT
# =============================================================================


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment):
    random.seed(e.SEED)

    # ------------------------------------------------------------------
    # 1. Load ZINC and split by size
    # ------------------------------------------------------------------
    e.log("Loading ZINC train + validation splits...")
    zinc_train = get_split(dataset="zinc", split="train")
    zinc_valid = get_split(dataset="zinc", split="valid")
    all_data = list(zinc_train) + list(zinc_valid)
    e.log(f"Total molecules loaded: {len(all_data)}")

    small_pool = [d for d in all_data if d.x.size(0) <= e.SMALL_MAX_ATOMS]
    medium_pool = [d for d in all_data if e.MEDIUM_MIN_ATOMS <= d.x.size(0) <= e.MEDIUM_MAX_ATOMS]
    e.log(f"Small pool (<={e.SMALL_MAX_ATOMS} atoms): {len(small_pool)} molecules")
    e.log(f"Medium pool ({e.MEDIUM_MIN_ATOMS}-{e.MEDIUM_MAX_ATOMS} atoms): {len(medium_pool)} molecules")

    n = e.SAMPLES_PER_GROUP
    if n > 0:
        small_mols = random.sample(small_pool, min(n, len(small_pool)))
        medium_mols = random.sample(medium_pool, min(n, len(medium_pool)))
    else:
        small_mols = small_pool
        medium_mols = medium_pool

    e.log(f"Sampled: {len(small_mols)} small, {len(medium_mols)} medium")

    # ------------------------------------------------------------------
    # 2. Compute raw RW return probabilities
    # ------------------------------------------------------------------
    k_values = e.K_VALUES

    def collect_rw_probs(molecules, label):
        """Return dict mapping k -> np.array of all per-atom probabilities."""
        per_k = {k: [] for k in k_values}
        for data in tqdm(molecules, desc=f"RW probs ({label})", disable=not e.__DEBUG__):
            with torch.no_grad():
                probs = compute_rw_return_probabilities(
                    data.edge_index, data.x.size(0), k_values=k_values,
                )  # [N, len(k_values)]
            for i, k in enumerate(k_values):
                per_k[k].append(probs[:, i].numpy())
        return {k: np.concatenate(v) for k, v in per_k.items()}

    small_probs = collect_rw_probs(small_mols, "small")
    medium_probs = collect_rw_probs(medium_mols, "medium")

    for k in k_values:
        e.log(f"  k={k}: small atoms={len(small_probs[k])}, medium atoms={len(medium_probs[k])}")

    # ------------------------------------------------------------------
    # 3. Compute bin boundaries according to the chosen mode
    # ------------------------------------------------------------------
    bin_mode = e.BIN_MODE
    num_bins = e.NUM_BINS

    if bin_mode == "quantile":
        boundaries = get_zinc_rw_boundaries(num_bins)
    elif bin_mode == "clipped":
        lo, hi = e.CLIP_RANGE
        step = (hi - lo) / num_bins
        boundaries = {k: [lo + step * i for i in range(1, num_bins)] for k in k_values}
    elif bin_mode == "uniform":
        step = 1.0 / num_bins
        boundaries = {k: [step * i for i in range(1, num_bins)] for k in k_values}
    else:
        raise ValueError(f"Unknown BIN_MODE: {bin_mode!r} (expected 'quantile', 'clipped', or 'uniform')")

    e.log(f"Bin boundaries (num_bins={num_bins}, mode={bin_mode}):")
    for k in k_values:
        b = boundaries.get(k, [])
        e.log(f"  k={k}: {[f'{v:.6f}' for v in b]}")

    # ------------------------------------------------------------------
    # 4. Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(k_values), figsize=(6 * len(k_values), 5))
    if len(k_values) == 1:
        axes = [axes]

    colors = {"small": "#2196F3", "medium": "#FF9800"}
    hist_bins = e.HISTOGRAM_BINS

    for ax, k in zip(axes, k_values):
        sp = small_probs[k]
        mp = medium_probs[k]

        # Histogram (normalised to density)
        ax.hist(
            sp, bins=hist_bins, range=(0, 1), density=True, alpha=0.4,
            color=colors["small"], label=f"small (<={e.SMALL_MAX_ATOMS} atoms)",
        )
        ax.hist(
            mp, bins=hist_bins, range=(0, 1), density=True, alpha=0.4,
            color=colors["medium"], label=f"medium ({e.MEDIUM_MIN_ATOMS}-{e.MEDIUM_MAX_ATOMS} atoms)",
        )

        # KDE overlay
        bw = e.KDE_BANDWIDTH
        if bw > 0:
            from scipy.stats import gaussian_kde
            xs = np.linspace(0, 1, 500)
            for arr, col in [(sp, colors["small"]), (mp, colors["medium"])]:
                # Filter out constant values (all-zero) which cause singular KDE
                if np.std(arr) > 1e-8:
                    kde = gaussian_kde(arr, bw_method=bw)
                    ax.plot(xs, kde(xs), color=col, linewidth=2)

        # Quantile bin boundaries
        b = boundaries.get(k, [])
        for i, threshold in enumerate(b):
            ax.axvline(
                threshold, color="#E53935", linewidth=1.5, linestyle="--",
                label="bin boundaries" if i == 0 else None,
            )

        ax.set_title(f"k = {k}", fontsize=14)
        ax.set_xlabel("RW return probability", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)

    fig.suptitle(
        f"RW Return Probability Distributions: Small vs Medium Molecules\n"
        f"(ZINC, {num_bins}-bin {bin_mode} boundaries)",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()

    e.track("rw_distributions", fig)
    e.log("Done - figure saved as 'rw_distributions'.")

    # ------------------------------------------------------------------
    # 5. Log summary statistics
    # ------------------------------------------------------------------
    e.log("\nSummary statistics (mean +/- std):")
    for k in k_values:
        sp, mp = small_probs[k], medium_probs[k]
        e.log(f"  k={k}:")
        e.log(f"    small:  {sp.mean():.4f} +/- {sp.std():.4f}  (min={sp.min():.4f}, max={sp.max():.4f})")
        e.log(f"    medium: {mp.mean():.4f} +/- {mp.std():.4f}  (min={mp.min():.4f}, max={mp.max():.4f})")

    # Per-bin atom fractions
    e.log(f"\nPer-bin atom fractions (num_bins={num_bins}, mode={bin_mode}):")
    for k in k_values:
        b = boundaries.get(k, [])
        edges = [0.0] + b + [1.0]
        e.log(f"  k={k}:")
        for arr, label in [(small_probs[k], "small"), (medium_probs[k], "medium")]:
            fractions = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                frac = np.mean((arr >= lo) & (arr < hi)) if lo < hi else np.mean(arr == lo)
                fractions.append(frac)
            # Last bin is inclusive on the right
            fractions[-1] = np.mean(arr >= edges[-2])
            frac_str = " | ".join(f"{f:.1%}" for f in fractions)
            e.log(f"    {label:>6s}: [{frac_str}]")


experiment.run_if_main()
