#!/usr/bin/env python
"""
Test Flow Matching — End-to-End Generation Evaluation.

Chains a trained FlowMatchingModel with a trained FlowEdgeDecoder to
perform full molecular graph generation:
  1. Sample HDC vectors from the flow model
  2. Decode node identities from order-0 embeddings
  3. Generate edges via the FlowEdgeDecoder
  4. Convert to RDKit molecules
  5. Compute comprehensive statistics and visualizations

Optionally compares generated distributions against a reference dataset.

Usage:
    # Quick test
    python test_flow_matching.py --__TESTING__ True

    # Full generation run
    python test_flow_matching.py \\
        --HDC_ENCODER_PATH /path/to/encoder.ckpt \\
        --FLOW_DECODER_PATH /path/to/decoder.ckpt \\
        --FLOW_MATCHING_PATH /path/to/flow_model.ckpt \\
        --NUM_SAMPLES 100

    # With reference dataset comparison
    python test_flow_matching.py \\
        --HDC_ENCODER_PATH /path/to/encoder.ckpt \\
        --FLOW_DECODER_PATH /path/to/decoder.ckpt \\
        --FLOW_MATCHING_PATH /path/to/flow_model.ckpt \\
        --COMPARE_DATASET True --DATASET zinc
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from rdkit import Chem
from rdkit.Chem import Crippen, QED, Draw
from rdkit.Contrib.SA_Score import sascorer
from torch_geometric.data import Data

from graph_hdc.hypernet import load_hypernet
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.models.flow_edge_decoder import (
    FlowEdgeDecoder,
    NODE_FEATURE_DIM,
    get_node_feature_bins,
    node_tuples_to_onehot,
)
from graph_hdc.models.flow_matching import FlowMatchingModel, build_condition
from graph_hdc.utils.experiment_helpers import (
    compute_hdc_distance,
    create_test_dummy_models,
    decode_nodes_from_hdc,
    draw_mol_or_error,
    get_canonical_smiles,
    is_valid_mol,
    pyg_to_mol,
)
from graph_hdc.utils.evaluator import (
    calculate_internal_diversity,
    rdkit_logp,
    rdkit_qed,
    rdkit_sa_score,
)


# =============================================================================
# PARAMETERS
# =============================================================================

# -----------------------------------------------------------------------------
# Model Paths
# -----------------------------------------------------------------------------

# :param HDC_ENCODER_PATH:
#     Path to saved HyperNet/MultiHyperNet encoder checkpoint (.ckpt).
HDC_ENCODER_PATH: str = "/media/ssd2/Programming/graph_hdc_public/experiments/decoding/results/train_flow_edge_decoder_streaming/GOOD/hypernet_encoder.ckpt"

# :param FLOW_DECODER_PATH:
#     Path to saved FlowEdgeDecoder checkpoint (.ckpt).
FLOW_DECODER_PATH: str = "/media/ssd2/Programming/graph_hdc_public/experiments/decoding/results/train_flow_edge_decoder_streaming/GOOD/last.ckpt"

# :param FLOW_MATCHING_PATH:
#     Path to saved FlowMatchingModel Lightning checkpoint (.ckpt).
FLOW_MATCHING_PATH: str = "/media/ssd2/Programming/graph_hdc_public/experiments/generation/results/train_flow_matching/debug/last.ckpt"

# -----------------------------------------------------------------------------
# Sampling Configuration
# -----------------------------------------------------------------------------

# :param NUM_SAMPLES:
#     Number of HDC vectors to sample from the flow model.
NUM_SAMPLES: int = 100

# :param NUM_REPETITIONS:
#     Best-of-N for FlowEdgeDecoder edge generation (batched).
NUM_REPETITIONS: int = 64

# :param SAMPLE_STEPS:
#     Number of ODE steps for FlowEdgeDecoder discrete flow sampling.
SAMPLE_STEPS: int = 50

# :param ETA:
#     Stochasticity parameter for FlowEdgeDecoder sampling.
ETA: float = 0.0

# :param OMEGA:
#     Target guidance strength for FlowEdgeDecoder sampling.
OMEGA: float = 0.0

# :param SAMPLE_TIME_DISTORTION:
#     Time distortion schedule for FlowEdgeDecoder sampling.
SAMPLE_TIME_DISTORTION: str = "polydec"

# :param FM_SAMPLE_STEPS:
#     Number of ODE steps for FlowMatchingModel sampling.
FM_SAMPLE_STEPS: int = 2_500

# -----------------------------------------------------------------------------
# Conditioning (optional)
# -----------------------------------------------------------------------------

# :param CONDITIONS:
#     List of condition names for conditional generation. None = unconditional.
CONDITIONS: Optional[List[str]] = None

# :param CONDITION_VALUES:
#     Dict mapping condition name to target value. Required if CONDITIONS is set.
CONDITION_VALUES: Optional[Dict[str, float]] = None

# -----------------------------------------------------------------------------
# Reference Dataset Comparison
# -----------------------------------------------------------------------------

# :param DATASET:
#     Dataset type for atom feature encoding ("zinc" or "qm9").
DATASET: str = "zinc"

# :param COMPARE_DATASET:
#     Whether to load reference dataset for distribution comparison.
COMPARE_DATASET: bool = True

# :param DATASET_SUBSAMPLE:
#     Number of reference molecules to subsample for comparison.
DATASET_SUBSAMPLE: int = 1000

# -----------------------------------------------------------------------------
# System Configuration
# -----------------------------------------------------------------------------

# :param DEVICE:
#     Device for model inference. "auto" prefers GPU.
DEVICE: str = "cuda"

# :param HDC_DEVICE:
#     Device for the HyperNet HDC encoder. Options: "auto" (prefer GPU),
#     "cpu", "cuda".
HDC_DEVICE: str = "cpu"

# :param PRUNE_DECODING_CODEBOOK:
#     Prune the RRWP codebook to only dataset-observed feature tuples before
#     node decoding. Reduces false positives from the overcomplete codebook.
PRUNE_DECODING_CODEBOOK: bool = True

# :param RUN_DIAGNOSTIC:
#     Whether to run the norm-decay diagnostic on real encoded molecules
#     before generation. Useful for debugging but adds startup time.
RUN_DIAGNOSTIC: bool = False

# :param PLOT_NORM_DECAY:
#     Save per-sample residual norm decay plots during node decoding.
PLOT_NORM_DECAY: bool = True

# :param SEED:
#     Random seed for reproducibility.
SEED: int = 42

# :param __DEBUG__:
#     Debug mode — reuses same output folder during development.
__DEBUG__: bool = True

# :param __TESTING__:
#     Testing mode — runs with dummy models and minimal iterations.
__TESTING__: bool = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def compute_mol_properties(mol: Chem.Mol) -> Dict[str, float]:
    """Compute molecular properties for a valid RDKit molecule."""
    props = {}
    try:
        props["logp"] = rdkit_logp(mol)
    except Exception:
        props["logp"] = float("nan")
    try:
        props["qed"] = rdkit_qed(mol)
    except Exception:
        props["qed"] = float("nan")
    try:
        props["sa_score"] = rdkit_sa_score(mol)
    except Exception:
        props["sa_score"] = float("nan")
    try:
        props["mol_weight"] = Chem.Descriptors.MolWt(mol)
    except Exception:
        try:
            props["mol_weight"] = Chem.rdMolDescriptors.CalcExactMolWt(mol)
        except Exception:
            props["mol_weight"] = float("nan")
    return props


def compute_structural_features(mol: Chem.Mol) -> Dict:
    """Compute structural features for a valid RDKit molecule."""
    feats = {}
    feats["num_atoms"] = mol.GetNumHeavyAtoms()
    feats["num_bonds"] = mol.GetNumBonds()

    # Atom types
    atom_types = []
    for atom in mol.GetAtoms():
        atom_types.append(atom.GetSymbol())
    feats["atom_types"] = atom_types

    # Bond types
    bond_type_map = {
        Chem.BondType.SINGLE: "Single",
        Chem.BondType.DOUBLE: "Double",
        Chem.BondType.TRIPLE: "Triple",
        Chem.BondType.AROMATIC: "Aromatic",
    }
    bond_types = []
    for bond in mol.GetBonds():
        bt = bond_type_map.get(bond.GetBondType(), "Other")
        bond_types.append(bt)
    feats["bond_types"] = bond_types

    # Rings
    ring_info = mol.GetRingInfo()
    feats["num_rings"] = ring_info.NumRings()
    ring_sizes = [len(r) for r in ring_info.AtomRings()]
    feats["ring_sizes"] = ring_sizes

    # Degree distribution
    degrees = [atom.GetDegree() for atom in mol.GetAtoms()]
    feats["degrees"] = degrees

    return feats


def collect_distribution_data(mols: List[Chem.Mol]) -> Dict:
    """Collect structural and property distributions from a list of valid molecules."""
    data = {
        "num_atoms": [],
        "num_bonds": [],
        "atom_types": Counter(),
        "bond_types": Counter(),
        "num_rings": [],
        "ring_sizes": Counter(),
        "degrees": Counter(),
        "logp": [],
        "qed": [],
        "sa_score": [],
        "mol_weight": [],
    }

    for mol in mols:
        sf = compute_structural_features(mol)
        data["num_atoms"].append(sf["num_atoms"])
        data["num_bonds"].append(sf["num_bonds"])
        for at in sf["atom_types"]:
            data["atom_types"][at] += 1
        for bt in sf["bond_types"]:
            data["bond_types"][bt] += 1
        data["num_rings"].append(sf["num_rings"])
        for rs in sf["ring_sizes"]:
            data["ring_sizes"][rs] += 1
        for d in sf["degrees"]:
            data["degrees"][d] += 1

        props = compute_mol_properties(mol)
        for key in ["logp", "qed", "sa_score", "mol_weight"]:
            val = props[key]
            if not math.isnan(val):
                data[key].append(val)

    return data


def plot_structural_distributions(
    gen_data: Dict,
    ref_data: Optional[Dict],
    save_path: Path,
) -> None:
    """Create 2x4 grid of structural distribution plots."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    gen_color = "#4C72B0"
    ref_color = "#DD8452"

    # 1. Atom count distribution (histogram)
    ax = axes[0, 0]
    bins = _shared_bins(gen_data["num_atoms"], ref_data["num_atoms"] if ref_data else None)
    ax.hist(gen_data["num_atoms"], bins=bins, alpha=0.7, color=gen_color, label="Generated", density=True)
    if ref_data:
        ax.hist(ref_data["num_atoms"], bins=bins, alpha=0.5, color=ref_color, label="Reference", density=True)
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Density")
    ax.set_title("Atom Count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Bond count distribution (histogram)
    ax = axes[0, 1]
    bins = _shared_bins(gen_data["num_bonds"], ref_data["num_bonds"] if ref_data else None)
    ax.hist(gen_data["num_bonds"], bins=bins, alpha=0.7, color=gen_color, label="Generated", density=True)
    if ref_data:
        ax.hist(ref_data["num_bonds"], bins=bins, alpha=0.5, color=ref_color, label="Reference", density=True)
    ax.set_xlabel("Number of Bonds")
    ax.set_ylabel("Density")
    ax.set_title("Bond Count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Atom type distribution (bar chart)
    ax = axes[0, 2]
    _plot_counter_comparison(ax, gen_data["atom_types"], ref_data["atom_types"] if ref_data else None,
                             gen_color, ref_color, "Atom Type", "Frequency")

    # 4. Bond type distribution (bar chart)
    ax = axes[0, 3]
    _plot_counter_comparison(ax, gen_data["bond_types"], ref_data["bond_types"] if ref_data else None,
                             gen_color, ref_color, "Bond Type", "Frequency")

    # 5. Ring size distribution (bar chart)
    ax = axes[1, 0]
    _plot_counter_comparison(ax, gen_data["ring_sizes"], ref_data["ring_sizes"] if ref_data else None,
                             gen_color, ref_color, "Ring Size", "Count")

    # 6. Rings per molecule (histogram)
    ax = axes[1, 1]
    bins = _shared_bins(gen_data["num_rings"], ref_data["num_rings"] if ref_data else None, integer=True)
    ax.hist(gen_data["num_rings"], bins=bins, alpha=0.7, color=gen_color, label="Generated", density=True)
    if ref_data:
        ax.hist(ref_data["num_rings"], bins=bins, alpha=0.5, color=ref_color, label="Reference", density=True)
    ax.set_xlabel("Rings per Molecule")
    ax.set_ylabel("Density")
    ax.set_title("Rings per Molecule")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 7. Degree distribution (bar chart)
    ax = axes[1, 2]
    _plot_counter_comparison(ax, gen_data["degrees"], ref_data["degrees"] if ref_data else None,
                             gen_color, ref_color, "Atom Degree", "Frequency")

    # 8. Molecular weight distribution (KDE-like histogram)
    ax = axes[1, 3]
    if gen_data["mol_weight"]:
        ax.hist(gen_data["mol_weight"], bins=30, alpha=0.7, color=gen_color, label="Generated", density=True)
    if ref_data and ref_data["mol_weight"]:
        ax.hist(ref_data["mol_weight"], bins=30, alpha=0.5, color=ref_color, label="Reference", density=True)
    ax.set_xlabel("Molecular Weight")
    ax.set_ylabel("Density")
    ax.set_title("Molecular Weight")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Structural Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_property_distributions(
    gen_data: Dict,
    ref_data: Optional[Dict],
    save_path: Path,
) -> None:
    """Create 1x3 grid of property distribution plots (KDE-like histograms)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    gen_color = "#4C72B0"
    ref_color = "#DD8452"

    properties = [
        ("logp", "LogP (Crippen)"),
        ("qed", "QED"),
        ("sa_score", "SA Score"),
    ]

    for i, (key, title) in enumerate(properties):
        ax = axes[i]
        gen_vals = gen_data[key]
        if gen_vals:
            ax.hist(gen_vals, bins=30, alpha=0.7, color=gen_color, label="Generated", density=True)
        if ref_data and ref_data[key]:
            ax.hist(ref_data[key], bins=30, alpha=0.5, color=ref_color, label="Reference", density=True)
        ax.set_xlabel(title)
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Property Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_generation_summary(
    validity: float,
    uniqueness: float,
    novelty: float,
    save_path: Path,
) -> None:
    """Create summary bar chart of generation quality metrics."""
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["Validity", "Uniqueness", "Novelty"]
    values = [validity, uniqueness, novelty]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=1.2)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold",
        )

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Generation Quality Metrics", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_norm_decay(
    norm_histories: List[List[float]],
    save_path: Path,
) -> None:
    """Plot residual norm decay during iterative node decoding.

    Each curve shows how the residual norm decreases as nodes are
    successively subtracted from the HDC vector for one sample.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    single = len(norm_histories) == 1

    for i, norms in enumerate(norm_histories):
        if not norms:
            continue
        ax.plot(
            range(len(norms)),
            norms,
            alpha=1.0 if single else max(0.15, 1.0 / len(norm_histories) * 5),
            linewidth=1.5 if single else 1.0,
            color="#4C72B0",
            marker="o" if single else None,
            markersize=4 if single else None,
        )

    # Overlay the median curve (only when multiple samples)
    if not single:
        max_len = max((len(n) for n in norm_histories if n), default=0)
        if max_len > 0:
            padded = np.full((len(norm_histories), max_len), np.nan)
            for i, norms in enumerate(norm_histories):
                if norms:
                    padded[i, : len(norms)] = norms
            median_curve = np.nanmedian(padded, axis=0)
            ax.plot(
                range(max_len),
                median_curve,
                color="#C44E52",
                linewidth=2.0,
                label="Median",
            )

    ax.set_xlabel("Decoding Iteration")
    ax.set_ylabel("Residual Norm")
    ax.set_title("Node Decoding — Residual Norm Decay")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _shared_bins(data1: List, data2: Optional[List], integer: bool = False) -> np.ndarray:
    """Compute shared bins for overlaid histograms."""
    all_data = list(data1)
    if data2:
        all_data.extend(data2)
    if not all_data:
        return np.arange(0, 2)
    lo, hi = min(all_data), max(all_data)
    if integer:
        return np.arange(lo - 0.5, hi + 1.5, 1)
    n_bins = min(30, max(10, int(hi - lo) + 1))
    return np.linspace(lo, hi, n_bins + 1)


def _plot_counter_comparison(
    ax,
    gen_counter: Counter,
    ref_counter: Optional[Counter],
    gen_color: str,
    ref_color: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Plot paired bar chart from two Counter objects."""
    all_keys = sorted(set(gen_counter.keys()) | (set(ref_counter.keys()) if ref_counter else set()))
    if not all_keys:
        ax.set_title(xlabel)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Normalize to frequencies
    gen_total = sum(gen_counter.values()) or 1
    gen_freqs = [gen_counter.get(k, 0) / gen_total for k in all_keys]

    x = np.arange(len(all_keys))
    width = 0.35 if ref_counter else 0.7

    if ref_counter:
        ref_total = sum(ref_counter.values()) or 1
        ref_freqs = [ref_counter.get(k, 0) / ref_total for k in all_keys]
        ax.bar(x - width / 2, gen_freqs, width, alpha=0.7, color=gen_color, label="Generated")
        ax.bar(x + width / 2, ref_freqs, width, alpha=0.7, color=ref_color, label="Reference")
    else:
        ax.bar(x, gen_freqs, width, alpha=0.7, color=gen_color, label="Generated")

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in all_keys], fontsize=8, rotation=45)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(xlabel)
    ax.legend(fontsize=8)
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
    """Test Flow Matching — End-to-End Generation Evaluation."""

    e.log("=" * 60)
    e.log("Flow Matching Generation Evaluation")
    e.log("=" * 60)
    e.log_parameters()

    torch.manual_seed(e.SEED)

    # Device setup
    if e.DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(e.DEVICE)
    if e.HDC_DEVICE == "auto":
        hdc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        hdc_device = torch.device(e.HDC_DEVICE)
    e.log(f"Decoder device: {device}")
    e.log(f"HyperNet device: {hdc_device}")

    # =========================================================================
    # Step 1: Load Models
    # =========================================================================

    hypernet, decoder, flow_model, base_hdc_dim = e.apply_hook(
        "load_models",
        device=device,
    )

    hypernet.to(hdc_device)
    hypernet.eval()
    e.log(str(hypernet))

    # Prune RRWP codebook to observed features for better node decoding
    if e.PRUNE_DECODING_CODEBOOK and hasattr(hypernet, '_limit_full_codebook'):
        from graph_hdc.datasets.utils import scan_node_features_with_rw

        rw_cfg = hypernet.rw_config

        @e.cache.cached(
            name="observed_rw_features",
            scope=lambda _e: (
                "rw_scan",
                _e.DATASET,
                str(rw_cfg.k_values),
                str(rw_cfg.num_bins),
            ),
        )
        def scan_observed_features():
            return scan_node_features_with_rw(
                dataset_name=e.DATASET,
                rw_config=rw_cfg,
            )

        observed = scan_observed_features()
        size_before = hypernet.nodes_codebook_full.shape[0]
        hypernet._limit_full_codebook(observed)
        size_after = hypernet.nodes_codebook_full.shape[0]
        e.log(f"Pruned RRWP codebook: {size_before:,} -> {size_after:,} entries")

    decoder.to(device)
    decoder.eval()
    flow_model.to(device)
    flow_model.eval()

    e["model/base_hdc_dim"] = base_hdc_dim
    e["model/concat_hdc_dim"] = 2 * base_hdc_dim

    # =========================================================================
    # Step 1b: Diagnostic — norm decay on real encoded molecules
    # =========================================================================

    if e.RUN_DIAGNOSTIC:
        e.log("\n" + "=" * 60)
        e.log("Diagnostic: norm decay on real encoded molecules")
        e.log("=" * 60)

        from graph_hdc.datasets.utils import get_split
        from torch_geometric.data import Batch as PyGBatch

        ref_dataset = get_split("train", dataset=e.DATASET)
        n_diag = min(20, len(ref_dataset))
        diag_indices = torch.randperm(len(ref_dataset))[:n_diag].tolist()
        diag_data = [ref_dataset[i] for i in diag_indices]

        ref_norm_histories: List[List[float]] = []
        with torch.no_grad():
            # Augment with RW features if the hypernet expects them
            if hasattr(hypernet, "rw_config") and hypernet.rw_config.enabled:
                from graph_hdc.utils.rw_features import augment_data_with_rw
                diag_data = [
                    augment_data_with_rw(
                        d.clone(),
                        k_values=hypernet.rw_config.k_values,
                        num_bins=hypernet.rw_config.num_bins,
                        bin_boundaries=hypernet.rw_config.bin_boundaries,
                        clip_range=hypernet.rw_config.clip_range,
                    )
                    for d in diag_data
                ]

            batch = PyGBatch.from_data_list(diag_data).to(hdc_device)
            enc_out = hypernet.forward(batch)
            node_terms = enc_out["node_terms"].cpu()
            graph_emb = enc_out["graph_embedding"].cpu()

            for i in range(n_diag):
                hdc_vec = torch.cat([node_terms[i], graph_emb[i]], dim=-1)
                _, _, norms, _ = decode_nodes_from_hdc(
                    hypernet, hdc_vec.unsqueeze(0), base_hdc_dim, debug=True
                )
                ref_norm_histories.append(norms)
                smiles = diag_data[i].smiles if hasattr(diag_data[i], "smiles") else "?"
                e.log(
                    f"  Real mol {i}: {smiles:30s}  "
                    f"init_norm={norms[0]:.2f}  final_norm={norms[-1]:.2f}  "
                    f"nodes_decoded={len(norms)-1}"
                )

        diag_path = Path(e.path) / "norm_decay_reference.png"
        plot_norm_decay(ref_norm_histories, diag_path)
        e.log(f"  Reference norm decay saved: {diag_path}")
    else:
        e.log("\nDiagnostic skipped (RUN_DIAGNOSTIC=False)")

    # =========================================================================
    # Step 2: Sample HDC Vectors
    # =========================================================================

    e.log("\n" + "=" * 60)
    e.log(f"Sampling {e.NUM_SAMPLES} HDC vectors from flow model...")
    e.log("=" * 60)

    # Build condition tensor if conditioning is requested
    condition_tensor = None
    if e.CONDITIONS is not None and e.CONDITION_VALUES is not None:
        multi_cond = build_condition(e.CONDITIONS)
        single_cond = multi_cond.sample_condition(e.CONDITION_VALUES)
        condition_tensor = single_cond.unsqueeze(0).expand(e.NUM_SAMPLES, -1).to(device)
        e.log(f"Conditioning on: {e.CONDITIONS} = {e.CONDITION_VALUES}")

    sample_start = time.time()
    with torch.no_grad():
        hdc_vectors = flow_model.sample(
            num_samples=e.NUM_SAMPLES,
            condition=condition_tensor,
            num_steps=e.FM_SAMPLE_STEPS,
            device=device,
        )
    sample_time = time.time() - sample_start
    e.log(f"Sampled {hdc_vectors.shape[0]} vectors of dim {hdc_vectors.shape[1]} in {sample_time:.2f}s")

    # =========================================================================
    # Step 3: Decode Each HDC Vector → Molecular Graph
    # =========================================================================

    e.log("\n" + "=" * 60)
    e.log("Decoding HDC vectors to molecules...")
    e.log("=" * 60)

    molecules_dir = Path(e.path) / "molecules"
    molecules_dir.mkdir(exist_ok=True)

    results = []
    valid_mols = []
    all_smiles = []
    all_norm_histories: List[List[float]] = []
    decode_start = time.time()

    for idx in range(hdc_vectors.shape[0]):
        hdc_vector = hdc_vectors[idx].cpu()

        # 3a. Decode nodes from order_0 part (with debug info for norm plots)
        node_tuples, num_nodes, norms_history, _ = decode_nodes_from_hdc(
            hypernet, hdc_vector.unsqueeze(0), base_hdc_dim, debug=True
        )
        all_norm_histories.append(norms_history)

        if num_nodes == 0:
            e.log(f"  Sample {idx + 1}/{e.NUM_SAMPLES}: No nodes decoded, skipping")
            results.append({
                "idx": idx,
                "smiles": None,
                "status": "no_nodes",
                "num_nodes": 0,
            })
            continue

        # 3b. Convert node tuples to one-hot features
        feature_bins = get_node_feature_bins(hypernet.rw_config)
        node_features = node_tuples_to_onehot(node_tuples, device=device, feature_bins=feature_bins).unsqueeze(0)
        node_mask = torch.ones(1, num_nodes, dtype=torch.bool, device=device)
        hdc_vec_device = hdc_vector.unsqueeze(0).to(device)

        # 3c. Generate edges
        num_reps = e.NUM_REPETITIONS

        sample_kwargs = dict(
            sample_steps=e.SAMPLE_STEPS,
            eta=e.ETA,
            omega=e.OMEGA,
            time_distortion=e.SAMPLE_TIME_DISTORTION,
            show_progress=False,
            device=device,
        )

        with torch.no_grad():
            if num_reps > 1:
                def score_fn(s):
                    return compute_hdc_distance(
                        s, hdc_vec_device, base_hdc_dim,
                        hypernet, device, dataset=e.DATASET,
                    )

                best_sample, best_distance, _avg_distance = decoder.sample_best_of_n(
                    hdc_vectors=hdc_vec_device,
                    node_features=node_features,
                    node_mask=node_mask,
                    num_repetitions=num_reps,
                    score_fn=score_fn,
                    **sample_kwargs,
                )
                generated_data = best_sample
            else:
                generated_samples = decoder.sample(
                    hdc_vectors=hdc_vec_device,
                    node_features=node_features,
                    node_mask=node_mask,
                    **sample_kwargs,
                )
                generated_data = generated_samples[0]

        # 3d. Convert to RDKit molecule
        mol = pyg_to_mol(generated_data)
        smiles = get_canonical_smiles(mol)
        valid = is_valid_mol(mol)

        # 3e. Compute HDC distance to original embedding
        hdc_dist = compute_hdc_distance(
            generated_data, hdc_vec_device, base_hdc_dim,
            hypernet, device, dataset=e.DATASET,
        )

        # 3f. Save molecule PNG with SMILES title and HDC distance
        mol_img = draw_mol_or_error(mol, size=(300, 300))
        title = f"{smiles or 'N/A'}  |  HDC dist: {hdc_dist:.4f}"
        # Create composite image with title bar
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except OSError:
            font = ImageFont.load_default()
        title_h = 30
        composite = Image.new("RGB", (mol_img.width, mol_img.height + title_h), "white")
        draw_ctx = ImageDraw.Draw(composite)
        draw_ctx.text((5, 5), title, fill="black", font=font)
        composite.paste(mol_img, (0, title_h))
        composite.save(molecules_dir / f"molecule_{idx:04d}.png")

        if e.PLOT_NORM_DECAY and norms_history:
            plot_norm_decay([norms_history], molecules_dir / f"norm_decay_{idx:04d}.png")

        # 3g. Collect results
        result = {
            "idx": idx,
            "smiles": smiles,
            "status": "valid" if valid else "invalid",
            "num_nodes": num_nodes,
            "hdc_distance": hdc_dist,
        }

        if valid and mol is not None:
            valid_mols.append(mol)
            all_smiles.append(smiles)
            props = compute_mol_properties(mol)
            result.update(props)
            result["num_atoms"] = mol.GetNumHeavyAtoms()
            result["num_bonds"] = mol.GetNumBonds()

        results.append(result)

        if (idx + 1) % 10 == 0 or idx == 0:
            e.log(f"  Sample {idx + 1}/{e.NUM_SAMPLES}: {smiles or 'N/A'} [{result['status']}]")

    decode_time = time.time() - decode_start
    e.log(f"\nDecoding completed in {decode_time:.2f}s")

    # =========================================================================
    # Step 4: Compute Statistics
    # =========================================================================

    e.log("\n" + "=" * 60)
    e.log("Computing statistics...")
    e.log("=" * 60)

    n_total = len(results)
    n_valid = len(valid_mols)
    n_no_nodes = sum(1 for r in results if r["status"] == "no_nodes")
    n_invalid = sum(1 for r in results if r["status"] == "invalid")

    # Validity rate
    validity_rate = 100.0 * n_valid / n_total if n_total > 0 else 0.0

    # Uniqueness
    unique_smiles = set(s for s in all_smiles if s is not None)
    uniqueness_rate = 100.0 * len(unique_smiles) / n_valid if n_valid > 0 else 0.0

    # Novelty (only if reference dataset loaded)
    training_smiles = set()
    novelty_rate = 0.0

    # Internal diversity
    diversity = 0.0
    if len(valid_mols) >= 2:
        diversity = calculate_internal_diversity(valid_mols)

    e.log(f"  Total samples: {n_total}")
    e.log(f"  Valid: {n_valid} ({validity_rate:.1f}%)")
    e.log(f"  Invalid: {n_invalid}")
    e.log(f"  No nodes decoded: {n_no_nodes}")
    e.log(f"  Unique: {len(unique_smiles)} ({uniqueness_rate:.1f}%)")
    e.log(f"  Internal diversity: {diversity:.1f}%")

    # =========================================================================
    # Step 5: Reference Dataset Comparison (optional)
    # =========================================================================

    ref_dist_data = None
    if e.COMPARE_DATASET and not e.__TESTING__:
        e.log("\n" + "=" * 60)
        e.log("Loading reference dataset for comparison...")
        e.log("=" * 60)

        from graph_hdc.datasets.utils import get_split

        ref_dataset = get_split("train", dataset=e.DATASET)
        n_ref = min(e.DATASET_SUBSAMPLE, len(ref_dataset))

        # Subsample
        indices = torch.randperm(len(ref_dataset))[:n_ref].tolist()
        ref_mols = []
        for i in indices:
            data = ref_dataset[i]
            mol = Chem.MolFromSmiles(data.smiles)
            if mol is not None and is_valid_mol(mol):
                ref_mols.append(mol)
                training_smiles.add(get_canonical_smiles(mol))

        e.log(f"  Reference molecules: {len(ref_mols)}")
        ref_dist_data = collect_distribution_data(ref_mols)

        # Recompute novelty with training smiles
        novel_smiles = unique_smiles - training_smiles
        novelty_rate = 100.0 * len(novel_smiles) / n_valid if n_valid > 0 else 0.0
        e.log(f"  Novel: {len(novel_smiles)} ({novelty_rate:.1f}%)")

    # =========================================================================
    # Step 6: Generate Plots
    # =========================================================================

    e.log("\n" + "=" * 60)
    e.log("Generating plots...")
    e.log("=" * 60)

    gen_dist_data = collect_distribution_data(valid_mols)

    # Figure 1: Structural distributions
    structural_path = Path(e.path) / "structural_distributions.png"
    plot_structural_distributions(gen_dist_data, ref_dist_data, structural_path)
    e.log(f"  Structural distributions saved: {structural_path}")

    # Figure 2: Property distributions
    property_path = Path(e.path) / "property_distributions.png"
    plot_property_distributions(gen_dist_data, ref_dist_data, property_path)
    e.log(f"  Property distributions saved: {property_path}")

    # Figure 3: Summary bar chart
    summary_path = Path(e.path) / "generation_summary.png"
    plot_generation_summary(validity_rate, uniqueness_rate, novelty_rate, summary_path)
    e.log(f"  Summary chart saved: {summary_path}")

    # Figure 4: Node decoding residual norm decay (aggregate)
    if e.PLOT_NORM_DECAY and all_norm_histories:
        norm_decay_path = Path(e.path) / "norm_decay.png"
        plot_norm_decay(all_norm_histories, norm_decay_path)
        e.log(f"  Norm decay plot saved: {norm_decay_path}")

    # =========================================================================
    # Step 7: Save Results
    # =========================================================================

    e.log("\n" + "=" * 60)
    e.log("SUMMARY")
    e.log("=" * 60)

    e["results/validity"] = validity_rate
    e["results/uniqueness"] = uniqueness_rate
    e["results/novelty"] = novelty_rate
    e["results/diversity"] = diversity
    e["results/n_total"] = n_total
    e["results/n_valid"] = n_valid
    e["results/n_invalid"] = n_invalid
    e["results/n_no_nodes"] = n_no_nodes
    e["results/n_unique"] = len(unique_smiles)
    e["results/sample_time_seconds"] = sample_time
    e["results/decode_time_seconds"] = decode_time

    # Property statistics
    if gen_dist_data["logp"]:
        e["results/logp_mean"] = float(np.mean(gen_dist_data["logp"]))
        e["results/logp_std"] = float(np.std(gen_dist_data["logp"]))
    if gen_dist_data["qed"]:
        e["results/qed_mean"] = float(np.mean(gen_dist_data["qed"]))
        e["results/qed_std"] = float(np.std(gen_dist_data["qed"]))
    if gen_dist_data["sa_score"]:
        e["results/sa_mean"] = float(np.mean(gen_dist_data["sa_score"]))
        e["results/sa_std"] = float(np.std(gen_dist_data["sa_score"]))

    summary_dict = {
        "config": {
            "hdc_encoder_path": e.HDC_ENCODER_PATH,
            "flow_decoder_path": e.FLOW_DECODER_PATH,
            "flow_matching_path": e.FLOW_MATCHING_PATH,
            "dataset": e.DATASET,
            "num_samples": e.NUM_SAMPLES,
            "num_repetitions": e.NUM_REPETITIONS,
            "sample_steps": e.SAMPLE_STEPS,
            "eta": e.ETA,
            "omega": e.OMEGA,
            "fm_sample_steps": e.FM_SAMPLE_STEPS,
            "conditions": e.CONDITIONS,
            "condition_values": e.CONDITION_VALUES,
            "seed": e.SEED,
        },
        "summary": {
            "validity_pct": validity_rate,
            "uniqueness_pct": uniqueness_rate,
            "novelty_pct": novelty_rate,
            "diversity_pct": diversity,
            "n_total": n_total,
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "n_no_nodes": n_no_nodes,
            "n_unique": len(unique_smiles),
            "sample_time_seconds": sample_time,
            "decode_time_seconds": decode_time,
        },
        "properties": {
            "logp_mean": float(np.mean(gen_dist_data["logp"])) if gen_dist_data["logp"] else None,
            "logp_std": float(np.std(gen_dist_data["logp"])) if gen_dist_data["logp"] else None,
            "qed_mean": float(np.mean(gen_dist_data["qed"])) if gen_dist_data["qed"] else None,
            "qed_std": float(np.std(gen_dist_data["qed"])) if gen_dist_data["qed"] else None,
            "sa_mean": float(np.mean(gen_dist_data["sa_score"])) if gen_dist_data["sa_score"] else None,
            "sa_std": float(np.std(gen_dist_data["sa_score"])) if gen_dist_data["sa_score"] else None,
        },
        "results": results,
    }

    e.commit_json("generation_results.json", summary_dict)

    e.log(f"\n  Validity:    {validity_rate:.1f}%")
    e.log(f"  Uniqueness:  {uniqueness_rate:.1f}%")
    e.log(f"  Novelty:     {novelty_rate:.1f}%")
    e.log(f"  Diversity:   {diversity:.1f}%")
    e.log(f"  Sample time: {sample_time:.2f}s")
    e.log(f"  Decode time: {decode_time:.2f}s")
    e.log(f"\n  Molecule PNGs saved to: {molecules_dir}")
    e.log("\nExperiment completed!")


# =============================================================================
# HOOKS
# =============================================================================


@experiment.hook("load_models", default=True)
def load_models(
    e: Experiment,
    device: torch.device,
) -> Tuple[HyperNet, FlowEdgeDecoder, FlowMatchingModel, int]:
    """
    Load HyperNet encoder, FlowEdgeDecoder, and FlowMatchingModel.

    Returns:
        Tuple of (hypernet, decoder, flow_model, base_hdc_dim).
    """
    e.log("\nLoading models...")

    if e.__TESTING__:
        e.log("TESTING MODE: Creating dummy models...")
        test_device = torch.device("cpu")
        hypernet, decoder, base_hdc_dim = create_test_dummy_models(test_device)

        # Create matching FlowMatchingModel
        data_dim = 2 * base_hdc_dim
        flow_model = FlowMatchingModel(
            data_dim=data_dim,
            hidden_dim=64,
            num_blocks=2,
            time_embed_dim=32,
            condition_dim=0,
            default_sample_steps=10,
        )

        return hypernet, decoder, flow_model, base_hdc_dim

    # Real model loading
    if not e.HDC_ENCODER_PATH:
        raise ValueError("HDC_ENCODER_PATH is required")
    if not e.FLOW_DECODER_PATH:
        raise ValueError("FLOW_DECODER_PATH is required")
    if not e.FLOW_MATCHING_PATH:
        raise ValueError("FLOW_MATCHING_PATH is required")

    e.log(f"Loading HyperNet from: {e.HDC_ENCODER_PATH}")
    hypernet = load_hypernet(e.HDC_ENCODER_PATH, device="cpu")
    base_hdc_dim = hypernet.hv_dim
    e.log(f"  HyperNet hv_dim: {base_hdc_dim}")

    e.log(f"Loading FlowEdgeDecoder from: {e.FLOW_DECODER_PATH}")
    decoder = FlowEdgeDecoder.load(e.FLOW_DECODER_PATH, device=device)
    e.log(f"  FlowEdgeDecoder hdc_dim: {decoder.hdc_dim}")

    e.log(f"Loading FlowMatchingModel from: {e.FLOW_MATCHING_PATH}")
    flow_model = FlowMatchingModel.load_from_checkpoint(
        e.FLOW_MATCHING_PATH,
        map_location=device,
    )
    e.log(f"  FlowMatchingModel data_dim: {flow_model.data_dim}")
    e.log(f"  FlowMatchingModel condition_dim: {flow_model.condition_dim}")

    return hypernet, decoder, flow_model, base_hdc_dim


# =============================================================================
# Testing Mode
# =============================================================================


@experiment.testing
def testing(e: Experiment) -> None:
    """Quick test mode with reduced parameters."""
    e.NUM_SAMPLES = 5
    e.NUM_REPETITIONS = 1
    e.SAMPLE_STEPS = 10
    e.FM_SAMPLE_STEPS = 10
    e.COMPARE_DATASET = False
    e.DATASET = "zinc"


# =============================================================================
# Entry Point
# =============================================================================

experiment.run_if_main()
