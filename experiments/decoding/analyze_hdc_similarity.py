#!/usr/bin/env python
"""
Analyze HDC Embedding Similarity Between Molecular Variants.

This experiment computes cosine distances between HDC embeddings of a reference
molecule and its structural variants. Multiple encoder initializations (seeds)
are used to assess embedding stability across different random codebook generations.

For each encoder seed:
1. Encode the original molecule to a hypervector
2. Encode each variant molecule to a hypervector
3. Compute cosine distance from original to each variant

Outputs:
- Raw distance values for each (encoder, variant) pair
- Aggregated statistics (mean, std) per variant
- Summary bar chart with error bars
- Per-variant figures showing molecules side-by-side with distance annotations

Usage:
    python analyze_hdc_similarity.py

    # Override default molecules
    python analyze_hdc_similarity.py --ORIGINAL_SMILES "c1ccccc1" --NUM_ENCODERS 20
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from rdkit import Chem
from rdkit.Chem import Draw

from graph_hdc.datasets.qm9_smiles import mol_to_data
from graph_hdc.hypernet.configs import (
    DSHDCConfig,
    FeatureConfig,
    Features,
    IndexRange,
)
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.hypernet.feature_encoders import CombinatoricIntegerEncoder
from graph_hdc.hypernet.types import VSAModel

# =============================================================================
# PARAMETERS
# =============================================================================

# :param ORIGINAL_SMILES:
#     Reference molecule in SMILES format. All variant molecules are compared against
#     this original. Must be a valid SMILES string that can be parsed by RDKit and
#     contain only atoms supported by the QM9 dataset (C, N, O, F). Default is aspirin.
ORIGINAL_SMILES: str = "CC(=O)Oc1ccccc1C(=O)O"

# :param VARIANTS:
#     Dictionary mapping human-readable variant names (keys) to SMILES strings (values).
#     Each variant will be encoded and compared against the original molecule. Invalid
#     SMILES strings are automatically skipped with a warning.
VARIANTS: dict[str, str] = {
    "salicylic_acid": "O=C(O)c1ccccc1O",  # Aspirin without acetyl group
    "benzoic_acid": "O=C(O)c1ccccc1",  # Simple benzoic acid
    "phenol": "Oc1ccccc1",  # Just phenol ring
    "acetylsalicylic_methyl_ester": "CC(=O)Oc1ccccc1C(=O)OC",  # Methyl ester variant
    "benzene": "c1ccccc1",  # Core benzene ring
}

# :param NUM_ENCODERS:
#     Number of different HyperNet encoder initializations to use. Each encoder uses
#     a different random seed for codebook generation, allowing assessment of embedding
#     stability. Higher values give more reliable statistics but increase runtime.
NUM_ENCODERS: int = 10

# :param BASE_SEED:
#     Starting seed value for encoder initialization. The experiment will use seeds
#     BASE_SEED, BASE_SEED+1, ..., BASE_SEED+NUM_ENCODERS-1. This ensures reproducibility
#     while allowing multiple independent encoder initializations.
BASE_SEED: int = 42

# :param HDC_DIM:
#     Hypervector dimension for the HDC encoder. Larger dimensions generally improve
#     encoding fidelity but increase memory usage and computation time. Common values
#     are 256, 512, 1024, or 2048.
HDC_DIM: int = 1024

# :param DEPTH:
#     Message passing depth for the HyperNet encoder. Controls how many hops of
#     neighborhood information are aggregated into node embeddings. Depth 1 captures
#     immediate neighbors, depth 2 captures 2-hop neighborhoods, etc.
DEPTH: int = 7

# Debug/Testing modes
__DEBUG__: bool = True
__TESTING__: bool = False


# =============================================================================
# Helper Functions
# =============================================================================


def create_hdc_config(
    hv_dim: int,
    depth: int,
    seed: int,
    device: str = "cpu",
) -> DSHDCConfig:
    """
    Create a HyperNet configuration for QM9-style molecules.

    Args:
        hv_dim: Hypervector dimension
        depth: Message passing depth
        seed: Random seed for codebook generation
        device: Device string

    Returns:
        DSHDCConfig for HyperNet initialization
    """
    # QM9: 4 atom types, 5 degrees, 3 charges, 5 hydrogens
    node_feature_config = FeatureConfig(
        count=math.prod([4, 5, 3, 5]),  # 300 combinations
        encoder_cls=CombinatoricIntegerEncoder,
        index_range=IndexRange((0, 4)),
        bins=[4, 5, 3, 5],
    )

    return DSHDCConfig(
        name=f"QM9_HRR_{hv_dim}_depth{depth}_seed{seed}",
        hv_dim=hv_dim,
        vsa=VSAModel.HRR,
        base_dataset="qm9",
        hypernet_depth=depth,
        device=device,
        seed=seed,
        normalize=True,
        dtype="float64",
        node_feature_configs=OrderedDict([
            (Features.NODE_FEATURES, node_feature_config),
        ]),
    )


def smiles_to_embedding(hypernet: HyperNet, smiles: str, device: torch.device) -> torch.Tensor:
    """
    Convert SMILES to HDC graph embedding.

    Args:
        hypernet: HyperNet encoder instance
        smiles: SMILES string
        device: Torch device

    Returns:
        Graph embedding tensor [1, hv_dim]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    data = mol_to_data(mol)
    data = data.to(device)

    # Add batch index for single molecule
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    with torch.no_grad():
        output = hypernet.forward(data, normalize=True)

    return output["graph_embedding"]


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute cosine distance between two vectors.

    Cosine distance = 1 - cosine_similarity

    Args:
        a: First vector [1, dim] or [dim]
        b: Second vector [1, dim] or [dim]

    Returns:
        Cosine distance (0 = identical, 2 = opposite)
    """
    a = a.flatten()
    b = b.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
    return (1.0 - cos_sim.item())


def mol_to_image(smiles: str, size: tuple[int, int] = (300, 300)) -> np.ndarray:
    """
    Convert SMILES to molecule image as numpy array.

    Args:
        smiles: SMILES string
        size: Image size (width, height)

    Returns:
        RGB image as numpy array
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return blank image for invalid SMILES
        return np.ones((*size, 3), dtype=np.uint8) * 255

    img = Draw.MolToImage(mol, size=size)
    return np.array(img)


def create_summary_bar_chart(
    variant_names: list[str],
    means: list[float],
    stds: list[float],
    output_path: str,
) -> plt.Figure:
    """
    Create bar chart showing distances to all variants with error bars.

    Args:
        variant_names: Names of variants
        means: Mean distances per variant
        stds: Standard deviations per variant
        output_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(variant_names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", edgecolor="black")

    ax.set_xlabel("Variant", fontsize=12)
    ax.set_ylabel("Cosine Distance", fontsize=12)
    ax.set_title("HDC Embedding Distance: Original to Variants", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(variant_names, rotation=45, ha="right")

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(
            f"{mean:.3f}\n(±{std:.3f})",
            xy=(bar.get_x() + bar.get_width() / 2, height + std),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0, max(means) + max(stds) + 0.1)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def create_variant_comparison_figure(
    original_smiles: str,
    variant_name: str,
    variant_smiles: str,
    mean_distance: float,
    std_distance: float,
    output_path: str,
) -> plt.Figure:
    """
    Create figure showing original and variant molecules side-by-side.

    Args:
        original_smiles: Original molecule SMILES
        variant_name: Name of the variant
        variant_smiles: Variant molecule SMILES
        mean_distance: Mean cosine distance
        std_distance: Standard deviation of distance
        output_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original molecule
    orig_img = mol_to_image(original_smiles, size=(400, 400))
    axes[0].imshow(orig_img)
    axes[0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Variant molecule
    var_img = mol_to_image(variant_smiles, size=(400, 400))
    axes[1].imshow(var_img)
    axes[1].set_title(f"Variant: {variant_name}", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Add distance annotation
    fig.suptitle(
        f"Cosine Distance: {mean_distance:.4f} (±{std_distance:.4f})",
        fontsize=14,
        fontweight="bold",
        y=0.02,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# EXPERIMENT
# =============================================================================


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    """Main experiment: analyze HDC similarity between molecular variants."""
    e.log("=" * 60)
    e.log("HDC Embedding Similarity Analysis")
    e.log("=" * 60)

    # Store configuration
    e["config/original_smiles"] = e.ORIGINAL_SMILES
    e["config/variants"] = e.VARIANTS
    e["config/num_encoders"] = e.NUM_ENCODERS
    e["config/base_seed"] = e.BASE_SEED
    e["config/hdc_dim"] = e.HDC_DIM
    e["config/depth"] = e.DEPTH

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e.log(f"Using device: {device}")
    e["config/device"] = str(device)

    # Validate SMILES
    e.log("\nValidating molecules...")
    orig_mol = Chem.MolFromSmiles(e.ORIGINAL_SMILES)
    if orig_mol is None:
        raise ValueError(f"Invalid original SMILES: {e.ORIGINAL_SMILES}")
    e.log(f"  Original: {e.ORIGINAL_SMILES} (valid)")

    valid_variants = {}
    for name, smiles in e.VARIANTS.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            e.log(f"  {name}: {smiles} (INVALID - skipping)")
        else:
            valid_variants[name] = smiles
            e.log(f"  {name}: {smiles} (valid)")

    if not valid_variants:
        raise ValueError("No valid variant SMILES provided")

    e["data/valid_variants"] = list(valid_variants.keys())

    # Initialize storage for results
    # raw_distances[variant_name][seed] = distance
    raw_distances: dict[str, dict[int, float]] = {name: {} for name in valid_variants}

    # Process each encoder seed
    e.log(f"\nProcessing {e.NUM_ENCODERS} encoder initializations...")
    seeds = list(range(e.BASE_SEED, e.BASE_SEED + e.NUM_ENCODERS))

    for seed in seeds:
        e.log(f"\n  Seed {seed}:")

        # Create encoder with this seed
        config = create_hdc_config(
            hv_dim=e.HDC_DIM,
            depth=e.DEPTH,
            seed=seed,
            device=str(device),
        )
        hypernet = HyperNet(config)
        hypernet.to(device)
        hypernet.eval()

        # Encode original molecule
        orig_embedding = smiles_to_embedding(hypernet, e.ORIGINAL_SMILES, device)

        # Compute distance to each variant
        for var_name, var_smiles in valid_variants.items():
            var_embedding = smiles_to_embedding(hypernet, var_smiles, device)
            dist = cosine_distance(orig_embedding, var_embedding)
            raw_distances[var_name][seed] = dist
            e.log(f"    {var_name}: {dist:.4f}")

        # Track per-seed metrics (use underscores for track keys)
        for var_name in valid_variants:
            e.track(f"distance_{var_name}", raw_distances[var_name][seed])

    # Compute aggregated statistics
    e.log("\n" + "=" * 60)
    e.log("AGGREGATED RESULTS")
    e.log("=" * 60)

    aggregated_stats: dict[str, dict[str, float]] = {}
    variant_names = []
    means = []
    stds = []

    for var_name in valid_variants:
        distances = list(raw_distances[var_name].values())
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        aggregated_stats[var_name] = {
            "mean": float(mean_dist),
            "std": float(std_dist),
            "min": float(min_dist),
            "max": float(max_dist),
        }

        variant_names.append(var_name)
        means.append(mean_dist)
        stds.append(std_dist)

        e.log(f"\n  {var_name}:")
        e.log(f"    Mean distance: {mean_dist:.4f}")
        e.log(f"    Std deviation: {std_dist:.4f}")
        e.log(f"    Range: [{min_dist:.4f}, {max_dist:.4f}]")

    # Store results
    e["results/raw_distances"] = {
        var_name: {str(seed): dist for seed, dist in seed_dists.items()}
        for var_name, seed_dists in raw_distances.items()
    }
    e["results/aggregated_stats"] = aggregated_stats

    # Save detailed JSON
    results_data = {
        "config": {
            "original_smiles": e.ORIGINAL_SMILES,
            "variants": valid_variants,
            "num_encoders": e.NUM_ENCODERS,
            "base_seed": e.BASE_SEED,
            "hdc_dim": e.HDC_DIM,
            "depth": e.DEPTH,
            "seeds": seeds,
        },
        "raw_distances": {
            var_name: {str(seed): dist for seed, dist in seed_dists.items()}
            for var_name, seed_dists in raw_distances.items()
        },
        "aggregated_stats": aggregated_stats,
    }
    e.commit_json("similarity_results.json", results_data)

    # Create visualizations
    e.log("\n" + "=" * 60)
    e.log("CREATING VISUALIZATIONS")
    e.log("=" * 60)

    # Summary bar chart
    e.log("\n  Creating summary bar chart...")
    summary_fig = create_summary_bar_chart(
        variant_names=variant_names,
        means=means,
        stds=stds,
        output_path=os.path.join(e.path, "summary_bar_chart.png"),
    )
    plt.close(summary_fig)
    e.log("    Saved: summary_bar_chart.png")

    # Per-variant comparison figures
    e.log("\n  Creating per-variant comparison figures...")
    for var_name, var_smiles in valid_variants.items():
        stats = aggregated_stats[var_name]
        output_path = os.path.join(e.path, f"comparison_{var_name}.png")

        var_fig = create_variant_comparison_figure(
            original_smiles=e.ORIGINAL_SMILES,
            variant_name=var_name,
            variant_smiles=var_smiles,
            mean_distance=stats["mean"],
            std_distance=stats["std"],
            output_path=output_path,
        )
        plt.close(var_fig)
        e.log(f"    Saved: comparison_{var_name}.png")

    # Summary table
    e.log("\n" + "=" * 60)
    e.log("SUMMARY TABLE")
    e.log("=" * 60)
    e.log(f"{'Variant':<30} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    e.log("-" * 70)

    # Sort by mean distance
    sorted_variants = sorted(aggregated_stats.items(), key=lambda x: x[1]["mean"])
    for var_name, stats in sorted_variants:
        e.log(
            f"{var_name:<30} {stats['mean']:<10.4f} {stats['std']:<10.4f} "
            f"{stats['min']:<10.4f} {stats['max']:<10.4f}"
        )

    e.log("\n" + "=" * 60)
    e.log("Experiment completed successfully!")
    e.log("=" * 60)


@experiment.testing
def testing(e: Experiment) -> None:
    """Quick test mode - reduced parameters."""
    e.NUM_ENCODERS = 3
    e.VARIANTS = {
        "salicylic_acid": "O=C(O)c1ccccc1O",
        "benzene": "c1ccccc1",
    }


@experiment.analysis
def analysis(e: Experiment) -> None:
    """Post-experiment analysis."""
    e.log("\n" + "=" * 60)
    e.log("ANALYSIS")
    e.log("=" * 60)

    # Find most and least similar variants
    aggregated = e["results/aggregated_stats"]
    sorted_by_distance = sorted(aggregated.items(), key=lambda x: x[1]["mean"])

    most_similar = sorted_by_distance[0]
    least_similar = sorted_by_distance[-1]

    e.log(f"\nMost similar variant: {most_similar[0]} (distance: {most_similar[1]['mean']:.4f})")
    e.log(f"Least similar variant: {least_similar[0]} (distance: {least_similar[1]['mean']:.4f})")

    # Analyze embedding stability (average std across variants)
    avg_std = np.mean([stats["std"] for stats in aggregated.values()])
    e.log(f"\nAverage std across variants: {avg_std:.4f}")
    e.log("  (Lower values indicate more stable embeddings across seeds)")

    e["analysis/most_similar_variant"] = most_similar[0]
    e["analysis/least_similar_variant"] = least_similar[0]
    e["analysis/avg_std_across_variants"] = float(avg_std)


experiment.run_if_main()
