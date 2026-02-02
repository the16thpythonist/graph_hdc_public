#!/usr/bin/env python
"""
Evaluate Node Set Decoding from HDC Embeddings (QM9).

This experiment evaluates the accuracy of decoding node multisets from order-0
hyperdimensional computing (HDC) embeddings. For each molecule in the QM9 dataset:
1. Extract ground truth node multiset from data.x
2. Encode to order-0 embedding (bundled node hypervectors, no message passing)
3. Decode using decode_order_zero_counter_iterative() (iterative unbinding method)
4. Compare decoded multiset with ground truth

Metrics reported:
- Exact match accuracy: % of molecules where decoded == ground truth
- Per-atom-type breakdown
- Error analysis by feature dimension

Usage:
    # Quick test
    python evaluate_node_decoding.py --__TESTING__ True

    # Full evaluation
    python evaluate_node_decoding.py --__DEBUG__ False
"""

from __future__ import annotations

import math
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import torch
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from torch_geometric.data import Batch
from tqdm import tqdm

from graph_hdc.datasets.utils import get_split, get_dataset_info
from graph_hdc.hypernet.configs import (
    DSHDCConfig,
    FeatureConfig,
    Features,
    IndexRange,
)
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.hypernet.feature_encoders import CombinatoricIntegerEncoder
from graph_hdc.hypernet.types import VSAModel
from graph_hdc.utils.helpers import scatter_hd

# =============================================================================
# PARAMETERS
# =============================================================================

# HDC Encoder configuration
HDC_DIM: int = 512  # Hypervector dimension
HDC_DEPTH: int = 3  # Message passing depth (not used for order-0, but needed for HyperNet init)

# Processing
BATCH_SIZE: int = 512  # Batch size for encoding

# System configuration
SEED: int = 42

# Debug/Testing modes
__DEBUG__: bool = True
__TESTING__: bool = False


# =============================================================================
# Helper Functions
# =============================================================================


def create_hdc_config(
    hv_dim: int,
    depth: int,
    device: str = "cpu",
) -> DSHDCConfig:
    """
    Create a HyperNet configuration for QM9.

    Args:
        hv_dim: Hypervector dimension
        depth: Message passing depth
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
        name=f"QM9_HRR_{hv_dim}_depth{depth}",
        hv_dim=hv_dim,
        vsa=VSAModel.HRR,
        base_dataset="qm9",
        hypernet_depth=depth,
        device=device,
        seed=42,
        normalize=True,
        dtype="float64",
        node_feature_configs=OrderedDict([
            (Features.NODE_FEATURES, node_feature_config),
        ]),
    )


def get_ground_truth_node_counter(data: Batch | torch.Tensor) -> dict[int, Counter]:
    """
    Extract ground truth node multiset from PyG data.

    Args:
        data: PyG Data or Batch object with x tensor

    Returns:
        Dictionary mapping batch index to Counter of node tuples
    """
    x = data.x if hasattr(data, "x") else data
    batch = data.batch if hasattr(data, "batch") else torch.zeros(x.size(0), dtype=torch.long)

    counters: dict[int, Counter] = defaultdict(Counter)

    for node_idx in range(x.size(0)):
        batch_idx = int(batch[node_idx].item())
        # Convert features to tuple of ints
        features = tuple(int(f) for f in x[node_idx].tolist())
        counters[batch_idx][features] += 1

    return dict(counters)


def compute_order_zero_embedding(
    hypernet: HyperNet,
    data: Batch,
) -> torch.Tensor:
    """
    Compute order-0 embedding (bundled node hypervectors, no message passing).

    Args:
        hypernet: HyperNet encoder instance
        data: PyG Batch object

    Returns:
        Order-0 graph embeddings [batch_size, hv_dim]
    """
    # Encode node features to hypervectors
    data = hypernet.encode_properties(data)

    # Bundle node hypervectors per graph (no message passing)
    order_zero_embedding = scatter_hd(src=data.node_hv, index=data.batch, op="bundle")

    return order_zero_embedding


def compare_counters(decoded: Counter, ground_truth: Counter) -> dict:
    """
    Compare decoded and ground truth counters.

    Args:
        decoded: Decoded node counter
        ground_truth: Ground truth node counter

    Returns:
        Dictionary with comparison metrics
    """
    exact_match = decoded == ground_truth

    # Compute detailed metrics
    all_keys = set(decoded.keys()) | set(ground_truth.keys())

    missing_in_decoded = Counter()  # In GT but not decoded (or under-counted)
    extra_in_decoded = Counter()  # In decoded but not GT (or over-counted)

    for key in all_keys:
        gt_count = ground_truth.get(key, 0)
        dec_count = decoded.get(key, 0)
        diff = dec_count - gt_count
        if diff > 0:
            extra_in_decoded[key] = diff
        elif diff < 0:
            missing_in_decoded[key] = -diff

    return {
        "exact_match": exact_match,
        "missing_in_decoded": missing_in_decoded,
        "extra_in_decoded": extra_in_decoded,
        "decoded_total": sum(decoded.values()),
        "ground_truth_total": sum(ground_truth.values()),
    }


def get_atom_type_name(atom_idx: int) -> str:
    """Get human-readable atom type name for QM9."""
    atom_map = {0: "C", 1: "N", 2: "O", 3: "F"}
    return atom_map.get(atom_idx, f"Unknown({atom_idx})")


# =============================================================================
# EXPERIMENT
# =============================================================================


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    """Main experiment: evaluate node decoding accuracy on QM9."""
    e.log("=" * 60)
    e.log("Node Decoding Accuracy Evaluation (QM9)")
    e.log("=" * 60)

    # Store configuration
    e["config/dataset"] = "qm9"
    e["config/hdc_dim"] = e.HDC_DIM
    e["config/hdc_depth"] = e.HDC_DEPTH
    e["config/batch_size"] = e.BATCH_SIZE
    e["config/seed"] = e.SEED

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e.log(f"Using device: {device}")
    e["config/device"] = str(device)

    # Create HyperNet encoder
    e.log("Creating HyperNet encoder for QM9...")
    config = create_hdc_config(e.HDC_DIM, e.HDC_DEPTH, device=str(device))
    hypernet = HyperNet(config)
    hypernet.to(device)
    hypernet.eval()

    e.log(f"  Node codebook size: {hypernet.nodes_codebook.shape[0]}")
    e.log(f"  Hypervector dimension: {hypernet.hv_dim}")
    e["model/node_codebook_size"] = hypernet.nodes_codebook.shape[0]
    e["model/hv_dim"] = hypernet.hv_dim

    # Get dataset info for feature names
    dataset_info = get_dataset_info("qm9")
    e["data/num_unique_node_types"] = len(dataset_info.node_features)

    # Feature dimension names for QM9
    feature_names = ["atom_type", "degree_idx", "formal_charge_idx", "explicit_hs"]
    e["data/feature_names"] = feature_names

    # Process each split
    splits = ["train", "valid", "test"]
    overall_results = {
        "total_molecules": 0,
        "total_exact_matches": 0,
        "per_split": {},
        "error_analysis": {
            "missing_by_atom_type": Counter(),
            "extra_by_atom_type": Counter(),
            "node_count_errors": Counter(),  # (decoded_count - gt_count)
        },
    }

    for split in splits:
        e.log(f"\nProcessing {split} split...")
        dataset = get_split(split, dataset="qm9")

        if e.__TESTING__:
            # Limit to 100 samples in testing mode
            dataset = dataset[:100]

        num_samples = len(dataset)
        e.log(f"  Number of samples: {num_samples}")

        split_exact_matches = 0
        split_results = []

        # Process in batches
        from torch_geometric.loader import DataLoader
        loader = DataLoader(dataset, batch_size=e.BATCH_SIZE, shuffle=False)

        for batch in tqdm(loader, desc=f"  {split}", disable=not e.__DEBUG__):
            batch = batch.to(device)

            # Get ground truth
            gt_counters = get_ground_truth_node_counter(batch)

            # Compute order-0 embedding
            with torch.no_grad():
                order_zero_emb = compute_order_zero_embedding(hypernet, batch)

            # Decode using iterative unbinding method
            decoded_counters = hypernet.decode_order_zero_counter_iterative(order_zero_emb)

            # Compare each sample in batch
            batch_size = order_zero_emb.size(0)
            for b_idx in range(batch_size):
                gt_counter = gt_counters.get(b_idx, Counter())
                dec_counter = decoded_counters.get(b_idx, Counter())

                comparison = compare_counters(dec_counter, gt_counter)
                split_results.append(comparison)

                if comparison["exact_match"]:
                    split_exact_matches += 1
                else:
                    # Track errors by atom type
                    for node_tuple, count in comparison["missing_in_decoded"].items():
                        atom_type = node_tuple[0]
                        overall_results["error_analysis"]["missing_by_atom_type"][atom_type] += count
                    for node_tuple, count in comparison["extra_in_decoded"].items():
                        atom_type = node_tuple[0]
                        overall_results["error_analysis"]["extra_by_atom_type"][atom_type] += count

                    # Track node count errors
                    count_diff = comparison["decoded_total"] - comparison["ground_truth_total"]
                    overall_results["error_analysis"]["node_count_errors"][count_diff] += 1

        # Compute split accuracy
        split_accuracy = split_exact_matches / num_samples if num_samples > 0 else 0.0
        e.log(f"  Exact match accuracy: {split_accuracy:.4f} ({split_exact_matches}/{num_samples})")

        overall_results["per_split"][split] = {
            "num_samples": num_samples,
            "exact_matches": split_exact_matches,
            "accuracy": split_accuracy,
        }
        overall_results["total_molecules"] += num_samples
        overall_results["total_exact_matches"] += split_exact_matches

        # Track metrics
        e.track(f"accuracy/{split}", split_accuracy)
        e[f"results/{split}/num_samples"] = num_samples
        e[f"results/{split}/exact_matches"] = split_exact_matches
        e[f"results/{split}/accuracy"] = split_accuracy

    # Compute overall accuracy
    overall_accuracy = (
        overall_results["total_exact_matches"] / overall_results["total_molecules"]
        if overall_results["total_molecules"] > 0
        else 0.0
    )

    e.log("\n" + "=" * 60)
    e.log("OVERALL RESULTS")
    e.log("=" * 60)
    e.log(f"Total molecules: {overall_results['total_molecules']}")
    e.log(f"Total exact matches: {overall_results['total_exact_matches']}")
    e.log(f"Overall accuracy: {overall_accuracy:.4f}")

    e["results/overall/total_molecules"] = overall_results["total_molecules"]
    e["results/overall/total_exact_matches"] = overall_results["total_exact_matches"]
    e["results/overall/accuracy"] = overall_accuracy

    # Error analysis
    e.log("\n" + "-" * 60)
    e.log("ERROR ANALYSIS")
    e.log("-" * 60)

    # Missing nodes by atom type
    e.log("\nMissing nodes by atom type (under-decoded):")
    missing_by_atom = overall_results["error_analysis"]["missing_by_atom_type"]
    for atom_idx, count in sorted(missing_by_atom.items()):
        atom_name = get_atom_type_name(atom_idx)
        e.log(f"  {atom_name} (idx={atom_idx}): {count}")
        e[f"error_analysis/missing_by_atom_type/{atom_name}"] = count

    # Extra nodes by atom type
    e.log("\nExtra nodes by atom type (over-decoded):")
    extra_by_atom = overall_results["error_analysis"]["extra_by_atom_type"]
    for atom_idx, count in sorted(extra_by_atom.items()):
        atom_name = get_atom_type_name(atom_idx)
        e.log(f"  {atom_name} (idx={atom_idx}): {count}")
        e[f"error_analysis/extra_by_atom_type/{atom_name}"] = count

    # Node count error distribution
    e.log("\nNode count error distribution (decoded - ground_truth):")
    count_errors = overall_results["error_analysis"]["node_count_errors"]
    for diff, count in sorted(count_errors.items()):
        e.log(f"  {diff:+d}: {count} molecules")
        e[f"error_analysis/node_count_errors/{diff}"] = count

    # Per-atom-type accuracy (based on missing/extra counts)
    e.log("\n" + "-" * 60)
    e.log("PER-ATOM-TYPE SUMMARY")
    e.log("-" * 60)

    all_atom_indices = set(missing_by_atom.keys()) | set(extra_by_atom.keys())
    for atom_idx in sorted(all_atom_indices):
        atom_name = get_atom_type_name(atom_idx)
        missing = missing_by_atom.get(atom_idx, 0)
        extra = extra_by_atom.get(atom_idx, 0)
        total_errors = missing + extra
        e.log(f"  {atom_name}: missing={missing}, extra={extra}, total_errors={total_errors}")

    # Summary table
    e.log("\n" + "=" * 60)
    e.log("SUMMARY TABLE")
    e.log("=" * 60)
    e.log(f"{'Split':<10} {'Samples':<10} {'Matches':<10} {'Accuracy':<10}")
    e.log("-" * 40)
    for split in splits:
        s = overall_results["per_split"][split]
        e.log(f"{split:<10} {s['num_samples']:<10} {s['exact_matches']:<10} {s['accuracy']:<10.4f}")
    e.log("-" * 40)
    e.log(f"{'TOTAL':<10} {overall_results['total_molecules']:<10} {overall_results['total_exact_matches']:<10} {overall_accuracy:<10.4f}")

    e.log("\nExperiment completed successfully!")


@experiment.testing
def testing(e: Experiment) -> None:
    """Quick test mode - reduced parameters."""
    e.BATCH_SIZE = 32


experiment.run_if_main()
