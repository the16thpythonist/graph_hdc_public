#!/usr/bin/env python
"""
Evaluate Node Set Decoding from HDC Embeddings.

This experiment evaluates the accuracy of decoding node multisets from order-0
hyperdimensional computing (HDC) embeddings. For each molecule in the dataset:
1. Extract ground truth node multiset from data.x (optionally augmented with RW features)
2. Encode to order-0 embedding (bundled node hypervectors, no message passing)
3. Decode using decode_order_zero_counter_iterative() (iterative unbinding method)
4. Compare decoded multiset with ground truth

The experiment sweeps over multiple HV dimensions to find the minimal dimension
that achieves ~98% node decoding recovery.

Metrics reported per dimension:
- Exact match accuracy per split and overall
- Summary table: hv_dim | train_acc | valid_acc | test_acc | overall_acc

Usage:
    # Quick test (2 dims, 100 samples)
    python evaluate_node_decoding.py --__TESTING__ True

    # Full evaluation without RW
    python evaluate_node_decoding.py --__DEBUG__ False

    # Full evaluation with RW features
    python evaluate_node_decoding.py --USE_RW True --__DEBUG__ False
"""

from __future__ import annotations

from collections import Counter, defaultdict

import torch
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from graph_hdc.datasets.utils import get_split
from graph_hdc.hypernet.configs import RWConfig, create_config_with_rw
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.utils.helpers import scatter_hd
from graph_hdc.utils.rw_features import augment_data_with_rw

# =============================================================================
# PARAMETERS
# =============================================================================

# -----------------------------------------------------------------------------
# Evaluation Configuration
# -----------------------------------------------------------------------------

# :param HV_DIMS:
#     List of hypervector dimensions to sweep. For each dimension a fresh
#     HyperNet is created and node decoding accuracy is evaluated.
HV_DIMS: list[int] = [512, 768, 1024, 1536, 2048]

# :param BASE_DATASET:
#     Dataset name for evaluation. Supported: "qm9", "zinc".
BASE_DATASET: str = "zinc"

# :param SAMPLES_PER_SPLIT:
#     Number of molecules to randomly sample from each split. Set to 0 to
#     use the full split.
SAMPLES_PER_SPLIT: int = 1000

# :param BATCH_SIZE:
#     Batch size for batched HDC encoding during evaluation.
BATCH_SIZE: int = 256

# :param DEVICE:
#     Device for HDC encoding. Set to "auto" (default) to pick CUDA if
#     available, or specify "cpu" / "cuda" explicitly.
DEVICE: str = "cuda"

# -----------------------------------------------------------------------------
# HDC Encoder Configuration
# -----------------------------------------------------------------------------

# :param USE_RW:
#     Whether to augment HDC node features with random walk return probabilities.
#     When True, each node's feature tuple is extended with binned RW return
#     probabilities at each step in RW_K_VALUES, making the HDC conditioning
#     vector more expressive about global graph topology.
USE_RW: bool = True

# :param RW_K_VALUES:
#     Random walk steps at which to compute return probabilities. Only used
#     when USE_RW is True.
RW_K_VALUES: tuple[int, ...] = (2, 6, 16)

# :param RW_NUM_BINS:
#     Number of uniform bins for discretising RW return probabilities on [0,1].
#     Only used when USE_RW is True.
RW_NUM_BINS: int = 6

# :param PRUNE_CODEBOOK:
#     Whether to prune the HDC codebook to only feature tuples observed in the
#     dataset. When True (default), unseen tuples cause encoding errors — set to
#     False when training on generated molecules (e.g. streaming fragments) whose
#     topology may produce novel feature combinations.
PRUNE_CODEBOOK: bool = False

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------

# :param SEED:
#     Random seed for reproducibility (codebook generation and subsampling).
SEED: int = 42

# :param __DEBUG__:
#     Debug mode — reuses same output folder during development.
__DEBUG__: bool = True

# :param __TESTING__:
#     Testing mode — runs with minimal iterations for validation.
__TESTING__: bool = False


# =============================================================================
# Helper Functions
# =============================================================================


def get_ground_truth_node_counter(data: Batch) -> dict[int, Counter]:
    """Extract ground truth node multiset from PyG data."""
    x = data.x
    batch = data.batch

    counters: dict[int, Counter] = defaultdict(Counter)
    for node_idx in range(x.size(0)):
        batch_idx = int(batch[node_idx].item())
        features = tuple(int(f) for f in x[node_idx].tolist())
        counters[batch_idx][features] += 1

    return dict(counters)


def compute_order_zero_embedding(
    hypernet: HyperNet,
    data: Batch,
) -> torch.Tensor:
    """Compute order-0 embedding (bundled node HVs, no message passing)."""
    data = hypernet.encode_properties(data)
    return scatter_hd(src=data.node_hv, index=data.batch, op="bundle")


def compare_counters(decoded: Counter, ground_truth: Counter) -> dict:
    """Compare decoded and ground truth counters."""
    exact_match = decoded == ground_truth

    all_keys = set(decoded.keys()) | set(ground_truth.keys())
    missing_in_decoded = Counter()
    extra_in_decoded = Counter()

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


# =============================================================================
# EXPERIMENT
# =============================================================================


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    """Evaluate node decoding accuracy across HV dimensions."""
    e.log("=" * 70)
    e.log("Node Decoding Accuracy Evaluation — Dimension Sweep")
    e.log("=" * 70)

    import random as _random
    _random.seed(e.SEED)

    if e.DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(e.DEVICE)
    e.log(f"Device: {device}")
    e.log(f"Dataset: {e.BASE_DATASET}")
    e.log(f"Use RW features: {e.USE_RW}")
    e.log(f"Codebook pruning: {e.PRUNE_CODEBOOK}")
    e.log(f"Samples per split: {e.SAMPLES_PER_SPLIT or 'all'}")

    rw_config = RWConfig(
        enabled=e.USE_RW,
        k_values=e.RW_K_VALUES,
        num_bins=e.RW_NUM_BINS,
    )

    # ------------------------------------------------------------------
    # Prepare datasets (loaded once, reused across dims)
    # ------------------------------------------------------------------
    splits = ["train", "valid", "test"]
    datasets = {}
    for split in splits:
        ds = list(get_split(split, dataset=e.BASE_DATASET))
        if e.__TESTING__:
            ds = ds[:100]
        elif e.SAMPLES_PER_SPLIT > 0 and len(ds) > e.SAMPLES_PER_SPLIT:
            ds = _random.sample(ds, e.SAMPLES_PER_SPLIT)
        datasets[split] = ds
        e.log(f"  {split}: {len(ds)} samples")

    # ------------------------------------------------------------------
    # Pre-scan for observed node features (needed for codebook pruning)
    # Scans the exact same data we'll evaluate on, avoiding mismatches.
    # ------------------------------------------------------------------
    observed_node_features: set[tuple] | None = None
    if rw_config.enabled and e.PRUNE_CODEBOOK:
        e.log("Pre-scanning datasets for RW-augmented node features...")
        observed_node_features = set()
        for split, ds in datasets.items():
            for data in tqdm(ds, desc=f"  Scanning {split}", disable=not e.__DEBUG__):
                d = data.clone()
                d = augment_data_with_rw(d, k_values=rw_config.k_values, num_bins=rw_config.num_bins, bin_boundaries=rw_config.bin_boundaries, clip_range=rw_config.clip_range)
                for row in d.x.int():
                    observed_node_features.add(tuple(row.tolist()))
        e.log(f"  Found {len(observed_node_features)} unique node feature tuples (with RW)")
        e["scan/num_observed_features"] = len(observed_node_features)

    # ------------------------------------------------------------------
    # Dimension sweep
    # ------------------------------------------------------------------
    summary_rows: list[dict] = []

    for hv_dim in e.HV_DIMS:
        e.log(f"\n{'─' * 70}")
        e.log(f"HV_DIM = {hv_dim}")
        e.log(f"{'─' * 70}")

        # Build config + HyperNet for this dimension
        cfg = create_config_with_rw(
            base_dataset=e.BASE_DATASET,
            hv_dim=hv_dim,
            rw_config=rw_config,
            prune_codebook=e.PRUNE_CODEBOOK,
        )
        hn = HyperNet(cfg, observed_node_features=observed_node_features)
        hn.to(device)
        hn.eval()
        e.log(f"Codebook size: {hn.nodes_codebook.shape[0]} entries")

        e.log(f"  Codebook size: {hn.nodes_codebook.shape[0]}")

        row = {"hv_dim": hv_dim}
        total_mols = 0
        total_matches = 0

        for split in splits:
            ds = datasets[split]
            loader = DataLoader(ds, batch_size=e.BATCH_SIZE, shuffle=False)

            split_matches = 0
            split_count = 0

            for batch in tqdm(loader, desc=f"  {split} (dim={hv_dim})", disable=not e.__DEBUG__):
                # Augment with RW features on-the-fly
                if rw_config.enabled:
                    data_list = batch.to_data_list()
                    data_list = [
                        augment_data_with_rw(d, k_values=rw_config.k_values, num_bins=rw_config.num_bins, bin_boundaries=rw_config.bin_boundaries, clip_range=rw_config.clip_range)
                        for d in data_list
                    ]
                    batch = Batch.from_data_list(data_list)

                batch = batch.to(device)

                gt_counters = get_ground_truth_node_counter(batch)

                with torch.no_grad():
                    order_zero_emb = compute_order_zero_embedding(hn, batch)

                decoded_counters = hn.decode_order_zero_counter_iterative(order_zero_emb)

                batch_size = order_zero_emb.size(0)
                for b_idx in range(batch_size):
                    gt = gt_counters.get(b_idx, Counter())
                    dec = decoded_counters.get(b_idx, Counter())
                    if dec == gt:
                        split_matches += 1
                    split_count += 1

            acc = split_matches / split_count if split_count > 0 else 0.0
            row[split] = acc
            total_mols += split_count
            total_matches += split_matches

            e.log(f"  {split}: {acc:.4f} ({split_matches}/{split_count})")
            e[f"results/dim_{hv_dim}/{split}/accuracy"] = acc
            e[f"results/dim_{hv_dim}/{split}/exact_matches"] = split_matches
            e[f"results/dim_{hv_dim}/{split}/num_samples"] = split_count

        overall_acc = total_matches / total_mols if total_mols > 0 else 0.0
        row["overall"] = overall_acc
        summary_rows.append(row)

        e.log(f"  OVERALL: {overall_acc:.4f} ({total_matches}/{total_mols})")
        e[f"results/dim_{hv_dim}/overall_accuracy"] = overall_acc

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    e.log("\n" + "=" * 70)
    e.log("SUMMARY TABLE")
    e.log("=" * 70)

    header = f"{'hv_dim':>8} | {'train':>8} | {'valid':>8} | {'test':>8} | {'overall':>8}"
    e.log(header)
    e.log("-" * len(header))

    for row in summary_rows:
        line = (
            f"{row['hv_dim']:>8d} | "
            f"{row.get('train', 0.0):>8.4f} | "
            f"{row.get('valid', 0.0):>8.4f} | "
            f"{row.get('test', 0.0):>8.4f} | "
            f"{row['overall']:>8.4f}"
        )
        e.log(line)

    e["summary"] = summary_rows
    e.log("\nExperiment completed.")


@experiment.testing
def testing(e: Experiment) -> None:
    """Quick test mode — reduced dims and samples."""
    e.HV_DIMS = [256, 512]
    e.BATCH_SIZE = 32


experiment.run_if_main()
