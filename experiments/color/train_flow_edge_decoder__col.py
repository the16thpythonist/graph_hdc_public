"""
Color graph FlowEdgeDecoder training — finite dataset variant.

Extends the domain-agnostic base with color-graph-specific settings:

- ``FEATURE_BINS = [17]`` (single color feature, 17 bins)
- ``NUM_EDGE_CLASSES = 2`` (no-edge, edge)
- Hook implementations for computing statistics and evaluating
  reconstruction via :class:`ColoredGraphDomain`.

Data loading hooks (``load_train_data``, ``load_valid_data``) raise
``NotImplementedError`` by default — override them in a child
experiment (e.g. ``train_flow_edge_decoder__col__stream.py``) or
provide a COGILES dataset path.
"""

from __future__ import annotations

import itertools
import os
from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGLoader
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from graph_hdc.domains.color.domain import ColoredGraphDomain
from graph_hdc.domains.color.preprocessing import (
    NUM_EDGE_CLASSES as COLOR_NUM_EDGE_CLASSES,
    preprocess_graphs,
)
from graph_hdc.models.flow_edge_decoder import (
    FlowEdgeDecoder,
    compute_edge_marginals,
    compute_node_counts,
    compute_size_edge_marginals,
    node_tuples_to_onehot,
    onehot_to_raw_features,
)

# =====================================================================
# Extend base experiment
# =====================================================================

experiment = Experiment.extend(
    "train_flow_edge_decoder.py",
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

# =====================================================================
# Domain-specific parameter overrides
# =====================================================================

FEATURE_BINS: list[int] = [17]
NUM_EDGE_CLASSES: int = COLOR_NUM_EDGE_CLASSES  # 2

# Smaller defaults sensible for color graphs
HDC_DIM: int = 256
HDC_DEPTH: int = 3

# =====================================================================
# Additional parameters
# =====================================================================

# :param COGILES_DATASET_PATH:
#     Path to a text file with COGILES strings (one per line).
#     Used by the default ``load_train_data`` / ``load_valid_data``
#     hooks.  Set to "" to skip (child must override hooks).
COGILES_DATASET_PATH: str = ""

# :param VALID_FRACTION:
#     Fraction of the loaded dataset to use for validation.
VALID_FRACTION: float = 0.1

# :param NUM_RECONSTRUCTION_SAMPLES:
#     Number of samples to reconstruct after training.
NUM_RECONSTRUCTION_SAMPLES: int = 50


# =====================================================================
# Hook: load_train_data
# =====================================================================


@experiment.hook("load_train_data", default=False, replace=True)
def load_train_data(e, hypernet, rw_config, feature_bins, device):
    """Load and preprocess color graph training data.

    If ``COGILES_DATASET_PATH`` is set, loads graphs from the file.
    Otherwise raises ``NotImplementedError`` so that a child experiment
    (e.g. the streaming variant) can provide data.
    """
    if not e.COGILES_DATASET_PATH:
        raise NotImplementedError(
            "Set COGILES_DATASET_PATH or override 'load_train_data' in "
            "a child experiment (e.g. the streaming variant)."
        )

    from graph_hdc.domains.color.datasets import CogilesDataset

    domain = ColoredGraphDomain()
    ds = CogilesDataset(e.COGILES_DATASET_PATH, domain)
    all_graphs = list(ds)

    n_valid = max(1, int(len(all_graphs) * e.VALID_FRACTION))
    train_graphs = all_graphs[n_valid:]

    e.log(f"Loaded {len(all_graphs)} graphs, {len(train_graphs)} for training")

    train_data = preprocess_graphs(
        train_graphs,
        hypernet,
        feature_bins,
        rw_config=rw_config,
        device=device,
        show_progress=True,
    )

    train_loader = PyGLoader(
        train_data,
        batch_size=e.BATCH_SIZE,
        shuffle=True,
        num_workers=e.NUM_WORKERS,
    )

    return train_loader, train_data


# =====================================================================
# Hook: load_valid_data
# =====================================================================


@experiment.hook("load_valid_data", default=False, replace=True)
def load_valid_data(e, hypernet, rw_config, feature_bins, device):
    """Load and preprocess color graph validation data."""
    if not e.COGILES_DATASET_PATH:
        raise NotImplementedError(
            "Set COGILES_DATASET_PATH or override 'load_valid_data' in "
            "a child experiment."
        )

    from graph_hdc.domains.color.datasets import CogilesDataset

    domain = ColoredGraphDomain()
    ds = CogilesDataset(e.COGILES_DATASET_PATH, domain)
    all_graphs = list(ds)

    n_valid = max(1, int(len(all_graphs) * e.VALID_FRACTION))
    valid_graphs = all_graphs[:n_valid]

    e.log(f"Validation set: {len(valid_graphs)} graphs")

    valid_data = preprocess_graphs(
        valid_graphs,
        hypernet,
        feature_bins,
        rw_config=rw_config,
        device=device,
        show_progress=True,
    )

    valid_loader = PyGLoader(
        valid_data,
        batch_size=e.BATCH_SIZE,
        shuffle=False,
        num_workers=e.NUM_WORKERS,
    )

    vis_samples = valid_data[: min(10, len(valid_data))]

    return valid_data, valid_loader, vis_samples


# =====================================================================
# Hook: compute_statistics
# =====================================================================


@experiment.hook("compute_statistics", default=False, replace=True)
def compute_statistics(e, train_data):
    """Compute edge marginals and node-count distribution from training data."""
    if train_data is None:
        raise ValueError(
            "train_data is None — cannot compute statistics.  "
            "The streaming child experiment must override this hook."
        )

    edge_marginals = compute_edge_marginals(
        train_data, num_edge_classes=e.NUM_EDGE_CLASSES,
    )
    node_counts = compute_node_counts(train_data)
    max_nodes = int(node_counts.shape[0]) - 1
    size_edge_marginals = compute_size_edge_marginals(
        train_data,
        max_nodes + 10,
        num_edge_classes=e.NUM_EDGE_CLASSES,
    )

    return edge_marginals, node_counts, max_nodes, size_edge_marginals


# =====================================================================
# Hook: evaluate_reconstruction
# =====================================================================


@experiment.hook("evaluate_reconstruction", default=False, replace=True)
def evaluate_reconstruction(
    e, model, hypernet, rw_config, feature_bins, valid_data, vis_samples, device
):
    """Reconstruct validation graphs and check validity via the domain."""
    domain = ColoredGraphDomain()

    if not vis_samples:
        e.log("No validation samples for reconstruction.")
        return

    import networkx as nx
    from graph_hdc.models.flow_edge_decoder import node_tuples_to_onehot

    num_samples = min(e.NUM_RECONSTRUCTION_SAMPLES, len(vis_samples))
    e.log(f"Reconstructing {num_samples} graphs...")

    valid_count = 0
    match_count = 0
    results = []

    model.eval()
    model = model.to(device)

    for idx in range(num_samples):
        sample = vis_samples[idx]

        # Decode nodes from HDC vector
        hdc_vector = sample.hdc_vector.to(device)  # [1, 2*hv_dim]
        hv_dim = hypernet.hv_dim
        order_zero = hdc_vector[0, :hv_dim]

        # Use iterative decoding for node tuples
        try:
            decoded_nodes = hypernet.decode_order_zero_iterative(
                order_zero.to(dtype=hypernet.nodes_codebook.dtype),
            )
            if isinstance(decoded_nodes, tuple):
                decoded_nodes = decoded_nodes[0]
        except Exception:
            results.append({"idx": idx, "valid": False, "match": False, "error": "decode_failed"})
            continue

        if not decoded_nodes:
            results.append({"idx": idx, "valid": False, "match": False, "error": "no_nodes"})
            continue

        num_nodes = len(decoded_nodes)

        # Trim to base features (drop RW columns) for node tuples
        base_len = len(e.FEATURE_BINS)
        base_tuples = [t[:base_len] for t in decoded_nodes]

        # Convert to one-hot
        node_features = node_tuples_to_onehot(
            decoded_nodes, device=device, feature_bins=feature_bins,
        )  # [num_nodes, sum(feature_bins)]

        # Prepare for edge generation
        node_mask = torch.ones(1, num_nodes, dtype=torch.bool, device=device)
        node_feat = node_features.unsqueeze(0)  # [1, num_nodes, feat_dim]
        hdc_cond = hdc_vector.to(device)  # [1, 2*hv_dim]

        # Sample edges
        with torch.no_grad():
            generated = model.sample(
                hdc_vectors=hdc_cond,
                node_features=node_feat,
                node_mask=node_mask,
                sample_steps=e.SAMPLE_STEPS,
                eta=e.ETA,
                omega=e.OMEGA,
                time_distortion=e.SAMPLE_TIME_DISTORTION,
            )

        if not generated:
            results.append({"idx": idx, "valid": False, "match": False, "error": "sample_failed"})
            continue

        gen_data = generated[0]

        # Convert generated data to nx.Graph with features
        G = nx.Graph()
        # Get raw features from one-hot
        gen_x = gen_data.x  # one-hot
        raw_x = onehot_to_raw_features(gen_x, feature_bins=feature_bins)
        for n_idx in range(raw_x.size(0)):
            feats = raw_x[n_idx, :base_len].tolist()
            G.add_node(n_idx, features=[int(f) for f in feats], color=int(feats[0]))

        # Add edges from generated edge_index
        if gen_data.edge_index is not None and gen_data.edge_index.size(1) > 0:
            edge_set = set()
            ei = gen_data.edge_index.cpu()
            for e_idx in range(ei.size(1)):
                u, v = int(ei[0, e_idx]), int(ei[1, e_idx])
                if u < v:
                    edge_set.add((u, v))
            for u, v in edge_set:
                G.add_edge(u, v)

        # Evaluate via domain
        result = domain.unprocess(G)

        is_valid = result.is_valid
        canonical_key = result.canonical_key

        # Compare with original
        original_key = None
        if hasattr(sample, "canonical_key"):
            original_key = sample.canonical_key

        is_match = (
            canonical_key is not None
            and original_key is not None
            and canonical_key == original_key
        )

        if is_valid:
            valid_count += 1
        if is_match:
            match_count += 1

        results.append({
            "idx": idx,
            "valid": is_valid,
            "match": is_match,
            "num_nodes": num_nodes,
            "canonical_key": canonical_key,
        })

    valid_rate = valid_count / num_samples if num_samples > 0 else 0.0
    match_rate = match_count / num_samples if num_samples > 0 else 0.0

    e.log(f"Reconstruction: {valid_count}/{num_samples} valid ({valid_rate:.1%})")
    e.log(f"Reconstruction: {match_count}/{num_samples} exact match ({match_rate:.1%})")

    e["reconstruction/num_samples"] = num_samples
    e["reconstruction/valid_count"] = valid_count
    e["reconstruction/match_count"] = match_count
    e["reconstruction/valid_rate"] = valid_rate
    e["reconstruction/match_rate"] = match_rate


# =====================================================================
# Testing mode
# =====================================================================


@experiment.testing
def testing(e: Experiment) -> None:
    e.EPOCHS = 2
    e.BATCH_SIZE = 4
    e.HDC_DIM = 64
    e.HDC_DEPTH = 2
    e.N_LAYERS = 2
    e.HIDDEN_DIM = 32
    e.HIDDEN_MLP_DIM = 32
    e.CONDITION_DIM = 32
    e.TIME_EMBED_DIM = 32
    e.SAMPLE_STEPS = 3
    e.NUM_RECONSTRUCTION_SAMPLES = 2
    e.RECONSTRUCTION_BATCH_SIZE = 2


experiment.run_if_main()
