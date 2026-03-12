"""
Preprocessing utilities for the color graph domain.

Converts nx.Graphs with integer color features into the dense format
expected by :class:`FlowEdgeDecoder`:  one-hot node features, one-hot
edge attributes, and concatenated HDC conditioning vectors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from graph_hdc.models.flow_edge_decoder import raw_features_to_onehot
from graph_hdc.utils.rw_features import augment_data_with_rw

if TYPE_CHECKING:
    from graph_hdc.hypernet.configs import RWConfig
    from graph_hdc.hypernet.encoder import HyperNet

import networkx as nx


# Color graphs have exactly 2 edge classes: 0 = no-edge, 1 = edge.
NUM_EDGE_CLASSES: int = 2


def nx_to_pyg(graph: nx.Graph) -> Data:
    """Convert a color-domain nx.Graph to a PyG Data object.

    Parameters
    ----------
    graph : nx.Graph
        Graph with ``"features"`` (list[int]) on each node.

    Returns
    -------
    Data
        PyG Data with integer ``x`` and bidirectional ``edge_index``.
    """
    nodes = sorted(graph.nodes())
    x = torch.tensor(
        [graph.nodes[n]["features"] for n in nodes],
        dtype=torch.long,
    )

    edges = list(graph.edges())
    if edges:
        src = [u for u, v in edges] + [v for u, v in edges]
        dst = [v for u, v in edges] + [u for u, v in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def preprocess_graph(
    graph: nx.Graph,
    hypernet: "HyperNet",
    feature_bins: list[int],
    rw_config: "RWConfig | None" = None,
    device: torch.device | str = "cpu",
) -> Data:
    """Convert a single nx.Graph to FlowEdgeDecoder-compatible Data.

    Steps:
    1. ``nx.Graph`` → PyG Data (integer features, bidirectional edges)
    2. Augment with RW return probabilities (if *rw_config* enabled)
    3. Encode with HyperNet → HDC conditioning vector
    4. Convert integer features to one-hot
    5. Create one-hot edge attributes (class 1 = edge)

    Parameters
    ----------
    graph : nx.Graph
        Color-domain graph with ``"features"`` node attributes.
    hypernet : HyperNet
        Encoder for computing HDC embeddings.
    feature_bins : list[int]
        Full feature bins (base + RW if applicable).
    rw_config : RWConfig or None
        Random walk configuration.  ``None`` or disabled = no RW.
    device : str or torch.device
        Device for HyperNet encoding.

    Returns
    -------
    Data
        Preprocessed data with ``x`` (one-hot), ``edge_index``,
        ``edge_attr`` (one-hot), ``hdc_vector``, and ``original_x``.
    """
    data = nx_to_pyg(graph)

    # RW augmentation (appends binned RW columns to data.x)
    if rw_config is not None and rw_config.enabled:
        data = augment_data_with_rw(
            data,
            k_values=rw_config.k_values,
            num_bins=rw_config.num_bins,
            bin_boundaries=rw_config.bin_boundaries,
            clip_range=rw_config.clip_range,
        )

    # HDC encoding
    hdc_data = data.clone().to(device)
    # Add batch vector for single graph
    hdc_data.batch = torch.zeros(hdc_data.x.size(0), dtype=torch.long, device=device)
    with torch.no_grad():
        hdc_out = hypernet.forward(hdc_data, normalize=True)
    order_zero = hdc_out["node_terms"].detach().cpu()
    order_n = hdc_out["graph_embedding"].detach().cpu()
    hdc_vector = torch.cat([order_zero, order_n], dim=-1)  # [1, 2*hv_dim]

    # One-hot features
    x_onehot = raw_features_to_onehot(data.x, feature_bins=feature_bins)

    # Edge attributes: all actual edges are class 1
    num_edges = data.edge_index.size(1)
    edge_attr = torch.zeros(num_edges, NUM_EDGE_CLASSES)
    if num_edges > 0:
        edge_attr[:, 1] = 1.0

    return Data(
        x=x_onehot,
        edge_index=data.edge_index,
        edge_attr=edge_attr,
        hdc_vector=hdc_vector,
        original_x=data.x,
    )


def preprocess_graphs(
    graphs: list[nx.Graph],
    hypernet: "HyperNet",
    feature_bins: list[int],
    rw_config: "RWConfig | None" = None,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
    show_progress: bool = False,
) -> list[Data]:
    """Preprocess a list of nx.Graphs in batches for efficiency.

    HyperNet encoding is batched for throughput.  All other steps
    are per-graph.

    Parameters
    ----------
    graphs : list[nx.Graph]
        Color-domain graphs.
    hypernet : HyperNet
        HDC encoder.
    feature_bins : list[int]
        Full feature bins (base + RW).
    rw_config : RWConfig or None
        Random walk configuration.
    device : str or torch.device
        Device for HyperNet encoding.
    batch_size : int
        Number of graphs per encoding batch.
    show_progress : bool
        Show tqdm progress bar.

    Returns
    -------
    list[Data]
        Preprocessed Data objects.
    """
    # Step 1: convert all graphs to PyG and optionally augment with RW
    pyg_data_list: list[Data] = []
    for graph in graphs:
        data = nx_to_pyg(graph)
        if rw_config is not None and rw_config.enabled:
            data = augment_data_with_rw(
                data,
                k_values=rw_config.k_values,
                num_bins=rw_config.num_bins,
                bin_boundaries=rw_config.bin_boundaries,
                clip_range=rw_config.clip_range,
            )
        pyg_data_list.append(data)

    # Step 2: batch-encode with HyperNet
    from torch_geometric.loader import DataLoader as PyGLoader

    iterator = range(0, len(pyg_data_list), batch_size)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Encoding", total=len(iterator))
        except ImportError:
            pass

    all_order_zero: list[Tensor] = []
    all_order_n: list[Tensor] = []

    hypernet_device = next(iter(hypernet.parameters())).device if hasattr(hypernet, 'parameters') else torch.device(device)

    for start in iterator:
        batch_list = pyg_data_list[start : start + batch_size]
        batch = Batch.from_data_list(batch_list).to(hypernet_device)
        with torch.no_grad():
            hdc_out = hypernet.forward(batch, normalize=True)
        all_order_zero.append(hdc_out["node_terms"].detach().cpu())
        all_order_n.append(hdc_out["graph_embedding"].detach().cpu())

    order_zero = torch.cat(all_order_zero, dim=0)  # [N, hv_dim]
    order_n = torch.cat(all_order_n, dim=0)          # [N, hv_dim]

    # Step 3: assemble final Data objects
    results: list[Data] = []
    for i, data in enumerate(pyg_data_list):
        x_onehot = raw_features_to_onehot(data.x, feature_bins=feature_bins)

        num_edges = data.edge_index.size(1)
        edge_attr = torch.zeros(num_edges, NUM_EDGE_CLASSES)
        if num_edges > 0:
            edge_attr[:, 1] = 1.0

        hdc_vector = torch.cat(
            [order_zero[i : i + 1], order_n[i : i + 1]], dim=-1
        )

        results.append(
            Data(
                x=x_onehot,
                edge_index=data.edge_index,
                edge_attr=edge_attr,
                hdc_vector=hdc_vector,
                original_x=data.x,
            )
        )

    return results
