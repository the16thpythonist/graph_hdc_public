"""
Streaming color graph FlowEdgeDecoder training.

Extends ``train_flow_edge_decoder__col.py`` with infinite streaming data
from the color graph generators (:class:`DelaunayMeshGenerator`,
:class:`RandomTreeGenerator`, :class:`GridSubgraphGenerator`).

Training data is generated on-the-fly using :class:`MixedStreamDataset`
with configurable generator weights.  Validation data is a fixed set
generated once at experiment start.
"""

from __future__ import annotations

import itertools

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGLoader
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from graph_hdc.domains.base import MixedStreamDataset
from graph_hdc.domains.color.domain import ColoredGraphDomain
from graph_hdc.domains.color.generators import (
    DelaunayMeshGenerator,
    GridSubgraphGenerator,
    RandomTreeGenerator,
)
from graph_hdc.domains.color.datasets import ColorGraphStreamingDataset
from graph_hdc.domains.color.preprocessing import (
    preprocess_graph,
    preprocess_graphs,
)
from graph_hdc.models.flow_edge_decoder import (
    compute_edge_marginals,
    compute_node_counts,
    compute_size_edge_marginals,
)

# =====================================================================
# Extend color graph experiment
# =====================================================================

experiment = Experiment.extend(
    "train_flow_edge_decoder__col.py",
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

# =====================================================================
# Streaming parameters
# =====================================================================

# :param STEPS_PER_EPOCH:
#     Number of training steps (batches) per epoch.  Since data is
#     infinite, this defines when validation runs.
STEPS_PER_EPOCH: int = 1000

# :param NUM_VALID_GRAPHS:
#     Number of graphs to generate for the fixed validation set.
NUM_VALID_GRAPHS: int = 1000

# :param NUM_STATISTICS_GRAPHS:
#     Number of graphs to generate for computing edge marginals and
#     node count distributions.
NUM_STATISTICS_GRAPHS: int = 5000

# =====================================================================
# Generator parameters
# =====================================================================

# :param MIN_NODES:
#     Minimum number of nodes per generated graph.
MIN_NODES: int = 4

# :param MAX_NODES:
#     Maximum number of nodes per generated graph.
MAX_NODES: int = 40

# :param MAX_DEGREE:
#     Maximum node degree constraint.
MAX_DEGREE: int = 6

# :param GENERATOR_SEED:
#     Base seed for graph generators.  Each generator gets a
#     derived seed for independence.
GENERATOR_SEED: int = 42

# :param DELAUNAY_WEIGHT:
#     Sampling weight for the Delaunay mesh generator.
DELAUNAY_WEIGHT: float = 1.0

# :param TREE_WEIGHT:
#     Sampling weight for the random tree generator.
TREE_WEIGHT: float = 1.0

# :param GRID_WEIGHT:
#     Sampling weight for the grid subgraph generator.
GRID_WEIGHT: float = 1.0

# :param DELAUNAY_EDGE_KEEP:
#     Fraction of non-spanning-tree edges to keep in Delaunay graphs.
DELAUNAY_EDGE_KEEP: float = 0.7

# :param GRID_NODE_KEEP:
#     Fraction of grid nodes to keep before taking the largest
#     connected component.
GRID_NODE_KEEP: float = 0.7

# :param GRID_EDGE_KEEP:
#     Fraction of non-spanning-tree edges to keep in grid subgraphs.
GRID_EDGE_KEEP: float = 0.9

# :param COLOR_SUBSET:
#     Optional subset of color indices to use.  None means all 17.
COLOR_SUBSET: list[int] | None = None

# :param COLOR_WEIGHTS:
#     Optional sampling weights for colors.  None means uniform.
COLOR_WEIGHTS: list[float] | None = None

# =====================================================================
# Training overrides
# =====================================================================

EPOCHS: int = 200
BATCH_SIZE: int = 16


# =====================================================================
# Helpers
# =====================================================================


def _make_generators(e) -> list:
    """Create the three generators from experiment parameters."""
    common = dict(
        min_nodes=e.MIN_NODES,
        max_nodes=e.MAX_NODES,
        max_degree=e.MAX_DEGREE,
        num_colors=17,
        color_subset=e.COLOR_SUBSET,
        color_weights=e.COLOR_WEIGHTS,
    )

    return [
        DelaunayMeshGenerator(
            edge_keep_fraction=e.DELAUNAY_EDGE_KEEP,
            seed=e.GENERATOR_SEED,
            **common,
        ),
        RandomTreeGenerator(
            seed=e.GENERATOR_SEED + 1,
            **common,
        ),
        GridSubgraphGenerator(
            node_keep_fraction=e.GRID_NODE_KEEP,
            edge_keep_fraction=e.GRID_EDGE_KEEP,
            seed=e.GENERATOR_SEED + 2,
            **common,
        ),
    ]


def _make_mixed_dataset(e, generators, domain):
    """Create a MixedStreamDataset from generators."""
    datasets = [
        ColorGraphStreamingDataset(gen, domain) for gen in generators
    ]
    weights = [e.DELAUNAY_WEIGHT, e.TREE_WEIGHT, e.GRID_WEIGHT]

    return MixedStreamDataset(
        datasets,
        weights=weights,
        seed=e.GENERATOR_SEED + 100,
    )


class _StreamingDataLoader:
    """Minimal DataLoader-compatible wrapper around a streaming dataset.

    Generates and preprocesses ``steps_per_epoch`` batches per
    iteration.  Each batch is created by drawing ``batch_size``
    graphs from *mixed_dataset*, preprocessing them, and collating
    into a PyG Batch.
    """

    def __init__(
        self,
        mixed_dataset,
        preprocess_fn,
        batch_size: int,
        steps_per_epoch: int,
    ):
        self.mixed_iter = iter(mixed_dataset)
        self.preprocess_fn = preprocess_fn
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            graphs = [next(self.mixed_iter) for _ in range(self.batch_size)]
            data_list = [self.preprocess_fn(g) for g in graphs]
            yield Batch.from_data_list(data_list)

    def __len__(self):
        return self.steps_per_epoch


# =====================================================================
# Hook: load_train_data
# =====================================================================


@experiment.hook("load_train_data", default=False, replace=True)
def load_train_data(e, hypernet, rw_config, feature_bins, device):
    """Create an infinite streaming DataLoader from color graph generators."""
    domain = ColoredGraphDomain()
    generators = _make_generators(e)
    mixed_ds = _make_mixed_dataset(e, generators, domain)

    def preprocess_fn(graph):
        return preprocess_graph(
            graph, hypernet, feature_bins,
            rw_config=rw_config, device=device,
        )

    train_loader = _StreamingDataLoader(
        mixed_dataset=mixed_ds,
        preprocess_fn=preprocess_fn,
        batch_size=e.BATCH_SIZE,
        steps_per_epoch=e.STEPS_PER_EPOCH,
    )

    e.log(
        f"Streaming training: {e.STEPS_PER_EPOCH} steps/epoch, "
        f"batch_size={e.BATCH_SIZE}, "
        f"weights=[delaunay={e.DELAUNAY_WEIGHT}, tree={e.TREE_WEIGHT}, grid={e.GRID_WEIGHT}]"
    )

    # train_data is None because data is infinite
    return train_loader, None


# =====================================================================
# Hook: load_valid_data
# =====================================================================


@experiment.hook("load_valid_data", default=False, replace=True)
def load_valid_data(e, hypernet, rw_config, feature_bins, device):
    """Generate a fixed validation set from the generators."""
    domain = ColoredGraphDomain()
    generators = _make_generators(e)
    mixed_ds = _make_mixed_dataset(e, generators, domain)

    valid_graphs = list(itertools.islice(mixed_ds, e.NUM_VALID_GRAPHS))
    e.log(f"Generated {len(valid_graphs)} validation graphs")

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
    """Compute marginals from a freshly generated sample.

    Since training data is infinite (``train_data is None``), we
    generate a separate fixed sample for statistics.
    """
    domain = ColoredGraphDomain()
    generators = _make_generators(e)
    mixed_ds = _make_mixed_dataset(e, generators, domain)

    e.log(f"Generating {e.NUM_STATISTICS_GRAPHS} graphs for statistics...")

    stat_graphs = list(itertools.islice(mixed_ds, e.NUM_STATISTICS_GRAPHS))

    # For marginal computation we only need x (one-hot) and
    # edge_index / edge_attr.  HDC vectors are not used.
    from graph_hdc.domains.color.preprocessing import nx_to_pyg
    from graph_hdc.models.flow_edge_decoder import raw_features_to_onehot

    base_bins = list(e.FEATURE_BINS)

    stat_data: list[Data] = []
    for graph in stat_graphs:
        data = nx_to_pyg(graph)
        x_onehot = raw_features_to_onehot(data.x, feature_bins=base_bins)
        num_edges = data.edge_index.size(1)
        edge_attr = torch.zeros(num_edges, e.NUM_EDGE_CLASSES)
        if num_edges > 0:
            edge_attr[:, 1] = 1.0
        stat_data.append(Data(
            x=x_onehot,
            edge_index=data.edge_index,
            edge_attr=edge_attr,
        ))

    edge_marginals = compute_edge_marginals(
        stat_data, num_edge_classes=e.NUM_EDGE_CLASSES,
    )
    node_counts = compute_node_counts(stat_data)
    max_nodes = int(node_counts.shape[0]) - 1
    size_edge_marginals = compute_size_edge_marginals(
        stat_data,
        max_nodes + 10,
        num_edge_classes=e.NUM_EDGE_CLASSES,
    )

    e.log(f"Statistics from {len(stat_data)} graphs: max_nodes={max_nodes}")
    e.log(f"Edge marginals: {edge_marginals}")

    return edge_marginals, node_counts, max_nodes, size_edge_marginals


# =====================================================================
# Testing mode
# =====================================================================


@experiment.testing
def testing(e: Experiment) -> None:
    e.EPOCHS = 2
    e.BATCH_SIZE = 4
    e.STEPS_PER_EPOCH = 5
    e.NUM_VALID_GRAPHS = 20
    e.NUM_STATISTICS_GRAPHS = 50
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
    e.MIN_NODES = 4
    e.MAX_NODES = 10


experiment.run_if_main()
