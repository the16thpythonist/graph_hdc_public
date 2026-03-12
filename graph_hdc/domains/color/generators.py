"""
Graph generators for the color domain.

Each generator produces random planar, connected graphs with node colors
drawn from the :class:`ColoredGraphDomain` palette.  Three structurally
distinct generators are provided:

- :class:`DelaunayMeshGenerator` — dense, triangulated mesh graphs
- :class:`RandomTreeGenerator` — sparse, acyclic trees
- :class:`GridSubgraphGenerator` — regular, lattice-like subgraphs
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import networkx as nx
import numpy as np


class ColorGraphGenerator(ABC):
    """Abstract base for colored graph generators.

    Subclasses implement :meth:`generate_topology` to produce an unlabeled
    graph structure.  Color assignment is handled by the base class via
    :meth:`_assign_colors`.

    Parameters
    ----------
    min_nodes : int
        Minimum number of nodes (inclusive).
    max_nodes : int
        Maximum number of nodes (inclusive).
    max_degree : int
        Maximum allowed degree for any node.
    num_colors : int
        Total number of available colors.
    color_weights : list[float] or None
        Sampling weights for *eligible* colors.  Length must match
        ``color_subset`` (if given) or ``num_colors``.  Normalized
        internally.  ``None`` means uniform.
    color_subset : list[int] or None
        Indices of eligible colors.  ``None`` means all colors
        ``[0, num_colors)`` are eligible.
    seed : int or None
        Seed for the internal RNG.
    """

    def __init__(
        self,
        *,
        min_nodes: int = 4,
        max_nodes: int = 40,
        max_degree: int = 6,
        num_colors: int = 17,
        color_weights: list[float] | None = None,
        color_subset: list[int] | None = None,
        seed: int | None = None,
    ) -> None:
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.max_degree = max_degree
        self.num_colors = num_colors
        self.color_subset = color_subset
        self.color_weights = color_weights
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def generate_topology(self, rng: np.random.Generator) -> nx.Graph:
        """Generate an unlabeled graph structure.

        Must return a connected, planar, simple graph with:

        - Node count in ``[min_nodes, max_nodes]``
        - Max degree ``<= max_degree``
        - No self-loops or parallel edges
        - Nodes labeled as contiguous integers starting from 0
        """
        ...

    def generate(self, rng: np.random.Generator | None = None) -> nx.Graph:
        """Generate a colored graph conforming to the color domain schema.

        Parameters
        ----------
        rng : numpy.random.Generator or None
            Optional RNG override.  If ``None``, uses ``self.rng``.

        Returns
        -------
        nx.Graph
            Graph with ``"features"`` (``list[int]``) and ``"color"``
            (``int``) attributes on each node.
        """
        if rng is None:
            rng = self.rng

        graph = self.generate_topology(rng)
        self._assign_colors(graph, rng)
        return graph

    def _assign_colors(self, graph: nx.Graph, rng: np.random.Generator) -> None:
        """Assign random colors to all nodes in-place."""
        eligible = (
            self.color_subset
            if self.color_subset is not None
            else list(range(self.num_colors))
        )

        weights = None
        if self.color_weights is not None:
            w = np.array(self.color_weights, dtype=float)
            weights = w / w.sum()

        for node in graph.nodes():
            color_idx = int(rng.choice(eligible, p=weights))
            graph.nodes[node]["features"] = [color_idx]
            graph.nodes[node]["color"] = color_idx


# ---------------------------------------------------------------------------
# Topology helpers (shared across generators)
# ---------------------------------------------------------------------------


def _enforce_max_degree(
    G: nx.Graph,
    max_degree: int,
    rng: np.random.Generator,
) -> None:
    """Remove edges from nodes exceeding *max_degree*.

    Prefers removing non-spanning-tree edges to preserve connectivity.
    When only tree edges remain, removes those too (the caller is
    expected to handle any resulting disconnection via
    :func:`_largest_component`).
    """
    tree_edges = set(map(frozenset, nx.minimum_spanning_tree(G).edges()))

    for node in list(G.nodes()):
        if node not in G:
            continue
        while G.degree(node) > max_degree:
            neighbors = list(G.neighbors(node))
            rng.shuffle(neighbors)
            # Prefer non-tree edges
            non_tree = [nb for nb in neighbors if frozenset({node, nb}) not in tree_edges]
            if non_tree:
                G.remove_edge(node, non_tree[0])
            else:
                # Remove a tree edge — may disconnect the graph
                G.remove_edge(node, neighbors[0])


def _largest_component(G: nx.Graph) -> nx.Graph:
    """Return the largest connected component, relabeled to 0..n-1."""
    if nx.is_connected(G):
        return nx.convert_node_labels_to_integers(G)

    largest = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(largest).copy())


def _thin_edges(
    G: nx.Graph,
    keep_fraction: float,
    rng: np.random.Generator,
) -> None:
    """Remove a random fraction of non-spanning-tree edges in-place."""
    tree_edges = set(map(frozenset, nx.minimum_spanning_tree(G).edges()))
    non_tree = [e for e in G.edges() if frozenset(e) not in tree_edges]

    n_remove = int(len(non_tree) * (1 - keep_fraction))
    if n_remove > 0 and len(non_tree) > 0:
        indices = rng.choice(len(non_tree), size=n_remove, replace=False)
        for idx in indices:
            G.remove_edge(*non_tree[idx])


# ---------------------------------------------------------------------------
# Concrete generators
# ---------------------------------------------------------------------------


class DelaunayMeshGenerator(ColorGraphGenerator):
    """Generates planar graphs via Delaunay triangulation of random 2D points.

    Produces dense, organic, mesh-like graphs with many triangular cycles.
    Edge density is controlled by ``edge_keep_fraction``: 1.0 keeps the full
    triangulation, lower values thin non-spanning-tree edges.

    Parameters
    ----------
    edge_keep_fraction : float
        Fraction of non-spanning-tree edges to keep (0.0 = spanning tree
        only, 1.0 = full Delaunay triangulation).
    **kwargs
        Forwarded to :class:`ColorGraphGenerator`.
    """

    def __init__(self, *, edge_keep_fraction: float = 0.7, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.edge_keep_fraction = edge_keep_fraction

    def generate_topology(self, rng: np.random.Generator) -> nx.Graph:
        from scipy.spatial import Delaunay

        n = int(rng.integers(self.min_nodes, self.max_nodes + 1))
        points = rng.uniform(0, 1, size=(n, 2))
        tri = Delaunay(points)

        G = nx.Graph()
        G.add_nodes_from(range(n))
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    G.add_edge(int(simplex[i]), int(simplex[j]))

        _thin_edges(G, self.edge_keep_fraction, rng)
        _enforce_max_degree(G, self.max_degree, rng)
        G = _largest_component(G)

        if len(G) < self.min_nodes:
            return self.generate_topology(rng)

        return G


class RandomTreeGenerator(ColorGraphGenerator):
    """Generates random trees via iterative random attachment.

    Produces sparse, acyclic, branching structures.  Each new node attaches
    to a uniformly chosen existing node that has not reached ``max_degree``.
    """

    def generate_topology(self, rng: np.random.Generator) -> nx.Graph:
        n = int(rng.integers(self.min_nodes, self.max_nodes + 1))

        G = nx.Graph()
        G.add_node(0)

        for i in range(1, n):
            eligible = [v for v in G.nodes() if G.degree(v) < self.max_degree]
            if not eligible:
                break
            parent = int(rng.choice(eligible))
            G.add_edge(i, parent)

        return G


class GridSubgraphGenerator(ColorGraphGenerator):
    """Generates random connected subgraphs of 2D grid lattices.

    Produces regular, rectilinear graphs with many 4-cycles.  A grid
    slightly larger than the target size is created, then random nodes
    and edges are removed to add irregularity.

    Parameters
    ----------
    node_keep_fraction : float
        Fraction of grid nodes to keep before taking the largest
        connected component.
    edge_keep_fraction : float
        Fraction of non-spanning-tree edges to keep (applied after
        node removal).
    **kwargs
        Forwarded to :class:`ColorGraphGenerator`.
    """

    def __init__(
        self,
        *,
        node_keep_fraction: float = 0.7,
        edge_keep_fraction: float = 0.9,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.node_keep_fraction = node_keep_fraction
        self.edge_keep_fraction = edge_keep_fraction

    def generate_topology(self, rng: np.random.Generator) -> nx.Graph:
        n_target = int(rng.integers(self.min_nodes, self.max_nodes + 1))

        # Build a grid large enough to yield n_target nodes after removal.
        side = max(2, int(np.ceil(np.sqrt(n_target / self.node_keep_fraction))))
        G = nx.grid_2d_graph(side, side)

        # Remove random nodes down to roughly n_target.
        nodes = list(G.nodes())
        n_to_remove = max(0, len(nodes) - n_target)
        if n_to_remove > 0:
            remove_indices = rng.choice(len(nodes), size=n_to_remove, replace=False)
            G.remove_nodes_from([nodes[i] for i in remove_indices])

        if len(G) == 0:
            return self.generate_topology(rng)

        # Thin non-spanning-tree edges for variety.
        if nx.is_connected(G) and self.edge_keep_fraction < 1.0:
            _thin_edges(G, self.edge_keep_fraction, rng)

        G = _largest_component(G)

        if len(G) < self.min_nodes:
            return self.generate_topology(rng)

        return G
