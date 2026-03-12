"""Tests for color graph generators and streaming datasets."""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest

from graph_hdc.domains.base import MixedStreamDataset
from graph_hdc.domains.color.domain import ColoredGraphDomain, NUM_COLORS
from graph_hdc.domains.color.generators import (
    ColorGraphGenerator,
    DelaunayMeshGenerator,
    GridSubgraphGenerator,
    RandomTreeGenerator,
)
from graph_hdc.domains.color.datasets import ColorGraphStreamingDataset


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture()
def domain():
    return ColoredGraphDomain()


GENERATOR_CLASSES = [DelaunayMeshGenerator, RandomTreeGenerator, GridSubgraphGenerator]

MIN_NODES = 4
MAX_NODES = 40
MAX_DEGREE = 6
N_SAMPLES = 20  # graphs per test


def _make_generator(cls, seed=42, **kwargs):
    return cls(
        min_nodes=MIN_NODES,
        max_nodes=MAX_NODES,
        max_degree=MAX_DEGREE,
        seed=seed,
        **kwargs,
    )


# ===================================================================
# ColorGraphGenerator contract tests
# ===================================================================


class TestGeneratorContract:
    """Tests that apply to all generator types."""

    @pytest.mark.parametrize("cls", GENERATOR_CLASSES)
    def test_generates_graph(self, cls):
        gen = _make_generator(cls)
        graph = gen.generate()
        assert isinstance(graph, nx.Graph)

    @pytest.mark.parametrize("cls", GENERATOR_CLASSES)
    def test_node_count_in_range(self, cls):
        gen = _make_generator(cls)
        for _ in range(N_SAMPLES):
            graph = gen.generate()
            assert MIN_NODES <= graph.number_of_nodes() <= MAX_NODES, (
                f"{cls.__name__}: got {graph.number_of_nodes()} nodes"
            )

    @pytest.mark.parametrize("cls", GENERATOR_CLASSES)
    def test_connected(self, cls):
        gen = _make_generator(cls)
        for _ in range(N_SAMPLES):
            graph = gen.generate()
            assert nx.is_connected(graph), f"{cls.__name__}: disconnected graph"

    @pytest.mark.parametrize("cls", GENERATOR_CLASSES)
    def test_max_degree(self, cls):
        gen = _make_generator(cls)
        for _ in range(N_SAMPLES):
            graph = gen.generate()
            max_deg = max(dict(graph.degree()).values())
            assert max_deg <= MAX_DEGREE, (
                f"{cls.__name__}: max degree {max_deg} > {MAX_DEGREE}"
            )

    @pytest.mark.parametrize("cls", GENERATOR_CLASSES)
    def test_no_self_loops(self, cls):
        gen = _make_generator(cls)
        for _ in range(N_SAMPLES):
            graph = gen.generate()
            assert nx.number_of_selfloops(graph) == 0

    @pytest.mark.parametrize("cls", GENERATOR_CLASSES)
    def test_simple_graph(self, cls):
        gen = _make_generator(cls)
        for _ in range(N_SAMPLES):
            graph = gen.generate()
            assert isinstance(graph, nx.Graph)
            assert not isinstance(graph, nx.MultiGraph)

    @pytest.mark.parametrize("cls", GENERATOR_CLASSES)
    def test_contiguous_node_labels(self, cls):
        gen = _make_generator(cls)
        for _ in range(N_SAMPLES):
            graph = gen.generate()
            assert sorted(graph.nodes()) == list(range(graph.number_of_nodes()))

    @pytest.mark.parametrize("cls", GENERATOR_CLASSES)
    def test_has_features_and_color(self, cls):
        gen = _make_generator(cls)
        graph = gen.generate()
        for node in graph.nodes():
            assert "features" in graph.nodes[node]
            assert "color" in graph.nodes[node]
            feats = graph.nodes[node]["features"]
            assert len(feats) == 1
            assert 0 <= feats[0] < NUM_COLORS

    @pytest.mark.parametrize("cls", GENERATOR_CLASSES)
    def test_domain_validates(self, cls, domain):
        gen = _make_generator(cls)
        for _ in range(5):
            graph = gen.generate()
            domain.validate(graph)

    @pytest.mark.parametrize("cls", GENERATOR_CLASSES)
    def test_planarity(self, cls):
        gen = _make_generator(cls)
        for _ in range(N_SAMPLES):
            graph = gen.generate()
            assert nx.check_planarity(graph)[0], (
                f"{cls.__name__}: generated non-planar graph"
            )

    @pytest.mark.parametrize("cls", GENERATOR_CLASSES)
    def test_seed_reproducibility(self, cls):
        g1 = _make_generator(cls, seed=123)
        g2 = _make_generator(cls, seed=123)
        graph1 = g1.generate()
        graph2 = g2.generate()
        assert graph1.number_of_nodes() == graph2.number_of_nodes()
        assert graph1.number_of_edges() == graph2.number_of_edges()
        for node in graph1.nodes():
            assert graph1.nodes[node]["features"] == graph2.nodes[node]["features"]

    @pytest.mark.parametrize("cls", GENERATOR_CLASSES)
    def test_rng_override(self, cls):
        gen = _make_generator(cls, seed=0)
        rng = np.random.default_rng(999)
        graph = gen.generate(rng=rng)
        assert isinstance(graph, nx.Graph)
        assert MIN_NODES <= graph.number_of_nodes() <= MAX_NODES


# ===================================================================
# Color assignment tests
# ===================================================================


class TestColorAssignment:

    def test_uniform_distribution(self):
        """With enough samples, all 17 colors should appear."""
        gen = _make_generator(DelaunayMeshGenerator, seed=42)
        seen_colors = set()
        for _ in range(100):
            graph = gen.generate()
            for node in graph.nodes():
                seen_colors.add(graph.nodes[node]["color"])
        assert len(seen_colors) == NUM_COLORS

    def test_color_subset(self):
        subset = [0, 3, 7]
        gen = _make_generator(
            RandomTreeGenerator,
            seed=42,
            color_subset=subset,
        )
        for _ in range(N_SAMPLES):
            graph = gen.generate()
            for node in graph.nodes():
                assert graph.nodes[node]["color"] in subset

    def test_color_weights(self):
        """Heavily weighted toward color 0 — it should dominate."""
        weights = [100.0] + [1.0] * 16
        gen = _make_generator(
            RandomTreeGenerator,
            seed=42,
            color_weights=weights,
        )
        color_counts = {i: 0 for i in range(NUM_COLORS)}
        for _ in range(50):
            graph = gen.generate()
            for node in graph.nodes():
                color_counts[graph.nodes[node]["color"]] += 1
        assert color_counts[0] > sum(color_counts[i] for i in range(1, NUM_COLORS))

    def test_color_subset_with_weights(self):
        subset = [2, 5]
        weights = [0.9, 0.1]
        gen = _make_generator(
            GridSubgraphGenerator,
            seed=42,
            color_subset=subset,
            color_weights=weights,
        )
        for _ in range(N_SAMPLES):
            graph = gen.generate()
            for node in graph.nodes():
                assert graph.nodes[node]["color"] in subset


# ===================================================================
# Generator-specific tests
# ===================================================================


class TestDelaunayMeshGenerator:

    def test_has_cycles(self):
        """Delaunay graphs should have cycles (not just a tree)."""
        gen = _make_generator(DelaunayMeshGenerator, seed=42)
        has_cycles = False
        for _ in range(10):
            graph = gen.generate()
            n = graph.number_of_nodes()
            e = graph.number_of_edges()
            if e > n - 1:
                has_cycles = True
                break
        assert has_cycles

    def test_edge_keep_fraction_full(self):
        gen = _make_generator(DelaunayMeshGenerator, seed=42, edge_keep_fraction=1.0)
        graph = gen.generate()
        n = graph.number_of_nodes()
        # Full Delaunay: typically ~3n - 6 edges for planar
        assert graph.number_of_edges() >= n - 1

    def test_edge_keep_fraction_minimal(self):
        gen = _make_generator(DelaunayMeshGenerator, seed=42, edge_keep_fraction=0.0)
        graph = gen.generate()
        n = graph.number_of_nodes()
        # Only spanning tree edges remain
        assert graph.number_of_edges() == n - 1


class TestRandomTreeGenerator:

    def test_is_tree(self):
        gen = _make_generator(RandomTreeGenerator, seed=42)
        for _ in range(N_SAMPLES):
            graph = gen.generate()
            assert nx.is_tree(graph)

    def test_no_cycles(self):
        gen = _make_generator(RandomTreeGenerator, seed=42)
        for _ in range(N_SAMPLES):
            graph = gen.generate()
            assert graph.number_of_edges() == graph.number_of_nodes() - 1


class TestGridSubgraphGenerator:

    def test_has_four_cycles(self):
        """Grid subgraphs should typically have 4-cycles."""
        gen = _make_generator(GridSubgraphGenerator, seed=42, edge_keep_fraction=1.0)
        has_4_cycle = False
        for _ in range(10):
            graph = gen.generate()
            for node in graph.nodes():
                neighbors = set(graph.neighbors(node))
                for n1 in neighbors:
                    for n2 in neighbors:
                        if n1 != n2 and graph.has_edge(n1, n2):
                            # Found a triangle, but grids should have 4-cycles
                            pass
                        if n1 < n2:
                            common = set(graph.neighbors(n1)) & set(graph.neighbors(n2))
                            common.discard(node)
                            if common:
                                has_4_cycle = True
                                break
                    if has_4_cycle:
                        break
                if has_4_cycle:
                    break
            if has_4_cycle:
                break
        assert has_4_cycle

    def test_max_degree_at_most_4(self):
        """Grid subgraph nodes have degree <= 4 (grid property)."""
        gen = _make_generator(GridSubgraphGenerator, seed=42, edge_keep_fraction=1.0)
        for _ in range(N_SAMPLES):
            graph = gen.generate()
            max_deg = max(dict(graph.degree()).values())
            assert max_deg <= 4


# ===================================================================
# ColorGraphStreamingDataset
# ===================================================================


class TestColorGraphStreamingDataset:

    def test_is_infinite(self, domain):
        gen = _make_generator(RandomTreeGenerator, seed=42)
        ds = ColorGraphStreamingDataset(gen, domain)
        assert ds.is_finite is False

    def test_no_length(self, domain):
        gen = _make_generator(RandomTreeGenerator, seed=42)
        ds = ColorGraphStreamingDataset(gen, domain)
        with pytest.raises(TypeError):
            len(ds)

    def test_yields_valid_graphs(self, domain):
        gen = _make_generator(DelaunayMeshGenerator, seed=42)
        ds = ColorGraphStreamingDataset(gen, domain)
        for graph in itertools.islice(ds, 10):
            assert isinstance(graph, nx.Graph)
            domain.validate(graph)

    def test_domain_attribute(self, domain):
        gen = _make_generator(RandomTreeGenerator, seed=42)
        ds = ColorGraphStreamingDataset(gen, domain)
        assert ds.domain is domain


# ===================================================================
# MixedStreamDataset
# ===================================================================


class TestMixedStreamDataset:

    def test_mixes_sources(self, domain):
        gen1 = _make_generator(RandomTreeGenerator, seed=10)
        gen2 = _make_generator(DelaunayMeshGenerator, seed=20)
        ds1 = ColorGraphStreamingDataset(gen1, domain)
        ds2 = ColorGraphStreamingDataset(gen2, domain)

        mixed = MixedStreamDataset([ds1, ds2], seed=42)
        graphs = list(itertools.islice(mixed, 20))
        assert len(graphs) == 20
        for g in graphs:
            assert isinstance(g, nx.Graph)
            domain.validate(g)

    def test_is_infinite(self, domain):
        gen = _make_generator(RandomTreeGenerator, seed=42)
        ds = ColorGraphStreamingDataset(gen, domain)
        mixed = MixedStreamDataset([ds])
        assert mixed.is_finite is False

    def test_no_length(self, domain):
        gen = _make_generator(RandomTreeGenerator, seed=42)
        ds = ColorGraphStreamingDataset(gen, domain)
        mixed = MixedStreamDataset([ds])
        with pytest.raises(TypeError):
            len(mixed)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least one dataset"):
            MixedStreamDataset([])

    def test_weights(self, domain):
        """With extreme weights, one source should dominate."""
        gen_tree = _make_generator(RandomTreeGenerator, seed=10)
        gen_grid = _make_generator(GridSubgraphGenerator, seed=20)
        ds_tree = ColorGraphStreamingDataset(gen_tree, domain)
        ds_grid = ColorGraphStreamingDataset(gen_grid, domain)

        # 99% tree, 1% grid
        mixed = MixedStreamDataset(
            [ds_tree, ds_grid],
            weights=[99.0, 1.0],
            seed=42,
        )

        tree_count = 0
        n_total = 100
        for graph in itertools.islice(mixed, n_total):
            if nx.is_tree(graph):
                tree_count += 1
        # Trees should dominate
        assert tree_count > 80

    def test_equal_weights(self, domain):
        gen1 = _make_generator(RandomTreeGenerator, seed=10)
        gen2 = _make_generator(RandomTreeGenerator, seed=20)
        ds1 = ColorGraphStreamingDataset(gen1, domain)
        ds2 = ColorGraphStreamingDataset(gen2, domain)

        mixed = MixedStreamDataset([ds1, ds2], weights=None, seed=42)
        graphs = list(itertools.islice(mixed, 20))
        assert len(graphs) == 20

    def test_three_sources(self, domain):
        generators = [
            _make_generator(cls, seed=i)
            for i, cls in enumerate(GENERATOR_CLASSES)
        ]
        datasets = [
            ColorGraphStreamingDataset(g, domain) for g in generators
        ]
        mixed = MixedStreamDataset(datasets, seed=42)
        graphs = list(itertools.islice(mixed, 30))
        assert len(graphs) == 30
        for g in graphs:
            domain.validate(g)

    def test_seed_reproducibility(self, domain):
        def make_mixed(seed):
            gen1 = _make_generator(RandomTreeGenerator, seed=10)
            gen2 = _make_generator(DelaunayMeshGenerator, seed=20)
            ds1 = ColorGraphStreamingDataset(gen1, domain)
            ds2 = ColorGraphStreamingDataset(gen2, domain)
            return MixedStreamDataset([ds1, ds2], seed=seed)

        m1 = make_mixed(42)
        m2 = make_mixed(42)
        for g1, g2 in itertools.islice(zip(m1, m2), 10):
            assert g1.number_of_nodes() == g2.number_of_nodes()
            assert g1.number_of_edges() == g2.number_of_edges()
