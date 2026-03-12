"""Tests for ColoredGraphDomain and CogilesDataset."""

from __future__ import annotations

import tempfile
from pathlib import Path

import networkx as nx
import pytest

from graph_hdc.domains.base import DomainResult
from graph_hdc.domains.color.domain import (
    COLOR_SYMBOLS,
    NUM_COLORS,
    SYMBOL_TO_IDX,
    ColoredGraphDomain,
)
from graph_hdc.domains.color.datasets import CogilesDataset


# ===================================================================
# ColoredGraphDomain basics
# ===================================================================

class TestColoredGraphDomain:

    @pytest.fixture()
    def domain(self):
        return ColoredGraphDomain()

    def test_feature_bins(self, domain):
        assert domain.feature_bins == [17]

    def test_feature_schema_keys(self, domain):
        assert list(domain.feature_schema.keys()) == ["color"]

    def test_color_count(self):
        assert NUM_COLORS == 17
        assert len(COLOR_SYMBOLS) == 17

    def test_symbol_to_idx_consistent(self):
        for i, sym in enumerate(COLOR_SYMBOLS):
            assert SYMBOL_TO_IDX[sym] == i


# ===================================================================
# process()
# ===================================================================

class TestProcess:

    @pytest.fixture()
    def domain(self):
        return ColoredGraphDomain()

    def test_single_node(self, domain):
        graph = domain.process("R")
        assert graph.number_of_nodes() == 1
        assert graph.number_of_edges() == 0
        assert graph.nodes[0]["features"] == [SYMBOL_TO_IDX["R"]]
        assert graph.nodes[0]["color"] == SYMBOL_TO_IDX["R"]

    def test_chain(self, domain):
        graph = domain.process("RGB")
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 2
        assert graph.nodes[0]["features"] == [SYMBOL_TO_IDX["R"]]
        assert graph.nodes[1]["features"] == [SYMBOL_TO_IDX["G"]]
        assert graph.nodes[2]["features"] == [SYMBOL_TO_IDX["B"]]

    def test_star(self, domain):
        graph = domain.process("R(G)(B)")
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 2
        # R is center, connected to G and B
        assert graph.has_edge(0, 1)
        assert graph.has_edge(0, 2)

    def test_cycle(self, domain):
        graph = domain.process("R-1GBYCK-1")
        assert graph.number_of_nodes() == 6
        # 5 chain edges + 1 anchor edge = 6
        assert graph.number_of_edges() == 6

    def test_disconnected(self, domain):
        graph = domain.process("RG.BY")
        assert graph.number_of_nodes() == 4
        assert graph.number_of_edges() == 2
        assert nx.number_connected_components(graph) == 2

    def test_two_letter_colors(self, domain):
        graph = domain.process("BrGrPiOl")
        assert graph.number_of_nodes() == 4
        assert graph.nodes[0]["features"] == [SYMBOL_TO_IDX["Br"]]
        assert graph.nodes[1]["features"] == [SYMBOL_TO_IDX["Gr"]]
        assert graph.nodes[2]["features"] == [SYMBOL_TO_IDX["Pi"]]
        assert graph.nodes[3]["features"] == [SYMBOL_TO_IDX["Ol"]]

    def test_validates(self, domain):
        graph = domain.process("RGB")
        domain.validate(graph)  # should not raise

    def test_stores_canonical_cogiles(self, domain):
        graph = domain.process("RGB")
        assert "cogiles" in graph.graph

    def test_invalid_string_raises(self, domain):
        with pytest.raises(ValueError, match="Cannot parse COGILES"):
            domain.process("")

    def test_all_colors_encodable(self, domain):
        """Every canonical color symbol can be processed."""
        for sym in COLOR_SYMBOLS:
            graph = domain.process(sym)
            assert graph.number_of_nodes() == 1
            assert graph.nodes[0]["features"] == [SYMBOL_TO_IDX[sym]]


# ===================================================================
# unprocess()
# ===================================================================

class TestUnprocess:

    @pytest.fixture()
    def domain(self):
        return ColoredGraphDomain()

    def test_roundtrip(self, domain):
        graph = domain.process("R(G)(B)")
        result = domain.unprocess(graph)
        assert isinstance(result, DomainResult)
        assert result.is_valid is True
        assert result.domain_object is not None
        assert result.canonical_key is not None

    def test_canonical_key_is_cogiles_string(self, domain):
        graph = domain.process("RGB")
        result = domain.unprocess(graph)
        assert isinstance(result.canonical_key, str)
        assert len(result.canonical_key) > 0

    def test_canonical_key_deterministic(self, domain):
        """Same graph produces the same canonical key."""
        g1 = domain.process("R(G)(B)")
        g2 = domain.process("R(G)(B)")
        r1 = domain.unprocess(g1)
        r2 = domain.unprocess(g2)
        assert r1.canonical_key == r2.canonical_key

    def test_empty_graph_invalid(self, domain):
        result = domain.unprocess(nx.Graph())
        assert result.is_valid is False

    def test_out_of_range_color_invalid(self, domain):
        G = nx.Graph()
        G.add_node(0, features=[99], color=99)
        result = domain.unprocess(G)
        assert result.is_valid is False

    def test_missing_features_invalid(self, domain):
        G = nx.Graph()
        G.add_node(0)  # no features
        result = domain.unprocess(G)
        assert result.is_valid is False

    def test_unprocessed_has_rgb_colors(self, domain):
        """The unprocessed graph has COGILES-compatible RGB color attributes."""
        graph = domain.process("R")
        result = domain.unprocess(graph)
        cogiles_graph = result.domain_object
        node_data = cogiles_graph.nodes[0]
        assert isinstance(node_data["color"], tuple)
        assert len(node_data["color"]) == 3
        assert "symbol" in node_data
        assert "color_name" in node_data

    def test_disconnected_graph_invalid(self, domain):
        """Disconnected graphs are invalid."""
        G = nx.Graph()
        G.add_node(0, features=[0], color=0)
        G.add_node(1, features=[1], color=1)
        # No edges — disconnected
        result = domain.unprocess(G)
        assert result.is_valid is False

    def test_high_degree_invalid(self, domain):
        """Graphs with degree > MAX_DEGREE are invalid."""
        G = nx.Graph()
        # Star graph with center degree 7 (> MAX_DEGREE=6)
        G.add_node(0, features=[0], color=0)
        for i in range(1, 8):
            G.add_node(i, features=[i % 17], color=i % 17)
            G.add_edge(0, i)
        result = domain.unprocess(G)
        assert result.is_valid is False

    def test_non_planar_invalid(self, domain):
        """Non-planar graphs (e.g. K5) are invalid."""
        G = nx.complete_graph(5)
        for n in G.nodes():
            G.nodes[n]["features"] = [n % 17]
            G.nodes[n]["color"] = n % 17
        result = domain.unprocess(G)
        assert result.is_valid is False

    def test_planar_connected_low_degree_valid(self, domain):
        """A simple path graph should be valid."""
        G = nx.path_graph(5)
        for n in G.nodes():
            G.nodes[n]["features"] = [n % 17]
            G.nodes[n]["color"] = n % 17
        result = domain.unprocess(G)
        assert result.is_valid is True


# ===================================================================
# Roundtrip consistency
# ===================================================================

class TestRoundtrip:

    @pytest.fixture()
    def domain(self):
        return ColoredGraphDomain()

    @pytest.mark.parametrize("cogiles_str", [
        "R",
        "RG",
        "RGB",
        "R(G)(B)",
        "R-1GBYCK-1",
        "Y(G)(G)(G)",
        "BrGrPiOl",
    ])
    def test_process_unprocess_preserves_structure(self, domain, cogiles_str):
        graph = domain.process(cogiles_str)
        result = domain.unprocess(graph)
        assert result.is_valid is True

        # Check node count and edge count match
        cogiles_graph = result.domain_object
        assert cogiles_graph.number_of_nodes() == graph.number_of_nodes()
        assert cogiles_graph.number_of_edges() == graph.number_of_edges()


# ===================================================================
# CogilesDataset
# ===================================================================

class TestCogilesDataset:

    def test_iteration(self):
        domain = ColoredGraphDomain()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("R\n")
            f.write("RGB\n")
            f.write("R(G)(B)\n")
            path = f.name
        try:
            ds = CogilesDataset(path, domain)
            graphs = list(ds)
            assert len(graphs) == 3
            for g in graphs:
                assert "features" in g.nodes[0]
                assert len(g.nodes[0]["features"]) == 1
        finally:
            Path(path).unlink()

    def test_length(self):
        domain = ColoredGraphDomain()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("R\nRG\nRGB\n")
            path = f.name
        try:
            ds = CogilesDataset(path, domain)
            assert len(ds) == 3
        finally:
            Path(path).unlink()

    def test_skip_empty_lines(self):
        domain = ColoredGraphDomain()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("R\n\n\nRG\n")
            path = f.name
        try:
            ds = CogilesDataset(path, domain)
            assert len(ds) == 2
        finally:
            Path(path).unlink()

    def test_skip_header(self):
        domain = ColoredGraphDomain()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("cogiles\n")
            f.write("R\n")
            f.write("RG\n")
            path = f.name
        try:
            ds = CogilesDataset(path, domain)
            assert len(ds) == 2
        finally:
            Path(path).unlink()

    def test_skip_invalid_during_iteration(self):
        """Invalid COGILES strings are skipped during iteration."""
        domain = ColoredGraphDomain()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("R\n")
            f.write("ZZZINVALID\n")
            f.write("RG\n")
            path = f.name
        try:
            ds = CogilesDataset(path, domain)
            assert len(ds) == 3  # all 3 loaded as strings
            graphs = list(ds)
            assert len(graphs) == 2  # only R and RG yielded
        finally:
            Path(path).unlink()
