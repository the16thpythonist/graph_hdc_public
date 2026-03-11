"""Contract tests for the GraphDomain ABC and related abstractions."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar

import networkx as nx
import pytest

from graph_hdc.domains.base import (
    BoolEncoder,
    DomainResult,
    GraphDataset,
    GraphDomain,
    IntegerEncoder,
    OneHotEncoder,
)


# ---------------------------------------------------------------------------
# Minimal test domain
# ---------------------------------------------------------------------------

class MinimalDomain(GraphDomain):
    """Minimal domain for contract testing (3 colors, 2 shapes)."""

    feature_schema: ClassVar[dict[str, Any]] = {
        "color": OneHotEncoder(["red", "green", "blue"]),
        "shape": OneHotEncoder(["circle", "square"]),
    }

    def process(self, data: str) -> nx.Graph:
        G = nx.Graph()
        # data is "color_idx,shape_idx" for a single node
        parts = [int(x) for x in data.split(",")]
        G.add_node(0, features=parts, color=parts[0], shape=parts[1])
        return G

    def unprocess(self, graph: nx.Graph) -> DomainResult:
        return DomainResult(domain_object=graph, is_valid=True, canonical_key="test")

    def visualize(self, ax: Any, obj: Any, **kwargs: Any) -> None:
        pass


class MinimalDataset(GraphDataset):
    """Finite dataset returning pre-built graphs."""

    def __init__(self, domain: GraphDomain, items: list[nx.Graph]) -> None:
        super().__init__(domain)
        self._items = items

    def __iter__(self) -> Iterator[nx.Graph]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


class StreamingDataset(GraphDataset):
    """Streaming dataset that doesn't support __len__."""

    def __init__(self, domain: GraphDomain) -> None:
        super().__init__(domain)

    def __iter__(self) -> Iterator[nx.Graph]:
        yield from ()

    @property
    def is_finite(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Feature bins
# ---------------------------------------------------------------------------

def test_feature_bins_derived_correctly():
    domain = MinimalDomain()
    assert domain.feature_bins == [3, 2]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_passes_valid_graph():
    domain = MinimalDomain()
    G = nx.Graph()
    G.add_node(0, features=[0, 1], color=0, shape=1)
    G.add_node(1, features=[2, 0], color=2, shape=0)
    G.add_edge(0, 1)
    domain.validate(G)  # should not raise


def test_validate_rejects_missing_features():
    domain = MinimalDomain()
    G = nx.Graph()
    G.add_node(0, color=0, shape=1)  # no "features"
    with pytest.raises(ValueError, match="missing 'features'"):
        domain.validate(G)


def test_validate_rejects_wrong_length():
    domain = MinimalDomain()
    G = nx.Graph()
    G.add_node(0, features=[0], color=0)  # only 1 feature, expected 2
    with pytest.raises(ValueError, match="expected 2 features, got 1"):
        domain.validate(G)


def test_validate_rejects_out_of_range():
    domain = MinimalDomain()
    G = nx.Graph()
    G.add_node(0, features=[5, 0], color=5, shape=0)  # color 5 out of [0, 3)
    with pytest.raises(ValueError, match="out of range"):
        domain.validate(G)


def test_validate_rejects_mismatched_named_attrs():
    domain = MinimalDomain()
    G = nx.Graph()
    G.add_node(0, features=[0, 1], color=1, shape=1)  # color=1 != features[0]=0
    with pytest.raises(ValueError, match="named attr 1 != features"):
        domain.validate(G)


def test_validate_rejects_missing_named_attr():
    domain = MinimalDomain()
    G = nx.Graph()
    G.add_node(0, features=[0, 1], color=0)  # missing "shape"
    with pytest.raises(ValueError, match="missing named attribute 'shape'"):
        domain.validate(G)


# ---------------------------------------------------------------------------
# GraphDataset
# ---------------------------------------------------------------------------

def test_graph_dataset_iteration():
    domain = MinimalDomain()
    g1 = nx.Graph()
    g1.add_node(0, features=[0, 0], color=0, shape=0)
    g2 = nx.Graph()
    g2.add_node(0, features=[1, 1], color=1, shape=1)
    ds = MinimalDataset(domain, [g1, g2])
    graphs = list(ds)
    assert len(graphs) == 2
    assert all(isinstance(g, nx.Graph) for g in graphs)


def test_graph_dataset_length():
    domain = MinimalDomain()
    ds = MinimalDataset(domain, [nx.Graph(), nx.Graph(), nx.Graph()])
    assert len(ds) == 3


def test_graph_dataset_streaming_raises_typeerror():
    domain = MinimalDomain()
    ds = StreamingDataset(domain)
    with pytest.raises(TypeError, match="does not have a fixed length"):
        len(ds)
    assert ds.is_finite is False


# ---------------------------------------------------------------------------
# OneHotEncoder
# ---------------------------------------------------------------------------

def test_onehot_encoder_roundtrip():
    enc = OneHotEncoder(["red", "green", "blue"])
    assert enc.num_bins == 3
    for i, val in enumerate(["red", "green", "blue"]):
        assert enc.encode(val) == i
        assert enc.decode(i) == val


def test_onehot_encoder_unknown_raises():
    enc = OneHotEncoder(["red", "green", "blue"])
    with pytest.raises(ValueError, match="Unknown value"):
        enc.encode("purple")


def test_onehot_encoder_decode_out_of_range():
    enc = OneHotEncoder(["a", "b"])
    with pytest.raises(ValueError, match="out of range"):
        enc.decode(5)


# ---------------------------------------------------------------------------
# IntegerEncoder
# ---------------------------------------------------------------------------

def test_integer_encoder_roundtrip():
    enc = IntegerEncoder(max_val=4)
    assert enc.num_bins == 5
    for i in range(5):
        assert enc.encode(i) == i
        assert enc.decode(i) == i


def test_integer_encoder_clips():
    enc = IntegerEncoder(max_val=3)
    assert enc.encode(-1) == 0
    assert enc.encode(10) == 3


def test_integer_encoder_decode_out_of_range():
    enc = IntegerEncoder(max_val=3)
    with pytest.raises(ValueError, match="out of range"):
        enc.decode(5)


# ---------------------------------------------------------------------------
# BoolEncoder
# ---------------------------------------------------------------------------

def test_bool_encoder_roundtrip():
    enc = BoolEncoder()
    assert enc.num_bins == 2
    assert enc.encode(True) == 1
    assert enc.encode(False) == 0
    assert enc.decode(1) is True
    assert enc.decode(0) is False


def test_bool_encoder_decode_out_of_range():
    enc = BoolEncoder()
    with pytest.raises(ValueError, match="out of range"):
        enc.decode(2)


# ---------------------------------------------------------------------------
# DomainResult
# ---------------------------------------------------------------------------

def test_domain_result_fields():
    result = DomainResult(
        domain_object="test",
        is_valid=True,
        canonical_key="key",
        properties={"score": 1.0},
    )
    assert result.domain_object == "test"
    assert result.is_valid is True
    assert result.canonical_key == "key"
    assert result.properties == {"score": 1.0}


def test_domain_result_defaults():
    result = DomainResult(domain_object=None, is_valid=False, canonical_key=None)
    assert result.properties == {}
