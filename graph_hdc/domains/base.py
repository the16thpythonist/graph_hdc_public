"""
Core domain abstractions for GraphHDC.

Defines the GraphDomain ABC, GraphDataset ABC, FeatureEncoder protocol,
DomainResult dataclass, and built-in generic encoders (OneHotEncoder,
IntegerEncoder, BoolEncoder).

All classes here are domain-agnostic — no RDKit or other domain-specific
imports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# FeatureEncoder protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class FeatureEncoder(Protocol):
    """Protocol for domain feature encoders in the attribute map."""

    @property
    def num_bins(self) -> int:
        """Number of discrete values this feature can take."""
        ...

    def encode(self, value: Any) -> int:
        """Encode a domain value to an integer index."""
        ...

    def decode(self, index: int) -> Any:
        """Decode an integer index back to a domain value."""
        ...


# ---------------------------------------------------------------------------
# Built-in encoders (domain-agnostic)
# ---------------------------------------------------------------------------

class OneHotEncoder:
    """Fixed-vocabulary encoder. Maps values to integer indices."""

    def __init__(self, values: list) -> None:
        self.values = list(values)
        self._val_to_idx: dict[Any, int] = {v: i for i, v in enumerate(self.values)}

    @property
    def num_bins(self) -> int:
        return len(self.values)

    def encode(self, value: Any) -> int:
        try:
            return self._val_to_idx[value]
        except KeyError:
            raise ValueError(
                f"Unknown value {value!r}. Expected one of {self.values}"
            ) from None

    def decode(self, index: int) -> Any:
        if not (0 <= index < len(self.values)):
            raise ValueError(
                f"Index {index} out of range [0, {len(self.values)})"
            )
        return self.values[index]

    def __repr__(self) -> str:
        return f"OneHotEncoder({self.values!r})"


class IntegerEncoder:
    """Identity encoder for integers in [0, max_val]."""

    def __init__(self, max_val: int) -> None:
        self.max_val = max_val

    @property
    def num_bins(self) -> int:
        return self.max_val + 1

    def encode(self, value: int) -> int:
        return max(0, min(int(value), self.max_val))

    def decode(self, index: int) -> int:
        if not (0 <= index <= self.max_val):
            raise ValueError(
                f"Index {index} out of range [0, {self.max_val}]"
            )
        return index

    def __repr__(self) -> str:
        return f"IntegerEncoder(max_val={self.max_val})"


class BoolEncoder:
    """Binary encoder for boolean values."""

    @property
    def num_bins(self) -> int:
        return 2

    def encode(self, value: Any) -> int:
        return int(bool(value))

    def decode(self, index: int) -> bool:
        if index not in (0, 1):
            raise ValueError(f"Index {index} out of range [0, 1]")
        return bool(index)

    def __repr__(self) -> str:
        return "BoolEncoder()"


# ---------------------------------------------------------------------------
# DomainResult
# ---------------------------------------------------------------------------

@dataclass
class DomainResult:
    """Result of unprocess(): domain object + metadata."""

    domain_object: Any
    is_valid: bool
    canonical_key: str | None
    properties: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GraphDomain ABC
# ---------------------------------------------------------------------------

class GraphDomain(ABC):
    """Abstract interface for a graph domain.

    Subclasses declare ``feature_schema`` as a class variable mapping
    feature names to :class:`FeatureEncoder` instances.  Customization
    is done by subclassing and overriding ``feature_schema``.
    """

    feature_schema: ClassVar[dict[str, FeatureEncoder]]

    @property
    def feature_bins(self) -> list[int]:
        """Derived: list of cardinalities, one per feature dimension."""
        return [enc.num_bins for enc in self.feature_schema.values()]

    @abstractmethod
    def process(self, domain_repr: Any) -> nx.Graph:
        """Convert a domain representation to a NetworkX graph.

        The returned graph must have ``"features"`` (list[int]) and named
        attributes on each node, conforming to ``feature_schema``.
        """
        ...

    @abstractmethod
    def unprocess(self, graph: nx.Graph) -> DomainResult:
        """Convert a NetworkX graph back to a domain object."""
        ...

    @abstractmethod
    def visualize(self, ax: Any, domain_repr_or_graph: Any, **kwargs: Any) -> None:
        """Draw a domain object or graph onto the given Axes."""
        ...

    @property
    def metrics(self) -> dict[str, Callable]:
        """Registry of domain-specific metric functions."""
        return {}

    def validate(self, graph: nx.Graph) -> None:
        """Check that a graph conforms to the feature schema.

        Raises :class:`ValueError` with a clear message on mismatch.
        """
        schema_names = list(self.feature_schema.keys())
        bins = self.feature_bins

        for node, attrs in graph.nodes(data=True):
            feats = attrs.get("features")
            if feats is None:
                raise ValueError(f"Node {node} missing 'features' attribute")
            if len(feats) != len(bins):
                raise ValueError(
                    f"Node {node}: expected {len(bins)} features, got {len(feats)}"
                )
            for i, (val, num_bins) in enumerate(zip(feats, bins)):
                if not (0 <= val < num_bins):
                    raise ValueError(
                        f"Node {node}, feature '{schema_names[i]}': "
                        f"value {val} out of range [0, {num_bins})"
                    )
                name = schema_names[i]
                named_val = attrs.get(name)
                if named_val is None:
                    raise ValueError(
                        f"Node {node} missing named attribute '{name}'"
                    )
                if named_val != val:
                    raise ValueError(
                        f"Node {node}, '{name}': "
                        f"named attr {named_val} != features[{i}] {val}"
                    )


# ---------------------------------------------------------------------------
# GraphDataset ABC
# ---------------------------------------------------------------------------

class GraphDataset(ABC):
    """Abstract base for graph datasets."""

    def __init__(self, domain: GraphDomain) -> None:
        self.domain = domain

    @abstractmethod
    def __iter__(self) -> Iterator[nx.Graph]:
        """Yield nx.Graphs conforming to ``self.domain.feature_schema``."""
        ...

    def __len__(self) -> int:
        raise TypeError(
            f"{type(self).__name__} does not have a fixed length (streaming)"
        )

    @property
    def is_finite(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# MixedStreamDataset (generic, domain-agnostic)
# ---------------------------------------------------------------------------


class MixedStreamDataset(GraphDataset):
    """Combines multiple streaming :class:`GraphDataset` instances with
    weighted sampling.

    On each iteration step, one of the source datasets is chosen at random
    according to *weights*, and the next graph from that source is yielded.

    All source datasets must share the same domain (i.e. the same
    ``feature_bins``).  Non-streaming (finite) sources are supported but
    will raise ``StopIteration`` when exhausted.

    Parameters
    ----------
    datasets : list[GraphDataset]
        Source datasets to mix.
    weights : list[float] or None
        Sampling weights (one per source).  Normalized internally.
        ``None`` means equal weight for every source.
    seed : int or None
        Seed for the source-selection RNG.
    """

    def __init__(
        self,
        datasets: list[GraphDataset],
        weights: list[float] | None = None,
        seed: int | None = None,
    ) -> None:
        if not datasets:
            raise ValueError("At least one dataset is required")
        super().__init__(datasets[0].domain)
        self.datasets = datasets
        self._weights = weights
        self.rng = np.random.default_rng(seed)

    @property
    def _normalized_weights(self) -> "np.ndarray":
        if self._weights is None:
            n = len(self.datasets)
            return np.ones(n) / n
        w = np.array(self._weights, dtype=float)
        return w / w.sum()

    def __iter__(self) -> Iterator[nx.Graph]:
        iterators = [iter(ds) for ds in self.datasets]
        probs = self._normalized_weights
        while True:
            idx = int(self.rng.choice(len(iterators), p=probs))
            yield next(iterators[idx])

    @property
    def is_finite(self) -> bool:
        return False
