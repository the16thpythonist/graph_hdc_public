"""
Domain abstractions for GraphHDC.

Provides the :class:`GraphDomain` ABC, :class:`GraphDataset` ABC,
:class:`FeatureEncoder` protocol, :class:`DomainResult` dataclass,
and built-in generic encoders.
"""

from graph_hdc.domains.base import (
    BoolEncoder,
    DomainResult,
    FeatureEncoder,
    GraphDataset,
    GraphDomain,
    IntegerEncoder,
    OneHotEncoder,
)

__all__ = [
    "BoolEncoder",
    "DomainResult",
    "FeatureEncoder",
    "GraphDataset",
    "GraphDomain",
    "IntegerEncoder",
    "OneHotEncoder",
]
