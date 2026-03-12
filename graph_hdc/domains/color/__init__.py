"""
Color graph domain for GraphHDC.

Provides :class:`ColoredGraphDomain` using COGILES as the string representation,
:class:`CogilesDataset` for loading colored graph datasets from text files,
:class:`ColorGraphStreamingDataset` for infinite streaming from generators,
and graph generators (:class:`DelaunayMeshGenerator`, :class:`RandomTreeGenerator`,
:class:`GridSubgraphGenerator`).
"""

from graph_hdc.domains.color.domain import (
    COLOR_SYMBOLS,
    ColoredGraphDomain,
    NUM_COLORS,
    SYMBOL_TO_IDX,
)
from graph_hdc.domains.color.datasets import (
    CogilesDataset,
    ColorGraphStreamingDataset,
)
from graph_hdc.domains.color.generators import (
    ColorGraphGenerator,
    DelaunayMeshGenerator,
    GridSubgraphGenerator,
    RandomTreeGenerator,
)

__all__ = [
    "COLOR_SYMBOLS",
    "CogilesDataset",
    "ColoredGraphDomain",
    "ColorGraphGenerator",
    "ColorGraphStreamingDataset",
    "DelaunayMeshGenerator",
    "GridSubgraphGenerator",
    "NUM_COLORS",
    "RandomTreeGenerator",
    "SYMBOL_TO_IDX",
]
