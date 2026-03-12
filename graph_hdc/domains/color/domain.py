"""
Colored graph domain for GraphHDC.

``ColoredGraphDomain`` uses `COGILES <https://github.com/the16thpythonist/cogiles>`_
as the domain representation — a SMILES-inspired string notation for colored graphs.

Each node has a single feature: **color**, drawn from a fixed palette of 17 canonical
COGILES colors.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import networkx as nx
import cogiles
from cogiles.colors import COLOR_LIST, symbol_to_color_info
from cogiles.errors import CogilesParseError, CogilesEncodeError

from graph_hdc.domains.base import (
    DomainResult,
    GraphDomain,
    OneHotEncoder,
)

# Canonical color symbols in the order used for encoding.
# This defines the integer mapping: index 0 = "R", index 1 = "G", etc.
COLOR_SYMBOLS: list[str] = [c.symbol for c in COLOR_LIST]

# Reverse lookup: symbol -> integer index
SYMBOL_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(COLOR_SYMBOLS)}

NUM_COLORS: int = len(COLOR_SYMBOLS)  # 17


class ColoredGraphDomain(GraphDomain):
    """Colored graph domain using COGILES as the string representation.

    The feature schema has a single dimension: ``"color"`` with 17 bins
    corresponding to the canonical COGILES color palette.

    Valid color graphs must be **connected**, **planar**, and have
    **max degree** ``<= MAX_DEGREE`` (default 6).

    Examples
    --------
    >>> domain = ColoredGraphDomain()
    >>> graph = domain.process("R(G)(B)")
    >>> graph.number_of_nodes()
    3
    >>> graph.nodes[0]["features"]
    [0]
    """

    feature_schema: ClassVar[dict[str, Any]] = {
        "color": OneHotEncoder(COLOR_SYMBOLS),
    }
    # feature_bins = [17]

    MAX_DEGREE: ClassVar[int] = 6

    def process(self, cogiles_string: str) -> nx.Graph:
        """Convert a COGILES string to a NetworkX graph with integer features.

        Parameters
        ----------
        cogiles_string : str
            COGILES representation of the colored graph.

        Returns
        -------
        nx.Graph
            Undirected graph with ``"features"`` and ``"color"`` (int) on
            each node, plus ``G.graph["cogiles"]`` storing the canonical
            COGILES string.

        Raises
        ------
        ValueError
            If the COGILES string cannot be parsed.
        """
        try:
            parsed = cogiles.parse(cogiles_string)
        except CogilesParseError as exc:
            raise ValueError(f"Cannot parse COGILES: {cogiles_string!r} ({exc})") from exc

        G = nx.Graph()
        encoder = self.feature_schema["color"]

        for node_idx in sorted(parsed.nodes()):
            attrs = parsed.nodes[node_idx]
            symbol = attrs["symbol"]
            color_int = encoder.encode(symbol)
            G.add_node(node_idx, features=[color_int], color=color_int)

        for u, v in parsed.edges():
            G.add_edge(u, v)

        # Store canonical COGILES for reference
        G.graph["cogiles"] = cogiles.encode(parsed)
        return G

    def unprocess(self, graph: nx.Graph) -> DomainResult:
        """Convert a NetworkX graph (with integer features) back to a COGILES graph.

        Parameters
        ----------
        graph : nx.Graph
            Graph with integer ``"features"`` node attributes.

        Returns
        -------
        DomainResult
            Contains the COGILES-attributed nx.Graph, validity flag,
            and canonical COGILES string as the key.
        """
        if graph.number_of_nodes() == 0:
            return DomainResult(
                domain_object=None,
                is_valid=False,
                canonical_key=None,
            )

        encoder = self.feature_schema["color"]

        # Build a COGILES-compatible graph with RGB color attributes
        cogiles_graph = nx.Graph()
        for node_idx in sorted(graph.nodes()):
            attrs = graph.nodes[node_idx]
            feats = attrs.get("features")
            if feats is None or len(feats) != 1:
                return DomainResult(
                    domain_object=None,
                    is_valid=False,
                    canonical_key=None,
                )

            color_int = feats[0]
            if not (0 <= color_int < NUM_COLORS):
                return DomainResult(
                    domain_object=None,
                    is_valid=False,
                    canonical_key=None,
                )

            symbol = encoder.decode(color_int)
            info = symbol_to_color_info(symbol)
            cogiles_graph.add_node(
                node_idx,
                color=info.rgb,
                symbol=info.symbol,
                color_name=info.name,
            )

        for u, v in graph.edges():
            cogiles_graph.add_edge(u, v)

        # Structural validity checks
        if not nx.is_connected(graph):
            return DomainResult(
                domain_object=cogiles_graph,
                is_valid=False,
                canonical_key=None,
            )

        max_deg = max(dict(graph.degree()).values())
        if max_deg > self.MAX_DEGREE:
            return DomainResult(
                domain_object=cogiles_graph,
                is_valid=False,
                canonical_key=None,
            )

        is_planar, _ = nx.check_planarity(graph)
        if not is_planar:
            return DomainResult(
                domain_object=cogiles_graph,
                is_valid=False,
                canonical_key=None,
            )

        try:
            canonical = cogiles.encode(cogiles_graph)
        except CogilesEncodeError:
            return DomainResult(
                domain_object=cogiles_graph,
                is_valid=False,
                canonical_key=None,
            )

        return DomainResult(
            domain_object=cogiles_graph,
            is_valid=True,
            canonical_key=canonical,
        )

    def visualize(self, ax: Any, domain_repr_or_graph: Any, **kwargs: Any) -> None:
        """Draw a colored graph.

        Accepts a COGILES string, a COGILES-attributed nx.Graph (with RGB
        ``color`` attrs), or a features-attributed nx.Graph (calls
        :meth:`unprocess` first).
        """
        if isinstance(domain_repr_or_graph, str):
            cogiles.draw(domain_repr_or_graph, ax=ax, **kwargs)
            return

        graph = domain_repr_or_graph
        # If graph has RGB "color" attrs, draw directly
        first_node = next(iter(graph.nodes()))
        first_color = graph.nodes[first_node].get("color")
        if isinstance(first_color, tuple):
            cogiles.draw(graph, ax=ax, **kwargs)
        else:
            # Integer features — unprocess first
            result = self.unprocess(graph)
            if result.domain_object is not None:
                cogiles.draw(result.domain_object, ax=ax, **kwargs)

    @property
    def metrics(self) -> dict[str, Callable]:
        return {}
