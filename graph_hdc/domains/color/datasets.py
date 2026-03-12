"""
Dataset implementations for the color graph domain.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import networkx as nx

from graph_hdc.domains.base import GraphDataset, GraphDomain
from graph_hdc.domains.color.domain import ColoredGraphDomain

# Avoid circular import at module level — generators import is deferred.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graph_hdc.domains.color.generators import ColorGraphGenerator


class CogilesDataset(GraphDataset):
    """Loads COGILES strings from a text file and converts to nx.Graphs.

    Works for any file with one COGILES string per line.
    Empty lines and unparseable strings are silently skipped.

    Parameters
    ----------
    path : str or Path
        Path to the COGILES text file.
    domain : ColoredGraphDomain
        The colored graph domain to use for ``process()``.
    """

    def __init__(
        self,
        path: str | Path,
        domain: ColoredGraphDomain,
    ) -> None:
        super().__init__(domain)
        self.path = Path(path)
        self._cogiles = self._load()

    def _load(self) -> list[str]:
        """Read COGILES strings from file, filtering empty lines."""
        strings: list[str] = []
        with self.path.open() as fh:
            for i, line in enumerate(fh):
                stripped = line.strip()
                if not stripped:
                    continue
                # Skip header
                if i == 0 and stripped.lower() == "cogiles":
                    continue
                strings.append(stripped)
        return strings

    def __iter__(self) -> Iterator[nx.Graph]:
        for cog in self._cogiles:
            try:
                yield self.domain.process(cog)
            except ValueError:
                continue

    def __len__(self) -> int:
        return len(self._cogiles)


class ColorGraphStreamingDataset(GraphDataset):
    """Infinite streaming dataset from a :class:`ColorGraphGenerator`.

    Each call to ``__iter__`` yields an endless stream of random colored
    graphs produced by the underlying generator.

    Parameters
    ----------
    generator : ColorGraphGenerator
        The graph generator to draw samples from.
    domain : GraphDomain
        The domain whose schema the generated graphs conform to.
    """

    def __init__(
        self,
        generator: ColorGraphGenerator,
        domain: GraphDomain,
    ) -> None:
        super().__init__(domain)
        self.generator = generator

    def __iter__(self) -> Iterator[nx.Graph]:
        while True:
            yield self.generator.generate()

    @property
    def is_finite(self) -> bool:
        return False
