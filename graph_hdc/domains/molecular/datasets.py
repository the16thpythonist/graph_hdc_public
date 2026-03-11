"""
Dataset implementations for the molecular domain.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import networkx as nx

from graph_hdc.domains.base import GraphDataset
from graph_hdc.domains.molecular.domain import MolecularDomain


class SmilesDataset(GraphDataset):
    """Loads SMILES strings from a text file and converts to nx.Graphs.

    Works for QM9, ZINC, MOSES, or any file with one SMILES per line.
    Disconnected molecules (containing ``'.'``) and unparseable SMILES
    are silently skipped.

    Parameters
    ----------
    path : str or Path
        Path to the SMILES text file.
    domain : MolecularDomain
        The molecular domain to use for ``process()``.
    """

    def __init__(
        self,
        path: str | Path,
        domain: MolecularDomain,
    ) -> None:
        super().__init__(domain)
        self.path = Path(path)
        self._smiles = self._load()

    def _load(self) -> list[str]:
        """Read SMILES from file, filtering disconnected molecules."""
        smiles: list[str] = []
        with self.path.open() as fh:
            for i, line in enumerate(fh):
                stripped = line.strip()
                if not stripped:
                    continue
                # Skip header
                if i == 0 and stripped.lower() == "smiles":
                    continue
                # Take first column (handles tab/space-separated files)
                smi = stripped.split()[0]
                # Skip disconnected molecules
                if "." in smi:
                    continue
                smiles.append(smi)
        return smiles

    def __iter__(self) -> Iterator[nx.Graph]:
        for smi in self._smiles:
            try:
                yield self.domain.process(smi)
            except ValueError:
                continue

    def __len__(self) -> int:
        return len(self._smiles)
