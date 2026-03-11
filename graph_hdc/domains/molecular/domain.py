"""
Molecular graph domain for GraphHDC.

``MolecularDomain`` has a sensible default feature schema covering common
drug-like molecules.  Dataset-specific variants (``QM9MolecularDomain``,
``ZINCMolecularDomain``) are thin subclasses that override ``feature_schema``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import networkx as nx
from rdkit import Chem

from graph_hdc.domains.base import (
    BoolEncoder,
    DomainResult,
    GraphDomain,
    IntegerEncoder,
    OneHotEncoder,
)
from graph_hdc.domains.molecular.encoders import DegreeEncoder, FormalChargeEncoder
from graph_hdc.domains.molecular.chem import (
    canonical_key,
    draw_mol,
    is_valid_molecule,
    reconstruct_for_eval,
)
from graph_hdc.domains.molecular.metrics import rdkit_logp, rdkit_qed, rdkit_sa_score


class MolecularDomain(GraphDomain):
    """Molecular graph domain with a comprehensive default feature set.

    The default ``feature_schema`` covers 9 common drug-like atom types
    (matching ZINC) with degree, formal charge, hydrogen count, and ring
    membership.  To restrict or extend features, subclass and override
    ``feature_schema`` (and optionally ``_atom_extractors``).

    Examples
    --------
    >>> domain = MolecularDomain()
    >>> graph = domain.process("CCO")
    >>> graph.number_of_nodes()
    3
    """

    # -- Extraction callbacks: feature_name -> (atom -> raw_value) ----------
    # Subclasses can extend this to add new feature types.
    _atom_extractors: ClassVar[dict[str, Callable]] = {
        "atom_type": lambda atom: atom.GetSymbol(),
        "degree": lambda atom: atom.GetDegree(),
        "formal_charge": lambda atom: atom.GetFormalCharge(),
        "num_hydrogens": lambda atom: atom.GetTotalNumHs(),
        "is_in_ring": lambda atom: atom.IsInRing(),
    }

    # -- Default feature schema (comprehensive drug-like set) ---------------
    feature_schema: ClassVar[dict[str, Any]] = {
        "atom_type": OneHotEncoder(["Br", "C", "Cl", "F", "I", "N", "O", "P", "S"]),
        "degree": DegreeEncoder(max_degree=6),
        "formal_charge": FormalChargeEncoder(),
        "num_hydrogens": IntegerEncoder(max_val=4),
        "is_in_ring": BoolEncoder(),
    }
    # Default bins: [9, 6, 3, 5, 2]

    def process(self, smiles: str) -> nx.Graph:
        """Convert a SMILES string to a NetworkX graph with integer features.

        Parameters
        ----------
        smiles : str
            SMILES representation of the molecule.

        Returns
        -------
        nx.Graph
            Undirected graph with ``"features"`` and named attributes on
            each node, plus ``G.graph["smiles"]`` storing the canonical
            SMILES.

        Raises
        ------
        ValueError
            If the SMILES cannot be parsed or contains atoms outside
            the ``feature_schema["atom_type"]`` vocabulary.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Cannot parse SMILES: {smiles!r}")

        G = nx.Graph()
        schema_items = list(self.feature_schema.items())

        for atom in mol.GetAtoms():
            features: list[int] = []
            named: dict[str, int] = {}
            for name, encoder in schema_items:
                extractor = self._atom_extractors.get(name)
                if extractor is None:
                    raise ValueError(
                        f"No extractor registered for feature '{name}'. "
                        f"Add it to _atom_extractors."
                    )
                raw = extractor(atom)
                idx = encoder.encode(raw)
                features.append(idx)
                named[name] = idx
            G.add_node(atom.GetIdx(), features=features, **named)

        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

        G.graph["smiles"] = Chem.MolToSmiles(mol, canonical=True)
        return G

    def unprocess(self, graph: nx.Graph) -> DomainResult:
        """Convert a NetworkX graph back to an RDKit molecule.

        This is a Phase 1 bridge: copies ``"features"`` to ``"type"``
        tuples and delegates to :func:`reconstruct_for_eval`.

        Parameters
        ----------
        graph : nx.Graph
            Graph with integer ``"features"`` node attributes.

        Returns
        -------
        DomainResult
            Contains the RDKit Mol (or None), validity flag, canonical
            key, and molecular properties.
        """
        bridge = graph.copy()
        for node in bridge.nodes:
            bridge.nodes[node]["type"] = tuple(bridge.nodes[node]["features"])

        # Infer compat dataset string from atom vocabulary
        atom_encoder = self.feature_schema["atom_type"]
        atom_set = set(atom_encoder.values)
        if atom_set <= {"C", "N", "O", "F"}:
            dataset_compat = "qm9"
        else:
            dataset_compat = "zinc"

        try:
            mol = reconstruct_for_eval(bridge, dataset=dataset_compat)
        except (ValueError, Exception):
            return DomainResult(
                domain_object=None,
                is_valid=False,
                canonical_key=None,
            )

        if mol is None or not is_valid_molecule(mol):
            return DomainResult(
                domain_object=mol,
                is_valid=False,
                canonical_key=None,
            )

        try:
            key = canonical_key(mol)
        except Exception:
            key = None

        props: dict[str, float] = {}
        try:
            props["logp"] = rdkit_logp(mol)
            props["qed"] = rdkit_qed(mol)
            props["sa_score"] = rdkit_sa_score(mol)
        except Exception:
            pass

        return DomainResult(
            domain_object=mol,
            is_valid=True,
            canonical_key=key,
            properties=props,
        )

    def visualize(self, ax: Any, domain_repr_or_graph: Any, **kwargs: Any) -> None:
        """Draw a molecule or graph.

        Accepts an RDKit Mol directly or an nx.Graph (calls
        :meth:`unprocess` first).
        """
        if isinstance(domain_repr_or_graph, nx.Graph):
            result = self.unprocess(domain_repr_or_graph)
            mol = result.domain_object
        else:
            mol = domain_repr_or_graph
        if mol is not None:
            draw_mol(mol, **kwargs)

    @property
    def metrics(self) -> dict[str, Callable]:
        return {
            "logp": rdkit_logp,
            "qed": rdkit_qed,
            "sa_score": rdkit_sa_score,
        }


# ---------------------------------------------------------------------------
# Preset subclasses
# ---------------------------------------------------------------------------

class QM9MolecularDomain(MolecularDomain):
    """QM9 feature set: 4 atom types, 4 features, bins [4, 5, 3, 5]."""

    feature_schema: ClassVar[dict[str, Any]] = {
        "atom_type": OneHotEncoder(["C", "N", "O", "F"]),
        "degree": DegreeEncoder(max_degree=5),
        "formal_charge": FormalChargeEncoder(),
        "num_hydrogens": IntegerEncoder(max_val=4),
    }


class ZINCMolecularDomain(MolecularDomain):
    """ZINC feature set: 9 atom types, 5 features, bins [9, 6, 3, 4, 2]."""

    feature_schema: ClassVar[dict[str, Any]] = {
        "atom_type": OneHotEncoder(["Br", "C", "Cl", "F", "I", "N", "O", "P", "S"]),
        "degree": DegreeEncoder(max_degree=6),
        "formal_charge": FormalChargeEncoder(),
        "num_hydrogens": IntegerEncoder(max_val=3),
        "is_in_ring": BoolEncoder(),
    }
