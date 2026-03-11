"""Tests for MolecularDomain, preset subclasses, and SmilesDataset."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, ClassVar

import pytest
from rdkit import Chem

from graph_hdc.domains.base import DomainResult, IntegerEncoder, OneHotEncoder
from graph_hdc.domains.molecular.domain import (
    MolecularDomain,
    QM9MolecularDomain,
    ZINCMolecularDomain,
)
from graph_hdc.domains.molecular.datasets import SmilesDataset
from graph_hdc.domains.molecular.encoders import DegreeEncoder, FormalChargeEncoder
from graph_hdc.utils.chem import canonical_key


# ===================================================================
# Default MolecularDomain
# ===================================================================

class TestDefaultMolecularDomain:

    @pytest.fixture()
    def domain(self):
        return MolecularDomain()

    def test_default_feature_bins(self, domain):
        assert domain.feature_bins == [9, 6, 3, 5, 2]

    def test_process_single_atom(self, domain):
        graph = domain.process("C")
        assert graph.number_of_nodes() == 1
        assert graph.number_of_edges() == 0
        feats = graph.nodes[0]["features"]
        assert len(feats) == 5
        # C is index 1 in default vocab ["Br", "C", "Cl", ...]
        assert feats[0] == 1  # atom_type

    def test_process_multi_atom(self, domain):
        graph = domain.process("CCO")
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 2  # undirected nx.Graph

    def test_process_aromatic(self, domain):
        graph = domain.process("c1ccccc1")  # benzene
        assert graph.number_of_nodes() == 6
        assert graph.number_of_edges() == 6

    def test_process_validates(self, domain):
        graph = domain.process("CCO")
        domain.validate(graph)  # should not raise

    def test_process_stores_canonical_smiles(self, domain):
        graph = domain.process("CCO")
        assert "smiles" in graph.graph
        assert graph.graph["smiles"] == Chem.MolToSmiles(
            Chem.MolFromSmiles("CCO"), canonical=True
        )

    def test_process_named_attributes(self, domain):
        graph = domain.process("C")
        node_data = graph.nodes[0]
        for name in domain.feature_schema:
            assert name in node_data, f"Missing named attribute '{name}'"
            idx = list(domain.feature_schema.keys()).index(name)
            assert node_data[name] == node_data["features"][idx]

    def test_process_invalid_smiles_raises(self, domain):
        with pytest.raises(ValueError, match="Cannot parse SMILES"):
            domain.process("INVALID")

    def test_process_unknown_atom_raises(self, domain):
        with pytest.raises(ValueError, match="Unknown value"):
            domain.process("[Xe]")

    def test_unprocess_roundtrip(self, domain):
        graph = domain.process("CCO")
        result = domain.unprocess(graph)
        assert isinstance(result, DomainResult)
        assert result.is_valid is True
        assert result.domain_object is not None

    def test_unprocess_canonical_key(self, domain):
        graph = domain.process("CCO")
        result = domain.unprocess(graph)
        mol = Chem.MolFromSmiles("CCO")
        expected_key = canonical_key(mol)
        assert result.canonical_key == expected_key

    def test_unprocess_properties(self, domain):
        graph = domain.process("CCO")
        result = domain.unprocess(graph)
        assert "logp" in result.properties
        assert "qed" in result.properties
        assert "sa_score" in result.properties

    def test_metrics_registry(self, domain):
        metrics = domain.metrics
        assert "logp" in metrics
        assert "qed" in metrics
        assert "sa_score" in metrics
        assert all(callable(m) for m in metrics.values())


# ===================================================================
# QM9 preset
# ===================================================================

class TestQM9MolecularDomain:

    @pytest.fixture()
    def domain(self):
        return QM9MolecularDomain()

    def test_feature_bins(self, domain):
        assert domain.feature_bins == [4, 5, 3, 5]

    def test_feature_schema_keys(self, domain):
        assert list(domain.feature_schema.keys()) == [
            "atom_type", "degree", "formal_charge", "num_hydrogens",
        ]

    def test_process_features_match_mol_to_data(self, domain):
        from graph_hdc.datasets.qm9_smiles import mol_to_data

        test_smiles = ["C", "CC", "CCO", "C=O", "C#N"]
        for smi in test_smiles:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None
            graph = domain.process(smi)

            pyg_data = mol_to_data(mol)
            for atom_idx in range(mol.GetNumAtoms()):
                expected_feats = [int(x) for x in pyg_data.x[atom_idx].tolist()]
                actual_feats = graph.nodes[atom_idx]["features"]
                assert actual_feats == expected_feats, (
                    f"Feature mismatch for '{smi}' atom {atom_idx}: "
                    f"expected {expected_feats}, got {actual_feats}"
                )

    def test_unprocess_qm9_molecule(self, domain):
        graph = domain.process("c1ccccc1")
        result = domain.unprocess(graph)
        assert result.is_valid is True


# ===================================================================
# ZINC preset
# ===================================================================

class TestZINCMolecularDomain:

    @pytest.fixture()
    def domain(self):
        return ZINCMolecularDomain()

    def test_feature_bins(self, domain):
        assert domain.feature_bins == [9, 6, 3, 4, 2]

    def test_feature_schema_keys(self, domain):
        assert list(domain.feature_schema.keys()) == [
            "atom_type", "degree", "formal_charge", "num_hydrogens", "is_in_ring",
        ]

    def test_process_features_match_mol_to_data(self, domain):
        from graph_hdc.datasets.zinc_smiles import mol_to_data

        test_smiles = ["c1ccccc1", "CC(=O)O", "c1ccc(O)cc1", "CS", "CCBr"]
        for smi in test_smiles:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None
            graph = domain.process(smi)

            pyg_data = mol_to_data(mol)
            for atom_idx in range(mol.GetNumAtoms()):
                expected_feats = [int(x) for x in pyg_data.x[atom_idx].tolist()]
                actual_feats = graph.nodes[atom_idx]["features"]
                assert actual_feats == expected_feats, (
                    f"Feature mismatch for '{smi}' atom {atom_idx}: "
                    f"expected {expected_feats}, got {actual_feats}"
                )

    def test_unprocess_zinc_molecule(self, domain):
        graph = domain.process("c1ccccc1")
        result = domain.unprocess(graph)
        assert result.is_valid is True


# ===================================================================
# Encoder tests
# ===================================================================

class TestDegreeEncoder:

    def test_encode(self):
        enc = DegreeEncoder(max_degree=5)
        assert enc.encode(0) == 0  # max(0, 0-1) = 0
        assert enc.encode(1) == 0  # max(0, 1-1) = 0
        assert enc.encode(2) == 1
        assert enc.encode(3) == 2
        assert enc.encode(5) == 4

    def test_decode(self):
        enc = DegreeEncoder(max_degree=5)
        assert enc.decode(0) == 1
        assert enc.decode(1) == 2
        assert enc.decode(4) == 5

    def test_num_bins(self):
        assert DegreeEncoder(max_degree=5).num_bins == 5
        assert DegreeEncoder(max_degree=6).num_bins == 6

    def test_encode_clips_high(self):
        enc = DegreeEncoder(max_degree=3)
        assert enc.encode(10) == 2  # clipped to max_degree - 1

    def test_decode_out_of_range(self):
        enc = DegreeEncoder(max_degree=5)
        with pytest.raises(ValueError, match="out of range"):
            enc.decode(5)


class TestFormalChargeEncoder:

    def test_encode(self):
        enc = FormalChargeEncoder()
        assert enc.encode(0) == 0
        assert enc.encode(1) == 1
        assert enc.encode(-1) == 2

    def test_decode(self):
        enc = FormalChargeEncoder()
        assert enc.decode(0) == 0
        assert enc.decode(1) == 1
        assert enc.decode(2) == -1

    def test_num_bins(self):
        assert FormalChargeEncoder().num_bins == 3

    def test_unknown_charge_raises(self):
        enc = FormalChargeEncoder()
        with pytest.raises(ValueError, match="Unknown formal charge"):
            enc.encode(3)

    def test_roundtrip(self):
        enc = FormalChargeEncoder()
        for charge in [0, 1, -1]:
            assert enc.decode(enc.encode(charge)) == charge


# ===================================================================
# Custom subclass
# ===================================================================

def test_custom_subclass():
    """Subclassing with a custom feature schema works correctly."""

    class SmallDomain(MolecularDomain):
        feature_schema: ClassVar[dict[str, Any]] = {
            "atom_type": OneHotEncoder(["C", "N", "O"]),
            "num_hydrogens": IntegerEncoder(max_val=3),
        }

    domain = SmallDomain()
    assert domain.feature_bins == [3, 4]
    graph = domain.process("C")
    assert graph.number_of_nodes() == 1
    feats = graph.nodes[0]["features"]
    assert len(feats) == 2
    assert feats[0] == 0  # C is index 0 in ["C", "N", "O"]


# ===================================================================
# SmilesDataset
# ===================================================================

class TestSmilesDataset:

    def test_iteration(self):
        domain = QM9MolecularDomain()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("smiles\n")
            f.write("C\n")
            f.write("CC\n")
            f.write("CCO\n")
            path = f.name
        try:
            ds = SmilesDataset(path, domain)
            graphs = list(ds)
            assert len(graphs) == 3
            for g in graphs:
                assert "features" in g.nodes[0]
                assert len(g.nodes[0]["features"]) == 4
        finally:
            Path(path).unlink()

    def test_length(self):
        domain = QM9MolecularDomain()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("C\nCC\nCCO\n")
            path = f.name
        try:
            ds = SmilesDataset(path, domain)
            assert len(ds) == 3
        finally:
            Path(path).unlink()

    def test_skip_disconnected(self):
        domain = QM9MolecularDomain()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("C\n")
            f.write("CC.OO\n")  # disconnected, should be skipped
            f.write("CCO\n")
            path = f.name
        try:
            ds = SmilesDataset(path, domain)
            assert len(ds) == 2  # only C and CCO
        finally:
            Path(path).unlink()

    def test_skip_invalid_during_iteration(self):
        """SMILES that can't be processed by the domain are skipped."""
        domain = QM9MolecularDomain()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("C\n")
            f.write("[Xe]\n")  # Xenon not in QM9 vocab, will raise ValueError
            f.write("CC\n")
            path = f.name
        try:
            ds = SmilesDataset(path, domain)
            assert len(ds) == 3  # all 3 loaded as SMILES strings
            graphs = list(ds)
            assert len(graphs) == 2  # only C and CC yielded
        finally:
            Path(path).unlink()

    def test_skip_header(self):
        domain = QM9MolecularDomain()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("SMILES\n")
            f.write("C\n")
            f.write("CC\n")
            path = f.name
        try:
            ds = SmilesDataset(path, domain)
            assert len(ds) == 2
        finally:
            Path(path).unlink()
