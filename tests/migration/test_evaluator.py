"""Migration tests for evaluator molecular property computations."""

import pytest
from rdkit import Chem

from graph_hdc.utils.chem import canonical_key
from graph_hdc.utils.evaluator import rdkit_logp, rdkit_qed, rdkit_sa_score


def test_evaluator_validity(evaluator_fixture):
    """Molecule validity flags must match golden reference."""
    for entry in evaluator_fixture["per_molecule"]:
        smi = entry["smiles"]
        mol = Chem.MolFromSmiles(smi)
        actual_valid = mol is not None
        assert actual_valid == entry["valid"], (
            f"Validity mismatch for '{smi}': expected {entry['valid']}, got {actual_valid}"
        )


def test_evaluator_validity_counts(evaluator_fixture):
    """Validity and uniqueness counts must match golden reference."""
    valid_count = sum(1 for e in evaluator_fixture["per_molecule"] if e["valid"])
    assert valid_count == evaluator_fixture["num_valid"], (
        f"Valid count mismatch: expected {evaluator_fixture['num_valid']}, got {valid_count}"
    )

    valid_keys = [
        e["canonical_key"] for e in evaluator_fixture["per_molecule"] if e["valid"]
    ]
    unique_count = len(set(valid_keys))
    assert unique_count == evaluator_fixture["num_unique"], (
        f"Unique count mismatch: expected {evaluator_fixture['num_unique']}, got {unique_count}"
    )


def test_evaluator_canonical_key(evaluator_fixture):
    """canonical_key() must match golden reference."""
    for entry in evaluator_fixture["per_molecule"]:
        if not entry["valid"]:
            continue
        smi = entry["smiles"]
        mol = Chem.MolFromSmiles(smi)
        actual = canonical_key(mol)
        assert actual == entry["canonical_key"], (
            f"canonical_key mismatch for '{smi}': expected '{entry['canonical_key']}', got '{actual}'"
        )


def test_evaluator_logp(evaluator_fixture):
    """LogP values must match golden reference."""
    for entry in evaluator_fixture["per_molecule"]:
        if not entry["valid"]:
            continue
        smi = entry["smiles"]
        mol = Chem.MolFromSmiles(smi)
        actual = rdkit_logp(mol)
        assert abs(actual - entry["logp"]) < 1e-6, (
            f"LogP mismatch for '{smi}': expected {entry['logp']}, got {actual}"
        )


def test_evaluator_qed(evaluator_fixture):
    """QED values must match golden reference."""
    for entry in evaluator_fixture["per_molecule"]:
        if not entry["valid"]:
            continue
        smi = entry["smiles"]
        mol = Chem.MolFromSmiles(smi)
        actual = rdkit_qed(mol)
        assert abs(actual - entry["qed"]) < 1e-6, (
            f"QED mismatch for '{smi}': expected {entry['qed']}, got {actual}"
        )


def test_evaluator_sa_score(evaluator_fixture):
    """SA score values must match golden reference."""
    for entry in evaluator_fixture["per_molecule"]:
        if not entry["valid"]:
            continue
        smi = entry["smiles"]
        mol = Chem.MolFromSmiles(smi)
        actual = rdkit_sa_score(mol)
        assert abs(actual - entry["sa_score"]) < 1e-6, (
            f"SA score mismatch for '{smi}': expected {entry['sa_score']}, got {actual}"
        )
