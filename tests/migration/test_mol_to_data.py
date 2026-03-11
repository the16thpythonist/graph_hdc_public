"""Migration tests for mol_to_data() determinism."""

import torch
from rdkit import Chem


def test_mol_to_data_features(mol_to_data_fixture):
    """mol_to_data() must produce identical node features."""
    entries = mol_to_data_fixture
    dataset = entries[0]["canonical_smiles"]  # detect dataset from first entry

    # Determine which mol_to_data to use
    if any(e["x"].shape[1] == 4 for e in entries):
        from graph_hdc.datasets.qm9_smiles import mol_to_data
    else:
        from graph_hdc.datasets.zinc_smiles import mol_to_data

    for entry in entries:
        smi = entry["smiles"]
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"Failed to parse SMILES '{smi}'"

        data = mol_to_data(mol)

        torch.testing.assert_close(
            data.x, entry["x"],
            msg=f"Node features mismatch for '{smi}'",
        )


def test_mol_to_data_edges(mol_to_data_fixture):
    """mol_to_data() must produce identical edge indices."""
    entries = mol_to_data_fixture

    if any(e["x"].shape[1] == 4 for e in entries):
        from graph_hdc.datasets.qm9_smiles import mol_to_data
    else:
        from graph_hdc.datasets.zinc_smiles import mol_to_data

    for entry in entries:
        smi = entry["smiles"]
        mol = Chem.MolFromSmiles(smi)
        data = mol_to_data(mol)

        torch.testing.assert_close(
            data.edge_index, entry["edge_index"],
            msg=f"Edge index mismatch for '{smi}'",
        )


def test_mol_to_data_shape(mol_to_data_fixture):
    """mol_to_data() must produce correct shapes."""
    entries = mol_to_data_fixture

    if any(e["x"].shape[1] == 4 for e in entries):
        from graph_hdc.datasets.qm9_smiles import mol_to_data
    else:
        from graph_hdc.datasets.zinc_smiles import mol_to_data

    for entry in entries:
        smi = entry["smiles"]
        mol = Chem.MolFromSmiles(smi)
        data = mol_to_data(mol)

        assert data.x.size(0) == entry["num_nodes"], (
            f"Node count mismatch for '{smi}': "
            f"expected {entry['num_nodes']}, got {data.x.size(0)}"
        )
        assert data.edge_index.size(1) == entry["num_edges"], (
            f"Edge count mismatch for '{smi}': "
            f"expected {entry['num_edges']}, got {data.edge_index.size(1)}"
        )


def test_mol_to_data_canonical_smiles(mol_to_data_fixture):
    """mol_to_data() must produce consistent canonical SMILES."""
    entries = mol_to_data_fixture

    if any(e["x"].shape[1] == 4 for e in entries):
        from graph_hdc.datasets.qm9_smiles import mol_to_data
    else:
        from graph_hdc.datasets.zinc_smiles import mol_to_data

    for entry in entries:
        smi = entry["smiles"]
        mol = Chem.MolFromSmiles(smi)
        data = mol_to_data(mol)

        assert data.smiles == entry["canonical_smiles"], (
            f"Canonical SMILES mismatch for '{smi}': "
            f"expected '{entry['canonical_smiles']}', got '{data.smiles}'"
        )
