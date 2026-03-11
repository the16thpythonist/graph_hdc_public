"""Shared fixtures for migration tests."""

import json
from pathlib import Path

import pytest
import torch

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _skip_if_missing(path: Path):
    if not path.exists():
        pytest.skip(f"Fixture not found: {path.name}. Run export_golden.py first.")


@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture(params=["qm9", "zinc"])
def codebook_fixture(request):
    path = FIXTURES_DIR / f"codebook_{request.param}.pt"
    _skip_if_missing(path)
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.fixture(params=["qm9", "zinc"])
def encoder_fixture(request):
    path = FIXTURES_DIR / f"encoder_{request.param}.pt"
    _skip_if_missing(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_molecule_params():
    """Load molecule entries for per-molecule parametrization."""
    params = []
    for dataset in ["qm9", "zinc"]:
        path = FIXTURES_DIR / f"encoder_{dataset}.pt"
        if not path.exists():
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        for mol in data["molecules"]:
            params.append(pytest.param(
                {"dataset": dataset, "config": data["config"], "molecule": mol},
                id=f"{dataset}-{mol['smiles']}",
            ))
    if not params:
        params = [pytest.param(None, id="no-fixtures",
                               marks=pytest.mark.skip(reason="No encoder fixtures"))]
    return params


@pytest.fixture(params=_load_molecule_params())
def molecule_entry(request):
    return request.param


@pytest.fixture(params=["qm9", "zinc"])
def node_decode_fixture(request):
    path = FIXTURES_DIR / f"node_decode_{request.param}.pt"
    _skip_if_missing(path)
    entries = torch.load(path, map_location="cpu", weights_only=False)
    return {"entries": entries, "dataset": request.param}


@pytest.fixture(params=["qm9", "zinc"])
def edge_decode_fixture(request):
    path = FIXTURES_DIR / f"edge_decode_{request.param}.pt"
    _skip_if_missing(path)
    entries = torch.load(path, map_location="cpu", weights_only=False)
    return {"entries": entries, "dataset": request.param}


@pytest.fixture
def rrwp_fixture():
    path = FIXTURES_DIR / "rrwp.json"
    _skip_if_missing(path)
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def flow_decoder_fixture():
    path = FIXTURES_DIR / "flow_decoder.pt"
    _skip_if_missing(path)
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.fixture
def evaluator_fixture():
    path = FIXTURES_DIR / "evaluator.json"
    _skip_if_missing(path)
    with open(path) as f:
        return json.load(f)


@pytest.fixture(params=["qm9", "zinc"])
def mol_to_data_fixture(request):
    path = FIXTURES_DIR / f"mol_to_data_{request.param}.pt"
    _skip_if_missing(path)
    return torch.load(path, map_location="cpu", weights_only=False)
