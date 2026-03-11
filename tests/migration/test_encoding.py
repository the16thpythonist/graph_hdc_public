"""Migration tests for HyperNet.forward() encoding determinism."""

import torch
from torch_geometric.data import Data

from graph_hdc.hypernet.configs import get_config
from graph_hdc.hypernet.encoder import HyperNet


def _get_hypernet(cfg):
    """Reconstruct HyperNet from config metadata."""
    if cfg["base_dataset"] == "qm9":
        config_name = "QM9_SMILES_HRR_256_F64_G1NG3"
    else:
        config_name = "ZINC_SMILES_HRR_256_F64_5G1NG4"

    config = get_config(config_name)
    config.device = "cpu"
    return config, HyperNet(config)


def test_encoding_config_matches_golden(encoder_fixture):
    """Config metadata must match golden reference."""
    f = encoder_fixture
    cfg = f["config"]
    config, hypernet = _get_hypernet(cfg)

    assert hypernet.hv_dim == cfg["hv_dim"], (
        f"hv_dim mismatch: expected {cfg['hv_dim']}, got {hypernet.hv_dim}"
    )
    assert hypernet.depth == cfg["depth"], (
        f"depth mismatch: expected {cfg['depth']}, got {hypernet.depth}"
    )
    assert hypernet.seed == cfg["seed"], (
        f"seed mismatch: expected {cfg['seed']}, got {hypernet.seed}"
    )
    assert str(hypernet.base_dataset) == cfg["base_dataset"] or hypernet.base_dataset.value == cfg["base_dataset"], (
        f"base_dataset mismatch: expected {cfg['base_dataset']}, got {hypernet.base_dataset}"
    )


def test_hypernet_forward_matches_golden(molecule_entry):
    """HyperNet.forward() must reproduce golden outputs for a single molecule."""
    cfg = molecule_entry["config"]
    entry = molecule_entry["molecule"]

    _, hypernet = _get_hypernet(cfg)

    data = Data(
        x=entry["input_x"],
        edge_index=entry["input_edge_index"],
    )
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)

    with torch.no_grad():
        output = hypernet.forward(data, normalize=True)

    smi = entry["smiles"]

    torch.testing.assert_close(
        output["graph_embedding"],
        entry["output_graph_embedding"],
        atol=1e-10, rtol=0,
        msg=f"graph_embedding mismatch for '{smi}'",
    )
    torch.testing.assert_close(
        output["node_terms"],
        entry["output_node_terms"],
        atol=1e-10, rtol=0,
        msg=f"node_terms mismatch for '{smi}'",
    )
    torch.testing.assert_close(
        output["edge_terms"],
        entry["output_edge_terms"],
        atol=1e-10, rtol=0,
        msg=f"edge_terms mismatch for '{smi}'",
    )
