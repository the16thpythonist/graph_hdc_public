"""Migration tests for HyperNet.decode_order_one() edge decoding determinism."""

from collections import Counter

from graph_hdc.hypernet.configs import get_config
from graph_hdc.hypernet.encoder import HyperNet


def test_edge_decoding_matches_golden(edge_decode_fixture):
    """decode_order_one() must reproduce golden edge tuples."""
    entries = edge_decode_fixture["entries"]
    dataset = edge_decode_fixture["dataset"]

    # Reconstruct HyperNet
    if dataset == "qm9":
        config_name = "QM9_SMILES_HRR_256_F64_G1NG3"
    else:
        config_name = "ZINC_SMILES_HRR_256_F64_5G1NG4"

    config = get_config(config_name)
    config.device = "cpu"
    hypernet = HyperNet(config)

    for entry in entries:
        smi = entry["smiles"]
        edge_terms = entry["edge_terms"]
        node_counter = Counter({
            tuple(k) if isinstance(k, list) else k: v
            for k, v in entry["node_counter"].items()
        })

        decoded_edges = hypernet.decode_order_one(edge_terms.clone(), node_counter)

        expected_edges = [
            (tuple(a), tuple(b)) if isinstance(a, list) else (a, b)
            for a, b in entry["decoded_edges"]
        ]
        actual_edges = list(decoded_edges)

        assert len(actual_edges) == entry["decoded_num_edges"], (
            f"Edge count mismatch for '{smi}': "
            f"expected {entry['decoded_num_edges']}, got {len(actual_edges)}"
        )

        assert actual_edges == expected_edges, (
            f"Edge tuples mismatch for '{smi}': "
            f"expected {expected_edges}, got {actual_edges}"
        )
