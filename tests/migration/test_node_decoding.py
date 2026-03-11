"""Migration tests for HyperNet.decode_order_zero_iterative() determinism."""

from graph_hdc.hypernet.configs import get_config
from graph_hdc.hypernet.encoder import HyperNet


def test_node_decoding_matches_golden(node_decode_fixture):
    """decode_order_zero_iterative() must reproduce golden node tuples."""
    entries = node_decode_fixture["entries"]
    dataset = node_decode_fixture["dataset"]

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
        node_terms = entry["node_terms"]  # [hv_dim], self-contained
        decoded_nodes = hypernet.decode_order_zero_iterative(node_terms)

        expected_tuples = [tuple(t) for t in entry["decoded_node_tuples"]]
        actual_tuples = list(decoded_nodes)

        # Compare in original order — iterative decoding is deterministic
        assert actual_tuples == expected_tuples, (
            f"Node tuples mismatch for '{smi}': "
            f"expected {expected_tuples}, got {actual_tuples}"
        )

        assert len(decoded_nodes) == entry["decoded_num_nodes"], (
            f"Node count mismatch for '{smi}': "
            f"expected {entry['decoded_num_nodes']}, got {len(decoded_nodes)}"
        )
