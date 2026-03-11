"""Migration tests for FlowEdgeDecoder forward pass determinism."""

import types

import torch

from graph_hdc.models.flow_edge_decoder import FlowEdgeDecoder


def test_flow_decoder_intermediates(flow_decoder_fixture):
    """condition_mlp and time_mlp must reproduce golden intermediate outputs."""
    f = flow_decoder_fixture
    cfg = f["config"]

    model = FlowEdgeDecoder(
        feature_bins=[cfg["num_node_classes"]],
        num_edge_classes=cfg["num_edge_classes"],
        hdc_dim=cfg["hdc_dim"],
        n_layers=cfg["n_layers"],
        hidden_dim=cfg["hidden_dim"],
        hidden_mlp_dim=cfg["hidden_mlp_dim"],
        n_heads=cfg["n_heads"],
        max_nodes=cfg["max_nodes"],
        dropout=cfg["dropout"],
        noise_type=cfg["noise_type"],
        lr=cfg["lr"],
    )
    model.load_state_dict(f["model_state_dict"], strict=True)
    model.eval()

    with torch.no_grad():
        actual_hdc_cond = model.condition_mlp(f["input_hdc"])
        actual_t_embed = model.time_mlp(f["input_t"])

    torch.testing.assert_close(
        actual_hdc_cond, f["input_hdc_cond"],
        atol=1e-6, rtol=0,
        msg="condition_mlp output mismatch",
    )
    torch.testing.assert_close(
        actual_t_embed, f["input_t_embed"],
        atol=1e-6, rtol=0,
        msg="time_mlp output mismatch",
    )


def test_flow_decoder_forward(flow_decoder_fixture):
    """FlowEdgeDecoder.forward() must reproduce golden output."""
    f = flow_decoder_fixture
    cfg = f["config"]

    # No seed needed — weights come entirely from load_state_dict
    model = FlowEdgeDecoder(
        feature_bins=[cfg["num_node_classes"]],
        num_edge_classes=cfg["num_edge_classes"],
        hdc_dim=cfg["hdc_dim"],
        n_layers=cfg["n_layers"],
        hidden_dim=cfg["hidden_dim"],
        hidden_mlp_dim=cfg["hidden_mlp_dim"],
        n_heads=cfg["n_heads"],
        max_nodes=cfg["max_nodes"],
        dropout=cfg["dropout"],
        noise_type=cfg["noise_type"],
        lr=cfg["lr"],
    )
    model.load_state_dict(f["model_state_dict"], strict=True)
    model.eval()

    # Use SimpleNamespace instead of importing PlaceHolder from defog
    extra_data = types.SimpleNamespace(
        X=f["extra_data_X"],
        E=f["extra_data_E"],
        y=f["extra_data_y"],
    )

    noisy_data = {
        "X_t": f["input_X"],
        "E_t": f["input_E"],
        "y_t": f["input_hdc_cond"],
        "t": f["input_t"],
    }

    with torch.no_grad():
        output = model.forward(noisy_data, extra_data, f["input_node_mask"])

    torch.testing.assert_close(
        output.X, f["output_X"],
        atol=1e-6, rtol=0,
        msg="FlowEdgeDecoder output X mismatch",
    )
    torch.testing.assert_close(
        output.E, f["output_E"],
        atol=1e-6, rtol=0,
        msg="FlowEdgeDecoder output E mismatch",
    )
