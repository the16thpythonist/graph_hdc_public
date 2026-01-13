"""
Smoke tests for normalizing flow models.
"""
import pytest
import torch

from graph_hdc.models.flows.real_nvp import FlowConfig, RealNVPV3Lightning
from graph_hdc.models.regressors.property_regressor import PropertyRegressor


def test_flow_creation():
    """Test creating a Real NVP V3 model."""

    cfg = FlowConfig(
        hv_dim=64,  # Small for testing
        num_flows=2,
        hidden_dim=128,
        num_hidden_layers=1,
    )

    model = RealNVPV3Lightning(cfg)

    assert model is not None
    assert model.D == 64
    assert model.flat_dim == 128  # 2 * 64


def test_flow_sampling():
    """Test sampling from a flow model."""


    cfg = FlowConfig(
        hv_dim=64,
        num_flows=2,
        hidden_dim=128,
        num_hidden_layers=1,
    )

    model = RealNVPV3Lightning(cfg)
    model.eval()

    # Sample
    with torch.no_grad():
        samples = model.sample_split(num_samples=5)

    assert "edge_terms" in samples
    assert "graph_terms" in samples
    assert samples["edge_terms"].shape == (5, 64)
    assert samples["graph_terms"].shape == (5, 64)


def test_flow_standardization():
    """Test setting standardization parameters."""

    cfg = FlowConfig(
        hv_dim=64,
        num_flows=2,
        hidden_dim=128,
        num_hidden_layers=1,
    )

    model = RealNVPV3Lightning(cfg)

    # Set standardization
    mu = torch.zeros(128)
    sigma = torch.ones(128)
    model.set_standardization(mu, sigma)

    assert torch.allclose(model.mu, mu)
    assert torch.allclose(model.log_sigma, torch.zeros(128))


def test_flow_forward_kld():
    """Test forward KLD computation."""

    cfg = FlowConfig(
        hv_dim=64,
        num_flows=2,
        hidden_dim=128,
        num_hidden_layers=1,
    )

    model = RealNVPV3Lightning(cfg)

    # Create random input
    x = torch.randn(3, 128)  # Batch of 3

    # Compute NLL
    nll = model.nf_forward_kld(x)

    assert nll.shape == (3,)
    assert torch.isfinite(nll).all()


def test_regressor_creation():
    """Test creating a property regressor."""

    model = PropertyRegressor(
        input_dim=64,
        hidden_dims=(128, 64),
        target_property="logp",
    )

    assert model is not None


def test_regressor_forward():
    """Test regressor forward pass."""

    model = PropertyRegressor(
        input_dim=64,
        hidden_dims=(128, 64),
        target_property="qed",
    )
    model.eval()

    # Random input
    x = torch.randn(5, 64)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (5,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
