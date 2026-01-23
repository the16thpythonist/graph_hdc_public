"""
Shared pytest fixtures for graph_hdc tests.
"""

import os

import pytest
import torch


@pytest.fixture(autouse=True)
def force_cpu():
    """Force CPU for all tests by disabling CUDA visibility."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    yield
    # Restore after test (optional, but good practice)


@pytest.fixture
def qm9_config():
    """Create QM9 configuration for testing."""
    from graph_hdc.hypernet.configs import get_config
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    return config


@pytest.fixture
def qm9_hypernet(qm9_config):
    """Create QM9 HyperNet instance for testing."""
    from graph_hdc.hypernet.encoder import HyperNet
    return HyperNet(qm9_config)


@pytest.fixture
def device():
    """Return the test device (always CPU for unit tests)."""
    return torch.device("cpu")
