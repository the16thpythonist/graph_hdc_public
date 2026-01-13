"""
Smoke tests for molecule generation and decoding.
"""

import pytest
import torch
from rdkit import Chem

from graph_hdc.hypernet.configs import DecoderSettings, FallbackDecoderSettings
from graph_hdc.utils.chem import canonical_key, is_valid_molecule
from graph_hdc.utils.helpers import TupleIndexer, pick_device


def test_decoder_settings():
    """Test decoder settings configuration."""

    # DecoderSettings: top_k is a private field (_top_k with init=False)
    settings = DecoderSettings(
        iteration_budget=5,
        max_graphs_per_iter=100,
    )

    assert settings.iteration_budget == 5
    assert settings.max_graphs_per_iter == 100

    # FallbackDecoderSettings uses beam_size not beam_width
    fallback = FallbackDecoderSettings(beam_size=16)
    assert fallback.beam_size == 16


def test_chem_utils():
    """Test chemistry utility functions."""

    # Valid molecule
    mol = Chem.MolFromSmiles("CCO")  # Ethanol
    assert is_valid_molecule(mol)
    key = canonical_key(mol)
    assert key is not None

    # Invalid molecule
    assert not is_valid_molecule(None)


def test_nx_utils():
    """Test NetworkX utility functions."""
    import networkx as nx
    from graph_hdc.utils.nx_utils import add_node_with_feat, add_edge_if_possible
    from graph_hdc.hypernet.types import Feat

    # Create a simple graph
    G = nx.Graph()

    # Add nodes with features
    feat1 = Feat(atom_type=0, degree_idx=1, formal_charge_idx=0, explicit_hs=3, is_in_ring=False)
    feat2 = Feat(atom_type=0, degree_idx=1, formal_charge_idx=0, explicit_hs=3, is_in_ring=False)

    node1 = add_node_with_feat(G, feat1)
    node2 = add_node_with_feat(G, feat2)

    assert G.number_of_nodes() == 2
    assert node1 == 0
    assert node2 == 1

    # Add edge
    success = add_edge_if_possible(G, 0, 1, strict=False)
    assert success
    assert G.number_of_edges() == 1


def test_feat_dataclass():
    """Test the Feat dataclass."""
    from graph_hdc.hypernet.types import Feat

    feat = Feat(
        atom_type=0,
        degree_idx=2,
        formal_charge_idx=0,
        explicit_hs=3,
        is_in_ring=False,
    )

    assert feat.atom_type == 0
    assert feat.to_tuple() == (0, 2, 0, 3, 0)

    # Test from tuple
    feat2 = Feat.from_tuple((1, 1, 0, 2, 1))
    assert feat2.atom_type == 1
    assert feat2.is_in_ring == 1


def test_torchhd_patch():
    """Test that TorchHD patch is applied correctly."""
    import torch
    import torchhd
    from torchhd import HRRTensor

    # Create HRR tensors
    hv1 = torchhd.random(1, 64, vsa="HRR")
    hv2 = torchhd.random(1, 64, vsa="HRR")

    # Test bind operation
    bound = torchhd.bind(hv1, hv2)
    assert bound.shape == (1, 64)

    # Test bundle operation
    bundled = torchhd.bundle(hv1, hv2)
    assert bundled.shape == (1, 64)

    # Test multibind (the patched operation)
    stacked = torch.stack([hv1, hv2], dim=-2)  # [1, 2, 64]
    stacked_hrr = stacked.as_subclass(HRRTensor)
    multibound = stacked_hrr.multibind()
    assert multibound.shape == (1, 64)


def test_helpers():
    """Test helper utilities."""

    # Test TupleIndexer - uses 'sizes' parameter
    indexer = TupleIndexer(sizes=[4, 3, 2])  # 4 * 3 * 2 = 24 combinations
    assert indexer.size() == 24  # Use size() method instead of len()

    # Get tuple from index using get_tuple method
    tup0 = indexer.get_tuple(0)
    assert tup0 == (0, 0, 0)

    # Test pick_device
    device = pick_device()
    assert isinstance(device, torch.device)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
