"""Tests for the A*-style best-first fallback decoder."""
from collections import Counter

import torch
from torch_geometric.data import Data

from graph_hdc.hypernet.configs import AStarDecoderSettings, get_config
from graph_hdc.hypernet.encoder import CorrectionLevel, HyperNet


def _make_hypernet_qm9() -> HyperNet:
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    return HyperNet(config)


def _encode(hypernet: HyperNet, x: torch.Tensor, edge_index: torch.Tensor):
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(x.size(0), dtype=torch.long, device="cpu")
    with torch.no_grad():
        output = hypernet.forward(data, normalize=True)
    return output["edge_terms"][0], output["graph_embedding"][0]


def test_astar_decoder_propane_roundtrip():
    """A* decoder reconstructs propane (linear C-C-C) exactly."""
    hypernet = _make_hypernet_qm9()

    # Propane: CH3-CH2-CH3
    # QM9 features: [atom_type, degree-1, formal_charge, total_H]
    x = torch.tensor([
        [0.0, 0.0, 0.0, 3.0],  # terminal C, degree 1, 3H
        [0.0, 1.0, 0.0, 2.0],  # middle C,   degree 2, 2H
        [0.0, 0.0, 0.0, 3.0],  # terminal C, degree 1, 3H
    ], device="cpu")
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long, device="cpu")
    edge_term, graph_term = _encode(hypernet, x, edge_index)

    settings = AStarDecoderSettings(budget_seconds=5.0, top_k=3)
    result = hypernet.decode_graph_astar(
        edge_term=edge_term,
        graph_term=graph_term,
        decoder_settings=settings,
    )

    assert result is not None
    assert len(result.nx_graphs) >= 1
    assert result.correction_level == CorrectionLevel.FAIL  # always FAIL for fallback

    top = result.nx_graphs[0]
    assert top.number_of_nodes() == 3
    assert top.number_of_edges() == 2
    assert result.final_flags[0] is True
    assert result.target_reached is True
    # Perfect reconstruction should hit cosine similarity 1 (up to fp noise)
    assert result.cos_similarities[0] > 0.99

    # Top graph should have the right atom-type multiset.
    type_counter = Counter(top.nodes[n]["type"] for n in top.nodes)
    assert type_counter[(0, 0, 0, 3)] == 2
    assert type_counter[(0, 1, 0, 2)] == 1


def test_astar_decoder_cyclopropane_requires_intra_edge():
    """A* decoder reconstructs cyclopropane, which requires the intra-edge
    expansion path to close the ring."""
    hypernet = _make_hypernet_qm9()

    # Cyclopropane (C3H6): three CH2 groups in a triangle.
    # Each carbon has degree 2, 2 hydrogens.
    x = torch.tensor([
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 1.0, 0.0, 2.0],
    ], device="cpu")
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long, device="cpu"
    )
    edge_term, graph_term = _encode(hypernet, x, edge_index)

    settings = AStarDecoderSettings(budget_seconds=5.0, top_k=3)
    result = hypernet.decode_graph_astar(
        edge_term=edge_term,
        graph_term=graph_term,
        decoder_settings=settings,
    )

    top = result.nx_graphs[0]
    assert top.number_of_nodes() == 3
    assert top.number_of_edges() == 3  # ring closed via intra-edge
    assert result.final_flags[0] is True
    assert result.target_reached is True
    assert result.cos_similarities[0] > 0.99


def test_astar_decoder_respects_budget():
    """Budget=0 still returns a DecodingResult (from seeds) rather than hanging."""
    hypernet = _make_hypernet_qm9()

    x = torch.tensor([
        [0.0, 0.0, 0.0, 3.0],
        [0.0, 0.0, 0.0, 3.0],
    ], device="cpu")
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device="cpu")
    edge_term, graph_term = _encode(hypernet, x, edge_index)

    # Essentially no budget for main loop — seeds are scored before the loop.
    settings = AStarDecoderSettings(budget_seconds=0.0, top_k=3)
    result = hypernet.decode_graph_astar(
        edge_term=edge_term,
        graph_term=graph_term,
        decoder_settings=settings,
    )
    # Ethane is a 2-atom molecule; the seed itself IS the target, so we expect
    # to get a complete match despite the zero budget (seeds are always scored).
    assert result is not None
    assert len(result.nx_graphs) >= 1
