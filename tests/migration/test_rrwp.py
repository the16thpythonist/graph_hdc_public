"""Migration tests for RRWP computation determinism."""

import torch
import pytest

from graph_hdc.utils.rw_features import (
    bin_rw_probabilities,
    compute_rw_return_probabilities,
    get_zinc_rw_boundaries,
)


def test_rw_return_probabilities(rrwp_fixture):
    """compute_rw_return_probabilities() must reproduce golden values."""
    for entry in rrwp_fixture:
        smi = entry["smiles"]
        edge_index = torch.tensor(entry["edge_index"], dtype=torch.long)
        num_nodes = entry["num_nodes"]
        k_values = tuple(entry["k_values"])

        rw_probs = compute_rw_return_probabilities(edge_index, num_nodes, k_values=k_values)
        expected = torch.tensor(entry["rw_probs"])

        torch.testing.assert_close(
            rw_probs, expected,
            atol=1e-12, rtol=0,
            msg=f"rw_probs mismatch for '{smi}'",
        )


def test_binned_uniform_10(rrwp_fixture):
    """Uniform 10-bin binning must match golden reference."""
    for entry in rrwp_fixture:
        smi = entry["smiles"]
        rw_probs = torch.tensor(entry["rw_probs"])

        binned = bin_rw_probabilities(rw_probs, num_bins=10)
        expected = torch.tensor(entry["binned_uniform_10"])

        torch.testing.assert_close(
            binned, expected,
            msg=f"binned_uniform_10 mismatch for '{smi}'",
        )


def test_binned_zinc_quantile_4(rrwp_fixture):
    """ZINC quantile 4-bin binning must match golden reference."""
    zinc_boundaries = get_zinc_rw_boundaries(4)

    for entry in rrwp_fixture:
        smi = entry["smiles"]
        k_values = tuple(entry["k_values"])
        rw_probs = torch.tensor(entry["rw_probs"])

        binned = bin_rw_probabilities(
            rw_probs, num_bins=4,
            bin_boundaries=zinc_boundaries,
            k_values=k_values,
        )
        expected = torch.tensor(entry["binned_zinc_quantile_4"])

        torch.testing.assert_close(
            binned, expected,
            msg=f"binned_zinc_quantile_4 mismatch for '{smi}'",
        )
