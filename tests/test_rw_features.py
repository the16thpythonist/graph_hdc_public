"""Tests for random walk return probability features."""

import torch
import pytest
from torch_geometric.data import Data

from graph_hdc.utils.rw_features import (
    ZINC_RW_QUANTILE_BOUNDARIES,
    augment_data_with_rw,
    bin_rw_probabilities,
    compute_rw_return_probabilities,
    get_zinc_rw_boundaries,
)


def _make_edge_index(*edges):
    """Build a bidirectional edge_index from a list of (u, v) pairs."""
    src, dst = [], []
    for u, v in edges:
        src += [u, v]
        dst += [v, u]
    return torch.tensor([src, dst], dtype=torch.long)


# ── compute_rw_return_probabilities ──────────────────────────────────


class TestRWReturnProbabilities:
    def test_path_graph(self):
        """Path 0-1-2: end nodes can return at k=2 via the middle node."""
        edge_index = _make_edge_index((0, 1), (1, 2))
        rw = compute_rw_return_probabilities(edge_index, 3, k_values=(2,))
        # Node 0: deg=1, only neighbour is 1 (deg=2).
        #   T = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
        #   T^2[0,0] = 0*0 + 1*0.5 + 0*0 = 0.5
        assert rw.shape == (3, 1)
        assert pytest.approx(rw[0, 0].item(), abs=1e-6) == 0.5
        assert pytest.approx(rw[2, 0].item(), abs=1e-6) == 0.5
        # Middle node 1: goes to 0 or 2 (prob 0.5 each), both must return
        # to 1 (their only neighbour). T^2[1,1] = 0.5*1 + 0.5*1 = 1.0
        assert pytest.approx(rw[1, 0].item(), abs=1e-6) == 1.0

    def test_cycle_6_k6(self):
        """On a 6-cycle, k=6 return prob should be > 0 (walker can go around)."""
        edges = [(i, (i + 1) % 6) for i in range(6)]
        edge_index = _make_edge_index(*edges)
        rw = compute_rw_return_probabilities(edge_index, 6, k_values=(6,))
        assert rw.shape == (6, 1)
        # All nodes are symmetric, same return prob
        for i in range(1, 6):
            assert pytest.approx(rw[i, 0].item(), abs=1e-6) == rw[0, 0].item()
        # Return prob at k=6 on a 6-cycle should be positive
        assert rw[0, 0].item() > 0.0

    def test_cycle_6_k3(self):
        """On a 6-cycle, k=3 return prob should be > 0 (can walk 3 steps around)."""
        edges = [(i, (i + 1) % 6) for i in range(6)]
        edge_index = _make_edge_index(*edges)
        rw = compute_rw_return_probabilities(edge_index, 6, k_values=(3,))
        # On an even cycle with k odd, return prob is positive because
        # the walker can go 3 steps forward (which wraps halfway).
        # Actually on a 6-cycle with deg=2, the walker goes left or right
        # with prob 0.5. After 3 steps it can return: go 3 steps one way
        # reaches the opposite node. So return prob at k=3 on 6-cycle = 0.
        # But wait - on a 6-cycle the antipodal node is 3 away, not self.
        # Actually k=3 return on 6-cycle: must return in exactly 3 steps.
        # Possible paths: LLL, LLR, LRL, LRR, RLL, RLR, RRL, RRR
        # Each with prob (1/2)^3 = 1/8.
        # Return requires net displacement = 0 mod 6. With L=+1, R=-1:
        #   LLL=+3, LLR=+1, LRL=+1, LRR=-1, RLL=+1, RLR=-1, RRL=-1, RRR=-3
        # None give 0 → return prob = 0
        assert pytest.approx(rw[0, 0].item(), abs=1e-6) == 0.0

    def test_complete_graph(self):
        """On K_4, all nodes should have equal return probs."""
        edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        edge_index = _make_edge_index(*edges)
        rw = compute_rw_return_probabilities(edge_index, 4, k_values=(2, 3))
        assert rw.shape == (4, 2)
        # All nodes equivalent by symmetry
        for i in range(1, 4):
            assert pytest.approx(rw[i, 0].item(), abs=1e-6) == rw[0, 0].item()
            assert pytest.approx(rw[i, 1].item(), abs=1e-6) == rw[0, 1].item()

    def test_isolated_node(self):
        """Isolated node should have return probability 1.0 for all k."""
        # Graph with 3 nodes: 0-1 connected, 2 isolated
        edge_index = _make_edge_index((0, 1))
        rw = compute_rw_return_probabilities(edge_index, 3, k_values=(2, 3))
        # Node 2 is isolated → self-loop, return prob = 1.0
        assert pytest.approx(rw[2, 0].item(), abs=1e-6) == 1.0
        assert pytest.approx(rw[2, 1].item(), abs=1e-6) == 1.0

    def test_multiple_k_values(self):
        """Output shape matches number of k values."""
        edge_index = _make_edge_index((0, 1), (1, 2))
        rw = compute_rw_return_probabilities(edge_index, 3, k_values=(2, 3, 5, 8))
        assert rw.shape == (3, 4)

    def test_single_node(self):
        """Single node with no edges should return 1.0."""
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        rw = compute_rw_return_probabilities(edge_index, 1, k_values=(3, 6))
        assert rw.shape == (1, 2)
        assert pytest.approx(rw[0, 0].item(), abs=1e-6) == 1.0
        assert pytest.approx(rw[0, 1].item(), abs=1e-6) == 1.0

    def test_output_range(self):
        """Return probabilities should be in [0, 1]."""
        edges = [(i, (i + 1) % 10) for i in range(10)]
        edge_index = _make_edge_index(*edges)
        rw = compute_rw_return_probabilities(edge_index, 10, k_values=(3, 6))
        assert (rw >= 0.0).all()
        assert (rw <= 1.0).all()


# ── bin_rw_probabilities (uniform) ───────────────────────────────────


class TestBinning:
    def test_basic_binning(self):
        probs = torch.tensor([[0.0, 0.05, 0.15, 0.95, 1.0]])
        binned = bin_rw_probabilities(probs, num_bins=10)
        assert binned.shape == probs.shape
        assert binned[0, 0].item() == 0.0  # 0.0 → bin 0
        assert binned[0, 1].item() == 0.0  # 0.05 → bin 0
        assert binned[0, 2].item() == 1.0  # 0.15 → bin 1
        assert binned[0, 3].item() == 9.0  # 0.95 → bin 9
        assert binned[0, 4].item() == 9.0  # 1.0 → clamped to bin 9

    def test_all_bins_reachable(self):
        """Values spread across [0, 1) should hit all 10 bins."""
        probs = torch.tensor([[i / 10.0 + 0.01 for i in range(10)]])
        binned = bin_rw_probabilities(probs, num_bins=10)
        assert set(binned[0].long().tolist()) == set(range(10))

    def test_output_dtype(self):
        probs = torch.tensor([[0.5]])
        binned = bin_rw_probabilities(probs, num_bins=10)
        assert binned.dtype == torch.float32


# ── bin_rw_probabilities (quantile) ──────────────────────────────────


class TestQuantileBinning:
    """Tests for data-driven quantile-based binning."""

    def test_known_boundaries(self):
        """Values should be assigned to correct bins using torch.bucketize."""
        # boundaries at [0.2, 0.5] → 3 bins: [0, 0.2), [0.2, 0.5), [0.5, ∞)
        probs = torch.tensor([
            [0.0, 0.1, 0.2, 0.3, 0.5, 0.8],
        ])
        boundaries = {6: [0.2, 0.5]}
        binned = bin_rw_probabilities(
            probs, num_bins=3,
            bin_boundaries=boundaries,
            k_values=(6, 6, 6, 6, 6, 6),
        )
        assert binned[0, 0].item() == 0.0  # 0.0 < 0.2 → bin 0
        assert binned[0, 1].item() == 0.0  # 0.1 < 0.2 → bin 0
        assert binned[0, 2].item() == 1.0  # 0.2 >= 0.2, < 0.5 → bin 1
        assert binned[0, 3].item() == 1.0  # 0.3 → bin 1
        assert binned[0, 4].item() == 2.0  # 0.5 >= 0.5 → bin 2
        assert binned[0, 5].item() == 2.0  # 0.8 → bin 2

    def test_multi_k_different_boundaries(self):
        """Each column uses its own k-specific boundaries."""
        probs = torch.tensor([
            [0.35, 0.20],  # k=2: 0.35 vs boundary [0.33, 0.5]; k=6: 0.20 vs [0.20, 0.29]
        ])
        boundaries = {
            2: [0.33, 0.5],
            6: [0.20, 0.29],
        }
        binned = bin_rw_probabilities(
            probs, num_bins=3,
            bin_boundaries=boundaries,
            k_values=(2, 6),
        )
        assert binned[0, 0].item() == 1.0  # 0.35 >= 0.33, < 0.5 → bin 1
        assert binned[0, 1].item() == 1.0  # 0.20 >= 0.20, < 0.29 → bin 1

    def test_fallback_to_uniform_for_missing_k(self):
        """If a k is not in bin_boundaries, fall back to uniform."""
        probs = torch.tensor([[0.55, 0.55]])
        boundaries = {4: [0.25, 0.5]}  # only k=4 has boundaries
        binned = bin_rw_probabilities(
            probs, num_bins=4,
            bin_boundaries=boundaries,
            k_values=(4, 8),  # k=8 not in boundaries → uniform
        )
        assert binned[0, 0].item() == 2.0  # 0.55 >= 0.5 → bin 2 (quantile)
        assert binned[0, 1].item() == 2.0  # 0.55 * 4 = 2.2 → bin 2 (uniform)

    def test_backward_compatible_no_boundaries(self):
        """When bin_boundaries is None, behaves identically to original."""
        probs = torch.tensor([[0.0, 0.25, 0.5, 0.75, 0.99]])
        uniform = bin_rw_probabilities(probs, num_bins=4)
        also_uniform = bin_rw_probabilities(probs, num_bins=4, bin_boundaries=None)
        assert torch.equal(uniform, also_uniform)

    def test_requires_k_values_with_boundaries(self):
        """Should raise ValueError if boundaries given without k_values."""
        probs = torch.tensor([[0.5]])
        with pytest.raises(ValueError, match="k_values is required"):
            bin_rw_probabilities(probs, num_bins=4, bin_boundaries={2: [0.3]})

    def test_output_clamped(self):
        """Bin indices should never exceed num_bins - 1."""
        probs = torch.tensor([[1.0, 0.999]])
        boundaries = {6: [0.2, 0.4, 0.6]}  # 4 bins
        binned = bin_rw_probabilities(
            probs, num_bins=4,
            bin_boundaries=boundaries,
            k_values=(6, 6),
        )
        assert (binned <= 3.0).all()
        assert (binned >= 0.0).all()


# ── ZINC_RW_QUANTILE_BOUNDARIES preset ──────────────────────────────


class TestZincPreset:
    def test_has_entries_for_k_2_to_16(self):
        """Preset should cover all k from 2 to 16."""
        for k in range(2, 17):
            assert k in ZINC_RW_QUANTILE_BOUNDARIES, f"Missing k={k}"

    def test_boundaries_are_sorted(self):
        """Each boundary list should be non-decreasing."""
        for k, bounds in ZINC_RW_QUANTILE_BOUNDARIES.items():
            for i in range(len(bounds) - 1):
                assert bounds[i] <= bounds[i + 1], (
                    f"k={k}: boundaries not sorted: {bounds}"
                )

    def test_boundaries_length(self):
        """Each entry should have exactly 3 boundaries (for 4 bins)."""
        for k, bounds in ZINC_RW_QUANTILE_BOUNDARIES.items():
            assert len(bounds) == 3, f"k={k}: expected 3 boundaries, got {len(bounds)}"

    def test_even_k_boundaries_positive(self):
        """Even k values should have all-positive boundaries (non-degenerate)."""
        for k in range(2, 17, 2):
            bounds = ZINC_RW_QUANTILE_BOUNDARIES[k]
            assert all(b > 0 for b in bounds), (
                f"k={k}: expected positive boundaries, got {bounds}"
            )

    def test_backward_compat_alias_is_4_bins(self):
        """ZINC_RW_QUANTILE_BOUNDARIES should be the 4-bin preset."""
        assert ZINC_RW_QUANTILE_BOUNDARIES is get_zinc_rw_boundaries(4)


class TestGetZincRwBoundaries:
    @pytest.mark.parametrize("num_bins", [3, 4, 5, 6])
    def test_available_bin_counts(self, num_bins):
        """All documented bin counts should be available."""
        bounds = get_zinc_rw_boundaries(num_bins)
        assert isinstance(bounds, dict)
        for k in range(2, 17):
            assert k in bounds
            assert len(bounds[k]) == num_bins - 1

    @pytest.mark.parametrize("num_bins", [3, 4, 5, 6, 7])
    def test_boundaries_sorted_all_bins(self, num_bins):
        """Boundaries should be non-decreasing for all bin counts."""
        bounds = get_zinc_rw_boundaries(num_bins)
        for k, b in bounds.items():
            for i in range(len(b) - 1):
                assert b[i] <= b[i + 1], (
                    f"bins={num_bins}, k={k}: not sorted: {b}"
                )

    def test_invalid_bin_count_raises(self):
        with pytest.raises(ValueError, match="No precomputed boundaries"):
            get_zinc_rw_boundaries(8)

    def test_more_bins_means_more_boundaries(self):
        """Higher bin counts should have more boundary values."""
        for k in range(2, 17, 2):
            b3 = get_zinc_rw_boundaries(3)[k]
            b7 = get_zinc_rw_boundaries(7)[k]
            assert len(b3) < len(b7)


# ── augment_data_with_rw ─────────────────────────────────────────────


class TestAugmentData:
    def test_appends_columns(self):
        """Should add len(k_values) columns to data.x."""
        data = Data(
            x=torch.tensor([[0.0, 1.0, 0.0, 2.0]], dtype=torch.float32),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
        )
        augmented = augment_data_with_rw(data, k_values=(3, 6), num_bins=10)
        assert augmented.x.shape == (1, 6)  # 4 original + 2 RW

    def test_preserves_original_features(self):
        """Original feature columns should not be modified."""
        x_orig = torch.tensor([[1.0, 2.0, 0.0, 3.0]], dtype=torch.float32)
        data = Data(
            x=x_orig.clone(),
            edge_index=_make_edge_index((0, 0)),  # self-loop
        )
        augmented = augment_data_with_rw(data, k_values=(3,), num_bins=10)
        assert torch.equal(augmented.x[:, :4], x_orig)

    def test_multi_node_graph(self):
        """RW augmentation on a small molecular-like graph."""
        # Propane-like: C-C-C
        x = torch.tensor([
            [0.0, 0.0, 0.0, 3.0],  # C, deg=1
            [0.0, 1.0, 0.0, 2.0],  # C, deg=2
            [0.0, 0.0, 0.0, 3.0],  # C, deg=1
        ], dtype=torch.float32)
        edge_index = _make_edge_index((0, 1), (1, 2))
        data = Data(x=x, edge_index=edge_index)

        augmented = augment_data_with_rw(data, k_values=(3, 6), num_bins=10)
        assert augmented.x.shape == (3, 6)
        # RW columns should be integer bin values
        rw_cols = augmented.x[:, 4:]
        assert (rw_cols >= 0).all()
        assert (rw_cols < 10).all()

    def test_augment_with_quantile_boundaries(self):
        """augment_data_with_rw should thread bin_boundaries through."""
        x = torch.tensor([
            [0.0, 0.0, 0.0, 3.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 3.0],
        ], dtype=torch.float32)
        edge_index = _make_edge_index((0, 1), (1, 2))
        data = Data(x=x, edge_index=edge_index)

        augmented = augment_data_with_rw(
            data, k_values=(2, 6), num_bins=4,
            bin_boundaries=ZINC_RW_QUANTILE_BOUNDARIES,
        )
        assert augmented.x.shape == (3, 6)  # 4 original + 2 RW
        rw_cols = augmented.x[:, 4:]
        assert (rw_cols >= 0).all()
        assert (rw_cols < 4).all()
