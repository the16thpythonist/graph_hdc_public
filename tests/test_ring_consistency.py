"""Tests for ring-membership hard-prune and leaf-time mismatch counter.

Constructs synthetic graphs with ZINC-style 5-feature tuples
``(atom_type, degree-1, charge, H, is_in_ring)`` — bypassing `Feat` entirely
to directly verify the graph-theoretic behaviour of the helper functions.
"""
from collections import Counter

import networkx as nx

from graph_hdc.hypernet.types import Feat
from graph_hdc.utils.nx_utils import (
    add_node_with_feat,
    count_ring_mismatches,
    ring_membership_consistent,
)

# ── Synthetic 5-feature tuples (ZINC-style) ──────────────────────────
# position 4 is is_in_ring (0 or 1). We only care about tuple equality,
# so the other fields are used as distinguishers.
RING_A = (0, 1, 0, 1, 1)   # ring atom, target_degree 2
RING_B = (0, 1, 0, 2, 1)   # another ring atom, target_degree 2
NONRING_X = (0, 0, 0, 3, 0)  # non-ring terminal, target_degree 1
NONRING_Y = (0, 1, 0, 2, 0)  # non-ring middle, target_degree 2


def _add_node(G: nx.Graph, nid: int, t: tuple) -> None:
    """Attach a node with both ``feat`` and ``type`` attributes set."""
    # add_node_with_feat picks node_id itself; we override to keep tests
    # deterministic when a specific nid is wanted.
    feat = Feat.from_tuple(t)
    G.add_node(nid, feat=feat, type=t, target_degree=feat.target_degree)


def _ctr(undirected_edges: list[tuple[tuple, tuple]]) -> Counter:
    """Build the bidirectional counter the A* decoder uses."""
    c: Counter = Counter()
    for a, b in undirected_edges:
        c[(a, b)] += 1
        c[(b, a)] += 1
    return c


def test_ring_check_noop_when_index_negative():
    """QM9-style (no ring bit): function must return True unconditionally."""
    G = nx.Graph()
    _add_node(G, 0, NONRING_X)
    # Even with nonsense inputs, ring_feature_index < 0 is a no-op.
    assert ring_membership_consistent(G, Counter(), Counter({NONRING_X: 1}), -1) is True


def test_accepts_valid_partial_propane_like():
    """Linear 3-atom partial, all non-ring: consistent."""
    G = nx.Graph()
    _add_node(G, 0, NONRING_X)
    _add_node(G, 1, NONRING_Y)
    _add_node(G, 2, NONRING_X)
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    node_counter = Counter({NONRING_X: 2, NONRING_Y: 1})
    remaining = Counter()  # all placed, all edges used
    assert ring_membership_consistent(G, remaining, node_counter, 4) is True


def test_rule_B_rejects_nonring_atom_on_cycle():
    """Triangle with one non-ring atom on it: infeasible."""
    G = nx.Graph()
    _add_node(G, 0, RING_A)
    _add_node(G, 1, RING_A)
    _add_node(G, 2, NONRING_Y)  # non-ring atom participating in a triangle
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 0)
    node_counter = Counter({RING_A: 2, NONRING_Y: 1})
    remaining = Counter()
    assert ring_membership_consistent(G, remaining, node_counter, 4) is False


def test_rule_B_accepts_ring_atom_on_cycle():
    """Triangle of ring atoms: consistent."""
    G = nx.Graph()
    _add_node(G, 0, RING_A)
    _add_node(G, 1, RING_A)
    _add_node(G, 2, RING_B)
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 0)
    node_counter = Counter({RING_A: 2, RING_B: 1})
    remaining = Counter()
    assert ring_membership_consistent(G, remaining, node_counter, 4) is True


def test_rule_C_rejects_pending_ring_with_zero_intra_budget():
    """A ring atom is placed as a leaf, all remaining budget goes to attachments.

    Setup: 2 ring atoms placed in a chain (neither on a cycle yet), one more
    ring atom to be placed, and exactly one remaining edge — which must be
    used to attach that unplaced atom. No intra-edges left → pending ring
    atoms can never reach a cycle → infeasible.
    """
    G = nx.Graph()
    _add_node(G, 0, RING_A)
    _add_node(G, 1, RING_A)
    G.add_edge(0, 1)
    node_counter = Counter({RING_A: 3})  # one more to place
    # 1 remaining undirected edge = 2 bidirectional entries; leftover = 1 atom
    # → remaining_intra = 1 - 1 = 0. Pending ring atoms > 0. Infeasible.
    remaining = Counter({(RING_A, RING_A): 2})
    assert ring_membership_consistent(G, remaining, node_counter, 4) is False


def test_rule_C_accepts_pending_ring_with_enough_intra_budget():
    """Same setup but enough remaining edges for the ring to close eventually."""
    G = nx.Graph()
    _add_node(G, 0, RING_A)
    _add_node(G, 1, RING_A)
    G.add_edge(0, 1)
    node_counter = Counter({RING_A: 3})
    # 2 remaining undirected edges (one for attachment, one for ring closure);
    # leftover = 1; remaining_intra = 2 - 1 = 1 > 0. Feasible.
    remaining = Counter({(RING_A, RING_A): 4})
    assert ring_membership_consistent(G, remaining, node_counter, 4) is True


def test_count_ring_mismatches_fully_consistent():
    """Benzene-like triangle: all declared ring, all on cycle → 0 mismatches."""
    G = nx.Graph()
    _add_node(G, 0, RING_A)
    _add_node(G, 1, RING_A)
    _add_node(G, 2, RING_A)
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 0)
    assert count_ring_mismatches(G, 4) == 0


def test_count_ring_mismatches_ring_atom_off_cycle():
    """Ring atom placed as a leaf (declared 1, actual 0) → 1 mismatch."""
    G = nx.Graph()
    _add_node(G, 0, RING_A)
    _add_node(G, 1, RING_A)
    G.add_edge(0, 1)  # both leaves of a bridge — neither on cycle
    assert count_ring_mismatches(G, 4) == 2  # both declared ring, neither on cycle


def test_count_ring_mismatches_mixed():
    """Triangle of ring atoms + a non-ring leaf dangling off one."""
    G = nx.Graph()
    _add_node(G, 0, RING_A)
    _add_node(G, 1, RING_A)
    _add_node(G, 2, RING_A)
    _add_node(G, 3, NONRING_X)
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 0)
    G.add_edge(2, 3)  # non-ring leaf attached to the ring
    assert count_ring_mismatches(G, 4) == 0  # everything matches


def test_count_ring_mismatches_noop():
    """No ring feature (QM9): function returns 0 regardless of topology."""
    G = nx.Graph()
    _add_node(G, 0, RING_A)
    _add_node(G, 1, RING_A)
    G.add_edge(0, 1)
    assert count_ring_mismatches(G, -1) == 0
