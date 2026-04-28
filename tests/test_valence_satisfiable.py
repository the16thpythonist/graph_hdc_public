"""Correctness tests for the valence-feasibility necessary-condition check.

The check (``graph_hdc.utils.nx_utils.valence_satisfiable``) must never
flag a genuinely feasible partial graph as infeasible (no false negatives).
False positives are allowed.
"""
from collections import Counter

import networkx as nx

from graph_hdc.hypernet.types import Feat
from graph_hdc.utils.nx_utils import (
    add_edge_if_possible,
    add_node_and_connect,
    add_node_with_feat,
    valence_satisfiable,
)

# QM9 feature tuple shape: (atom_type, degree-1, formal_charge, total_H)
CH3 = (0, 0, 0, 3)   # terminal C, target_degree 1
CH2 = (0, 1, 0, 2)   # middle   C, target_degree 2
CH  = (0, 2, 0, 1)   # branching C, target_degree 3


def _ctr_from_undirected(undirected_edges: list[tuple[tuple, tuple]]) -> Counter:
    """Build the bidirectional counter that the A* decoder uses."""
    c: Counter = Counter()
    for a, b in undirected_edges:
        c[(a, b)] += 1
        c[(b, a)] += 1
    return c


def test_propane_initial_state_is_feasible():
    """Empty graph with valid node/edge multiset must be declared feasible."""
    G = nx.Graph()
    node_counter = Counter({CH3: 2, CH2: 1})
    remaining = _ctr_from_undirected([(CH3, CH2), (CH2, CH3)])
    assert valence_satisfiable(G, remaining, node_counter) is True


def test_fully_built_propane_is_feasible():
    """Terminal state with all residuals zero and no unplaced atoms is feasible."""
    G = nx.Graph()
    node_counter = Counter({CH3: 2, CH2: 1})

    u = add_node_with_feat(G, Feat.from_tuple(CH3), raw_type=CH3)
    mid = add_node_and_connect(G, Feat.from_tuple(CH2), connect_to=[u], total_nodes=3, raw_type=CH2)
    add_node_and_connect(G, Feat.from_tuple(CH3), connect_to=[mid], total_nodes=3, raw_type=CH3)

    remaining: Counter = Counter()
    assert valence_satisfiable(G, remaining, node_counter) is True


def test_terminal_with_open_residual_is_infeasible():
    """No remaining edges but a placed atom still has an open valence → infeasible."""
    G = nx.Graph()
    node_counter = Counter({CH3: 2, CH2: 1})

    u = add_node_with_feat(G, Feat.from_tuple(CH3), raw_type=CH3)
    add_node_and_connect(G, Feat.from_tuple(CH2), connect_to=[u], total_nodes=3, raw_type=CH2)
    # We skipped adding the third carbon; middle-C has residual 1 but no edges left.

    remaining: Counter = Counter()
    assert valence_satisfiable(G, remaining, node_counter) is False


def test_edge_type_without_matching_atom_type_is_infeasible():
    """Remaining edge needs a type that neither placed nor unplaced atoms provide."""
    G = nx.Graph()
    # One placed CH3. node_counter says that's the only atom. But remaining
    # multiset demands a CH3–CH2 edge — impossible since no CH2 atoms exist.
    add_node_with_feat(G, Feat.from_tuple(CH3), raw_type=CH3)
    node_counter = Counter({CH3: 1})
    remaining = _ctr_from_undirected([(CH3, CH2)])
    assert valence_satisfiable(G, remaining, node_counter) is False


def test_insufficient_partners_is_infeasible():
    """A placed atom with residual 3 but only 1 potential partner → infeasible."""
    G = nx.Graph()
    # Branching C (target_degree 3) placed alone, plus one unplaced CH3 → only
    # one possible partner, but the branching C still needs 3 edges.
    u = add_node_with_feat(G, Feat.from_tuple(CH), raw_type=CH)
    node_counter = Counter({CH: 1, CH3: 1})
    # Pretend the remaining multiset wants 3 CH–CH3 edges, which also
    # mismatches atom counts but the Hall check should fail first.
    remaining = _ctr_from_undirected([(CH, CH3), (CH, CH3), (CH, CH3)])
    assert valence_satisfiable(G, remaining, node_counter) is False


def test_parallel_self_edges_exceed_capacity_is_infeasible():
    """Two CH2 atoms, remaining wants 3 self-type edges → infeasible (max 1 edge between 2 atoms)."""
    G = nx.Graph()
    u = add_node_with_feat(G, Feat.from_tuple(CH2), raw_type=CH2)
    add_node_and_connect(G, Feat.from_tuple(CH2), connect_to=[u], total_nodes=2, raw_type=CH2)
    # Now both atoms have residual 1. Remaining claims 3 more CH2–CH2 edges.
    node_counter = Counter({CH2: 2})
    # Contrive a counter that has 6 bidirectional entries for (CH2,CH2) = 3 physical edges.
    remaining: Counter = Counter({(CH2, CH2): 6})
    assert valence_satisfiable(G, remaining, node_counter) is False


def test_negative_residual_is_infeasible():
    """A placed atom with current_degree > target_degree → infeasible."""
    G = nx.Graph()
    u = add_node_with_feat(G, Feat.from_tuple(CH3), raw_type=CH3)  # target 1
    v = add_node_with_feat(G, Feat.from_tuple(CH3), raw_type=CH3)
    w = add_node_with_feat(G, Feat.from_tuple(CH3), raw_type=CH3)
    # Force u to have degree 2 against its target_degree 1 — constructed by
    # bypassing the residual checks.
    G.add_edge(u, v)
    G.add_edge(u, w)
    node_counter = Counter({CH3: 3})
    assert valence_satisfiable(G, Counter(), node_counter) is False


def test_mid_construction_cyclopropane_is_feasible():
    """3-atom partial with one edge in place and two more to come is feasible."""
    G = nx.Graph()
    u = add_node_with_feat(G, Feat.from_tuple(CH2), raw_type=CH2)
    v = add_node_and_connect(G, Feat.from_tuple(CH2), connect_to=[u], total_nodes=3, raw_type=CH2)
    # Only one of 3 ring edges placed; two CH2–CH2 edges remain (2 physical).
    node_counter = Counter({CH2: 3})
    remaining: Counter = Counter({(CH2, CH2): 4})  # 2 physical self-edges
    # One more unplaced CH2 still to come.
    assert valence_satisfiable(G, remaining, node_counter) is True
