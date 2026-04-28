"""
NetworkX graph utilities for molecular graph construction and manipulation.
"""

from collections import Counter
from collections.abc import Sequence
from itertools import chain, combinations

import networkx as nx
from networkx.algorithms.isomorphism import (
    GraphMatcher,
    categorical_edge_match,
    categorical_node_match,
)

from graph_hdc.hypernet.types import Feat


def is_induced_subgraph_by_features(
    g1: nx.Graph,
    g2: nx.Graph,
    *,
    node_keys: list[str] | None = None,
    edge_keys: Sequence[str] = (),
    require_connected: bool = True,
) -> bool:
    """
    Check if G1 is isomorphic to a node-induced subgraph of G2.

    Uses VF2 algorithm with semantic checks on node/edge attributes.

    Args:
        g1: Pattern graph
        g2: Target graph
        node_keys: Node attribute keys for matching (default: ["feat"])
        edge_keys: Edge attribute keys for matching
        require_connected: Fail fast if g1 is disconnected

    Returns:
        True if induced subgraph isomorphism exists
    """
    if require_connected and g1.number_of_nodes() and not nx.is_connected(g1):
        return False

    if node_keys is None:
        node_keys = ["feat"]

    def feat_tuple(G: nx.Graph, n) -> tuple:
        data = G.nodes[n]
        return tuple(data.get(k) for k in node_keys)

    # Quick multiset pre-check
    c1 = Counter(feat_tuple(g1, n) for n in g1.nodes)
    c2 = Counter(feat_tuple(g2, n) for n in g2.nodes)
    for k, need in c1.items():
        if c2.get(k, 0) < need:
            return False

    # Full isomorphism check
    nm = categorical_node_match(
        node_keys if len(node_keys) > 1 else node_keys[0],
        [None] * len(node_keys) if len(node_keys) > 1 else None,
    )
    em = categorical_edge_match(list(edge_keys), [None] * len(edge_keys)) if edge_keys else None

    GM = GraphMatcher(g2, g1, node_match=nm, edge_match=em)
    return GM.subgraph_is_isomorphic()


def feature_counter_from_graph(G: nx.Graph) -> Counter[tuple[int, int, int, int]]:
    """Count node features in a graph, keyed by raw type tuple."""
    c = Counter()
    for n in G.nodes:
        nd = G.nodes[n]
        c[nd.get("type", nd["feat"].to_tuple())] += 1
    return c


def leftover_features(full: Counter[tuple[int, int, int, int]], G: nx.Graph) -> Counter:
    """Remaining features to place given the current partial graph."""
    left = full.copy()
    left.subtract(feature_counter_from_graph(G))
    for k in list(left):
        if left[k] <= 0:
            del left[k]
    return left


def current_degree(G: nx.Graph, node: int) -> int:
    """Current degree of node in the graph."""
    return G.degree[node]


def residual_degree(G: nx.Graph, node: int) -> int:
    """Residual degree capacity = target_degree - current_degree."""
    return int(G.nodes[node]["target_degree"]) - current_degree(G, node)


def residuals(G: nx.Graph) -> dict[int, int]:
    """Residual degrees for all nodes."""
    return {n: residual_degree(G, n) for n in G.nodes}


def anchors(G: nx.Graph) -> list[int]:
    """Nodes that can still accept edges (residual > 0)."""
    return [n for n in G.nodes if residual_degree(G, n) > 0]


def add_edge_if_possible(G: nx.Graph, u: int, v: int, *, strict: bool = True) -> bool:
    """
    Add an undirected edge if constraints allow.

    Constraints:
    - u != v
    - Edge must not already exist
    - Both endpoints must have residual > 0 (if strict)

    Returns:
        True if edge was added
    """
    if u == v or G.has_edge(u, v):
        return False
    if strict and (residual_degree(G, u) <= 0 or residual_degree(G, v) <= 0):
        return False
    G.add_edge(u, v)
    if strict and (residual_degree(G, u) < 0 or residual_degree(G, v) < 0):
        G.remove_edge(u, v)
        return False
    return True


def total_edges_count(feat_ctr: Counter[tuple[int, int, int, int]]) -> int:
    """Compute total edges implied by feature multiset (sum of degrees / 2)."""
    return sum(((deg_idx + 1) * v) for (_, deg_idx, _, _), v in feat_ctr.items()) // 2


def add_node_with_feat(
    G: nx.Graph, feat: Feat, node_id: int | None = None, raw_type: tuple | None = None,
) -> int:
    """
    Add a node with frozen features.

    Args:
        G: Target graph (modified in place)
        feat: Node features
        node_id: Optional explicit node id
        raw_type: Raw feature tuple (preserves exact values that Feat
                  may booleanize, e.g. RRWP bins at position 4 for QM9)

    Returns:
        The node id used
    """
    if node_id is None:
        node_id = 0 if not G.nodes else (max(G.nodes) + 1)
    node_type = raw_type if raw_type is not None else feat.to_tuple()
    G.add_node(node_id, feat=feat, type=node_type, target_degree=feat.target_degree)
    return node_id


def add_node_and_connect(
    G: nx.Graph, feat: Feat, connect_to: Sequence[int], total_nodes: int,
    raw_type: tuple | None = None,
) -> int | None:
    """Add a node and connect to anchors (greedy, respects residuals)."""
    nid = add_node_with_feat(G, feat, raw_type=raw_type)
    return connect_all_if_possible(G, nid, connect_to, total_nodes)


def connect_all_if_possible(
    G: nx.Graph, nid: int, connect_to: Sequence[int], total_nodes: int
) -> int | None:
    """Connect node to anchors, remove if constraints violated."""
    ok = True
    for a in connect_to:
        if residual_degree(G, nid) <= 0:
            break
        if residual_degree(G, a) <= 0:
            continue
        if not add_edge_if_possible(G, nid, a, strict=True):
            ok = False
            break
    if not ok or (len(anchors(G)) <= 0 and G.number_of_nodes() != total_nodes):
        G.remove_node(nid)
        return None
    return nid


def powerset(iterable):
    """Return the power set of the input iterable."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def wl_hash(G: nx.Graph, *, iters: int = 3) -> str:
    """WL hash that respects both `feat` and `type` node attributes."""
    H = G.copy()
    for n in H.nodes:
        if "type" in H.nodes[n]:
            label = ",".join(map(str, H.nodes[n]["type"]))
        elif "feat" in H.nodes[n]:
            f = H.nodes[n]["feat"]
            label = ",".join(map(str, f.to_tuple()))
        else:
            label = "unknown"
        H.nodes[n]["__wl_label__"] = label
    return nx.weisfeiler_lehman_graph_hash(H, node_attr="__wl_label__", iterations=iters)


def graph_hash(G: nx.Graph) -> tuple[str, int, int]:
    """Hash a graph by (WL hash, num_nodes, num_edges)."""
    return wl_hash(G), G.number_of_nodes(), G.number_of_edges()


# Aliases for backward compatibility with greedy decoder
_wl_hash = wl_hash
_hash = graph_hash


def order_leftovers_by_degree_distinct(ctr: Counter) -> list[tuple[int, int, int, int]]:
    """Unique feature tuples, sorted by degree (asc), then lexicographically."""
    uniq = list(ctr.keys())
    uniq.sort(key=lambda t: (t[1] + 1, t))
    return uniq


def valence_satisfiable(
    G: nx.Graph,
    remaining_ctr: Counter,
    node_counter: Counter,
) -> bool:
    """Return True iff completion MIGHT still be possible from this partial graph.

    Necessary-only conditions: a return of False is a proof of infeasibility,
    but True does not imply feasibility (false positives are allowed). Intended
    as a cheap sanity check inside search loops.

    Applies four checks:

    1. **No overfilled atom** — every placed atom has ``residual_degree >= 0``.
    2. **Terminal fit** — if ``remaining_ctr`` is empty, every placed atom must
       have residual 0 and no unplaced atoms may remain.
    3. **Per-atom Hall-like condition** — for each placed atom ``u`` with
       residual ``r > 0``, count distinct candidate partners (placed atoms
       with residual > 0 not already adjacent to ``u``, plus unplaced atoms)
       whose type pairs with ``u`` in ``remaining_ctr``. If fewer than ``r``
       candidates exist, completion is impossible. This is the check that
       uses each atom's decoded target_degree directly (via ``residual_degree``).
    4. **Aggregate pairability** — for each remaining edge type ``(a, b)`` with
       physical count ``k_phys``, enough atoms of types ``a`` and ``b`` must
       exist under simple-graph capacity (``avail_a * avail_b`` for ``a != b``,
       ``C(avail_a, 2)`` for ``a == b``).

    Parameters
    ----------
    G : nx.Graph
        Partial graph; each node must carry a ``type`` attribute (the raw
        feature tuple) and a ``target_degree`` attribute.
    remaining_ctr : Counter
        Directed/bidirectional typed-edge multiset still to be placed. Using
        the same convention as the A*/greedy decoders: a non-self edge type
        ``(a, b)`` with ``a != b`` has matching counts under both orientations;
        a self-type edge ``(a, a)`` has count 2 per physical edge.
    node_counter : Counter
        Target multiset of atom types (from Phase 0 decoding).

    Returns
    -------
    bool
        False iff completion is provably impossible.
    """
    # (1) No overfilled atoms
    for n in G.nodes:
        if residual_degree(G, n) < 0:
            return False

    leftover = leftover_features(node_counter, G)

    # (2) Terminal fit
    if remaining_ctr.total() == 0:
        if any(residual_degree(G, n) > 0 for n in G.nodes):
            return False
        if leftover.total() > 0:
            return False
        return True

    # (3) Per-atom Hall-like condition
    for u in G.nodes:
        r_u = residual_degree(G, u)
        if r_u <= 0:
            continue
        u_t = G.nodes[u]["type"]
        partners = 0
        for v in G.nodes:
            if v == u:
                continue
            if G.has_edge(u, v):
                continue
            if residual_degree(G, v) <= 0:
                continue
            v_t = G.nodes[v]["type"]
            if remaining_ctr[(u_t, v_t)] > 0:
                partners += 1
                if partners >= r_u:
                    break
        if partners < r_u:
            for t, cnt in leftover.items():
                if cnt <= 0:
                    continue
                if remaining_ctr[(u_t, t)] > 0:
                    partners += cnt
                    if partners >= r_u:
                        break
        if partners < r_u:
            return False

    # (4) Aggregate pairability by edge type
    placed_avail: Counter = Counter()
    for n in G.nodes:
        if residual_degree(G, n) > 0:
            placed_avail[G.nodes[n]["type"]] += 1

    def _avail(t: tuple) -> int:
        return placed_avail.get(t, 0) + leftover.get(t, 0)

    checked: set = set()
    for (a, b), k in remaining_ctr.items():
        if k <= 0:
            continue
        unord = (a, b) if (a == b or a <= b) else (b, a)
        if unord in checked:
            continue
        checked.add(unord)
        if a == b:
            k_phys = k // 2
            ava = _avail(a)
            if ava < 2 or k_phys > ava * (ava - 1) // 2:
                return False
        else:
            k_phys = k
            ava, avb = _avail(a), _avail(b)
            if ava < 1 or avb < 1 or k_phys > ava * avb:
                return False

    return True


def _on_cycle_vertex_set(G: nx.Graph) -> set:
    """Return the set of vertices lying on at least one cycle in G.

    A vertex is on a cycle iff it belongs to a biconnected component with
    three or more vertices (a 2-vertex biconnected component is a bridge
    edge, which is not part of any cycle).
    """
    on_cycle: set = set()
    for comp in nx.biconnected_components(G):
        if len(comp) >= 3:
            on_cycle.update(comp)
    return on_cycle


def ring_membership_consistent(
    G: nx.Graph,
    remaining_ctr: Counter,
    node_counter: Counter,
    ring_feature_index: int,
) -> bool:
    """Return False only when ring-membership constraints are provably unsatisfiable.

    The decoded feature tuple carries an ``is_in_ring`` bit at
    ``ring_feature_index`` (typically 4 for ZINC/PubChem; <0 or None for
    QM9 means "no ring information available"). The search must ensure
    that every atom with ``t[idx] == 1`` ends up on a cycle in the final
    molecule and every atom with ``t[idx] == 0`` does NOT. Both are
    topological constraints that can be partially checked on a partial graph.

    Necessary-only conditions (a False return is a proof of infeasibility):

    1. **No non-ring atom on a current cycle.** Every cycle in the final
       molecule is entirely contained within the ring-atom subgraph; if any
       ``t[idx] == 0`` atom currently lies on a cycle, the partial cannot
       complete consistently.
    2. **Pending ring-atoms must fit the remaining intra-edge budget.** For
       any connected graph, ``cyclomatic = |E| - |V| + 1`` equals the number
       of intra-edges (cycle closures). The search's remaining intra-edge
       budget is therefore ``remaining_edges_undirected - leftover_count``
       (attachments consume one edge each). If there are still ring-atoms
       not on any cycle (placed or unplaced) and the remaining intra-edge
       budget is <=0, no future cycle can form and the partial is dead.

    When ``ring_feature_index < 0`` the check is a no-op (returns True).

    Parameters
    ----------
    G : nx.Graph
        Partial graph. Each node must carry a ``type`` attribute holding
        the raw feature tuple.
    remaining_ctr : Counter
        Bidirectional typed-edge multiset still to place (A*-style).
    node_counter : Counter
        Target multiset of atom types from Phase 0.
    ring_feature_index : int
        Position of the in-ring bit in the feature tuple, or <0 for no-op.

    Returns
    -------
    bool
        False iff ring-membership constraints are provably unsatisfiable.
    """
    if ring_feature_index < 0:
        return True

    on_cycle = _on_cycle_vertex_set(G)

    # (1) No non-ring atom on a cycle
    for n in G.nodes:
        if G.nodes[n]["type"][ring_feature_index] == 0 and n in on_cycle:
            return False

    # (2) Forward-feasibility on pending ring-atoms
    leftover = leftover_features(node_counter, G)
    placed_R_not_on_cycle = sum(
        1 for n in G.nodes
        if G.nodes[n]["type"][ring_feature_index] == 1 and n not in on_cycle
    )
    unplaced_R = sum(
        cnt for t, cnt in leftover.items()
        if t[ring_feature_index] == 1
    )
    pending_R = placed_R_not_on_cycle + unplaced_R

    # cyclomatic identity: remaining_intra = remaining_undirected - leftover_count
    remaining_undirected = remaining_ctr.total() // 2
    remaining_intra = remaining_undirected - leftover.total()

    if pending_R > 0 and remaining_intra <= 0:
        return False

    return True


def count_ring_mismatches(G: nx.Graph, ring_feature_index: int) -> int:
    """Count atoms whose declared ring-membership disagrees with actual topology.

    For every node, compare ``t[ring_feature_index]`` (1 iff declared in-ring)
    against the topological fact of whether the node lies on a cycle in G.
    Returns 0 when ``ring_feature_index < 0`` (no ring info).
    """
    if ring_feature_index < 0:
        return 0
    on_cycle = _on_cycle_vertex_set(G)
    mismatches = 0
    for n in G.nodes:
        declared = G.nodes[n]["type"][ring_feature_index] == 1
        actual = n in on_cycle
        if declared != actual:
            mismatches += 1
    return mismatches
