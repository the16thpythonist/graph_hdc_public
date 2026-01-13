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
    """Count node features in a graph, keyed by feat.to_tuple()."""
    c = Counter()
    for n in G.nodes:
        c[G.nodes[n]["feat"].to_tuple()] += 1
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


def add_node_with_feat(G: nx.Graph, feat: Feat, node_id: int | None = None) -> int:
    """
    Add a node with frozen features.

    Args:
        G: Target graph (modified in place)
        feat: Node features
        node_id: Optional explicit node id

    Returns:
        The node id used
    """
    if node_id is None:
        node_id = 0 if not G.nodes else (max(G.nodes) + 1)
    G.add_node(node_id, feat=feat, target_degree=feat.target_degree)
    return node_id


def add_node_and_connect(
    G: nx.Graph, feat: Feat, connect_to: Sequence[int], total_nodes: int
) -> int | None:
    """Add a node and connect to anchors (greedy, respects residuals)."""
    nid = add_node_with_feat(G, feat)
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
        if "feat" in H.nodes[n]:
            f = H.nodes[n]["feat"]
            label = ",".join(map(str, f.to_tuple()))
        elif "type" in H.nodes[n]:
            label = ",".join(map(str, H.nodes[n]["type"]))
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
