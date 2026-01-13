"""
Graph decoding utilities for HDC embeddings.

This module provides functions for decoding graphs from hypervectors,
including correction strategies and pattern matching.
"""

import random

from collections import defaultdict


import networkx as nx


def deduplicate_edges(edges_multiset: list[tuple]) -> list[tuple]:
    """Remove bidirectional duplicates from the edge multiset."""
    temp = []
    for edge_vec in edges_multiset:
        if edge_vec[0] > edge_vec[1]:
            temp.append((edge_vec[1], edge_vec[0]))
        else:
            temp.append((edge_vec[0], edge_vec[1]))
    temp.sort()
    deduplicated = [temp[i] for i in range(0, len(temp), 2)]
    return deduplicated


def compute_sampling_structure(nodes_multiset: list[tuple], edges_multiset: list[tuple]) -> tuple[dict, dict]:
    """
    Construct bipartite matching structure for graph sampling.

    Returns:
        matching_components: Dict mapping node types to nodes/edges lists
        id_to_type: Mapping from string IDs to feature tuples
    """
    nodes_multiset = sorted(nodes_multiset)
    edges_multiset = sorted(edges_multiset)
    deduplicated_edges = deduplicate_edges(edges_multiset)

    matching_components: dict = {}
    id_to_type: dict = {}

    for node_vec in nodes_multiset:
        matching_components.setdefault(node_vec, {"nodes": [], "edges": []})

    for i, node_vec in enumerate(nodes_multiset):
        node_degree = node_vec[1] + 1
        id_to_type[f"n{i}"] = node_vec
        for _ in range(node_degree):
            matching_components.setdefault(node_vec, {"nodes": []})["nodes"].append(f"n{i}")

    for k, edge_vec in enumerate(deduplicated_edges):
        edge_beginning = tuple(edge_vec[0])
        edge_ending = tuple(edge_vec[1])
        id_to_type[f"e{k}"] = (edge_beginning, edge_ending)
        matching_components.setdefault(edge_beginning, {"edges": []})["edges"].append(f"e{k}")
        matching_components.setdefault(edge_ending, {"edges": []})["edges"].append(f"e{k}")

    return matching_components, id_to_type


def draw_random_matching(sampling_structure: dict) -> list[tuple[str, str]]:
    """Draw a random matching from the sampling structure."""
    matching = []
    for component in sampling_structure.values():
        nodes = component["nodes"]
        edges = component["edges"]
        permuted_edges = edges[:]
        random.shuffle(permuted_edges)
        for node, edge in zip(nodes, permuted_edges, strict=False):
            matching.append((edge, node))
    return sorted(matching)


def compute_graph_from_matching(matching: list[tuple[str, str]], id_to_type: dict) -> nx.Graph:
    """Construct NetworkX graph from matching."""
    G = nx.Graph()
    for i in range(0, len(matching), 2):
        edge_id_1, node_id_1 = matching[i]
        edge_id_2, node_id_2 = matching[i + 1]
        G.add_edge(node_id_1, node_id_2)
        G.nodes[node_id_1]["type"] = id_to_type[node_id_1]
        G.nodes[node_id_2]["type"] = id_to_type[node_id_2]
    return G


def draw_random_graph_from_sampling_structure(matching_components: dict, id_to_type: dict) -> nx.Graph:
    """Sample a random molecular graph by drawing a random matching."""
    random_matching = draw_random_matching(matching_components)
    return compute_graph_from_matching(random_matching, id_to_type)


def graph_is_valid(G: nx.Graph) -> bool:
    """Check if the graph is connected and has no self-loops."""
    return nx.is_connected(G) and nx.number_of_selfloops(G) == 0


def has_valid_ring_structure(
    G: nx.Graph,
    processed_histogram: dict,
    single_ring_atom_types: set,
    *,
    is_partial: bool = False
) -> bool:
    """
    Validate ring structure constraints.

    Checks:
    1. IsInRing flag (acyclic vs cyclic)
    2. Ring histogram (allowed ring sizes)
    3. Single ring atoms (must be in exactly 1 ring)
    """
    try:
        all_cycles = list(nx.cycle_basis(G))
    except Exception:
        return False

    if len(all_cycles) == 0:
        return True

    nodes_must_be_acyclic = set()
    nodes_must_be_single_ring = set()
    nodes_must_be_in_ring = set()
    attr = "feat" if "feat" in G.nodes[next(iter(G.nodes))] else "type"

    for node_id, attrs in G.nodes(data=True):
        node_type = attrs[attr].to_tuple() if attr == "feat" else attrs[attr]
        if node_type[4] == 0:
            nodes_must_be_acyclic.add(node_id)
        else:
            nodes_must_be_in_ring.add(node_id)
            if node_type in single_ring_atom_types:
                nodes_must_be_single_ring.add(node_id)

    node_to_ring_lengths = defaultdict(set)
    node_to_cycle_count = defaultdict(int)

    for cycle in all_cycles:
        cycle_len = len(cycle)
        for node_id in cycle:
            if node_id in nodes_must_be_acyclic:
                return False
            node_to_ring_lengths[node_id].add(cycle_len)
            node_to_cycle_count[node_id] += 1

    for node_id in nodes_must_be_in_ring:
        if node_id not in node_to_cycle_count:
            if not is_partial:
                return False
            continue

        if node_id in nodes_must_be_single_ring and node_to_cycle_count[node_id] != 1:
            if not is_partial:
                return False
            continue

        actual_sizes = node_to_ring_lengths[node_id]
        node_type = G.nodes[node_id][attr].to_tuple() if attr == "feat" else G.nodes[node_id][attr]
        allowed_sizes = set(processed_histogram[node_type].keys())

        if not actual_sizes.issubset(allowed_sizes):
            return False

    return True


def try_find_isomorphic_graph(
    matching_components: dict,
    id_to_type: dict,
    *,
    max_samples: int = 200000,
    ring_histogram: dict | None = None,
    single_ring_atom_types: set | None = None,
) -> list[nx.Graph]:
    """Generate valid molecular graphs by sampling random matchings."""
    max_attempts = 10 * max_samples
    valid_graphs_found = []

    for i in range(max_attempts):
        G = draw_random_graph_from_sampling_structure(matching_components, id_to_type)

        if graph_is_valid(G) and (
            not ring_histogram or has_valid_ring_structure(G, ring_histogram, single_ring_atom_types)
        ):
            valid_graphs_found.append(G)

        if len(valid_graphs_found) >= max_samples:
            break

    return valid_graphs_found
