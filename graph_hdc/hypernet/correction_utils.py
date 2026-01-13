"""
Graph decoding correction utilities.

This module provides functions for correcting decoded edge sets that don't meet
target criteria by adding or removing edges based on node counter discrepancies.
"""
import math
import random
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CorrectionResult:
    """Holds the results of a graph correction attempt."""

    add_sets: list[list[tuple[tuple, tuple]]] = field(default_factory=list)
    remove_sets: list[list[tuple[tuple, tuple]]] = field(default_factory=list)
    add_edit_count: int = 0
    remove_edit_count: int = 0


def _is_pairing_possible(node_ctr: Counter[tuple], valid_pairs: set[tuple[tuple, tuple]]) -> bool:
    """
    Performs a pre-search check to see if a valid pairing is even possible.

    For each node, it checks if it has enough *potential* partners in the
    entire stub pool to satisfy its required degree.

    Parameters
    ----------
    node_ctr : Counter
        Counter mapping nodes to required degree counts.
    valid_pairs : set
        A set of (u, v) tuples that are considered valid pairs.

    Returns
    -------
    bool
        True if a solution *might* exist, False if it is *impossible*.
    """

    for node, required_degree in node_ctr.items():
        # Count all stubs that can form a valid pair with this node
        available_partners_count = 0
        for potential_partner_node, count_in_list in node_ctr.items():
            if tuple(sorted((node, potential_partner_node))) in valid_pairs:
                if node == potential_partner_node:
                    # Self-loops: Can pair with other instances of itself.
                    # A stub can't pair with itself.
                    available_partners_count += count_in_list - 1
                else:
                    # Standard edge
                    available_partners_count += count_in_list

        if available_partners_count < required_degree:
            # Fail fast
            return False

    return True


def get_node_counter(edges: list[tuple[tuple, tuple]], method: Literal["ceil", "floor"] = "floor") -> Counter[tuple]:
    # Only using the edges and the degree of the nodes we can count the number of nodes
    node_degree_counter = Counter(u for u, _ in edges)
    node_counter = Counter()
    for k, v in node_degree_counter.items():
        # By dividing the number of outgoing edges to the node degree, we can count the number of nodes
        if method == "ceil":
            node_counter[k] = math.ceil(v / (k[1] + 1))
        else:
            node_counter[k] = v // (k[1] + 1)
    return node_counter

def target_reached(edges: list) -> bool:
    if len(edges) == 0:
        return False
    available_edges_cnt = len(edges)  # directed
    target_count = sum((k[1] + 1) * v for k, v in get_node_counter(edges).items())
    return available_edges_cnt == target_count


def _find_corrective_sets(ctr_to_solve: Counter[tuple], valid_pairs: set, max_solutions=10, max_attempts=100) -> list:
    """Helper function to find N distinct corrective sets."""
    found_sets = []
    attempt = 0

    if not _is_pairing_possible(ctr_to_solve, valid_pairs):
        return found_sets

    while len(found_sets) < max_solutions and attempt < max_attempts:
        attempt += 1
        candidate = find_random_valid_sample_robust(deepcopy(ctr_to_solve), valid_pairs)

        if attempt > (max_attempts / 2) and len(found_sets) == 0:
            return found_sets

        if candidate:
            candidate_ctr = Counter(candidate)
            if candidate_ctr not in found_sets:
                found_sets.append(candidate_ctr)
    return found_sets


def get_corrected_sets(
        node_counter_fp: dict[tuple, float],
        decoded_edges_s: list[tuple[tuple, tuple]],
        valid_edge_tuples: set[tuple[tuple, tuple]],
) -> CorrectionResult:
    """
    Attempt to correct a decoded edge set to meet target criteria.

    This function analyzes node counter discrepancies and attempts to find valid
    corrected edge sets by either adding missing edges or removing extra edges.

    Parameters
    ----------
    node_counter_fp : dict
        Node counter with floating point values indicating fractional node degrees
    decoded_edges_s : list
        Initial decoded edge set that may need correction

    Returns
    -------
    list
        list of corrected edge sets that meet the target criteria
    """
    # Corrections
    corrected_edge_sets_add = []
    corrected_edge_sets_remove = []
    missing_ctr = {}
    extra_ctr = {}
    for k, v in node_counter_fp.items():
        if v - int(v) == 0.0:
            continue
        extra, missing = get_base_units(number=v, base_value=1 / (k[1] + 1))  # k[1] has the degree - 1 (0 indexed)
        missing_ctr[k] = missing
        extra_ctr[k] = extra

    missing_ctr = Counter(missing_ctr)
    extra_ctr = Counter(extra_ctr)

    corrective_sets = []

    max_solutions = 20
    max_attempts = 100

    removable_pairs = {tuple(sorted((u, v))) for u, v in decoded_edges_s if u <= v}
    possible_corrective_add_sets = _find_corrective_sets(
        missing_ctr, removable_pairs, max_solutions=max_solutions, max_attempts=max_attempts
    )
    for candidate_ctr in possible_corrective_add_sets:
        if candidate_ctr not in corrective_sets:
            corrective_sets.append(candidate_ctr)
            new_edge_set = deepcopy(decoded_edges_s)
            for k, v in candidate_ctr.items():
                a, b = k
                for _ in range(v):
                    new_edge_set.append((a, b))
                    new_edge_set.append((b, a))
            if target_reached(new_edge_set):
                corrected_edge_sets_add.append(new_edge_set)

    corrective_remove_sets = _find_corrective_sets(
        extra_ctr, valid_edge_tuples, max_solutions=max_solutions, max_attempts=max_attempts
    )
    for candidate_ctr in corrective_remove_sets:
        if candidate_ctr not in corrective_sets:
            corrective_sets.append(candidate_ctr)
            new_edge_set = deepcopy(decoded_edges_s)
            for k, v in candidate_ctr.items():
                a, b = k
                for _ in range(v):
                    if (a, b) in new_edge_set:
                        new_edge_set.remove((a, b))
                        new_edge_set.remove((b, a))
            if target_reached(new_edge_set):
                corrected_edge_sets_remove.append(new_edge_set)

    return CorrectionResult(
        add_sets=corrected_edge_sets_add,
        remove_sets=corrected_edge_sets_remove,
        add_edit_count=missing_ctr.total(),
        remove_edit_count=extra_ctr.total(),
    )


class _SolverTimeout(Exception):
    """Internal exception to signal the solver took too long."""


def find_random_valid_sample_robust(
        node_ctr: Counter[tuple],
        valid_pairs: set[tuple[tuple, tuple]],
        max_attempts: int = 100,
        timeout_sec: float = 2.0,
) -> list[tuple[tuple, tuple]] | None:
    """
    Find a random valid edge pairing that satisfies node counter requirements.

    This function attempts to find a valid matching of node "stubs" into edges,
    where each stub represents a required connection for a node. The algorithm
    uses randomized backtracking search with timeout protection.

    Parameters
    ----------
    node_ctr : Counter[tuple]
        Counter mapping node feature tuples to required degree counts.
        Each key is a node feature tuple, and the value indicates how many
        connections (stubs) that node type needs.
    valid_pairs : set[tuple[tuple, tuple]]
        Set of valid edge pairs as canonical (sorted) tuples of node features.
        Only edges in this set are considered valid solutions.
    max_attempts : int, optional
        Maximum number of random shuffles to try before giving up.
        Default is 100.
    timeout_sec : float, optional
        Maximum time in seconds for the solver before timing out.
        Default is 2.0 seconds.

    Returns
    -------
    list[tuple[tuple, tuple]] | None
        A list of canonical edge pairs forming a valid matching, or None if
        no valid solution was found within the attempt/time budget.
        Returns an empty list if node_ctr is empty.

    Notes
    -----
    The algorithm works by:
    1. Creating a flat list of "stubs" from node_ctr (one entry per required degree)
    2. Randomly shuffling the stubs
    3. Recursively pairing stubs from the end of the list
    4. Backtracking when a pairing is invalid (not in valid_pairs)
    5. Retrying with different random orderings up to max_attempts times
    """

    # 1. Create the flat list of all "stubs"
    item_list_base = [k for k, v in node_ctr.items() for _ in range(v)]
    if not item_list_base:
        return []

    start_time = time.monotonic()

    def solve_inplace(items: list[tuple], n_items: int) -> list[tuple[tuple, tuple]] | None:
        if n_items == 0:
            return []

        if time.monotonic() - start_time > timeout_sec:
            raise _SolverTimeout

        item1 = items[n_items - 1]
        partner_indices = list(range(n_items - 1))
        random.shuffle(partner_indices)

        for j in partner_indices:
            item2 = items[j]
            canonical_pair = tuple(sorted((item1, item2)))

            if canonical_pair not in valid_pairs:
                continue

            # In-place swap and recurse
            items[j], items[n_items - 2] = items[n_items - 2], items[j]
            solution_for_rest = solve_inplace(items, n_items - 2)
            # Swap back (backtrack)
            items[j], items[n_items - 2] = items[n_items - 2], items[j]

            if solution_for_rest is not None:
                return [canonical_pair, *solution_for_rest]

        return None

    for attempt in range(max_attempts):
        items_to_solve = list(item_list_base)
        random.shuffle(items_to_solve)

        try:
            solution = solve_inplace(items_to_solve, len(items_to_solve))

            if solution:
                return solution

        except _SolverTimeout:
            break

    return None


def get_base_units(number: float, base_value: float) -> tuple[int, int]:
    """
    Calculates how many "extra" or "missing" base units a number
    is from its nearest integers.

    Parameters
    ----------
    number : float
        The floating-point number (e.g., 2.5, 1.33)
    base_value : float
        The base resolution as a float (e.g., 0.5 for 1/2, 0.333... for 1/3)

    Returns
    -------
    tuple
        A tuple of (extra_units, missing_units) as integers

    Examples
    --------
    >>> get_base_units(2.5, 0.5)
    (1, 1)  # 0.5 extra from 2, 0.5 missing to 3

    >>> get_base_units(1.333, 0.333)
    (1, 2)  # ~1/3 extra from 1, ~2/3 missing to 2
    """
    # Find the distance to the floor and ceiling integers
    # "Extra" is the distance from the floor (e.g., 2.5 - floor(2.5) = 0.5)
    extra_value = number - math.floor(number)

    # "Missing" is the distance to the ceiling (e.g., ceil(2.5) - 2.5 = 0.5)
    # Use 1.0 - extra_value to handle precision and integer cases
    missing_value = 0.0 if number == math.floor(number) else math.ceil(number) - number

    # Calculate units by dividing the value by the base
    # We use round() to account for floating-point inaccuracies
    # (e.g., 1.33 vs 1/3 results in 0.99, which rounds to 1)
    return round(extra_value / base_value), round(missing_value / base_value)
