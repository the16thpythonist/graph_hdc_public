"""
Chemical-rules-based codebook pruning for HyperNet encoders.

Generates the set of chemically valid node feature tuples based on
per-atom valence rules, replacing dataset-scanning approaches that
miss novel feature combinations from BRICS fragment recombination.

Node features (ZINC format):
    [atom_type, degree-1, formal_charge, total_Hs, is_in_ring]

Feature encoding:
    - atom_type: Br=0, C=1, Cl=2, F=3, I=4, N=5, O=6, P=7, S=8
    - degree-1:  0..5  (actual degree = feature_value + 1; degree=0 maps to 0)
    - charge:    0=neutral, 1=+1, 2=-1
    - total_Hs:  0..3
    - is_in_ring: 0 or 1

For RRWP encoders, additional RW bin features are appended.  These are
graph-structural (not chemical) so all bin values 0..num_bins-1 are
valid for any chemically valid base tuple.
"""
from __future__ import annotations

from itertools import product
from typing import Optional


# ─────────────────────────────────────────────────────────────────────
# Per-atom-type valence rules
# ─────────────────────────────────────────────────────────────────────
#
# For each atom type, we define the allowed valence states as a list of
# (charge_idx, max_total_valence) pairs.  ``max_total_valence`` is the
# upper bound on ``degree + total_Hs`` for that charge state.
#
# These are intentionally generous to cover edge cases (hypervalent S/P,
# charged N/O, etc.) rather than restricting to the most common states.
#
# charge_idx: 0=neutral, 1=+1, 2=-1

_VALENCE_RULES: dict[int, list[tuple[int, int]]] = {
    # Br (idx=0): almost always monovalent
    0: [(0, 1)],
    # C (idx=1): valence 4; charged carbocations/carbanions rare but valid
    1: [(0, 4), (1, 3), (2, 3)],
    # Cl (idx=2): monovalent in organic molecules
    2: [(0, 1)],
    # F (idx=3): strictly monovalent
    3: [(0, 1)],
    # I (idx=4): monovalent (can be polyvalent but rare in drug-like)
    4: [(0, 1), (0, 3)],
    # N (idx=5): valence 3 (neutral) or 4 (cation, e.g. quaternary N+)
    5: [(0, 3), (1, 4), (2, 2)],
    # O (idx=6): valence 2 (neutral) or 1 (anion, e.g. carboxylate O-)
    #            or 3 (cation, e.g. oxonium)
    6: [(0, 2), (2, 1), (1, 3)],
    # P (idx=7): valence 3 or 5; can be charged in phosphonium/phosphate ions
    7: [(0, 3), (0, 5), (1, 4), (2, 4)],
    # S (idx=8): valence 2, 4, or 6; thiolate S- (valence 1),
    #            sulfonium S+ (valence 3)
    8: [(0, 2), (0, 4), (0, 6), (2, 1), (1, 3)],
}

# Maximum possible degree per atom type (across all charge states)
_MAX_DEGREE = {
    0: 1,   # Br
    1: 4,   # C
    2: 1,   # Cl
    3: 1,   # F
    4: 3,   # I
    5: 4,   # N (quaternary N+)
    6: 3,   # O (oxonium O+)
    7: 5,   # P
    8: 6,   # S
}


def get_valid_base_node_tuples() -> set[tuple[int, ...]]:
    """Enumerate all chemically valid base node feature 5-tuples.

    Returns a set of (atom_type, degree_idx, charge_idx, total_hs, is_in_ring)
    tuples that satisfy per-atom valence rules.

    Rules applied:
    - degree + total_Hs <= max_valence for at least one allowed valence
    - degree >= 1 (atom must be connected in a molecule)
    - degree <= max_degree for the atom type
    - is_in_ring=1 requires degree >= 2
    - total_Hs <= 3 (ZINC feature encoding limit)
    - halogens (F, Cl, Br, I): degree=1, Hs=0, neutral only
    """
    valid: set[tuple[int, ...]] = set()

    for atom_type in range(9):
        rules = _VALENCE_RULES[atom_type]
        max_deg = _MAX_DEGREE[atom_type]

        for charge_idx, max_valence in rules:
            for degree in range(1, min(max_deg, 5) + 1):  # degree 1..5 (idx 0..4)
                degree_idx = degree - 1
                for total_hs in range(min(3, max_valence - degree) + 1):
                    if degree + total_hs > max_valence:
                        continue
                    for is_in_ring in (0, 1):
                        # Ring membership requires at least 2 neighbors
                        if is_in_ring and degree < 2:
                            continue
                        valid.add((atom_type, degree_idx, charge_idx, total_hs, is_in_ring))

    return valid


def get_valid_node_tuples_with_rw(
    num_rw_features: int = 0,
    num_rw_bins: int = 8,
) -> set[tuple[int, ...]]:
    """Enumerate valid node feature tuples including RW bin features.

    For RRWP encoders, each base 5-tuple is crossed with all possible
    RW bin combinations.  RW return probabilities are graph-structural
    properties with no chemical constraints, so all bin values are valid
    for any chemically valid atom.

    Args:
        num_rw_features: Number of RW k-values (0 for non-RRWP encoders).
        num_rw_bins: Number of bins per RW feature.

    Returns:
        Set of valid node feature tuples (5 + num_rw_features dims).
    """
    base_tuples = get_valid_base_node_tuples()

    if num_rw_features == 0:
        return base_tuples

    # Cross base tuples with all RW bin combinations
    rw_combos = list(product(range(num_rw_bins), repeat=num_rw_features))

    valid: set[tuple[int, ...]] = set()
    for base in base_tuples:
        for rw in rw_combos:
            valid.add(base + rw)

    return valid


def get_valid_edge_pairs(
    node_tuples: Optional[set[tuple[int, ...]]] = None,
) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Generate all valid edge pairs from valid node tuples.

    An edge pair (src, dst) is valid if both src and dst are valid node
    tuples.  This is suitable for non-RRWP encoders where the total
    number of pairs is manageable.

    For RRWP encoders (with RW features), the Cartesian product is too
    large — use chemical-rules node pruning only and skip edge codebook
    pre-computation.

    Args:
        node_tuples: Set of valid node tuples.  If None, uses
            ``get_valid_base_node_tuples()``.

    Returns:
        Set of (src_tuple, dst_tuple) pairs.
    """
    if node_tuples is None:
        node_tuples = get_valid_base_node_tuples()

    sorted_tuples = sorted(node_tuples)
    pairs: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for src in sorted_tuples:
        for dst in sorted_tuples:
            pairs.add((src, dst))

    return pairs
