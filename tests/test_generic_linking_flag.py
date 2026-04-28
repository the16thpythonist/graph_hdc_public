"""
Unit tests for the ``use_generic_linking`` flag on ``FragmentLibrary``.

The flag controls whether the universal wildcard attachment label (0) is
permitted: when False, only canonical BRICS labels (1-16) and their official
compatibility table apply.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from rdkit import Chem

from graph_hdc.datasets.streaming_fragments import (
    BRICS_COMPATIBLE_PAIRS,
    BRICS_STRICT_PAIRS,
    FragmentLibrary,
    _brics_compatible,
    fast_combine_two_fragments,
)


# =============================================================================
# _brics_compatible
# =============================================================================


def test_brics_compatible_strict_mode_rejects_wildcard():
    # (0, anything) is in the wildcard set but not in strict pairs
    assert _brics_compatible(0, 5, allow_wildcard=True) is True
    assert _brics_compatible(0, 5, allow_wildcard=False) is False
    assert _brics_compatible(5, 0, allow_wildcard=False) is False


def test_brics_compatible_strict_mode_accepts_canonical_pairs():
    # (1, 3) is a canonical BRICS pair
    assert (1, 3) in BRICS_STRICT_PAIRS
    assert _brics_compatible(1, 3, allow_wildcard=False) is True
    assert _brics_compatible(1, 3, allow_wildcard=True) is True


def test_brics_compatible_rejects_invalid_pair():
    # (1, 2) is not in the BRICS table
    assert _brics_compatible(1, 2, allow_wildcard=False) is False
    assert _brics_compatible(1, 2, allow_wildcard=True) is False


def test_strict_pairs_subset_of_compatible_pairs():
    assert BRICS_STRICT_PAIRS.issubset(BRICS_COMPATIBLE_PAIRS)
    # Wildcard adds extra pairs
    assert len(BRICS_COMPATIBLE_PAIRS) > len(BRICS_STRICT_PAIRS)


# =============================================================================
# fast_combine_two_fragments
# =============================================================================


def _make_frag(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Failed to parse: {smiles}"
    return mol


def test_combine_strict_rejects_wildcard_fragment():
    # Wildcard (label 0) attachment on methyl, paired with a label-3 fragment.
    # In generic mode this combines; in strict mode it must fail.
    wildcard_frag = _make_frag("[*]C")  # isotope 0 by default on '*'
    brics_frag = _make_frag("[3*]C(=O)C")  # acetyl with BRICS label 3

    # Sanity: wildcard fragment really has label 0
    iso_labels = [
        a.GetIsotope() for a in wildcard_frag.GetAtoms() if a.GetSymbol() == "*"
    ]
    assert iso_labels == [0]

    generic = fast_combine_two_fragments(
        wildcard_frag, brics_frag, allow_wildcard=True,
    )
    strict = fast_combine_two_fragments(
        wildcard_frag, brics_frag, allow_wildcard=False,
    )

    assert generic is not None, "generic mode should accept wildcard pair"
    assert strict is None, "strict mode must reject wildcard pair"


def test_combine_strict_accepts_canonical_brics_pair():
    # Labels 1 and 3 are a canonical BRICS pair
    frag1 = _make_frag("[1*]CC")        # ethyl with BRICS label 1
    frag3 = _make_frag("[3*]C(=O)C")    # acetyl with BRICS label 3

    result = fast_combine_two_fragments(frag1, frag3, allow_wildcard=False)
    assert result is not None
    smiles = Chem.MolToSmiles(result)
    # No dummy atoms should remain in the product
    assert "*" not in smiles


def test_combine_strict_rejects_incompatible_canonical_pair():
    # (1, 2) is not in the BRICS pair table
    frag1 = _make_frag("[1*]CC")
    frag2 = _make_frag("[2*]CC")
    result = fast_combine_two_fragments(frag1, frag2, allow_wildcard=False)
    assert result is None


# =============================================================================
# FragmentLibrary defaults & save/load
# =============================================================================


def test_fragment_library_default_is_strict():
    lib = FragmentLibrary()
    assert lib.use_generic_linking is False


def test_fragment_library_save_load_roundtrip_strict():
    lib = FragmentLibrary(use_generic_linking=False)
    lib.fragments = ["[1*]CC", "[3*]C(=O)C"]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lib.pkl"
        lib.save(path)
        loaded = FragmentLibrary.load(path)

    assert loaded.use_generic_linking is False
    assert loaded.fragments == lib.fragments


def test_fragment_library_save_load_roundtrip_generic():
    lib = FragmentLibrary(use_generic_linking=True)
    lib.fragments = ["[0*]C", "[1*]CC"]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lib.pkl"
        lib.save(path)
        loaded = FragmentLibrary.load(path)

    assert loaded.use_generic_linking is True
    assert loaded.fragments == lib.fragments


# =============================================================================
# expand_with_enumerated_positions gating
# =============================================================================


def test_enumeration_no_op_in_strict_mode():
    lib = FragmentLibrary(use_generic_linking=False)
    # Use a real BRICS fragment with H-bearing positions available
    lib.fragments = ["[1*]CC", "[3*]C(=O)C"]
    n_before = lib.num_fragments

    n_added = lib.expand_with_enumerated_positions(max_new_points=5)

    assert n_added == 0
    assert lib.num_fragments == n_before


def test_enumeration_adds_variants_in_generic_mode():
    lib = FragmentLibrary(use_generic_linking=True)
    # Ethyl with one BRICS attachment — has H-bearing carbons available
    lib.fragments = ["[1*]CC"]
    n_before = lib.num_fragments

    n_added = lib.expand_with_enumerated_positions(max_new_points=5)

    assert n_added > 0
    assert lib.num_fragments == n_before + n_added

    # New variants must carry an extra dummy atom with the wildcard label (0).
    # Canonical SMILES renders isotope 0 as bare ``*`` (no isotope prefix).
    for s in lib.fragments[n_before:]:
        m = Chem.MolFromSmiles(s)
        assert m is not None, f"Invalid variant SMILES: {s}"
        wildcard_dummies = [
            a for a in m.GetAtoms()
            if a.GetAtomicNum() == 0 and a.GetIsotope() == 0
        ]
        assert wildcard_dummies, f"Expected wildcard dummy in variant: {s}"


# =============================================================================
# combine_fragments uses the flag
# =============================================================================


def test_library_combine_strict_rejects_wildcard_fragments():
    lib_strict = FragmentLibrary(use_generic_linking=False)
    lib_generic = FragmentLibrary(use_generic_linking=True)

    # Two fragments: one wildcard, one BRICS label 3
    frag_w = _make_frag("[*]C")
    frag_3 = _make_frag("[3*]C(=O)C")

    # Strict mode should fail to combine wildcard pair
    assert lib_strict.combine_fragments([frag_w, frag_3]) is None

    # Generic mode should succeed
    result = lib_generic.combine_fragments([frag_w, frag_3])
    assert result is not None


def test_library_combine_strict_accepts_canonical_pair():
    lib_strict = FragmentLibrary(use_generic_linking=False)
    frag1 = _make_frag("[1*]CC")
    frag3 = _make_frag("[3*]C(=O)C")

    result = lib_strict.combine_fragments([frag1, frag3])
    assert result is not None
    assert "*" not in Chem.MolToSmiles(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
