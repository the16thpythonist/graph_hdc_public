"""
Tests for strip_dummy_atoms.

Verifies that BRICS fragments with dummy (*) atoms are correctly cleaned
into valid molecules when used as single-fragment inputs to combine_fragments.
"""

import pytest
from rdkit import Chem
from rdkit.Chem import BRICS

from graph_hdc.datasets.streaming_fragments import (
    FragmentLibrary,
    strip_dummy_atoms,
)


class TestStripDummyAtomsIsNeeded:
    """Verify that BRICS fragments actually have dummies that need stripping."""

    ZINC_SMILES = [
        "c1ccccc1",           # benzene
        "CC(=O)O",            # acetic acid
        "c1ccncc1",           # pyridine
        "CC(=O)Nc1ccccc1",   # acetanilide
        "O=C(O)c1ccccc1",    # benzoic acid
        "c1ccc2[nH]ccc2c1",  # indole
    ]

    def test_brics_fragments_have_dummies(self):
        """BRICS decomposition produces fragments with dummy atoms."""
        fragments_with_dummies = 0
        total_fragments = 0

        for smiles in self.ZINC_SMILES:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            frags = BRICS.BRICSDecompose(mol, returnMols=False)
            for frag_smi in frags:
                frag = Chem.MolFromSmiles(frag_smi)
                if frag is None:
                    continue
                total_fragments += 1
                has_dummy = any(a.GetAtomicNum() == 0 for a in frag.GetAtoms())
                if has_dummy:
                    fragments_with_dummies += 1

        assert total_fragments > 0, "Should have produced some fragments"
        # Most BRICS fragments have dummies; molecules with no BRICS-cleavable
        # bonds are returned whole (without dummies).  strip_dummy_atoms
        # handles both cases.
        assert fragments_with_dummies > 0, (
            "At least some BRICS fragments should have dummy atoms"
        )

    def test_single_fragment_in_combine_needs_strip(self):
        """combine_fragments with n=1 calls strip_dummy_atoms."""
        library = FragmentLibrary(min_atoms=2, max_atoms=30)

        # Manually build a small library from benzene
        mol = Chem.MolFromSmiles("CC(=O)Nc1ccccc1")
        frags = BRICS.BRICSDecompose(mol, returnMols=False)
        library.fragments = list(frags)

        # Sample 1 fragment and try to combine
        frag_mols = library.sample_fragments(1)
        result = library.combine_fragments(frag_mols)

        # strip_dummy_atoms was called; result should be a valid mol
        # without any dummy atoms (or None if stripping failed)
        if result is not None:
            dummy_count = sum(
                1 for a in result.GetAtoms() if a.GetAtomicNum() == 0
            )
            assert dummy_count == 0, "Result should have no dummy atoms"


class TestStripDummyAtomsCorrectness:
    """Test that strip_dummy_atoms produces valid molecules."""

    def test_simple_fragment(self):
        """Stripping a single dummy from a simple fragment."""
        frag = Chem.MolFromSmiles("[1*]CC")
        result = strip_dummy_atoms(frag)
        assert result is not None
        assert result.GetNumAtoms() == 2  # CC
        assert Chem.MolToSmiles(result) == "CC"

    def test_aromatic_fragment(self):
        """Stripping dummies from an aromatic fragment preserves aromaticity."""
        # Benzene with one BRICS attachment
        frag = Chem.MolFromSmiles("[1*]c1ccccc1")
        result = strip_dummy_atoms(frag)
        assert result is not None
        smiles = Chem.MolToSmiles(result)
        assert smiles == "c1ccccc1", f"Expected benzene, got {smiles}"

    def test_aromatic_bonds_preserved(self):
        """After stripping, aromatic bonds have BondType.AROMATIC."""
        frag = Chem.MolFromSmiles("[1*]c1ccccc1")
        result = strip_dummy_atoms(frag)
        assert result is not None
        aromatic_bonds = [
            b for b in result.GetBonds()
            if b.GetBondType() == Chem.BondType.AROMATIC
        ]
        assert len(aromatic_bonds) > 0, (
            "Aromatic bonds should be preserved after strip_dummy_atoms"
        )

    def test_multiple_dummies(self):
        """Fragment with two dummy atoms."""
        frag = Chem.MolFromSmiles("[1*]c1ccc([3*])cc1")
        result = strip_dummy_atoms(frag)
        assert result is not None
        dummy_count = sum(
            1 for a in result.GetAtoms() if a.GetAtomicNum() == 0
        )
        assert dummy_count == 0

    def test_no_dummies_passthrough(self):
        """Fragment without dummies is returned unchanged."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        result = strip_dummy_atoms(mol)
        assert result is not None
        assert Chem.MolToSmiles(result) == "c1ccccc1"

    def test_too_small_after_strip(self):
        """Fragment that becomes too small (<2 atoms) returns None."""
        frag = Chem.MolFromSmiles("[1*]C")  # After stripping: just C (1 atom)
        result = strip_dummy_atoms(frag)
        assert result is None

    def test_disconnected_after_strip(self):
        """Fragment that becomes disconnected after stripping returns None."""
        # Dummy atom bridging two components: removing it disconnects them
        frag = Chem.MolFromSmiles("C([1*])([2*])C")
        # This particular case won't disconnect, but let's test the mechanism
        # with a manually constructed disconnected case
        frag = Chem.MolFromSmiles("[1*]C.[2*]N")
        result = strip_dummy_atoms(frag)
        # After stripping both dummies: C.N (disconnected) -> None
        assert result is None

    def test_real_brics_fragments(self):
        """strip_dummy_atoms works on real BRICS fragments from ZINC-like mols."""
        test_smiles = [
            "c1ccncc1",           # pyridine
            "CC(=O)Nc1ccccc1",   # acetanilide
            "O=C(O)c1ccccc1",    # benzoic acid
            "c1ccc2ccccc2c1",    # naphthalene
        ]

        successes = 0
        total = 0

        for smi in test_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            frags = BRICS.BRICSDecompose(mol, returnMols=False)
            for frag_smi in frags:
                frag = Chem.MolFromSmiles(frag_smi)
                if frag is None:
                    continue
                total += 1
                result = strip_dummy_atoms(frag)
                if result is not None:
                    successes += 1
                    # Verify no dummies remain
                    assert not any(
                        a.GetAtomicNum() == 0 for a in result.GetAtoms()
                    )
                    # Verify it's a valid molecule
                    assert Chem.MolToSmiles(result) is not None

        assert total > 0, "Should have tested some fragments"
        assert successes > 0, "At least some fragments should strip successfully"


class TestStripDummyAtomsKekulization:
    """Test that strip_dummy_atoms does not produce kekulization issues."""

    def test_no_kekulize_warnings_on_result(self):
        """Molecules from strip_dummy_atoms should kekulize cleanly."""
        test_smiles = [
            "c1ccncc1",
            "CC(=O)Nc1ccccc1",
            "O=C(O)c1ccccc1",
            "c1ccc2ccccc2c1",
            "c1ccc2[nH]ccc2c1",
        ]

        for smi in test_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            frags = BRICS.BRICSDecompose(mol, returnMols=False)
            for frag_smi in frags:
                frag = Chem.MolFromSmiles(frag_smi)
                if frag is None:
                    continue
                result = strip_dummy_atoms(frag)
                if result is None:
                    continue

                # The round-trip should produce the same molecule —
                # if it can't kekulize, MolFromSmiles returns None
                roundtrip = Chem.MolFromSmiles(Chem.MolToSmiles(result))
                assert roundtrip is not None, (
                    f"strip_dummy_atoms result for fragment '{frag_smi}' "
                    f"(from '{smi}') failed SMILES round-trip"
                )

                # Explicit kekulization should succeed
                try:
                    rw = Chem.RWMol(result)
                    Chem.Kekulize(rw, clearAromaticFlags=False)
                except Exception as exc:
                    pytest.fail(
                        f"Kekulization failed for stripped fragment "
                        f"'{frag_smi}' (from '{smi}'): {exc}"
                    )
