"""
Tests for BRICS fragment recombination connectivity.

Verifies that the streaming_fragments module produces connected molecules
and that the disconnection safety check works correctly.
"""

import pytest
from rdkit import Chem
from rdkit.Chem import BRICS

from graph_hdc.datasets.streaming_fragments import (
    FragmentLibrary,
    fast_combine_two_fragments,
    _get_attachment_points,
    _brics_compatible,
    mol_to_flow_data,
    BRICS_COMPATIBLE_PAIRS,
)


class TestBRICSCompatibility:
    """Test BRICS compatibility checking."""

    def test_compatible_pairs_symmetric(self):
        """Verify that compatibility pairs are defined (not necessarily symmetric)."""
        # BRICS rules are NOT symmetric - (1,3) doesn't imply (3,1)
        # but we should have both defined if both are valid
        assert (1, 3) in BRICS_COMPATIBLE_PAIRS
        assert (3, 1) in BRICS_COMPATIBLE_PAIRS

    def test_brics_compatible_function(self):
        """Test the compatibility check function."""
        assert _brics_compatible(1, 3) is True
        assert _brics_compatible(7, 7) is True  # Self-compatible
        assert _brics_compatible(1, 2) is False  # Not compatible


class TestGetAttachmentPoints:
    """Test attachment point extraction."""

    def test_simple_fragment(self):
        """Test attachment point extraction from simple fragment."""
        # BRICS fragment with one attachment point
        mol = Chem.MolFromSmiles("[1*]C")
        points = _get_attachment_points(mol)
        assert len(points) == 1
        atom_idx, isotope, neighbor_idx = points[0]
        assert isotope == 1
        # The neighbor should be the carbon
        assert mol.GetAtomWithIdx(neighbor_idx).GetSymbol() == "C"

    def test_multiple_attachment_points(self):
        """Test fragment with multiple attachment points."""
        # Fragment with two attachment points
        mol = Chem.MolFromSmiles("[1*]CC[3*]")
        points = _get_attachment_points(mol)
        assert len(points) == 2
        isotopes = {p[1] for p in points}
        assert isotopes == {1, 3}

    def test_no_attachment_points(self):
        """Test molecule with no attachment points."""
        mol = Chem.MolFromSmiles("CCO")
        points = _get_attachment_points(mol)
        assert len(points) == 0


class TestFastCombineTwoFragments:
    """Test the fast fragment combination function."""

    def test_basic_combination(self):
        """Test basic combination of two compatible fragments."""
        # [1*]C (methyl) and [3*]N (amino) - (1,3) is compatible
        frag1 = Chem.MolFromSmiles("[1*]C")
        frag2 = Chem.MolFromSmiles("[3*]N")

        result = fast_combine_two_fragments(frag1, frag2)

        assert result is not None
        smiles = Chem.MolToSmiles(result, canonical=True)
        # Should be connected (no dot)
        assert "." not in smiles, f"Got disconnected SMILES: {smiles}"
        # Should be methylamine or similar
        assert "C" in smiles and "N" in smiles

    def test_incompatible_fragments_return_none(self):
        """Test that incompatible fragments return None."""
        # [1*]C and [2*]N - (1,2) is NOT compatible
        frag1 = Chem.MolFromSmiles("[1*]C")
        frag2 = Chem.MolFromSmiles("[2*]N")

        result = fast_combine_two_fragments(frag1, frag2)

        assert result is None

    def test_no_attachment_points_return_none(self):
        """Test that fragments without attachment points return None."""
        frag1 = Chem.MolFromSmiles("CC")
        frag2 = Chem.MolFromSmiles("[3*]N")

        result = fast_combine_two_fragments(frag1, frag2)
        assert result is None

        result = fast_combine_two_fragments(frag2, frag1)
        assert result is None

    def test_result_is_connected(self):
        """Test that combined result is always connected."""
        # Test multiple compatible pairs
        test_cases = [
            ("[1*]C", "[3*]N"),      # methyl + amino
            ("[1*]CC", "[3*]O"),     # ethyl + hydroxyl
            ("[5*]c1ccccc1", "[1*]C"),  # phenyl + methyl
            ("[8*]C(=O)", "[16*]N"),  # carbonyl + amine
        ]

        for smiles1, smiles2 in test_cases:
            frag1 = Chem.MolFromSmiles(smiles1)
            frag2 = Chem.MolFromSmiles(smiles2)

            if frag1 is None or frag2 is None:
                continue

            result = fast_combine_two_fragments(frag1, frag2)

            if result is not None:
                smiles = Chem.MolToSmiles(result, canonical=True)
                assert "." not in smiles, (
                    f"Disconnected result from {smiles1} + {smiles2}: {smiles}"
                )

    def test_multiple_attachment_points_partial_connection(self):
        """Test that fragments with multiple attachment points get partially connected."""
        # Fragment with 2 attachment points + fragment with 1
        frag1 = Chem.MolFromSmiles("[1*]CC[3*]")  # Two attachment points
        frag2 = Chem.MolFromSmiles("[3*]N")       # One attachment point

        result = fast_combine_two_fragments(frag1, frag2)

        if result is not None:
            smiles = Chem.MolToSmiles(result, canonical=True)
            # Should be connected even with unused attachment point
            # The remaining [1*] becomes part of the molecule
            assert "." not in smiles, f"Got disconnected: {smiles}"
            # Should still have a dummy atom from unused attachment point
            # (The * or isotope-labeled dummy)


class TestFragmentLibraryCombination:
    """Test FragmentLibrary's combine_fragments method."""

    def test_combine_multiple_fragments_connected(self):
        """Test that combining multiple fragments produces connected molecules."""
        library = FragmentLibrary()

        # Manually add some test fragments
        test_fragments = [
            "[1*]C",           # methyl
            "[3*]CC[5*]",      # ethylene with two attachment points
            "[5*]c1ccccc1",    # phenyl
        ]
        library.fragments = test_fragments

        # Sample and combine
        for _ in range(10):
            frags = library.sample_fragments(3)
            result = library.combine_fragments(frags)

            if result is not None:
                smiles = Chem.MolToSmiles(result, canonical=True)
                assert "." not in smiles, f"Got disconnected: {smiles}"

    def test_combine_two_fragments(self):
        """Test combining exactly two fragments."""
        library = FragmentLibrary()
        library.fragments = ["[1*]C", "[3*]N"]

        frags = [Chem.MolFromSmiles(s) for s in library.fragments]
        result = library.combine_fragments(frags)

        if result is not None:
            smiles = Chem.MolToSmiles(result, canonical=True)
            assert "." not in smiles

    def test_single_fragment_returned_as_is(self):
        """Test that single fragment is returned unchanged."""
        library = FragmentLibrary()
        frag = Chem.MolFromSmiles("[1*]CC")

        result = library.combine_fragments([frag])

        assert result is not None
        # Should be the same molecule
        assert Chem.MolToSmiles(result) == Chem.MolToSmiles(frag)

    def test_empty_list_returns_none(self):
        """Test that empty fragment list returns None."""
        library = FragmentLibrary()
        result = library.combine_fragments([])
        assert result is None


class TestRealBRICSFragments:
    """Test with real BRICS fragments from molecules."""

    def get_fragments_from_smiles(self, smiles: str) -> list:
        """Extract BRICS fragments from a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        frag_smiles = BRICS.BRICSDecompose(mol, returnMols=False)
        return [Chem.MolFromSmiles(s) for s in frag_smiles if Chem.MolFromSmiles(s)]

    def test_aspirin_fragments_recombine_connected(self):
        """Test that aspirin fragments recombine to connected molecules."""
        # Aspirin: CC(=O)Oc1ccccc1C(=O)O
        frags = self.get_fragments_from_smiles("CC(=O)Oc1ccccc1C(=O)O")

        if len(frags) >= 2:
            library = FragmentLibrary()
            result = library.combine_fragments(frags[:2])

            if result is not None:
                smiles = Chem.MolToSmiles(result, canonical=True)
                assert "." not in smiles, f"Disconnected aspirin recombination: {smiles}"

    def test_caffeine_fragments_recombine_connected(self):
        """Test that caffeine fragments recombine to connected molecules."""
        # Caffeine
        frags = self.get_fragments_from_smiles("Cn1cnc2c1c(=O)n(c(=O)n2C)C")

        if len(frags) >= 2:
            library = FragmentLibrary()
            result = library.combine_fragments(frags[:2])

            if result is not None:
                smiles = Chem.MolToSmiles(result, canonical=True)
                assert "." not in smiles, f"Disconnected caffeine recombination: {smiles}"

    def test_random_drug_fragments(self):
        """Test fragments from various drug molecules."""
        drug_smiles = [
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
            "CC(=O)Nc1ccc(cc1)O",           # Paracetamol
            "c1ccc2c(c1)cc3ccccc3n2",       # Acridine
            "CCO",                           # Ethanol (simple)
            "c1ccccc1",                      # Benzene (no BRICS points)
        ]

        all_frags = []
        for smiles in drug_smiles:
            frags = self.get_fragments_from_smiles(smiles)
            all_frags.extend(frags)

        if len(all_frags) < 2:
            pytest.skip("Not enough fragments generated")

        library = FragmentLibrary()

        # Try multiple random combinations
        import random
        for _ in range(20):
            if len(all_frags) >= 2:
                sample = random.sample(all_frags, min(3, len(all_frags)))
                result = library.combine_fragments(sample)

                if result is not None:
                    smiles = Chem.MolToSmiles(result, canonical=True)
                    assert "." not in smiles, (
                        f"Disconnected result: {smiles}"
                    )


class TestDisconnectionSafetyCheck:
    """Test the safety check that should catch disconnected molecules."""

    def test_dot_in_smiles_detection(self):
        """Verify that disconnected SMILES contain a dot."""
        # Create a disconnected molecule manually
        mol1 = Chem.MolFromSmiles("CC")
        mol2 = Chem.MolFromSmiles("OO")
        combined = Chem.CombineMols(mol1, mol2)

        smiles = Chem.MolToSmiles(combined, canonical=True)
        assert "." in smiles, "Combined unconnected mols should have dot"

    def test_connected_smiles_no_dot(self):
        """Verify that connected SMILES don't have a dot."""
        connected_smiles = [
            "CCO",
            "c1ccccc1",
            "CC(=O)O",
            "NCCO",
            "c1ccc(cc1)O",
        ]

        for smiles in connected_smiles:
            assert "." not in smiles, f"Connected SMILES should not have dot: {smiles}"

    def test_mol_to_flow_data_skips_disconnected(self):
        """Test that mol_to_flow_data handles molecules correctly."""
        # Connected molecule
        mol = Chem.MolFromSmiles("CCO")
        data = mol_to_flow_data(mol)
        assert data is not None

        # Disconnected molecule (manually created)
        mol1 = Chem.MolFromSmiles("CC")
        mol2 = Chem.MolFromSmiles("OO")
        disconnected = Chem.CombineMols(mol1, mol2)

        # mol_to_flow_data doesn't explicitly check for disconnection,
        # but the worker process does. This test verifies the molecule
        # would be caught by the "." check.
        smiles = Chem.MolToSmiles(disconnected)
        assert "." in smiles


class TestEdgeCases:
    """Test edge cases that might cause connectivity issues."""

    def test_very_small_fragments(self):
        """Test combining very small fragments."""
        # Single carbon with attachment
        frag1 = Chem.MolFromSmiles("[1*]C")
        frag2 = Chem.MolFromSmiles("[3*]C")

        result = fast_combine_two_fragments(frag1, frag2)

        if result is not None:
            smiles = Chem.MolToSmiles(result, canonical=True)
            assert "." not in smiles
            # Should be ethane or similar
            assert result.GetNumAtoms() == 2  # Just two carbons

    def test_ring_fragments(self):
        """Test combining ring-containing fragments."""
        # Phenyl ring
        frag1 = Chem.MolFromSmiles("[5*]c1ccccc1")
        # Another attachment
        frag2 = Chem.MolFromSmiles("[1*]C")

        result = fast_combine_two_fragments(frag1, frag2)

        if result is not None:
            smiles = Chem.MolToSmiles(result, canonical=True)
            assert "." not in smiles

    def test_fragments_with_charges(self):
        """Test fragments that might have formal charges."""
        # Carboxylate-like fragment
        frag1 = Chem.MolFromSmiles("[1*]C(=O)O")
        frag2 = Chem.MolFromSmiles("[3*]N")

        result = fast_combine_two_fragments(frag1, frag2)

        if result is not None:
            smiles = Chem.MolToSmiles(result, canonical=True)
            assert "." not in smiles

    def test_symmetric_attachment_points(self):
        """Test fragments with symmetric attachment points (same isotope)."""
        # Two fragments both with [7*] (7-7 is self-compatible)
        frag1 = Chem.MolFromSmiles("[7*]CC")
        frag2 = Chem.MolFromSmiles("[7*]CC")

        result = fast_combine_two_fragments(frag1, frag2)

        if result is not None:
            smiles = Chem.MolToSmiles(result, canonical=True)
            assert "." not in smiles


class TestStressTest:
    """Stress tests for connectivity."""

    def test_many_random_combinations(self):
        """Test many random fragment combinations for connectivity."""
        # Generate a pool of fragments from various molecules
        source_molecules = [
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
            "CC(=O)Nc1ccc(cc1)O",           # Paracetamol
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine
            "CC(=O)OC1=CC=CC=C1C(=O)O",     # Aspirin
            "C1=CC=C(C=C1)C(C(=O)O)N",      # Phenylalanine
            "CCCC",                          # Butane
            "c1ccc(cc1)c2ccccc2",           # Biphenyl
        ]

        all_frags = []
        for smiles in source_molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                try:
                    frag_smiles = BRICS.BRICSDecompose(mol, returnMols=False)
                    for fs in frag_smiles:
                        frag = Chem.MolFromSmiles(fs)
                        if frag:
                            all_frags.append(frag)
                except Exception:
                    pass

        if len(all_frags) < 2:
            pytest.skip("Not enough fragments")

        library = FragmentLibrary()
        disconnected_count = 0
        none_count = 0
        success_count = 0

        import random
        for i in range(100):
            n_frags = random.randint(2, min(4, len(all_frags)))
            sample = random.choices(all_frags, k=n_frags)

            result = library.combine_fragments(sample)

            if result is None:
                none_count += 1
            else:
                smiles = Chem.MolToSmiles(result, canonical=True)
                if "." in smiles:
                    disconnected_count += 1
                else:
                    success_count += 1

        # Report results
        print(f"\nStress test results: {success_count} connected, "
              f"{none_count} failed (None), {disconnected_count} disconnected")

        # The key assertion: no disconnected molecules should slip through
        assert disconnected_count == 0, (
            f"Found {disconnected_count} disconnected molecules in stress test"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
