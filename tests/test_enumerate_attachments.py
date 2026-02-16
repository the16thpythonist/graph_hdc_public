"""
Unit tests for enumerate_attachment_positions.

Verifies that new attachment point variants are generated correctly
and that they combine with existing fragments to produce valid molecules.
"""

import pytest
from rdkit import Chem
from rdkit.Chem import BRICS

from graph_hdc.datasets.streaming_fragments import (
    WILDCARD_LABEL,
    _get_attachment_points,
    enumerate_attachment_positions,
    fast_combine_two_fragments,
)


def _brics_fragment(smiles: str) -> Chem.Mol:
    """Helper: parse a BRICS fragment SMILES into a Mol."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Failed to parse {smiles}"
    return mol


def _get_brics_fragments(smiles: str):
    """Decompose a molecule and return parsed fragment Mols."""
    mol = Chem.MolFromSmiles(smiles)
    frag_smiles = BRICS.BRICSDecompose(mol, returnMols=False)
    return [_brics_fragment(s) for s in frag_smiles]


class TestEnumerateAttachmentPositions:
    """Tests for the enumerate_attachment_positions function."""

    def test_benzene_fragment_gets_new_positions(self):
        """A benzene ring fragment should gain attachment points at other carbons."""
        frag = _brics_fragment("[16*]c1ccccc1")
        variants = enumerate_attachment_positions(frag)

        assert len(variants) > 0, "Should generate at least one variant"

        orig_dummies = sum(1 for a in frag.GetAtoms() if a.GetSymbol() == "*")
        for v in variants:
            new_dummies = sum(1 for a in v.GetAtoms() if a.GetSymbol() == "*")
            assert new_dummies == orig_dummies + 1

    def test_variants_are_valid_molecules(self):
        """All generated variants should pass RDKit sanitization."""
        frag = _brics_fragment("[16*]c1ccccc1")
        variants = enumerate_attachment_positions(frag)

        for v in variants:
            Chem.SanitizeMol(v)

    def test_new_points_use_wildcard_label(self):
        """New attachment points should use the universal wildcard label (0)."""
        frag = _brics_fragment("[16*]c1ccccc1")
        variants = enumerate_attachment_positions(frag)

        for v in variants:
            points = _get_attachment_points(v)
            labels = {iso for _, iso, _ in points}
            assert WILDCARD_LABEL in labels, (
                f"Expected wildcard label {WILDCARD_LABEL} in {labels}"
            )
            # Original label 16 should still be present
            assert 16 in labels

    def test_no_variants_for_fully_substituted(self):
        """A fragment with no available H-bearing atoms yields no variants."""
        frag = _brics_fragment("[1*]C([1*])([1*])[1*]")
        variants = enumerate_attachment_positions(frag)
        assert len(variants) == 0

    def test_max_new_points_limits_output(self):
        """max_new_points should cap the number of generated variants."""
        frag = _brics_fragment("[16*]c1ccccc1")
        variants = enumerate_attachment_positions(frag, max_new_points=2)
        assert len(variants) <= 2

    def test_no_duplicate_variants(self):
        """Symmetric positions should be deduplicated by canonical SMILES."""
        # Cyclohexane has many symmetric carbons that would produce duplicates
        frag = _brics_fragment("[1*]C1CCCCC1")
        variants = enumerate_attachment_positions(frag, max_new_points=20)
        smiles_list = [Chem.MolToSmiles(v) for v in variants]
        assert len(smiles_list) == len(set(smiles_list)), (
            f"Found duplicates: {smiles_list}"
        )

    def test_cyclohexane_dedup_count(self):
        """Cyclohexane should have exactly 3 unique variants (ortho, meta, para)."""
        frag = _brics_fragment("[1*]C1CCCCC1")
        variants = enumerate_attachment_positions(frag, max_new_points=20)
        # 1,2 (adjacent), 1,3 (one apart), 1,4 (opposite)
        assert len(variants) == 3

    def test_pyridine_dedup_count(self):
        """Pyridine [16*]c1ccncc1 should have exactly 2 unique variants."""
        frag = _brics_fragment("[16*]c1ccncc1")
        variants = enumerate_attachment_positions(frag, max_new_points=20)
        # Due to mirror symmetry through the N-C(attached) axis
        assert len(variants) == 2


class TestWildcardCompatibility:
    """Test that the wildcard label connects to all fragment types."""

    @pytest.mark.parametrize("partner_smiles,partner_label", [
        ("[3*]O", 3),           # hydroxyl
        ("[1*]C", 1),           # methyl
        ("[16*]c1ccccc1", 16),  # phenyl
        ("[6*]C(=O)O", 6),     # carboxyl
        ("[11*]S", 11),         # thiol
        ("[9*]n1cccc1", 9),     # aromatic N
    ])
    def test_wildcard_combines_with_all_labels(self, partner_smiles, partner_label):
        """A fragment with wildcard label should combine with any BRICS label."""
        # Cyclohexane variant with wildcard attachment
        frag = _brics_fragment("[1*]C1CCCCC1")
        variants = enumerate_attachment_positions(frag, max_new_points=20)
        assert len(variants) > 0

        partner = _brics_fragment(partner_smiles)
        # At least one variant should successfully combine with this partner
        results = []
        for v in variants:
            result = fast_combine_two_fragments(v, partner)
            if result is not None:
                results.append(result)

        assert len(results) > 0, (
            f"Wildcard label should combine with label {partner_label} "
            f"({partner_smiles}), but all attempts failed"
        )
        # Verify the result is a valid molecule
        for r in results:
            Chem.SanitizeMol(r)


class TestEnumeratedCombination:
    """Test that enumerated variants actually combine to produce new molecules."""

    def test_disubstituted_benzene_from_variants(self):
        """
        Core test: a benzene fragment with one attachment point can't make
        disubstituted benzene alone. With enumerated variants (2 attachment
        points), it should combine with two hydroxyl fragments.
        """
        ring = _brics_fragment("[16*]c1ccccc1")
        hydroxyl = _brics_fragment("[3*]O")

        # First, combine ring + OH to get phenol-like intermediate
        intermediate = fast_combine_two_fragments(ring, hydroxyl)
        assert intermediate is not None, "Should combine ring + OH"

        # The intermediate has no more attachment points — can't add second OH
        points = _get_attachment_points(intermediate)
        assert len(points) == 0, "Plain phenol has no remaining attachment points"

        # Now try with an enumerated variant that has TWO attachment points
        variants = enumerate_attachment_positions(ring, max_new_points=10)
        two_point_variants = [
            v for v in variants
            if sum(1 for a in v.GetAtoms() if a.GetSymbol() == "*") == 2
        ]
        assert len(two_point_variants) > 0, "Should have variants with 2 points"

        success_count = 0
        for variant in two_point_variants:
            step1 = fast_combine_two_fragments(variant, hydroxyl)
            if step1 is None:
                continue
            remaining = _get_attachment_points(step1)
            if len(remaining) == 0:
                continue
            step2 = fast_combine_two_fragments(step1, hydroxyl)
            if step2 is not None:
                success_count += 1
                smiles = Chem.MolToSmiles(step2)
                assert "O" in smiles, f"Expected hydroxyl groups in {smiles}"
                n_dummy = sum(1 for a in step2.GetAtoms() if a.GetSymbol() == "*")
                assert n_dummy == 0, f"Should have no dummy atoms left, got {n_dummy}"

        assert success_count > 0, (
            "At least one variant should produce a dihydroxybenzene"
        )

    def test_disubstituted_cyclohexane(self):
        """
        Cyclohexane with one BRICS point + methyl: previously impossible
        because label 1 can't connect to label 1. With wildcard label, it works.
        """
        ring = _brics_fragment("[1*]C1CCCCC1")
        methyl = _brics_fragment("[1*]C")

        # Original ring can combine with methyl via label 1-1? No — (1,1) not in BRICS
        # But the original ring has label 1 and methyl has label 1, and (1,1) is not
        # a standard BRICS pair. However, with wildcard the NEW point (label 0) works.
        variants = enumerate_attachment_positions(ring, max_new_points=10)

        success_count = 0
        for v in variants:
            result = fast_combine_two_fragments(v, methyl)
            if result is not None:
                success_count += 1
                Chem.SanitizeMol(result)

        assert success_count > 0, (
            "Cyclohexane wildcard points should combine with methyl fragments"
        )

    def test_heteroaromatic_combinations(self):
        """Enumerated points on heteroaromatic rings produce valid molecules."""
        test_cases = [
            ("[16*]c1ccncc1", "pyridine"),
            ("[16*]c1ccoc1", "furan"),
            ("[16*]c1ccsc1", "thiophene"),
            ("[16*]c1ccc2[nH]ccc2c1", "indole"),
        ]
        methyl = _brics_fragment("[1*]C")

        for frag_smi, name in test_cases:
            frag = _brics_fragment(frag_smi)
            variants = enumerate_attachment_positions(frag, max_new_points=10)
            assert len(variants) > 0, f"{name} should have variants"

            combined_any = False
            for v in variants:
                result = fast_combine_two_fragments(v, methyl)
                if result is not None:
                    Chem.SanitizeMol(result)
                    combined_any = True

            assert combined_any, f"{name} variants should combine with methyl"

    def test_enumerated_fragments_from_real_decomposition(self):
        """Fragments from real BRICS decomposition gain useful new positions."""
        fragments = _get_brics_fragments("CC(=O)Nc1ccccc1")

        total_new = 0
        for frag in fragments:
            variants = enumerate_attachment_positions(frag)
            total_new += len(variants)

        assert total_new > 0, (
            "Real BRICS fragments should yield at least some new attachment variants"
        )


# Drug-like molecules used by the connectivity/validity stress tests.
_DRUG_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",           # Aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",      # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",    # Caffeine
    "CC(=O)Nc1ccc(O)cc1",              # Paracetamol
    "c1ccc2c(c1)cc3ccccc3n2",          # Acridine
    "CCN(CC)CC",                        # Triethylamine
    "c1ccc(cc1)C(=O)O",               # Benzoic acid
    "CC(=O)Nc1ccccc1",                 # Acetanilide
    "OC(=O)c1cccnc1",                  # Nicotinic acid
    "c1ccc(-c2ccccc2)cc1",             # Biphenyl
    "CC(O)CC",                          # 2-Butanol
    "C1CCCCC1",                         # Cyclohexane
    "c1ccncc1",                         # Pyridine
    "c1ccoc1",                          # Furan
    "c1ccsc1",                          # Thiophene
    "c1ccc2[nH]ccc2c1",               # Indole
    "CC(=O)OCC",                        # Ethyl acetate
    "c1ccc(NC(=O)c2ccccc2)cc1",        # Benzanilide
    "OC(=O)CC(O)(CC(=O)O)C(=O)O",     # Citric acid
    "Clc1ccccc1",                       # Chlorobenzene
]


def _build_fragment_library():
    """Build a BRICS fragment library from the drug-like molecule set."""
    all_frags = set()
    for smiles in _DRUG_SMILES:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            frags = BRICS.BRICSDecompose(mol, returnMols=False)
            all_frags.update(frags)
        except Exception:
            continue
    return [Chem.MolFromSmiles(s) for s in all_frags if Chem.MolFromSmiles(s) is not None]


def _expand_library(frag_mols, max_new_points=5):
    """Return original + enumerated-variant fragment Mols."""
    expanded = list(frag_mols)
    seen = {Chem.MolToSmiles(m) for m in frag_mols}
    for mol in frag_mols:
        for v in enumerate_attachment_positions(mol, max_new_points=max_new_points):
            smi = Chem.MolToSmiles(v)
            if smi not in seen:
                seen.add(smi)
                expanded.append(v)
    return expanded


class TestCombinedMoleculesConnectedAndValid:
    """
    Stress tests that molecules produced by combining enumerated-variant
    fragments are always CONNECTED and chemically VALID.
    """

    @pytest.fixture(scope="class")
    def frag_mols(self):
        return _build_fragment_library()

    @pytest.fixture(scope="class")
    def expanded_mols(self, frag_mols):
        return _expand_library(frag_mols)

    # ------------------------------------------------------------------ #
    # Two-fragment combinations
    # ------------------------------------------------------------------ #

    def test_pairwise_combinations_connected_and_valid(self, expanded_mols):
        """
        Combine many random pairs from the expanded library.
        Every successful combination must be connected and valid.
        """
        import random
        rng = random.Random(42)

        n_trials = 500
        n_success = 0
        for _ in range(n_trials):
            f1, f2 = rng.sample(expanded_mols, 2)
            mol = fast_combine_two_fragments(f1, f2)
            if mol is None:
                continue

            n_success += 1
            smi = Chem.MolToSmiles(mol, canonical=True)

            # --- Connected ---
            assert "." not in smi, (
                f"Disconnected molecule from combining "
                f"{Chem.MolToSmiles(f1)} + {Chem.MolToSmiles(f2)}: {smi}"
            )

            # --- Valid ---
            try:
                Chem.SanitizeMol(mol)
            except Exception as exc:
                pytest.fail(
                    f"Invalid molecule from combining "
                    f"{Chem.MolToSmiles(f1)} + {Chem.MolToSmiles(f2)}: "
                    f"{smi} — {exc}"
                )

        assert n_success >= 50, (
            f"Expected at least 50 successful combinations out of {n_trials}, "
            f"got {n_success}"
        )

    # ------------------------------------------------------------------ #
    # Multi-step (3-4 fragment) combinations
    # ------------------------------------------------------------------ #

    def test_multistep_combinations_connected_and_valid(self, expanded_mols):
        """
        Sequentially combine 3-4 fragments (simulating the streaming worker).
        Every successful result must be connected and valid.
        """
        import random
        rng = random.Random(123)

        n_trials = 200
        n_success = 0
        for _ in range(n_trials):
            n_frags = rng.randint(3, 4)
            frags = rng.sample(expanded_mols, min(n_frags, len(expanded_mols)))

            result = frags[0]
            failed = False
            for i in range(1, len(frags)):
                combined = fast_combine_two_fragments(result, frags[i])
                if combined is None:
                    combined = fast_combine_two_fragments(frags[i], result)
                if combined is None:
                    failed = True
                    break
                result = combined

            if failed:
                continue

            n_success += 1
            smi = Chem.MolToSmiles(result, canonical=True)

            assert "." not in smi, (
                f"Disconnected molecule from multi-step combination: {smi}"
            )
            try:
                Chem.SanitizeMol(result)
            except Exception as exc:
                pytest.fail(
                    f"Invalid molecule from multi-step combination: {smi} — {exc}"
                )

        assert n_success >= 10, (
            f"Expected at least 10 successful multi-step combinations out of "
            f"{n_trials}, got {n_success}"
        )

    # ------------------------------------------------------------------ #
    # Wildcard-only combinations (no original BRICS labels involved)
    # ------------------------------------------------------------------ #

    def test_wildcard_only_combinations_valid(self, frag_mols):
        """
        Combine two variants that each ONLY have wildcard attachment points
        (original BRICS points already consumed). Must still be valid.
        """
        import random
        rng = random.Random(99)

        # Generate variants and keep those where ALL dummies are wildcard
        wildcard_only = []
        for mol in frag_mols:
            for v in enumerate_attachment_positions(mol, max_new_points=5):
                points = _get_attachment_points(v)
                if all(iso == WILDCARD_LABEL for _, iso, _ in points):
                    wildcard_only.append(v)

        if len(wildcard_only) < 2:
            pytest.skip("Not enough wildcard-only variants generated")

        n_success = 0
        for _ in range(200):
            f1, f2 = rng.sample(wildcard_only, 2)
            mol = fast_combine_two_fragments(f1, f2)
            if mol is None:
                continue

            n_success += 1
            smi = Chem.MolToSmiles(mol, canonical=True)
            assert "." not in smi, f"Disconnected wildcard-only result: {smi}"
            try:
                Chem.SanitizeMol(mol)
            except Exception as exc:
                pytest.fail(f"Invalid wildcard-only result: {smi} — {exc}")

        assert n_success >= 5, (
            f"Expected at least 5 wildcard-only successes, got {n_success}"
        )

    # ------------------------------------------------------------------ #
    # Exhaustive pairwise for a small subset
    # ------------------------------------------------------------------ #

    def test_exhaustive_small_subset(self):
        """
        Take a small set of fragments, enumerate variants, and try ALL
        pairwise combinations. Every success must be connected and valid.
        """
        base_smiles = [
            "[16*]c1ccccc1",       # phenyl
            "[1*]C",               # methyl
            "[3*]O",               # hydroxyl
            "[6*]C(=O)O",         # carboxyl
            "[1*]CC([3*])=O",     # keto-ethyl
        ]
        base_mols = [_brics_fragment(s) for s in base_smiles]

        # Expand each with up to 3 variants
        all_mols = list(base_mols)
        for mol in base_mols:
            all_mols.extend(enumerate_attachment_positions(mol, max_new_points=3))

        n_tried = 0
        n_success = 0
        for i, m1 in enumerate(all_mols):
            for j, m2 in enumerate(all_mols):
                if i == j:
                    continue
                n_tried += 1
                result = fast_combine_two_fragments(m1, m2)
                if result is None:
                    continue

                n_success += 1
                smi = Chem.MolToSmiles(result, canonical=True)
                assert "." not in smi, (
                    f"Disconnected: {Chem.MolToSmiles(m1)} + "
                    f"{Chem.MolToSmiles(m2)} -> {smi}"
                )
                try:
                    Chem.SanitizeMol(result)
                except Exception as exc:
                    pytest.fail(
                        f"Invalid: {Chem.MolToSmiles(m1)} + "
                        f"{Chem.MolToSmiles(m2)} -> {smi} — {exc}"
                    )

        assert n_success > 0, f"No successful combinations out of {n_tried} tries"

    # ------------------------------------------------------------------ #
    # No residual dummy atoms when all points consumed
    # ------------------------------------------------------------------ #

    def test_no_residual_dummy_atoms(self, expanded_mols):
        """
        When a combination consumes all attachment points, the result
        must contain zero dummy (*) atoms.
        """
        import random
        rng = random.Random(7)

        n_checked = 0
        for _ in range(300):
            f1, f2 = rng.sample(expanded_mols, 2)
            mol = fast_combine_two_fragments(f1, f2)
            if mol is None:
                continue

            remaining = _get_attachment_points(mol)
            if len(remaining) > 0:
                # Still has attachment points — that's fine, but they must
                # be real dummy atoms (symbol "*") with valid isotope labels
                for _, iso, _ in remaining:
                    assert isinstance(iso, int) and iso >= 0
                continue

            # All points consumed — verify no stray dummies
            n_checked += 1
            for atom in mol.GetAtoms():
                assert atom.GetSymbol() != "*", (
                    f"Residual dummy atom in fully-combined molecule: "
                    f"{Chem.MolToSmiles(mol)}"
                )

        assert n_checked > 0, "Should find at least one fully-consumed combination"
