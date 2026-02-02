"""
Unit tests for streaming fragment dataset.

Tests the FragmentLibrary, worker generation logic, and data format compatibility.
"""

import multiprocessing as mp
import pickle
import tempfile
import time
from pathlib import Path

import pytest
import torch
from rdkit import Chem
from torch_geometric.data import Batch, Data

from graph_hdc.datasets.streaming_fragments import (
    NODE_FEATURE_DIM,
    NUM_EDGE_CLASSES,
    FragmentLibrary,
    StreamingFragmentDataLoader,
    StreamingFragmentDataset,
    _worker_process,
    get_bond_type_idx,
    mol_to_flow_data,
    mol_to_zinc_data,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing."""
    return [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CC(C)CC",  # Isopentane
        "CCN(CC)CC",  # Triethylamine
        "c1ccc(O)cc1",  # Phenol
        "CC(=O)Nc1ccccc1",  # Acetanilide
        "c1ccc2ccccc2c1",  # Naphthalene
    ]


@pytest.fixture
def sample_dataset(sample_smiles):
    """Create a mock dataset with smiles attribute."""
    class MockData:
        def __init__(self, smiles):
            self.smiles = smiles
    return [MockData(s) for s in sample_smiles]


@pytest.fixture
def fragment_library(sample_dataset):
    """Create a fragment library from sample data."""
    library = FragmentLibrary(min_atoms=1, max_atoms=20)
    library.build_from_dataset(sample_dataset, show_progress=False)
    return library


# =============================================================================
# mol_to_flow_data tests
# =============================================================================


class TestMolToFlowData:
    """Tests for mol_to_flow_data conversion."""

    def test_basic_conversion(self):
        """Test basic molecule conversion."""
        mol = Chem.MolFromSmiles("CCO")
        data = mol_to_flow_data(mol)

        assert data is not None
        assert data.x.shape[1] == NODE_FEATURE_DIM
        assert data.edge_index.shape[0] == 2
        assert data.edge_attr.shape[1] == NUM_EDGE_CLASSES
        assert hasattr(data, "smiles")

    def test_node_features_one_hot(self):
        """Test that node features are 24-dim concatenated one-hot encoded."""
        mol = Chem.MolFromSmiles("CCO")
        data = mol_to_flow_data(mol)

        # Each row should sum to 5 (one-hot per feature: atom, degree, charge, Hs, ring)
        expected_sum = torch.full((data.x.size(0),), 5.0)
        assert torch.allclose(data.x.sum(dim=1), expected_sum)

    def test_edge_features_one_hot(self):
        """Test that edge features are one-hot encoded."""
        mol = Chem.MolFromSmiles("CC=O")  # Acetaldehyde with double bond
        data = mol_to_flow_data(mol)

        if data.edge_attr.numel() > 0:
            assert torch.allclose(data.edge_attr.sum(dim=1), torch.ones(data.edge_attr.size(0)))

    def test_unsupported_atom_returns_none(self):
        """Test that molecules with unsupported atoms return None."""
        mol = Chem.MolFromSmiles("[Pt]")  # Platinum - not in FLOW_ATOM_TYPES
        data = mol_to_flow_data(mol)
        assert data is None

    def test_none_mol_returns_none(self):
        """Test that None molecule returns None."""
        data = mol_to_flow_data(None)
        assert data is None

    def test_symmetric_edges(self):
        """Test that edges are symmetric (both directions)."""
        mol = Chem.MolFromSmiles("CC")
        data = mol_to_flow_data(mol)

        # Should have 2 edges for the C-C bond (both directions)
        assert data.edge_index.size(1) == 2


class TestMolToZincData:
    """Tests for mol_to_zinc_data conversion."""

    def test_basic_conversion(self):
        """Test basic ZINC-style conversion."""
        mol = Chem.MolFromSmiles("CCO")
        data = mol_to_zinc_data(mol)

        assert data is not None
        assert data.x.shape[1] == 5  # ZINC has 5 node features
        assert data.edge_index.shape[0] == 2
        assert hasattr(data, "smiles")

    def test_node_features(self):
        """Test ZINC node feature extraction."""
        mol = Chem.MolFromSmiles("C")
        data = mol_to_zinc_data(mol)

        # Single carbon: atom_type, degree-1, charge, Hs, ring
        assert data.x.shape == (1, 5)


class TestGetBondTypeIdx:
    """Tests for bond type index mapping."""

    def test_none_bond(self):
        """Test None bond returns 0 (no edge)."""
        assert get_bond_type_idx(None) == 0

    def test_single_bond(self):
        """Test single bond."""
        mol = Chem.MolFromSmiles("CC")
        bond = mol.GetBondBetweenAtoms(0, 1)
        assert get_bond_type_idx(bond) == 1

    def test_double_bond(self):
        """Test double bond."""
        mol = Chem.MolFromSmiles("C=C")
        bond = mol.GetBondBetweenAtoms(0, 1)
        assert get_bond_type_idx(bond) == 2

    def test_triple_bond(self):
        """Test triple bond."""
        mol = Chem.MolFromSmiles("C#C")
        bond = mol.GetBondBetweenAtoms(0, 1)
        assert get_bond_type_idx(bond) == 3

    def test_aromatic_bond(self):
        """Test aromatic bond."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        bond = mol.GetBondBetweenAtoms(0, 1)
        assert get_bond_type_idx(bond) == 4


# =============================================================================
# FragmentLibrary tests
# =============================================================================


class TestFragmentLibrary:
    """Tests for FragmentLibrary class."""

    def test_initialization(self):
        """Test library initialization."""
        library = FragmentLibrary(min_atoms=2, max_atoms=20)
        assert library.min_atoms == 2
        assert library.max_atoms == 20
        assert len(library.fragments) == 0

    def test_build_from_dataset(self, sample_dataset):
        """Test building library from dataset."""
        library = FragmentLibrary(min_atoms=1, max_atoms=20)
        library.build_from_dataset(sample_dataset, show_progress=False)

        assert len(library.fragments) > 0

    def test_sample_fragments(self, fragment_library):
        """Test sampling fragments."""
        fragments = fragment_library.sample_fragments(3)

        assert len(fragments) == 3
        for frag in fragments:
            assert isinstance(frag, Chem.Mol)

    def test_sample_with_replacement(self):
        """Test sampling with replacement when n > num_fragments."""
        library = FragmentLibrary(min_atoms=1, max_atoms=50)
        # Create a small library
        class MockData:
            def __init__(self, smiles):
                self.smiles = smiles
        library.build_from_dataset([MockData("CC")], show_progress=False)

        # Sample more than available
        fragments = library.sample_fragments(10)
        assert len(fragments) == 10

    def test_combine_fragments(self, fragment_library):
        """Test combining fragments."""
        fragments = fragment_library.sample_fragments(2)
        combined = fragment_library.combine_fragments(fragments)

        # Combination may or may not work depending on fragments
        # Just check it doesn't crash
        if combined is not None:
            assert isinstance(combined, Chem.Mol)

    def test_combine_single_fragment(self, fragment_library):
        """Test combining a single fragment returns it."""
        fragments = fragment_library.sample_fragments(1)
        combined = fragment_library.combine_fragments(fragments)

        assert combined == fragments[0]

    def test_combine_empty_list(self, fragment_library):
        """Test combining empty list returns None."""
        combined = fragment_library.combine_fragments([])
        assert combined is None

    def test_save_load(self, fragment_library):
        """Test saving and loading library."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "library.pkl"
            fragment_library.save(path)

            loaded = FragmentLibrary.load(path)

            assert loaded.min_atoms == fragment_library.min_atoms
            assert loaded.max_atoms == fragment_library.max_atoms
            assert loaded.fragments == fragment_library.fragments

    def test_num_fragments_property(self, fragment_library):
        """Test num_fragments property."""
        assert fragment_library.num_fragments == len(fragment_library.fragments)
        assert len(fragment_library) == fragment_library.num_fragments

    def test_empty_library_sample_raises(self):
        """Test that sampling from empty library raises error."""
        library = FragmentLibrary()
        with pytest.raises(ValueError, match="empty"):
            library.sample_fragments(3)


# =============================================================================
# BRICSBuild Performance Tests
# =============================================================================


class TestBRICSBuildPerformance:
    """
    Tests demonstrating BRICSBuild combinatorial explosion problem.

    These tests show why BRICSBuild can be slow and compare it with
    alternative approaches.
    """

    def test_count_attachment_points(self):
        """Analyze attachment point distribution in BRICS fragments."""
        from rdkit.Chem import BRICS

        # Test molecules of varying complexity
        test_smiles = [
            "CCO",  # Simple: ethanol
            "c1ccccc1",  # Benzene
            "CC(=O)Nc1ccc(O)cc1",  # Paracetamol - more complex
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine - complex
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
        ]

        print("\n=== Attachment Point Analysis ===")
        for smiles in test_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            frags = BRICS.BRICSDecompose(mol, returnMols=False)
            print(f"\nMolecule: {smiles}")
            print(f"  Num fragments: {len(frags)}")

            for frag_smiles in list(frags)[:5]:  # Show first 5
                frag_mol = Chem.MolFromSmiles(frag_smiles)
                if frag_mol:
                    n_dummy = sum(1 for a in frag_mol.GetAtoms() if a.GetSymbol() == "*")
                    print(f"    {frag_smiles}: {n_dummy} attachment points")

    def test_brics_build_timing_simple(self):
        """Test BRICSBuild timing with simple fragments (few attachment points)."""
        from rdkit.Chem import BRICS

        # Simple fragments with 1 attachment point each
        simple_frags = [
            "[1*]C",  # Methyl
            "[1*]CC",  # Ethyl
            "[1*]CCC",  # Propyl
        ]
        frag_mols = [Chem.MolFromSmiles(s) for s in simple_frags]

        start = time.time()
        products = BRICS.BRICSBuild(frag_mols, maxDepth=2)
        first_product = next(products, None)
        elapsed = time.time() - start

        print(f"\n=== Simple Fragments (1 attachment point each) ===")
        print(f"Time to get first product: {elapsed*1000:.2f} ms")
        if first_product:
            print(f"First product: {Chem.MolToSmiles(first_product)}")

        # Should be very fast
        assert elapsed < 0.1, f"Simple fragments took {elapsed}s, expected < 0.1s"

    def test_brics_build_timing_complex(self):
        """Test BRICSBuild timing with complex fragments (multiple attachment points)."""
        from rdkit.Chem import BRICS

        # Complex fragments with multiple attachment points
        complex_frags = [
            "[1*]c1ccc([2*])cc1",  # Para-substituted benzene (2 points)
            "[3*]C([4*])=O",  # Carbonyl (2 points)
            "[1*]NC([2*])=O",  # Amide (2 points)
            "[5*]c1ccc([6*])c([7*])c1",  # Tri-substituted benzene (3 points)
        ]
        frag_mols = [Chem.MolFromSmiles(s) for s in complex_frags if Chem.MolFromSmiles(s)]

        print(f"\n=== Complex Fragments ({len(frag_mols)} frags, multiple attachment points) ===")
        for i, (smiles, mol) in enumerate(zip(complex_frags, frag_mols)):
            if mol:
                n_dummy = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "*")
                print(f"  Fragment {i}: {smiles} - {n_dummy} attachment points")

        start = time.time()
        products = BRICS.BRICSBuild(frag_mols, maxDepth=2, scrambleReagents=True)

        # Try to get first few products
        n_products = 0
        for product in products:
            n_products += 1
            if n_products >= 5:
                break
            if time.time() - start > 5.0:  # Timeout after 5 seconds
                print(f"  TIMEOUT after {n_products} products")
                break

        elapsed = time.time() - start
        print(f"Time to get {n_products} products: {elapsed*1000:.2f} ms")
        print(f"Average time per product: {elapsed*1000/max(n_products,1):.2f} ms")

    def test_brics_build_enumeration_explosion(self):
        """Demonstrate combinatorial explosion by counting possible products."""
        from rdkit.Chem import BRICS

        # Two fragments with 2 attachment points each
        frags_2x2 = [
            "[1*]c1ccc([2*])cc1",  # 2 points
            "[1*]C([2*])C",  # 2 points
        ]
        frag_mols = [Chem.MolFromSmiles(s) for s in frags_2x2]

        print("\n=== Enumeration Count Test ===")
        print(f"Fragments: {frags_2x2}")

        start = time.time()
        products = list(BRICS.BRICSBuild(frag_mols, maxDepth=2))
        elapsed = time.time() - start

        print(f"Number of products from 2 fragments (2 attachment points each): {len(products)}")
        print(f"Time: {elapsed*1000:.2f} ms")

        # Now try with 3 fragments
        frags_3x2 = frags_2x2 + ["[1*]N([2*])C"]  # Add another 2-point fragment
        frag_mols_3 = [Chem.MolFromSmiles(s) for s in frags_3x2]

        start = time.time()
        products_3 = list(BRICS.BRICSBuild(frag_mols_3, maxDepth=2))
        elapsed_3 = time.time() - start

        print(f"Number of products from 3 fragments (2 attachment points each): {len(products_3)}")
        print(f"Time: {elapsed_3*1000:.2f} ms")
        print(f"Explosion factor: {len(products_3) / max(len(products), 1):.1f}x")

    def test_brics_build_with_real_fragments(self):
        """Test BRICSBuild with fragments from actual BRICS decomposition."""
        import random
        from rdkit.Chem import BRICS

        # Use real drug-like molecules that will produce compatible fragments
        drug_smiles = [
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "CC(=O)Nc1ccc(O)cc1",  # Paracetamol
            "c1ccc2c(c1)cc3ccccc3n2",  # Acridine
        ]

        # Collect all fragments
        all_frags = set()
        for smiles in drug_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                frags = BRICS.BRICSDecompose(mol, returnMols=False)
                all_frags.update(frags)

        frag_mols = [Chem.MolFromSmiles(s) for s in all_frags if Chem.MolFromSmiles(s)]
        print(f"\n=== Real BRICS Fragments Test ===")
        print(f"Collected {len(frag_mols)} fragments from {len(drug_smiles)} molecules")

        # Show fragments with their attachment points
        for frag_smiles in list(all_frags)[:8]:
            mol = Chem.MolFromSmiles(frag_smiles)
            if mol:
                dummies = [(a.GetSymbol(), a.GetIsotope()) for a in mol.GetAtoms() if a.GetSymbol() == "*"]
                print(f"  {frag_smiles}: {len(dummies)} points, labels={[d[1] for d in dummies]}")

        # Time getting first product with 2-4 fragments
        for n_frags in [2, 3, 4]:
            if len(frag_mols) < n_frags:
                continue

            times = []
            successes = 0
            for _ in range(20):
                sample = random.sample(frag_mols, n_frags)
                start = time.time()
                products = BRICS.BRICSBuild(sample, maxDepth=min(n_frags, 3))
                first = next(products, None)
                elapsed = time.time() - start
                times.append(elapsed)
                if first:
                    successes += 1

            avg_time = sum(times) / len(times) * 1000
            print(f"\n  {n_frags} fragments: avg {avg_time:.2f} ms, {successes}/20 successes")

    def test_fast_vs_brics_build_comparison(self):
        """Compare fast_combine_two_fragments with BRICSBuild."""
        import random
        from rdkit.Chem import BRICS
        from graph_hdc.datasets.streaming_fragments import fast_combine_two_fragments

        # Build a realistic fragment library from drug molecules
        drug_smiles = [
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
            "CC(=O)Nc1ccc(O)cc1",  # Paracetamol
            "c1ccc2c(c1)ccc3ccccc32",  # Phenanthrene
            "Cc1ccc(cc1)C(C)C(=O)O",  # Simplified ibuprofen
            "CC(=O)Nc1ccccc1",  # Acetanilide
            "c1ccc(cc1)C(=O)O",  # Benzoic acid
            "CCN(CC)CC",  # Triethylamine
        ]

        all_frags = set()
        for smiles in drug_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                frags = BRICS.BRICSDecompose(mol, returnMols=False)
                all_frags.update(frags)

        frag_mols = [Chem.MolFromSmiles(s) for s in all_frags if Chem.MolFromSmiles(s)]
        print(f"\n=== Fast vs BRICSBuild Comparison ===")
        print(f"Fragment library size: {len(frag_mols)}")

        n_iterations = 100

        # Test fast_combine_two_fragments
        fast_times = []
        fast_successes = 0
        for _ in range(n_iterations):
            f1, f2 = random.sample(frag_mols, 2)
            start = time.time()
            result = fast_combine_two_fragments(f1, f2)
            fast_times.append(time.time() - start)
            if result:
                fast_successes += 1

        # Test BRICSBuild
        brics_times = []
        brics_successes = 0
        for _ in range(n_iterations):
            f1, f2 = random.sample(frag_mols, 2)
            start = time.time()
            products = BRICS.BRICSBuild([f1, f2], maxDepth=1)
            result = next(products, None)
            brics_times.append(time.time() - start)
            if result:
                brics_successes += 1

        fast_avg = sum(fast_times) / len(fast_times) * 1000
        brics_avg = sum(brics_times) / len(brics_times) * 1000

        print(f"\nFast combine: {fast_avg:.3f} ms avg, {fast_successes}/{n_iterations} successes")
        print(f"BRICSBuild:   {brics_avg:.3f} ms avg, {brics_successes}/{n_iterations} successes")
        print(f"Speedup: {brics_avg/fast_avg:.1f}x")

        # Fast method should be significantly faster
        assert fast_avg < brics_avg, "Fast method should be faster than BRICSBuild"

    def test_manual_combination_speed(self):
        """Test faster manual fragment combination approach."""
        import random
        from rdkit.Chem import AllChem, BRICS

        def fast_combine_two_fragments(frag1: Chem.Mol, frag2: Chem.Mol) -> Chem.Mol | None:
            """
            Manually combine two fragments by connecting random attachment points.

            This is much faster than BRICSBuild for getting a single random product.
            """
            # Find attachment points (dummy atoms) in each fragment
            dummy1 = [(a.GetIdx(), a.GetIsotope()) for a in frag1.GetAtoms() if a.GetSymbol() == "*"]
            dummy2 = [(a.GetIdx(), a.GetIsotope()) for a in frag2.GetAtoms() if a.GetSymbol() == "*"]

            if not dummy1 or not dummy2:
                return None

            # Find compatible attachment points (matching BRICS types)
            # BRICS uses isotope labels to indicate compatible connection types
            compatible_pairs = []
            for idx1, iso1 in dummy1:
                for idx2, iso2 in dummy2:
                    # Check BRICS compatibility (simplified - in reality more complex)
                    if _brics_compatible(iso1, iso2):
                        compatible_pairs.append((idx1, idx2))

            if not compatible_pairs:
                # Try any pair as fallback
                compatible_pairs = [(dummy1[0][0], dummy2[0][0])]

            # Pick a random compatible pair
            idx1, idx2 = random.choice(compatible_pairs)

            # Combine molecules
            combined = Chem.CombineMols(frag1, frag2)
            edit_mol = Chem.RWMol(combined)

            # Get neighbor atoms of the dummy atoms
            dummy1_neighbors = list(frag1.GetAtomWithIdx(idx1).GetNeighbors())
            dummy2_neighbors = list(frag2.GetAtomWithIdx(idx2).GetNeighbors())

            if not dummy1_neighbors or not dummy2_neighbors:
                return None

            neighbor1_idx = dummy1_neighbors[0].GetIdx()
            # Adjust idx2 for combined molecule (offset by frag1 size)
            neighbor2_idx = dummy2_neighbors[0].GetIdx() + frag1.GetNumAtoms()

            # Add bond between neighbors
            edit_mol.AddBond(neighbor1_idx, neighbor2_idx, Chem.BondType.SINGLE)

            # Remove dummy atoms (remove higher index first to avoid reindexing issues)
            dummy1_combined = idx1
            dummy2_combined = idx2 + frag1.GetNumAtoms()
            if dummy1_combined > dummy2_combined:
                edit_mol.RemoveAtom(dummy1_combined)
                edit_mol.RemoveAtom(dummy2_combined)
            else:
                edit_mol.RemoveAtom(dummy2_combined)
                edit_mol.RemoveAtom(dummy1_combined)

            try:
                mol = edit_mol.GetMol()
                Chem.SanitizeMol(mol)
                return mol
            except Exception:
                return None

        def _brics_compatible(iso1: int, iso2: int) -> bool:
            """Check if two BRICS attachment points are compatible."""
            # BRICS compatibility rules (simplified)
            # In reality, specific isotope pairs are compatible
            # See RDKit BRICS.py for full rules
            compatible_pairs = {
                (1, 1), (1, 2), (1, 3),
                (2, 1), (2, 2),
                (3, 1), (3, 3), (3, 4),
                (4, 3), (4, 4), (4, 5),
                (5, 4), (5, 5),
                (6, 6), (6, 7),
                (7, 6), (7, 7),
                (8, 8),
            }
            return (iso1, iso2) in compatible_pairs or (iso2, iso1) in compatible_pairs

        # Test with complex fragments
        complex_frags = [
            "[1*]c1ccc([2*])cc1",
            "[3*]C([4*])=O",
            "[1*]NC([2*])=O",
        ]
        frag_mols = [Chem.MolFromSmiles(s) for s in complex_frags if Chem.MolFromSmiles(s)]

        print("\n=== Manual Combination Speed Test ===")

        # Time BRICSBuild
        start = time.time()
        n_brics = 0
        for _ in range(10):
            products = BRICS.BRICSBuild(frag_mols[:2], maxDepth=1)
            mol = next(products, None)
            if mol:
                n_brics += 1
        brics_time = time.time() - start
        print(f"BRICSBuild (10 iterations): {brics_time*1000:.2f} ms ({n_brics} successes)")

        # Time manual combination
        start = time.time()
        n_manual = 0
        for _ in range(10):
            mol = fast_combine_two_fragments(frag_mols[0], frag_mols[1])
            if mol:
                n_manual += 1
        manual_time = time.time() - start
        print(f"Manual combine (10 iterations): {manual_time*1000:.2f} ms ({n_manual} successes)")

        if brics_time > 0 and manual_time > 0:
            print(f"Speedup: {brics_time/manual_time:.1f}x")


# =============================================================================
# Data format compatibility tests
# =============================================================================


class TestDataFormatCompatibility:
    """Test that generated data is compatible with FlowEdgeDecoder."""

    def test_data_has_required_attributes(self):
        """Test that mol_to_flow_data produces all required attributes."""
        mol = Chem.MolFromSmiles("CCO")
        data = mol_to_flow_data(mol)

        # Required attributes for FlowEdgeDecoder
        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert hasattr(data, "edge_attr")
        assert hasattr(data, "smiles")

    def test_x_shape(self):
        """Test node feature shape."""
        mol = Chem.MolFromSmiles("CCCCC")  # 5 carbons
        data = mol_to_flow_data(mol)

        assert data.x.shape == (5, NODE_FEATURE_DIM)

    def test_edge_attr_shape(self):
        """Test edge attribute shape."""
        mol = Chem.MolFromSmiles("C-C-C")  # 2 bonds = 4 directed edges
        data = mol_to_flow_data(mol)

        num_edges = data.edge_index.size(1)
        assert data.edge_attr.shape == (num_edges, NUM_EDGE_CLASSES)

    def test_batch_collation(self):
        """Test that data can be collated into a batch."""
        mols = [Chem.MolFromSmiles(s) for s in ["CC", "CCC", "CCCC"]]
        data_list = [mol_to_flow_data(m) for m in mols]

        batch = Batch.from_data_list(data_list)

        assert batch.num_graphs == 3
        assert hasattr(batch, "batch")
        assert hasattr(batch, "x")
        assert hasattr(batch, "edge_index")
        assert hasattr(batch, "edge_attr")


# =============================================================================
# Integration tests (lightweight, no actual multiprocessing)
# =============================================================================


class TestStreamingDatasetInitialization:
    """Test StreamingFragmentDataset initialization without starting workers."""

    def test_initialization(self, fragment_library):
        """Test dataset initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy checkpoint path (we won't actually start workers)
            checkpoint_path = Path(tmpdir) / "hypernet.pt"
            checkpoint_path.touch()

            dataset = StreamingFragmentDataset(
                fragment_library=fragment_library,
                hypernet_checkpoint_path=checkpoint_path,
                buffer_size=100,
                num_workers=2,
                fragments_range=(2, 4),
                max_nodes=50,
            )

            assert dataset.buffer_size == 100
            assert dataset.num_workers == 2
            assert dataset.fragments_range == (2, 4)
            assert dataset.max_nodes == 50
            assert not dataset._started


class TestStreamingDataLoaderInitialization:
    """Test StreamingFragmentDataLoader initialization."""

    def test_initialization(self, fragment_library):
        """Test dataloader initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "hypernet.pt"
            checkpoint_path.touch()

            dataset = StreamingFragmentDataset(
                fragment_library=fragment_library,
                hypernet_checkpoint_path=checkpoint_path,
                buffer_size=100,
                num_workers=2,
            )

            loader = StreamingFragmentDataLoader(
                dataset=dataset,
                batch_size=16,
                steps_per_epoch=50,
            )

            assert loader.batch_size == 16
            assert loader.steps_per_epoch == 50
            assert len(loader) == 50


# =============================================================================
# Full pipeline tests (require HyperNet, run only if explicitly requested)
# =============================================================================


@pytest.mark.slow
class TestFullPipeline:
    """
    Full pipeline tests with actual HyperNet and multiprocessing.

    These tests are marked as slow and should be run explicitly.
    Run with: pytest -m slow tests/test_streaming_fragments.py
    """

    @pytest.fixture
    def hypernet_checkpoint(self, tmp_path):
        """Create a real HyperNet checkpoint for testing."""
        from graph_hdc.hypernet.encoder import HyperNet
        from graph_hdc.hypernet.configs import get_config

        config = get_config("ZINC_SMILES_HRR_256_F64_5G1NG4")
        hypernet = HyperNet(config)

        checkpoint_path = tmp_path / "hypernet_test.pt"
        hypernet.save(checkpoint_path)

        return checkpoint_path

    @pytest.fixture
    def full_fragment_library(self):
        """Create a more substantial fragment library."""
        smiles_list = [
            "CCO", "CCCO", "CCCCO",
            "CC(=O)O", "CC(=O)OC",
            "c1ccccc1", "c1ccc(O)cc1", "c1ccc(N)cc1",
            "CCN", "CCNC", "CCN(C)C",
            "CC(C)C", "CC(C)(C)C",
        ]
        class MockData:
            def __init__(self, smiles):
                self.smiles = smiles

        library = FragmentLibrary(min_atoms=1, max_atoms=30)
        library.build_from_dataset([MockData(s) for s in smiles_list], show_progress=False)
        return library

    def test_worker_generates_valid_data(self, full_fragment_library, hypernet_checkpoint):
        """Test that a single worker generates valid data."""
        # Use spawn context for compatibility
        ctx = mp.get_context('spawn')
        queue = ctx.Queue(maxsize=10)
        stop_event = ctx.Event()

        worker = ctx.Process(
            target=_worker_process,
            args=(
                0,  # worker_id
                full_fragment_library,
                queue,
                stop_event,
                str(hypernet_checkpoint),
                (2, 3),  # fragments_range
                50,  # max_nodes
                "zinc",  # dataset_name
                10,  # log_interval
            ),
            daemon=True,
        )

        worker.start()

        # Wait for some samples
        samples = []
        timeout = 60  # seconds
        start = time.time()

        while len(samples) < 5 and (time.time() - start) < timeout:
            try:
                data = queue.get(timeout=5)
                samples.append(data)
            except Exception:
                continue

        # Stop worker
        stop_event.set()
        worker.join(timeout=5)

        # Verify samples
        assert len(samples) > 0, "Worker should generate at least one sample"

        for data in samples:
            # Worker sends serialized dicts, not Data objects
            assert "x" in data
            assert "edge_index" in data
            assert "edge_attr" in data
            assert "hdc_vector" in data
            assert "smiles" in data

            # Check shapes (numpy arrays)
            assert data["x"].shape[1] == NODE_FEATURE_DIM
            assert data["edge_attr"].shape[1] == NUM_EDGE_CLASSES
            assert data["hdc_vector"].ndim == 2
            assert data["hdc_vector"].shape[0] == 1

    def test_streaming_dataset_produces_batches(self, full_fragment_library, hypernet_checkpoint):
        """Test full streaming dataset with actual workers."""
        dataset = StreamingFragmentDataset(
            fragment_library=full_fragment_library,
            hypernet_checkpoint_path=hypernet_checkpoint,
            buffer_size=50,
            num_workers=2,
            fragments_range=(2, 3),
            max_nodes=50,
            prefill_fraction=0.1,
        )

        loader = StreamingFragmentDataLoader(
            dataset=dataset,
            batch_size=4,
            steps_per_epoch=5,
        )

        try:
            batches = []
            for batch in loader:
                batches.append(batch)
                if len(batches) >= 3:
                    break

            assert len(batches) >= 3

            for batch in batches:
                assert batch.num_graphs == 4
                assert hasattr(batch, "hdc_vector")

        finally:
            dataset.stop_workers()
