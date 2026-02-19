"""
Streaming Fragment Dataset for FlowEdgeDecoder.

Generates infinite molecular training data by combining BRICS fragments on-the-fly,
encoding with HyperNet, and feeding to the model via a producer-consumer architecture.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import pickle
import random
import time
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import BRICS
from torch import Tensor
from torch_geometric.data import Batch, Data
from tqdm.auto import tqdm


# Configure logging
logger = logging.getLogger(__name__)

# BRICS compatibility rules: which isotope labels can connect
# Based on RDKit BRICS.py environsTable
# Label 0 is a universal wildcard used for enumerated attachment points
# (positions added by enumerate_attachment_positions, not from BRICS cuts).
_BRICS_LABELS = range(0, 17)
BRICS_COMPATIBLE_PAIRS = {
    (1, 3), (1, 5), (1, 10),
    (2, 12), (2, 14), (2, 16),
    (3, 1), (3, 4), (3, 13), (3, 14), (3, 15), (3, 16),
    (4, 3), (4, 5), (4, 11),
    (5, 1), (5, 4), (5, 5), (5, 13), (5, 14), (5, 15), (5, 16),
    (6, 13), (6, 14), (6, 15), (6, 16),
    (7, 7),
    (8, 8), (8, 9), (8, 10), (8, 13), (8, 14), (8, 15), (8, 16),
    (9, 8), (9, 13), (9, 14), (9, 15), (9, 16),
    (10, 1), (10, 8), (10, 13), (10, 14), (10, 15), (10, 16),
    (11, 4), (11, 13), (11, 14), (11, 15), (11, 16),
    (12, 2),
    (13, 3), (13, 5), (13, 6), (13, 8), (13, 9), (13, 10), (13, 11), (13, 13), (13, 14), (13, 15), (13, 16),
    (14, 2), (14, 3), (14, 5), (14, 6), (14, 8), (14, 9), (14, 10), (14, 11), (14, 13), (14, 14), (14, 15), (14, 16),
    (15, 3), (15, 5), (15, 6), (15, 8), (15, 9), (15, 10), (15, 11), (15, 13), (15, 14), (15, 15), (15, 16),
    (16, 2), (16, 3), (16, 5), (16, 6), (16, 8), (16, 9), (16, 10), (16, 11), (16, 13), (16, 14), (16, 15), (16, 16),
}
# Add universal wildcard pairs: label 0 is compatible with every label (both directions)
for _label in _BRICS_LABELS:
    BRICS_COMPATIBLE_PAIRS.add((0, _label))
    BRICS_COMPATIBLE_PAIRS.add((_label, 0))


def _brics_compatible(iso1: int, iso2: int) -> bool:
    """Check if two BRICS attachment points are compatible."""
    return (iso1, iso2) in BRICS_COMPATIBLE_PAIRS


def strip_dummy_atoms(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Remove dummy (``*``) atoms from a fragment, returning a sanitized molecule.

    BRICS fragments retain ``*`` atoms at cut points.  Stripping them and
    re-sanitizing allows single-fragment molecules to be used as valid small
    molecules in the training pipeline.

    Returns ``None`` if the result is invalid (disconnected, too small, or
    fails sanitization).
    """
    dummy_indices = sorted(
        [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0],
        reverse=True,  # remove from end to preserve earlier indices
    )
    if not dummy_indices:
        return mol  # no dummies to strip
    rw = Chem.RWMol(mol)
    for idx in dummy_indices:
        rw.RemoveAtom(idx)
    try:
        result = rw.GetMol()
        Chem.SanitizeMol(result)
        if result.GetNumAtoms() < 2:
            return None
        smiles = Chem.MolToSmiles(result)
        if "." in smiles:
            return None  # disconnected
        # Round-trip through SMILES to force consistent aromaticity/kekulization
        result = Chem.MolFromSmiles(smiles)
        return result
    except Exception:
        return None


# Universal wildcard label for enumerated attachment points.
# Unlike BRICS labels 1-16 which encode specific retrosynthetic bond environments,
# label 0 is compatible with every other label, allowing new attachment points to
# bond freely. Valence correctness is enforced by SanitizeMol at combination time.
WILDCARD_LABEL = 0


def enumerate_attachment_positions(
    frag: Chem.Mol,
    max_new_points: int = 3,
) -> List[Chem.Mol]:
    """
    Generate fragment variants with attachment points at additional H-bearing positions.

    For each hydrogen-bearing non-dummy atom that doesn't already have an attachment
    point, creates a new fragment variant with a dummy atom (*) at that position.
    Only single-site additions are generated (one new * per variant).

    New attachment points use the universal wildcard label (0) which is compatible
    with all BRICS labels. This avoids the problem of environment-specific labels
    being too restrictive (e.g. label 1 for aliphatic C only connects to 3 partners).

    Variants are deduplicated by canonical SMILES to avoid redundant entries from
    symmetric positions (e.g. equivalent carbons on a cyclohexane ring).

    Args:
        frag: A BRICS fragment (Mol with existing dummy atoms).
        max_new_points: Maximum number of unique variants to return.
            If there are more candidate positions than this, a random subset is chosen
            (after deduplication).

    Returns:
        List of new fragment Mol variants (does NOT include the original).
    """
    # Find atoms that already neighbor a dummy atom — skip these
    existing_attachment_neighbors = set()
    for atom in frag.GetAtoms():
        if atom.GetSymbol() == "*":
            for nbr in atom.GetNeighbors():
                existing_attachment_neighbors.add(nbr.GetIdx())

    # Find candidate positions: H-bearing, non-dummy, not already an attachment site
    candidates = []
    for atom in frag.GetAtoms():
        if atom.GetSymbol() == "*":
            continue
        if atom.GetIdx() in existing_attachment_neighbors:
            continue
        if atom.GetTotalNumHs() == 0:
            continue
        candidates.append(atom.GetIdx())

    if not candidates:
        return []

    # Generate all variants first, then deduplicate
    seen_smiles = set()
    variants = []
    for target_idx in candidates:
        try:
            rw = Chem.RWMol(frag)
            # Add a dummy atom with universal wildcard label
            dummy_idx = rw.AddAtom(Chem.Atom(0))  # atomic num 0 = *
            rw.GetAtomWithIdx(dummy_idx).SetIsotope(WILDCARD_LABEL)
            # Bond it to the target atom (single bond)
            rw.AddBond(target_idx, dummy_idx, Chem.BondType.SINGLE)
            # Reduce hydrogen count on target atom
            target_atom = rw.GetAtomWithIdx(target_idx)
            n_explicit = target_atom.GetNumExplicitHs()
            if n_explicit > 0:
                target_atom.SetNumExplicitHs(n_explicit - 1)
            else:
                target_atom.SetNoImplicit(False)
            mol = rw.GetMol()
            Chem.SanitizeMol(mol)
            # Deduplicate by canonical SMILES
            canon = Chem.MolToSmiles(mol)
            if canon not in seen_smiles:
                seen_smiles.add(canon)
                variants.append(mol)
        except Exception:
            continue

    # Subsample if too many unique variants
    if len(variants) > max_new_points:
        variants = random.sample(variants, max_new_points)

    return variants


def _get_attachment_points(mol: Chem.Mol) -> List[Tuple[int, int, int]]:
    """
    Get attachment points (dummy atoms) from a molecule.

    Returns:
        List of (atom_idx, isotope_label, neighbor_idx) tuples
    """
    points = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            neighbors = atom.GetNeighbors()
            if neighbors:
                points.append((atom.GetIdx(), atom.GetIsotope(), neighbors[0].GetIdx()))
    return points


def fast_combine_two_fragments(frag1: Chem.Mol, frag2: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Manually combine two BRICS fragments at compatible attachment points.

    This is much faster than BRICSBuild for getting a single random product
    because it doesn't enumerate all possibilities.

    Args:
        frag1: First fragment with BRICS attachment points
        frag2: Second fragment with BRICS attachment points

    Returns:
        Combined molecule or None if no compatible attachment points
    """
    # Get attachment points from each fragment
    points1 = _get_attachment_points(frag1)
    points2 = _get_attachment_points(frag2)

    if not points1 or not points2:
        return None

    # Find compatible pairs
    compatible_pairs = []
    for idx1, iso1, neighbor1 in points1:
        for idx2, iso2, neighbor2 in points2:
            if _brics_compatible(iso1, iso2):
                compatible_pairs.append((idx1, neighbor1, idx2, neighbor2))

    if not compatible_pairs:
        return None

    # Pick a random compatible pair
    dummy1_idx, neighbor1_idx, dummy2_idx, neighbor2_idx = random.choice(compatible_pairs)

    # Combine molecules
    combined = Chem.CombineMols(frag1, frag2)
    edit_mol = Chem.RWMol(combined)

    # Adjust indices for combined molecule (frag2 atoms are offset)
    offset = frag1.GetNumAtoms()
    dummy2_idx_combined = dummy2_idx + offset
    neighbor2_idx_combined = neighbor2_idx + offset

    # Add bond between the neighbor atoms (not the dummy atoms)
    edit_mol.AddBond(neighbor1_idx, neighbor2_idx_combined, Chem.BondType.SINGLE)

    # Remove dummy atoms (remove higher index first to avoid reindexing issues)
    indices_to_remove = sorted([dummy1_idx, dummy2_idx_combined], reverse=True)
    for idx in indices_to_remove:
        edit_mol.RemoveAtom(idx)

    try:
        mol = edit_mol.GetMol()
        Chem.SanitizeMol(mol)
        # Round-trip through SMILES to force consistent aromaticity/kekulization
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        return mol
    except Exception:
        return None

# Import node feature constants from flow_edge_decoder
from graph_hdc.models.flow_edge_decoder import (
    NODE_FEATURE_DIM,
    NODE_FEATURE_BINS,
    ZINC_ATOM_TO_IDX,
)

# 5-class bond type mapping
BOND_TYPE_TO_IDX = {
    None: 0,  # No edge
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}
NUM_EDGE_CLASSES = 5


def get_bond_type_idx(bond) -> int:
    """Convert RDKit bond to 5-class index."""
    if bond is None:
        return 0
    return BOND_TYPE_TO_IDX.get(bond.GetBondType(), 0)


def mol_to_flow_data(mol: Chem.Mol) -> Optional[Data]:
    """
    Convert RDKit molecule to PyG Data with 24-dim ZINC node features.

    This creates data in the format expected by FlowEdgeDecoder, but without
    the HDC vector (which is added separately by the worker).

    Node features are concatenated one-hot encodings:
    - atom_type: 9 classes (Br, C, Cl, F, I, N, O, P, S)
    - degree: 6 classes (0-5)
    - formal_charge: 3 classes (0=neutral, 1=+1, 2=-1)
    - total_Hs: 4 classes (0-3)
    - is_in_ring: 2 classes (0, 1)
    Total: 24 dimensions

    Args:
        mol: RDKit molecule object

    Returns:
        PyG Data object or None if molecule has unsupported atoms
    """
    if mol is None:
        return None

    # Check all atoms are supported
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ZINC_ATOM_TO_IDX:
            return None

    n_atoms = mol.GetNumAtoms()

    # Build raw features for each atom
    raw_features = []
    for atom in mol.GetAtoms():
        atom_type = ZINC_ATOM_TO_IDX[atom.GetSymbol()]
        degree = max(0, min(5, atom.GetDegree() - 1))  # Clamp to 0-5
        charge = atom.GetFormalCharge()
        charge_idx = 0 if charge == 0 else (1 if charge > 0 else 2)
        num_hs = min(3, atom.GetTotalNumHs())  # Clamp to 0-3
        is_ring = int(atom.IsInRing())
        raw_features.append([atom_type, degree, charge_idx, num_hs, is_ring])

    raw_features = torch.tensor(raw_features, dtype=torch.long)

    # Create concatenated one-hot encoding (24 dims)
    x = torch.zeros(n_atoms, NODE_FEATURE_DIM, dtype=torch.float32)
    offset = 0
    for feat_idx, num_classes in enumerate(NODE_FEATURE_BINS):
        feat_vals = raw_features[:, feat_idx]
        onehot = F.one_hot(feat_vals, num_classes=num_classes).float()
        x[:, offset:offset + num_classes] = onehot
        offset += num_classes

    # Build edge index and attributes
    src, dst = [], []
    edge_attr_list = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = get_bond_type_idx(bond)
        # Add both directions
        src.extend([i, j])
        dst.extend([j, i])
        edge_attr_list.extend([bond_type, bond_type])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = F.one_hot(
        torch.tensor(edge_attr_list, dtype=torch.long),
        num_classes=NUM_EDGE_CLASSES
    ).float()

    smiles = Chem.MolToSmiles(mol, canonical=True)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        smiles=smiles,
        original_x=raw_features.float(),  # Keep raw indices for HDC-guided sampling
    )


def mol_to_zinc_data(mol: Chem.Mol) -> Optional[Data]:
    """
    Convert RDKit molecule to PyG Data with ZINC-style features.

    This is needed for HyperNet encoding since HyperNet expects the original
    dataset format (not the 7-class format).

    Node features: [atom_type, degree-1, formal_charge, total_Hs, is_in_ring]

    Args:
        mol: RDKit molecule object

    Returns:
        PyG Data object or None if molecule has unsupported atoms
    """
    # ZINC atom type mapping
    ZINC_ATOM_TO_IDX = {
        "Br": 0, "C": 1, "Cl": 2, "F": 3, "I": 4, "N": 5, "O": 6, "P": 7, "S": 8
    }

    if mol is None:
        return None

    x = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in ZINC_ATOM_TO_IDX:
            return None
        x.append([
            float(ZINC_ATOM_TO_IDX[sym]),
            float(max(0, atom.GetDegree() - 1)),
            float(atom.GetFormalCharge() if atom.GetFormalCharge() >= 0 else 2),
            float(atom.GetTotalNumHs()),
            float(atom.IsInRing()),
        ])

    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src.extend([i, j])
        dst.extend([j, i])

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        smiles=Chem.MolToSmiles(mol, canonical=True),
    )


class FragmentLibrary:
    """
    Library of molecular fragments extracted using BRICS decomposition.

    Fragments can be sampled and recombined to generate new molecules.
    """

    def __init__(
        self,
        min_atoms: int = 2,
        max_atoms: int = 30,
        remove_hydrogens: bool = True,
        remove_stereo: bool = True,
        remove_charges: bool = False,
    ):
        """
        Initialize fragment library.

        Args:
            min_atoms: Minimum fragment size to keep
            max_atoms: Maximum fragment size to keep
            remove_hydrogens: Remove explicit hydrogens from fragments
            remove_stereo: Remove all stereochemistry (chiral centers and E/Z)
            remove_charges: Remove formal charges; discard fragments that fail
                sanitization after charge removal
        """
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.remove_hydrogens = remove_hydrogens
        self.remove_stereo = remove_stereo
        self.remove_charges = remove_charges
        self.fragments: List[str] = []  # Store as SMILES for memory efficiency
        self._fragment_mols: Optional[List[Chem.Mol]] = None  # Lazy cache

    def _clean_fragment(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        Apply configured cleaning steps to a fragment molecule.

        Returns cleaned Mol or None if the fragment becomes invalid.
        """
        if self.remove_hydrogens:
            mol = Chem.RemoveHs(mol)

        if self.remove_stereo:
            Chem.RemoveStereochemistry(mol)

        if self.remove_charges:
            for atom in mol.GetAtoms():
                atom.SetFormalCharge(0)
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                return None

        return mol

    def build_from_dataset(
        self,
        dataset,
        show_progress: bool = True,
        max_molecules: Optional[int] = None,
    ) -> None:
        """
        Build fragment library from a molecular dataset.

        The dataset should have a 'smiles' attribute on each item.

        Args:
            dataset: PyG dataset or list of Data objects with .smiles attribute
            show_progress: Whether to show progress bar
            max_molecules: Maximum number of molecules to process (None for all)
        """
        fragment_set = set()

        iterator = dataset
        if max_molecules is not None:
            iterator = list(dataset)[:max_molecules]
        if show_progress:
            iterator = tqdm(iterator, desc="Building fragment library")

        for data in iterator:
            smiles = data.smiles if hasattr(data, 'smiles') else None
            if smiles is None:
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            try:
                # BRICS decomposition returns a set of fragment SMILES
                frags = BRICS.BRICSDecompose(mol, returnMols=False)
                for frag_smiles in frags:
                    # Parse fragment to check validity and size
                    frag_mol = Chem.MolFromSmiles(frag_smiles)
                    if frag_mol is None:
                        continue

                    frag_mol = self._clean_fragment(frag_mol)
                    if frag_mol is None:
                        continue

                    n_atoms = frag_mol.GetNumAtoms()
                    if self.min_atoms <= n_atoms <= self.max_atoms:
                        # Check all atoms are in the supported set
                        valid = all(
                            atom.GetSymbol() in ZINC_ATOM_TO_IDX or atom.GetSymbol() == "*"
                            for atom in frag_mol.GetAtoms()
                        )
                        if valid:
                            # Re-canonicalize after cleaning
                            clean_smiles = Chem.MolToSmiles(frag_mol)
                            fragment_set.add(clean_smiles)
            except Exception as e:
                logger.debug(f"BRICS decomposition failed for {smiles}: {e}")
                continue

        self.fragments = list(fragment_set)
        self._fragment_mols = None  # Reset cache
        logger.info(f"Built fragment library with {len(self.fragments)} fragments")

    def save(self, path: Path) -> None:
        """Save fragment library to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "min_atoms": self.min_atoms,
            "max_atoms": self.max_atoms,
            "remove_hydrogens": self.remove_hydrogens,
            "remove_stereo": self.remove_stereo,
            "remove_charges": self.remove_charges,
            "fragments": self.fragments,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Saved fragment library to {path}")

    @classmethod
    def load(cls, path: Path) -> "FragmentLibrary":
        """Load fragment library from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        library = cls(
            min_atoms=state["min_atoms"],
            max_atoms=state["max_atoms"],
            remove_hydrogens=state.get("remove_hydrogens", True),
            remove_stereo=state.get("remove_stereo", True),
            remove_charges=state.get("remove_charges", True),
        )
        library.fragments = state["fragments"]
        return library

    def expand_with_enumerated_positions(self, max_new_points: int = 3) -> int:
        """
        Expand the library by enumerating new attachment positions on existing fragments.

        For each fragment, generates variants with additional attachment points at
        H-bearing positions using the universal wildcard label. Variants are
        deduplicated against the existing library and against each other.

        Args:
            max_new_points: Maximum number of new variants to generate per fragment.

        Returns:
            Number of new fragments added.
        """
        existing = set(self.fragments)
        new_fragments = []

        for smiles in self.fragments:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            variants = enumerate_attachment_positions(mol, max_new_points=max_new_points)
            for v in variants:
                v_smiles = Chem.MolToSmiles(v)
                if v_smiles not in existing:
                    existing.add(v_smiles)
                    new_fragments.append(v_smiles)

        self.fragments.extend(new_fragments)
        self._fragment_mols = None  # Reset cache
        logger.info(
            f"Expanded fragment library with {len(new_fragments)} enumerated variants "
            f"(total: {len(self.fragments)})"
        )
        return len(new_fragments)

    def _get_fragment_mols(self) -> List[Chem.Mol]:
        """Get cached fragment Mol objects."""
        if self._fragment_mols is None:
            self._fragment_mols = []
            for smiles in self.fragments:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    self._fragment_mols.append(mol)
        return self._fragment_mols

    def sample_fragments(self, n: int) -> List[Chem.Mol]:
        """
        Sample n random fragments.

        Args:
            n: Number of fragments to sample

        Returns:
            List of RDKit Mol objects
        """
        if len(self.fragments) == 0:
            raise ValueError("Fragment library is empty. Call build_from_dataset first.")

        mols = self._get_fragment_mols()
        if len(mols) < n:
            # Sample with replacement if not enough fragments
            return random.choices(mols, k=n)
        return random.sample(mols, n)

    def combine_fragments(self, fragments: List[Chem.Mol]) -> Optional[Chem.Mol]:
        """
        Combine fragments using BRICS attachment points.

        Uses a fast manual combination approach that respects BRICS compatibility
        rules but doesn't enumerate all possibilities like BRICSBuild does.

        Args:
            fragments: List of fragment Mol objects

        Returns:
            Combined molecule or None if combination failed
        """
        if len(fragments) == 0:
            return None
        if len(fragments) == 1:
            return strip_dummy_atoms(fragments[0])

        # Sequentially combine fragments using fast manual approach
        result = fragments[0]
        for i in range(1, len(fragments)):
            combined = fast_combine_two_fragments(result, fragments[i])
            if combined is None:
                # Try combining in different order as fallback
                combined = fast_combine_two_fragments(fragments[i], result)
            if combined is None:
                return None
            result = combined

        return result

    def combine_fragments_brics(self, fragments: List[Chem.Mol]) -> Optional[Chem.Mol]:
        """
        Combine fragments using RDKit's BRICSBuild (slower but more thorough).

        This is the original implementation using BRICSBuild which enumerates
        all possible combinations. Use combine_fragments() for faster single
        random combinations.

        Args:
            fragments: List of fragment Mol objects

        Returns:
            Combined molecule or None if combination failed
        """
        if len(fragments) < 2:
            return fragments[0] if fragments else None

        try:
            # BRICSBuild returns a generator of possible combinations
            # We take the first valid one
            products = BRICS.BRICSBuild(fragments)
            for product in products:
                # Sanitize and return first valid product
                try:
                    Chem.SanitizeMol(product)
                    return product
                except Exception:
                    continue
            return None
        except Exception as e:
            logger.debug(f"BRICS combination failed: {e}")
            return None

    @property
    def num_fragments(self) -> int:
        """Number of fragments in library."""
        return len(self.fragments)

    def __len__(self) -> int:
        return len(self.fragments)


def _worker_process(
    worker_id: int,
    fragment_library: FragmentLibrary,
    output_queue: mp.Queue,
    stop_event: mp.Event,
    hypernet_checkpoint_path: str,
    fragments_range: Tuple[int, int],
    max_nodes: int,
    dataset_name: str,
    log_interval: int = 1000,
    encoding_batch_size: int = 32,
) -> None:
    """
    Worker process function that generates samples with batched HDC encoding.

    This is a standalone function to avoid pickling issues with Process subclass.
    Uses batch encoding to significantly speed up HDC computation.

    Args:
        worker_id: Unique identifier for logging
        fragment_library: Fragment library to sample from
        output_queue: Queue to push generated samples
        stop_event: Event to signal shutdown
        hypernet_checkpoint_path: Path to saved HyperNet checkpoint
        fragments_range: (min, max) fragments per molecule
        max_nodes: Maximum allowed atoms
        dataset_name: Dataset name for preprocessing
        log_interval: Log stats every N samples
        encoding_batch_size: Number of molecules to encode together (default 32)
    """
    import sys
    import time as time_module

    from rdkit import RDLogger as _RDLogger
    _RDLogger.DisableLog('rdApp.*')

    def log_msg(msg: str) -> None:
        """Print to stderr for reliable subprocess logging."""
        print(f"[Worker {worker_id}] {msg}", file=sys.stderr, flush=True)

    log_msg("Starting worker process...")

    try:
        from graph_hdc.hypernet import load_hypernet

        log_msg(f"Loading HyperNet from {hypernet_checkpoint_path}")
        hypernet = load_hypernet(hypernet_checkpoint_path, device="cpu")
        hypernet.eval()

        # Check if RW augmentation is needed
        _use_rw = hasattr(hypernet, "rw_config") and hypernet.rw_config.enabled
        if _use_rw:
            from graph_hdc.utils.rw_features import augment_data_with_rw
            _rw_k = hypernet.rw_config.k_values
            _rw_bins = hypernet.rw_config.num_bins
            _rw_boundaries = hypernet.rw_config.bin_boundaries
            _rw_clip_range = hypernet.rw_config.clip_range
            bin_mode = "quantile" if _rw_boundaries else ("clipped" if _rw_clip_range else "uniform")
            log_msg(f"RW augmentation enabled: k={_rw_k}, bins={_rw_bins}, mode={bin_mode}")

        log_msg(f"HyperNet loaded successfully (batch_size={encoding_batch_size})")

    except Exception as e:
        log_msg(f"FATAL: Failed to load HyperNet: {e}")
        import traceback
        traceback.print_exc()
        return

    iteration = 0
    retry_count = 0
    sanitize_fail_count = 0  # Molecules that failed RDKit SanitizeMol
    disconnected_count = 0   # Molecules with disconnected components

    log_msg("Worker ready, starting generation loop")

    # Profiling: track time spent in each step
    profile_fragment = 0.0  # Fragment sampling + BRICS combination
    profile_validation = 0.0  # Size check, SMILES, disconnected check
    profile_conversion = 0.0  # mol_to_zinc_data + mol_to_flow_data
    profile_hdc = 0.0  # HyperNet encoding (batched)
    profile_serialize = 0.0  # Numpy conversion
    profile_queue = 0.0  # Queue put

    while not stop_event.is_set():
        try:
            # === Collect a batch of valid molecules ===
            zinc_data_list = []
            flow_data_list = []

            while len(zinc_data_list) < encoding_batch_size and not stop_event.is_set():
                # === Fragment sampling + BRICS combination ===
                t0 = time_module.perf_counter()
                n_fragments = random.randint(fragments_range[0], fragments_range[1])
                fragments = fragment_library.sample_fragments(n_fragments)
                mol = fragment_library.combine_fragments(fragments)
                t1 = time_module.perf_counter()
                profile_fragment += t1 - t0

                if mol is None:
                    retry_count += 1
                    continue

                # === Validation ===
                t0 = time_module.perf_counter()
                # Check size
                if mol.GetNumAtoms() > max_nodes:
                    t1 = time_module.perf_counter()
                    profile_validation += t1 - t0
                    retry_count += 1
                    continue

                # Check for disconnected fragments
                smiles = Chem.MolToSmiles(mol, canonical=True)
                if "." in smiles:
                    t1 = time_module.perf_counter()
                    profile_validation += t1 - t0
                    disconnected_count += 1
                    retry_count += 1
                    continue

                # Verify chemical validity
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    t1 = time_module.perf_counter()
                    profile_validation += t1 - t0
                    sanitize_fail_count += 1
                    retry_count += 1
                    continue
                t1 = time_module.perf_counter()
                profile_validation += t1 - t0

                # === Data conversion ===
                t0 = time_module.perf_counter()
                zinc_data = mol_to_zinc_data(mol)
                if zinc_data is None:
                    t1 = time_module.perf_counter()
                    profile_conversion += t1 - t0
                    retry_count += 1
                    continue

                flow_data = mol_to_flow_data(mol)
                if flow_data is None:
                    t1 = time_module.perf_counter()
                    profile_conversion += t1 - t0
                    retry_count += 1
                    continue

                # Skip molecules with no bonds
                if flow_data.edge_index.numel() == 0:
                    t1 = time_module.perf_counter()
                    profile_conversion += t1 - t0
                    retry_count += 1
                    continue
                t1 = time_module.perf_counter()
                profile_conversion += t1 - t0

                # Augment with RW features if needed
                if _use_rw:
                    zinc_data = augment_data_with_rw(zinc_data, k_values=_rw_k, num_bins=_rw_bins, bin_boundaries=_rw_boundaries, clip_range=_rw_clip_range)
                    # Extend flow_data node features with one-hot RW bins
                    rw_bin_cols = zinc_data.x[:, 5:]  # (n, len(k_values))
                    rw_onehot_parts = []
                    for col_idx in range(rw_bin_cols.size(1)):
                        rw_onehot_parts.append(
                            F.one_hot(rw_bin_cols[:, col_idx].long(), num_classes=_rw_bins).float()
                        )
                    flow_data.x = torch.cat([flow_data.x] + rw_onehot_parts, dim=-1)

                # Add to batch
                zinc_data_list.append(zinc_data)
                flow_data_list.append(flow_data)

            if stop_event.is_set() or not zinc_data_list:
                break

            # === Batched HyperNet encoding ===
            t0 = time_module.perf_counter()
            with torch.no_grad():
                # Create batched data and encode
                zinc_batch = Batch.from_data_list(zinc_data_list)
                hdc_out = hypernet.forward(zinc_batch)

                # Concatenate [node_terms | graph_embedding] for each graph
                order_zero = hdc_out["node_terms"]
                order_n = hdc_out["graph_embedding"]
                hdc_vectors = torch.cat([order_zero, order_n], dim=-1).float()

            t1 = time_module.perf_counter()
            profile_hdc += t1 - t0

            # === Serialize and push each sample ===
            for i, flow_data in enumerate(flow_data_list):
                # Check stop event before each push
                if stop_event.is_set():
                    break

                t0 = time_module.perf_counter()
                hdc_vector = hdc_vectors[i:i+1].clone().detach()

                serialized_data = {
                    "x": flow_data.x.numpy(),
                    "edge_index": flow_data.edge_index.numpy(),
                    "edge_attr": flow_data.edge_attr.numpy(),
                    "hdc_vector": hdc_vector.numpy(),
                    "smiles": flow_data.smiles,
                }
                t1 = time_module.perf_counter()
                profile_serialize += t1 - t0

                # === Queue put with timeout to allow checking stop_event ===
                t0 = time_module.perf_counter()
                while not stop_event.is_set():
                    try:
                        output_queue.put(serialized_data, timeout=1.0)
                        break  # Successfully put, exit the retry loop
                    except Exception:
                        # Queue full, retry after checking stop_event
                        continue
                t1 = time_module.perf_counter()
                profile_queue += t1 - t0

                if stop_event.is_set():
                    break

                iteration += 1

            # Log with profiling breakdown
            if iteration >= log_interval and iteration % log_interval < encoding_batch_size:
                total_time = (profile_fragment + profile_validation + profile_conversion +
                              profile_hdc + profile_serialize + profile_queue)
                if total_time > 0:
                    pct_frag = 100 * profile_fragment / total_time
                    pct_val = 100 * profile_validation / total_time
                    pct_conv = 100 * profile_conversion / total_time
                    pct_hdc = 100 * profile_hdc / total_time
                    pct_ser = 100 * profile_serialize / total_time
                    pct_queue = 100 * profile_queue / total_time
                    samples_per_sec = iteration / total_time if total_time > 0 else 0
                    total_attempts = iteration + retry_count
                    log_msg(
                        f"Generated {iteration} samples, {retry_count} retries, {samples_per_sec:.1f} samples/sec\n"
                        f"  Fragment+BRICS: {pct_frag:.1f}% | Validation: {pct_val:.1f}% | "
                        f"Conversion: {pct_conv:.1f}% | HDC: {pct_hdc:.1f}% | "
                        f"Serialize: {pct_ser:.1f}% | Queue: {pct_queue:.1f}%\n"
                        f"  Rejected: {disconnected_count} disconnected, "
                        f"{sanitize_fail_count} invalid "
                        f"({100 * (disconnected_count + sanitize_fail_count) / max(total_attempts, 1):.1f}% of attempts)"
                    )

        except Exception as e:
            log_msg(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            retry_count += 1
            continue

    log_msg(f"Stopped after {iteration} samples")


class FragmentWorker(mp.Process):
    """
    Multiprocessing worker that continuously generates molecular samples.

    Each worker loads its own HyperNet from a checkpoint and generates
    molecules by combining BRICS fragments.
    """

    def __init__(
        self,
        worker_id: int,
        fragment_library: FragmentLibrary,
        output_queue: mp.Queue,
        stop_event: mp.Event,
        hypernet_checkpoint_path: Path,
        fragments_range: Tuple[int, int] = (2, 5),
        max_nodes: int = 70,
        dataset_name: str = "zinc",
        log_interval: int = 1000,
        encoding_batch_size: int = 32,
    ):
        """
        Initialize worker.

        Args:
            worker_id: Unique identifier for logging
            fragment_library: Fragment library to sample from
            output_queue: Queue to push generated samples
            stop_event: Event to signal shutdown
            hypernet_checkpoint_path: Path to saved HyperNet checkpoint
            fragments_range: (min, max) fragments per molecule
            max_nodes: Maximum allowed atoms
            dataset_name: Dataset name for preprocessing
            log_interval: Log stats every N iterations
            encoding_batch_size: Number of molecules to encode together
        """
        super().__init__()
        self.worker_id = worker_id
        self.fragment_library = fragment_library
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.hypernet_checkpoint_path = str(hypernet_checkpoint_path)
        self.fragments_range = fragments_range
        self.max_nodes = max_nodes
        self.dataset_name = dataset_name
        self.log_interval = log_interval
        self.encoding_batch_size = encoding_batch_size

    def run(self) -> None:
        """Main worker loop."""
        _worker_process(
            worker_id=self.worker_id,
            fragment_library=self.fragment_library,
            output_queue=self.output_queue,
            stop_event=self.stop_event,
            hypernet_checkpoint_path=self.hypernet_checkpoint_path,
            fragments_range=self.fragments_range,
            max_nodes=self.max_nodes,
            dataset_name=self.dataset_name,
            log_interval=self.log_interval,
            encoding_batch_size=self.encoding_batch_size,
        )


class StreamingFragmentDataset(torch.utils.data.IterableDataset):
    """
    PyTorch IterableDataset that generates molecular data from fragment combinations.

    Manages worker processes that continuously generate samples and push them
    to a shared buffer.
    """

    def __init__(
        self,
        fragment_library: FragmentLibrary,
        hypernet_checkpoint_path: Path,
        buffer_size: int = 10000,
        num_workers: int = 4,
        fragments_range: Tuple[int, int] = (2, 5),
        max_nodes: int = 70,
        dataset_name: str = "zinc",
        prefill_fraction: float = 0.1,
        log_interval: int = 1000,
        encoding_batch_size: int = 32,
    ):
        """
        Initialize streaming dataset.

        Args:
            fragment_library: Fragment library to sample from
            hypernet_checkpoint_path: Path to saved HyperNet checkpoint
            buffer_size: Maximum buffer size
            num_workers: Number of worker processes
            fragments_range: (min, max) fragments per molecule
            max_nodes: Maximum atoms per generated molecule
            dataset_name: Dataset name for preprocessing
            prefill_fraction: Wait for buffer to reach this fraction before yielding
            log_interval: Worker log interval
            encoding_batch_size: Number of molecules to encode together per worker
        """
        super().__init__()
        self.fragment_library = fragment_library
        self.hypernet_checkpoint_path = Path(hypernet_checkpoint_path)
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.fragments_range = fragments_range
        self.max_nodes = max_nodes
        self.dataset_name = dataset_name
        self.prefill_fraction = prefill_fraction
        self.log_interval = log_interval
        self.encoding_batch_size = encoding_batch_size

        self._queue: Optional[mp.Queue] = None
        self._stop_event: Optional[mp.Event] = None
        self._workers: List[FragmentWorker] = []
        self._started = False

    def start_workers(self) -> None:
        """Start all worker processes."""
        if self._started:
            logger.warning("Workers already started")
            return

        # Use 'spawn' context to avoid issues with forking and PyTorch/CUDA
        # This is safer and more portable across platforms
        try:
            ctx = mp.get_context('spawn')
        except ValueError:
            # Fall back to default if spawn not available
            ctx = mp

        # Create queue and stop event using the context
        self._queue = ctx.Queue(maxsize=self.buffer_size)
        self._stop_event = ctx.Event()

        # Create and start workers using the context's Process
        self._workers = []
        # Hide GPU from worker processes — they only need CPU for HDC encoding.
        # We set CUDA_VISIBLE_DEVICES before spawning so that the child
        # processes inherit the empty value and never initialise a CUDA
        # context (~300-500 MB VRAM per process).  The parent's value is
        # restored immediately after all workers have been started.
        import os
        _prev_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        for i in range(self.num_workers):
            worker = ctx.Process(
                target=_worker_process,
                args=(
                    i,  # worker_id
                    self.fragment_library,
                    self._queue,
                    self._stop_event,
                    str(self.hypernet_checkpoint_path),
                    self.fragments_range,
                    self.max_nodes,
                    self.dataset_name,
                    self.log_interval,
                    self.encoding_batch_size,
                ),
                daemon=True,  # Workers should exit when main process exits
            )
            worker.start()
            self._workers.append(worker)

        # Restore parent's CUDA visibility
        if _prev_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = _prev_cuda

        logger.info(f"Started {self.num_workers} workers")
        print(f"[StreamingDataset] Started {self.num_workers} workers (encoding_batch_size={self.encoding_batch_size})", flush=True)

        # Wait for buffer to prefill by trying to get samples
        target_size = int(self.buffer_size * self.prefill_fraction)
        print(f"[StreamingDataset] Waiting for buffer to reach {target_size} samples...", flush=True)

        # Use a timeout-based approach instead of qsize() which is unreliable
        prefill_timeout = 120  # seconds
        start_time = time.time()
        samples_received = 0

        while samples_received < target_size:
            elapsed = time.time() - start_time
            if elapsed > prefill_timeout:
                # Check if any workers are alive
                alive_workers = [w for w in self._workers if w.is_alive()]
                if not alive_workers:
                    raise RuntimeError(
                        f"All workers died during prefill. "
                        f"Check worker logs for errors."
                    )
                else:
                    print(
                        f"[StreamingDataset] Warning: Prefill timeout after {elapsed:.1f}s, "
                        f"but {len(alive_workers)} workers still alive. Continuing...",
                        flush=True
                    )
                    break

            try:
                # Try to get a sample with timeout
                serialized_data = self._queue.get(timeout=5.0)
                # Put it back - we're just checking the queue is filling
                self._queue.put(serialized_data)
                samples_received += 1

                if samples_received % 100 == 0:
                    print(f"[StreamingDataset] Prefill progress: {samples_received}/{target_size}", flush=True)

            except Exception:
                # Queue was empty, check workers
                alive = sum(1 for w in self._workers if w.is_alive())
                if alive == 0:
                    raise RuntimeError("All workers died during prefill")
                # Small sleep before retry
                time.sleep(0.5)

        print(f"[StreamingDataset] Buffer prefilled with ~{samples_received} samples", flush=True)
        self._started = True

    def stop_workers(self) -> None:
        """Gracefully stop all workers."""
        if not self._started:
            return

        print("[StreamingDataset] Stopping workers...", flush=True)
        logger.info("Stopping workers...")

        # Signal workers to stop
        if self._stop_event is not None:
            self._stop_event.set()

        # Drain queue while waiting for workers to stop
        # This unblocks workers that might be stuck on queue.put()
        if self._queue is not None:
            drained = 0
            while True:
                try:
                    self._queue.get_nowait()
                    drained += 1
                except Exception:
                    break
            if drained > 0:
                print(f"[StreamingDataset] Drained {drained} items from queue", flush=True)

        # Wait for workers to finish with periodic queue draining
        for i, worker in enumerate(self._workers):
            # Try to join with short timeout, drain queue if needed
            for _ in range(10):  # Up to 10 seconds per worker
                worker.join(timeout=1.0)
                if not worker.is_alive():
                    break
                # Drain any new items that might have been added
                if self._queue is not None:
                    try:
                        while True:
                            self._queue.get_nowait()
                    except Exception:
                        pass

            if worker.is_alive():
                print(f"[StreamingDataset] Worker {i} did not stop, terminating", flush=True)
                logger.warning(f"Worker {i} did not stop, terminating")
                worker.terminate()
                worker.join(timeout=2)  # Wait for termination

        # Final queue cleanup
        if self._queue is not None:
            try:
                while True:
                    self._queue.get_nowait()
            except Exception:
                pass
            try:
                self._queue.close()
                self._queue.join_thread()
            except Exception:
                pass

        self._workers = []
        self._started = False
        print("[StreamingDataset] All workers stopped", flush=True)
        logger.info("All workers stopped")

    def __iter__(self) -> Iterator[Data]:
        """Yield samples from buffer."""
        if not self._started:
            self.start_workers()

        while True:
            try:
                # Get sample from queue (blocks if empty)
                serialized_data = self._queue.get(timeout=30)

                # Convert from serialized format back to PyG Data
                data = Data(
                    x=torch.from_numpy(serialized_data["x"]),
                    edge_index=torch.from_numpy(serialized_data["edge_index"]),
                    edge_attr=torch.from_numpy(serialized_data["edge_attr"]),
                    hdc_vector=torch.from_numpy(serialized_data["hdc_vector"]),
                    smiles=serialized_data["smiles"],
                )
                yield data
            except Exception as e:
                # Check if workers are still alive
                alive = sum(1 for w in self._workers if w.is_alive())
                if alive == 0:
                    logger.error("All workers died")
                    break
                logger.warning(f"Queue get timeout: {e}")

    def __del__(self) -> None:
        """Cleanup workers on deletion."""
        self.stop_workers()

    @property
    def current_buffer_size(self) -> int:
        """Current number of samples in buffer."""
        if self._queue is None:
            return 0
        return self._queue.qsize()


class StreamingFragmentDataLoader:
    """
    PyG-compatible DataLoader wrapper with epoch concept.

    Batches samples from a StreamingFragmentDataset and defines
    an epoch as a fixed number of steps.
    """

    def __init__(
        self,
        dataset: StreamingFragmentDataset,
        batch_size: int = 32,
        steps_per_epoch: int = 1000,
        collate_fn: Optional[Callable] = None,
        log_buffer_interval: int = 100,
    ):
        """
        Initialize dataloader.

        Args:
            dataset: StreamingFragmentDataset instance
            batch_size: Samples per batch
            steps_per_epoch: Number of batches per "epoch"
            collate_fn: Custom collation function (defaults to Batch.from_data_list)
            log_buffer_interval: Log buffer size every N batches (0 to disable)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.collate_fn = collate_fn or Batch.from_data_list
        self.log_buffer_interval = log_buffer_interval

        self._iterator: Optional[Iterator[Data]] = None

        # Buffer tracking statistics
        self._buffer_samples: List[int] = []
        self._batch_count: int = 0

    def __iter__(self) -> Iterator[Batch]:
        """Yield batches for one epoch."""
        if self._iterator is None:
            self._iterator = iter(self.dataset)

        for step in range(self.steps_per_epoch):
            batch_data = []
            for _ in range(self.batch_size):
                try:
                    data = next(self._iterator)
                    batch_data.append(data)
                except StopIteration:
                    # Restart iterator
                    self._iterator = iter(self.dataset)
                    data = next(self._iterator)
                    batch_data.append(data)

            self._batch_count += 1

            # Track and log buffer size
            if self.log_buffer_interval > 0 and self._batch_count % self.log_buffer_interval == 0:
                buffer_size = self.dataset.current_buffer_size
                self._buffer_samples.append(buffer_size)
                buffer_pct = 100 * buffer_size / self.dataset.buffer_size
                print(
                    f"[DataLoader] Batch {self._batch_count}: "
                    f"buffer={buffer_size}/{self.dataset.buffer_size} ({buffer_pct:.1f}%)",
                    flush=True
                )

            yield self.collate_fn(batch_data)

    def __len__(self) -> int:
        """Return steps per epoch."""
        return self.steps_per_epoch

    def stop(self) -> None:
        """Stop the underlying dataset workers."""
        self.dataset.stop_workers()

    def get_buffer_stats(self) -> dict:
        """
        Get buffer utilization statistics.

        Returns:
            Dict with min, max, mean, and samples of buffer sizes observed.
        """
        if not self._buffer_samples:
            return {
                "min": 0,
                "max": 0,
                "mean": 0.0,
                "samples": [],
                "utilization_pct": 0.0,
            }

        mean_size = sum(self._buffer_samples) / len(self._buffer_samples)
        return {
            "min": min(self._buffer_samples),
            "max": max(self._buffer_samples),
            "mean": mean_size,
            "samples": self._buffer_samples[-10:],  # Last 10 samples
            "utilization_pct": 100 * mean_size / self.dataset.buffer_size,
        }

    def test_iteration(self, num_batches: int = 3) -> bool:
        """
        Test that the dataloader can produce batches.

        Call this before training to verify the streaming pipeline works.

        Args:
            num_batches: Number of batches to test

        Returns:
            True if successful, raises exception otherwise
        """
        print(f"[StreamingDataLoader] Testing {num_batches} batches...", flush=True)

        if self._iterator is None:
            self._iterator = iter(self.dataset)

        for i in range(num_batches):
            batch_data = []
            for _ in range(self.batch_size):
                try:
                    data = next(self._iterator)
                    batch_data.append(data)
                except StopIteration:
                    self._iterator = iter(self.dataset)
                    data = next(self._iterator)
                    batch_data.append(data)

            batch = self.collate_fn(batch_data)
            print(
                f"[StreamingDataLoader] Test batch {i+1}/{num_batches}: "
                f"{batch.num_graphs} graphs, "
                f"x shape: {batch.x.shape}, "
                f"hdc_vector shape: {batch.hdc_vector.shape}",
                flush=True
            )

        print("[StreamingDataLoader] Test successful!", flush=True)
        return True
