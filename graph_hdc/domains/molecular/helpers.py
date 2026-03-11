"""
Molecular helper functions for experiment workflows.

Canonical location for molecular-specific experiment helpers.
Backward-compatible re-exports exist at ``graph_hdc.utils.experiment_helpers``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from torch_geometric.data import Data

from graph_hdc.domains.molecular.preprocessing import NUM_ATOM_CLASSES, ZINC_IDX_TO_ATOM


# ZINC atom type mapping (index -> symbol) — kept for backward compat
ZINC_IDX_TO_ATOM_COMPAT = ZINC_IDX_TO_ATOM


def pyg_to_mol(data: Data) -> Optional[Chem.Mol]:
    """
    Convert FlowEdgeDecoder output (PyG Data) to RDKit Mol.

    Expects 24-dim one-hot node features where first 9 dims are atom type.
    Edge classes: 0=no-edge, 1=single, 2=double, 3=triple, 4=aromatic

    Args:
        data: PyG Data from FlowEdgeDecoder.sample()

    Returns:
        RDKit Mol or None if conversion fails
    """
    try:
        mol = Chem.RWMol()

        # Get atom types from first 9 dimensions of 24-dim one-hot
        if data.x.dim() > 1:
            atom_type_probs = data.x[:, :NUM_ATOM_CLASSES]
            node_types = atom_type_probs.argmax(dim=-1).cpu().numpy()
        else:
            node_types = data.x.cpu().numpy()

        # Add atoms
        atom_map = {}
        for i, atom_type_idx in enumerate(node_types):
            atom_symbol = ZINC_IDX_TO_ATOM[int(atom_type_idx)]
            atom = Chem.Atom(atom_symbol)
            rdkit_idx = mol.AddAtom(atom)
            atom_map[i] = rdkit_idx

        # Get edge info
        if data.edge_index is None or data.edge_index.numel() == 0:
            return mol.GetMol()

        edge_index = data.edge_index.cpu().numpy()
        if data.edge_attr is not None:
            if data.edge_attr.dim() > 1:
                edge_types = data.edge_attr.argmax(dim=-1).cpu().numpy()
            else:
                edge_types = data.edge_attr.cpu().numpy()
        else:
            edge_types = [1] * edge_index.shape[1]

        bond_type_map = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
            4: Chem.BondType.AROMATIC,
        }

        added_bonds = set()
        for k in range(edge_index.shape[1]):
            i, j = int(edge_index[0, k]), int(edge_index[1, k])
            edge_type = int(edge_types[k])

            if edge_type == 0 or i == j:
                continue

            bond_key = (min(i, j), max(i, j))
            if bond_key in added_bonds:
                continue
            added_bonds.add(bond_key)

            if edge_type in bond_type_map:
                mol.AddBond(atom_map[i], atom_map[j], bond_type_map[edge_type])

        mol = mol.GetMol()

        # Try full sanitization first
        try:
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            pass

        # If full sanitization fails, try partial sanitization
        try:
            Chem.FastFindRings(mol)
            mol.UpdatePropertyCache(strict=False)
            return mol
        except Exception:
            pass

        # If even partial init fails, return None
        return None

    except Exception:
        return None


def scrub_smiles(smiles: str) -> Optional[str]:
    """
    Clean a SMILES string by neutralizing formal charges, removing
    stereochemistry, and stripping explicit hydrogens.

    Args:
        smiles: Input SMILES string.

    Returns:
        Canonical scrubbed SMILES, or None on failure.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Remove stereochemistry (E/Z, R/S)
        Chem.RemoveStereochemistry(mol)

        # Strip explicit hydrogens
        mol = Chem.RemoveAllHs(mol)

        # Neutralize formal charges
        from rdkit.Chem.MolStandardize import rdMolStandardize
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def is_valid_mol(mol: Optional[Chem.Mol]) -> bool:
    """
    Check if molecule is valid (not None and has valid SMILES).

    Args:
        mol: RDKit molecule or None

    Returns:
        True if valid, False otherwise
    """
    if mol is None:
        return False
    try:
        smiles = Chem.MolToSmiles(mol)
        return smiles is not None and len(smiles) > 0
    except Exception:
        return False


def get_canonical_smiles(mol: Optional[Chem.Mol]) -> Optional[str]:
    """
    Get canonical SMILES from molecule.

    Args:
        mol: RDKit molecule or None

    Returns:
        Canonical SMILES string or None if conversion fails
    """
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


# ========= Visualization =========


def draw_mol_or_error(
    mol: Optional[Chem.Mol],
    size: Tuple[int, int] = (200, 200),
) -> Image.Image:
    """
    Draw molecule image or red X if drawing fails.

    Args:
        mol: RDKit molecule or None
        size: Image size (width, height)

    Returns:
        PIL Image
    """
    if mol is not None:
        try:
            # Try to compute 2D coords if not present
            if mol.GetNumConformers() == 0:
                AllChem.Compute2DCoords(mol)
            return Draw.MolToImage(mol, size=size)
        except Exception:
            # Try drawing without 2D coords computation
            try:
                return Draw.MolToImage(mol, size=size)
            except Exception:
                pass

    # Draw red X for invalid/failed molecules
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    margin = 20
    draw.line([(margin, margin), (size[0] - margin, size[1] - margin)], fill="red", width=5)
    draw.line([(size[0] - margin, margin), (margin, size[1] - margin)], fill="red", width=5)
    return img


def create_reconstruction_plot(
    original_mol: Chem.Mol,
    generated_mol: Optional[Chem.Mol],
    original_smiles: str,
    generated_smiles: Optional[str],
    is_valid: bool,
    is_match: bool,
    sample_idx: int,
    save_path: Path,
) -> None:
    """
    Create side-by-side plot of original vs generated molecule.

    Args:
        original_mol: Original RDKit molecule
        generated_mol: Generated RDKit molecule (can be None)
        original_smiles: Original SMILES string
        generated_smiles: Generated SMILES string (can be None)
        is_valid: Whether generated molecule is valid
        is_match: Whether SMILES match
        sample_idx: Sample index for labeling
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original molecule
    try:
        img_original = Draw.MolToImage(original_mol, size=(400, 400))
        axes[0].imshow(img_original)
    except Exception:
        axes[0].text(0.5, 0.5, "Failed to draw", ha="center", va="center", fontsize=14)
    axes[0].set_title(f"Original\n{original_smiles[:50]}{'...' if len(original_smiles) > 50 else ''}", fontsize=10)
    axes[0].axis("off")

    # Plot generated molecule
    if generated_mol is not None:
        try:
            img_generated = Draw.MolToImage(generated_mol, size=(400, 400))
            axes[1].imshow(img_generated)
        except Exception:
            axes[1].text(0.5, 0.5, "Failed to draw", ha="center", va="center", fontsize=14)
    else:
        axes[1].text(0.5, 0.5, "Generation failed", ha="center", va="center", fontsize=14)

    gen_smiles_display = generated_smiles or "N/A"
    if len(gen_smiles_display) > 50:
        gen_smiles_display = gen_smiles_display[:50] + "..."
    axes[1].set_title(f"Generated\n{gen_smiles_display}", fontsize=10)
    axes[1].axis("off")

    # Add status information
    status_color = "green" if is_match else ("orange" if is_valid else "red")
    status_text = "MATCH" if is_match else ("Valid" if is_valid else "Invalid")
    fig.suptitle(f"Sample {sample_idx + 1}: {status_text}", fontsize=14, color=status_color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_tanimoto_similarity(
    mol1: Optional[Chem.Mol],
    mol2: Optional[Chem.Mol],
    radius: int = 2,
    n_bits: int = 2048,
) -> float:
    """
    Compute Tanimoto similarity between two molecules.

    Args:
        mol1: First RDKit molecule
        mol2: Second RDKit molecule
        radius: Morgan fingerprint radius
        n_bits: Fingerprint bit length

    Returns:
        Tanimoto similarity (0.0 if either molecule is invalid)
    """
    if mol1 is None or mol2 is None:
        return 0.0

    # Check that molecules have atoms
    if mol1.GetNumAtoms() == 0 or mol2.GetNumAtoms() == 0:
        return 0.0

    try:
        # Use the newer MorganGenerator API to avoid deprecation warnings
        from rdkit.Chem import rdFingerprintGenerator

        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp1 = generator.GetFingerprint(mol1)
        fp2 = generator.GetFingerprint(mol2)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        # Fallback: return 0.0 for any error
        return 0.0


def load_smiles_from_csv(csv_path: str) -> list[str]:
    """
    Load SMILES strings from a CSV file.

    Args:
        csv_path: Path to CSV file. Must contain a "smiles" column.

    Returns:
        List of SMILES strings (NaN values dropped).

    Raises:
        ValueError: If CSV file does not contain a "smiles" column.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "smiles" not in df.columns:
        raise ValueError(f"CSV file must have 'smiles' column. Found: {list(df.columns)}")

    return df["smiles"].dropna().tolist()


def smiles_to_pyg_data(smiles: str, dataset: str = "zinc") -> Optional[Data]:
    """
    Convert SMILES string to PyG Data object with node features.

    Supports both QM9 (4-dim features) and ZINC (5-dim features) datasets.

    Args:
        smiles: SMILES string to convert.
        dataset: Dataset name for feature encoding ("qm9" or "zinc").

    Returns:
        PyG Data object with ``x``, ``edge_index``, and ``smiles`` attributes,
        or ``None`` if the SMILES cannot be parsed or contains unsupported atoms.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Get atom type mapping based on dataset
    if dataset.lower() == "qm9":
        from graph_hdc.datasets.qm9_smiles import QM9_ATOM_TO_IDX
        atom_to_idx = QM9_ATOM_TO_IDX
        num_features = 4
    else:
        from graph_hdc.datasets.zinc_smiles import ZINC_ATOM_TO_IDX
        atom_to_idx = ZINC_ATOM_TO_IDX
        num_features = 5

    # Build node features
    x = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in atom_to_idx:
            return None  # Unsupported atom type

        atom_type = atom_to_idx[sym]
        degree = max(0, min(5, atom.GetDegree() - 1))
        charge = atom.GetFormalCharge()
        charge_idx = 0 if charge == 0 else (1 if charge > 0 else 2)
        explicit_hs = min(3, atom.GetTotalNumHs())

        if num_features == 4:
            x.append([float(atom_type), float(degree), float(charge_idx), float(explicit_hs)])
        else:
            is_in_ring = float(atom.IsInRing())
            x.append([float(atom_type), float(degree), float(charge_idx), float(explicit_hs), is_in_ring])

    x = torch.tensor(x, dtype=torch.float)

    # Build edge index
    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, smiles=smiles)
