"""
Chemistry utilities for molecular graph handling.
"""

from dataclasses import dataclass

import networkx as nx
import torch
from rdkit import Chem
from rdkit.Chem import QED, SanitizeFlags, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Data

from graph_hdc.hypernet.types import Feat

# Atom symbol mappings
QM9_ATOM_SYMBOLS = ["C", "N", "O", "F"]
ZINC_ATOM_SYMBOLS = ["Br", "C", "Cl", "F", "I", "N", "O", "P", "S"]

QM9_ATOM_TO_IDX = {s: i for i, s in enumerate(QM9_ATOM_SYMBOLS)}
ZINC_ATOM_TO_IDX = {s: i for i, s in enumerate(ZINC_ATOM_SYMBOLS)}

# Formal charge mapping: 0 -> 0, 1 -> +1, 2 -> -1
FORMAL_CHARGE_IDX_TO_VAL = {0: 0, 1: +1, 2: -1}


def draw_mol(
    mol: Chem.Mol,
    save_path: str | None = None,
    size: tuple[int, int] = (300, 300),
    fmt: str = "svg",
    bond_width: float = 2.0,
    bw_palette: bool = False,
    font_size: int = 14,
    transparent: bool = True,
) -> None:
    """
    Draw an RDKit molecule with publication-quality styling.

    Args:
        mol: RDKit molecule object
        save_path: Path to save the image (optional)
        size: Image size (width, height)
        fmt: Output format ('svg' or 'png')
        bond_width: Line thickness for bonds
        bw_palette: Use black-and-white palette
        font_size: Font size for atom labels
        transparent: Transparent background
    """
    if mol is None:
        return

    try:
        if mol.GetNumConformers() == 0:
            rdDepictor.Compute2DCoords(mol)
        Chem.NormalizeDepiction(mol)
    except Exception:
        pass

    if fmt == "svg":
        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    elif fmt == "png":
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    opts = drawer.drawOptions()
    opts.bondLineWidth = bond_width
    opts.minFontSize = font_size
    opts.padding = 0.05
    opts.multipleBondOffset = 0.15

    if bw_palette:
        opts.useBWAtomPalette()

    if transparent:
        opts.clearBackground = False
        opts.setBackgroundColour((1, 1, 1, 0))

    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        data = drawer.GetDrawingText()

        if save_path:
            mode = "w" if fmt == "svg" else "wb"
            with open(save_path, mode) as f:
                f.write(data)
    except Exception as e:
        print(f"Error drawing molecule: {e}")


def is_valid_molecule(mol: Chem.Mol) -> bool:
    """Check if molecule is valid (passes sanitization)."""
    if mol is None or mol.GetNumAtoms() == 0:
        return False
    err = Chem.SanitizeMol(mol, catchErrors=True)
    return err == SanitizeFlags.SANITIZE_NONE


def canonical_key(mol: Chem.Mol) -> str:
    """
    Generate a stable canonical SMILES key for uniqueness/novelty checks.

    Removes explicit Hs, sanitizes, and returns canonical isomeric SMILES.
    """
    m = Chem.RemoveHs(Chem.Mol(mol))
    Chem.SanitizeMol(m)
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True, kekuleSmiles=False)


def compute_qed(mol: Chem.Mol) -> float:
    """Compute QED (Quantitative Estimate of Drug-likeness)."""
    try:
        return float(QED.qed(mol))
    except Exception:
        return float("nan")


@dataclass
class ReconstructionResult:
    """Result of molecule reconstruction with diagnostics."""
    mol: Chem.Mol
    strategy: str
    confidence: float
    warnings: list[str]


RECONSTRUCTION_CONFIDENCE = {
    "standard": 1.0,
    "kekulized": 0.95,
    "single_bonds": 0.7,
    "partial_sanitize": 0.6,
}


def nx_to_mol(
    G: nx.Graph,
    dataset: str = "qm9",
    infer_bonds: bool = True,
    sanitize: bool = True,
    kekulize: bool = True,
) -> tuple[Chem.Mol | None, dict[int, int]]:
    """
    Convert a NetworkX graph with node features to an RDKit molecule.

    Args:
        G: NetworkX graph with node 'feat' or 'type' attributes
        dataset: Dataset name ('qm9' or 'zinc') for atom symbol mapping
        infer_bonds: Infer bond orders from valence
        sanitize: Run sanitization after construction
        kekulize: Kekulize aromatic rings

    Returns:
        Tuple of (RDKit Mol, node_id to atom_idx mapping)
    """
    atom_symbols = QM9_ATOM_SYMBOLS if dataset == "qm9" else ZINC_ATOM_SYMBOLS

    if G.number_of_nodes() == 0:
        return None, {}

    # Get node features
    nodes = sorted(G.nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Create editable molecule
    mol = Chem.RWMol()

    # Add atoms
    for n in nodes:
        data = G.nodes[n]
        if "feat" in data:
            feat = data["feat"]
            t = feat.to_tuple() if hasattr(feat, "to_tuple") else tuple(feat)
        elif "type" in data:
            t = data["type"]
        else:
            raise ValueError(f"Node {n} has no 'feat' or 'type' attribute")

        atom_type_idx = int(t[0])
        formal_charge_idx = int(t[2])
        explicit_hs = int(t[3])

        symbol = atom_symbols[atom_type_idx]
        atom = Chem.Atom(symbol)
        atom.SetFormalCharge(FORMAL_CHARGE_IDX_TO_VAL.get(formal_charge_idx, 0))
        atom.SetNumExplicitHs(explicit_hs)
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)

    # Collect edges
    edges = []
    for u, v in G.edges():
        if u != v:
            ui, vi = node_to_idx[u], node_to_idx[v]
            edges.append((min(ui, vi), max(ui, vi)))
    # BUG FIX (2026-04-17): use sorted() for deterministic edge ordering.
    # list(set()) produces hash-dependent ordering, making the greedy bond
    # inference non-deterministic — different graph constructions could yield
    # different bond assignments for the same topology.
    edges = sorted(set(edges))

    # Infer bond orders if requested
    if infer_bonds:
        edges = _infer_bond_orders(mol, edges, G, nodes, atom_symbols)

    # Add bonds
    for ui, vi, btype in edges:
        mol.AddBond(ui, vi, btype)

    mol = mol.GetMol()

    if sanitize:
        try:
            if kekulize:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            Chem.SanitizeMol(mol)
        except Exception:
            pass

    # BUG FIX (2026-04-17): Clear noImplicit flags and re-sanitize.
    #
    # Problem: nx_to_mol builds atoms with SetNoImplicit(True) and
    # SetNumExplicitHs(h) to strictly validate valence during construction.
    # Combined with imperfect greedy bond inference for aromatic rings, this
    # forces RDKit to write bracket notation ([CH], [NH], etc.) in SMILES
    # output even for atoms with standard valence. These non-canonical SMILES
    # produce significantly wrong FCD scores (e.g., 7.2 vs 2.1 for the same
    # molecules) because ChemNet encodes bracketed atoms differently.
    #
    # Fix: clear only the noImplicit flag (keep explicit Hs unchanged), then
    # re-sanitize. This lets RDKit detect aromaticity and treat the explicit
    # Hs as deliberately chosen rather than forced by noImplicit. RDKit will
    # then omit brackets for atoms with standard valence while preserving
    # them where genuinely needed (e.g., [nH] in pyrrole).
    rwmol = Chem.RWMol(mol)
    for atom in rwmol.GetAtoms():
        atom.SetNoImplicit(False)
    mol = rwmol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    return mol, node_to_idx


def _infer_bond_orders(mol, edges, G, nodes, atom_symbols):
    """Infer bond orders based on valence requirements."""
    # Get target valences
    target_valence = {}
    atom_info = {}  # i -> (symbol, charge, explicit_hs, degree)
    for i, n in enumerate(nodes):
        data = G.nodes[n]
        if "feat" in data:
            feat = data["feat"]
            t = feat.to_tuple() if hasattr(feat, "to_tuple") else tuple(feat)
        else:
            t = data["type"]

        atom_type_idx = int(t[0])
        degree_idx = int(t[1])
        formal_charge_idx = int(t[2])
        explicit_hs = int(t[3])

        symbol = atom_symbols[atom_type_idx]
        charge = FORMAL_CHARGE_IDX_TO_VAL.get(formal_charge_idx, 0)

        # BUG FIX (2026-04-17): Multi-valent atom support.
        #
        # Previously, this used a single-valence lookup table:
        #   valences = {"C": 4, "N": 3, "O": 2, ..., "S": 2, "P": 3}
        #   base_val = valences.get(symbol, 4)
        #
        # This fails for elements with multiple allowed valences. For example,
        # sulfur in DMSO (CS(=O)C) has 3 heavy-atom bonds and needs valence 4
        # to accommodate the S=O double bond. But the old table hardcodes S=2,
        # giving target = 2 - 0 = 2 with 3 bonds → deficit = -1. The algorithm
        # thinks S is over-saturated and never assigns the double bond.
        #
        # Fix: two-pass approach (see below). First pass stores per-atom info,
        # second pass resolves multi-valent atoms using neighbor context.
        atom_info[i] = (symbol, charge, explicit_hs, degree_idx + 1)

    # BUG FIX (2026-04-17): charge sign was inverted (was base_val - charge).
    # Positive charge means the atom needs MORE bonds (e.g., N+ needs 4),
    # negative charge means fewer (e.g., O- needs 1).

    # Standard valences per element, sorted ascending. Elements with multiple
    # entries (S, P, N, halogens) can appear in different oxidation states.
    multi_valences = {
        "C": [4], "N": [3, 5], "O": [2], "F": [1],
        "Cl": [1, 3, 5, 7], "Br": [1, 3, 5, 7], "I": [1, 3, 5, 7],
        "S": [2, 4, 6], "P": [3, 5],
    }

    # Build adjacency from edge list for neighbor lookups
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    # Pass 1: compute preliminary targets using smallest feasible valence.
    # "Feasible" means target >= degree (at minimum, all bonds can be single).
    prelim_target = {}
    for i, (symbol, charge, explicit_hs, degree) in atom_info.items():
        candidates = multi_valences.get(symbol, [4])
        base_val = candidates[-1]  # fallback to highest if none feasible
        for v in candidates:
            target = v + charge - explicit_hs
            if target >= degree:
                base_val = v
                break
        prelim_target[i] = base_val + charge - explicit_hs

    # Pass 2: for multi-valent atoms, check if neighbors demand more bond
    # orders than the preliminary target can provide. A neighbor with only
    # 1 heavy-atom bond and target > 1 needs that bond to be double or
    # triple — the multi-valent atom must accommodate this.
    #
    # Example: dimethylsulfone CS(=O)(=O)C
    #   Pass 1: S gets val=4, target=4. With 4 single bonds, deficit=0.
    #   But each O (degree=1, target=2) needs a double bond from S.
    #   Neighbor demand = 1(C) + 2(O) + 2(O) + 1(C) = 6 > 4.
    #   Pass 2: upgrade S to val=6, target=6. Now deficit=2, allowing
    #   2 double bonds to the oxygens.
    for i, (symbol, charge, explicit_hs, degree) in atom_info.items():
        candidates = multi_valences.get(symbol, [4])
        if len(candidates) <= 1:
            continue  # single-valence atom, nothing to adjust

        # Sum the minimum bond order each neighbor requires from this atom.
        # For a neighbor with only 1 bond (degree=1), the minimum bond order
        # to that neighbor equals the neighbor's full target. For neighbors
        # with multiple bonds, each bond is at minimum single (order 1).
        neighbor_demand = 0
        for nbr in adj.get(i, []):
            nbr_sym, nbr_charge, nbr_hs, nbr_degree = atom_info[nbr]
            nbr_target = prelim_target[nbr]
            if nbr_degree == 1:
                # Neighbor has only this one bond; it must carry the full target
                neighbor_demand += max(1, nbr_target)
            else:
                # Neighbor has other bonds too; this bond is at minimum single
                neighbor_demand += 1

        required = max(prelim_target[i], neighbor_demand)
        # Pick the smallest valence that achieves the required target
        for v in candidates:
            target = v + charge - explicit_hs
            if target >= required:
                base_val = v
                break
        else:
            base_val = candidates[-1]  # fallback to highest
        target_valence[i] = base_val + charge - explicit_hs
        continue

    # Fill in single-valence atoms (skipped in pass 2)
    for i in atom_info:
        if i not in target_valence:
            target_valence[i] = prelim_target[i]

    # Start with single bonds
    bond_orders = {(u, v): 1 for u, v in edges}
    current_valence = {i: 0 for i in range(len(nodes))}
    for u, v in edges:
        current_valence[u] += 1
        current_valence[v] += 1

    # BUG FIX (2026-04-17): Fix bond orders in aromatic rings.
    #
    # A single greedy pass depends on edge iteration order: for a 6-membered
    # ring it may assign doubles to bonds (0,1) and (3,4) only, leaving atoms
    # 2 and 5 with valence deficit. This prevents aromaticity detection and
    # produces non-canonical SMILES with brackets ([CH]).
    #
    # Fix: after each greedy pass, look for an unsatisfied atom and find an
    # augmenting path — downgrade a neighbor's double bond to free capacity,
    # then upgrade the bond to the unsatisfied atom. Re-run the greedy pass
    # after each augmentation to catch newly-created opportunities (e.g., two
    # atoms that both gained deficit can now have their shared bond increased).
    #
    # BUG FIX (2026-04-17): Added maximum iteration guard.
    #
    # Previously, the outer loop ran without a bound. This caused infinite
    # loops in two cases:
    #
    # 1. Fused ring systems (e.g., naphthalene): the 1-hop augmenting step
    #    cannot propagate a deficit through multiple bonds. It swaps a double
    #    bond from edge A to edge B, creating a new deficit at the far end of
    #    A. The next iteration swaps it back, oscillating forever. This
    #    affected ~10-25% of random edge orderings for naphthalene.
    #
    # 2. Genuinely unsatisfiable constraints (e.g., an odd cycle where every
    #    atom needs one more bond order, but total deficit is odd): no valid
    #    assignment exists, but the algorithm keeps trying to redistribute
    #    the deficit around the ring.
    #
    # Fix: cap iterations at len(edges) * 3. A correct assignment needs at
    # most len(edges) augmentations (each edge upgraded at most twice, from
    # single→double→triple). If the algorithm hasn't converged by then, it
    # returns the best partial result — some atoms may have unsatisfied
    # valence, but the algorithm terminates.
    max_iterations = len(edges) * 3

    overall_changed = True
    iteration = 0
    while overall_changed and iteration < max_iterations:
        overall_changed = False
        iteration += 1

        # Greedy pass: increase bonds where both endpoints have deficit
        changed = True
        while changed:
            changed = False
            for u, v in edges:
                deficit_u = target_valence[u] - current_valence[u]
                deficit_v = target_valence[v] - current_valence[v]
                if deficit_u > 0 and deficit_v > 0 and bond_orders[(u, v)] < 3:
                    bond_orders[(u, v)] += 1
                    current_valence[u] += 1
                    current_valence[v] += 1
                    changed = True
                    overall_changed = True

        # Augmenting step: find one swap to unblock a stuck atom
        augmented = False
        for atom in range(len(nodes)):
            deficit = target_valence[atom] - current_valence[atom]
            if deficit <= 0:
                continue
            for nbr in adj.get(atom, []):
                edge_key = (min(atom, nbr), max(atom, nbr))
                if bond_orders[edge_key] >= 3:
                    continue
                nbr_deficit = target_valence[nbr] - current_valence[nbr]
                if nbr_deficit > 0:
                    continue  # Both have deficit — greedy pass handles this
                # nbr is satisfied; find one of its OTHER double+ bonds to downgrade
                for nbr2 in adj.get(nbr, []):
                    if nbr2 == atom:
                        continue
                    edge_key2 = (min(nbr, nbr2), max(nbr, nbr2))
                    if bond_orders[edge_key2] <= 1:
                        continue
                    # Downgrade nbr-nbr2, upgrade atom-nbr
                    bond_orders[edge_key2] -= 1
                    current_valence[nbr2] -= 1
                    bond_orders[edge_key] += 1
                    current_valence[atom] += 1
                    augmented = True
                    overall_changed = True
                    break
                if augmented:
                    break
            if augmented:
                break

    # Convert to RDKit bond types
    bond_type_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }
    return [(u, v, bond_type_map[bond_orders[(u, v)]]) for u, v in edges]


def reconstruct_for_eval(
    nx_graph: nx.Graph,
    dataset: str = "qm9",
    return_diagnostics: bool = False,
) -> Chem.Mol | ReconstructionResult:
    """
    Reconstruct RDKit molecule from NetworkX graph with fallback strategies.

    Tries progressively more permissive strategies:
    1. Standard aromatic
    2. Kekulized
    3. Single bonds only
    4. Partial sanitization

    Args:
        nx_graph: NetworkX graph with node features
        dataset: Dataset name ('qm9' or 'zinc')
        return_diagnostics: Return ReconstructionResult with metadata

    Returns:
        RDKit Mol or ReconstructionResult if return_diagnostics=True

    Raises:
        ValueError: If all strategies fail
    """
    warnings = []

    # Strategy 1: Standard with kekulize
    try:
        mol, _ = nx_to_mol(nx_graph, dataset=dataset, infer_bonds=True, sanitize=True, kekulize=True)
        if mol is not None and is_valid_molecule(mol):
            if return_diagnostics:
                return ReconstructionResult(mol, "standard", 1.0, warnings)
            return mol
    except Exception as e:
        warnings.append(f"Standard failed: {type(e).__name__}")

    # Strategy 2: Without kekulize
    try:
        mol, _ = nx_to_mol(nx_graph, dataset=dataset, infer_bonds=True, sanitize=True, kekulize=False)
        if mol is not None and is_valid_molecule(mol):
            if return_diagnostics:
                return ReconstructionResult(mol, "kekulized", 0.95, warnings)
            return mol
    except Exception as e:
        warnings.append(f"Kekulized failed: {type(e).__name__}")

    # Strategy 3: Single bonds only
    try:
        mol, _ = nx_to_mol(nx_graph, dataset=dataset, infer_bonds=False, sanitize=True, kekulize=False)
        if mol is not None and is_valid_molecule(mol):
            warnings.append("Used single bonds only")
            if return_diagnostics:
                return ReconstructionResult(mol, "single_bonds", 0.7, warnings)
            return mol
    except Exception as e:
        warnings.append(f"Single bonds failed: {type(e).__name__}")

    # Strategy 4: Partial sanitize
    try:
        mol, _ = nx_to_mol(nx_graph, dataset=dataset, infer_bonds=True, sanitize=False, kekulize=False)
        if mol is not None:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=(
                    SanitizeFlags.SANITIZE_CLEANUP
                    | SanitizeFlags.SANITIZE_SYMMRINGS
                    | SanitizeFlags.SANITIZE_SETAROMATICITY
                ),
            )
            Chem.SanitizeMol(mol)
            if is_valid_molecule(mol):
                warnings.append("Used partial sanitization")
                if return_diagnostics:
                    return ReconstructionResult(mol, "partial_sanitize", 0.6, warnings)
                return mol
    except Exception as e:
        warnings.append(f"Partial sanitize failed: {type(e).__name__}")

    raise ValueError(
        f"All reconstruction strategies failed. "
        f"Nodes: {nx_graph.number_of_nodes()}, Edges: {nx_graph.number_of_edges()}. "
        f"Warnings: {warnings}"
    )


def mol_to_data(mol: Chem.Mol, dataset: str = "qm9") -> Data:
    """Convert RDKit molecule to PyG Data object."""
    atom_to_idx = QM9_ATOM_TO_IDX if dataset == "qm9" else ZINC_ATOM_TO_IDX

    x = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in atom_to_idx:
            raise ValueError(f"Unknown atom symbol: {symbol}")

        atom_type = atom_to_idx[symbol]
        degree = max(0, atom.GetDegree() - 1)
        charge_idx = {0: 0, 1: 1, -1: 2}.get(atom.GetFormalCharge(), 0)
        explicit_hs = atom.GetTotalNumHs()

        x.append([atom_type, degree, charge_idx, explicit_hs])

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
