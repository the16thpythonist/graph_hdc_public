#!/usr/bin/env python
"""
Test FlowEdgeDecoder with HDC Distance Tracking and Early Stopping.

This experiment tests the sample_with_hdc() method which:
1. Performs standard sampling (no R^HDC guidance term)
2. At each timestep, computes deterministic argmax edges and encodes with HyperNet
3. Tracks the best match (lowest cosine distance) per sample
4. Early stops when all samples have distance below threshold
5. Returns the best-seen edges (not final edges)

This is useful for testing reconstruction quality - if the model can find
edge configurations that closely match the target HDC encoding.

Usage:
    # Test with SMILES list
    python test_flow_edge_decoder_hdc.py \
        --HDC_ENCODER_PATH /path/to/encoder.ckpt \
        --FLOW_DECODER_PATH /path/to/decoder.ckpt \
        --DATASET qm9 \
        --DISTANCE_THRESHOLD 0.001

    # Quick test
    python test_flow_edge_decoder_hdc.py --__TESTING__ True
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Data

from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.models.flow_edge_decoder import (
    ZINC_ATOM_TYPES,
    ZINC_IDX_TO_ATOM,
    NODE_FEATURE_DIM,
    NUM_EDGE_CLASSES,
    FlowEdgeDecoder,
    node_tuples_to_onehot,
)
from graph_hdc.utils.helpers import scatter_hd

# =============================================================================
# PARAMETERS
# =============================================================================

# Model paths (required)
HDC_ENCODER_PATH: str = "/media/ssd2/Programming/graph_hdc_public/experiments/decoding/results/train_flow_edge_decoder_streaming/debug/hypernet_encoder.ckpt"
FLOW_DECODER_PATH: str = "/media/ssd2/Programming/graph_hdc_public/experiments/decoding/results/train_flow_edge_decoder_streaming/debug/last.ckpt"

# Dataset type for atom mapping (required)
DATASET: str = "zinc"  # "qm9" or "zinc" - determines atom type mapping

# Input SMILES - either provide CSV path or direct list
SMILES_CSV_PATH: str = ""  # Path to CSV file with "smiles" column
SMILES_LIST: list[str] = [
    "CCO",      # Ethanol
    "CC(=O)O",  # Acetic acid
    "c1ccccc1", # Benzene
    "CCN",      # Ethylamine
    "CC=O",     # Acetaldehyde
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cn1cnc2N(C)C(=O)N(C)C(=O)c12",
    "CN1C(=O)C=C(Cc2ccc(F)c(F)c2)N(C)C1=O",
    "CN1C(=O)C=C(Cc2cc(F)ccc2F)N(C)C1=O",
    "CN1CC=C(Cc2cc(F)c(=O)cc2F)C(N)C1=O",
]

# Standard Sampling configuration
SAMPLE_STEPS: int = 1000
ETA: float = 1.5
OMEGA: float = 0.0
SAMPLE_TIME_DISTORTION: str = "polydec"
NOISE_TYPE_OVERRIDE: Optional[str] = None

# HDC Early Stopping configuration
DISTANCE_THRESHOLD: float = 0.00001  # Cosine distance threshold for early stopping

# System configuration
SEED: int = 42

# Debug/Testing modes
__DEBUG__: bool = True
__TESTING__: bool = False


# =============================================================================
# Helper Functions
# =============================================================================


def load_smiles_from_csv(csv_path: str) -> list[str]:
    """Load SMILES from a CSV file."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "smiles" not in df.columns:
        raise ValueError(f"CSV file must have 'smiles' column. Found: {list(df.columns)}")

    return df["smiles"].dropna().tolist()


def smiles_to_pyg_data(smiles: str, dataset: str = "zinc") -> Optional[Data]:
    """
    Convert SMILES to PyG Data object with ZINC node features.

    Args:
        smiles: SMILES string
        dataset: Dataset name (only "zinc" supported)

    Returns:
        PyG Data object or None if conversion fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    from graph_hdc.datasets.zinc_smiles import ZINC_ATOM_TO_IDX
    atom_to_idx = ZINC_ATOM_TO_IDX

    # Build node features (5-dim: atom_type, degree, charge, Hs, ring)
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
        is_in_ring = int(atom.IsInRing())
        x.append([float(atom_type), float(degree), float(charge_idx), float(explicit_hs), float(is_in_ring)])

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


def decode_nodes_from_hdc(
    hypernet: HyperNet,
    hdc_vector: torch.Tensor,
    base_hdc_dim: int,
) -> tuple[list[tuple[int, ...]], int]:
    """
    Decode node tuples from HDC embedding.

    Args:
        hypernet: HyperNet encoder instance
        hdc_vector: Concatenated HDC vector (2 * base_hdc_dim,) or (batch, 2 * base_hdc_dim)
        base_hdc_dim: Base hypervector dimension

    Returns:
        Tuple of (list of node tuples (atom, degree, charge, Hs, ring), number of nodes)
    """
    # Extract order_0 from the first half
    if hdc_vector.dim() == 1:
        order_zero = hdc_vector[:base_hdc_dim].unsqueeze(0)
    else:
        order_zero = hdc_vector[:, :base_hdc_dim]

    # Move to same device as HyperNet's codebook (typically CPU)
    codebook_device = hypernet.nodes_codebook.device
    if order_zero.device != codebook_device:
        order_zero = order_zero.to(codebook_device)

    # Decode node counts from order-0 embedding using iterative unbinding
    node_counter_dict = hypernet.decode_order_zero_counter_iterative(order_zero)
    node_counter = node_counter_dict.get(0, Counter())

    # Convert to list of node tuples
    node_tuples = []
    for node_tuple, count in node_counter.items():
        node_tuples.extend([node_tuple] * count)

    return node_tuples, len(node_tuples)


def pyg_to_mol(data: Data) -> Optional[Chem.Mol]:
    """Convert FlowEdgeDecoder output (PyG Data) to RDKit Mol.

    Expects 24-dim one-hot node features where first 9 dims are atom type.
    """
    try:
        mol = Chem.RWMol()

        # Get atom types from first 9 dimensions of 24-dim one-hot
        if data.x.dim() > 1:
            # Extract atom type from first 9 dims (ZINC atom classes)
            atom_type_probs = data.x[:, :9]
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
            mol = mol.GetMol()
            return mol

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

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass

        return mol

    except Exception:
        return None


def is_valid_mol(mol: Optional[Chem.Mol]) -> bool:
    """Check if molecule is valid."""
    if mol is None:
        return False
    try:
        smiles = Chem.MolToSmiles(mol)
        return smiles is not None and len(smiles) > 0
    except Exception:
        return False


def get_canonical_smiles(mol: Optional[Chem.Mol]) -> Optional[str]:
    """Get canonical SMILES from molecule."""
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def create_comparison_plot(
    original_mol: Chem.Mol,
    generated_mol: Optional[Chem.Mol],
    original_smiles: str,
    generated_smiles: Optional[str],
    is_valid: bool,
    is_match: bool,
    sample_idx: int,
    save_path: Path,
) -> None:
    """Create side-by-side plot of original vs generated molecule."""
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


def create_summary_bar_chart(
    match_count: int,
    valid_count: int,
    invalid_count: int,
    total_count: int,
    save_path: Path,
) -> None:
    """Create summary bar chart showing match/valid/invalid counts."""
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ["Match", "Valid (no match)", "Invalid"]
    valid_no_match = valid_count - match_count
    counts = [match_count, valid_no_match, invalid_count]
    colors = ["green", "orange", "red"]

    bars = ax.bar(categories, counts, color=colors, edgecolor="black", linewidth=1.2)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percentage = 100 * count / total_count if total_count > 0 else 0
        ax.annotate(
            f"{count}\n({percentage:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"HDC Early-Stop Reconstruction Results (n={total_count})", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.2 if max(counts) > 0 else 1)

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# EXPERIMENT
# =============================================================================


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    """Test FlowEdgeDecoder with HDC Distance Tracking and Early Stopping."""

    e.log("=" * 60)
    e.log("FlowEdgeDecoder HDC Sampling with Early Stopping")
    e.log("=" * 60)

    # Validate required parameters
    if not e.HDC_ENCODER_PATH and not e.__TESTING__:
        e.log("ERROR: HDC_ENCODER_PATH is required")
        return
    if not e.FLOW_DECODER_PATH and not e.__TESTING__:
        e.log("ERROR: FLOW_DECODER_PATH is required")
        return

    # Store configuration
    e["config/hdc_encoder_path"] = e.HDC_ENCODER_PATH
    e["config/flow_decoder_path"] = e.FLOW_DECODER_PATH
    e["config/dataset"] = e.DATASET
    e["config/sample_steps"] = e.SAMPLE_STEPS
    e["config/eta"] = e.ETA
    e["config/omega"] = e.OMEGA
    e["config/distance_threshold"] = e.DISTANCE_THRESHOLD
    e["config/sample_time_distortion"] = e.SAMPLE_TIME_DISTORTION
    e["config/noise_type_override"] = e.NOISE_TYPE_OVERRIDE

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e.log(f"Using device: {device}")
    e["config/device"] = str(device)

    # Load SMILES
    if e.SMILES_CSV_PATH:
        e.log(f"Loading SMILES from CSV: {e.SMILES_CSV_PATH}")
        smiles_list = load_smiles_from_csv(e.SMILES_CSV_PATH)
        e["config/smiles_source"] = "csv"
    else:
        smiles_list = e.SMILES_LIST
        e["config/smiles_source"] = "list"

    e.log(f"Number of SMILES to test: {len(smiles_list)}")
    e["data/num_smiles"] = len(smiles_list)

    # HDC Early Stopping parameters
    e.log(f"\nHDC Early Stopping Parameters:")
    e.log(f"  distance_threshold: {e.DISTANCE_THRESHOLD}")

    # Load models
    e.log("\nLoading models...")

    if e.__TESTING__:
        # Create dummy models for testing
        from collections import OrderedDict
        import math
        from graph_hdc.hypernet.configs import DSHDCConfig, FeatureConfig, Features, IndexRange
        from graph_hdc.hypernet.feature_encoders import CombinatoricIntegerEncoder
        from graph_hdc.hypernet.types import VSAModel

        e.log("TESTING MODE: Creating dummy models (using CPU)...")
        device = torch.device("cpu")  # Force CPU for testing

        # Create minimal HyperNet (ZINC config with 5 features)
        node_feature_config = FeatureConfig(
            count=math.prod([9, 6, 3, 4, 2]),  # ZINC: 9*6*3*4*2 = 1296
            encoder_cls=CombinatoricIntegerEncoder,
            index_range=IndexRange((0, 5)),
            bins=[9, 6, 3, 4, 2],
        )
        config = DSHDCConfig(
            name="TEST_ZINC",
            hv_dim=256,
            vsa=VSAModel.HRR,
            base_dataset="zinc",
            hypernet_depth=3,
            device="cpu",
            seed=42,
            normalize=True,
            dtype="float64",
            node_feature_configs=OrderedDict([(Features.NODE_FEATURES, node_feature_config)]),
        )
        hypernet = HyperNet(config)

        # Create minimal FlowEdgeDecoder (24-dim node features)
        decoder = FlowEdgeDecoder(
            num_node_classes=NODE_FEATURE_DIM,
            hdc_dim=512,
            condition_dim=64,
            n_layers=2,
            hidden_dim=64,
            hidden_mlp_dim=128,
        )
        base_hdc_dim = 256
    else:
        # Load HyperNet encoder
        e.log(f"Loading HyperNet from: {e.HDC_ENCODER_PATH}")
        hypernet = HyperNet.load(e.HDC_ENCODER_PATH, device=str(device))
        base_hdc_dim = hypernet.hv_dim
        e.log(f"  HyperNet hv_dim: {base_hdc_dim}")

        # Load FlowEdgeDecoder
        e.log(f"Loading FlowEdgeDecoder from: {e.FLOW_DECODER_PATH}")
        decoder = FlowEdgeDecoder.load(e.FLOW_DECODER_PATH, device=device)
        e.log(f"  FlowEdgeDecoder hdc_dim: {decoder.hdc_dim}")

    hypernet.to(device)
    hypernet.eval()
    decoder.to(device)
    decoder.eval()

    e["model/base_hdc_dim"] = base_hdc_dim

    # Create output directories
    plots_dir = Path(e.path) / "comparison_plots"
    plots_dir.mkdir(exist_ok=True)

    # Process each SMILES
    e.log("\n" + "=" * 60)
    e.log("Processing SMILES with HDC Early Stopping")
    e.log("=" * 60)

    results = []
    valid_count = 0
    match_count = 0
    invalid_count = 0
    skipped_count = 0
    early_stop_count = 0

    start_time = time.time()

    for idx, smiles in enumerate(smiles_list):
        e.log(f"\nSample {idx + 1}/{len(smiles_list)}: {smiles}")

        # Parse SMILES to molecule
        original_mol = Chem.MolFromSmiles(smiles)
        if original_mol is None:
            e.log("  WARNING: Failed to parse SMILES, skipping...")
            skipped_count += 1
            results.append({
                "idx": idx,
                "original_smiles": smiles,
                "generated_smiles": None,
                "status": "skipped",
                "error": "Failed to parse SMILES",
            })
            continue

        # Convert to PyG Data
        data = smiles_to_pyg_data(smiles, e.DATASET)
        if data is None:
            e.log("  WARNING: Failed to convert to PyG Data, skipping...")
            skipped_count += 1
            results.append({
                "idx": idx,
                "original_smiles": smiles,
                "generated_smiles": None,
                "status": "skipped",
                "error": "Unsupported atom types",
            })
            continue

        # Check for molecules with no edges (single atoms)
        if data.edge_index.numel() == 0:
            e.log("  WARNING: Molecule has no edges, skipping...")
            skipped_count += 1
            results.append({
                "idx": idx,
                "original_smiles": smiles,
                "generated_smiles": None,
                "status": "skipped",
                "error": "No edges in molecule",
            })
            continue

        # Add batch attribute
        data = data.to(device)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        # Store original node features for HyperNet encoding
        original_node_features = data.x.clone()

        # Encode with HyperNet
        with torch.no_grad():
            data = hypernet.encode_properties(data)
            # Ensure node_hv is on the correct device
            if data.node_hv.device != device:
                data.node_hv = data.node_hv.to(device)
            order_zero = scatter_hd(src=data.node_hv, index=data.batch, op="bundle")
            encoder_output = hypernet.forward(data, normalize=True)
            order_n = encoder_output["graph_embedding"]
            hdc_vector = torch.cat([order_zero, order_n], dim=-1).squeeze(0)

        # Decode nodes from HDC embedding (returns node tuples)
        node_tuples, num_nodes = decode_nodes_from_hdc(
            hypernet, hdc_vector.unsqueeze(0), base_hdc_dim
        )

        original_num_nodes = original_node_features.shape[0]
        e.log(f"  Decoded {num_nodes} nodes (original: {original_num_nodes})")

        if num_nodes == 0:
            e.log("  WARNING: No nodes decoded, skipping...")
            invalid_count += 1
            results.append({
                "idx": idx,
                "original_smiles": smiles,
                "generated_smiles": None,
                "status": "invalid",
                "error": "No nodes decoded",
            })
            continue

        # Check if decoded node count matches original
        if num_nodes != original_num_nodes:
            e.log(f"  WARNING: Node count mismatch ({num_nodes} decoded vs {original_num_nodes} original), skipping...")
            skipped_count += 1
            results.append({
                "idx": idx,
                "original_smiles": smiles,
                "generated_smiles": None,
                "status": "skipped",
                "error": f"Node count mismatch: {num_nodes} decoded vs {original_num_nodes} original",
            })
            continue

        # Prepare inputs for sample_with_hdc
        # Convert node tuples to 24-dim one-hot encoding
        node_features = node_tuples_to_onehot(node_tuples, device=device).unsqueeze(0)

        node_mask = torch.ones(1, num_nodes, dtype=torch.bool, device=device)
        hdc_vectors = hdc_vector.unsqueeze(0)

        # Use the original node features for HyperNet encoding
        original_x_for_hypernet = original_node_features.unsqueeze(0)

        # Generate edges with HDC early stopping
        sample_start = time.time()
        with torch.no_grad():
            generated_samples = decoder.sample_with_hdc(
                hdc_vectors=hdc_vectors,
                node_features=node_features,
                node_mask=node_mask,
                original_node_features=original_x_for_hypernet,
                hypernet=hypernet,
                distance_threshold=e.DISTANCE_THRESHOLD,
                eta=e.ETA,
                omega=e.OMEGA,
                sample_steps=e.SAMPLE_STEPS,
                time_distortion=e.SAMPLE_TIME_DISTORTION,
                noise_type_override=e.NOISE_TYPE_OVERRIDE,
                show_progress=True,
            )
        sample_time = time.time() - sample_start

        # Convert to RDKit molecule
        generated_data = generated_samples[0]
        generated_mol = pyg_to_mol(generated_data)
        generated_smiles = get_canonical_smiles(generated_mol)

        e.log(f"  Generated: {generated_smiles or 'N/A'} (took {sample_time:.2f}s)")

        # Check validity and match
        original_canonical = Chem.MolToSmiles(Chem.RemoveHs(original_mol), canonical=True)
        is_valid = is_valid_mol(generated_mol)
        is_match = is_valid and generated_smiles == original_canonical

        if is_match:
            e.log("  Status: MATCH")
            match_count += 1
            valid_count += 1
        elif is_valid:
            e.log("  Status: Valid (no match)")
            valid_count += 1
        else:
            e.log("  Status: Invalid")
            invalid_count += 1

        # Store result
        results.append({
            "idx": idx,
            "original_smiles": smiles,
            "original_canonical": original_canonical,
            "generated_smiles": generated_smiles,
            "status": "match" if is_match else ("valid" if is_valid else "invalid"),
            "sample_time": sample_time,
        })

        # Create comparison plot
        plot_path = plots_dir / f"sample_{idx:04d}.png"
        create_comparison_plot(
            original_mol, generated_mol,
            smiles, generated_smiles,
            is_valid, is_match, idx, plot_path
        )

    # Timing
    total_time = time.time() - start_time
    processed = len(smiles_list) - skipped_count

    # Summary
    e.log("\n" + "=" * 60)
    e.log("SUMMARY")
    e.log("=" * 60)
    e.log(f"Total SMILES: {len(smiles_list)}")
    e.log(f"Processed: {processed}")
    e.log(f"Skipped: {skipped_count}")
    e.log(f"Valid: {valid_count} ({100 * valid_count / processed:.1f}%)" if processed > 0 else "Valid: 0")
    e.log(f"Match: {match_count} ({100 * match_count / processed:.1f}%)" if processed > 0 else "Match: 0")
    e.log(f"Invalid: {invalid_count} ({100 * invalid_count / processed:.1f}%)" if processed > 0 else "Invalid: 0")
    e.log(f"Total time: {total_time:.2f}s")
    e.log(f"Avg time per sample: {total_time / processed:.2f}s" if processed > 0 else "")

    # Store summary
    e["results/total"] = len(smiles_list)
    e["results/processed"] = processed
    e["results/skipped"] = skipped_count
    e["results/valid"] = valid_count
    e["results/match"] = match_count
    e["results/invalid"] = invalid_count
    e["results/total_time"] = total_time
    e["results/details"] = results

    # Create summary chart
    if processed > 0:
        chart_path = Path(e.path) / "summary_chart.png"
        create_summary_bar_chart(match_count, valid_count, invalid_count, processed, chart_path)
        e.log(f"\nSummary chart saved to: {chart_path}")

    e.log("\nExperiment complete!")


# =============================================================================
# Testing Mode
# =============================================================================


@experiment.testing
def testing(e: Experiment) -> None:
    """Quick test mode with reduced parameters."""
    e.SAMPLE_STEPS = 10
    e.SMILES_LIST = ["CCO", "CC=O"]  # Just 2 simple molecules
    e.DATASET = "zinc"  # Match the dummy HyperNet config
    e.DISTANCE_THRESHOLD = 0.5  # Higher threshold for faster testing


# =============================================================================
# Entry Point
# =============================================================================

experiment.run_if_main()
