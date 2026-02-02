#!/usr/bin/env python
"""
Test FlowEdgeDecoder - Evaluate trained models on custom SMILES.

This experiment loads pre-trained HDC encoder and FlowEdgeDecoder models,
then evaluates them on a list of SMILES strings (from CSV or direct input).

For each molecule:
1. Parse SMILES to RDKit molecule
2. Encode with HyperNet (concatenated order-0 and order-N embeddings)
3. Decode nodes from order-0 embedding
4. Generate edges with FlowEdgeDecoder
5. Compare generated molecule with original

Outputs:
- Individual side-by-side plots (original vs generated)
- Summary bar chart (valid/match/invalid counts)
- Timing statistics

Usage:
    # Test with SMILES list
    python test_flow_edge_decoder.py \
        --HDC_ENCODER_PATH /path/to/encoder.ckpt \
        --FLOW_DECODER_PATH /path/to/decoder.ckpt \
        --DATASET qm9

    # Test with CSV file
    python test_flow_edge_decoder.py \
        --HDC_ENCODER_PATH /path/to/encoder.ckpt \
        --FLOW_DECODER_PATH /path/to/decoder.ckpt \
        --SMILES_CSV_PATH /path/to/molecules.csv \
        --DATASET qm9

    # Test on CPU (useful when GPU memory is limited)
    python test_flow_edge_decoder.py \
        --HDC_ENCODER_PATH /path/to/encoder.ckpt \
        --FLOW_DECODER_PATH /path/to/decoder.ckpt \
        --DATASET qm9 \
        --DEVICE cpu

    # Quick test
    python test_flow_edge_decoder.py --__TESTING__ True
"""

from __future__ import annotations

import io
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import imageio
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
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
HDC_ENCODER_PATH: str = "/media/ssd2/Programming/graph_hdc_public/experiments/decoding/results/train_flow_edge_decoder_streaming/debug/hypernet_encoder.ckpt"  # Path to saved HyperNet encoder checkpoint
FLOW_DECODER_PATH: str = "/media/ssd2/Programming/graph_hdc_public/experiments/decoding/results/train_flow_edge_decoder_streaming/debug/last.ckpt"  # Path to saved FlowEdgeDecoder checkpoint

# Dataset type (ZINC only supported)
DATASET: str = "zinc"

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

# Sampling configuration
SAMPLE_STEPS: int = 1000
ETA: float = 0.0
OMEGA: float = 0.0
SAMPLE_TIME_DISTORTION: str = "polydec"
NOISE_TYPE_OVERRIDE: Optional[str] = None  # Override noise type: "uniform", "marginal", or None (use trained)
DETERMINISTIC: bool = False  # Use argmax instead of sampling for deterministic trajectories

# GIF Animation Configuration
GENERATE_GIF: bool = True                       # Enable/disable GIF generation
GIF_FRAME_INTERVAL: int = 10                    # Capture frame every N steps
GIF_FPS: int = 10                               # Frames per second (100ms per frame)
GIF_IMAGE_SIZE: Tuple[int, int] = (400, 400)    # Size of molecule rendering

# System configuration
SEED: int = 42
DEVICE: str = "auto"  # "auto", "cpu", or "cuda" - device for inference

# Debug/Testing modes
__DEBUG__: bool = True
__TESTING__: bool = False


# =============================================================================
# Helper Functions
# =============================================================================


def load_smiles_from_csv(csv_path: str) -> list[str]:
    """
    Load SMILES from a CSV file.

    Args:
        csv_path: Path to CSV file with "smiles" column

    Returns:
        List of SMILES strings
    """
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

    The hdc_vector is a concatenation of [order_0 | order_N], where:
    - order_0: Bundled node hypervectors (first base_hdc_dim dimensions)
    - order_N: Graph embedding after message passing (last base_hdc_dim dimensions)

    Node decoding uses the order_0 part.

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

    # Decode node counts from order-0 embedding using iterative unbinding
    node_counter_dict = hypernet.decode_order_zero_counter_iterative(order_zero)
    node_counter = node_counter_dict.get(0, Counter())

    # Convert to list of node tuples
    node_tuples = []
    for node_tuple, count in node_counter.items():
        node_tuples.extend([node_tuple] * count)

    return node_tuples, len(node_tuples)


def pyg_to_mol(data: Data) -> Optional[Chem.Mol]:
    """
    Convert FlowEdgeDecoder output (PyG Data) to RDKit Mol.

    Expects 24-dim one-hot node features where first 9 dims are atom type.

    Args:
        data: PyG Data from FlowEdgeDecoder.sample()

    Returns:
        RDKit Mol or None if conversion fails
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


# =============================================================================
# GIF Animation Functions
# =============================================================================


def dense_E_to_pyg_data(
    X: torch.Tensor,
    E: torch.Tensor,
    node_mask: torch.Tensor,
    sample_idx: int = 0,
) -> Data:
    """
    Convert dense edge tensor to PyG Data object for visualization.

    Args:
        X: Node features (bs, n, dx) - one-hot encoded
        E: Edge features (bs, n, n, de) - one-hot encoded
        node_mask: Valid node mask (bs, n)
        sample_idx: Which sample in the batch to extract (default: 0)

    Returns:
        PyG Data object with x, edge_index, edge_attr
    """
    # Extract single sample
    x = X[sample_idx]  # (n, dx)
    e = E[sample_idx]  # (n, n, de)
    mask = node_mask[sample_idx]  # (n,)

    # Get number of valid nodes
    n_valid = mask.sum().item()
    x_valid = x[:n_valid]  # (n_valid, dx)
    e_valid = e[:n_valid, :n_valid]  # (n_valid, n_valid, de)

    # Convert edges to sparse format
    # Get argmax for edge types
    e_labels = torch.argmax(e_valid, dim=-1)  # (n_valid, n_valid)

    # Build edge_index and edge_attr for non-zero edges
    edge_src = []
    edge_dst = []
    edge_types = []

    for i in range(n_valid):
        for j in range(i + 1, n_valid):  # Upper triangle only (symmetric)
            edge_type = e_labels[i, j].item()
            if edge_type > 0:  # Non-zero edge (has a bond)
                edge_src.extend([i, j])
                edge_dst.extend([j, i])
                edge_types.extend([edge_type, edge_type])

    if len(edge_src) > 0:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros(0, dtype=torch.long)

    return Data(x=x_valid, edge_index=edge_index, edge_attr=edge_attr)


def render_frame(
    mol: Optional[Chem.Mol],
    t_value: float,
    smiles: Optional[str],
    image_size: Tuple[int, int] = (400, 400),
) -> Image.Image:
    """
    Render a single frame for the GIF animation.

    Args:
        mol: RDKit molecule (can be None or invalid)
        t_value: Current time step value (0.0 to 1.0)
        smiles: SMILES string (can be None)
        image_size: Size of the output image

    Returns:
        PIL Image with the rendered molecule and annotations (fixed size)
    """
    # Fixed output size (slightly larger to accommodate title and SMILES)
    output_width = image_size[0]
    output_height = image_size[1] + 60  # Extra space for title and SMILES

    # Create figure with fixed size - do NOT use tight_layout or bbox_inches="tight"
    fig, ax = plt.subplots(figsize=(output_width / 100, output_height / 100), dpi=100)

    # Try to render the molecule
    if mol is not None:
        try:
            img = Draw.MolToImage(mol, size=image_size)
            ax.imshow(img)
        except Exception:
            ax.text(
                0.5, 0.5, "Invalid\nMolecule",
                ha="center", va="center", fontsize=16, transform=ax.transAxes
            )
    else:
        ax.text(
            0.5, 0.5, "No Molecule",
            ha="center", va="center", fontsize=16, transform=ax.transAxes
        )

    ax.axis("off")

    # Add annotations
    title = f"t = {t_value:.3f}"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # Add SMILES at bottom (truncated if too long)
    if smiles:
        smiles_display = smiles[:40] + "..." if len(smiles) > 40 else smiles
    else:
        smiles_display = "N/A"

    fig.text(0.5, 0.02, smiles_display, ha="center", fontsize=8, family="monospace")

    # Adjust subplot to leave room for title and SMILES (no tight_layout)
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.02, right=0.98)

    # Convert matplotlib figure to PIL Image with FIXED size (no bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white", edgecolor="none")
    buf.seek(0)
    pil_image = Image.open(buf).copy()
    buf.close()
    plt.close(fig)

    # Ensure exact output size by resizing if necessary
    if pil_image.size != (output_width, output_height):
        pil_image = pil_image.resize((output_width, output_height), Image.Resampling.LANCZOS)

    return pil_image


def create_reconstruction_gif(
    frames: List[Image.Image],
    save_path: Path,
    fps: int = 10,
) -> None:
    """
    Create animated GIF from list of PIL Images.

    Args:
        frames: List of PIL Image frames
        save_path: Output path for the GIF
        fps: Frames per second
    """
    if len(frames) == 0:
        return

    # Calculate duration in seconds
    duration = 1.0 / fps

    # Convert PIL Images to numpy arrays
    frame_arrays = [np.array(frame) for frame in frames]

    imageio.mimsave(
        save_path,
        frame_arrays,
        format="GIF",
        duration=duration,
        loop=0,  # Loop forever
    )


class FrameCollector:
    """
    Collects intermediate frames during sampling for GIF generation.

    This class is used as a callback during FlowEdgeDecoder.sample() to
    capture intermediate states at specified intervals.
    """

    def __init__(
        self,
        capture_interval: int,
        sample_steps: int,
        image_size: Tuple[int, int] = (400, 400),
    ):
        """
        Initialize frame collector.

        Args:
            capture_interval: Capture frame every N steps
            sample_steps: Total number of sampling steps
            image_size: Size for molecule rendering
        """
        self.capture_interval = capture_interval
        self.sample_steps = sample_steps
        self.image_size = image_size
        # (step, t_value, noisy_pyg_data, prediction_pyg_data or None)
        self.frames: List[Tuple[int, float, Data, Optional[Data]]] = []

    def __call__(
        self,
        step: int,
        t_value: float,
        X: torch.Tensor,
        E: torch.Tensor,
        node_mask: torch.Tensor,
        pred_E: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Callback invoked at each sampling step.

        Captures frame data if step matches the capture interval.

        Args:
            step: Current step index
            t_value: Current time value (0.0 to 1.0)
            X: Node features tensor
            E: Current noisy edge features tensor
            node_mask: Valid node mask
            pred_E: Model's predicted clean edge distribution (None at t=0)
        """
        # Capture at step 0 (initial), every capture_interval steps, and final step
        should_capture = (
            step == 0
            or step % self.capture_interval == 0
            or step == self.sample_steps
        )

        if should_capture:
            # Convert dense tensors to PyG Data (for single sample, idx=0)
            noisy_pyg_data = dense_E_to_pyg_data(X, E, node_mask, sample_idx=0)

            # Convert prediction if available
            pred_pyg_data = None
            if pred_E is not None:
                pred_pyg_data = dense_E_to_pyg_data(X, pred_E, node_mask, sample_idx=0)

            self.frames.append((step, t_value, noisy_pyg_data, pred_pyg_data))

    def render_gif(
        self,
        save_path: Path,
        fps: int = 10,
    ) -> None:
        """
        Render all captured noisy state frames to GIF.

        Args:
            save_path: Output path for GIF
            fps: Frames per second
        """
        rendered_frames = []

        for step, t_value, noisy_pyg_data, _ in self.frames:
            # Convert PyG Data to RDKit molecule
            mol = pyg_to_mol(noisy_pyg_data)

            # Get SMILES if valid
            smiles = get_canonical_smiles(mol)

            # Render frame
            frame_img = render_frame(mol, t_value, smiles, self.image_size)
            rendered_frames.append(frame_img)

        # Create GIF
        create_reconstruction_gif(rendered_frames, save_path, fps)

    def render_prediction_gif(
        self,
        save_path: Path,
        fps: int = 10,
    ) -> None:
        """
        Render all captured prediction frames to GIF.

        Shows the model's predicted final clean graph at each timestep.

        Args:
            save_path: Output path for GIF
            fps: Frames per second
        """
        rendered_frames = []

        for step, t_value, _, pred_pyg_data in self.frames:
            # Skip frames without prediction (t=0)
            if pred_pyg_data is None:
                continue

            # Convert PyG Data to RDKit molecule
            mol = pyg_to_mol(pred_pyg_data)

            # Get SMILES if valid
            smiles = get_canonical_smiles(mol)

            # Render frame
            frame_img = render_frame(mol, t_value, smiles, self.image_size)
            rendered_frames.append(frame_img)

        # Create GIF (only if we have frames)
        if rendered_frames:
            create_reconstruction_gif(rendered_frames, save_path, fps)

    def clear(self) -> None:
        """Clear collected frames."""
        self.frames = []


# =============================================================================
# Visualization Functions
# =============================================================================


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
    # Valid count from the caller includes matches, so we need to subtract
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
    ax.set_title(f"Reconstruction Results (n={total_count})", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.2 if max(counts) > 0 else 1)

    # Add grid
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
    """Test FlowEdgeDecoder on custom SMILES."""

    e.log("=" * 60)
    e.log("FlowEdgeDecoder Testing")
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
    e["config/sample_time_distortion"] = e.SAMPLE_TIME_DISTORTION
    e["config/noise_type_override"] = e.NOISE_TYPE_OVERRIDE
    e["config/deterministic"] = e.DETERMINISTIC
    e["config/generate_gif"] = e.GENERATE_GIF
    e["config/gif_frame_interval"] = e.GIF_FRAME_INTERVAL
    e["config/gif_fps"] = e.GIF_FPS
    e["config/gif_image_size"] = e.GIF_IMAGE_SIZE
    e["config/device_setting"] = e.DEVICE

    # Device setup
    if e.DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(e.DEVICE)
    e.log(f"Using device: {device}")
    e["config/device"] = str(device)

    # Load SMILES
    if e.SMILES_CSV_PATH:
        e.log(f"Loading SMILES from CSV: {e.SMILES_CSV_PATH}")
        smiles_list = load_smiles_from_csv(e.SMILES_CSV_PATH)
        e["config/smiles_source"] = "csv"
        e["config/smiles_csv_path"] = e.SMILES_CSV_PATH
    else:
        smiles_list = e.SMILES_LIST
        e["config/smiles_source"] = "list"

    e.log(f"Number of SMILES to test: {len(smiles_list)}")
    e["data/num_smiles"] = len(smiles_list)

    # Load models
    e.log("\nLoading models...")

    if e.__TESTING__:
        # Create dummy models for testing
        from collections import OrderedDict
        import math
        from graph_hdc.hypernet.configs import DSHDCConfig, FeatureConfig, Features, IndexRange
        from graph_hdc.hypernet.feature_encoders import CombinatoricIntegerEncoder
        from graph_hdc.hypernet.types import VSAModel

        e.log("TESTING MODE: Creating dummy models...")

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
            device=str(device),
            seed=42,
            normalize=True,
            dtype="float64",
            node_feature_configs=OrderedDict([(Features.NODE_FEATURES, node_feature_config)]),
        )
        hypernet = HyperNet(config)

        # Create minimal FlowEdgeDecoder (24-dim node features)
        decoder = FlowEdgeDecoder(
            num_node_classes=NODE_FEATURE_DIM,
            hdc_dim=512,  # 2x base dim for concatenated embeddings
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
        e.log(f"  FlowEdgeDecoder condition_dim: {decoder.condition_dim}")

    hypernet.to(device)
    hypernet.eval()
    decoder.to(device)
    decoder.eval()

    e["model/base_hdc_dim"] = base_hdc_dim
    e["model/concat_hdc_dim"] = 2 * base_hdc_dim

    # Create output directories
    plots_dir = Path(e.path) / "comparison_plots"
    plots_dir.mkdir(exist_ok=True)

    # Process each SMILES
    e.log("\n" + "=" * 60)
    e.log("Processing SMILES")
    e.log("=" * 60)

    results = []
    valid_count = 0
    match_count = 0
    invalid_count = 0
    skipped_count = 0

    # Start timing
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
            e.log("  WARNING: Failed to convert to PyG Data (unsupported atoms?), skipping...")
            skipped_count += 1
            results.append({
                "idx": idx,
                "original_smiles": smiles,
                "generated_smiles": None,
                "status": "skipped",
                "error": "Unsupported atom types",
            })
            continue

        # Add batch attribute
        data = data.to(device)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        # Encode with HyperNet - compute concatenated [order_0 | order_N]
        with torch.no_grad():
            # Encode node properties
            data = hypernet.encode_properties(data)

            # Order-0: Bundle node hypervectors (no message passing)
            order_zero = scatter_hd(src=data.node_hv, index=data.batch, op="bundle")

            # Order-N: Full graph embedding with message passing
            encoder_output = hypernet.forward(data, normalize=True)
            order_n = encoder_output["graph_embedding"]

            # Concatenate [order_0 | order_N]
            hdc_vector = torch.cat([order_zero, order_n], dim=-1).squeeze(0)

        # Decode nodes from HDC embedding
        node_tuples, num_nodes = decode_nodes_from_hdc(
            hypernet, hdc_vector.unsqueeze(0), base_hdc_dim
        )

        e.log(f"  Decoded {num_nodes} nodes")

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

        # Prepare inputs for FlowEdgeDecoder.sample()
        # Convert node tuples to 24-dim one-hot
        node_features = node_tuples_to_onehot(node_tuples, device=device).unsqueeze(0)

        node_mask = torch.ones(1, num_nodes, dtype=torch.bool, device=device)
        hdc_vectors = hdc_vector.unsqueeze(0)

        # Create frame collector if GIF generation is enabled
        frame_collector = None
        if e.GENERATE_GIF:
            frame_collector = FrameCollector(
                capture_interval=e.GIF_FRAME_INTERVAL,
                sample_steps=e.SAMPLE_STEPS,
                image_size=e.GIF_IMAGE_SIZE,
            )

        # Generate edges
        with torch.no_grad():
            generated_samples = decoder.sample(
                hdc_vectors=hdc_vectors,
                node_features=node_features,
                node_mask=node_mask,
                sample_steps=e.SAMPLE_STEPS,
                eta=e.ETA,
                omega=e.OMEGA,
                time_distortion=e.SAMPLE_TIME_DISTORTION,
                noise_type_override=e.NOISE_TYPE_OVERRIDE,
                show_progress=False,
                step_callback=frame_collector,
                deterministic=e.DETERMINISTIC,
            )

        # Convert to RDKit molecule
        generated_data = generated_samples[0]
        generated_mol = pyg_to_mol(generated_data)
        generated_smiles = get_canonical_smiles(generated_mol)

        e.log(f"  Generated: {generated_smiles or 'N/A'}")

        # Check validity and match
        # For comparison, remove all Hs from original and recanonicalize
        is_valid = is_valid_mol(generated_mol)
        original_canonical = None
        if original_mol is not None:
            try:
                original_mol_no_h = Chem.RemoveAllHs(original_mol)
                original_canonical = Chem.MolToSmiles(original_mol_no_h, canonical=True)
            except Exception:
                original_canonical = get_canonical_smiles(original_mol)

        is_match = (
            is_valid
            and generated_smiles is not None
            and original_canonical is not None
            and generated_smiles == original_canonical
        )

        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
        if is_match:
            match_count += 1

        status = "match" if is_match else ("valid" if is_valid else "invalid")
        e.log(f"  Status: {status.upper()}")

        # Create comparison plot
        plot_path = plots_dir / f"comparison_{idx + 1:04d}.png"
        create_comparison_plot(
            original_mol=original_mol,
            generated_mol=generated_mol,
            original_smiles=smiles,
            generated_smiles=generated_smiles or "N/A",
            is_valid=is_valid,
            is_match=is_match,
            sample_idx=idx,
            save_path=plot_path,
        )

        # Generate GIF animations if enabled
        if e.GENERATE_GIF and frame_collector is not None:
            # GIF showing noisy states progression
            gif_path = plots_dir / f"reconstruction_{idx + 1:04d}_animation.gif"
            frame_collector.render_gif(gif_path, fps=e.GIF_FPS)
            e.log(f"  GIF saved: {gif_path.name}")

            # GIF showing model's predictions at each timestep
            pred_gif_path = plots_dir / f"reconstruction_{idx + 1:04d}_prediction.gif"
            frame_collector.render_prediction_gif(pred_gif_path, fps=e.GIF_FPS)
            e.log(f"  Prediction GIF saved: {pred_gif_path.name}")

        results.append({
            "idx": idx,
            "original_smiles": smiles,
            "original_canonical": original_canonical,
            "generated_smiles": generated_smiles,
            "status": status,
            "is_valid": is_valid,
            "is_match": is_match,
        })

    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    num_processed = len(smiles_list) - skipped_count

    # Create summary bar chart
    summary_plot_path = Path(e.path) / "summary_bar_chart.png"
    create_summary_bar_chart(
        match_count=match_count,
        valid_count=valid_count,
        invalid_count=invalid_count,
        total_count=num_processed,
        save_path=summary_plot_path,
    )

    # Log summary
    e.log("\n" + "=" * 60)
    e.log("SUMMARY")
    e.log("=" * 60)
    e.log(f"Total SMILES: {len(smiles_list)}")
    e.log(f"Skipped: {skipped_count}")
    e.log(f"Processed: {num_processed}")
    e.log(f"Valid molecules: {valid_count} ({100 * valid_count / num_processed:.1f}%)" if num_processed > 0 else "Valid molecules: 0")
    e.log(f"Exact matches: {match_count} ({100 * match_count / num_processed:.1f}%)" if num_processed > 0 else "Exact matches: 0")
    e.log(f"Invalid: {invalid_count} ({100 * invalid_count / num_processed:.1f}%)" if num_processed > 0 else "Invalid: 0")
    e.log("-" * 40)
    e.log(f"Total sampling time: {total_time:.2f} seconds")
    e.log(f"Average time per sample: {total_time / num_processed:.2f} seconds" if num_processed > 0 else "Average time per sample: N/A")
    e.log("=" * 60)

    # Store results
    e["results/total_smiles"] = len(smiles_list)
    e["results/skipped"] = skipped_count
    e["results/processed"] = num_processed
    e["results/valid_count"] = valid_count
    e["results/match_count"] = match_count
    e["results/invalid_count"] = invalid_count
    e["results/valid_rate"] = valid_count / num_processed if num_processed > 0 else 0
    e["results/match_rate"] = match_count / num_processed if num_processed > 0 else 0
    e["results/total_sampling_time_seconds"] = total_time
    e["results/avg_time_per_sample_seconds"] = total_time / num_processed if num_processed > 0 else 0
    e["results/details"] = results

    # Save results as JSON
    e.commit_json("test_results.json", {
        "config": {
            "hdc_encoder_path": e.HDC_ENCODER_PATH,
            "flow_decoder_path": e.FLOW_DECODER_PATH,
            "dataset": e.DATASET,
            "sample_steps": e.SAMPLE_STEPS,
            "eta": e.ETA,
            "omega": e.OMEGA,
            "noise_type_override": e.NOISE_TYPE_OVERRIDE,
            "deterministic": e.DETERMINISTIC,
        },
        "summary": {
            "total_smiles": len(smiles_list),
            "skipped": skipped_count,
            "processed": num_processed,
            "valid_count": valid_count,
            "match_count": match_count,
            "invalid_count": invalid_count,
            "valid_rate": valid_count / num_processed if num_processed > 0 else 0,
            "match_rate": match_count / num_processed if num_processed > 0 else 0,
            "total_sampling_time_seconds": total_time,
        },
        "results": results,
    })

    e.log("\nExperiment completed!")
    e.log(f"Comparison plots saved to: {plots_dir}")
    e.log(f"Summary chart saved to: {summary_plot_path}")


# =============================================================================
# Testing Mode
# =============================================================================


@experiment.testing
def testing(e: Experiment) -> None:
    """Quick test mode with reduced parameters."""
    e.SAMPLE_STEPS = 10
    e.SMILES_LIST = ["CCO", "CC=O"]  # Just 2 simple molecules
    e.DATASET = "zinc"  # Match the dummy HyperNet config
    e.GENERATE_GIF = True
    e.GIF_FRAME_INTERVAL = 2  # Capture more frames in test mode (every 2 steps)


# =============================================================================
# Entry Point
# =============================================================================

experiment.run_if_main()
