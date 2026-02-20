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

This is the BASE TEST EXPERIMENT. Child experiments can inherit via
Experiment.extend() and override hooks for custom sampling behavior
(e.g. HDC-guided sampling, HDC early stopping).

Usage:
    # Test with SMILES list
    python test_flow_edge_decoder.py \\
        --HDC_ENCODER_PATH /path/to/encoder.ckpt \\
        --FLOW_DECODER_PATH /path/to/decoder.ckpt \\
        --DATASET qm9

    # Test with CSV file
    python test_flow_edge_decoder.py \\
        --HDC_ENCODER_PATH /path/to/encoder.ckpt \\
        --FLOW_DECODER_PATH /path/to/decoder.ckpt \\
        --SMILES_CSV_PATH /path/to/molecules.csv \\
        --DATASET qm9

    # Test on CPU (useful when GPU memory is limited)
    python test_flow_edge_decoder.py \\
        --HDC_ENCODER_PATH /path/to/encoder.ckpt \\
        --FLOW_DECODER_PATH /path/to/decoder.ckpt \\
        --DATASET qm9 \\
        --DEVICE cpu

    # Quick test
    python test_flow_edge_decoder.py --__TESTING__ True
"""

from __future__ import annotations

import gc
import io
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import imageio
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import torch
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Data

from graph_hdc.hypernet import load_hypernet
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.models.flow_edge_decoder import (
    FlowEdgeDecoder,
    get_node_feature_bins,
    node_tuples_to_onehot,
)
from graph_hdc.utils.experiment_helpers import (
    compute_hdc_distance,
    create_accuracy_by_size_chart,
    create_reconstruction_plot,
    create_summary_bar_chart,
    create_test_dummy_models,
    decode_nodes_from_hdc,
    get_canonical_smiles,
    is_valid_mol,
    load_smiles_from_csv,
    pyg_to_mol,
    smiles_to_pyg_data,
)


# =============================================================================
# PARAMETERS
# =============================================================================

# -----------------------------------------------------------------------------
# Model Paths
# -----------------------------------------------------------------------------

# :param HDC_ENCODER_PATH:
#     Path to saved HyperNet encoder checkpoint (.ckpt). Required unless
#     running in __TESTING__ mode.
HDC_ENCODER_PATH: str = "/media/ssd2/Programming/graph_hdc_public/experiments/decoding/results/train_flow_edge_decoder_streaming/GOOD/hypernet_encoder.ckpt"

# :param FLOW_DECODER_PATH:
#     Path to saved FlowEdgeDecoder checkpoint (.ckpt). Required unless
#     running in __TESTING__ mode.
FLOW_DECODER_PATH: str = "/media/ssd2/Programming/graph_hdc_public/experiments/decoding/results/train_flow_edge_decoder_streaming/GOOD/last.ckpt"

# -----------------------------------------------------------------------------
# Dataset Configuration
# -----------------------------------------------------------------------------

# :param DATASET:
#     Dataset type for atom feature encoding. Determines node feature
#     dimensions and atom type mapping. Options: "zinc", "qm9".
DATASET: str = "zinc"

# -----------------------------------------------------------------------------
# Input SMILES
# -----------------------------------------------------------------------------

# :param SMILES_CSV_PATH:
#     Path to CSV file with a "smiles" column. If empty, uses SMILES_LIST
#     instead.
SMILES_CSV_PATH: str = ""

# :param SMILES_LIST:
#     List of SMILES strings to test. Used when SMILES_CSV_PATH is empty.
#     Default: 100 diverse SMILES from ZINC (MaxMin on Morgan FP r=2, 2048 bits),
#     cleaned of charges, stereochemistry, and explicit hydrogens.
SMILES_LIST: list[str] = [
    "CCN(Cc1ccc(OC)c(OC)c1)C(=O)c2cscc2",
    "C(NCC1=CC=C(C)C=C1)1=NC=NC(N2C3CC(C)(C)CC(C)(C3)C2)=C1N",
    "C(=O)1N(C)CC(NC(C2SC=CC=2)C2CC2)C1",
    "C1C(C(C#N)N2C(C3=CC=CC=C3)CC(C)=N2)=COC=1",
    "FC1CCCCC1Br",
    "NCCCCNC(=S)S",
    "CCOCC(N)CSCC",
    "CC(Br)=CCCCBr",
    "C=CC(N)CCC(F)(F)F",
    "O=C(CO)C(O)C(O)CO",
    "Fc1cc(F)c(F)c(I)c1",
    "FC=CC1C=CC=CC1(F)F",
    "OC1COCOC1C1OCOCC1O",
    "CCC(C)C(C)n1nnnc1N",
    "CCCCC1C(C)C1(Br)Br",
    "N#CCCCCOC1CCCC(N)C1",
    "FOC(F)=C(F)C(F)(F)F",
    "NS(=O)(=O)c1cscc1Br",
    "Oc1ncnc2cnc(Cl)nc12",
    "CSc1nc(C)c2c(n1)SCC2",
    "CCCc1ccc(C=CCCNC)cc1",
    "OCC1OC(O)C(F)C(O)C1O",
    "CC1(c2ccc(Cl)nn2)CC1",
    "C=CCN(CC=C)S(C)(=O)=O",
    "CCCC(C)CC(NCC)C1CSCCS1",
    "CN(C)CCN1CCN(CCCCS)CC1",
    "Nc1cc2c3c(c1)CCCN3CCC2",
    "Nc1ccccc1SCCCOc1ccccc1",
    "C1=C(COCC2=CCCSC2)CSCC1",
    "CN(C)Cc1noc(C2CNCCO2)n1",
    "CC(C)(CCO)c1cc(Cl)sc1Cl",
    "CC1(C)SC(C)(C)SC(C)(C)S1",
    "N#CC1=C2C(=S)N=CN=C2N=N1",
    "CC(CCl)CN(C)C(C)Cc1cccs1",
    "NC1=C(Br)C(C(F)(F)F)N=N1",
    "O=C(O)C1=CC2C=CC1C1C=CC21",
    "CC(C)CC(C)(O)CNCC(Cl)=CCl",
    "CN1C(=O)C2=NN=NC2N(C)C1=O",
    "Oc1ccc(C2CNCc3sccc32)cc1O",
    "CC1CC(C)CC2(C1)NC(=O)NC2=N",
    "Clc1ccc(-n2ccc3ccccc32)cc1",
    "Brc1ccc(C=NN=Cc2ccc(Br)o2)o1",
    "C#CCN(CC(=O)O)C(=O)c1snnc1CC",
    "CNC1CCC(N(C)C2CCCC(C)CC2)CC1",
    "CSCCCNCc1coc(-c2ccc(C)cc2)n1",
    "CCC(Nc1ccc(F)cc1C#N)c1ccncc1",
    "COc1ccc(C(N)c2cc(C)ccc2C)cc1C",
    "CC1CN2CCCC2CN1CC1CCC2CCCCC2N1",
    "Cc1nnc2ccc(Oc3cncc(Br)c3)nn12",
    "CCCc1nn(C)c2c1nc(CCl)n2C1CC1C",
    "NCC1(C2(O)CCCC(C3CC3)C2)CCOC1",
    "COC(=O)C(C)C(C)NC(C)C(=O)N(C)C",
    "O=C(NCCCn1ccnc1)c1csc(C#CCO)c1",
    "CC(C)OP(=S)(OC(C)C)OP1OCCC(C)O1",
    "CC(C)(C)SCC(=O)NC1(CC(=O)O)CCC1",
    "COCCNC12CC3CC(C)(CC(C)(C3)C1)C2",
    "CCOC1(C)OC(N)=C(C#N)C1=C(C#N)C#N",
    "CC(Nc1ncc(F)c(N(C)C)n1)C1=CCCCC1",
    "Cc1noc(C)c1C(C)NS(=O)(=O)C(C)C#N",
    "CCCN(C(C)C)C1(CN)CC(C)(C)OC1(C)C",
    "O=S(=O)(N1CCCCC1)N1CCOC2(CCCC2)C1",
    "O=c1c2sccc2ncn1Cc1csc(-c2ccsc2)n1",
    "N#CC1(NC2CC2)CCC(Sc2ccc(Br)cc2)C1",
    "C=CCNc1nnc(SCc2cc3c(cc2Br)OCO3)s1",
    "CCN=C(NCC1CCOC1C(C)(C)C)NC(C)(C)C",
    "CC(C)=Nn1cnc2c1=NC(C)(C)NC=2C(N)=O",
    "CCC1(C)CC2=C3C(O)=NC(=S)N=C3SC2CO1",
    "Cc1nn(CCO)c(C)c1CNC1CCCCC1Cc1ccccc1",
    "CC(=O)Oc1ccc2c3c(ccc(C(C)=O)c13)CC2",
    "Cc1cc2oc(Br)c(CC(=O)N3CCOCC3)c2cc1Cl",
    "CS(=O)(=O)N1CCN(Cc2cc3n(n2)CCNC3)CC1",
    "CC(C)(C)c1noc(CCc2nc(-c3cnccn3)no2)n1",
    "O=C(NC1CC1)C1CCN(C(=O)C2CC3CCC2C3)CC1",
    "C=Cn1ccnc1P(=S)(c1ccn(C)c1)c1nccn1C=C",
    "CC(C)(C)C(CBr)CN1C(=O)C(C)(C)S1(=O)=O",
    "N#Cc1ccc2ncc(CN3CCCC4(CC=CCC4)C3)n2c1",
    "CC(O)C1C(O)CC2C3CCC4CCCCC4(C)C3CCC21C",
    "O=S1(=O)CC(Cl)C(SSC2CS(=O)(=O)CC2Cl)C1",
    "O=CC1=CN=c2c1ccc1c3c(ccc21)C(C=O)=CN=3",
    "O=C(OCC1CC=CCC1)C1CCCN1C(=O)OCC(F)(F)F",
    "CCC1(CC)C(OC)C(C)C1N1C(=O)C(CCSC)NC1=S",
    "CC(O)CNC(=O)C1NC(C2=CC3N=CC=C3C=C2)=NO1",
    "N#Cc1cccc(NC(=O)N(Cc2ccco2)Cc2ccccc2O)c1",
    "CCOc1cccc2c1NC(c1ccc(N(C)C)cc1)C1CC=CC21",
    "C1=CC(c2cccc(-c3cc(-n4cccn4)ncn3)c2)N=N1",
    "Nc1nn2c(-c3ccccc3)cc(O)nc2c1N=Nc1ccc(O)cc1",
    "FC(F)(F)c1ccc(Cl)c(NC(=S)NN=C2CC3C=CCC23)c1",
    "C1=NN=c2cc3c(cc21)=NC(c1cn(C2CCNCC2)nn1)=N3",
    "CC(C)Sc1nnc2n3ncnc3c3c4c(sc3n12)COC(C)(C)C4",
    "CS(=O)CCNS(=O)(=O)c1ccc2c(c1)NC(=O)C(C)(C)O2",
    "CS(=NS(=O)(=O)c1cccc2nsnc12)c1ncccc1C(F)(F)F",
    "COCCCN1COc2ccc3c(c2C1)OC(=Cc1ccc(Br)cc1)C3=O",
    "CC#CC1(O)CCC2C3CCC4=C(CCC5(C4)OCCO5)C3=CCC21C",
    "CCOC(=O)CCn1c(=O)c2c(nc3n(C)c(C)cn23)n(C)c1=O",
    "OC(c1ccccc1)C(F)(F)C1(F)OC(F)(F)C(F)(F)C1(F)F",
    "CC(C)(C)C1=CC(C2CC(=O)NCC3N=C4C=CC=CN4C32)N=N1",
    "Fc1ccccc1C1Oc2ccccc2C2=C1C(c1cccnc1)N1N=CNC1=N2",
    "C=CCN1C(=O)C(CC(=O)c2ccc(F)cc2)SC1=Nc1ccc(F)cc1",
    "CC1(C)OCC(C2OC3OC(C)(C)OC3C2OP2OCCN2C(C)(C)C)O1",
    "CC(OC(=O)c1ccc(-n2cncn2)cc1)C(=O)C1=c2ccccc2=NC1",
    "CCCC1CCc2c(sc(NC(=O)C3C4CCC(O4)C3C(=O)O)c2C(N)=O)C1",
    "Cc1cccc(N2C(=O)C(=C3SC(=S)N(c4cccc(C)c4C)C3=O)SC2=S)c1C",
    "C=C1CCCC2(C)CC3OC(=O)C(CN4CCc5cc(OC)c(OC)cc5C4CCO)C3CC12",
    "O=C1N=C(O)C(=Cc2ccc(OC(=O)c3sc4cc(Cl)ccc4c3Cl)cc2)C(=O)N1",
]

# -----------------------------------------------------------------------------
# Sampling Configuration
# -----------------------------------------------------------------------------

# :param SAMPLE_STEPS:
#     Number of denoising steps during discrete flow matching sampling.
#     Higher values give better results but are slower.
SAMPLE_STEPS: int = 50

# :param ETA:
#     Stochasticity parameter for sampling. 0.0 = deterministic CTMC.
ETA: float = 0.0

# :param OMEGA:
#     Target guidance strength parameter for sampling.
OMEGA: float = 0.0

# :param SAMPLE_TIME_DISTORTION:
#     Time distortion schedule for sampling. Options: "identity", "polydec".
SAMPLE_TIME_DISTORTION: str = "polydec"

# :param NOISE_TYPE_OVERRIDE:
#     Override the noise type used during sampling. Options: "uniform",
#     "marginal", or None (use the type the model was trained with).
NOISE_TYPE_OVERRIDE: Optional[str] = None

# :param DETERMINISTIC:
#     If True, use argmax instead of sampling for deterministic trajectories.
DETERMINISTIC: bool = False

# -----------------------------------------------------------------------------
# Repetition Configuration
# -----------------------------------------------------------------------------

# :param NUM_REPETITIONS:
#     Number of independent edge generation attempts per molecule.
#     When > 1, each attempt is scored by HDC cosine distance to the
#     original order_N embedding, and the best result is kept.
NUM_REPETITIONS: int = 128

# :param INIT_MODE:
#     Initialization mode for the edge matrix at the start of sampling.
#     Options:
#       - "noise": Sample initial edges from the limit distribution (default,
#         original stochastic behavior).
#       - "empty": Start from an all-no-edge graph (class 0 everywhere).
#         Fully deterministic when combined with DETERMINISTIC=True.
INIT_MODE: str = "noise"

# -----------------------------------------------------------------------------
# GIF Animation Configuration
# -----------------------------------------------------------------------------

# :param GENERATE_GIF:
#     Whether to generate animated GIFs showing the sampling trajectory
#     for each molecule.
GENERATE_GIF: bool = True

# :param GIF_FRAME_INTERVAL:
#     Capture a frame every N sampling steps for the GIF animation.
GIF_FRAME_INTERVAL: int = 10

# :param GIF_FPS:
#     Frames per second for the output GIF animation.
GIF_FPS: int = 10

# :param GIF_IMAGE_SIZE:
#     Size (width, height) of molecule rendering in GIF frames.
GIF_IMAGE_SIZE: Tuple[int, int] = (400, 400)

# -----------------------------------------------------------------------------
# System Configuration
# -----------------------------------------------------------------------------

# :param SEED:
#     Random seed for reproducibility.
SEED: int = 42

# :param DEVICE:
#     Device for the FlowEdgeDecoder inference. Options: "auto" (prefer GPU),
#     "cpu", "cuda".
DEVICE: str = "cuda"

# :param HDC_DEVICE:
#     Device for the HyperNet HDC encoder. Options: "auto" (prefer GPU),
#     "cpu", "cuda".
HDC_DEVICE: str = "cuda"

# -----------------------------------------------------------------------------
# Debug/Testing Modes
# -----------------------------------------------------------------------------

# :param __DEBUG__:
#     Debug mode - reuses same output folder during development.
__DEBUG__: bool = True

# :param __TESTING__:
#     Testing mode - runs with minimal iterations for validation.
__TESTING__: bool = False


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
    x = X[sample_idx]
    e = E[sample_idx]
    mask = node_mask[sample_idx]

    n_valid = mask.sum().item()
    x_valid = x[:n_valid]
    e_valid = e[:n_valid, :n_valid]

    e_labels = torch.argmax(e_valid, dim=-1)

    edge_src = []
    edge_dst = []
    edge_types = []

    for i in range(n_valid):
        for j in range(i + 1, n_valid):
            edge_type = e_labels[i, j].item()
            if edge_type > 0:
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
    output_width = image_size[0]
    output_height = image_size[1] + 60

    fig, ax = plt.subplots(figsize=(output_width / 100, output_height / 100), dpi=100)

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

    title = f"t = {t_value:.3f}"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    if smiles:
        smiles_display = smiles[:40] + "..." if len(smiles) > 40 else smiles
    else:
        smiles_display = "N/A"

    fig.text(0.5, 0.02, smiles_display, ha="center", fontsize=8, family="monospace")

    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.02, right=0.98)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white", edgecolor="none")
    buf.seek(0)
    pil_image = Image.open(buf).copy()
    buf.close()
    plt.close(fig)

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

    duration = 1.0 / fps
    frame_arrays = [np.array(frame) for frame in frames]

    imageio.mimsave(
        save_path,
        frame_arrays,
        format="GIF",
        duration=duration,
        loop=0,
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
        self.capture_interval = capture_interval
        self.sample_steps = sample_steps
        self.image_size = image_size
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
        should_capture = (
            step == 0
            or step % self.capture_interval == 0
            or step == self.sample_steps
        )

        if should_capture:
            noisy_pyg_data = dense_E_to_pyg_data(X, E, node_mask, sample_idx=0)

            pred_pyg_data = None
            if pred_E is not None:
                pred_pyg_data = dense_E_to_pyg_data(X, pred_E, node_mask, sample_idx=0)

            self.frames.append((step, t_value, noisy_pyg_data, pred_pyg_data))

    def render_gif(
        self,
        save_path: Path,
        fps: int = 10,
    ) -> None:
        """Render all captured noisy state frames to GIF."""
        rendered_frames = []

        for step, t_value, noisy_pyg_data, _ in self.frames:
            mol = pyg_to_mol(noisy_pyg_data)
            smiles = get_canonical_smiles(mol)
            frame_img = render_frame(mol, t_value, smiles, self.image_size)
            rendered_frames.append(frame_img)

        create_reconstruction_gif(rendered_frames, save_path, fps)

    def render_prediction_gif(
        self,
        save_path: Path,
        fps: int = 10,
    ) -> None:
        """Render all captured prediction frames to GIF."""
        rendered_frames = []

        for step, t_value, _, pred_pyg_data in self.frames:
            if pred_pyg_data is None:
                continue

            mol = pyg_to_mol(pred_pyg_data)
            smiles = get_canonical_smiles(mol)
            frame_img = render_frame(mol, t_value, smiles, self.image_size)
            rendered_frames.append(frame_img)

        if rendered_frames:
            create_reconstruction_gif(rendered_frames, save_path, fps)

    def clear(self) -> None:
        """Clear collected frames."""
        self.frames = []


def create_evolution_grid(
    frames: List[Tuple[int, float, Data, Optional[Data]]],
    save_path: Path,
    n_rows: int = 4,
    n_cols: int = 5,
) -> None:
    """
    Create a grid of subplots showing the molecule's evolution over sampling time.

    Selects ``n_rows * n_cols`` evenly-spaced frames from the captured list and
    renders each as an RDKit molecule image.  Subplots are arranged left-to-right,
    top-to-bottom (time 0 at top-left, final time at bottom-right).

    Args:
        frames: List of ``(step, t_value, noisy_pyg_data, pred_pyg_data)`` tuples
                as produced by :class:`FrameCollector`.
        save_path: Output path for the PNG figure.
        n_rows: Number of rows in the grid (default 4).
        n_cols: Number of columns in the grid (default 5).
    """
    n_cells = n_rows * n_cols

    if len(frames) == 0:
        return

    # Sub-sample to exactly n_cells frames, evenly spaced
    if len(frames) <= n_cells:
        selected = frames
    else:
        indices = np.linspace(0, len(frames) - 1, n_cells, dtype=int)
        selected = [frames[i] for i in indices]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3 * n_cols, 3 * n_rows),
        dpi=150,
    )

    for cell_idx, ax in enumerate(axes.flat):
        if cell_idx < len(selected):
            _step, t_value, noisy_pyg_data, _ = selected[cell_idx]
            mol = pyg_to_mol(noisy_pyg_data)

            if mol is not None:
                try:
                    img = Draw.MolToImage(mol, size=(300, 300))
                    ax.imshow(img)
                except Exception:
                    ax.text(
                        0.5, 0.5, "Render\nFailed",
                        ha="center", va="center", fontsize=10,
                        transform=ax.transAxes,
                    )
            else:
                ax.text(
                    0.5, 0.5, "Invalid",
                    ha="center", va="center", fontsize=10,
                    transform=ax.transAxes,
                )

            ax.set_title(f"t = {t_value:.3f}", fontsize=9)
        else:
            ax.set_visible(False)

        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
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
    e.log_parameters()

    # Device setup
    if e.DEVICE == "auto":
        decoder_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        decoder_device = torch.device(e.DEVICE)
    if e.HDC_DEVICE == "auto":
        hdc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        hdc_device = torch.device(e.HDC_DEVICE)
    e.log(f"Decoder device: {decoder_device}")
    e.log(f"HyperNet device: {hdc_device}")
    e["config/device"] = str(decoder_device)

    # =========================================================================
    # Apply Hooks
    # =========================================================================

    # Load models
    hypernet, decoder, base_hdc_dim = e.apply_hook(
        "load_models",
        device=decoder_device,
    )

    hypernet.to(hdc_device)
    hypernet.eval()
    decoder.to(decoder_device)
    decoder.eval()

    e["model/base_hdc_dim"] = base_hdc_dim
    e["model/concat_hdc_dim"] = 2 * base_hdc_dim

    # Load SMILES
    smiles_list = e.apply_hook("load_smiles")

    e.log(f"Number of SMILES to test: {len(smiles_list)}")
    e["data/num_smiles"] = len(smiles_list)

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
    e["config/num_repetitions"] = e.NUM_REPETITIONS
    e["config/init_mode"] = e.INIT_MODE
    e["config/generate_gif"] = e.GENERATE_GIF
    e["config/device_setting"] = e.DEVICE

    # Create output directories
    plots_dir = Path(e.path) / "comparison_plots"
    plots_dir.mkdir(exist_ok=True)

    # =========================================================================
    # Process Each SMILES
    # =========================================================================

    e.log("\n" + "=" * 60)
    e.log("Processing SMILES")
    e.log("=" * 60)

    results = []
    valid_count = 0
    match_count = 0
    invalid_count = 0
    skipped_count = 0

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

        num_atoms = original_mol.GetNumHeavyAtoms()

        # Add batch attribute (HyperNet runs on CPU)
        data = data.to(hdc_device)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=hdc_device)

        # Augment with RW features if the hypernet expects them
        if hasattr(hypernet, "rw_config") and hypernet.rw_config.enabled:
            from graph_hdc.utils.rw_features import augment_data_with_rw
            data = augment_data_with_rw(
                data,
                k_values=hypernet.rw_config.k_values,
                num_bins=hypernet.rw_config.num_bins,
                bin_boundaries=hypernet.rw_config.bin_boundaries,
                clip_range=hypernet.rw_config.clip_range,
            )

        # Encode with HyperNet - compute concatenated [order_0 | order_N]
        with torch.no_grad():
            encoder_output = hypernet.forward(data, normalize=True)
            order_zero = encoder_output["node_terms"]
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
                "num_atoms": num_atoms,
                "is_match": False,
            })
            continue

        # Prepare inputs for edge generation (on decoder device)
        feature_bins = get_node_feature_bins(hypernet.rw_config)
        node_features = node_tuples_to_onehot(node_tuples, device=decoder_device, feature_bins=feature_bins).unsqueeze(0)
        node_mask = torch.ones(1, num_nodes, dtype=torch.bool, device=decoder_device)
        hdc_vectors = hdc_vector.unsqueeze(0).to(decoder_device)

        # Generate edges via hook
        generated_samples = e.apply_hook(
            "generate_edges",
            decoder=decoder,
            hypernet=hypernet,
            hdc_vectors=hdc_vectors,
            node_features=node_features,
            node_mask=node_mask,
            node_tuples=node_tuples,
            num_nodes=num_nodes,
            original_data=data,
            base_hdc_dim=base_hdc_dim,
            device=decoder_device,
            idx=idx,
            plots_dir=plots_dir,
        )

        # Handle skip (None return from hook)
        if generated_samples is None:
            e.log("  Skipped by generate_edges hook")
            skipped_count += 1
            results.append({
                "idx": idx,
                "original_smiles": smiles,
                "generated_smiles": None,
                "status": "skipped",
                "error": "Skipped by generate_edges hook",
            })
            continue

        # Convert to RDKit molecule
        generated_data = generated_samples[0]
        generated_mol = pyg_to_mol(generated_data)
        generated_smiles = get_canonical_smiles(generated_mol)

        e.log(f"  Generated: {generated_smiles or 'N/A'}")

        # Check validity and match
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
        create_reconstruction_plot(
            original_mol=original_mol,
            generated_mol=generated_mol,
            original_smiles=smiles,
            generated_smiles=generated_smiles or "N/A",
            is_valid=is_valid,
            is_match=is_match,
            sample_idx=idx,
            save_path=plot_path,
        )

        # Per-sample timing / ETA
        elapsed_total = time.time() - start_time
        processed_so_far = idx + 1 - skipped_count
        if processed_so_far > 0:
            avg_per_sample = elapsed_total / processed_so_far
            remaining = len(smiles_list) - (idx + 1)
            eta_seconds = avg_per_sample * remaining
            eta_time = datetime.now() + timedelta(seconds=eta_seconds)
            valid_pct = 100 * valid_count / processed_so_far
            match_pct = 100 * match_count / processed_so_far
            e.log(f"  Time: {elapsed_total:.1f}s elapsed | "
                  f"avg {avg_per_sample:.2f}s/sample | "
                  f"~{eta_seconds:.0f}s remaining | "
                  f"ETA {eta_time:%Y-%m-%d %H:%M:%S}")
            e.log(f"  Accuracy: {match_count}/{processed_so_far} match ({match_pct:.1f}%) | "
                  f"{valid_count}/{processed_so_far} valid ({valid_pct:.1f}%)")

        results.append({
            "idx": idx,
            "original_smiles": smiles,
            "original_canonical": original_canonical,
            "generated_smiles": generated_smiles,
            "status": status,
            "is_valid": is_valid,
            "is_match": is_match,
            "num_atoms": num_atoms,
        })

        # Free GPU memory: drop per-iteration tensors and flush CUDA cache
        del data, hdc_vector, node_features, node_mask, hdc_vectors
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # Summary
    # =========================================================================

    end_time = time.time()
    total_time = end_time - start_time
    num_processed = len(smiles_list) - skipped_count

    # Create summary visualization via hook
    e.apply_hook(
        "create_summary_visualization",
        match_count=match_count,
        valid_count=valid_count,
        invalid_count=invalid_count,
        total_count=num_processed,
    )

    # Create accuracy-by-molecule-size chart
    accuracy_by_size_path = Path(e.path) / "accuracy_by_size.png"
    create_accuracy_by_size_chart(results, accuracy_by_size_path)
    e.log(f"Accuracy by size chart saved to: {accuracy_by_size_path}")

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
            "num_repetitions": e.NUM_REPETITIONS,
            "init_mode": e.INIT_MODE,
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


# =============================================================================
# HOOKS
# =============================================================================


@experiment.hook("load_models", default=True)
def load_models(
    e: Experiment,
    device: torch.device,
) -> Tuple[HyperNet, FlowEdgeDecoder, int]:
    """
    Load HyperNet encoder and FlowEdgeDecoder models.

    Default implementation loads from checkpoint paths specified in parameters,
    or creates dummy models in testing mode.

    Args:
        e: Experiment instance for accessing parameters.
        device: Device to load models to.

    Returns:
        Tuple of (hypernet, decoder, base_hdc_dim).
    """
    e.log("\nLoading models...")

    if e.__TESTING__:
        e.log("TESTING MODE: Creating dummy models...")
        device = torch.device("cpu")
        hypernet, decoder, base_hdc_dim = create_test_dummy_models(device)
    else:
        if not e.HDC_ENCODER_PATH:
            raise ValueError("HDC_ENCODER_PATH is required")
        if not e.FLOW_DECODER_PATH:
            raise ValueError("FLOW_DECODER_PATH is required")

        e.log(f"Loading HyperNet from: {e.HDC_ENCODER_PATH}")
        hypernet = load_hypernet(e.HDC_ENCODER_PATH, device="cpu")
        base_hdc_dim = hypernet.hv_dim
        e.log(f"  HyperNet hv_dim: {base_hdc_dim}")

        # Rebuild full unpruned codebook so that any valid feature tuple
        # can be encoded (not just the ones observed during training).
        if hasattr(hypernet, "rebuild_unpruned_codebook"):
            hypernet.rebuild_unpruned_codebook()
            e.log("  Rebuilt unpruned codebook")

        e.log(f"Loading FlowEdgeDecoder from: {e.FLOW_DECODER_PATH}")
        decoder = FlowEdgeDecoder.load(e.FLOW_DECODER_PATH, device=device)
        e.log(f"  FlowEdgeDecoder hdc_dim: {decoder.hdc_dim}")
        e.log(f"  FlowEdgeDecoder condition_dim: {decoder.condition_dim}")

    return hypernet, decoder, base_hdc_dim


@experiment.hook("load_smiles", default=True)
def load_smiles(e: Experiment) -> list[str]:
    """
    Load SMILES strings for testing.

    Default implementation loads from CSV file if SMILES_CSV_PATH is set,
    otherwise uses the SMILES_LIST parameter.

    Args:
        e: Experiment instance.

    Returns:
        List of SMILES strings.
    """
    if e.SMILES_CSV_PATH:
        e.log(f"Loading SMILES from CSV: {e.SMILES_CSV_PATH}")
        smiles_list = load_smiles_from_csv(e.SMILES_CSV_PATH)
        e["config/smiles_source"] = "csv"
        e["config/smiles_csv_path"] = e.SMILES_CSV_PATH
    else:
        smiles_list = e.SMILES_LIST
        e["config/smiles_source"] = "list"

    return smiles_list


@experiment.hook("generate_edges", default=True)
def generate_edges(
    e: Experiment,
    decoder: FlowEdgeDecoder,
    hypernet: HyperNet,
    hdc_vectors: torch.Tensor,
    node_features: torch.Tensor,
    node_mask: torch.Tensor,
    node_tuples: list,
    num_nodes: int,
    original_data: Data,
    base_hdc_dim: int,
    device: torch.device,
    idx: int,
    plots_dir: Path,
) -> Optional[List[Data]]:
    """
    Generate edges with optional best-of-N repetitions.

    When ``NUM_REPETITIONS > 1``, edge generation is run N times in a
    single batched ``sample_best_of_n()`` call and the result with the
    lowest HDC cosine distance to the original molecule is kept.

    Child experiments can override this hook to use different sampling methods
    (e.g. ``sample_with_hdc()``, ``sample_with_hdc_guidance()``).

    Args:
        e: Experiment instance.
        decoder: FlowEdgeDecoder model.
        hypernet: HyperNet encoder.
        hdc_vectors: Concatenated HDC vectors (1, 2*hdc_dim).
        node_features: One-hot node features (1, n, 24).
        node_mask: Valid node mask (1, n).
        node_tuples: Decoded node tuples.
        num_nodes: Number of decoded nodes.
        original_data: Original PyG Data object (with original features).
        base_hdc_dim: Base hypervector dimension.
        device: Device for computation.
        idx: Current molecule index.
        plots_dir: Directory for saving per-molecule outputs.

    Returns:
        List of generated PyG Data objects, or None to skip this molecule.
    """
    num_reps = e.NUM_REPETITIONS

    # GIF handling: only supported for single repetition
    generate_gif = e.GENERATE_GIF and num_reps == 1
    if num_reps > 1 and e.GENERATE_GIF and idx == 0:
        e.log("  NOTE: GIF generation disabled when NUM_REPETITIONS > 1")

    # Capture interval for the evolution grid (~20 frames)
    grid_capture_interval = max(1, e.SAMPLE_STEPS // 19)

    # Build initial edges based on INIT_MODE
    init_edges = None
    if e.INIT_MODE == "empty":
        n_max = node_features.size(1)
        de = decoder.num_edge_classes
        init_edges = torch.zeros(1, n_max, n_max, de, device=device)
        init_edges[..., 0] = 1.0  # class 0 = no_edge
    elif e.INIT_MODE != "noise":
        raise ValueError(f"Unknown INIT_MODE: {e.INIT_MODE}. Use 'noise' or 'empty'.")

    # Common sampling kwargs
    sample_kwargs = dict(
        sample_steps=e.SAMPLE_STEPS,
        eta=e.ETA,
        omega=e.OMEGA,
        time_distortion=e.SAMPLE_TIME_DISTORTION,
        noise_type_override=e.NOISE_TYPE_OVERRIDE,
        show_progress=False,
        deterministic=e.DETERMINISTIC,
        initial_edges=init_edges,
        device=device,
    )

    best_frame_collector = None

    if num_reps > 1:
        # ─── Batched best-of-N via sample_best_of_n ───
        # Use decoded node tuples as raw features for HDC distance computation.
        # We cannot rely on onehot_to_raw_features(generated_data.x) because
        # dense_to_pyg treats the 24-dim concatenated multi-feature one-hot as
        # a single-class one-hot (argmax→one_hot), destroying all features
        # except atom type.
        raw_x = torch.tensor(node_tuples, dtype=torch.float)

        def score_fn(s):
            return compute_hdc_distance(
                s, hdc_vectors, base_hdc_dim,
                hypernet, device, dataset=e.DATASET,
                original_x=raw_x,
            )

        with torch.no_grad():
            best_sample, best_distance, avg_distance = decoder.sample_best_of_n(
                hdc_vectors=hdc_vectors,
                node_features=node_features,
                node_mask=node_mask,
                num_repetitions=num_reps,
                score_fn=score_fn,
                **sample_kwargs,
            )

        best_samples = [best_sample]
        e.log(f"  Best HDC distance across {num_reps} reps: {best_distance:.6f} (avg: {avg_distance:.6f})")

    else:
        # ─── Single repetition: preserve GIF / evolution grid support ───
        if generate_gif:
            capture_interval = min(grid_capture_interval, e.GIF_FRAME_INTERVAL)
        else:
            capture_interval = grid_capture_interval

        frame_collector = FrameCollector(
            capture_interval=capture_interval,
            sample_steps=e.SAMPLE_STEPS,
            image_size=e.GIF_IMAGE_SIZE,
        )

        with torch.no_grad():
            best_samples = decoder.sample(
                hdc_vectors=hdc_vectors,
                node_features=node_features,
                node_mask=node_mask,
                step_callback=frame_collector,
                **sample_kwargs,
            )

        if generate_gif:
            gif_path = plots_dir / f"reconstruction_{idx + 1:04d}_animation.gif"
            frame_collector.render_gif(gif_path, fps=e.GIF_FPS)
            e.log(f"  GIF saved: {gif_path.name}")
            pred_gif_path = plots_dir / f"reconstruction_{idx + 1:04d}_prediction.gif"
            frame_collector.render_prediction_gif(pred_gif_path, fps=e.GIF_FPS)
            e.log(f"  Prediction GIF saved: {pred_gif_path.name}")

        best_frame_collector = frame_collector

    # Save evolution grid for the best repetition
    if best_frame_collector is not None and len(best_frame_collector.frames) > 0:
        grid_path = plots_dir / f"evolution_{idx + 1:04d}.png"
        create_evolution_grid(best_frame_collector.frames, grid_path)
        e.log(f"  Evolution grid saved: {grid_path.name}")

    return best_samples


@experiment.hook("create_summary_visualization", default=True)
def create_summary_visualization(
    e: Experiment,
    match_count: int,
    valid_count: int,
    invalid_count: int,
    total_count: int,
) -> None:
    """
    Create summary visualization of test results.

    Default implementation creates a bar chart showing match/valid/invalid counts.

    Args:
        e: Experiment instance.
        match_count: Number of exact SMILES matches.
        valid_count: Number of valid molecules (includes matches).
        invalid_count: Number of invalid molecules.
        total_count: Total number of processed molecules.
    """
    summary_plot_path = Path(e.path) / "summary_bar_chart.png"
    create_summary_bar_chart(
        match_count=match_count,
        valid_count=valid_count,
        invalid_count=invalid_count,
        total_count=total_count,
        save_path=summary_plot_path,
    )
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
    e.GIF_FRAME_INTERVAL = 2  # Capture more frames in test mode


# =============================================================================
# Entry Point
# =============================================================================

experiment.run_if_main()
