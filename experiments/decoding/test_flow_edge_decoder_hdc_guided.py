#!/usr/bin/env python
"""
Test FlowEdgeDecoder with Soft HDC Gradient Guidance.

This experiment extends test_flow_edge_decoder.py (base test experiment) and
overrides the generate_edges hook to use soft HDC gradient guidance.

At each timestep during sampling:
1. The model's predicted edge logits are relaxed into a soft adjacency matrix
2. A differentiable soft HDC encoder computes a graph embedding via
   matrix-multiply message passing + FFT circular convolution
3. The cosine distance to the target order_N is back-propagated to get
   per-edge, per-class gradients
4. These gradients steer sampling toward HDC-consistent edge configurations,
   either by blending into the predicted distribution or by adding an R^HDC
   rate matrix term

Usage:
    # Test with SMILES list
    python test_flow_edge_decoder_hdc_guided.py \\
        --HDC_ENCODER_PATH /path/to/encoder.ckpt \\
        --FLOW_DECODER_PATH /path/to/decoder.ckpt \\
        --DATASET qm9 \\
        --GAMMA 0.5 \\
        --INTEGRATION_MODE blend

    # Quick test
    python test_flow_edge_decoder_hdc_guided.py --__TESTING__ True
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from torch_geometric.data import Data

from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.models.flow_edge_decoder import FlowEdgeDecoder


# =============================================================================
# PARAMETER OVERRIDES
# =============================================================================

# -----------------------------------------------------------------------------
# Model Paths
# -----------------------------------------------------------------------------

# :param HDC_ENCODER_PATH:
#     Path to saved HyperNet encoder checkpoint (.ckpt).
HDC_ENCODER_PATH: str = "/media/ssd2/Programming/graph_hdc_public/experiments/decoding/results/train_flow_edge_decoder_streaming/GOOD/hypernet_encoder.ckpt"

# :param FLOW_DECODER_PATH:
#     Path to saved FlowEdgeDecoder checkpoint (.ckpt).
FLOW_DECODER_PATH: str = "/media/ssd2/Programming/graph_hdc_public/experiments/decoding/results/train_flow_edge_decoder_streaming/GOOD/last.ckpt"

# -----------------------------------------------------------------------------
# Input SMILES (smaller default list for guided sampling which is slower)
# -----------------------------------------------------------------------------

# :param SMILES_LIST:
#     List of SMILES strings to test. Guided sampling is slower per molecule,
#     so the default list is smaller.
SMILES_LIST: list[str] = [
    "CCO",      # Ethanol
    "CC(=O)O",  # Acetic acid
    "c1ccccc1", # Benzene
    "CCN",      # Ethylamine
    "CC=O",     # Acetaldehyde
    "CCN(Cc1ccc(OC)c(OC)c1)C(=O)c2cscc2",
    "C(NCC1=CC=C(C)C=C1)1=NC=NC(N2C3CC(C)(C)CC(C)(C3)C2)=C1N",
    "C(=O)1N(C)CC(NC(C2SC=CC=2)C2CC2)C1",
    "C1C(C(C#N)N2C(C3=CC=CC=C3)CC(C)=N2)=COC=1",
]

# -----------------------------------------------------------------------------
# Sampling Configuration (override base defaults)
# -----------------------------------------------------------------------------

# :param SAMPLE_STEPS:
#     Number of denoising steps during sampling.
SAMPLE_STEPS: int = 100

# :param SAMPLE_TIME_DISTORTION:
#     Time distortion schedule for sampling.
SAMPLE_TIME_DISTORTION: str = "polydec"

# :param NOISE_TYPE_OVERRIDE:
#     Override noise type. HDC-guided variant defaults to marginal noise.
NOISE_TYPE_OVERRIDE: Optional[str] = "marginal"

# -----------------------------------------------------------------------------
# Soft HDC Gradient Guidance Configuration
# -----------------------------------------------------------------------------

# :param GAMMA:
#     HDC guidance strength. 0.0 = no guidance, higher values bias sampling
#     more strongly toward HDC-consistent edge configurations.
GAMMA: float = 0.5

# :param TAU:
#     Softmax temperature for the soft edge probabilities and for converting
#     the negative gradient into a distribution. Lower values make the soft
#     edges sharper (closer to discrete) but the gradient becomes noisier.
TAU: float = 0.5

# :param INTEGRATION_MODE:
#     How to apply the gradient signal. Options:
#       - "blend": mix gradient-derived target distribution with model's
#         prediction before rate matrix computation (most principled).
#       - "rate_matrix": add R^HDC = relu(-grad) * alpha to the rate matrix
#         after standard computation.
INTEGRATION_MODE: str = "blend"

# :param SCHEDULE:
#     Time-dependent guidance strength schedule. Options:
#       - "constant": alpha = gamma at every step.
#       - "linear_decay": alpha = gamma * (1 - t), more guidance early.
#       - "linear_ramp": alpha = gamma * t, more guidance late.
SCHEDULE: str = "linear_decay"

# -----------------------------------------------------------------------------
# GIF Animation (now supported via step_callback)
# -----------------------------------------------------------------------------

# :param GENERATE_GIF:
#     Whether to generate animated GIFs showing the sampling trajectory.
GENERATE_GIF: bool = True

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
# INHERIT FROM BASE TEST EXPERIMENT
# =============================================================================

experiment = Experiment.extend(
    "test_flow_edge_decoder.py",
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


# =============================================================================
# HOOK OVERRIDES
# =============================================================================


@experiment.hook("generate_edges", default=False, replace=True)
def generate_edges_hdc_guided(
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
    Generate edges using soft HDC gradient guidance.

    Converts the decoded ``node_tuples`` to raw integer features for
    codebook lookup, then calls
    ``decoder.sample_with_soft_hdc_guidance()`` which computes a
    differentiable soft HDC encoding at each timestep and uses the
    gradient to steer sampling.

    Unlike the old K-candidate approach, this does **not** require the
    decoded node count to match the original molecule's node count.

    Args:
        e: Experiment instance.
        decoder: FlowEdgeDecoder model.
        hypernet: HyperNet encoder (for codebook and depth).
        hdc_vectors: Concatenated HDC vectors (1, 2*hdc_dim).
        node_features: One-hot node features (1, n, 24).
        node_mask: Valid node mask (1, n).
        node_tuples: Decoded node tuples [(atom, deg, chg, hs, ring), ...].
        num_nodes: Number of decoded nodes.
        original_data: Original PyG Data (unused by soft approach).
        base_hdc_dim: Base hypervector dimension.
        device: Device for computation.
        idx: Current molecule index.
        plots_dir: Directory for saving per-molecule outputs.

    Returns:
        List of generated PyG Data objects, or None to skip this molecule.
    """
    # Build raw_node_features from decoded node_tuples for codebook lookup
    raw_node_features = torch.tensor(
        [list(t) for t in node_tuples],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)  # (1, n, 5)

    with torch.no_grad():
        generated_samples = decoder.sample_with_soft_hdc_guidance(
            hdc_vectors=hdc_vectors,
            node_features=node_features,
            node_mask=node_mask,
            hypernet=hypernet,
            raw_node_features=raw_node_features,
            gamma=e.GAMMA,
            tau=e.TAU,
            integration_mode=e.INTEGRATION_MODE,
            schedule=e.SCHEDULE,
            eta=e.ETA,
            omega=e.OMEGA,
            sample_steps=e.SAMPLE_STEPS,
            time_distortion=e.SAMPLE_TIME_DISTORTION,
            noise_type_override=e.NOISE_TYPE_OVERRIDE,
            show_progress=False,
            deterministic=e.DETERMINISTIC,
            device=device,
        )

    return generated_samples


@experiment.hook("create_summary_visualization", default=False, replace=True)
def create_summary_visualization_guided(
    e: Experiment,
    match_count: int,
    valid_count: int,
    invalid_count: int,
    total_count: int,
) -> None:
    """Create summary visualization with soft HDC-guided-specific title."""
    from graph_hdc.utils.experiment_helpers import create_summary_bar_chart

    if total_count > 0:
        summary_plot_path = Path(e.path) / "summary_chart.png"
        create_summary_bar_chart(
            match_count=match_count,
            valid_count=valid_count,
            invalid_count=invalid_count,
            total_count=total_count,
            save_path=summary_plot_path,
            title_prefix="Soft HDC-Guided",
        )
        e.log(f"Summary chart saved to: {summary_plot_path}")


# =============================================================================
# Testing Mode
# =============================================================================


@experiment.testing
def testing(e: Experiment) -> None:
    """Quick test mode with reduced parameters for soft HDC-guided variant."""
    e.SAMPLE_STEPS = 10
    e.SMILES_LIST = ["CCO", "CC=O"]
    e.DATASET = "zinc"
    e.GAMMA = 0.5
    e.TAU = 0.5
    e.INTEGRATION_MODE = "blend"
    e.SCHEDULE = "linear_decay"
    e.GENERATE_GIF = False


# =============================================================================
# Entry Point
# =============================================================================

experiment.run_if_main()
