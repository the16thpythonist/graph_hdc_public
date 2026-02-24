#!/usr/bin/env python
"""
Train Flow Matching on graph_terms conditioned on node_terms (Stage 2).

Child experiment of ``train_flow_matching.py`` that trains a flow model
to generate ``graph_terms`` (graph-level topology encoding) conditioned
on ``node_terms`` (atom identities). Uses real node_terms from the
dataset as conditioning (teacher forcing).

In two-stage hierarchical generation, this model is trained after the
node stage. At inference time, node_terms sampled from Stage 1 are
passed as the conditioning signal.

Usage:
    # Quick test
    python train_flow_matching__graph.py --__TESTING__ True

    # Full training
    python train_flow_matching__graph.py --ENCODER_PATH /path/to/encoder.ckpt

    # With auxiliary cosine loss
    python train_flow_matching__graph.py --ENCODER_PATH /path/to/encoder.ckpt --COSINE_LOSS_WEIGHT 0.1
"""
from __future__ import annotations

from typing import Optional, List
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path


# =============================================================================
# PARAMETER OVERRIDES
# =============================================================================

# Train only on the graph_terms half, conditioned on node_terms.
# The parent automatically sets condition_dim = hv_dim for this mode.
VECTOR_PART: str = "graph_terms"

# Auxiliary cosine loss weight (0 = disabled).
COSINE_LOSS_WEIGHT: float = 0.0

# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------

# :param HIDDEN_DIM:
#     Hidden dimension for the velocity MLP.
HIDDEN_DIM: int = 512

# :param NUM_BLOCKS:
#     Number of residual FiLM blocks in the velocity MLP.
NUM_BLOCKS: int = 18

# :param TIME_EMBED_DIM:
#     Dimension of sinusoidal time embedding.
TIME_EMBED_DIM: int = 128

# :param CONDITIONS:
#     List of condition names from the registry. None or empty = unconditional.
#     Available: "logp" (Crippen LogP), "heavy_atoms" (heavy atom count).
#     Example: ["logp"] for single, ["logp", "heavy_atoms"] for multi.
CONDITIONS: Optional[List[str]] = None

# :param CONDITION_EMBED_DIM:
#     Dimension to project raw condition values into via a learned MLP.
#     Only used when CONDITIONS is set.
CONDITION_EMBED_DIM: int = 128

# :param DROPOUT:
#     Dropout rate in residual blocks.
DROPOUT: float = 0.1

# :param VELOCITY_ARCH:
#     Velocity network architecture. Options: "mlp" (FiLM residual blocks),
#     "dit" (DiT-style Transformer with adaLN-Zero conditioning).
VELOCITY_ARCH: str = "mlp"

# :param NUM_HEADS:
#     Number of attention heads (DiT only). Must divide HIDDEN_DIM evenly.
NUM_HEADS: int = 8

# :param NUM_TOKENS:
#     Number of tokens to chunk the HDC vector into (DiT only).
#     data_dim is padded to the nearest multiple of NUM_TOKENS if needed.
#     Default 32 means each token covers 16 dims for data_dim=512.
NUM_TOKENS: int = 32

# :param MLP_RATIO:
#     FFN hidden dimension as multiple of HIDDEN_DIM (DiT only).
MLP_RATIO: float = 4.0

# :param VECTOR_PART:
#     Which part of the HDC vector to train on.
#     "both": Full [node_terms | graph_terms] (default, standard training).
#     "node_terms": Train only on node_terms (Stage 1 of hierarchical).
#     "graph_terms": Train on graph_terms conditioned on node_terms (Stage 2).
VECTOR_PART: str = "graph_terms"

# :param COSINE_LOSS_WEIGHT:
#     Weight for auxiliary cosine similarity loss. 0.0 disables it.
#     Encourages directional alignment between predictions and targets,
#     which matters for HDC decoding via circular correlation.
COSINE_LOSS_WEIGHT: float = 0.0

# :param DEQUANT_SIGMA:
#     Dequantization noise scale (std) applied to training targets in
#     standardized space. Smooths the discrete lattice of HDC vectors
#     into a continuous distribution. 0.0 disables dequantization.
#     Suggested range: [0.01, 0.2]. Start with 0.05.
DEQUANT_SIGMA: float = 0.05

# :param COND_DROPOUT_PROB:
#     Probability of zeroing out the entire condition vector per sample
#     during training. Regularizes the condition pathway to prevent
#     memorization of condition->target pairs. Also enables classifier-
#     free guidance at inference. 0.0 disables condition dropout.
COND_DROPOUT_PROB: float = 0.15

# :param PCA_N_COMPONENTS:
#     Number of PCA components for dimensionality reduction (0 = disabled).
#     Only supported with VECTOR_PART="node_terms".  Reduces the flow's
#     data_dim from hv_dim to this value, eliminating dead dimensions
#     in the node_terms distribution.
PCA_N_COMPONENTS: int = 0

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------

# :param NUM_SUBSAMPLE:
#     Optional subsample size for quick testing. When set, only this many
#     training samples (and 20% for validation) are used. None = full dataset.
NUM_SUBSAMPLE: Optional[int] = None

# =============================================================================
# EXPERIMENT
# =============================================================================

experiment = Experiment.extend(
    "train_flow_matching.py",
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

experiment.run_if_main()
