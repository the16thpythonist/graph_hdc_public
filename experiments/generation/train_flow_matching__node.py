#!/usr/bin/env python
"""
Train Flow Matching on node_terms only (Stage 1 of hierarchical generation).

Child experiment of ``train_flow_matching.py`` that trains a flow model
to generate only the ``node_terms`` (order-0 bundled atom hypervectors).
These are the first half of the HDC vector and encode atom identities.

In two-stage hierarchical generation, this model is trained first, then
its samples serve as conditioning input for the graph_terms stage.

Usage:
    # Quick test
    python train_flow_matching__node.py --__TESTING__ True

    # Full training
    python train_flow_matching__node.py --ENCODER_PATH /path/to/encoder.ckpt

    # With auxiliary cosine loss
    python train_flow_matching__node.py --ENCODER_PATH /path/to/encoder.ckpt --COSINE_LOSS_WEIGHT 0.1
"""
from __future__ import annotations

from typing import Optional
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path


# =============================================================================
# PARAMETER OVERRIDES
# =============================================================================

# Train only on the node_terms half of the HDC vector.
VECTOR_PART: str = "node_terms"

# Auxiliary cosine loss weight (0 = disabled).
COSINE_LOSS_WEIGHT: float = 1.0

# :param DEQUANT_SIGMA:
#     Dequantization noise scale (std) applied to training targets in
#     standardized space. Smooths the discrete lattice of HDC vectors
#     into a continuous distribution. 0.0 disables dequantization.
#     Suggested range: [0.01, 0.2]. Start with 0.05.
DEQUANT_SIGMA: float = 0.2

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------

# :param NUM_SUBSAMPLE:
#     Optional subsample size for quick testing. When set, only this many
#     training samples (and 20% for validation) are used. None = full dataset.
NUM_SUBSAMPLE: Optional[int] = None

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------

# :param EPOCHS:
#     Number of training epochs.
EPOCHS: int = 1000

# :param BATCH_SIZE:
#     Batch size for training.
BATCH_SIZE: int = 512

# :param LEARNING_RATE:
#     Learning rate for AdamW optimizer.
LEARNING_RATE: float = 2e-4

# :param WEIGHT_DECAY:
#     Weight decay for AdamW.
WEIGHT_DECAY: float = 0e-5

# :param WARMUP_EPOCHS:
#     Number of linear warmup epochs.
WARMUP_EPOCHS: int = 5

# :param ENCODER_BATCH_SIZE:
#     Batch size for HDC encoding. Lower values reduce GPU memory pressure
#     during FFT operations (relevant for MultiHyperNet).
ENCODER_BATCH_SIZE: int = 256

# :param GRADIENT_CLIP_VAL:
#     Gradient clipping value.
GRADIENT_CLIP_VAL: float = 1.0

# :param LOSS_TYPE:
#     Loss function for velocity matching. Options: "mse", "pseudo_huber".
LOSS_TYPE: str = "mse"

# :param TIME_SAMPLING:
#     Timestep sampling distribution. Options: "uniform", "logit_normal".
#     Logit-normal concentrates on mid-range timesteps (SD3 recipe).
TIME_SAMPLING: str = "logit_normal"

# :param PREDICTION_TYPE:
#     Prediction parameterization. Options: "velocity" (predict dx/dt),
#     "x_prediction" (predict clean data x_1, converted to velocity at inference).
PREDICTION_TYPE: str = "x_prediction"

# :param EVAL_TRACKING_EVERY_N:
#     How often (in epochs) to compute expensive tracking metrics
#     (sample quality, reconstruction, velocity norm, etc.).
EVAL_TRACKING_EVERY_N: int = 5

# :param USE_OT_COUPLING:
#     Whether to use minibatch optimal transport coupling.
USE_OT_COUPLING: bool = True

# :param USE_EMA:
#     Whether to use Exponential Moving Average of model weights.
USE_EMA: bool = True

# :param EMA_DECAY:
#     EMA decay rate. 0.9999 is standard (DiT, SD3).
EMA_DECAY: float = 0.9999

# :param STANDARDIZE:
#     Whether to standardize input vectors (zero mean, unit variance).
STANDARDIZE: bool = True

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
