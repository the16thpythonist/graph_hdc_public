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
COSINE_LOSS_WEIGHT: float = 0.0

# :param DEQUANT_SIGMA:
#     Dequantization noise scale (std) applied to training targets in
#     standardized space. Smooths the discrete lattice of HDC vectors
#     into a continuous distribution. 0.0 disables dequantization.
#     Suggested range: [0.01, 0.2]. Start with 0.05.
DEQUANT_SIGMA: float = 0.1

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
