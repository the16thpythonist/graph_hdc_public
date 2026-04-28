#!/usr/bin/env python
"""
Train Flow Matching on [edge_terms | graph_terms] with RRWP encoder.

Child experiment of ``train_flow_matching.py`` that trains a single-stage
unconditional flow model over concatenated edge_terms and graph_terms.
This mirrors the setup used in the HPO sweep (``hpo_flow_matching.py``):
the flow learns to generate full molecular HDC embeddings in one shot.

The encoder must be created with RRWP features using
``experiments/scripts/create_rrwp_encoder.py``. Default config:
dim=1024, depth=3, k=(6,10,14), bins=8.

After encoding, ``edge_terms`` is copied into the ``node_terms`` slot
so the parent's ``_extract_vectors(vector_part="both")`` returns
``[edge_terms | graph_terms]`` — a 2048-dim vector for dim=1024.

Usage:
    # 1. Create the encoder (one-time)
    python experiments/scripts/create_rrwp_encoder.py

    # 2. Train the flow
    python experiments/generation/train_flow_matching__edge_graph.py \
        --ENCODER_PATH encoders/zinc_d1024_depth3_k6_10_14_b8.ckpt
"""
from __future__ import annotations

from typing import Optional

from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.models.flow_matching import MultiCondition


# =============================================================================
# PARAMETER OVERRIDES
# =============================================================================

# Default encoder path (created by create_rrwp_encoder.py with default args)
ENCODER_PATH: str = "/media/ssd2/Programming/_branch/graph_hdc_public/experiments/encoders/zinc_d1024_depth3_k6_10_14_b8.ckpt"

# Full [edge_terms | graph_terms] vector, unconditional.
VECTOR_PART: str = "both"

# Cosine alignment loss to improve angular precision for HDC decoding.
COSINE_LOSS_WEIGHT: float = 0.0

# Standard MSE loss with x_prediction.
LOSS_TYPE: str = "mse"
PREDICTION_TYPE: str = "x_prediction"

# HPO-optimized hyperparameters.
BATCH_SIZE: int = 256
LEARNING_RATE: float = 7.839966958361505e-05
WEIGHT_DECAY: float = 0.00011290133559092664
HIDDEN_DIM: int = 1280
NUM_BLOCKS: int = 4
TIME_EMBED_DIM: int = 256
DROPOUT: float = 0.03881699724000254

# =============================================================================
# EXPERIMENT
# =============================================================================

experiment = Experiment.extend(
    "train_flow_matching.py",
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.hook("load_and_encode_data", default=False)
def load_and_encode_data(
    e: Experiment,
    hypernet: HyperNet,
    device,
    conditioning: Optional[MultiCondition] = None,
):
    """Load data, encode, then swap edge_terms into node_terms slot.

    After ``post_compute_encodings``, each Data object has ``node_terms``,
    ``edge_terms``, and ``graph_terms``.  We overwrite ``node_terms`` with
    ``edge_terms`` so that ``_extract_vectors(vector_part="both")`` returns
    ``[edge_terms | graph_terms]``.
    """
    import torch
    from graph_hdc.datasets.utils import get_split, post_compute_encodings

    e.log("Loading dataset...")
    train_ds = get_split("train", dataset=e.DATASET.lower())
    valid_ds = get_split("valid", dataset=e.DATASET.lower())

    if e.__TESTING__:
        train_ds = train_ds[:64]
        valid_ds = valid_ds[:16]
    elif e.NUM_SUBSAMPLE is not None:
        n = e.NUM_SUBSAMPLE
        n_val = max(1, n // 5)
        train_ds = train_ds[:n]
        valid_ds = valid_ds[:n_val]

    e.log(f"Train: {len(train_ds)}, Valid: {len(valid_ds)}")

    e.log("Computing HDC encodings...")
    train_encoded = post_compute_encodings(
        train_ds, hypernet, device=device, batch_size=e.ENCODER_BATCH_SIZE,
    )
    torch.cuda.empty_cache()
    valid_encoded = post_compute_encodings(
        valid_ds, hypernet, device=device, batch_size=e.ENCODER_BATCH_SIZE,
    )
    torch.cuda.empty_cache()
    e.log(f"Encoded: {len(train_encoded)} train, {len(valid_encoded)} valid")

    # Swap edge_terms -> node_terms
    e.log("Swapping edge_terms into node_terms slot for [edge_terms | graph_terms] training")
    for d in train_encoded + valid_encoded:
        d.node_terms = d.edge_terms

    return train_encoded, valid_encoded


experiment.run_if_main()
