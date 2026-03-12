"""
Domain-agnostic base experiment for training a FlowEdgeDecoder.

This experiment defines the full training loop for a discrete flow-matching
edge decoder conditioned on HDC embeddings.  All domain-specific behaviour
(data loading, statistics, reconstruction evaluation) is delegated to hooks
that **must** be overridden by child experiments via
``Experiment.extend()``.

Child experiments are expected to set at minimum:

- ``FEATURE_BINS`` — list of class counts per node feature dimension
- ``NUM_EDGE_CLASSES`` — number of edge types (including "no-edge")

and override the four data hooks.
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from graph_hdc.hypernet.configs import RWConfig
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.hypernet.rrwp_hypernet import RRWPHyperNet
from graph_hdc.models.flow_edge_decoder import (
    FlowEdgeDecoder,
    extend_feature_bins,
)

# =====================================================================
# Parameters — Domain Configuration (MUST be overridden)
# =====================================================================

# :param FEATURE_BINS:
#     List of class counts per node feature dimension.
#     E.g. [17] for color graphs, [9, 6, 3, 4, 2] for ZINC molecules.
FEATURE_BINS: list[int] = []

# :param NUM_EDGE_CLASSES:
#     Number of edge classes including "no-edge".
#     E.g. 2 for color graphs (no-edge, edge), 5 for ZINC.
NUM_EDGE_CLASSES: int = 2

# =====================================================================
# Parameters — HDC Encoder
# =====================================================================

# :param HDC_DIM:
#     Hypervector dimension for the HyperNet encoder.
HDC_DIM: int = 512

# :param HDC_DEPTH:
#     Number of message-passing layers in HyperNet.
HDC_DEPTH: int = 3

# :param USE_RW:
#     Whether to augment node features with random walk return
#     probabilities.
USE_RW: bool = True

# :param RW_K_VALUES:
#     Random walk step counts at which to compute return probabilities.
RW_K_VALUES: tuple = (3, 6)

# :param RW_NUM_BINS:
#     Number of bins for discretising RW return probabilities.
RW_NUM_BINS: int = 4

# :param RW_CLIP_RANGE:
#     When using uniform binning, clip RW values to this range before
#     binning.  Set to None for full [0, 1] range.
RW_CLIP_RANGE: tuple | None = (0, 0.8)

# :param USE_RRWP_HYPERNET:
#     When True, use RRWPHyperNet (split codebook) so that RRWP features
#     only affect order-0 (node_terms), not structural embeddings.
USE_RRWP_HYPERNET: bool = True

# :param NORMALIZE_GRAPH_EMBEDDING:
#     Whether to L2-normalize the graph embedding (order-N) before
#     concatenation into the HDC conditioning vector.
NORMALIZE_GRAPH_EMBEDDING: bool = True

# =====================================================================
# Parameters — Model Architecture
# =====================================================================

# :param N_LAYERS:
#     Number of transformer layers in the FlowEdgeDecoder.
N_LAYERS: int = 6

# :param HIDDEN_DIM:
#     Hidden dimension for transformer layers.
HIDDEN_DIM: int = 256

# :param HIDDEN_MLP_DIM:
#     Hidden dimension for MLP blocks in transformer.
HIDDEN_MLP_DIM: int = 256

# :param N_HEADS:
#     Number of attention heads in transformer layers.
N_HEADS: int = 4

# :param DROPOUT:
#     Dropout probability in transformer layers.
DROPOUT: float = 0.0

# :param CONDITION_DIM:
#     Dimension for HDC conditioning after MLP projection.
CONDITION_DIM: int = 256

# :param TIME_EMBED_DIM:
#     Dimension for sinusoidal time embedding.
TIME_EMBED_DIM: int = 256

# :param NODE_HDC_EMBED_DIM:
#     Dimension for per-node HDC codebook embedding.
#     Set to 0 to disable.
NODE_HDC_EMBED_DIM: int = 0

# :param USE_CROSS_ATTN:
#     Whether to use cross-attention HDC conditioning.
USE_CROSS_ATTN: bool = False

# :param CROSS_ATTN_TOKENS:
#     Number of tokens to decompose the HDC vector into for
#     cross-attention.
CROSS_ATTN_TOKENS: int = 8

# :param CROSS_ATTN_HEADS:
#     Number of attention heads for the cross-attention conditioner.
CROSS_ATTN_HEADS: int = 4

# =====================================================================
# Parameters — Training
# =====================================================================

# :param EPOCHS:
#     Number of training epochs.
EPOCHS: int = 50

# :param BATCH_SIZE:
#     Batch size for training and validation.
BATCH_SIZE: int = 8

# :param LEARNING_RATE:
#     Initial learning rate for Adam optimizer.
LEARNING_RATE: float = 1e-4

# :param WEIGHT_DECAY:
#     Weight decay (L2 regularisation) for optimizer.
WEIGHT_DECAY: float = 1e-5

# :param TRAIN_TIME_DISTORTION:
#     Time distortion type during training ("identity" or "polydec").
TRAIN_TIME_DISTORTION: str = "identity"

# :param GRADIENT_CLIP_VAL:
#     Gradient clipping value.  Set to 0.0 to disable.
GRADIENT_CLIP_VAL: float = 1.0

# :param ACCUMULATE_GRAD_BATCHES:
#     Number of batches to accumulate gradients over.
ACCUMULATE_GRAD_BATCHES: int = 1

# =====================================================================
# Parameters — Sampling / Evaluation
# =====================================================================

# :param SAMPLE_STEPS:
#     Number of denoising steps during sampling.
SAMPLE_STEPS: int = 100

# :param ETA:
#     Stochasticity parameter for sampling (0.0 = deterministic).
ETA: float = 0.0

# :param OMEGA:
#     Target guidance strength parameter.
OMEGA: float = 0.0

# :param SAMPLE_TIME_DISTORTION:
#     Time distortion type during sampling ("identity" or "polydec").
SAMPLE_TIME_DISTORTION: str = "polydec"

# :param EXTRA_FEATURES_TYPE:
#     Type of extra positional features ("rrwp" or "none").
EXTRA_FEATURES_TYPE: str = "rrwp"

# :param RRWP_STEPS:
#     Number of random walk steps for RRWP positional encoding
#     inside the FlowEdgeDecoder.
RRWP_STEPS: int = 10

# :param NOISE_TYPE:
#     Noise distribution type ("uniform" or "marginal").
NOISE_TYPE: str = "uniform"

# =====================================================================
# Parameters — System
# =====================================================================

# :param SEED:
#     Random seed for reproducibility.
SEED: int = 1

# :param NUM_WORKERS:
#     Number of data loader workers.
NUM_WORKERS: int = 0

# :param PRECISION:
#     Training precision ("32", "16", "bf16").
PRECISION: str = "32"

# :param ACCELERATOR:
#     PyTorch Lightning accelerator ("auto", "gpu", "cpu").
ACCELERATOR: str = "auto"

# :param NUM_RECONSTRUCTION_SAMPLES:
#     Number of samples to reconstruct after training.
NUM_RECONSTRUCTION_SAMPLES: int = 100

# :param RECONSTRUCTION_BATCH_SIZE:
#     Batch size for parallel edge generation during reconstruction.
RECONSTRUCTION_BATCH_SIZE: int = 16

# :param __DEBUG__:
#     Debug mode — reuses same output folder during development.
__DEBUG__: bool = True

# :param __TESTING__:
#     Testing mode — runs with minimal iterations for validation.
__TESTING__: bool = False


# =====================================================================
# Experiment
# =====================================================================


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    """Train a FlowEdgeDecoder model."""

    # --- Setup -----------------------------------------------------------

    pl.seed_everything(e.SEED)
    device = torch.device(
        "cuda" if e.ACCELERATOR in ("auto", "gpu") and torch.cuda.is_available()
        else "cpu"
    )
    e.log(f"Device: {device}")

    # Validate that domain parameters are set
    if not e.FEATURE_BINS:
        raise ValueError(
            "FEATURE_BINS is empty.  Set it in a child experiment "
            "(e.g. FEATURE_BINS = [17] for color graphs)."
        )

    # --- RW Config -------------------------------------------------------

    rw_config = RWConfig(enabled=False)
    if e.USE_RW:
        rw_config = RWConfig(
            enabled=True,
            k_values=tuple(e.RW_K_VALUES),
            num_bins=e.RW_NUM_BINS,
            clip_range=tuple(e.RW_CLIP_RANGE) if e.RW_CLIP_RANGE else None,
        )

    base_bins = list(e.FEATURE_BINS)
    full_bins = extend_feature_bins(base_bins, rw_config)
    e.log(f"Base feature bins: {base_bins}")
    e.log(f"Full feature bins (with RW): {full_bins}")

    # --- Create HyperNet Encoder -----------------------------------------

    if e.USE_RRWP_HYPERNET:
        if not rw_config.enabled:
            raise ValueError("USE_RRWP_HYPERNET=True requires USE_RW=True")
        hypernet = RRWPHyperNet(
            feature_bins=base_bins,
            rw_config=rw_config,
            hv_dim=e.HDC_DIM,
            hypernet_depth=e.HDC_DEPTH,
            seed=e.SEED,
            normalize=True,
            normalize_graph_embedding=e.NORMALIZE_GRAPH_EMBEDDING,
            device=str(device),
        )
    else:
        hypernet = HyperNet(
            feature_bins=full_bins,
            hv_dim=e.HDC_DIM,
            hypernet_depth=e.HDC_DEPTH,
            seed=e.SEED,
            normalize=True,
            normalize_graph_embedding=e.NORMALIZE_GRAPH_EMBEDDING,
            device=str(device),
        )

    hypernet = hypernet.to(device)
    hypernet.eval()
    concat_hdc_dim = 2 * hypernet.hv_dim  # order_0 + order_N
    e.log(f"HyperNet hv_dim={hypernet.hv_dim}, concat_hdc_dim={concat_hdc_dim}")

    # Save encoder
    encoder_path = os.path.join(e.path, "encoder.pt")
    hypernet.save(encoder_path)
    e.log(f"Saved encoder to {encoder_path}")

    # --- Data Loading (via hooks) ----------------------------------------

    train_loader, train_data = e.apply_hook(
        "load_train_data",
        hypernet=hypernet,
        rw_config=rw_config,
        feature_bins=full_bins,
        device=device,
    )

    valid_data, valid_loader, vis_samples = e.apply_hook(
        "load_valid_data",
        hypernet=hypernet,
        rw_config=rw_config,
        feature_bins=full_bins,
        device=device,
    )

    edge_marginals, node_counts, max_nodes, size_edge_marginals = e.apply_hook(
        "compute_statistics",
        train_data=train_data,
    )

    e.log(f"Edge marginals: {edge_marginals}")
    e.log(f"Max nodes: {max_nodes}")

    # --- Create Model ----------------------------------------------------

    model = FlowEdgeDecoder(
        feature_bins=full_bins,
        num_edge_classes=e.NUM_EDGE_CLASSES,
        hdc_dim=concat_hdc_dim,
        condition_dim=e.CONDITION_DIM,
        time_embed_dim=e.TIME_EMBED_DIM,
        n_layers=e.N_LAYERS,
        hidden_dim=e.HIDDEN_DIM,
        hidden_mlp_dim=e.HIDDEN_MLP_DIM,
        n_heads=e.N_HEADS,
        dropout=e.DROPOUT,
        lr=e.LEARNING_RATE,
        weight_decay=e.WEIGHT_DECAY,
        noise_type=e.NOISE_TYPE,
        edge_marginals=edge_marginals,
        node_counts=node_counts,
        max_nodes=max_nodes + 10,
        extra_features_type=e.EXTRA_FEATURES_TYPE,
        rrwp_steps=e.RRWP_STEPS,
        size_edge_marginals=size_edge_marginals,
        train_time_distortion=e.TRAIN_TIME_DISTORTION,
        sample_steps=e.SAMPLE_STEPS,
        eta=e.ETA,
        omega=e.OMEGA,
        sample_time_distortion=e.SAMPLE_TIME_DISTORTION,
        use_cross_attn=e.USE_CROSS_ATTN,
        cross_attn_tokens=e.CROSS_ATTN_TOKENS,
        cross_attn_heads=e.CROSS_ATTN_HEADS,
        node_hdc_embed_dim=e.NODE_HDC_EMBED_DIM,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    e.log(f"FlowEdgeDecoder: {num_params:,} trainable parameters")

    # --- Callbacks -------------------------------------------------------

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(e.path, "models"),
            filename="best-{epoch}-{val_loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        ),
        ModelCheckpoint(
            dirpath=os.path.join(e.path, "models"),
            filename="latest",
            every_n_epochs=1,
            save_top_k=1,
        ),
    ]

    callbacks = e.apply_hook(
        "modify_callbacks",
        callbacks=callbacks,
        train_loader=train_loader,
    )

    # --- Train -----------------------------------------------------------

    logger = CSVLogger(save_dir=os.path.join(e.path, "logs"), name="train")
    trainer = Trainer(
        max_epochs=e.EPOCHS,
        accelerator=e.ACCELERATOR,
        devices=1,
        precision=e.PRECISION,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=e.GRADIENT_CLIP_VAL if e.GRADIENT_CLIP_VAL > 0 else None,
        accumulate_grad_batches=e.ACCUMULATE_GRAD_BATCHES,
    )

    e.log("Starting training...")
    try:
        trainer.fit(model, train_loader, valid_loader)
    except KeyboardInterrupt:
        e.log("Training interrupted by user.")

    e.log("Training complete.")

    # --- Reconstruction Evaluation (via hook) ----------------------------

    try:
        e.apply_hook(
            "evaluate_reconstruction",
            model=model,
            hypernet=hypernet,
            rw_config=rw_config,
            feature_bins=full_bins,
            valid_data=valid_data,
            vis_samples=vis_samples,
            device=device,
        )
    except Exception:
        e.log(f"Reconstruction evaluation failed:\n{traceback.format_exc()}")


# =====================================================================
# Hooks — must be overridden by domain-specific child experiments
# =====================================================================


@experiment.hook("load_train_data", default=True)
def load_train_data(e, hypernet, rw_config, feature_bins, device):
    """Return ``(train_loader, train_data_list_or_None)``."""
    raise NotImplementedError(
        "Override 'load_train_data' in a child experiment."
    )


@experiment.hook("load_valid_data", default=True)
def load_valid_data(e, hypernet, rw_config, feature_bins, device):
    """Return ``(valid_data_list, valid_loader, vis_samples_list)``."""
    raise NotImplementedError(
        "Override 'load_valid_data' in a child experiment."
    )


@experiment.hook("compute_statistics", default=True)
def compute_statistics(e, train_data):
    """Return ``(edge_marginals, node_counts, max_nodes, size_edge_marginals)``."""
    raise NotImplementedError(
        "Override 'compute_statistics' in a child experiment."
    )


@experiment.hook("modify_callbacks", default=True)
def modify_callbacks(e, callbacks, train_loader):
    """Optionally modify the callback list.  Default: passthrough."""
    return callbacks


@experiment.hook("evaluate_reconstruction", default=True)
def evaluate_reconstruction(
    e, model, hypernet, rw_config, feature_bins, valid_data, vis_samples, device
):
    """Post-training reconstruction evaluation.  Default: no-op."""
    e.log("No reconstruction evaluation configured (override 'evaluate_reconstruction').")


# =====================================================================
# Testing mode
# =====================================================================


@experiment.testing
def testing(e: Experiment) -> None:
    e.EPOCHS = 2
    e.BATCH_SIZE = 4
    e.HDC_DIM = 128
    e.HDC_DEPTH = 2
    e.N_LAYERS = 2
    e.HIDDEN_DIM = 32
    e.HIDDEN_MLP_DIM = 32
    e.CONDITION_DIM = 32
    e.TIME_EMBED_DIM = 32
    e.SAMPLE_STEPS = 5
    e.NUM_RECONSTRUCTION_SAMPLES = 4
    e.RECONSTRUCTION_BATCH_SIZE = 2


experiment.run_if_main()
