#!/usr/bin/env python
"""
Train FlowEdgeDecoder - Edge-only DeFoG decoder conditioned on HDC vectors.

This experiment trains a discrete flow matching model that generates molecular
edges given:
1. Pre-computed HDC vectors as conditioning
2. Fixed node features (24-dim one-hot: atom type, degree, charge, Hs, ring)

The model learns to predict edges (5 classes: no-edge, single, double, triple,
aromatic) through a denoising process while keeping nodes fixed.

This is the BASE EXPERIMENT. Child experiments can inherit via Experiment.extend()
and override hooks for custom data loading behavior.

Usage:
    # Quick test
    python train_flow_edge_decoder.py --__TESTING__ True

    # Full training with new encoder
    python train_flow_edge_decoder.py --__DEBUG__ False --EPOCHS 100

    # Train with existing encoder checkpoint
    python train_flow_edge_decoder.py --HDC_CONFIG_PATH /path/to/encoder.ckpt
"""

from __future__ import annotations

import os
import random
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from rdkit import Chem
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph_hdc.datasets.utils import get_split
from graph_hdc.hypernet.configs import RWConfig
from graph_hdc.utils.rw_features import get_zinc_rw_boundaries
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.hypernet.multi_hypernet import MultiHyperNet
from graph_hdc.models.flow_edge_decoder import (
    NODE_FEATURE_DIM,
    FlowEdgeDecoder,
    compute_edge_marginals,
    compute_size_edge_marginals,
    compute_node_counts,
    get_node_feature_bins,
    preprocess_dataset,
    node_tuples_to_onehot,
)
from graph_hdc.utils.experiment_helpers import (
    GracefulInterruptHandler,
    LossTrackingCallback,
    ReconstructionVisualizationCallback,
    TrainingMetricsCallback,
    create_reconstruction_plot,
    decode_nodes_from_hdc,
    get_canonical_smiles,
    is_valid_mol,
    load_or_create_encoder,
    pyg_to_mol,
    scrub_smiles,
)

# =============================================================================
# PARAMETERS
# =============================================================================

# :param DATASET:
#     Dataset name for training. Currently only "zinc" is supported.
DATASET: str = "zinc"

# -----------------------------------------------------------------------------
# HDC Encoder Configuration
# -----------------------------------------------------------------------------

# :param HDC_CONFIG_PATH:
#     Path to saved HyperNet encoder checkpoint (.ckpt). If empty, creates a new
#     encoder with HDC_DIM and HDC_DEPTH parameters.
HDC_CONFIG_PATH: str = ""

# :param HDC_DIM:
#     Hypervector dimension for the HyperNet encoder. Only used if HDC_CONFIG_PATH
#     is empty. Typical values: 256, 512, 1024.
HDC_DIM: int = 1024

# :param HDC_DEPTH:
#     Message passing depth for the HyperNet encoder. Only used if HDC_CONFIG_PATH
#     is empty. Higher values capture longer-range structural information.
HDC_DEPTH: int = 6

# :param USE_RW:
#     Whether to augment HDC node features with random walk return probabilities.
#     When True, each node's feature tuple is extended with binned RW return
#     probabilities at each step in RW_K_VALUES, making the HDC conditioning
#     vector more expressive about global graph topology. When USE_RRWP_HYPERNET
#     is also True, the FlowEdgeDecoder's one-hot node features are extended
#     with the additional RW bins (e.g. 24-dim → 32-dim for 2 k-values × 4 bins).
USE_RW: bool = True

# :param RW_K_VALUES:
#     Random walk steps at which to compute return probabilities. Only used
#     when USE_RW is True.
RW_K_VALUES: tuple = (3, 6)

# :param RW_NUM_BINS:
#     Number of bins for discretising RW return probabilities. When
#     USE_QUANTILE_BINS is True, must be one of {3, 4, 5, 6} (precomputed).
#     Only used when USE_RW is True.
RW_NUM_BINS: int = 4

# :param USE_QUANTILE_BINS:
#     Whether to use precomputed quantile-based bin boundaries for RW features
#     instead of uniform bins on [0,1]. Quantile binning distributes atoms
#     equally across bins for each k value, avoiding the near-degenerate
#     distributions that uniform binning produces at higher k (e.g. k=10 puts
#     81% of atoms into bin 0 with uniform bins). Requires RW_NUM_BINS in
#     {3, 4, 5, 6}. Only used when USE_RW is True.
USE_QUANTILE_BINS: bool = False

# :param RW_CLIP_RANGE:
#     When set to a (lo, hi) tuple and USE_QUANTILE_BINS is False, uniform bins
#     are placed over [lo, hi] instead of [0, 1]. Values outside this range are
#     clamped to the first / last bin. This concentrates bin resolution in the
#     region where most RW return probabilities fall. Ignored when
#     USE_QUANTILE_BINS is True. Only used when USE_RW is True.
RW_CLIP_RANGE: tuple | None = (0, 0.8)

# :param PRUNE_CODEBOOK:
#     Whether to prune the HDC codebook to only feature tuples observed in the
#     dataset. When True (default), unseen tuples cause encoding errors — set to
#     False when training on generated molecules (e.g. streaming fragments) whose
#     topology may produce novel feature combinations.
PRUNE_CODEBOOK: bool = True

# :param USE_RRWP_HYPERNET:
#     When True (requires USE_RW=True), creates RRWPHyperNet which uses
#     RRWP-enriched features only for order-0 (node_terms) readout, while
#     message passing operates on base features only. This prevents positional
#     information from interfering with structural binding. Works with both
#     single HyperNet and MultiHyperNet ensembles.
USE_RRWP_HYPERNET: bool = True

# :param NORMALIZE_GRAPH_EMBEDDING:
#     Whether to L2-normalize the graph embedding (order-N) output of each
#     HyperNet before concatenation into the HDC conditioning vector. Without
#     normalization, the embedding magnitude scales with ~sqrt(num_atoms),
#     causing the decoder's conditioning signal to be systematically weaker
#     for small molecules.
NORMALIZE_GRAPH_EMBEDDING: bool = True

# :param ENSEMBLE_CONFIGS:
#     Optional list of (hv_dim, depth) tuples for a MultiHyperNet ensemble.
#     Each tuple creates an independently-initialized HyperNet with its own
#     random codebook, providing a different "perspective" on the same graph.
#     Seeds are auto-generated as SEED+0, SEED+1, etc.
#     When empty/None, a single HyperNet with HDC_DIM/HDC_DEPTH is used instead.
#     Example: [(256, 6), (512, 4), (256, 8)] creates 3 HyperNets.
ENSEMBLE_CONFIGS: Optional[List[Tuple[int, int]]] = None

# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------

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
#     Dimension for per-node HDC codebook embedding. When >0, each node's
#     codebook hypervector is projected to this dimension and concatenated
#     with the one-hot node features. Set to 0 to disable.
NODE_HDC_EMBED_DIM: int = 64

# :param USE_CROSS_ATTN:
#     Whether to use cross-attention HDC conditioning. When True, the raw HDC
#     vector is decomposed into learnable tokens, nodes cross-attend to these
#     tokens, and node-pair combinations produce edge-specific conditioning
#     signals — supplementing the broadcast FiLM conditioning.
USE_CROSS_ATTN: bool = False

# :param CROSS_ATTN_TOKENS:
#     Number of tokens to decompose the HDC vector into for cross-attention.
#     Only used when USE_CROSS_ATTN is True.
CROSS_ATTN_TOKENS: int = 8

# :param CROSS_ATTN_HEADS:
#     Number of attention heads for the cross-attention conditioner.
#     Only used when USE_CROSS_ATTN is True.
CROSS_ATTN_HEADS: int = 4

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------

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
#     Weight decay (L2 regularization) for optimizer.
WEIGHT_DECAY: float = 1e-5

# :param TRAIN_TIME_DISTORTION:
#     Time distortion type during training. Options: "identity", "polydec".
TRAIN_TIME_DISTORTION: str = "identity"

# :param GRADIENT_CLIP_VAL:
#     Gradient clipping value. Set to 0.0 to disable.
GRADIENT_CLIP_VAL: float = 1.0

# :param ACCUMULATE_GRAD_BATCHES:
#     Number of batches to accumulate gradients over before performing an
#     optimizer step. Effective batch size becomes BATCH_SIZE * ACCUMULATE_GRAD_BATCHES.
ACCUMULATE_GRAD_BATCHES: int = 1

# -----------------------------------------------------------------------------
# Sampling Configuration
# -----------------------------------------------------------------------------

# :param SAMPLE_STEPS:
#     Number of denoising steps during sampling.
SAMPLE_STEPS: int = 100

# :param ETA:
#     Stochasticity parameter for sampling. 0.0 = deterministic.
ETA: float = 0.0

# :param OMEGA:
#     Target guidance strength parameter.
OMEGA: float = 0.0

# :param SAMPLE_TIME_DISTORTION:
#     Time distortion type during sampling. Options: "identity", "polydec".
SAMPLE_TIME_DISTORTION: str = "polydec"

# -----------------------------------------------------------------------------
# Extra Features
# -----------------------------------------------------------------------------

# :param EXTRA_FEATURES_TYPE:
#     Type of extra positional features. Options: "rrwp", "none".
EXTRA_FEATURES_TYPE: str = "rrwp"

# :param RRWP_STEPS:
#     Number of random walk steps for RRWP positional encoding.
RRWP_STEPS: int = 10

# -----------------------------------------------------------------------------
# Noise Distribution
# -----------------------------------------------------------------------------

# :param NOISE_TYPE:
#     Noise distribution type. Options: "uniform", "marginal".
NOISE_TYPE: str = "uniform"

# -----------------------------------------------------------------------------
# System Configuration
# -----------------------------------------------------------------------------

# :param SEED:
#     Random seed for reproducibility.
SEED: int = 1

# :param NUM_WORKERS:
#     Number of data loader workers. Set to 0 to avoid multiprocessing issues.
NUM_WORKERS: int = 0

# :param PRECISION:
#     Training precision. Options: "32", "16", "bf16".
PRECISION: str = "32"

# :param ACCELERATOR:
#     PyTorch Lightning accelerator. Options: "auto", "gpu", "cpu".
#     Use "gpu" to force GPU training, "cpu" to force CPU training.
ACCELERATOR: str = "auto"

# -----------------------------------------------------------------------------
# Reconstruction Evaluation
# -----------------------------------------------------------------------------

# :param NUM_RECONSTRUCTION_SAMPLES:
#     Number of molecules to reconstruct and visualize after training.
NUM_RECONSTRUCTION_SAMPLES: int = 100

# :param RECONSTRUCTION_BATCH_SIZE:
#     Batch size for parallel edge generation during reconstruction evaluation.
#     Higher values are faster but use more GPU memory.
RECONSTRUCTION_BATCH_SIZE: int = 16

# :param NUM_VALIDATION_VISUALIZATIONS:
#     Number of molecules to visualize during each validation epoch.
NUM_VALIDATION_VISUALIZATIONS: int = 10

# :param NUM_VALIDATION_REPETITIONS:
#     Number of parallel decodings per molecule during validation visualization.
#     Each molecule is decoded this many times in a single batched call, and
#     the result with the lowest HDC cosine distance is kept (best-of-N).
NUM_VALIDATION_REPETITIONS: int = 8

# -----------------------------------------------------------------------------
# Resume Training
# -----------------------------------------------------------------------------

# :param RESUME_CHECKPOINT_PATH:
#     Path to PyTorch Lightning checkpoint (.ckpt) to resume training from.
#     If set, RESUME_ENCODER_PATH must also be provided.
RESUME_CHECKPOINT_PATH: Optional[str] = None

# :param RESUME_ENCODER_PATH:
#     Path to HyperNet encoder checkpoint when resuming training.
#     Required if RESUME_CHECKPOINT_PATH is set.
RESUME_ENCODER_PATH: Optional[str] = None

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
# EXPERIMENT
# =============================================================================


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    """Train FlowEdgeDecoder model."""

    # =========================================================================
    # Setup
    # =========================================================================

    # Fix for PyTorch Lightning's _atomic_save on systems with limited tmpfs
    custom_tmpdir = Path(e.path) / ".tmp_checkpoints"
    custom_tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(custom_tmpdir)
    tempfile.tempdir = str(custom_tmpdir)

    # Set seed
    pl.seed_everything(e.SEED)

    # Validate resume configuration
    resuming = bool(e.RESUME_CHECKPOINT_PATH)
    if resuming:
        resume_ckpt_path = Path(e.RESUME_CHECKPOINT_PATH)
        if not resume_ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt_path}")

        if not e.RESUME_ENCODER_PATH:
            raise ValueError("RESUME_ENCODER_PATH is required when RESUME_CHECKPOINT_PATH is set")

        resume_encoder_path = Path(e.RESUME_ENCODER_PATH)
        if not resume_encoder_path.exists():
            raise FileNotFoundError(f"Resume encoder not found: {resume_encoder_path}")

    e.log("=" * 60)
    if resuming:
        e.log("FlowEdgeDecoder Training (RESUMING from checkpoint)")
    else:
        e.log("FlowEdgeDecoder Training")
    use_ensemble = bool(e.ENSEMBLE_CONFIGS)

    # Build RW config from parameters
    rw_bin_boundaries = None
    if e.USE_RW and e.USE_QUANTILE_BINS:
        rw_bin_boundaries = get_zinc_rw_boundaries(e.RW_NUM_BINS)

    rw_config = RWConfig(
        enabled=e.USE_RW,
        k_values=e.RW_K_VALUES,
        num_bins=e.RW_NUM_BINS,
        bin_boundaries=rw_bin_boundaries,
        clip_range=None if rw_bin_boundaries else e.RW_CLIP_RANGE,
    )

    if e.USE_RRWP_HYPERNET and not e.USE_RW:
        raise ValueError("USE_RRWP_HYPERNET requires USE_RW=True")

    e.log("=" * 60)
    e.log(f"Dataset: {e.DATASET.upper()}")
    e.log(f"HDC config path: {e.HDC_CONFIG_PATH or '(creating new)'}")
    if use_ensemble:
        e.log(f"Ensemble: {len(e.ENSEMBLE_CONFIGS)} HyperNets, configs={e.ENSEMBLE_CONFIGS}")
    else:
        e.log(f"HDC dim: {e.HDC_DIM}, depth: {e.HDC_DEPTH}")
    if rw_config.enabled:
        bin_mode = "quantile" if rw_config.bin_boundaries else ("clipped" if rw_config.clip_range else "uniform")
        e.log(f"RW features: k={rw_config.k_values}, bins={rw_config.num_bins} ({bin_mode})")
    if e.USE_RRWP_HYPERNET:
        e.log(f"RRWP HyperNet: enabled (split order-0 encoding)")
    e.log(f"Codebook pruning: {e.PRUNE_CODEBOOK}")
    e.log(f"Architecture: {e.N_LAYERS} layers, {e.HIDDEN_DIM} hidden dim")
    e.log(f"Training: {e.EPOCHS} epochs, batch size {e.BATCH_SIZE}")
    e.log(f"Debug mode: {e.__DEBUG__}")
    if resuming:
        e.log(f"Resuming from: {e.RESUME_CHECKPOINT_PATH}")
        e.log(f"Using encoder: {e.RESUME_ENCODER_PATH}")
    e.log("=" * 60)

    # Store config
    e["config/dataset"] = e.DATASET
    e["config/hdc_config_path"] = e.HDC_CONFIG_PATH
    e["config/hdc_dim"] = e.HDC_DIM
    e["config/hdc_depth"] = e.HDC_DEPTH
    e["config/ensemble_configs"] = e.ENSEMBLE_CONFIGS
    e["config/use_ensemble"] = use_ensemble
    e["config/use_rw"] = e.USE_RW
    e["config/rw_k_values"] = list(e.RW_K_VALUES)
    e["config/rw_num_bins"] = e.RW_NUM_BINS
    e["config/rw_quantile_bins"] = e.USE_QUANTILE_BINS
    e["config/rw_clip_range"] = e.RW_CLIP_RANGE
    e["config/prune_codebook"] = e.PRUNE_CODEBOOK
    e["config/use_rrwp_hypernet"] = e.USE_RRWP_HYPERNET
    e["config/n_layers"] = e.N_LAYERS
    e["config/hidden_dim"] = e.HIDDEN_DIM
    e["config/epochs"] = e.EPOCHS
    e["config/batch_size"] = e.BATCH_SIZE
    e["config/lr"] = e.LEARNING_RATE
    e["config/resuming"] = resuming

    # Device
    if e.ACCELERATOR == "gpu":
        device = torch.device("cuda")
    elif e.ACCELERATOR == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e.log(f"Using device: {device} (accelerator={e.ACCELERATOR})")
    e["config/device"] = str(device)

    # =========================================================================
    # Load or Create HyperNet Encoder
    # =========================================================================

    e.log("\nLoading/Creating HyperNet encoder...")

    hdc_device = torch.device("cpu")

    if resuming:
        # When resuming, load saved encoder (auto-detects type)
        from graph_hdc.hypernet import load_hypernet
        resume_encoder_file = Path(e.RESUME_ENCODER_PATH)
        hypernet = load_hypernet(str(resume_encoder_file), device=str(hdc_device))
        hypernet.eval()
    elif use_ensemble:
        hn_cls = None
        if e.USE_RRWP_HYPERNET:
            from graph_hdc.hypernet.rrwp_hypernet import RRWPHyperNet
            hn_cls = RRWPHyperNet

        if rw_config.enabled:
            from graph_hdc.hypernet.configs import create_config_with_rw
            from graph_hdc.datasets.utils import scan_node_features_with_rw
            e.log("Scanning dataset for observed RW-augmented node features (ensemble)...")
            observed = scan_node_features_with_rw(e.DATASET.lower(), rw_config)
            e.log(f"Found {len(observed)} unique node feature tuples with RW")
            base_config = create_config_with_rw(
                base_dataset=e.DATASET.lower(),
                hv_dim=e.ENSEMBLE_CONFIGS[0][0],
                rw_config=rw_config,
                hypernet_depth=e.ENSEMBLE_CONFIGS[0][1],
                prune_codebook=e.PRUNE_CODEBOOK,
                normalize_graph_embedding=e.NORMALIZE_GRAPH_EMBEDDING,
                device=str(hdc_device),
            )
            hypernet = MultiHyperNet.from_dim_depth_pairs(
                base_config=base_config,
                dim_depth_pairs=e.ENSEMBLE_CONFIGS,
                base_seed=e.SEED,
                observed_node_features=observed if e.PRUNE_CODEBOOK else None,
                hypernet_cls=hn_cls,
            )
        else:
            from graph_hdc.utils.experiment_helpers import create_hdc_config
            base_config = create_hdc_config(
                dataset=e.DATASET,
                hv_dim=e.ENSEMBLE_CONFIGS[0][0],
                depth=e.ENSEMBLE_CONFIGS[0][1],
                device=str(hdc_device),
            )
            base_config.normalize_graph_embedding = e.NORMALIZE_GRAPH_EMBEDDING
            hypernet = MultiHyperNet.from_dim_depth_pairs(
                base_config=base_config,
                dim_depth_pairs=e.ENSEMBLE_CONFIGS,
                base_seed=e.SEED,
            )
        hypernet = hypernet.to(hdc_device)
        hypernet.eval()
    else:
        hypernet = load_or_create_encoder(
            config_path=e.HDC_CONFIG_PATH,
            dataset=e.DATASET,
            hv_dim=e.HDC_DIM,
            depth=e.HDC_DEPTH,
            device=hdc_device,
            rw_config=rw_config,
            prune_codebook=e.PRUNE_CODEBOOK,
            use_rrwp_hypernet=e.USE_RRWP_HYPERNET,
            normalize_graph_embedding=e.NORMALIZE_GRAPH_EMBEDDING,
        )

    # Compute dimensions
    if isinstance(hypernet, MultiHyperNet):
        primary_hdc_dim = hypernet.hv_dim  # primary's dim, used for order_0
        ensemble_graph_dim = hypernet.ensemble_graph_dim  # sum of all dims for order_N
        e.log(f"MultiHyperNet initialized: K={hypernet.num_hypernets}, "
              f"primary_dim={primary_hdc_dim}, ensemble_graph_dim={ensemble_graph_dim}")
        e.log(f"  Per-HyperNet: {[(hn.hv_dim, hn.depth, hn.seed) for hn in hypernet._hypernets]}")
        e["config/actual_hdc_dim"] = primary_hdc_dim
        e["config/ensemble_graph_dim"] = ensemble_graph_dim
        actual_hdc_dim = primary_hdc_dim
    else:
        actual_hdc_dim = hypernet.hv_dim
        ensemble_graph_dim = actual_hdc_dim  # single HyperNet: order_N dim == base dim
        e.log(f"HyperNet initialized: hv_dim={actual_hdc_dim}, depth={hypernet.depth}")
        e["config/actual_hdc_dim"] = actual_hdc_dim
        e["config/actual_hdc_depth"] = hypernet.depth

    # Compute node feature bins (includes RW bins when enabled)
    feature_bins = get_node_feature_bins(hypernet.rw_config)
    num_node_classes = sum(feature_bins)
    e.log(f"Node feature bins: {feature_bins} ({num_node_classes}-dim)")
    e["config/node_feature_bins"] = feature_bins
    e["config/num_node_classes"] = num_node_classes

    # Save the encoder if we created a new one
    if not e.HDC_CONFIG_PATH and not resuming:
        encoder_path = Path(e.path) / "hypernet_encoder.ckpt"
        hypernet.save(encoder_path)
        e.log(f"Saved new encoder to: {encoder_path}")
        e["results/encoder_path"] = str(encoder_path)

    # =========================================================================
    # Apply Hooks for Data Loading
    # =========================================================================

    train_loader, train_data = e.apply_hook(
        "load_train_data",
        hypernet=hypernet,
        device=hdc_device,
    )

    valid_data, valid_loader, vis_samples, vis_samples_small = e.apply_hook(
        "load_valid_data",
        hypernet=hypernet,
        device=hdc_device,
    )

    edge_marginals, node_counts, max_nodes, size_edge_marginals = e.apply_hook(
        "compute_statistics",
        train_data=train_data,
        hypernet=hypernet,
        device=device,
    )

    # Store statistics
    e["data/edge_marginals"] = edge_marginals.tolist()
    e["data/max_nodes"] = max_nodes
    e["data/train_size"] = len(train_data) if train_data else "streaming"
    e["data/valid_size"] = len(valid_data)

    # =========================================================================
    # Create Model
    # =========================================================================

    e.log("\nCreating FlowEdgeDecoder model...")

    # HDC vector dimension = order_0 (primary dim) + order_N (ensemble dim)
    # For single HyperNet: ensemble_graph_dim == actual_hdc_dim, so this is 2*dim
    # For ensemble: ensemble_graph_dim == sum of all sub-HyperNet dims
    concat_hdc_dim = actual_hdc_dim + ensemble_graph_dim
    e.log(f"Concatenated HDC dim: {concat_hdc_dim} (order_0={actual_hdc_dim} + order_N={ensemble_graph_dim})")
    e.log(f"Condition dim after MLP: {e.CONDITION_DIM}")
    e.log(f"Time embedding dim: {e.TIME_EMBED_DIM}")
    e["config/concat_hdc_dim"] = concat_hdc_dim
    e["config/condition_dim"] = e.CONDITION_DIM
    e["config/time_embed_dim"] = e.TIME_EMBED_DIM
    e["config/use_cross_attn"] = e.USE_CROSS_ATTN
    e["config/cross_attn_tokens"] = e.CROSS_ATTN_TOKENS
    e["config/cross_attn_heads"] = e.CROSS_ATTN_HEADS
    e["config/node_hdc_embed_dim"] = e.NODE_HDC_EMBED_DIM

    model = FlowEdgeDecoder(
        num_node_classes=num_node_classes,  # dynamic: base 24 + optional RW bins
        num_edge_classes=5,
        hdc_dim=concat_hdc_dim,  # Input dim is 2x base (concatenated)
        condition_dim=e.CONDITION_DIM,  # Reduced dim after MLP
        time_embed_dim=e.TIME_EMBED_DIM,  # Dimension for time embedding
        n_layers=e.N_LAYERS,
        hidden_dim=e.HIDDEN_DIM,
        hidden_mlp_dim=e.HIDDEN_MLP_DIM,
        n_heads=e.N_HEADS,
        dropout=e.DROPOUT,
        noise_type=e.NOISE_TYPE,
        edge_marginals=edge_marginals,
        node_counts=node_counts,
        max_nodes=max_nodes + 10,  # Buffer
        extra_features_type=e.EXTRA_FEATURES_TYPE,
        rrwp_steps=e.RRWP_STEPS,
        lr=e.LEARNING_RATE,
        weight_decay=e.WEIGHT_DECAY,
        train_time_distortion=e.TRAIN_TIME_DISTORTION,
        sample_steps=e.SAMPLE_STEPS,
        eta=e.ETA,
        omega=e.OMEGA,
        sample_time_distortion=e.SAMPLE_TIME_DISTORTION,
        use_cross_attn=e.USE_CROSS_ATTN,
        cross_attn_tokens=e.CROSS_ATTN_TOKENS,
        cross_attn_heads=e.CROSS_ATTN_HEADS,
        node_hdc_embed_dim=e.NODE_HDC_EMBED_DIM,
        nodes_codebook=hypernet.nodes_codebook.clone() if e.NODE_HDC_EMBED_DIM > 0 else None,
        size_edge_marginals=size_edge_marginals,
    )

    e.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    e["model/num_parameters"] = sum(p.numel() for p in model.parameters())

    # =========================================================================
    # Create Standard Callbacks
    # =========================================================================

    # Checkpoint directory for latest checkpoint
    checkpoint_dir = Path(e.path) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Best model checkpoint (based on validation loss)
        ModelCheckpoint(
            dirpath=e.path,
            filename="best-{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        # Save checkpoint after every epoch (overwrites previous)
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="latest",
            save_top_k=1,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            enable_version_counter=False,
        ),
        LossTrackingCallback(experiment=e),
        ReconstructionVisualizationCallback(
            experiment=e,
            vis_samples=vis_samples,
            sample_steps=e.SAMPLE_STEPS,
            eta=e.ETA,
            omega=e.OMEGA,
            time_distortion=e.SAMPLE_TIME_DISTORTION,
            hypernet=hypernet,
            num_repetitions=e.NUM_VALIDATION_REPETITIONS,
            dataset=e.DATASET,
        ),
        *([ReconstructionVisualizationCallback(
            experiment=e,
            vis_samples=vis_samples_small,
            sample_steps=e.SAMPLE_STEPS,
            eta=e.ETA,
            omega=e.OMEGA,
            time_distortion=e.SAMPLE_TIME_DISTORTION,
            hypernet=hypernet,
            num_repetitions=e.NUM_VALIDATION_REPETITIONS,
            dataset=e.DATASET,
            label="small_molecules",
        )] if vis_samples_small else []),
        TrainingMetricsCallback(
            experiment=e,
            num_timestep_bins=10,
            num_edge_classes=5,
        ),
    ]

    # Allow child experiments to modify callbacks (add, remove, or change)
    callbacks = e.apply_hook(
        "modify_callbacks",
        callbacks=callbacks,
        train_loader=train_loader,
    )

    # =========================================================================
    # Setup Training
    # =========================================================================

    e.log("\nSetting up training...")

    # Logger
    logger = CSVLogger(e.path, name="logs")

    # Trainer
    trainer = Trainer(
        max_epochs=e.EPOCHS,
        accelerator=e.ACCELERATOR,
        devices=1,
        precision=e.PRECISION,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=e.path,
        log_every_n_steps=10,
        gradient_clip_val=e.GRADIENT_CLIP_VAL,
        accumulate_grad_batches=e.ACCUMULATE_GRAD_BATCHES,
        enable_progress_bar=True,
    )

    # =========================================================================
    # Train with Graceful Interrupt Handling
    # =========================================================================

    e.log("\nStarting training...")
    if resuming:
        e.log(f"Resuming from checkpoint: {resume_ckpt_path}")
    e.log("(Press CTRL+C to gracefully stop)")
    e.log("-" * 40)

    interrupted = False
    with GracefulInterruptHandler() as handler:
        handler.set_trainer(trainer)
        try:
            trainer.fit(
                model, train_loader, valid_loader,
                ckpt_path=str(resume_ckpt_path) if resuming else None,
            )
        except KeyboardInterrupt:
            interrupted = True
            e.log("\nTraining interrupted by user (force quit)")

    if trainer.should_stop and not interrupted:
        e.log("-" * 40)
        e.log("Training gracefully stopped by user")
    elif not interrupted:
        e.log("-" * 40)
        e.log("Training complete!")

    # =========================================================================
    # Save Results
    # =========================================================================

    # Get best metrics
    best_val_loss = trainer.callback_metrics.get("val/loss")
    if best_val_loss is not None:
        e["results/best_val_loss"] = float(best_val_loss)
        e.log(f"Best validation loss: {best_val_loss:.4f}")

    # Save final model
    final_path = Path(e.path) / "final_model.ckpt"
    model.save(str(final_path))
    e.log(f"Model saved to: {final_path}")
    e["results/model_path"] = str(final_path)

    # Find best checkpoint
    best_ckpts = list(Path(e.path).glob("best-*.ckpt"))
    if best_ckpts:
        best_ckpt = best_ckpts[0]
        e.log(f"Best checkpoint: {best_ckpt}")
        e["results/best_checkpoint"] = str(best_ckpt)

    # =========================================================================
    # Reconstruction Evaluation
    # =========================================================================

    e.log("\n" + "=" * 60)
    e.log("Reconstruction Evaluation")
    e.log("=" * 60)

    try:
        # Use final model for reconstruction
        recon_model = model
        recon_model.eval()
        recon_model.to(device)

        # Get raw validation dataset (with SMILES)
        raw_valid_ds = get_split("valid", dataset=e.DATASET.lower())

        # Select random samples
        num_samples = min(e.NUM_RECONSTRUCTION_SAMPLES, len(raw_valid_ds))
        sample_indices = random.sample(range(len(raw_valid_ds)), num_samples)

        e.log(f"Evaluating reconstruction on {num_samples} samples...")

        # Create output directory for plots
        recon_dir = Path(e.path) / "reconstruction_plots"
        recon_dir.mkdir(exist_ok=True)

        # Track metrics
        valid_count = 0
        match_count = 0
        results = []

        # Start timing
        sampling_start_time = time.time()

        # ── Phase 1: Encode all molecules and decode nodes (sequential) ──
        prepared = []  # list of dicts with encoding results
        for idx, sample_idx in enumerate(sample_indices):
            original_data = raw_valid_ds[sample_idx]
            raw_smiles = original_data.smiles if hasattr(original_data, "smiles") else "N/A"
            original_smiles = scrub_smiles(raw_smiles) if raw_smiles != "N/A" else "N/A"
            if original_smiles is None:
                original_smiles = raw_smiles
            original_mol = Chem.MolFromSmiles(original_smiles) if original_smiles != "N/A" else None

            # Encode with HyperNet (on CPU)
            data_for_encoding = original_data.clone()
            data_for_encoding = data_for_encoding.to(hdc_device)
            if not hasattr(data_for_encoding, "batch") or data_for_encoding.batch is None:
                data_for_encoding.batch = torch.zeros(
                    data_for_encoding.x.size(0), dtype=torch.long, device=hdc_device
                )

            with torch.no_grad():
                # Augment with RW features if the encoder expects them
                if hasattr(hypernet, "rw_config") and hypernet.rw_config.enabled:
                    from graph_hdc.utils.rw_features import augment_data_with_rw
                    data_for_encoding = augment_data_with_rw(
                        data_for_encoding,
                        k_values=hypernet.rw_config.k_values,
                        num_bins=hypernet.rw_config.num_bins,
                        bin_boundaries=hypernet.rw_config.bin_boundaries,
                        clip_range=hypernet.rw_config.clip_range,
                    )

                # forward() handles encode_properties internally and returns
                # the correct node_terms for both HyperNet and RRWPHyperNet
                encoder_output = hypernet.forward(data_for_encoding, normalize=True)
                order_zero = encoder_output["node_terms"]
                order_n = encoder_output["graph_embedding"]
                graph_embedding = torch.cat([order_zero, order_n], dim=-1).squeeze(0)

            node_tuples, num_nodes = decode_nodes_from_hdc(
                hypernet, graph_embedding.unsqueeze(0), actual_hdc_dim
            )

            if num_nodes == 0:
                e.log(f"\nSample {idx + 1}/{num_samples}: {original_smiles}")
                e.log("  WARNING: No nodes decoded, skipping...")
                results.append({
                    "sample_idx": sample_idx,
                    "original_smiles": original_smiles,
                    "generated_smiles": None,
                    "is_valid": False,
                    "is_match": False,
                    "error": "No nodes decoded",
                })
                continue

            node_features = node_tuples_to_onehot(node_tuples, device=device, feature_bins=feature_bins)
            prepared.append({
                "idx": idx,
                "sample_idx": sample_idx,
                "original_smiles": original_smiles,
                "original_mol": original_mol,
                "graph_embedding": graph_embedding,
                "node_features": node_features,
                "num_nodes": num_nodes,
                "node_tuples": node_tuples,
            })

        # ── Phase 2: Batch edge generation ──
        recon_batch_size = e.RECONSTRUCTION_BATCH_SIZE
        e.log(f"\nGenerating edges for {len(prepared)} molecules "
              f"(batch_size={recon_batch_size}, {e.SAMPLE_STEPS} steps)...")

        generated_results = [None] * len(prepared)
        for batch_start in range(0, len(prepared), recon_batch_size):
            batch_items = prepared[batch_start:batch_start + recon_batch_size]
            bs = len(batch_items)
            max_nodes = max(item["num_nodes"] for item in batch_items)
            node_feature_dim = batch_items[0]["node_features"].size(-1)
            hdc_dim = batch_items[0]["graph_embedding"].size(-1)

            batch_hdc = torch.stack(
                [item["graph_embedding"] for item in batch_items], dim=0
            ).to(device)
            batch_node_features = torch.zeros(
                bs, max_nodes, node_feature_dim, device=device
            )
            batch_node_mask = torch.zeros(
                bs, max_nodes, dtype=torch.bool, device=device
            )
            for i, item in enumerate(batch_items):
                n = item["num_nodes"]
                batch_node_features[i, :n] = item["node_features"].to(device)
                batch_node_mask[i, :n] = True

            with torch.no_grad():
                batch_samples = recon_model.sample(
                    hdc_vectors=batch_hdc,
                    node_features=batch_node_features,
                    node_mask=batch_node_mask,
                    sample_steps=e.SAMPLE_STEPS,
                    show_progress=False,
                    device=device,
                )

            for i, item in enumerate(batch_items):
                generated_results[batch_start + i] = batch_samples[i]

            e.log(f"  Batch {batch_start // recon_batch_size + 1}/"
                  f"{(len(prepared) + recon_batch_size - 1) // recon_batch_size}: "
                  f"generated {bs} molecules")

        # ── Phase 3: Evaluate results ──
        for i, item in enumerate(prepared):
            idx = item["idx"]
            sample_idx = item["sample_idx"]
            original_smiles = item["original_smiles"]
            original_mol = item["original_mol"]

            generated_data = generated_results[i]
            generated_mol = pyg_to_mol(generated_data)
            generated_smiles = get_canonical_smiles(generated_mol)

            e.log(f"\nSample {idx + 1}/{num_samples}:")
            e.log(f"  Original:  {original_smiles}")
            e.log(f"  Generated: {generated_smiles or 'N/A'}")

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
            if is_match:
                match_count += 1

            status = "MATCH" if is_match else ("Valid" if is_valid else "Invalid")
            e.log(f"  Status: {status}")

            if original_mol is not None:
                plot_path = recon_dir / f"reconstruction_{idx + 1:03d}.png"
                create_reconstruction_plot(
                    original_mol=original_mol,
                    generated_mol=generated_mol,
                    original_smiles=original_smiles,
                    generated_smiles=generated_smiles or "N/A",
                    is_valid=is_valid,
                    is_match=is_match,
                    sample_idx=idx,
                    save_path=plot_path,
                )

            results.append({
                "sample_idx": sample_idx,
                "original_smiles": original_smiles,
                "generated_smiles": generated_smiles,
                "is_valid": is_valid,
                "is_match": is_match,
            })

        # End timing
        total_sampling_time = time.time() - sampling_start_time

        # Log summary
        e.log("\n" + "-" * 40)
        e.log("Reconstruction Summary:")
        e.log(f"  Total samples: {num_samples}")
        e.log(f"  Valid molecules: {valid_count} ({100 * valid_count / num_samples:.1f}%)")
        e.log(f"  Exact matches: {match_count} ({100 * match_count / num_samples:.1f}%)")
        e.log(f"  Total sampling time: {total_sampling_time:.2f} seconds")
        e.log(f"  Average time per sample: {total_sampling_time / num_samples:.2f} seconds")
        e.log("-" * 40)

        # Store metrics
        e["reconstruction/num_samples"] = num_samples
        e["reconstruction/valid_count"] = valid_count
        e["reconstruction/match_count"] = match_count
        e["reconstruction/valid_rate"] = valid_count / num_samples if num_samples > 0 else 0
        e["reconstruction/match_rate"] = match_count / num_samples if num_samples > 0 else 0
        e["reconstruction/total_sampling_time_seconds"] = total_sampling_time
        e["reconstruction/avg_time_per_sample_seconds"] = total_sampling_time / num_samples if num_samples > 0 else 0
        e["reconstruction/results"] = results

        # Save results as JSON
        e.commit_json("reconstruction_results.json", {
            "num_samples": num_samples,
            "valid_count": valid_count,
            "match_count": match_count,
            "valid_rate": valid_count / num_samples if num_samples > 0 else 0,
            "match_rate": match_count / num_samples if num_samples > 0 else 0,
            "results": results,
        })

    except Exception as ex:
        e.log(f"Reconstruction evaluation failed: {ex}")
        import traceback
        e.log(traceback.format_exc())
        e["reconstruction/status"] = "failed"
        e["reconstruction/error"] = str(ex)

    e.log("\n" + "=" * 60)
    e.log("Experiment completed!")
    e.log("=" * 60)


# =============================================================================
# HOOKS
# =============================================================================


@experiment.hook("load_train_data", default=True)
def load_train_data(
    e: Experiment,
    hypernet: HyperNet,
    device: torch.device,
) -> Tuple[DataLoader, List[Data]]:
    """
    Load and preprocess training data.

    Default implementation loads ZINC training split and preprocesses with HyperNet.
    Child experiments can override this hook to provide custom data sources.

    Args:
        e: Experiment instance for accessing parameters
        hypernet: HyperNet encoder for preprocessing
        device: Device for HDC computation

    Returns:
        Tuple of (train_loader, train_data list)
    """
    e.log("\nLoading training data...")
    train_ds = get_split("train", dataset=e.DATASET.lower())
    e.log(f"Raw train: {len(train_ds)}")

    e.log("Preprocessing training data...")
    e.log("(Computing HDC embeddings and converting features...)")
    train_data = preprocess_dataset(
        train_ds,
        hypernet,
        device=device,
        show_progress=True,
    )
    e.log(f"Processed train: {len(train_data)}")

    if len(train_data) == 0:
        raise ValueError("No training data after preprocessing!")

    train_loader = DataLoader(
        train_data,
        batch_size=e.BATCH_SIZE,
        shuffle=True,
        num_workers=e.NUM_WORKERS,
        pin_memory=(e.NUM_WORKERS > 0),
    )

    return train_loader, train_data


@experiment.hook("load_valid_data", default=True)
def load_valid_data(
    e: Experiment,
    hypernet: HyperNet,
    device: torch.device,
) -> Tuple[List[Data], DataLoader, List[Data], List[Data]]:
    """
    Load and preprocess validation data.

    Default implementation loads ZINC validation split and preprocesses with HyperNet.

    Args:
        e: Experiment instance
        hypernet: HyperNet encoder for preprocessing
        device: Device for HDC computation

    Returns:
        Tuple of (valid_data, valid_loader, vis_samples, vis_samples_small)
    """
    e.log("\nLoading validation data...")
    valid_ds = get_split("valid", dataset=e.DATASET.lower())
    e.log(f"Raw valid: {len(valid_ds)}")

    e.log("Preprocessing validation data...")
    valid_data = preprocess_dataset(
        valid_ds,
        hypernet,
        device=device,
        show_progress=True,
    )
    e.log(f"Processed valid: {len(valid_data)}")

    valid_loader = DataLoader(
        valid_data,
        batch_size=e.BATCH_SIZE,
        shuffle=False,
        num_workers=e.NUM_WORKERS,
        pin_memory=(e.NUM_WORKERS > 0),
    )

    # Select visualization samples
    num_vis = min(e.NUM_VALIDATION_VISUALIZATIONS, len(valid_data))
    vis_samples = valid_data[:num_vis]
    e.log(f"Selected {num_vis} samples for validation visualization")

    # Select small molecule visualization samples (4-10 atoms)
    vis_samples_small = [d for d in valid_data if 4 <= d.x.size(0) <= 10][:num_vis]
    e.log(f"Selected {len(vis_samples_small)} small molecule samples (4-10 atoms) for visualization")

    return valid_data, valid_loader, vis_samples, vis_samples_small


@experiment.hook("compute_statistics", default=True)
def compute_statistics(
    e: Experiment,
    train_data: Optional[List[Data]],
    hypernet: HyperNet,
    device: torch.device,
) -> Tuple[Tensor, Tensor, int]:
    """
    Compute edge marginals and node counts for model initialization.

    Default implementation computes statistics from the training data.
    Child experiments can override to compute from different data sources.

    Args:
        e: Experiment instance
        train_data: Training data (None for streaming)
        hypernet: HyperNet encoder (needed if train_data is None)
        device: Device for computation

    Returns:
        Tuple of (edge_marginals, node_counts, max_nodes, size_edge_marginals)
    """
    if train_data is None:
        raise ValueError("train_data is required for base experiment statistics computation")

    e.log("\nComputing edge marginals...")
    edge_marginals = compute_edge_marginals(train_data)
    e.log(f"Edge marginals: {edge_marginals.tolist()}")

    node_counts = compute_node_counts(train_data)
    max_nodes = int(node_counts.nonzero()[-1].item()) if node_counts.sum() > 0 else 50
    e.log(f"Max nodes: {max_nodes}")

    size_edge_marginals = compute_size_edge_marginals(train_data, max_nodes + 10)
    e.log(f"Size-conditional marginals computed for {size_edge_marginals.size(0)} sizes")

    return edge_marginals, node_counts, max_nodes, size_edge_marginals


@experiment.hook("modify_callbacks", default=True)
def modify_callbacks(
    e: Experiment,
    callbacks: List[Callback],
    train_loader: DataLoader,
) -> List[Callback]:
    """
    Modify the standard callbacks list.

    Override this hook to add, remove, or modify callbacks. The standard
    callbacks (ModelCheckpoint, LearningRateMonitor, LossTrackingCallback,
    ReconstructionVisualizationCallback) are created in the main function.

    The default implementation returns the callbacks unchanged.

    Args:
        e: Experiment instance
        callbacks: List of standard callbacks to modify
        train_loader: Training data loader (useful for cleanup callbacks)

    Returns:
        Modified list of callbacks
    """
    return callbacks


# =============================================================================
# Entry Point
# =============================================================================

experiment.run_if_main()
