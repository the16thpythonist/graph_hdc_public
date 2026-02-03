#!/usr/bin/env python
"""
Train FlowEdgeDecoder with Streaming Fragment Dataset.

This experiment extends train_flow_edge_decoder.py (base experiment) and overrides
data loading hooks to use a streaming dataset that generates infinite molecular
data by combining BRICS fragments on-the-fly.

Key differences from base:
1. Uses StreamingFragmentDataset for training (infinite data from fragment combinations)
2. Uses fixed ZINC samples for validation
3. Defines epochs as a fixed number of training steps
4. Fragment library is built from ZINC and saved to archive

Usage:
    # Quick test
    python train_flow_edge_decoder_streaming.py --__TESTING__ True

    # Full training
    python train_flow_edge_decoder_streaming.py --__DEBUG__ False --EPOCHS 5000

    # Custom configuration
    python train_flow_edge_decoder_streaming.py --NUM_FRAGMENT_WORKERS 8 --BUFFER_SIZE 20000
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from pytorch_lightning.callbacks import Callback
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph_hdc.datasets.streaming_fragments import (
    FragmentLibrary,
    StreamingFragmentDataLoader,
    StreamingFragmentDataset,
)
from graph_hdc.datasets.utils import get_split
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.models.flow_edge_decoder import (
    compute_edge_marginals,
    compute_node_counts,
    preprocess_dataset,
)


# =============================================================================
# PARAMETER OVERRIDES (must be defined BEFORE Experiment.extend())
# =============================================================================

# -----------------------------------------------------------------------------
# Model Architecture (different from base)
# -----------------------------------------------------------------------------

# :param N_LAYERS:
#     Number of transformer layers. Streaming uses more layers for higher capacity.
N_LAYERS: int = 12

# :param HIDDEN_MLP_DIM:
#     Hidden dimension for MLP blocks. Streaming uses smaller MLP.
HIDDEN_MLP_DIM: int = 256

# :param CONDITION_DIM:
#     Dimension for HDC conditioning. Streaming uses smaller conditioning.
CONDITION_DIM: int = 256

# :param TIME_EMBED_DIM:
#     Dimension for time embedding. Streaming uses larger time embedding.
TIME_EMBED_DIM: int = 256

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

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------

# :param EPOCHS:
#     Number of training epochs.
EPOCHS: int = 1000

# :param BATCH_SIZE:
#     Batch size for training and validation.
BATCH_SIZE: int = 8

# :param LEARNING_RATE:
#     Initial learning rate for Adam optimizer.
LEARNING_RATE: float = 1e-4

# :param WEIGHT_DECAY:
#     Weight decay (L2 regularization) for optimizer.
WEIGHT_DECAY: float = 0e-5

# :param TRAIN_TIME_DISTORTION:
#     Time distortion type during training. Options: "identity", "polydec".
TRAIN_TIME_DISTORTION: str = "identity"

# :param GRADIENT_CLIP_VAL:
#     Gradient clipping value. Set to 0.0 to disable.
GRADIENT_CLIP_VAL: float = 1.5

# -----------------------------------------------------------------------------
# Noise Distribution (different from base)
# -----------------------------------------------------------------------------

# :param NOISE_TYPE:
#     Noise distribution type. Streaming uses marginal noise distribution.
NOISE_TYPE: str = "marginal"

# =============================================================================
# STREAMING-SPECIFIC PARAMETERS (must be defined BEFORE Experiment.extend())
# =============================================================================

# -----------------------------------------------------------------------------
# Streaming Dataset Configuration
# -----------------------------------------------------------------------------

# :param BUFFER_SIZE:
#     Number of samples to keep in the streaming buffer. Larger buffers provide
#     more diversity but use more memory.
BUFFER_SIZE: int = 10000

# :param NUM_FRAGMENT_WORKERS:
#     Number of worker processes generating samples from fragments.
NUM_FRAGMENT_WORKERS: int = 4

# :param FRAGMENTS_RANGE_MIN:
#     Minimum number of fragments per generated molecule.
FRAGMENTS_RANGE_MIN: int = 1

# :param FRAGMENTS_RANGE_MAX:
#     Maximum number of fragments per generated molecule.
FRAGMENTS_RANGE_MAX: int = 4

# :param MAX_GENERATED_NODES:
#     Maximum number of atoms in generated molecules. Molecules exceeding this
#     are discarded.
MAX_GENERATED_NODES: int = 50

# :param PREFILL_FRACTION:
#     Fraction of buffer to fill before starting training.
PREFILL_FRACTION: float = 0.1

# :param STEPS_PER_EPOCH:
#     Number of training steps per epoch. Since streaming data is infinite,
#     this defines when validation runs.
STEPS_PER_EPOCH: int = 1000

# -----------------------------------------------------------------------------
# Validation Configuration
# -----------------------------------------------------------------------------

# :param NUM_VALID_SAMPLES:
#     Number of ZINC samples to use for validation.
NUM_VALID_SAMPLES: int = 1000

# :param NUM_MARGINAL_SAMPLES:
#     Number of ZINC samples for computing edge marginals.
NUM_MARGINAL_SAMPLES: int = 50_000

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
# INHERIT FROM BASE EXPERIMENT
# =============================================================================

experiment = Experiment.extend(
    "train_flow_edge_decoder.py",
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


# =============================================================================
# STREAMING-SPECIFIC CALLBACK
# =============================================================================


class StreamingDatasetCleanupCallback(Callback):
    """Cleanup streaming dataset workers on training end."""

    def __init__(self, streaming_loader: StreamingFragmentDataLoader):
        super().__init__()
        self.streaming_loader = streaming_loader

    def on_train_end(self, trainer, pl_module):
        self.streaming_loader.stop()


# =============================================================================
# HOOK OVERRIDES
# =============================================================================


@experiment.hook("load_train_data", default=False, replace=True)
def load_train_data(
    e: Experiment,
    hypernet: HyperNet,
    device: torch.device,
) -> Tuple[StreamingFragmentDataLoader, None]:
    """
    Create streaming fragment dataset for training.

    Builds a FragmentLibrary from ZINC and creates a StreamingFragmentDataset
    that generates infinite molecular data by combining BRICS fragments.

    Args:
        e: Experiment instance
        hypernet: HyperNet encoder for preprocessing
        device: Device for computation

    Returns:
        Tuple of (streaming_train_loader, None)
        Note: Returns None for train_data because streaming has no static data.
    """
    e.log("\nBuilding fragment library from ZINC...")

    zinc_train = get_split("train", dataset="zinc")
    e.log(f"ZINC train size: {len(zinc_train)}")

    fragment_library = FragmentLibrary(min_atoms=2, max_atoms=30)

    if e.__DEBUG__:
        fragment_library.build_from_dataset(
            zinc_train, show_progress=True, max_molecules=25_000
        )
    else:
        fragment_library.build_from_dataset(zinc_train, show_progress=True)

    e.log(f"Fragment library built: {fragment_library.num_fragments} fragments")
    e["data/num_fragments"] = fragment_library.num_fragments

    # Save fragment library
    library_path = Path(e.path) / "fragment_library.pkl"
    fragment_library.save(library_path)
    e.log(f"Saved fragment library to: {library_path}")

    # Save encoder for workers
    encoder_path = Path(e.path) / "hypernet_encoder.ckpt"
    hypernet.save(encoder_path)
    e["results/encoder_path"] = str(encoder_path)

    # Create streaming dataset
    e.log("\nCreating streaming fragment dataset...")
    batch_size = 16 if e.__DEBUG__ else e.BATCH_SIZE

    streaming_dataset = StreamingFragmentDataset(
        fragment_library=fragment_library,
        hypernet_checkpoint_path=encoder_path,
        buffer_size=e.BUFFER_SIZE,
        num_workers=e.NUM_FRAGMENT_WORKERS,
        fragments_range=(e.FRAGMENTS_RANGE_MIN, e.FRAGMENTS_RANGE_MAX),
        max_nodes=e.MAX_GENERATED_NODES,
        dataset_name="zinc",
        prefill_fraction=e.PREFILL_FRACTION,
    )

    train_loader = StreamingFragmentDataLoader(
        dataset=streaming_dataset,
        batch_size=batch_size,
        steps_per_epoch=e.STEPS_PER_EPOCH,
    )

    e.log(f"Streaming loader: {len(train_loader)} steps per epoch")

    # Test streaming dataloader
    e.log("\nTesting streaming dataloader (3 batches)...")
    try:
        train_loader.test_iteration(num_batches=3)
        e.log("Streaming dataloader test passed!")
    except Exception as ex:
        e.log(f"ERROR: Streaming dataloader test failed: {ex}")
        train_loader.stop()
        raise

    # Return None for train_data - streaming doesn't have static data
    return train_loader, None


@experiment.hook("load_valid_data", default=False, replace=True)
def load_valid_data(
    e: Experiment,
    hypernet: HyperNet,
    device: torch.device,
) -> Tuple[List[Data], DataLoader, List[Data]]:
    """
    Load ZINC subset for validation.

    Args:
        e: Experiment instance
        hypernet: HyperNet encoder for preprocessing
        device: Device for computation

    Returns:
        Tuple of (valid_data list, valid_loader, vis_samples)
    """
    e.log("\nPreparing validation data from ZINC...")

    zinc_valid = get_split("valid", dataset="zinc")
    num_valid = min(e.NUM_VALID_SAMPLES, len(zinc_valid))
    zinc_valid_subset = list(zinc_valid)[:num_valid]

    valid_data = preprocess_dataset(
        zinc_valid_subset,
        hypernet,
        device=device,
        show_progress=True,
    )

    e.log(f"Validation data: {len(valid_data)} samples")
    e["data/valid_size"] = len(valid_data)

    batch_size = 16 if e.__DEBUG__ else e.BATCH_SIZE
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Select visualization samples
    num_vis = min(e.NUM_VALIDATION_VISUALIZATIONS, len(valid_data))
    vis_samples = valid_data[:num_vis]
    e.log(f"Selected {num_vis} samples for visualization")

    return valid_data, valid_loader, vis_samples


@experiment.hook("compute_statistics", default=False, replace=True)
def compute_statistics(
    e: Experiment,
    train_data: Optional[List[Data]],
    hypernet: HyperNet,
    device: torch.device,
) -> Tuple[Tensor, Tensor, int]:
    """
    Compute edge marginals from ZINC (not from streaming data).

    Streaming experiment computes marginals from a separate ZINC subset
    because the streaming data is infinite.

    Args:
        e: Experiment instance
        train_data: Not used (None for streaming)
        hypernet: HyperNet encoder for preprocessing
        device: Device for computation

    Returns:
        Tuple of (edge_marginals, node_counts, max_nodes)
    """
    e.log("\nComputing edge marginals from ZINC...")

    zinc_train = get_split("train", dataset="zinc")
    num_marginal = min(e.NUM_MARGINAL_SAMPLES, len(zinc_train))
    zinc_for_marginals = list(zinc_train)[:num_marginal]

    marginal_data = preprocess_dataset(
        zinc_for_marginals,
        hypernet,
        device=device,
        show_progress=True,
    )

    edge_marginals = compute_edge_marginals(marginal_data)
    e.log(f"Edge marginals: {edge_marginals.tolist()}")

    node_counts = compute_node_counts(marginal_data)
    max_nodes = int(node_counts.nonzero()[-1].item()) if node_counts.sum() > 0 else 50
    max_nodes = max(max_nodes, e.MAX_GENERATED_NODES)
    e.log(f"Max nodes: {max_nodes}")

    return edge_marginals, node_counts, max_nodes


@experiment.hook("modify_callbacks", default=False, replace=True)
def modify_callbacks(
    e: Experiment,
    callbacks: List[Callback],
    train_loader: StreamingFragmentDataLoader,
) -> List[Callback]:
    """
    Add streaming cleanup callback.

    Adds StreamingDatasetCleanupCallback to ensure worker processes are
    properly terminated when training ends.

    Args:
        e: Experiment instance
        callbacks: List of standard callbacks from base experiment
        train_loader: Streaming loader for cleanup callback

    Returns:
        Modified list of callbacks with cleanup callback added
    """
    callbacks.append(StreamingDatasetCleanupCallback(train_loader))
    return callbacks


# =============================================================================
# Testing Mode
# =============================================================================


@experiment.testing
def testing(e: Experiment):
    """Quick test mode with reduced parameters for streaming."""
    e.EPOCHS = 2
    e.BATCH_SIZE = 4
    e.SAMPLE_STEPS = 10
    e.NUM_RECONSTRUCTION_SAMPLES = 3
    e.NUM_VALIDATION_VISUALIZATIONS = 2
    e.STEPS_PER_EPOCH = 10
    e.BUFFER_SIZE = 100
    e.NUM_FRAGMENT_WORKERS = 1
    e.NUM_VALID_SAMPLES = 20
    e.NUM_MARGINAL_SAMPLES = 100


# =============================================================================
# Entry Point
# =============================================================================

experiment.run_if_main()
