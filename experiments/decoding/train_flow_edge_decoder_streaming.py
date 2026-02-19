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

from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from pycomex import INHERIT
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from pytorch_lightning.callbacks import Callback
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph_hdc.datasets.mixed_streaming import MixedStreamingDataLoader, StreamingSource
from graph_hdc.datasets.streaming_fragments import (
    FragmentLibrary,
    StreamingFragmentDataLoader,
    StreamingFragmentDataset,
)
from graph_hdc.datasets.streaming_small_molecules import (
    SmallMoleculePool,
    SmallMoleculeStreamingDataset,
)
from graph_hdc.datasets.utils import get_split
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.models.flow_edge_decoder import (
    compute_edge_marginals,
    compute_size_edge_marginals,
    compute_node_counts,
    preprocess_dataset,
)


# =============================================================================
# PARAMETER OVERRIDES (must be defined BEFORE Experiment.extend())
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
HDC_CONFIG_PATH: str = ''

# :param HDC_DIM:
#     Hypervector dimension for the HyperNet encoder. Only used if HDC_CONFIG_PATH
#     is empty. Typical values: 256, 512, 1024.
HDC_DIM: int = 1024

# :param HDC_DEPTH:
#     Message passing depth for the HyperNet encoder. Only used if HDC_CONFIG_PATH
#     is empty. Higher values capture longer-range structural information.
HDC_DEPTH: int = 8

# :param USE_RW:
#     Whether to augment node features with random walk return probabilities.
#     When True, each node's feature tuple is extended with binned RW return
#     probabilities at each step in RW_K_VALUES. This affects both the HDC
#     conditioning vector and the FlowEdgeDecoder's one-hot node features
#     (extended from 24-dim to 24 + RW_NUM_BINS * len(RW_K_VALUES)).
USE_RW: bool = True

# :param RW_K_VALUES:
#     Random walk steps at which to compute return probabilities. Only used
#     when USE_RW is True.
RW_K_VALUES: tuple = (4, 6, 10, 16)

# :param USE_QUANTILE_BINS:
#     Whether to use precomputed quantile-based bin boundaries for RW features
#     instead of uniform bins on [0,1]. Quantile binning distributes atoms
#     equally across bins for each k value, avoiding the near-degenerate
#     distributions that uniform binning produces at higher k (e.g. k=10 puts
#     81% of atoms into bin 0 with uniform bins). Requires RW_NUM_BINS in
#     {3, 4, 5, 6}. Only used when USE_RW is True.
USE_QUANTILE_BINS: bool = True

# :param RW_CLIP_RANGE:
#     When set to a (lo, hi) tuple and USE_QUANTILE_BINS is False, uniform bins
#     are placed over [lo, hi] instead of [0, 1]. Values outside this range are
#     clamped to the first / last bin. This concentrates bin resolution in the
#     region where most RW return probabilities fall. Ignored when
#     USE_QUANTILE_BINS is True. Only used when USE_RW is True.
RW_CLIP_RANGE: tuple | None = (0, 0.8)

# :param RW_NUM_BINS:
#     Number of uniform bins for discretising RW return probabilities on [0,1].
#     Only used when USE_RW is True.
RW_NUM_BINS: int = 5

# :param PRUNE_CODEBOOK:
#     Whether to prune the HDC codebook to only feature tuples observed in the
#     dataset. When True (default), unseen tuples cause encoding errors — set to
#     False when training on generated molecules (e.g. streaming fragments) whose
#     topology may produce novel feature combinations.
PRUNE_CODEBOOK: bool = False

# :param NORMALIZE_GRAPH_EMBEDDING:
#     Whether to L2-normalize the graph embedding (order-N) output of each
#     HyperNet before concatenation into the HDC conditioning vector. Without
#     normalization, the embedding magnitude scales with ~sqrt(num_atoms),
#     causing the decoder's conditioning signal to be systematically weaker
#     for small molecules.
NORMALIZE_GRAPH_EMBEDDING: bool = True

# :param USE_RRWP_HYPERNET:
#     When True (requires USE_RW=True), creates RRWPHyperNet which uses
#     RRWP-enriched features only for order-0 (node_terms) readout, while
#     message passing operates on base features only. This prevents positional
#     information from interfering with structural binding. Works with both
#     single HyperNet and MultiHyperNet ensembles.
USE_RRWP_HYPERNET: bool = True

# :param ENSEMBLE_CONFIGS:
#     Optional list of (hv_dim, depth) tuples for a MultiHyperNet ensemble.
#     Each tuple creates an independently-initialized HyperNet with its own
#     random codebook, providing a different "perspective" on the same graph.
#     Seeds are auto-generated as SEED+0, SEED+1, etc.
#     When empty/None, a single HyperNet with HDC_DIM/HDC_DEPTH is used instead.
#     Example: [(256, 6), (512, 4), (256, 8)] creates 3 HyperNets.
ENSEMBLE_CONFIGS: Optional[List[Tuple[int, int]]] = None

# -----------------------------------------------------------------------------
# Model Architecture (different from base)
# -----------------------------------------------------------------------------

# :param N_LAYERS:
#     Number of transformer layers. Streaming uses more layers for higher capacity.
N_LAYERS: int = 16

# :param HIDDEN_DIM:
#     Hidden dimension for transformer layers.
HIDDEN_DIM: int = 256

# :param HIDDEN_MLP_DIM:
#     Hidden dimension for MLP blocks in transformer.
HIDDEN_MLP_DIM: int = 256

# :param N_HEADS:
#     Number of attention heads in transformer layers.
N_HEADS: int = INHERIT

# :param DROPOUT:
#     Dropout probability in transformer layers.
DROPOUT: float = INHERIT

# :param CONDITION_DIM:
#     Dimension for HDC conditioning after MLP projection.
CONDITION_DIM: int = 512

# :param TIME_EMBED_DIM:
#     Dimension for sinusoidal time embedding.
TIME_EMBED_DIM: int = 128

# :param USE_CROSS_ATTN:
#     Whether to use cross-attention HDC conditioning. When True, the raw HDC
#     vector is decomposed into learnable tokens, nodes cross-attend to these
#     tokens, and node-pair combinations produce edge-specific conditioning
#     signals — supplementing the broadcast FiLM conditioning.
USE_CROSS_ATTN: bool = True

# :param CROSS_ATTN_TOKENS:
#     Number of tokens to decompose the HDC vector into for cross-attention.
#     Only used when USE_CROSS_ATTN is True.
CROSS_ATTN_TOKENS: int = 10

# :param CROSS_ATTN_HEADS:
#     Number of attention heads for the cross-attention conditioner.
#     Only used when USE_CROSS_ATTN is True.
CROSS_ATTN_HEADS: int = INHERIT

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
TRAIN_TIME_DISTORTION: str = "polydec"

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
ETA: float = INHERIT

# :param OMEGA:
#     Target guidance strength parameter.
OMEGA: float = INHERIT

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
# Noise Distribution (different from base)
# -----------------------------------------------------------------------------

# :param NOISE_TYPE:
#     Noise distribution type. Streaming uses marginal noise distribution.
NOISE_TYPE: str = "marginal"

# -----------------------------------------------------------------------------
# System Configuration
# -----------------------------------------------------------------------------

# :param SEED:
#     Random seed for reproducibility.
SEED: int = INHERIT

# :param ACCELERATOR:
#     PyTorch Lightning accelerator. Options: "auto", "gpu", "cpu".
#     Use "gpu" to force GPU training.
ACCELERATOR: str = "gpu"

# -----------------------------------------------------------------------------
# Reconstruction Evaluation
# -----------------------------------------------------------------------------

# :param NUM_RECONSTRUCTION_SAMPLES:
#     Number of molecules to reconstruct and visualize after training.
NUM_RECONSTRUCTION_SAMPLES: int = 100

# :param RECONSTRUCTION_BATCH_SIZE:
#     Batch size for parallel edge generation during reconstruction evaluation.
#     Higher values are faster but use more GPU memory.
RECONSTRUCTION_BATCH_SIZE: int = 1

# :param NUM_VALIDATION_VISUALIZATIONS:
#     Number of molecules to visualize during each validation epoch.
NUM_VALIDATION_VISUALIZATIONS: int = 10

# :param NUM_VALIDATION_REPETITIONS:
#     Number of parallel decodings per molecule during validation visualization.
#     Each molecule is decoded this many times in a single batched call, and
#     the result with the lowest HDC cosine distance is kept (best-of-N).
NUM_VALIDATION_REPETITIONS: int = 10

# -----------------------------------------------------------------------------
# Resume Training
# -----------------------------------------------------------------------------

# :param RESUME_CHECKPOINT_PATH:
#     Path to PyTorch Lightning checkpoint (.ckpt) to resume training from.
#     If set, RESUME_ENCODER_PATH must also be provided.
RESUME_CHECKPOINT_PATH: Optional[str] = "/media/ssd2/Programming/graph_hdc_public/experiments/decoding/results/train_flow_edge_decoder_streaming/GOOD/last.ckpt"

# :param RESUME_ENCODER_PATH:
#     Path to HyperNet encoder checkpoint when resuming training.
#     Required if RESUME_CHECKPOINT_PATH is set.
RESUME_ENCODER_PATH: Optional[str] = "/media/ssd2/Programming/graph_hdc_public/experiments/decoding/results/train_flow_edge_decoder_streaming/GOOD/hypernet_encoder.ckpt"

# =============================================================================
# STREAMING-SPECIFIC PARAMETERS (must be defined BEFORE Experiment.extend())
# =============================================================================

# -----------------------------------------------------------------------------
# Streaming Dataset Configuration
# -----------------------------------------------------------------------------

# :param BUFFER_SIZE:
#     Number of samples to keep in the streaming buffer. Larger buffers smooth
#     throughput by absorbing production/consumption rate mismatches, but use
#     more memory.
BUFFER_SIZE: int = 2000

# :param NUM_FRAGMENT_WORKERS:
#     Number of worker processes generating samples from fragments.
NUM_FRAGMENT_WORKERS: int = 2

# :param FRAGMENTS_RANGE_MIN:
#     Minimum number of fragments per generated molecule.
FRAGMENTS_RANGE_MIN: int = 1

# :param FRAGMENTS_RANGE_MAX:
#     Maximum number of fragments per generated molecule.
FRAGMENTS_RANGE_MAX: int = 4

# :param MAX_GENERATED_NODES:
#     Maximum number of atoms in generated molecules. Molecules exceeding this
#     are discarded.
MAX_GENERATED_NODES: int = 40

# :param PREFILL_FRACTION:
#     Fraction of buffer to fill before starting training.
PREFILL_FRACTION: float = 0.1

# :param ENUMERATE_ATTACHMENTS:
#     If True, expand the fragment library by enumerating additional attachment
#     positions on each fragment using a universal wildcard label. This increases
#     the diversity of fragment combinations (e.g. positional isomers).
ENUMERATE_ATTACHMENTS: bool = False

# :param ENUM_MAX_VARIANTS_PER_FRAGMENT:
#     Maximum number of enumerated attachment-point variants to generate per
#     fragment. Only used when ENUMERATE_ATTACHMENTS is True.
ENUM_MAX_VARIANTS_PER_FRAGMENT: int = 10

# :param STEPS_PER_EPOCH:
#     Number of training steps per epoch. Since streaming data is infinite,
#     this defines when validation runs.
STEPS_PER_EPOCH: int = 1000

# -----------------------------------------------------------------------------
# Small Molecule Mixing Configuration
# -----------------------------------------------------------------------------

# :param SMALL_MOL_MIXING_WEIGHT:
#     Relative weight for small molecule samples in mixed streaming.
#     Default 0.1 means ~10% of training samples come from small molecules.
#     Set to 0.0 to disable small molecule mixing entirely.
SMALL_MOL_MIXING_WEIGHT: float = 0.2

# :param FRAGMENT_MIXING_WEIGHT:
#     Relative weight for fragment-based samples in mixed streaming.
FRAGMENT_MIXING_WEIGHT: float = 0.8

# :param NUM_SMALL_MOL_WORKERS:
#     Number of worker processes for small molecule streaming.
NUM_SMALL_MOL_WORKERS: int = 1

# :param SMALL_MOL_BUFFER_SIZE:
#     Buffer size for small molecule streaming queue.
SMALL_MOL_BUFFER_SIZE: int = 2000

# :param SMALL_MOL_CSV_PATH:
#     Path to pre-built CSV containing small molecule SMILES and source.
SMALL_MOL_CSV_PATH: str = "data/small_molecules.csv"

# -----------------------------------------------------------------------------
# Validation Configuration
# -----------------------------------------------------------------------------

# :param NUM_VALID_SAMPLES:
#     Number of ZINC samples to use for validation.
NUM_VALID_SAMPLES: int = 2000

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

    def __init__(self, streaming_loader):
        super().__init__()
        self.streaming_loader = streaming_loader

    def on_train_end(self, trainer, pl_module):
        self.streaming_loader.stop()


# =============================================================================
# HOOK OVERRIDES
# =============================================================================


# CPK-inspired element colors
_ELEMENT_COLORS = {
    "C": "#909090",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "F": "#00CED1",
    "S": "#FFFF30",
    "Cl": "#1FF01F",
    "Br": "#A62929",
    "P": "#FF8000",
    "I": "#940094",
}


def _plot_fragment_atom_distribution(e: Experiment, library: FragmentLibrary) -> None:
    """Plot and save the atom type distribution across all fragments."""
    from rdkit import Chem

    atom_counts: Counter = Counter()
    for smiles in library.fragments:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym != "*":
                atom_counts[sym] += 1

    if not atom_counts:
        e.log("WARNING: No atoms found in fragment library, skipping distribution plot")
        return

    # Sort elements by count (descending) for a cleaner plot
    elements = sorted(atom_counts.keys(), key=lambda s: atom_counts[s], reverse=True)
    counts = [atom_counts[el] for el in elements]
    colors = [_ELEMENT_COLORS.get(el, "#CCCCCC") for el in elements]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(elements, counts, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Element")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"Fragment Library Atom Distribution ({library.num_fragments} fragments)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    plot_path = Path(e.path) / "fragment_atom_distribution.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    e.log(f"Saved fragment atom distribution plot to: {plot_path}")

    # Log the counts
    for el, count in zip(elements, counts):
        e.log(f"  {el}: {count}")


@experiment.hook("load_train_data", default=False, replace=True)
def load_train_data(
    e: Experiment,
    hypernet: HyperNet,
    device: torch.device,
) -> Tuple[MixedStreamingDataLoader, None]:
    """
    Create mixed streaming dataset for training.

    Builds a FragmentLibrary from ZINC, optionally adds a SmallMoleculePool,
    and creates a MixedStreamingDataLoader that pulls from both sources
    according to configurable weights.

    Args:
        e: Experiment instance
        hypernet: HyperNet encoder for preprocessing
        device: Device for computation

    Returns:
        Tuple of (mixed_train_loader, None)
        Note: Returns None for train_data because streaming has no static data.
    """
    # -----------------------------------------------------------------
    # Fragment library (unchanged)
    # -----------------------------------------------------------------
    e.log("\nBuilding fragment library from ZINC...")

    zinc_train = get_split("train", dataset="zinc")
    e.log(f"ZINC train size: {len(zinc_train)}")

    fragment_library = FragmentLibrary(min_atoms=2, max_atoms=30)

    fragment_library.build_from_dataset(zinc_train, show_progress=True)

    e.log(f"Fragment library built: {fragment_library.num_fragments} fragments")

    # Optionally expand library with enumerated attachment positions
    if e.ENUMERATE_ATTACHMENTS:
        n_before = fragment_library.num_fragments
        n_added = fragment_library.expand_with_enumerated_positions(
            max_new_points=e.ENUM_MAX_VARIANTS_PER_FRAGMENT,
        )
        e.log(f"Enumerated attachment expansion: {n_before} -> {fragment_library.num_fragments} "
              f"fragments (+{n_added} variants)")
        e["data/num_enumerated_variants"] = n_added

    e["data/num_fragments"] = fragment_library.num_fragments

    # Save fragment library
    library_path = Path(e.path) / "fragment_library.pkl"
    fragment_library.save(library_path)
    e.log(f"Saved fragment library to: {library_path}")

    # Plot atom distribution of the fragment library
    _plot_fragment_atom_distribution(e, fragment_library)

    # Save encoder for workers
    encoder_path = Path(e.path) / "hypernet_encoder.ckpt"
    hypernet.save(encoder_path)
    e["results/encoder_path"] = str(encoder_path)

    # -----------------------------------------------------------------
    # Fragment streaming source
    # -----------------------------------------------------------------
    e.log("\nCreating streaming fragment dataset...")

    fragment_dataset = StreamingFragmentDataset(
        fragment_library=fragment_library,
        hypernet_checkpoint_path=encoder_path,
        buffer_size=e.BUFFER_SIZE,
        num_workers=e.NUM_FRAGMENT_WORKERS,
        fragments_range=(e.FRAGMENTS_RANGE_MIN, e.FRAGMENTS_RANGE_MAX),
        max_nodes=e.MAX_GENERATED_NODES,
        dataset_name="zinc",
        prefill_fraction=e.PREFILL_FRACTION,
    )

    sources = [
        StreamingSource(
            name="fragments",
            dataset=fragment_dataset,
            weight=e.FRAGMENT_MIXING_WEIGHT,
        ),
    ]

    # -----------------------------------------------------------------
    # Small molecule streaming source (optional)
    # -----------------------------------------------------------------
    if e.SMALL_MOL_MIXING_WEIGHT > 0:
        e.log("\nLoading small molecule pool...")
        small_mol_pool = SmallMoleculePool(e.SMALL_MOL_CSV_PATH)
        e.log(f"Small molecule pool: {small_mol_pool.size} unique SMILES")
        e["data/small_mol_pool_size"] = small_mol_pool.size

        small_mol_dataset = SmallMoleculeStreamingDataset(
            smiles_pool=small_mol_pool,
            hypernet_checkpoint_path=encoder_path,
            buffer_size=e.SMALL_MOL_BUFFER_SIZE,
            num_workers=e.NUM_SMALL_MOL_WORKERS,
            max_nodes=e.MAX_GENERATED_NODES,
            prefill_fraction=e.PREFILL_FRACTION,
        )

        sources.append(
            StreamingSource(
                name="small_molecules",
                dataset=small_mol_dataset,
                weight=e.SMALL_MOL_MIXING_WEIGHT,
            ),
        )

    # -----------------------------------------------------------------
    # Mixed loader
    # -----------------------------------------------------------------
    batch_size = 16 if e.__DEBUG__ else e.BATCH_SIZE

    train_loader = MixedStreamingDataLoader(
        sources=sources,
        batch_size=batch_size,
        steps_per_epoch=e.STEPS_PER_EPOCH,
    )

    e.log(f"Mixed streaming loader: {len(train_loader)} steps per epoch, "
          f"{len(sources)} source(s)")
    for src in sources:
        e.log(f"  - {src.name}: weight={src.weight}")

    # Test mixed dataloader
    e.log("\nTesting mixed streaming dataloader (3 batches)...")
    try:
        train_loader.test_iteration(num_batches=3)
        e.log("Mixed streaming dataloader test passed!")
    except Exception as ex:
        e.log(f"ERROR: Mixed streaming dataloader test failed: {ex}")
        train_loader.stop()
        raise

    # Return None for train_data - streaming doesn't have static data
    return train_loader, None


@experiment.hook("load_valid_data", default=False, replace=True)
def load_valid_data(
    e: Experiment,
    hypernet: HyperNet,
    device: torch.device,
) -> Tuple[List[Data], DataLoader, List[Data], List[Data]]:
    """
    Load ZINC subset for validation.

    Args:
        e: Experiment instance
        hypernet: HyperNet encoder for preprocessing
        device: Device for computation

    Returns:
        Tuple of (valid_data, valid_loader, vis_samples, vis_samples_small)
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

    # Select small molecule visualization samples (4-10 atoms)
    vis_samples_small = [d for d in valid_data if 4 <= d.x.size(0) <= 10][:num_vis]
    e.log(f"Selected {len(vis_samples_small)} small molecule samples (4-10 atoms) for visualization")

    return valid_data, valid_loader, vis_samples, vis_samples_small


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
        Tuple of (edge_marginals, node_counts, max_nodes, size_edge_marginals)
    """
    e.log("\nComputing edge marginals from ZINC...")

    zinc_train = get_split("train", dataset="zinc")
    num_marginal = min(e.NUM_MARGINAL_SAMPLES, len(zinc_train))
    zinc_for_marginals = list(zinc_train)[:num_marginal]

    # Force CPU to avoid moving hypernet to GPU as a side effect of
    # preprocess_dataset (which calls hypernet.to(device) in-place).
    # Marginals only need graph structure, not GPU acceleration.
    marginal_data = preprocess_dataset(
        zinc_for_marginals,
        hypernet,
        device=torch.device("cpu"),
        show_progress=True,
    )

    edge_marginals = compute_edge_marginals(marginal_data)
    e.log(f"Edge marginals: {edge_marginals.tolist()}")

    node_counts = compute_node_counts(marginal_data)
    max_nodes = int(node_counts.nonzero()[-1].item()) if node_counts.sum() > 0 else 50
    max_nodes = max(max_nodes, e.MAX_GENERATED_NODES)
    e.log(f"Max nodes: {max_nodes}")

    size_edge_marginals = compute_size_edge_marginals(marginal_data, max_nodes + 10)
    e.log(f"Size-conditional marginals computed for {size_edge_marginals.size(0)} sizes")

    return edge_marginals, node_counts, max_nodes, size_edge_marginals


@experiment.hook("modify_callbacks", default=False, replace=True)
def modify_callbacks(
    e: Experiment,
    callbacks: List[Callback],
    train_loader,
) -> List[Callback]:
    """
    Add streaming cleanup callback.

    Adds StreamingDatasetCleanupCallback to ensure worker processes are
    properly terminated when training ends.  Works with both
    ``MixedStreamingDataLoader`` and ``StreamingFragmentDataLoader`` since
    both implement ``stop()``.

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
    """Quick test mode with reduced parameters for streaming.

    Shrinks model architecture and ensemble to run on CPU with minimal RAM.
    Still exercises the ensemble code path with a tiny MultiHyperNet.
    """
    # --- Data / iteration budget ---
    e.EPOCHS = 2
    e.BATCH_SIZE = 4
    e.SAMPLE_STEPS = 10
    e.NUM_RECONSTRUCTION_SAMPLES = 3
    e.NUM_VALIDATION_VISUALIZATIONS = 2
    e.NUM_VALIDATION_REPETITIONS = 2
    e.STEPS_PER_EPOCH = 10
    e.BUFFER_SIZE = 100
    e.NUM_FRAGMENT_WORKERS = 1
    e.NUM_VALID_SAMPLES = 20
    e.NUM_MARGINAL_SAMPLES = 100
    # Small molecule mixing
    e.SMALL_MOL_MIXING_WEIGHT = 0.2
    e.NUM_SMALL_MOL_WORKERS = 1
    e.SMALL_MOL_BUFFER_SIZE = 50
    # --- Tiny ensemble (exercises MultiHyperNet code path) ---
    e.ENSEMBLE_CONFIGS = [(512, 2), (512, 2)]
    e.HDC_DIM = 512
    e.HDC_DEPTH = 2
    # --- Small model architecture ---
    e.N_LAYERS = 2
    e.HIDDEN_DIM = 32
    e.HIDDEN_MLP_DIM = 32
    e.CONDITION_DIM = 64
    e.TIME_EMBED_DIM = 16
    e.USE_CROSS_ATTN = False


# =============================================================================
# Entry Point
# =============================================================================

experiment.run_if_main()
