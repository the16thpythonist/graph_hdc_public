#!/usr/bin/env python
"""
Test FlowEdgeDecoder on molecules sampled from a dataset.

Child experiment of ``test_flow_edge_decoder.py`` that replaces the
fixed SMILES_LIST with random sampling from a PyG dataset (ZINC by
default). The complete dataset (train + valid + test) is loaded and
``NUM_DATASET_SAMPLES`` molecules are drawn uniformly at random.

Usage:
    # Sample 200 random molecules from ZINC
    python test_flow_edge_decoder__dataset.py \
        --HDC_ENCODER_PATH /path/to/encoder.ckpt \
        --FLOW_DECODER_PATH /path/to/decoder.ckpt \
        --NUM_DATASET_SAMPLES 200

    # Sample from QM9 instead
    python test_flow_edge_decoder__dataset.py \
        --HDC_ENCODER_PATH /path/to/encoder.ckpt \
        --FLOW_DECODER_PATH /path/to/decoder.ckpt \
        --DATASET qm9 --NUM_DATASET_SAMPLES 200

    # Quick test
    python test_flow_edge_decoder__dataset.py --__TESTING__ True
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

from rdkit import Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from graph_hdc.datasets.utils import get_split


def _sanitize_smiles(smiles: str) -> Optional[str]:
    """
    Sanitize a molecule: remove charges, explicit H, and all stereochemistry.

    Converts stereo bonds to basic types, strips chiral center annotations,
    zeroes all formal charges, removes explicit hydrogens, and returns a
    new canonical SMILES.  Returns ``None`` if the resulting molecule is
    chemically invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Remove explicit hydrogens
    mol = Chem.RemoveHs(mol)

    # Remove all stereochemistry (chiral centers + bond E/Z)
    Chem.RemoveStereochemistry(mol)

    # Zero all formal charges
    for atom in mol.GetAtoms():
        atom.SetFormalCharge(0)

    # First sanitize pass – may fail if charge removal broke valence rules
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    # Fix radicals AFTER sanitize (SanitizeMol can recompute radical
    # electrons from valence).  Zero them out and force implicit-H
    # recalculation via UpdatePropertyCache.
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            atom.SetNumRadicalElectrons(0)
            atom.SetNoImplicit(False)
            atom.UpdatePropertyCache(strict=False)

    # Re-perceive aromaticity: charge removal can leave rings with stale
    # aromatic flags.  Kekulize first (assigns explicit single/double bonds),
    # then re-apply aromaticity detection from scratch.
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
        Chem.SetAromaticity(mol)
    except Exception:
        pass  # keep whatever SanitizeMol produced

    new_smiles = Chem.MolToSmiles(mol)
    return new_smiles if new_smiles else None


# =============================================================================
# PARAMETERS (copied from parent experiment)
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
# Dataset Sampling Configuration (NEW in this child experiment)
# -----------------------------------------------------------------------------

# :param NUM_DATASET_SAMPLES:
#     Number of molecules to randomly sample from the complete dataset
#     (train + valid + test splits combined).
NUM_DATASET_SAMPLES: int = 1000

# :param DATASET_SEED:
#     Random seed for reproducible dataset sampling. Separate from the
#     model SEED to allow re-drawing different subsets independently.
DATASET_SEED: int = 0

# :param MAX_MOLECULE_SIZE:
#     Maximum number of atoms (nodes) a molecule may have to be included
#     in the sample pool. Molecules with more atoms are filtered out
#     before sampling. Set to None to disable the filter.
MAX_MOLECULE_SIZE: Optional[int] = 40

# -----------------------------------------------------------------------------
# Input SMILES (inherited but ignored — overridden by load_smiles hook)
# -----------------------------------------------------------------------------

# :param SMILES_CSV_PATH:
#     Path to CSV file with a "smiles" column. Not used in this child
#     experiment (dataset sampling takes precedence).
SMILES_CSV_PATH: str = ""

# :param SMILES_LIST:
#     Not used in this child experiment. The load_smiles hook samples
#     directly from the dataset instead.
SMILES_LIST: list[str] = []

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
HDC_DEVICE: str = "cpu"

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
# EXPERIMENT (extends parent)
# =============================================================================

experiment = Experiment.extend(
    "test_flow_edge_decoder.py",
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


# =============================================================================
# HOOKS
# =============================================================================


@experiment.hook("load_smiles", replace=True, default=False)
def load_smiles_from_dataset(e: Experiment) -> list[str]:
    """
    Load SMILES by randomly sampling from the complete dataset.

    Loads all three splits (train + valid + test), combines them, and
    draws ``NUM_DATASET_SAMPLES`` molecules uniformly at random using
    ``DATASET_SEED`` for reproducibility.

    Args:
        e: Experiment instance.

    Returns:
        List of SMILES strings sampled from the dataset.
    """
    dataset_name = e.DATASET
    num_samples = e.NUM_DATASET_SAMPLES
    seed = e.DATASET_SEED
    max_size = e.MAX_MOLECULE_SIZE

    e.log(f"Loading complete {dataset_name.upper()} dataset (all splits)...")

    all_smiles: list[str] = []
    for split in ["train", "valid", "test"]:
        ds = get_split(split=split, dataset=dataset_name)
        split_smiles = [data.smiles for data in ds if max_size is None or data.num_nodes <= max_size]
        e.log(f"  {split}: {len(split_smiles)} molecules (after size filter)")
        all_smiles.extend(split_smiles)

    e.log(f"Total molecules after filtering (max_size={max_size}): {len(all_smiles)}")

    # Shuffle the full pool with fixed seed, then iterate and sanitize
    # until we have enough valid molecules (drop-and-resample strategy).
    rng = random.Random(seed)
    rng.shuffle(all_smiles)

    sampled: list[str] = []
    num_dropped = 0
    for smi in all_smiles:
        if len(sampled) >= num_samples:
            break

        clean = _sanitize_smiles(smi)
        if clean is not None:
            sampled.append(clean)
        else:
            num_dropped += 1

    if num_dropped > 0:
        e.log(f"Dropped {num_dropped} molecules that became invalid after sanitization")

    if len(sampled) < num_samples:
        e.log(f"WARNING: Only {len(sampled)} valid molecules available "
               f"(requested {num_samples})")

    e.log(f"Sampled {len(sampled)} sanitized molecules (seed={seed})")
    e["config/smiles_source"] = "dataset"
    e["config/dataset_name"] = dataset_name
    e["config/num_dataset_samples"] = len(sampled)
    e["config/dataset_seed"] = seed
    e["config/max_molecule_size"] = max_size
    e["config/total_dataset_size"] = len(all_smiles)
    e["config/num_dropped_sanitization"] = num_dropped

    return sampled


# =============================================================================
# Testing Mode
# =============================================================================


@experiment.testing
def testing(e: Experiment) -> None:
    """Quick test mode: sample just 2 molecules."""
    e.NUM_DATASET_SAMPLES = 2
    e.SAMPLE_STEPS = 10
    e.GENERATE_GIF = True
    e.GIF_FRAME_INTERVAL = 2


# =============================================================================
# Entry Point
# =============================================================================

experiment.run_if_main()
