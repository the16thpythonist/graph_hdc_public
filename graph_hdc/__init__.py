"""
Graph HDC: Hyperdimensional Computing for Molecular Graph Generation.

This package implements HDC-based molecular graph encoding and generation
using Vector Symbolic Architectures (VSA) with Holographic Reduced
Representations (HRR).
"""

__version__ = "0.1.0"

# Apply TorchHD patches for correct HRR multibind/multibundle operations
# Must be imported before any torchhd usage
from graph_hdc.utils.torchhd_patch import apply_patches as _apply_torchhd_patches

_apply_torchhd_patches()

# Core encoder
from graph_hdc.hypernet.encoder import CorrectionLevel, DecodingResult, HyperNet
from graph_hdc.hypernet.multi_hypernet import MultiHyperNet

# Configurations
from graph_hdc.hypernet.configs import (
    DecoderSettings,
    DSHDCConfig,
    FallbackDecoderSettings,
    RWConfig,
    create_config_with_rw,
    get_config,
)

# Datasets
from graph_hdc.datasets.utils import get_split, post_compute_encodings, scan_node_features_with_rw

# Evaluator and metrics
from graph_hdc.utils.evaluator import (
    GenerationEvaluator,
    calculate_internal_diversity,
    rdkit_logp,
    rdkit_qed,
    rdkit_sa_score,
)

__all__ = [
    # Encoder
    "HyperNet",
    "MultiHyperNet",
    "CorrectionLevel",
    "DecodingResult",
    # Configs
    "get_config",
    "create_config_with_rw",
    "DSHDCConfig",
    "RWConfig",
    "DecoderSettings",
    "FallbackDecoderSettings",
    # Datasets
    "get_split",
    "post_compute_encodings",
    "scan_node_features_with_rw",
    # Evaluator
    "GenerationEvaluator",
    "rdkit_logp",
    "rdkit_qed",
    "rdkit_sa_score",
    "calculate_internal_diversity",
]
