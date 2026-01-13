"""
HyperNet: Hyperdimensional Graph Encoder module.

Provides the core encoding and decoding functionality for molecular graphs.
"""

from graph_hdc.hypernet.configs import (
    DecoderSettings,
    DSHDCConfig,
    FallbackDecoderSettings,
    Features,
    get_config,
)
from graph_hdc.hypernet.encoder import CorrectionLevel, DecodingResult, HyperNet
from graph_hdc.hypernet.types import Feat, VSAModel

__all__ = [
    "HyperNet",
    "CorrectionLevel",
    "DecodingResult",
    "get_config",
    "DSHDCConfig",
    "DecoderSettings",
    "FallbackDecoderSettings",
    "Features",
    "Feat",
    "VSAModel",
]
