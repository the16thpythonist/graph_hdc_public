"""
HyperNet: Hyperdimensional Graph Encoder module.

Provides the core encoding and decoding functionality for molecular graphs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from graph_hdc.hypernet.configs import (
    DecoderSettings,
    DSHDCConfig,
    FallbackDecoderSettings,
    Features,
    RWConfig,
    create_config_with_rw,
    get_config,
)
from graph_hdc.hypernet.encoder import CorrectionLevel, DecodingResult, HyperNet
from graph_hdc.hypernet.multi_hypernet import MultiHyperNet
from graph_hdc.hypernet.rrwp_hypernet import RRWPHyperNet
from graph_hdc.hypernet.types import Feat, VSAModel


def load_hypernet(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> Union[HyperNet, MultiHyperNet]:
    """
    Load a HyperNet or MultiHyperNet from a checkpoint file.

    Probes the checkpoint to detect the type and dispatches to the
    appropriate ``load()`` class method.

    Parameters
    ----------
    path : str | Path
        Path to saved checkpoint.
    device : str | torch.device
        Device to load to.

    Returns
    -------
    HyperNet | MultiHyperNet
        The loaded encoder.
    """
    checkpoint_data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint_data, dict):
        ckpt_type = checkpoint_data.get("type")
        if ckpt_type == "MultiHyperNet":
            return MultiHyperNet.load(path, device=device)
        if ckpt_type == "RRWPHyperNet":
            return RRWPHyperNet.load(path, device=device)
    return HyperNet.load(path, device=device)


__all__ = [
    "HyperNet",
    "MultiHyperNet",
    "RRWPHyperNet",
    "CorrectionLevel",
    "DecodingResult",
    "get_config",
    "create_config_with_rw",
    "DSHDCConfig",
    "RWConfig",
    "DecoderSettings",
    "FallbackDecoderSettings",
    "Features",
    "Feat",
    "VSAModel",
    "load_hypernet",
]
