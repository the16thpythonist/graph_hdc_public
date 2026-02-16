"""
Dataset configurations for HDC encoding.

This module defines configurations for QM9 and ZINC datasets with their
node feature specifications and HDC parameters.
"""

import enum
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, replace
from typing import Literal

from graph_hdc.hypernet.feature_encoders import CombinatoricIntegerEncoder
from graph_hdc.hypernet.types import VSAModel
from graph_hdc.utils.helpers import pick_device_str

IndexRange = tuple[int, int]
BaseDataset = Literal["qm9", "zinc"]


class Features(enum.Enum):
    """Feature type enum."""
    NODE_FEATURES = ("node_features", 0)

    def __new__(cls, value, idx):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.idx = idx
        return obj


@dataclass
class FeatureConfig:
    """Configuration for a single feature's hypervector codebook."""
    count: int
    encoder_cls: type
    index_range: IndexRange = (0, 1)
    idx_offset: int = 0
    bins: list[int] | None = None


@dataclass
class RWConfig:
    """Configuration for random walk return probability features.

    When *bin_boundaries* is set, quantile-based binning is used instead
    of uniform bins on ``[0, 1]``.  Use
    :data:`graph_hdc.utils.rw_features.ZINC_RW_QUANTILE_BOUNDARIES` for
    a ready-made preset derived from 5 000 ZINC training molecules.
    """
    enabled: bool = False
    k_values: tuple[int, ...] = (3, 6)
    num_bins: int = 10
    bin_boundaries: dict[int, list[float]] | None = None


@dataclass
class DSHDCConfig:
    """Configuration for hyperdimensional encoding of a dataset."""
    name: str
    hv_dim: int = 256
    hv_count: int = 2
    vsa: VSAModel = field(default_factory=lambda: VSAModel.HRR)
    node_feature_configs: dict[Features, FeatureConfig] = field(default_factory=OrderedDict)
    edge_feature_configs: dict[Features, FeatureConfig] | None = field(default_factory=OrderedDict)
    graph_feature_configs: dict[Features, FeatureConfig] | None = field(default_factory=OrderedDict)
    device: str = "cpu"
    seed: int | None = None
    nha_bins: int | None = None
    nha_depth: int | None = None
    dtype: str = "float64"
    base_dataset: BaseDataset = "qm9"
    hypernet_depth: int = 3
    normalize: bool = False
    rw_config: RWConfig = field(default_factory=RWConfig)
    prune_codebook: bool = True


@dataclass
class FallbackDecoderSettings:
    """Settings for the greedy fallback decoder."""
    initial_limit: int = 4096
    limit: int = 4096
    beam_size: int = 4096
    pruning_method: str = "cos_sim"
    use_size_aware_pruning: bool = True
    use_one_initial_population: bool = False
    use_g3_instead_of_h3: bool = False
    use_modified_graph_embedding: bool = False
    random_sample_ratio: float = 0.0
    validate_ring_structure: bool | None = None
    top_k: int | None = None

    @property
    def pruning_fn(self) -> str:
        return self.pruning_method

    @property
    def graph_embedding_attr(self) -> str:
        return "g3" if self.use_g3_instead_of_h3 else "graph_embedding"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DecoderSettings:
    """Main decoder settings for graph decoding."""
    iteration_budget: int = 3
    max_graphs_per_iter: int = 1024
    _top_k: int = field(default=3, init=False, repr=False)
    sim_eps: float = 0.0001
    early_stopping: bool = False
    prefer_smaller_corrective_edits: bool = False
    use_correction: bool = True
    _validate_ring_structure: bool = field(default=False, init=False, repr=False)
    fallback_decoder_settings: FallbackDecoderSettings = field(default_factory=FallbackDecoderSettings)
    max_solutions: int = 1000

    def __post_init__(self):
        if not hasattr(self, "_top_k"):
            self._top_k = 10
        if not hasattr(self, "_validate_ring_structure"):
            self._validate_ring_structure = False

        updates = {}
        if self.fallback_decoder_settings.top_k is None:
            updates["top_k"] = self._top_k
        if self.fallback_decoder_settings.validate_ring_structure is None:
            updates["validate_ring_structure"] = self._validate_ring_structure

        if updates:
            self.fallback_decoder_settings = replace(self.fallback_decoder_settings, **updates)

    @property
    def top_k(self) -> int:
        return self._top_k

    @top_k.setter
    def top_k(self, value: int) -> None:
        self._top_k = value
        self.fallback_decoder_settings = replace(self.fallback_decoder_settings, top_k=value)

    @property
    def validate_ring_structure(self) -> bool:
        return self._validate_ring_structure

    @validate_ring_structure.setter
    def validate_ring_structure(self, value: bool) -> None:
        self._validate_ring_structure = value
        self.fallback_decoder_settings = replace(self.fallback_decoder_settings, validate_ring_structure=value)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def get_default_for(cls, base_dataset: BaseDataset) -> "DecoderSettings":
        """Returns default decoder settings for a given dataset."""
        if base_dataset == "qm9":
            return cls(
                iteration_budget=3,
                max_graphs_per_iter=1024,
                fallback_decoder_settings=FallbackDecoderSettings(limit=2048, beam_size=2048),
            )
        return cls(
            iteration_budget=25,
            max_graphs_per_iter=1024,
            fallback_decoder_settings=FallbackDecoderSettings(limit=1024, beam_size=64),
        )


def _get_qm9_config(hv_dim: int = 256) -> DSHDCConfig:
    """Create QM9 dataset configuration."""

    return DSHDCConfig(
        seed=42,
        name=f"QM9SmilesHRR{hv_dim}F64G1NG3",
        base_dataset="qm9",
        vsa=VSAModel.HRR,
        hv_dim=hv_dim,
        device=pick_device_str(),
        node_feature_configs=OrderedDict([
            (
                Features.NODE_FEATURES,
                FeatureConfig(
                    count=math.prod([4, 5, 3, 5]),
                    encoder_cls=CombinatoricIntegerEncoder,
                    index_range=IndexRange((0, 4)),
                    bins=[4, 5, 3, 5],
                ),
            ),
        ]),
        normalize=True,
        hypernet_depth=3,
        dtype="float64",
        hv_count=2,
    )


def _get_zinc_config(hv_dim: int = 256) -> DSHDCConfig:
    """Create ZINC dataset configuration."""

    return DSHDCConfig(
        seed=42,
        name=f"ZincSmilesHRR{hv_dim}F645G1NG4",
        base_dataset="zinc",
        vsa=VSAModel.HRR,
        hv_dim=hv_dim,
        device=pick_device_str(),
        node_feature_configs=OrderedDict([
            (
                Features.NODE_FEATURES,
                FeatureConfig(
                    count=math.prod([9, 6, 3, 4, 2]),
                    encoder_cls=CombinatoricIntegerEncoder,
                    index_range=IndexRange((0, 5)),
                    bins=[9, 6, 3, 4, 2],
                ),
            ),
        ]),
        normalize=True,
        hypernet_depth=4,
        dtype="float64",
    )


class SupportedDataset(enum.Enum):
    """Supported datasets with their configurations."""
    QM9_SMILES_HRR_256_F64_G1NG3 = ("QM9_SMILES_HRR_256_F64_G1NG3", None)
    ZINC_SMILES_HRR_256_F64_5G1NG4 = ("ZINC_SMILES_HRR_256_F64_5G1NG4", None)

    def __new__(cls, value: str, _):
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    @property
    def default_cfg(self) -> DSHDCConfig:
        """Get the default configuration for this dataset."""
        if "QM9" in self._value_:
            return _get_qm9_config(hv_dim=256)
        return _get_zinc_config(hv_dim=256)


def get_config(dataset_name: str) -> DSHDCConfig:
    """Get dataset configuration by name."""
    try:
        return SupportedDataset(dataset_name).default_cfg
    except ValueError:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {[e.value for e in SupportedDataset]}")


# ── Dataset-specific base bins ──────────────────────────────────────

_BASE_BINS: dict[BaseDataset, list[int]] = {
    "qm9": [4, 5, 3, 5],
    "zinc": [9, 6, 3, 4, 2],
}

_BASE_DEPTH: dict[BaseDataset, int] = {
    "qm9": 3,
    "zinc": 4,
}


def create_config_with_rw(
    base_dataset: BaseDataset,
    hv_dim: int,
    rw_config: RWConfig | None = None,
    **overrides,
) -> DSHDCConfig:
    """
    Create a DSHDCConfig with optional RW feature augmentation.

    This factory decouples the HDC representation from the dataset choice.
    When ``rw_config.enabled`` is True, the node feature bins and index range
    are extended with ``[num_bins] * len(k_values)`` additional dimensions.

    Parameters
    ----------
    base_dataset : {"qm9", "zinc"}
        Which dataset's base feature scheme to use.
    hv_dim : int
        Hypervector dimensionality.
    rw_config : RWConfig, optional
        Random walk configuration. Defaults to ``RWConfig()`` (disabled).
    **overrides
        Additional keyword arguments forwarded to ``DSHDCConfig``.

    Returns
    -------
    DSHDCConfig
    """
    if rw_config is None:
        rw_config = RWConfig()

    bins = list(_BASE_BINS[base_dataset])
    base_dim = len(bins)

    if rw_config.enabled:
        bins.extend([rw_config.num_bins] * len(rw_config.k_values))

    total_dim = len(bins)

    rw_suffix = "RW" if rw_config.enabled else ""
    name = f"{base_dataset.upper()}{rw_suffix}HRR{hv_dim}"

    cfg_kwargs = dict(
        name=name,
        base_dataset=base_dataset,
        vsa=VSAModel.HRR,
        hv_dim=hv_dim,
        seed=42,
        device=pick_device_str(),
        node_feature_configs=OrderedDict([
            (
                Features.NODE_FEATURES,
                FeatureConfig(
                    count=math.prod(bins),
                    encoder_cls=CombinatoricIntegerEncoder,
                    index_range=IndexRange((0, total_dim)),
                    bins=bins,
                ),
            ),
        ]),
        normalize=True,
        hypernet_depth=_BASE_DEPTH[base_dataset],
        dtype="float64",
        hv_count=2,
        rw_config=rw_config,
    )
    cfg_kwargs.update(overrides)
    return DSHDCConfig(**cfg_kwargs)
