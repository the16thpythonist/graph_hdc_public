"""
Models for molecular generation.
"""

from graph_hdc.models.flows import (
    FlowConfig,
    QM9_FLOW_CONFIG,
    RealNVPV3Lightning,
    ZINC_FLOW_CONFIG,
)
from graph_hdc.models.regressors import (
    MolecularProperty,
    PropertyRegressor,
    QM9_LOGP_CONFIG,
    QM9_QED_CONFIG,
    RegressorConfig,
    ZINC_LOGP_CONFIG,
    ZINC_QED_CONFIG,
)
from graph_hdc.models.flow_edge_decoder import (
    FlowEdgeDecoder,
    FlowEdgeDecoderConfig,
    QM9_EDGE_DECODER_CONFIG,
    ZINC_EDGE_DECODER_CONFIG,
    preprocess_for_flow_edge_decoder,
    preprocess_dataset,
    compute_edge_marginals,
    compute_node_counts,
    FLOW_ATOM_TYPES,
    FLOW_ATOM_TO_IDX,
    FLOW_IDX_TO_ATOM,
    NUM_ATOM_CLASSES,
    NUM_EDGE_CLASSES,
    QM9_TO_7CLASS,
    ZINC_TO_7CLASS,
)

__all__ = [
    # Flows
    "RealNVPV3Lightning",
    "FlowConfig",
    "QM9_FLOW_CONFIG",
    "ZINC_FLOW_CONFIG",
    # Regressors
    "PropertyRegressor",
    "RegressorConfig",
    "MolecularProperty",
    "QM9_LOGP_CONFIG",
    "QM9_QED_CONFIG",
    "ZINC_LOGP_CONFIG",
    "ZINC_QED_CONFIG",
    # Flow Edge Decoder
    "FlowEdgeDecoder",
    "FlowEdgeDecoderConfig",
    "QM9_EDGE_DECODER_CONFIG",
    "ZINC_EDGE_DECODER_CONFIG",
    "preprocess_for_flow_edge_decoder",
    "preprocess_dataset",
    "compute_edge_marginals",
    "compute_node_counts",
    "FLOW_ATOM_TYPES",
    "FLOW_ATOM_TO_IDX",
    "FLOW_IDX_TO_ATOM",
    "NUM_ATOM_CLASSES",
    "NUM_EDGE_CLASSES",
    "QM9_TO_7CLASS",
    "ZINC_TO_7CLASS",
]
