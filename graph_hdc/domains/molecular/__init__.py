"""
Molecular graph domain for GraphHDC.

Provides :class:`MolecularDomain` with a sensible default feature schema,
plus dataset-specific presets (:class:`QM9MolecularDomain`,
:class:`ZINCMolecularDomain`).

Domain classes are loaded lazily to avoid circular imports when
``preprocessing`` constants are needed early in the import chain
(e.g. by ``streaming_fragments``).
"""

# Lightweight preprocessing constants — no circular dependencies
from graph_hdc.domains.molecular.preprocessing import (
    BOND_TYPE_TO_IDX,
    BOND_TYPES,
    NODE_FEATURE_BINS,
    NODE_FEATURE_DIM,
    NUM_ATOM_CLASSES,
    NUM_BOND_CLASSES,
    NUM_CHARGE_CLASSES,
    NUM_DEGREE_CLASSES,
    NUM_EDGE_CLASSES,
    NUM_HS_CLASSES,
    NUM_RING_CLASSES,
    ZINC_ATOM_TO_IDX,
    ZINC_ATOM_TYPES,
    ZINC_EDGE_DECODER_CONFIG,
    ZINC_FEATURE_BINS,
    ZINC_FEATURE_DIM,
    ZINC_IDX_TO_ATOM,
    get_bond_type_idx,
    preprocess_dataset,
    preprocess_for_flow_edge_decoder,
)

__all__ = [
    # Domain classes (lazy)
    "DegreeEncoder",
    "FormalChargeEncoder",
    "MolecularDomain",
    "QM9MolecularDomain",
    "SmilesDataset",
    "ZINCMolecularDomain",
    # Chem utilities (lazy)
    "canonical_key",
    "is_valid_molecule",
    "reconstruct_for_eval",
    "nx_to_mol",
    "mol_to_data",
    "draw_mol",
    # Metrics (lazy)
    "GenerationEvaluator",
    "rdkit_logp",
    "rdkit_qed",
    "rdkit_sa_score",
    "calculate_internal_diversity",
    # Preprocessing constants
    "BOND_TYPE_TO_IDX",
    "BOND_TYPES",
    "NODE_FEATURE_BINS",
    "NODE_FEATURE_DIM",
    "NUM_ATOM_CLASSES",
    "NUM_BOND_CLASSES",
    "NUM_CHARGE_CLASSES",
    "NUM_DEGREE_CLASSES",
    "NUM_EDGE_CLASSES",
    "NUM_HS_CLASSES",
    "NUM_RING_CLASSES",
    "ZINC_ATOM_TO_IDX",
    "ZINC_ATOM_TYPES",
    "ZINC_EDGE_DECODER_CONFIG",
    "ZINC_FEATURE_BINS",
    "ZINC_FEATURE_DIM",
    "ZINC_IDX_TO_ATOM",
    "get_bond_type_idx",
    "preprocess_dataset",
    "preprocess_for_flow_edge_decoder",
]

# Lazy imports for domain classes to avoid circular dependencies.
# These modules import from graph_hdc.utils.chem which can cause cycles
# when streaming_fragments.py triggers this __init__.py early.
_LAZY_IMPORTS = {
    # Domain classes
    "MolecularDomain": "graph_hdc.domains.molecular.domain",
    "QM9MolecularDomain": "graph_hdc.domains.molecular.domain",
    "ZINCMolecularDomain": "graph_hdc.domains.molecular.domain",
    "SmilesDataset": "graph_hdc.domains.molecular.datasets",
    "DegreeEncoder": "graph_hdc.domains.molecular.encoders",
    "FormalChargeEncoder": "graph_hdc.domains.molecular.encoders",
    # Chem utilities
    "canonical_key": "graph_hdc.domains.molecular.chem",
    "is_valid_molecule": "graph_hdc.domains.molecular.chem",
    "reconstruct_for_eval": "graph_hdc.domains.molecular.chem",
    "nx_to_mol": "graph_hdc.domains.molecular.chem",
    "mol_to_data": "graph_hdc.domains.molecular.chem",
    "draw_mol": "graph_hdc.domains.molecular.chem",
    # Metrics
    "GenerationEvaluator": "graph_hdc.domains.molecular.metrics",
    "rdkit_logp": "graph_hdc.domains.molecular.metrics",
    "rdkit_qed": "graph_hdc.domains.molecular.metrics",
    "rdkit_sa_score": "graph_hdc.domains.molecular.metrics",
    "calculate_internal_diversity": "graph_hdc.domains.molecular.metrics",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
