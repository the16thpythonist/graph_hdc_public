"""
Utility functions for graph_hdc.
"""

from graph_hdc.utils.chem import canonical_key, is_valid_molecule, reconstruct_for_eval
from graph_hdc.utils.evaluator import (
    GenerationEvaluator,
    calculate_internal_diversity,
    rdkit_logp,
    rdkit_qed,
    rdkit_sa_score,
)
from graph_hdc.utils.experiment_helpers import (
    GracefulInterruptHandler,
    LossTrackingCallback,
    ReconstructionVisualizationCallback,
    TrainingMetricsCallback,
    compute_tanimoto_similarity,
    create_hdc_config,
    create_reconstruction_plot,
    decode_nodes_from_hdc,
    draw_mol_or_error,
    get_canonical_smiles,
    is_valid_mol,
    load_or_create_encoder,
    pyg_to_mol,
)
from graph_hdc.utils.helpers import DataTransformer, TupleIndexer, pick_device

__all__ = [
    # Evaluator
    "GenerationEvaluator",
    "rdkit_logp",
    "rdkit_qed",
    "rdkit_sa_score",
    "calculate_internal_diversity",
    # Helpers
    "DataTransformer",
    "TupleIndexer",
    "pick_device",
    # Chemistry
    "is_valid_molecule",
    "canonical_key",
    "reconstruct_for_eval",
    # Experiment helpers
    "create_hdc_config",
    "load_or_create_encoder",
    "decode_nodes_from_hdc",
    "pyg_to_mol",
    "is_valid_mol",
    "get_canonical_smiles",
    "draw_mol_or_error",
    "create_reconstruction_plot",
    "compute_tanimoto_similarity",
    "LossTrackingCallback",
    "ReconstructionVisualizationCallback",
    "TrainingMetricsCallback",
    "GracefulInterruptHandler",
]
