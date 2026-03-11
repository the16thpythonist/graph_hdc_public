"""Backward-compatible re-exports. Canonical: graph_hdc.domains.molecular.chem"""
from graph_hdc.domains.molecular.chem import (  # noqa: F401
    FORMAL_CHARGE_IDX_TO_VAL,
    QM9_ATOM_SYMBOLS,
    QM9_ATOM_TO_IDX,
    RECONSTRUCTION_CONFIDENCE,
    ReconstructionResult,
    ZINC_ATOM_SYMBOLS,
    ZINC_ATOM_TO_IDX,
    canonical_key,
    compute_qed,
    draw_mol,
    is_valid_molecule,
    mol_to_data,
    nx_to_mol,
    reconstruct_for_eval,
)
