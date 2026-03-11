"""Backward-compatible re-exports. Canonical: graph_hdc.domains.molecular.metrics"""
from graph_hdc.domains.molecular.metrics import (  # noqa: F401
    GenerationEvaluator,
    calculate_internal_diversity,
    rdkit_logp,
    rdkit_max_ring_size,
    rdkit_qed,
    rdkit_sa_score,
)
