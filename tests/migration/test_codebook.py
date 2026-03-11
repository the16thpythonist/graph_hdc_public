"""Migration tests for CombinatoricIntegerEncoder codebook determinism."""

import hashlib
import math

import torch
import pytest

from graph_hdc.hypernet.feature_encoders import CombinatoricIntegerEncoder
from graph_hdc.utils.helpers import TupleIndexer


def _tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def test_codebook_shape(codebook_fixture):
    """Codebook shape must match golden reference."""
    f = codebook_fixture
    bins = f["bins"]
    hv_dim = f["hv_dim"]
    seed = f["seed"]

    indexer = TupleIndexer(bins)
    encoder = CombinatoricIntegerEncoder(
        dim=hv_dim,
        num_categories=math.prod(bins),
        indexer=indexer,
        vsa="HRR",
        seed=seed,
        dtype="float64",
    )

    assert list(encoder.codebook.shape) == f["codebook_shape"]


def test_codebook_hash(codebook_fixture):
    """Codebook bytes must be identical (deterministic generation)."""
    f = codebook_fixture
    bins = f["bins"]
    hv_dim = f["hv_dim"]
    seed = f["seed"]

    indexer = TupleIndexer(bins)
    encoder = CombinatoricIntegerEncoder(
        dim=hv_dim,
        num_categories=math.prod(bins),
        indexer=indexer,
        vsa="HRR",
        seed=seed,
        dtype="float64",
    )

    assert _tensor_sha256(encoder.codebook.cpu()) == f["codebook_hash"]


def test_encode_indices(codebook_fixture):
    """Encoding test tuples must produce the same flat indices."""
    f = codebook_fixture
    test_tuples = [tuple(t) for t in f["test_tuples"]]
    expected_indices = f["encoded_indices"]

    indexer = TupleIndexer(f["bins"])
    actual_indices = indexer.get_idxs(test_tuples)
    assert actual_indices == expected_indices


def test_encode_hypervectors(codebook_fixture):
    """Encoded hypervectors must match golden reference."""
    f = codebook_fixture
    bins = f["bins"]
    hv_dim = f["hv_dim"]
    seed = f["seed"]

    indexer = TupleIndexer(bins)
    encoder = CombinatoricIntegerEncoder(
        dim=hv_dim,
        num_categories=math.prod(bins),
        indexer=indexer,
        vsa="HRR",
        seed=seed,
        dtype="float64",
    )

    indices = torch.tensor(f["encoded_indices"], dtype=torch.long)
    actual_hvs = encoder.codebook[indices]

    torch.testing.assert_close(
        actual_hvs, f["encoded_hvs"],
        atol=1e-12, rtol=0,
    )


def test_decode_roundtrip(codebook_fixture):
    """Encode -> decode roundtrip must match golden reference."""
    f = codebook_fixture
    bins = f["bins"]
    hv_dim = f["hv_dim"]
    seed = f["seed"]

    indexer = TupleIndexer(bins)
    encoder = CombinatoricIntegerEncoder(
        dim=hv_dim,
        num_categories=math.prod(bins),
        indexer=indexer,
        vsa="HRR",
        seed=seed,
        dtype="float64",
    )

    # Decode all hvs as a batch (single-vector decode has a known issue with squeeze)
    decoded_all = encoder.decode(f["encoded_hvs"])
    expected_all = torch.tensor(f["decode_results"])
    torch.testing.assert_close(decoded_all, expected_all)
