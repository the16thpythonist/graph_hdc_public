"""
Tests to isolate the TypeError in compute_hdc_distance when encoding
generated molecules with MultiHyperNet.

The error chain:
  compute_hdc_distance → hypernet.encode_properties → _slice_encode_bind
  → CombinatoricIntegerEncoder.encode → indexer.get_idxs → None values
  → torch.tensor([..., None, ...]) → TypeError
"""

import pytest
import torch
from rdkit import Chem
from torch_geometric.data import Data

from graph_hdc.hypernet import load_hypernet
from graph_hdc.models.flow_edge_decoder import (
    FlowEdgeDecoder,
    NODE_FEATURE_BINS,
    node_tuples_to_onehot,
    onehot_to_raw_features,
)
from graph_hdc.utils.experiment_helpers import (
    compute_hdc_distance,
    decode_nodes_from_hdc,
    smiles_to_pyg_data,
)
from graph_hdc.utils.helpers import scatter_hd, TupleIndexer

# Paths to the trained models
ENCODER_PATH = "experiments/decoding/results/train_flow_edge_decoder_streaming/GOOD/hypernet_encoder.ckpt"
DECODER_PATH = "experiments/decoding/results/train_flow_edge_decoder_streaming/GOOD/last.ckpt"


@pytest.fixture(scope="module")
def hypernet():
    hn = load_hypernet(ENCODER_PATH, device="cpu")
    hn.rebuild_unpruned_codebook()
    return hn


@pytest.fixture(scope="module")
def decoder():
    return FlowEdgeDecoder.load(DECODER_PATH, device="cpu")


# ─────────────────────────────────────────────────────────────────────
# 1. Test that the TupleIndexer covers all valid index combinations
# ─────────────────────────────────────────────────────────────────────

class TestTupleIndexerCoverage:
    """Verify that every possible argmax result maps to a valid codebook entry."""

    def test_all_combinations_exist(self):
        """Every tuple in the Cartesian product of NODE_FEATURE_BINS
        should exist in the TupleIndexer."""
        bins = NODE_FEATURE_BINS  # [9, 6, 3, 4, 2]
        indexer = TupleIndexer(bins)

        missing = []
        for a in range(bins[0]):
            for b in range(bins[1]):
                for c in range(bins[2]):
                    for d in range(bins[3]):
                        for e in range(bins[4]):
                            tup = (a, b, c, d, e)
                            idx = indexer.get_idx(tup)
                            if idx is None:
                                missing.append(tup)

        assert len(missing) == 0, f"Missing tuples in TupleIndexer: {missing[:10]}..."

    def test_indexer_from_loaded_encoder(self, hypernet):
        """Check that the loaded HyperNet's node encoder indexer has
        the expected sizes and covers all combinations."""
        primary = hypernet._primary if hasattr(hypernet, "_primary") else hypernet
        enc_map = primary.node_encoder_map

        for name, (encoder, idx_range) in enc_map.items():
            sizes = encoder.indexer.sizes
            print(f"Encoder '{name}': range={idx_range}, sizes={sizes}, "
                  f"idx_offset={encoder.idx_offset}, "
                  f"codebook_len={len(encoder.indexer.tuple_to_idx)}")

            expected_count = 1
            for s in sizes:
                expected_count *= s
            assert len(encoder.indexer.tuple_to_idx) == expected_count, (
                f"Codebook size {len(encoder.indexer.tuple_to_idx)} != "
                f"expected {expected_count} for sizes {sizes}"
            )


# ─────────────────────────────────────────────────────────────────────
# 2. Test the one-hot ↔ raw features round-trip
# ─────────────────────────────────────────────────────────────────────

class TestOnehotRawRoundTrip:
    """Verify that onehot → raw → encode doesn't produce None indices."""

    def test_roundtrip_identity(self):
        """Convert raw features to one-hot and back; verify exact match."""
        bins = NODE_FEATURE_BINS  # [9, 6, 3, 4, 2]
        # Create a few representative raw feature vectors
        raw = torch.tensor([
            [0, 0, 0, 0, 0],  # all zeros
            [8, 5, 2, 3, 1],  # all maxes
            [1, 2, 1, 1, 0],  # Carbon-like
            [5, 1, 0, 3, 1],  # Nitrogen-like
        ], dtype=torch.float)

        onehot = node_tuples_to_onehot(
            [tuple(int(v) for v in row) for row in raw.tolist()],
            device="cpu",
        )
        raw_back = onehot_to_raw_features(onehot)

        assert torch.allclose(raw, raw_back), (
            f"Round-trip mismatch!\n  original: {raw}\n  recovered: {raw_back}"
        )

    def test_roundtrip_all_atom_types(self):
        """Verify round-trip for every atom type index (0-8)."""
        bins = NODE_FEATURE_BINS
        for atom_idx in range(bins[0]):
            raw = torch.tensor([[atom_idx, 1, 0, 0, 0]], dtype=torch.float)
            onehot = node_tuples_to_onehot(
                [(atom_idx, 1, 0, 0, 0)], device="cpu"
            )
            raw_back = onehot_to_raw_features(onehot)
            assert raw_back[0, 0].item() == atom_idx, (
                f"Atom type {atom_idx}: expected {atom_idx}, got {raw_back[0, 0].item()}"
            )

    def test_raw_features_encode_with_hypernet(self, hypernet):
        """Raw features from onehot_to_raw_features should be encodable
        by the HyperNet without errors."""
        bins = NODE_FEATURE_BINS
        raw = torch.tensor([
            [1, 2, 0, 1, 0],
            [5, 1, 0, 3, 1],
        ], dtype=torch.float)

        data = Data(
            x=raw,
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        )
        data.batch = torch.zeros(2, dtype=torch.long)

        # This is the call that fails in compute_hdc_distance
        data = hypernet.encode_properties(data)
        assert data.node_hv is not None
        assert data.node_hv.shape[0] == 2

    def test_onehot_from_dense_to_pyg_encodes_correctly(self, hypernet):
        """Simulate what dense_to_pyg produces and verify it encodes."""
        # dense_to_pyg outputs 24-dim one-hot x
        # Create a plausible one-hot x
        bins = NODE_FEATURE_BINS
        raw_indices = torch.tensor([
            [1, 2, 0, 1, 0],
            [5, 1, 0, 3, 1],
        ], dtype=torch.long)

        # Build one-hot manually (same as dense_to_pyg)
        onehot_parts = []
        for i, b in enumerate(bins):
            onehot_parts.append(
                torch.nn.functional.one_hot(raw_indices[:, i], num_classes=b).float()
            )
        x_onehot = torch.cat(onehot_parts, dim=-1)
        assert x_onehot.shape == (2, 24)

        # Reverse to raw (what compute_hdc_distance does)
        raw_back = onehot_to_raw_features(x_onehot)
        assert raw_back.shape == (2, 5)

        # Encode with HyperNet
        data = Data(
            x=raw_back,
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        )
        data.batch = torch.zeros(2, dtype=torch.long)
        data = hypernet.encode_properties(data)
        assert data.node_hv is not None


# ─────────────────────────────────────────────────────────────────────
# 3. Test the actual failing flow: decode nodes → generate edges → score
# ─────────────────────────────────────────────────────────────────────

class TestComputeHDCDistanceFlow:
    """Reproduce the exact flow from test_flow_edge_decoder.py."""

    def _encode_smiles(self, hypernet, smiles: str):
        """Replicate the encoding steps from the test experiment."""
        data = smiles_to_pyg_data(smiles, "zinc")
        assert data is not None, f"Failed to parse {smiles}"

        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)

        with torch.no_grad():
            data = hypernet.encode_properties(data)
            order_zero = scatter_hd(
                src=data.node_hv, index=data.batch, op="bundle"
            )
            encoder_output = hypernet.forward(data, normalize=True)
            order_n = encoder_output["graph_embedding"]
            hdc_vector = torch.cat([order_zero, order_n], dim=-1).squeeze(0)

        return data, hdc_vector

    def test_decode_nodes_produce_valid_tuples(self, hypernet):
        """Check that decoded node tuples are within valid ZINC ranges."""
        bins = NODE_FEATURE_BINS  # [9, 6, 3, 4, 2]
        data, hdc_vector = self._encode_smiles(hypernet, "CCO")
        base_hdc_dim = hypernet.hv_dim

        node_tuples, num_nodes = decode_nodes_from_hdc(
            hypernet, hdc_vector.unsqueeze(0), base_hdc_dim
        )
        print(f"Decoded {num_nodes} nodes: {node_tuples}")

        for i, tup in enumerate(node_tuples):
            for dim_idx, (val, max_val) in enumerate(zip(tup, bins)):
                assert 0 <= val < max_val, (
                    f"Node {i}, feature {dim_idx}: value {val} outside "
                    f"valid range [0, {max_val})"
                )

    def test_node_tuples_to_onehot_roundtrip(self, hypernet):
        """Verify node_tuples_to_onehot → onehot_to_raw_features round-trip
        with actual decoded tuples."""
        data, hdc_vector = self._encode_smiles(hypernet, "CCO")
        base_hdc_dim = hypernet.hv_dim

        node_tuples, num_nodes = decode_nodes_from_hdc(
            hypernet, hdc_vector.unsqueeze(0), base_hdc_dim
        )

        # This is what the test experiment does
        onehot = node_tuples_to_onehot(node_tuples, device="cpu")
        print(f"One-hot shape: {onehot.shape}")
        assert onehot.shape == (num_nodes, 24)

        # This is what compute_hdc_distance does
        raw_back = onehot_to_raw_features(onehot)
        print(f"Raw features shape: {raw_back.shape}")
        print(f"Raw features:\n{raw_back}")

        # Verify they match the original tuples
        for i, tup in enumerate(node_tuples):
            recovered = tuple(int(v) for v in raw_back[i].tolist())
            assert recovered == tup, (
                f"Node {i}: original={tup}, recovered={recovered}"
            )

    def test_compute_hdc_distance_with_real_smiles(self, hypernet, decoder):
        """Run the full compute_hdc_distance flow for a simple molecule."""
        data, hdc_vector = self._encode_smiles(hypernet, "CCO")
        base_hdc_dim = hypernet.hv_dim

        # Decode nodes
        node_tuples, num_nodes = decode_nodes_from_hdc(
            hypernet, hdc_vector.unsqueeze(0), base_hdc_dim
        )
        assert num_nodes > 0, "No nodes decoded"

        # Prepare inputs for decoder (same as test experiment)
        node_features = node_tuples_to_onehot(
            node_tuples, device="cpu"
        ).unsqueeze(0)
        node_mask = torch.ones(1, num_nodes, dtype=torch.bool)
        hdc_vectors = hdc_vector.unsqueeze(0)

        # Generate edges (single sample, few steps)
        with torch.no_grad():
            samples = decoder.sample(
                hdc_vectors=hdc_vectors,
                node_features=node_features,
                node_mask=node_mask,
                sample_steps=10,
                eta=0.0,
                omega=0.0,
                time_distortion="polydec",
                show_progress=False,
                device=torch.device("cpu"),
            )

        generated_data = samples[0]
        print(f"Generated data.x shape: {generated_data.x.shape}")
        print(f"Generated data.x dtype: {generated_data.x.dtype}")
        print(f"Generated data.x[:3]:\n{generated_data.x[:3]}")

        # Now test the raw features conversion
        raw_features = onehot_to_raw_features(generated_data.x)
        print(f"Raw features shape: {raw_features.shape}")
        print(f"Raw features dtype: {raw_features.dtype}")
        print(f"Raw features:\n{raw_features}")

        # Check all values are within ZINC bins
        bins = NODE_FEATURE_BINS
        for node_idx in range(raw_features.shape[0]):
            tup = tuple(int(v) for v in raw_features[node_idx].tolist())
            for feat_idx, (val, max_val) in enumerate(zip(tup, bins)):
                assert 0 <= val < max_val, (
                    f"Node {node_idx}, feature {feat_idx}: "
                    f"value {val} outside [0, {max_val}). Full tuple: {tup}"
                )

        # Now call compute_hdc_distance — the function that actually fails
        distance = compute_hdc_distance(
            generated_data=generated_data,
            original_hdc_vectors=hdc_vectors,
            base_hdc_dim=base_hdc_dim,
            hypernet=hypernet,
            device=torch.device("cpu"),
            dataset="zinc",
            original_x=None,  # Force the onehot→raw path
        )
        print(f"HDC distance: {distance}")
        assert distance != float("inf"), "compute_hdc_distance returned inf (exception occurred)"

    def test_compute_hdc_distance_with_original_x(self, hypernet, decoder):
        """Test the original_x code path (when raw features are provided)."""
        data, hdc_vector = self._encode_smiles(hypernet, "CCO")
        base_hdc_dim = hypernet.hv_dim

        node_tuples, num_nodes = decode_nodes_from_hdc(
            hypernet, hdc_vector.unsqueeze(0), base_hdc_dim
        )

        node_features = node_tuples_to_onehot(
            node_tuples, device="cpu"
        ).unsqueeze(0)
        node_mask = torch.ones(1, num_nodes, dtype=torch.bool)
        hdc_vectors = hdc_vector.unsqueeze(0)

        with torch.no_grad():
            samples = decoder.sample(
                hdc_vectors=hdc_vectors,
                node_features=node_features,
                node_mask=node_mask,
                sample_steps=10,
                show_progress=False,
                device=torch.device("cpu"),
            )

        generated_data = samples[0]

        # Provide original_x as the test experiment would
        # Note: original_data from smiles_to_pyg_data does NOT have original_x
        raw_x = getattr(data, "original_x", None)
        print(f"original_x from data: {raw_x}")

        # Test with explicit raw features
        raw_from_tuples = torch.tensor(
            [list(t) for t in node_tuples], dtype=torch.float
        )
        distance = compute_hdc_distance(
            generated_data=generated_data,
            original_hdc_vectors=hdc_vectors,
            base_hdc_dim=base_hdc_dim,
            hypernet=hypernet,
            device=torch.device("cpu"),
            dataset="zinc",
            original_x=raw_from_tuples,
        )
        print(f"HDC distance (with original_x): {distance}")
        assert distance != float("inf"), "compute_hdc_distance returned inf"


# ─────────────────────────────────────────────────────────────────────
# 4. Direct test of encode_properties with various x formats
# ─────────────────────────────────────────────────────────────────────

class TestEncodePropertiesFormats:
    """Test encode_properties with different data.x formats to find
    what exactly triggers the None in the indexer."""

    def test_encode_with_int_features(self, hypernet):
        """Encode with integer raw features (the expected format)."""
        raw = torch.tensor([[1, 2, 0, 1, 0]], dtype=torch.long)
        data = Data(
            x=raw.float(),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )
        data.batch = torch.zeros(1, dtype=torch.long)
        data = hypernet.encode_properties(data)
        assert data.node_hv is not None

    def test_encode_with_float_features(self, hypernet):
        """Encode with float raw features (from onehot_to_raw_features)."""
        raw = torch.tensor([[1.0, 2.0, 0.0, 1.0, 0.0]])
        data = Data(
            x=raw,
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )
        data.batch = torch.zeros(1, dtype=torch.long)
        data = hypernet.encode_properties(data)
        assert data.node_hv is not None

    def test_encode_with_onehot_features_fails(self, hypernet):
        """Encoding with 24-dim one-hot features should fail or produce
        wrong results, since the encoder expects 5-dim raw features."""
        # This would be the case if someone accidentally passed one-hot
        # features directly to encode_properties
        bins = NODE_FEATURE_BINS
        raw = torch.tensor([[1, 2, 0, 1, 0]], dtype=torch.long)
        onehot_parts = []
        for i, b in enumerate(bins):
            onehot_parts.append(
                torch.nn.functional.one_hot(raw[:, i], num_classes=b).float()
            )
        x_onehot = torch.cat(onehot_parts, dim=-1)  # (1, 24)

        data = Data(
            x=x_onehot,
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )
        data.batch = torch.zeros(1, dtype=torch.long)

        # The encoder map slices x[:, 0:5] — these would be the first 5
        # values of the one-hot encoding, NOT raw feature indices.
        # This should either error or produce garbage.
        print(f"One-hot x[:, 0:5] = {x_onehot[0, :5].tolist()}")
        print(f"  (these would be interpreted as raw feature values)")

        # Check what tuples this would produce
        feat_slice = x_onehot[:, 0:5]
        tup = feat_slice.long().tolist()[0]
        print(f"  Tuple that would be looked up: {tuple(tup)}")

        indexer = TupleIndexer(bins)
        idx = indexer.get_idx(tuple(tup))
        print(f"  Indexer result: {idx}")

    def test_what_node_encoder_map_slices(self, hypernet):
        """Print exactly what the encoder map does to various x shapes."""
        primary = hypernet._primary if hasattr(hypernet, "_primary") else hypernet

        for name, (encoder, (start, end)) in primary.node_encoder_map.items():
            print(f"\nEncoder '{name}': slices x[:, {start}:{end}]")
            print(f"  indexer sizes: {encoder.indexer.sizes}")
            print(f"  idx_offset: {encoder.idx_offset}")
            print(f"  Expected feature dim: {end - start}")

            # Test with a 5-dim raw feature vector
            raw_5 = torch.tensor([[1.0, 2.0, 0.0, 1.0, 0.0]])
            feat = raw_5[:, start:end]
            print(f"  Slice from 5-dim raw: {feat.tolist()}")

            # Test with a 24-dim one-hot vector (what dense_to_pyg produces)
            raw_24 = torch.zeros(1, 24)
            raw_24[0, 1] = 1.0  # atom_type=1
            raw_24[0, 9 + 2] = 1.0  # degree=2
            feat_24 = raw_24[:, start:end]
            print(f"  Slice from 24-dim onehot: {feat_24.tolist()}")
            print(f"  (Would be interpreted as raw features!)")


# ─────────────────────────────────────────────────────────────────────
# 5. Directly inspect what happens inside compute_hdc_distance
# ─────────────────────────────────────────────────────────────────────

class TestDebugComputeHDCDistance:
    """Trace through compute_hdc_distance step by step to find the
    exact point of failure."""

    def test_trace_encoding_step_by_step(self, hypernet, decoder):
        """Manually walk through each step of compute_hdc_distance."""
        smiles = "CCO"
        data = smiles_to_pyg_data(smiles, "zinc")
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)

        print(f"Original data.x shape: {data.x.shape}")
        print(f"Original data.x:\n{data.x}")

        with torch.no_grad():
            data = hypernet.encode_properties(data)
            order_zero = scatter_hd(
                src=data.node_hv, index=data.batch, op="bundle"
            )
            encoder_output = hypernet.forward(data, normalize=True)
            order_n = encoder_output["graph_embedding"]
            hdc_vector = torch.cat([order_zero, order_n], dim=-1).squeeze(0)

        base_hdc_dim = hypernet.hv_dim
        node_tuples, num_nodes = decode_nodes_from_hdc(
            hypernet, hdc_vector.unsqueeze(0), base_hdc_dim
        )

        node_features = node_tuples_to_onehot(
            node_tuples, device="cpu"
        ).unsqueeze(0)
        node_mask = torch.ones(1, num_nodes, dtype=torch.bool)
        hdc_vectors = hdc_vector.unsqueeze(0)

        with torch.no_grad():
            samples = decoder.sample(
                hdc_vectors=hdc_vectors,
                node_features=node_features,
                node_mask=node_mask,
                sample_steps=10,
                show_progress=False,
                device=torch.device("cpu"),
            )

        generated_data = samples[0]

        # Step 1: What does generated_data.x look like?
        print(f"\n--- Step 1: generated_data.x ---")
        print(f"Shape: {generated_data.x.shape}")
        print(f"Dtype: {generated_data.x.dtype}")
        print(f"Values:\n{generated_data.x}")

        # Step 2: What does onehot_to_raw_features produce?
        print(f"\n--- Step 2: onehot_to_raw_features ---")
        raw_x = onehot_to_raw_features(generated_data.x)
        print(f"Shape: {raw_x.shape}")
        print(f"Dtype: {raw_x.dtype}")
        print(f"Values:\n{raw_x}")

        # Step 3: What tuples does this produce?
        print(f"\n--- Step 3: Feature tuples ---")
        bins = NODE_FEATURE_BINS
        for i in range(raw_x.shape[0]):
            tup = tuple(int(v) for v in raw_x[i].tolist())
            in_range = all(0 <= v < b for v, b in zip(tup, bins))
            indexer = TupleIndexer(bins)
            idx = indexer.get_idx(tup)
            print(f"  Node {i}: {tup} | in_range={in_range} | idx={idx}")

        # Step 4: What does the HyperNet's encoder actually see?
        print(f"\n--- Step 4: HyperNet encoder view ---")
        primary = hypernet._primary if hasattr(hypernet, "_primary") else hypernet
        for name, (encoder, (start, end)) in primary.node_encoder_map.items():
            feat_slice = raw_x[:, start:end]
            print(f"  Encoder '{name}' gets x[:, {start}:{end}]:")
            print(f"    Shape: {feat_slice.shape}")
            print(f"    Values:\n    {feat_slice}")

            # Manually run the encoder's tuple conversion
            tup = feat_slice.long() - encoder.idx_offset
            tup_list = list(map(tuple, tup.tolist()))
            idxs = encoder.indexer.get_idxs(tup_list)
            print(f"    Tuples: {tup_list}")
            print(f"    Indices: {idxs}")
            none_positions = [i for i, x in enumerate(idxs) if x is None]
            if none_positions:
                print(f"    *** None at positions: {none_positions} ***")
                for pos in none_positions:
                    print(f"        Failing tuple: {tup_list[pos]}")

        # Step 5: Try encode_properties and see if it fails
        print(f"\n--- Step 5: encode_properties ---")
        gen_data = Data(
            x=raw_x,
            edge_index=generated_data.edge_index,
        )
        gen_data.batch = torch.zeros(gen_data.x.size(0), dtype=torch.long)

        try:
            gen_data = hypernet.encode_properties(gen_data)
            print("SUCCESS: encode_properties completed")
        except Exception as exc:
            print(f"FAILED: {exc}")
            # Print the exact state that caused the failure
            raise
