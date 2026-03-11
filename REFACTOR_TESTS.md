# Migration Reference Tests

Safety net for the domain-agnostic refactoring. We export input/output pairs from the **current working codebase** as golden fixtures, then write tests that the **refactored code must pass**.

## Strategy

1. **Export script** (`tests/migration/export_golden.py`): Runs against the current codebase, produces fixture files.
2. **Golden fixtures** (`tests/migration/fixtures/`): Stored in the repo. `.pt` files for tensors, `.json` for simple structures.
3. **Migration tests** (`tests/migration/test_*.py`): Load fixtures, instantiate refactored components, compare outputs.

All exports use `torch.manual_seed(42)` and run on CPU with `float64` for reproducibility.

## Test Molecules

Pick 10 molecules from each dataset covering a range of sizes and feature diversity:

**QM9** (features: `[atom_type, degree-1, formal_charge, total_Hs]`, bins `[4, 5, 3, 5]`):
- `"C"` — methane, single atom
- `"CC"` — ethane, minimal chain
- `"CCO"` — ethanol, 3 atoms, O heteroatom
- `"C=O"` — formaldehyde, double bond
- `"C#N"` — hydrogen cyanide, triple bond
- `"c1ccccc1"` — benzene, aromatic ring
- `"CC(=O)O"` — acetic acid, branching + double bond
- `"c1ccncc1"` — pyridine, N in aromatic ring
- `"C1CC1"` — cyclopropane, small ring
- `"OC(=O)c1ccccc1"` — benzoic acid, 9 heavy atoms (near QM9 max)

**ZINC** (features: `[atom_type, degree-1, formal_charge, total_Hs, is_in_ring]`, bins `[9, 6, 3, 4, 2]`):
- `"CC"` — ethane, minimal
- `"c1ccccc1"` — benzene, simple ring
- `"c1ccc(cc1)N"` — aniline, ring + NH2
- `"O=C(O)c1ccccc1"` — benzoic acid, multiple bond types
- `"c1ccc(cc1)Cl"` — chlorobenzene, halogen
- `"c1ccc2c(c1)cccc2"` — naphthalene, fused rings
- `"CC(=O)Nc1ccccc1"` — acetanilide, amide bond
- `"O=S(=O)(O)c1ccccc1"` — benzenesulfonic acid, S + high degree
- `"c1cc(ccc1F)Br"` — 4-bromofluorobenzene, two halogens
- `"c1ccc(cc1)c2ccccc2"` — biphenyl, two connected rings

## Components & What to Export

### 1. CombinatoricIntegerEncoder

**What**: Codebook generation is deterministic given seed + bins. Verify that encode/decode roundtrips and codebook shape are preserved.

**Export**:
```python
{
    "qm9": {
        "bins": [4, 5, 3, 5],
        "hv_dim": 256,
        "seed": 42,
        "codebook_shape": [300, 256],         # math.prod(bins) x hv_dim
        "codebook_hash": str,                  # SHA256 of codebook bytes
        "test_tuples": [[0,0,0,0], [2,1,0,3], [3,4,2,4]],
        "encoded_indices": [int, int, int],    # TupleIndexer results
        "encoded_hvs": Tensor,                 # [3, 256] encoded hypervectors
        "decode_of_encoded": [[int,...], ...],  # roundtrip: encode→decode
    },
    "zinc": { ... same structure with bins=[9,6,3,4,2] ... }
}
```

**Fixture format**: `.pt` (tensors + metadata in one dict)

**Tolerance**: Exact match for indices and codebook hash. `atol=1e-12` for hypervectors.

### 2. HyperNet Encoding

**What**: Given a PyG Data object, `forward()` must produce identical `graph_embedding`, `node_terms`, `edge_terms`.

**Export** (per molecule, per dataset):
```python
{
    "smiles": str,
    "input_x": Tensor,              # [num_nodes, num_features]
    "input_edge_index": Tensor,     # [2, num_edges]
    "config": {                      # Minimal config to reconstruct
        "bins": list[int],
        "hv_dim": 256,
        "depth": int,
        "seed": 42,
        "normalize": True,
    },
    "output_graph_embedding": Tensor,  # [1, hv_dim]
    "output_node_terms": Tensor,       # [1, hv_dim]
    "output_edge_terms": Tensor,       # [1, hv_dim]
}
```

**Fixture format**: `.pt` per dataset (dict with list of molecule entries)

**Tolerance**: `atol=1e-10` for float64 outputs.

### 3. HyperNet Node Decoding

**What**: Given an HDC embedding (from step 2), `decode_order_zero_iterative()` must produce the same node tuples.

**Export** (per molecule, per dataset):
```python
{
    "smiles": str,
    "input_node_terms": Tensor,         # [hv_dim] — the order-0 embedding
    "decoded_node_tuples": list[list[int]],  # e.g. [[0,0,0,3], [0,0,0,3]]
    "decoded_num_nodes": int,
}
```

**Fixture format**: `.json` (node tuples are small integer lists)

**Tolerance**: Exact match for node tuples and count.

### 4. RRWP Computation

**What**: `compute_rw_return_probabilities()` and `bin_rw_probabilities()` must produce identical results for given graphs.

**Export** (per molecule, using the QM9/ZINC edge_index from step 2):
```python
{
    "smiles": str,
    "edge_index": list[list[int]],     # JSON-friendly
    "num_nodes": int,
    "k_values": [3, 6],
    "rw_probs": list[list[float]],     # [num_nodes, len(k_values)]
    "binned_uniform_10": list[list[int]],   # uniform binning, 10 bins
    "binned_zinc_quantile_4": list[list[int]],  # ZINC quantile, 4 bins
}
```

**Fixture format**: `.json`

**Tolerance**: `atol=1e-12` for rw_probs. Exact match for binned values.

### 5. FlowEdgeDecoder Forward Pass

**What**: Given fixed random weights and a specific input, the model's `forward()` must produce the same loss/output. This verifies that parameterization changes (removing hardcoded constants) don't alter the computation.

**Export**:
```python
{
    "config": {
        "num_node_classes": 24,        # sum([9,6,3,4,2]) for ZINC
        "num_edge_classes": 5,
        "hdc_dim": 512,
        "n_layers": 2,                 # Small for test speed
        "hidden_dim": 64,
        "n_heads": 4,
        "max_nodes": 10,
    },
    "model_state_dict": OrderedDict,   # Saved weights
    "input_X": Tensor,                 # [1, max_nodes, num_node_classes]
    "input_E": Tensor,                 # [1, max_nodes, max_nodes, num_edge_classes]
    "input_node_mask": Tensor,         # [1, max_nodes]
    "input_hdc": Tensor,               # [1, hdc_dim]
    "input_time": Tensor,              # [1]
    "output": Tensor,                  # Model output
}
```

**Fixture format**: `.pt` (large tensors)

**Tolerance**: `atol=1e-6` (float32 model weights).

### 6. Evaluator Metrics

**What**: Given a small set of known molecules (as SMILES), the evaluator must compute the same validity, uniqueness, novelty, and property values.

**Export**:
```python
{
    "base_dataset": "qm9",
    "test_smiles": ["CCO", "c1ccccc1", "INVALID", "CCO", "CC(=O)O"],
    "expected_validity": float,
    "expected_uniqueness": float,     # among valid
    "expected_novelty": float,        # vs training set
    "per_molecule_properties": [
        {"smiles": "CCO", "logp": float, "qed": float, "sa": float},
        ...
    ],
}
```

**Fixture format**: `.json`

**Tolerance**: `atol=1e-6` for float properties. Exact for validity/uniqueness/novelty ratios.

## Export Script Structure

```
tests/migration/
├── export_golden.py           # Run once against current codebase
├── fixtures/
│   ├── encoder_qm9.pt         # HyperNet encoding (QM9 molecules)
│   ├── encoder_zinc.pt        # HyperNet encoding (ZINC molecules)
│   ├── codebook_qm9.pt        # CombinatoricIntegerEncoder (QM9 bins)
│   ├── codebook_zinc.pt       # CombinatoricIntegerEncoder (ZINC bins)
│   ├── node_decode_qm9.json   # Node decoding results
│   ├── node_decode_zinc.json  # Node decoding results
│   ├── rrwp.json              # RRWP computation results
│   ├── flow_decoder.pt        # FlowEdgeDecoder forward pass
│   └── evaluator.json         # Evaluator metrics
├── test_codebook.py           # Tests for CombinatoricIntegerEncoder
├── test_encoding.py           # Tests for HyperNet.forward()
├── test_node_decoding.py      # Tests for decode_order_zero_iterative
├── test_rrwp.py               # Tests for RRWP computation
├── test_flow_decoder.py       # Tests for FlowEdgeDecoder forward
└── test_evaluator.py          # Tests for evaluator metrics
```

## Export Script Outline

```python
"""
Export golden reference data from the current codebase.

Run ONCE before starting the refactoring:
    python tests/migration/export_golden.py

Requires: RDKit, current graph_hdc package, QM9/ZINC data files.
"""

import json
import hashlib
import torch
from pathlib import Path
from rdkit import Chem

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)

QM9_SMILES = [
    "C", "CC", "CCO", "C=O", "C#N",
    "c1ccccc1", "CC(=O)O", "c1ccncc1", "C1CC1", "OC(=O)c1ccccc1",
]
ZINC_SMILES = [
    "CC", "c1ccccc1", "c1ccc(cc1)N", "O=C(O)c1ccccc1", "c1ccc(cc1)Cl",
    "c1ccc2c(c1)cccc2", "CC(=O)Nc1ccccc1", "O=S(=O)(O)c1ccccc1",
    "c1cc(ccc1F)Br", "c1ccc(cc1)c2ccccc2",
]

def export_codebook(dataset: str, bins: list[int]):
    """Export CombinatoricIntegerEncoder golden data."""
    ...

def export_encoding(dataset: str, smiles_list: list[str]):
    """Export HyperNet.forward() golden data."""
    ...

def export_node_decoding(dataset: str, smiles_list: list[str]):
    """Export decode_order_zero_iterative golden data."""
    ...

def export_rrwp(smiles_list: list[str]):
    """Export RRWP computation golden data."""
    ...

def export_flow_decoder():
    """Export FlowEdgeDecoder forward pass golden data."""
    ...

def export_evaluator():
    """Export evaluator metrics golden data."""
    ...

if __name__ == "__main__":
    export_codebook("qm9", [4, 5, 3, 5])
    export_codebook("zinc", [9, 6, 3, 4, 2])
    export_encoding("qm9", QM9_SMILES)
    export_encoding("zinc", ZINC_SMILES)
    export_node_decoding("qm9", QM9_SMILES)
    export_node_decoding("zinc", ZINC_SMILES)
    export_rrwp(QM9_SMILES + ZINC_SMILES)
    export_flow_decoder()
    export_evaluator()
    print("All golden fixtures exported.")
```

## Test Structure (Example)

```python
# tests/migration/test_encoding.py

import torch
import pytest
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"

@pytest.fixture(params=["qm9", "zinc"])
def golden(request):
    return torch.load(FIXTURES / f"encoder_{request.param}.pt", weights_only=False)

def test_hypernet_forward_matches_golden(golden):
    """HyperNet.forward() must reproduce golden outputs."""
    from graph_hdc.core.hypernet import HyperNet  # refactored import

    cfg = golden["config"]
    hypernet = HyperNet(
        feature_bins=cfg["bins"],
        hv_dim=cfg["hv_dim"],
        hypernet_depth=cfg["depth"],
        seed=cfg["seed"],
        normalize=True,
        dtype="float64",
        device="cpu",
    )

    for entry in golden["molecules"]:
        data = Data(x=entry["input_x"], edge_index=entry["input_edge_index"])
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)

        with torch.no_grad():
            output = hypernet.forward(data, normalize=True)

        torch.testing.assert_close(
            output["graph_embedding"],
            entry["output_graph_embedding"],
            atol=1e-10, rtol=0,
        )
        torch.testing.assert_close(
            output["node_terms"],
            entry["output_node_terms"],
            atol=1e-10, rtol=0,
        )
        torch.testing.assert_close(
            output["edge_terms"],
            entry["output_edge_terms"],
            atol=1e-10, rtol=0,
        )
```

## Execution Order

1. **Before refactoring**: Run `python tests/migration/export_golden.py` → generates fixtures
2. **Commit fixtures** to the repo (they are the ground truth)
3. **During each refactoring phase**: Run `pytest tests/migration/` → must pass
4. **After refactoring complete**: Migration tests become permanent regression tests

## Notes

- The export script depends on the **current** codebase (old imports, DSHDCConfig, etc.). It will stop working after Phase 4 (legacy removal). That's fine — the fixtures are already committed.
- FlowEdgeDecoder test uses a **small model** (2 layers, 64 hidden dim) for speed. The test verifies the computation graph, not training quality.
- Evaluator test includes an intentionally invalid SMILES (`"INVALID"`) and a duplicate (`"CCO"` twice) to test validity and uniqueness counting.
- All tests run on CPU with deterministic settings. No GPU required.
