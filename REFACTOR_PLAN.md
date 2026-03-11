# Refactoring Plan: Domain-Agnostic GraphHDC

This document describes the target architecture after refactoring — the classes, interfaces, exchange data structures, and how the system fits together.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  DOMAIN LAYER (interchangeable per graph domain)                    │
│                                                                     │
│  MolecularDomain / ColoredGraphDomain / SceneGraphDomain / ...      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ process() │  │unprocess()│ │visualize()│  │ metrics  │           │
│  └─────┬─────┘  └─────▲─────┘ └──────────┘  └──────────┘           │
│        │               │                                            │
│   domain repr     domain repr                                       │
│   (SMILES,etc)    + validity                                        │
└────────┼───────────────┼────────────────────────────────────────────┘
         │               │
    nx.Graph          nx.Graph
    (features)        (features)
         │               │
┌────────┼───────────────┼────────────────────────────────────────────┐
│  CORE LAYER (domain-agnostic)                                       │
│        │               │                                            │
│        ▼               │                                            │
│  ┌──────────┐    ┌──────────┐                                       │
│  │ nx→PyG   │    │ dense→nx │                                       │
│  │ adapter  │    │ adapter  │                                       │
│  └─────┬────┘    └─────▲────┘                                       │
│        │               │                                            │
│   PyG Data        dense (X, E)                                      │
│        │               │                                            │
│        ▼               │                                            │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────┐              │
│  │ HyperNet │    │FlowEdgeDecoder│◄───│  Flow Model  │              │
│  │ (encoder)│    │   (decoder)   │    │ (Real NVP /  │              │
│  └─────┬────┘    └───────────────┘    │  FlowMatch)  │              │
│        │                              └──────▲───────┘              │
│   HDC embeddings                             │                      │
│   (edge_terms,                          HDC embeddings              │
│    graph_terms)──────────────────────────────┘                      │
│                                                                     │
│  ┌──────────────┐                                                   │
│  │  Evaluator   │  (calls domain.metrics + generic graph metrics)   │
│  └──────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Exchange Data Structures

### 1. NetworkX Graph (domain ↔ core boundary)

The universal intermediate representation. Each node carries:
- `"features"`: a `list[int]` — the **source of truth** for the core pipeline (codebook, PyG conversion, etc.)
- Named attributes: redundant copies of each feature under its schema name, for human-readable debugging.

The named attributes are derived from `features` using the domain's `feature_schema` key order. They are convenience copies — the core only ever reads `"features"`.

```python
# Molecular example (feature_schema keys: "atom_type", "formal_charge", "num_hydrogens")
G = nx.Graph()
G.add_node(0, features=[2, 0, 1], atom_type=2, formal_charge=0, num_hydrogens=1)
G.add_node(1, features=[0, 1, 0], atom_type=0, formal_charge=1, num_hydrogens=0)
G.add_edge(0, 1)

# Colored graph example (feature_schema keys: "color")
G = nx.Graph()
G.add_node(0, features=[2], color=2)
G.add_node(1, features=[0], color=0)
G.add_edge(0, 1)
```

Convention:
- Undirected, simple graphs only (no self-loops, no parallel edges)
- Node attribute key: `"features"` (always a `list[int]`) — **source of truth**
- Named attributes: one per feature_schema entry, integer values matching `features` list
- Feature values are 0-indexed integers within the declared bin range
- Edge attributes: optional, domain-provided if needed
- `process()` is responsible for populating both `features` and named attributes
- `validate()` checks consistency between `features` and named attributes

### 2. PyG Data (internal to core)

Standard PyTorch Geometric format. Produced by the nx→PyG adapter, consumed by HyperNet.

```python
Data(
    x=Tensor([[2, 0, 1], [0, 1, 0]]),  # [num_nodes, num_features] int
    edge_index=Tensor([[0, 1], [1, 0]]),  # [2, num_edges] undirected
    # After post_compute_encodings:
    edge_terms=Tensor([...]),    # [hv_dim]
    graph_terms=Tensor([...]),   # [hv_dim]
    node_terms=Tensor([...]),    # [hv_dim]
)
```

### 3. Dense Graph Representation (FlowEdgeDecoder I/O)

The FlowEdgeDecoder operates on dense tensors:

```python
X: Tensor  # [batch, max_nodes, sum(feature_bins)]  one-hot node features
E: Tensor  # [batch, max_nodes, max_nodes, num_edge_classes]  edge types
node_mask: Tensor  # [batch, max_nodes]  which nodes are real
```

### 4. HDC Embeddings (HyperNet ↔ Flow Model)

```python
{
    "edge_terms": Tensor,   # [batch, hv_dim] — encodes structure + node features
    "graph_terms": Tensor,  # [batch, hv_dim] — full graph embedding
    "node_terms": Tensor,   # [batch, hv_dim] — node features only
}
```

### 5. DomainResult (domain.unprocess() output)

```python
@dataclass
class DomainResult:
    domain_object: Any          # The domain-specific object (RDKit Mol, colored graph, etc.)
    is_valid: bool              # Whether the reconstruction succeeded
    canonical_key: str | None   # For uniqueness/novelty (canonical SMILES, WL hash, etc.)
    properties: dict[str, float]  # Optional domain properties (LogP, QED, etc.)
```

## Interfaces

### GraphDomain ABC

```python
from abc import ABC, abstractmethod

class GraphDomain(ABC):
    """Abstract interface for a graph domain."""

    # Subclasses define this as a class variable (plain dict in the class body).
    # Can be overridden with @property if dynamic behavior is needed.
    #
    # Example (molecular):
    #     feature_schema = {
    #         "atom_type": OneHotEncoder(["C", "N", "O", "F"]),
    #         "formal_charge": OneHotEncoder([0, 1, -1]),
    #         "num_hydrogens": IntegerEncoder(max_val=4),
    #     }
    #
    # Example (colored graph):
    #     feature_schema = {
    #         "color": OneHotEncoder(["red", "green", "blue"]),
    #     }
    feature_schema: ClassVar[dict[str, FeatureEncoder]]

    @property
    def feature_bins(self) -> list[int]:
        """Derived: list of cardinalities, one per feature dimension."""
        return [enc.num_bins for enc in self.feature_schema.values()]

    @abstractmethod
    def process(self, domain_repr) -> nx.Graph:
        """
        Convert a domain-specific representation to a NetworkX graph
        with integer features under the "features" node attribute.
        """
        ...

    @abstractmethod
    def unprocess(self, graph: nx.Graph) -> DomainResult:
        """
        Convert a NetworkX graph (with integer features) back to a domain object.
        Returns a DomainResult with the domain object, validity flag,
        canonical key, and optional properties.
        """
        ...

    @abstractmethod
    def visualize(self, ax: plt.Axes, domain_repr_or_graph, **kwargs) -> None:
        """Draw a domain object or graph onto the given Axes."""
        ...

    # Note: GraphDomain does NOT handle data loading. Datasets are a separate
    # abstraction (see GraphDataset below). A domain can have many dataset types.

    @property
    def metrics(self) -> dict[str, Callable]:
        """
        Registry of domain-specific metric functions.
        Each metric: (list[DomainResult], **kwargs) -> dict[str, float]
        """
        return {}

    def validate(self, graph: nx.Graph) -> None:
        """
        Check that a graph conforms to the feature schema.
        Raises ValueError with a clear message on mismatch.
        Called automatically during process() in debug mode.

        Checks:
        1. "features" list exists and has correct length
        2. Each feature value is within [0, num_bins)
        3. Named attributes exist and match the corresponding features entry
        """
        schema_names = list(self.feature_schema.keys())
        for node, attrs in graph.nodes(data=True):
            feats = attrs.get("features")
            if feats is None:
                raise ValueError(f"Node {node} missing 'features' attribute")
            if len(feats) != len(self.feature_bins):
                raise ValueError(
                    f"Node {node}: expected {len(self.feature_bins)} features, got {len(feats)}"
                )
            for i, (val, num_bins) in enumerate(zip(feats, self.feature_bins)):
                if not (0 <= val < num_bins):
                    raise ValueError(
                        f"Node {node}, feature {i}: value {val} out of range [0, {num_bins})"
                    )
                name = schema_names[i]
                named_val = attrs.get(name)
                if named_val is None:
                    raise ValueError(f"Node {node} missing named attribute '{name}'")
                if named_val != val:
                    raise ValueError(
                        f"Node {node}, '{name}': named attr {named_val} != features[{i}] {val}"
                    )
```

### FeatureEncoder Protocol

```python
class FeatureEncoder(Protocol):
    """Protocol for domain feature encoders in the attribute map."""

    @property
    def num_bins(self) -> int:
        """Number of discrete values this feature can take."""
        ...

    def encode(self, value: Any) -> int:
        """Encode a domain value to an integer index."""
        ...

    def decode(self, index: int) -> Any:
        """Decode an integer index back to a domain value."""
        ...
```

### GraphDataset ABC

Datasets are a separate abstraction from domains. A domain defines *what* a graph type is; a dataset defines *where* the data comes from. A single domain can have many dataset types (e.g., molecular → `SmilesDataset`, `XyzDataset`, `FragmentStreamDataset`).

Dataset types, not concrete datasets: `SmilesDataset("data/qm9.txt", domain)` and `SmilesDataset("data/zinc.txt", domain)` are the same class, different files. No need for `QM9Dataset` vs `ZINCDataset`.

```python
from abc import ABC, abstractmethod

class GraphDataset(ABC):
    """Abstract base for graph datasets. Knows its domain."""

    def __init__(self, domain: GraphDomain):
        self.domain = domain

    @abstractmethod
    def __iter__(self) -> Iterator[nx.Graph]:
        """Yield nx.Graphs conforming to self.domain.feature_schema."""
        ...

    def __len__(self) -> int:
        """Optional. Streaming datasets raise TypeError."""
        raise TypeError(f"{type(self).__name__} does not have a fixed length (streaming)")

    @property
    def is_finite(self) -> bool:
        """Whether this dataset has a fixed size."""
        return True
```

**Finite dataset type example** (molecular):

```python
class SmilesDataset(GraphDataset):
    """Loads SMILES strings from a text file. Works for QM9, ZINC, MOSES, etc."""

    def __init__(self, path: str | Path, domain: MolecularDomain, split: str | None = None):
        super().__init__(domain)
        self.path = Path(path)
        self.split = split
        self._smiles = self._load()

    def _load(self) -> list[str]:
        # Read SMILES from file, filter by split if needed
        ...

    def __iter__(self) -> Iterator[nx.Graph]:
        for smi in self._smiles:
            yield self.domain.process(smi)

    def __len__(self) -> int:
        return len(self._smiles)
```

**Streaming dataset type example** (molecular, infinite BRICS fragment recombination):

```python
class FragmentStreamDataset(GraphDataset):
    """Infinite stream of molecules from BRICS fragment recombination."""

    def __init__(self, fragment_library: Path, domain: MolecularDomain):
        super().__init__(domain)
        self.fragment_library = fragment_library

    def __iter__(self) -> Iterator[nx.Graph]:
        while True:
            mol = self._recombine_fragments()
            yield self.domain.process(mol)

    @property
    def is_finite(self) -> bool:
        return False
```

### HyperNet (refactored — encoder + node decoding)

The legacy *graph* decoders (decode_order_one, decode_graph, greedy decoder) are removed. The **node decoding** methods stay — they are used by the FlowEdgeDecoder pipeline to recover node tuples from HDC embeddings before edge prediction.

```python
class HyperNet:
    def __init__(
        self,
        feature_bins: list[int],        # REQUIRED: [4, 3, 5] etc.
        hv_dim: int = 256,
        vsa: str | VSAModel = "HRR",
        hypernet_depth: int = 3,
        seed: int | None = None,
        normalize: bool = False,
        normalize_graph_embedding: bool = False,
        prune_codebook: bool = False,
        observed_node_features: set[tuple] | None = None,
        dtype: str = "float64",
        device: str = "cpu",
    ):
        ...

    # ── Encoding ──

    def forward(
        self,
        data: Data | Batch,
        *,
        normalize: bool | None = None,
    ) -> dict[str, Tensor]:
        """Returns {"graph_embedding", "node_terms", "edge_terms"}"""
        ...

    def encode_properties(self, data: Data) -> Data:
        """Adds node_hv (and optionally edge_hv, graph_hv) to data."""
        ...

    # ── Node decoding (used by FlowEdgeDecoder pipeline) ──

    def decode_order_zero_iterative(
        self, embedding: Tensor, debug: bool = False,
    ) -> list[tuple[int, ...]] | tuple[list[tuple[int, ...]], list[float], list[float]]:
        """Iteratively unbind node features from order-0 HDC embedding."""
        ...

    def decode_order_zero_counter_iterative(
        self, embedding: Tensor,
    ) -> dict[int, Counter]:
        """Decode node counts as Counter dict using iterative unbinding."""
        ...

    # ── Persistence ──

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "HyperNet": ...
```

### FlowEdgeDecoder (refactored — no hardcoded constants)

```python
class FlowEdgeDecoder:
    def __init__(
        self,
        feature_bins: list[int],         # REQUIRED: replaces NODE_FEATURE_BINS
        num_edge_classes: int = 2,        # REQUIRED: replaces NUM_EDGE_CLASSES
        hdc_dim: int = 512,
        max_nodes: int = 50,
        # ... architecture params with defaults ...
        n_layers: int = 6,
        hidden_dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
        # ... training params ...
        lr: float = 1e-4,
        # ... sampling params ...
        sample_steps: int = 100,
        eta: float = 0.0,
        omega: float = 0.0,
    ):
        # num_node_classes = sum(feature_bins), computed internally
        ...
```

### GenerationEvaluator (refactored — domain-agnostic)

```python
class GenerationEvaluator:
    def __init__(
        self,
        domain: GraphDomain,
        training_keys: set[str] | None = None,  # canonical keys of training set
        validation_keys: set[str] | None = None,
    ):
        ...

    def evaluate(
        self,
        generated_graphs: list[nx.Graph],
    ) -> dict[str, float]:
        """
        1. Call domain.unprocess() on each graph → DomainResult
        2. Compute generic metrics: validity%, uniqueness%, novelty%
        3. Call each domain metric function
        4. Return combined metrics dict
        """
        ...
```

## Adapters

### nx_to_pyg (NetworkX → PyG)

```python
def nx_to_pyg(graph: nx.Graph) -> Data:
    """
    Convert a NetworkX graph with "features" node attributes to PyG Data.

    Returns:
        Data(x=Tensor[num_nodes, num_features], edge_index=Tensor[2, num_edges])
    """
```

### dense_to_nx (Dense Tensors → NetworkX)

```python
def dense_to_nx(
    X: Tensor,         # [max_nodes, sum(feature_bins)] one-hot
    E: Tensor,         # [max_nodes, max_nodes, num_edge_classes]
    node_mask: Tensor,  # [max_nodes] bool
    feature_bins: list[int],
    feature_names: list[str] | None = None,  # from domain.feature_schema.keys()
) -> nx.Graph:
    """
    Convert FlowEdgeDecoder dense output to NetworkX graph.
    Argmax on X to get integer features, argmax on E to get edges.
    If feature_names is provided, also sets named attributes on each node.
    """
```

## Data Pipeline

### Training

```python
# 1. Initialize domain and dataset separately
domain = MolecularDomain()
dataset = SmilesDataset("data/qm9_train.txt", domain)

# 2. Convert to PyG (cached)
train_data = load_and_cache_dataset(dataset)
# Internally: iter(dataset) → nx.Graph per molecule
#           → nx_to_pyg(graph) → PyG Data
#           → cache to disk as .pt

# 3. Create HyperNet from domain
hypernet = HyperNet(
    feature_bins=domain.feature_bins,
    hv_dim=256,
    hypernet_depth=3,
)

# 4. Compute HDC embeddings (unchanged)
train_encoded = post_compute_encodings(train_data, hypernet)

# 5. Train flow model (unchanged)
flow = RealNVPV3Lightning(input_dim=hypernet.hv_dim, ...)
trainer.fit(flow, DataLoader(train_encoded))

# 6. Train FlowEdgeDecoder
edge_decoder = FlowEdgeDecoder(
    feature_bins=domain.feature_bins,
    num_edge_classes=2,  # or domain-provided
    hdc_dim=hypernet.hv_dim,
)
```

### Generation & Evaluation

```python
# 1. Sample from flow
samples = flow.sample_split(n_samples=1000)

# 2. Decode with FlowEdgeDecoder → dense (X, E)
decoded = edge_decoder.sample(
    hdc_condition=samples["edge_terms"],
    ...
)

# 3. Convert dense → nx.Graph (with named attributes for debugging)
feature_names = list(domain.feature_schema.keys())
generated_graphs = [
    dense_to_nx(X[i], E[i], mask[i], domain.feature_bins, feature_names)
    for i in range(n_samples)
]

# 4. Evaluate
evaluator = GenerationEvaluator(
    domain=domain,
    training_keys=get_canonical_keys(train_data, domain),
)
metrics = evaluator.evaluate(generated_graphs)
# → {"validity": 0.85, "uniqueness": 0.92, "novelty": 0.78, ...}
```

## Package Structure (After Refactoring)

```
graph_hdc/
├── __init__.py                    # Core public API (no RDKit import)
├── core/
│   ├── hypernet.py                # HyperNet encoder (explicit args, no DSHDCConfig)
│   ├── feature_encoders.py        # CombinatoricIntegerEncoder, TupleIndexer
│   ├── types.py                   # VSAModel enum (Feat removed)
│   └── rw_features.py             # RRWP computation (domain-agnostic)
├── models/
│   ├── flow_edge_decoder.py       # FlowEdgeDecoder (parameterized, no ZINC constants)
│   ├── flows.py                   # RealNVP, FlowMatching
│   └── regressors.py              # PropertyRegressor (generic)
├── adapters/
│   ├── nx_pyg.py                  # nx_to_pyg, pyg_to_nx
│   ├── dense_nx.py                # dense_to_nx, nx_to_dense
│   └── dataset.py                 # load_and_cache_dataset(GraphDataset), post_compute_encodings
├── evaluation/
│   └── evaluator.py               # GenerationEvaluator (domain-agnostic)
├── domains/
│   ├── base.py                    # GraphDomain ABC, GraphDataset ABC, FeatureEncoder protocol, DomainResult
│   ├── molecular/
│   │   ├── __init__.py
│   │   ├── domain.py              # MolecularDomain(GraphDomain)
│   │   ├── chem.py                # RDKit utilities (nx_to_mol, reconstruct_for_eval, etc.)
│   │   ├── datasets.py            # SmilesDataset, XyzDataset (dataset types, not concrete datasets)
│   │   ├── encoders.py            # AtomTypeEncoder, ChargeEncoder, etc.
│   │   ├── metrics.py             # molecular metrics (LogP, QED, SA, FCD, diversity)
│   │   ├── visualization.py       # draw_mol
│   │   └── fragments/             # BRICS streaming, fragment library
│   │       ├── streaming.py       # FragmentStreamDataset(GraphDataset)
│   │       └── small_molecules.py
│   └── colored_graphs/            # Second domain (Phase 6)
│       ├── __init__.py
│       ├── domain.py              # ColoredGraphDomain(GraphDomain)
│       ├── datasets.py            # CogilesDataset, etc.
│       └── ...
├── utils/
│   ├── helpers.py                 # scatter_hd, generic utilities
│   └── nx_utils.py                # wl_hash, graph_hash, is_induced_subgraph (no Feat)
└── experiments/
    ├── train_flow.py              # Domain-parameterized training
    ├── train_edge_decoder.py      # Domain-parameterized FlowEdgeDecoder training
    ├── evaluate_generation.py     # Domain-parameterized evaluation
    └── run_optimization.py        # Moved to domains/molecular/ (molecular-specific)
```

## Migration Phases

### Phase 1: Define GraphDomain ABC, GraphDataset ABC + MolecularDomain wrapper — DONE

All items completed. Existing code untouched, all tests pass.

**What was implemented:**
- `graph_hdc/domains/base.py`: `GraphDomain` ABC, `GraphDataset` ABC, `FeatureEncoder` protocol (runtime-checkable), `DomainResult` dataclass, plus built-in generic encoders (`OneHotEncoder`, `IntegerEncoder`, `BoolEncoder`). `GraphDomain` includes `feature_schema`, `feature_bins`, `process()`, `unprocess()`, `visualize()`, `metrics`, and `validate()`.
- `graph_hdc/domains/molecular/domain.py`: `MolecularDomain` with default 5-feature schema (atom_type, degree, formal_charge, num_hydrogens, is_in_ring), `process()` (SMILES → nx.Graph with both `"features"` and named attributes), `unprocess()` (nx.Graph → DomainResult via bridge to legacy `reconstruct_for_eval()`), `visualize()`, and metrics (logp, qed, sa_score). Preset subclasses `QM9MolecularDomain` (bins [4, 5, 3, 5]) and `ZINCMolecularDomain` (bins [9, 6, 3, 4, 2]).
- `graph_hdc/domains/molecular/encoders.py`: `DegreeEncoder` (degree-1 convention) and `FormalChargeEncoder` ({0→0, +1→1, -1→2}).
- `graph_hdc/domains/molecular/datasets.py`: `SmilesDataset(GraphDataset)` — loads SMILES from text file, skips disconnected/invalid, works for QM9/ZINC/MOSES.
- `graph_hdc/domains/__init__.py`: Re-exports all base abstractions.
- `tests/test_domain_contract.py` (254 lines): Contract tests with `MinimalDomain` (color+shape), `MinimalDataset`, `StreamingDataset`. Covers `feature_bins`, `validate()` (6 error cases), dataset iteration/length/streaming, encoder roundtrips.
- `tests/test_molecular_domain.py` (355 lines): MolecularDomain tests (process, unprocess, roundtrip, properties, metrics), QM9/ZINC backward compatibility (feature matching against legacy `mol_to_data()`), encoder tests, SmilesDataset tests.
- `tests/migration/`: Golden fixture export script (`export_golden.py`, 501 lines) + 8 migration test suites (codebook, encoding, node decoding, edge decoding, mol_to_data, RRWP, FlowEdgeDecoder, evaluator) with `conftest.py`. ~4.9 MB of golden fixtures.
- `REFACTOR_TESTS.md`: Migration test strategy documentation.

### Phase 2: Make HyperNet accept explicit args — DONE

All items completed. Old tests still pass via DSHDCConfig path.

**What was implemented:**
- `graph_hdc/hypernet/encoder.py`: Added keyword-only `feature_bins`, `hv_dim`, `vsa`, `hypernet_depth`, `seed`, `normalize`, etc. as alternative to `config`. Mutual exclusivity check (config XOR feature_bins). New `_config_from_bins()` static method builds internal `DSHDCConfig` from explicit bins. New `feature_bins` property. New `from_domain(GraphDomain)` classmethod. Codebook pruning raises `ValueError` if `prune_codebook=True` without `observed_node_features` and `base_dataset=None`. Save/load preserves `feature_bins` in state dict.
- `graph_hdc/hypernet/configs.py`: `BaseDataset` type now allows `None` for domain-agnostic configs.
- `tests/test_hypernet_explicit.py` (319 lines): Creation tests (8), encoding equivalence tests against config-based HyperNet for QM9/ZINC (exact match at atol=1e-10), synthetic graph tests (color+shape domains), domain integration tests (`from_domain()`), save/load roundtrip tests, pruning error tests.

### Phase 3: Decouple FlowEdgeDecoder — ~95% DONE

The primary objective — decoupling FlowEdgeDecoder from hardcoded molecular constants — is achieved. The model accepts `feature_bins` and `num_edge_classes` as required arguments and works with any graph domain.

**What was implemented:**
- ✅ `flow_edge_decoder.py`: `feature_bins` and `num_edge_classes` are mandatory positional args. Zero hardcoded constants, zero RDKit imports, zero preprocessing imports. All node/edge dimensions computed from constructor arguments.
- ✅ `BOND_TYPE_TO_IDX` and all ZINC/molecular constants moved to `domains/molecular/preprocessing.py` with clear docstring. RDKit imports localized there.
- ✅ Migration tests (`tests/migration/test_flow_decoder.py`) derive all dimensions from fixture config dicts, not hardcoded values.
- ✅ Main test files (`tests/test_flow_edge_decoder.py`, `tests/test_transformer_edge_decoder.py`) import constants from preprocessing (appropriate since they are molecular-domain tests).
- ✅ Core library (`graph_hdc/__init__.py`, `graph_hdc/models/__init__.py`) does not export molecular-specific constants.

**Remaining (minor, non-blocking):**
- ⚠️ `transformer_edge_decoder.py` still imports `NODE_FEATURE_DIM` and `NUM_EDGE_CLASSES` from preprocessing as **default argument values** (lines 150-151). The model itself accepts explicit overrides. This is intentional backward compatibility for a secondary model — not a blocker.
- ⚠️ `preprocessing.py` keeps legacy aliases (`NODE_FEATURE_DIM`, `NODE_FEATURE_BINS`, `NUM_EDGE_CLASSES`) for backward compatibility. Could be deprecated in Phase 4 cleanup.

### Phase 4: Remove legacy code — NOT STARTED

None of the Phase 4 deletion tasks have been performed. All legacy code is still fully present and actively used throughout the codebase.

**Detailed status of each item:**

- ❌ **Legacy graph decoders in `encoder.py`** — ALL STILL PRESENT:
  - `decode_order_one()` (line ~803)
  - `decode_order_one_no_node_terms()` (line ~875)
  - `decode_graph()` (line ~1120)
  - `decode_graph_greedy()` (line ~1286)
  - Supporting helpers: `_is_feasible_set()`, `_apply_edge_corrections()`, `_find_top_k_isomorphic_graphs()`
  - These methods actively import from `correction_utils` and `decoder` modules.
  - **Keep as required**: `decode_order_zero_iterative` (line ~633) and `decode_order_zero_counter_iterative` (line ~715) — correctly present and used by FlowEdgeDecoder pipeline.

- ❌ **`decoder.py`** — STILL EXISTS at `graph_hdc/hypernet/decoder.py`. Contains `compute_sampling_structure()`, `try_find_isomorphic_graph()`, `has_valid_ring_structure()`, etc. Actively imported by legacy decoder methods in `encoder.py`.

- ❌ **`correction_utils.py`** — STILL EXISTS at `graph_hdc/hypernet/correction_utils.py`. Contains `CorrectionResult`, `get_corrected_sets()`, `get_node_counter()`, `target_reached()`. Actively imported by legacy decoder methods in `encoder.py`.

- ❌ **`Feat` dataclass in `types.py`** — STILL EXISTS at `graph_hdc/hypernet/types.py` (lines ~39-90). Actively used by `decode_graph()` and `decode_graph_greedy()` via `Feat.from_tuple()`.

- ❌ **Degree-dependent functions in `nx_utils.py`** — ALL STILL PRESENT at `graph_hdc/utils/nx_utils.py`:
  - `current_degree()`, `residual_degree()`, `residuals()`, `anchors()`
  - `add_node_with_feat()`, `add_node_and_connect()`, `connect_all_if_possible()`
  - `order_leftovers_by_degree_distinct()`
  - All depend on the `Feat` dataclass and `target_degree` convention. Actively used by legacy decoders.

- ❌ **`DecoderSettings` / `FallbackDecoderSettings`** — STILL EXIST in `graph_hdc/hypernet/configs.py` (lines ~87-177). Exported from `__init__.py`. Used in experiment scripts (`evaluate_generation.py`, `run_optimization.py`, `run_retrieval_experiment.py`).

- ❌ **`DSHDCConfig` / `get_config()` / `SupportedDataset`** — ALL STILL EXIST in `graph_hdc/hypernet/configs.py`:
  - `DSHDCConfig` (lines ~64-84)
  - `_get_qm9_config()`, `_get_zinc_config()` helper functions
  - `SupportedDataset` enum (lines ~234-249)
  - `get_config()` function (lines ~252-257)
  - Widely used: `encoder.py`, `rrwp_hypernet.py`, `experiment_helpers.py`, multiple experiment scripts.

**Note:** All legacy code forms a tightly coupled cluster. The deletion order matters: legacy decoder methods → `decoder.py`/`correction_utils.py` → `Feat` → degree functions → `DecoderSettings`. `DSHDCConfig`/`get_config()` can be removed independently once all experiment scripts are updated to use explicit args or `from_domain()`.

### Phase 5: Relocate molecular code — ~85% DONE

The core goal — relocating molecular code to `domains/molecular/` and establishing backward-compatible shims — is largely achieved.

**What was implemented:**

- ✅ **`utils/chem.py` → `domains/molecular/chem.py`**: FULL MOVE (391 lines in new location). All functions moved: `draw_mol()`, `is_valid_molecule()`, `canonical_key()`, `compute_qed()`, `nx_to_mol()`, `_infer_bond_orders()`, `reconstruct_for_eval()`, `mol_to_data()`, `ReconstructionResult`, atom symbol mappings, charge mappings. Old `utils/chem.py` is now an 18-line backward-compatible re-export wrapper.

- ✅ **Evaluator → `domains/molecular/metrics.py`**: FULL MOVE (389 lines in new location). Contains `rdkit_logp()`, `rdkit_qed()`, `rdkit_sa_score()`, `rdkit_max_ring_size()`, `calculate_internal_diversity()`, and the full `GenerationEvaluator` class (210 lines). Old `utils/evaluator.py` is now a 10-line re-export wrapper.

- ✅ **Experiment helpers → `domains/molecular/helpers.py`**: EXTRACTED (408 lines in new location). Molecular-specific functions moved: `pyg_to_mol()`, `scrub_smiles()`, `is_valid_mol()`, `get_canonical_smiles()`, `draw_mol_or_error()`, `create_reconstruction_plot()`, `compute_tanimoto_similarity()`, `load_smiles_from_csv()`, `smiles_to_pyg_data()`. Old `utils/experiment_helpers.py` is now a hybrid: domain-agnostic core logic (HDC config, encoding, callbacks) plus re-exports from `domains.molecular.helpers`.

- ✅ **Datasets → `domains/molecular/datasets.py`**: NEW generic `SmilesDataset(GraphDataset)` class (68 lines). Loads SMILES from text files, converts via `domain.process()`. Legacy PyG dataset classes (`QM9Smiles`, `ZincSmiles`) remain in `graph_hdc/datasets/` for backward compatibility — appropriate since the refactored pipeline prefers `SmilesDataset`.

- ✅ **Import updates**: Backward-compatible re-export shims in all old locations. `domains/molecular/__init__.py` uses `__getattr__` lazy import pattern to avoid circular dependencies and defer RDKit loading.

- ✅ **New domain infrastructure**: `domains/molecular/preprocessing.py` centralizes all molecular constants (`NODE_FEATURE_BINS`, `NUM_EDGE_CLASSES`, `BOND_TYPE_TO_IDX`, `ZINC_ATOM_TYPES`, etc.). `domains/molecular/encoders.py` has `DegreeEncoder` and `FormalChargeEncoder`.

**Remaining:**

- ❌ **ZINC quantile bins not moved**: Still in `graph_hdc/utils/rw_features.py` (lines 26-115: `_ZINC_RW_QUANTILE_BOUNDARIES`, `ZINC_RW_QUANTILE_BOUNDARIES`, `get_zinc_rw_boundaries()`). Referenced by `streaming_fragments.py` and `experiment_helpers.py`. Should move to `domains/molecular/` — RRWP computation itself is domain-agnostic, but ZINC-specific quantile presets belong in the molecular domain.

- ⚠️ **Top-level RDKit dependency not fully lazy**: `graph_hdc/__init__.py` (lines 36-42) unconditionally imports from `utils.evaluator`, which triggers RDKit import. `from graph_hdc import HyperNet` works fine, but `from graph_hdc import GenerationEvaluator` requires RDKit. To make the top-level package fully RDKit-optional, evaluator imports in `__init__.py` should use the same `__getattr__` lazy pattern already used in `domains/molecular/__init__.py`.

- ⚠️ **Legacy dataset classes still present**: `graph_hdc/datasets/{qm9_smiles.py, zinc_smiles.py}` still exist alongside the new `SmilesDataset`. This is intentional — backward compatibility for existing training scripts.

### Phase 6: Implement colored graph domain
- Create `domains/colored_graphs/domain.py`
- Write contract tests
- Run end-to-end training + generation
- Validates the abstraction works for a non-molecular domain
