# Domain-Agnostic Refactoring

## Motivation

The GraphHDC method — HDC encoding via VSA message passing + normalizing flow for generation — is fundamentally **graph-domain agnostic**. The core pipeline (encode graph → learn distribution → sample → decode graph) operates on integer node features and adjacency structures. It does not inherently require molecular chemistry.

However, the current codebase is deeply coupled to molecular graphs: RDKit imports span ~34 files, atom types and valence rules are hardcoded throughout, and evaluation metrics assume SMILES-based validity/uniqueness. This makes it impossible to apply the same method to other graph domains (colored graphs, scene graphs, social networks, etc.) without invasive changes.

## Goal

Introduce a **GraphDomain** abstraction that cleanly separates domain-specific concerns from the core HDC+flow pipeline. The result should be a plug-and-play system where:

- The **core models** (HyperNet encoder, FlowEdgeDecoder, normalizing flow) only see domain-agnostic graph representations and arbitrary integer node features.
- Each **domain** (molecules, colored graphs, scene graphs, ...) is an interchangeable module that handles data loading, feature definition, graph-to-domain conversion, visualization, and evaluation metrics.

## Legacy Code Removal

This refactoring is also a cleanup opportunity. The following legacy components are **removed entirely**, not refactored:

- **Deterministic decoders**: The greedy beam search decoder, correction-based decoder, and all associated code in `encoder.py` (decode_order_zero, decode_order_one, decode_graph, greedy decoder methods), `decoder.py`, `correction_utils.py`. The `DecoderSettings` and `FallbackDecoderSettings` dataclasses are also removed.
- **`Feat` dataclass**: The molecule-specific `Feat` type in `types.py` with its `atom_type`, `degree_idx`, `formal_charge_idx`, `explicit_hs`, `is_in_ring` fields. Replaced by plain integer tuples.
- **nx_utils degree helpers**: `residual_degree()`, `add_node_with_feat()`, `total_edges_count()`, `anchors()`, and other functions that depend on the `Feat` dataclass or the `k[1] + 1` degree convention.

The **FlowEdgeDecoder** is the only decoder going forward. It predicts edges directly from learned node embeddings and does not suffer from the degree-feature coupling that plagued the deterministic decoders. This eliminates the single hardest coupling point in the codebase.

## Design Decisions

### Transfer Format: NetworkX Graphs

NetworkX graph objects serve as the **domain-agnostic intermediate representation** between domain-specific code and the core pipeline. Node features are stored as node attributes (plain integers/lists). The core converts to PyG `Data` internally for model operations.

**Rationale**: Serialization is straightforward via `json_graph` or pickle. NetworkX provides essential graph algorithms (cycle detection, connectivity, isomorphism, WL hashing) needed for evaluation. The memory/speed overhead is negligible at our scale.

### GraphDomain Interface (inspired by visual_graph_datasets)

Each domain provides a processing class with a consistent interface, inspired by the `ProcessingBase` pattern from [visual_graph_datasets](https://github.com/aimat-lab/visual_graph_datasets), but tailored to the HDC pipeline's needs. Key responsibilities:

- **`process()`**: Convert domain representation (e.g., SMILES string, COGILES string, image annotation) → NetworkX graph with integer node features.
- **`unprocess()`**: Convert NetworkX graph → domain representation. This is a full reconstruction pipeline, not just a simple fixup. For molecules, this is a multi-strategy cascade (kekulize, no-kekulize, single-bonds, partial sanitization). Should return a typed result: domain object + validity flag + canonical key for uniqueness/novelty.
- **`visualize()`**: Render a domain object or graph for inspection.
- **Feature schema**: An attribute map (VGD-style) mapping feature names to encoder objects with `encode()`/`decode()` methods. The core reads only the cardinalities (bin counts) for codebook construction; domains use the full schema for encoding/decoding semantics. All features are **intrinsic** node properties (atom type, color, label).
- **`canonical_key()`**: Domain-provided function to produce a canonical string representation for uniqueness/novelty computation (e.g., canonical SMILES for molecules, WL hash for generic graphs).
- **Metric registry**: Domains register named metric functions for evaluation. Generic graph metrics (structural uniqueness, connectivity) are provided by the core. Domains add their own (e.g., molecular validity via RDKit sanitization, SMILES-based novelty).
- **Validation**: Eager `validate()` method that checks `process()` output against the declared schema, catching feature cardinality mismatches and structural issues early with clear error messages.

### Degree Is Not a Node Feature

Node degree is a **structural property** of the graph adjacency, not an intrinsic node feature. With the removal of the deterministic decoders (which hardcoded `k[1] + 1` for degree), this distinction becomes clean. The FlowEdgeDecoder predicts edges directly and does not need degree encoded as a feature. Domains provide only intrinsic features (atom type, color, charge, etc.).

### Domain-Specific Extensions

All molecular-specific code (BRICS fragment recombination, property prediction regressors, LogP/QED computation) moves into the **molecular domain module**. From the core's perspective, these are just a data stream and optional domain properties — fragment streaming vs. dataset loading is a data source concern, not a model concern.

### Constructor Design: Explicit Arguments with Good Defaults

Core components (HyperNet, FlowEdgeDecoder, etc.) receive their parameters as explicit constructor arguments with sensible defaults — no config dataclasses. A few parameters are required (e.g., `feature_bins`), the rest have defaults. This trades the convenience of a config object for explicitness and eliminates the God-object config anti-pattern.

### Backwards Compatibility

None preserved. All existing experiment scripts and configs will be migrated to the new domain-based API. No legacy code paths or shim layers.

### Migration Strategy: Incremental Phases

The refactoring proceeds in incremental phases, each with passing tests:

1. **Define GraphDomain ABC + MolecularDomain wrapper** (non-breaking, existing code untouched)
2. **Make HyperNet accept domain-agnostic parameters** (alternative constructor alongside old one)
3. **Decouple FlowEdgeDecoder constants** (remove hardcoded ZINC dims, parameterize bins)
4. **Remove legacy code** (deterministic decoders, Feat, degree-dependent nx_utils, DSHDCConfig)
5. **Relocate molecular code** to `graph_hdc/domains/molecular/`
6. **Implement second domain** (colored graphs) to validate the abstraction

### Scope Constraints

The abstraction targets **undirected simple graphs** with:
- Fixed-arity integer node features
- Bounded edge types (or no edge types)
- No self-loops, no multigraph support, no directed edges

## What Stays (Core)

- **HyperNet encoder**: VSA message passing on integer features — already domain-agnostic in logic, just needs config decoupling. The decode methods are removed; only encoding (`forward`, `encode_properties`) remains.
- **FlowEdgeDecoder**: The sole decoder going forward. Predicts edges from node feature embeddings. Already accepts `feature_bins` as optional params; just needs hardcoded ZINC defaults removed.
- **CombinatoricIntegerEncoder**: Maps integer feature vectors to codebook indices — domain-agnostic, just needs bin counts.
- **RRWP computation**: Random walk return probabilities are already domain-agnostic (operate on edge_index). Only the precomputed ZINC quantile bins are molecular-specific and move to the molecular domain.

## What Gets Removed (Legacy)

- **Deterministic decoders**: `decoder.py`, `correction_utils.py`, all `decode_*` methods in `encoder.py`, `DecoderSettings`, `FallbackDecoderSettings`.
- **`Feat` dataclass**: `types.py` — replaced by plain integer tuples.
- **Degree-dependent nx_utils**: `residual_degree()`, `add_node_with_feat()`, `total_edges_count()`, `anchors()`, `add_node_and_connect()`, `connect_all_if_possible()`.
- **`DSHDCConfig`**: The config dataclass with hardcoded `base_dataset`, `nha_bins`, etc.
- **`get_config()` / `SupportedDataset`**: The named config registry.

## What Changes (Refactored)

- **Feature definitions**: Domain-provided attribute maps with `encode()`/`decode()` per feature.
- **Configs → explicit args**: Core components take explicit constructor arguments with good defaults.
- **Data loading**: Each domain provides its own dataset class producing NetworkX graphs; a shared adapter converts to PyG. NX→PyG conversion cached at dataset processing time (not per batch).
- **Evaluation**: Generic evaluator calls domain-registered metrics. Domain provides `canonical_key()` for uniqueness/novelty.
- **Visualization**: Delegated to the domain.
- **Experiment scripts**: Parameterized by domain name rather than dataset name.
- **Node attribute convention**: Standardized to a single key (e.g., `"features"`) instead of the current `"feat"` vs `"type"` dual convention.
- **Serialization**: `save()`/`load()` records feature schema (not `base_dataset`) so loaded models reconstruct codebooks without the domain class.
- **RDKit as soft dependency**: Core package importable without RDKit. Molecular domain is optional.

## Example Domains

1. **Molecular** (existing): QM9, ZINC — SMILES ↔ graph, RDKit validation, LogP/QED metrics, BRICS fragments.
2. **Colored Graphs**: COGILES ↔ graph, color consistency checks, structural metrics.
3. **Scene Graphs**: Image annotations ↔ graph, object/relationship features, semantic validity.

## Expert Review Findings

This design was reviewed by 4 expert agents (Architecture, ML Pipeline, Developer Experience, Testing/Reliability). Key findings incorporated:

- **Incremental phased migration** reduces regression risk
- **Eager validation** at domain boundary catches mismatches early
- **`unprocess()` as full reconstruction pipeline** (not a simple hook)
- **Standardize NX node attribute key** to eliminate `"feat"` vs `"type"` ambiguity
- **Cache NX→PyG conversion** during dataset processing, not per-batch
- **RDKit must be a soft dependency** — core imports without it
- **The degree-coupling problem** (identified as the hardest challenge by all 4 experts) is resolved by removing the deterministic decoders entirely rather than refactoring them
