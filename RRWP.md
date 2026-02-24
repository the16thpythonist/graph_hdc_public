# RRWP Features — Random Walk Return Probabilities

This document explains the RRWP (Random Walk Return Probability) feature system: what it captures, the underlying math, and how it is implemented across the codebase.

## Motivation

Local message passing in the HDC encoder aggregates information from a node's immediate neighbourhood. However, certain global structural properties — ring membership, node centrality, bridge positions — cannot be recovered from local neighbourhoods alone. RRWP features address this gap by annotating each node with a compact summary of its position within the overall graph topology.

The key idea: the probability that a random walk starting at node *i* returns to node *i* after exactly *k* steps reveals structural context that is invisible to bounded-depth message passing.

## Math (light version)

Given an undirected graph with adjacency matrix **A** and degree matrix **D** = diag(d₁, …, dₙ):

1. **Transition matrix**: T = D⁻¹A — the row-stochastic matrix where T_ij = 1/dᵢ if edge (i,j) exists, else 0. Isolated nodes get a self-loop (T_ii = 1).

2. **k-step return probability**: For each node *i*, compute [T^k]_ii — the (i,i) entry of the k-th matrix power of T. This gives the probability that a length-k random walk starting at *i* returns to *i*.

3. **Feature vector**: For a set of k-values (e.g. k ∈ {3, 6}), each node gets a vector of return probabilities in [0, 1]:

   ```
   rrwp(i) = ( [T³]_ii, [T⁶]_ii )
   ```

4. **Discretisation**: The continuous probabilities are binned into integer indices (0 … num_bins-1) and appended to the base node feature vector. Three binning modes are available:
   - **Quantile**: per-k boundary vectors from precomputed dataset statistics.
   - **Clipped uniform**: uniform bins over a specified [lo, hi] range.
   - **Uniform**: uniform bins over [0, 1].

Quantile turned out to work the best and is currently used throughout the implementation

### What the features capture

| Pattern | Effect on return probability |
|---|---|
| Node in a short ring (e.g. 3-cycle) | High return prob at k = ring length |
| Bridge node between clusters | Low return prob (walks disperse) |
| High-degree hub | Lower return prob (more exit paths) |
| Leaf / low-degree node | Higher return prob |
| Odd k on bipartite-like structures | Near-zero (can't return in odd steps) |

## Architecture: Split Codebook Design

The RRWP features are only used for the **order-0 (node-level) embedding**. They are deliberately excluded from message passing to prevent positional information from leaking into structural edge binding.

This is achieved through `RRWPHyperNet`, which maintains two separate codebooks:

```
                           data.x = [base_features | rrwp_bins]
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
              base features               full features
             (first F cols)            (first F+R cols)
                    │                           │
                    ▼                           ▼
           ┌───────────────┐          ┌─────────────────┐
           │nodes_codebook │          │nodes_codebook_   │
           │   (base)      │          │   full           │
           │               │          │  (base + RRWP)   │
           └───────┬───────┘          └────────┬────────┘
                   │                           │
                   ▼                           │
           data.node_hv                data.node_hv_full
                   │                           │
                   ▼                           │
        ┌──────────────────┐                   │
        │  Message Passing │                   │
        │  (VSA binding)   │                   │
        └──────┬───────────┘                   │
               │                               │
      ┌────────┼────────┐                      │
      │        │        │                      │
      ▼        ▼        ▼                      ▼
  edge_    graph_    node_terms ◄──── REPLACED by
  terms  embedding  (discarded)       bundle(node_hv_full)
```

**Why split?** Message passing uses binding and permutation operations to encode edge structure. If RRWP features were mixed in, the resulting edge-level hypervectors would conflate topology (who is connected to whom) with position (where a node sits globally). By restricting RRWP to order-0, the graph embedding stays purely structural while the node-level readout gains global context.

Additionally, it just turned out that using the RRWP features also for the convolution does not work for the reconstruction at all. The hypothesis is that when the nodes become super distinct through the RRWP features, the local convolutions of higher order are all completely distinct symbols which are likely never seen twice during training and therefore no network can learn a pattern from them either.

## Implementation Map

| File | Role |
|---|---|
| `graph_hdc/utils/rw_features.py` | Core computation: `compute_rw_return_probabilities`, `bin_rw_probabilities`, `augment_data_with_rw` |
| `graph_hdc/hypernet/rrwp_hypernet.py` | `RRWPHyperNet` — split-codebook encoder/decoder |
| `graph_hdc/hypernet/configs.py` | `RWConfig` dataclass, `create_config_with_rw` factory, `_BASE_BINS` |
| `graph_hdc/datasets/utils.py` | `encode_dataset` (augments batches before encoding), `scan_node_features_with_rw` (collects observed tuples for pruning) |
| `graph_hdc/utils/experiment_helpers.py` | `create_hypernet` — end-to-end helper that wires config + scanning + RRWPHyperNet |

## Usage

### 1. Create a config with RRWP enabled

```python
from graph_hdc.hypernet.configs import RWConfig, create_config_with_rw

rw_config = RWConfig(
    enabled=True,
    k_values=(3, 6),    # random walk step counts
    num_bins=4,          # discretisation bins per k
)

config = create_config_with_rw("qm9", hv_dim=256, rw_config=rw_config)
# Node feature bins: [4, 5, 3, 5] (base) + [4, 4] (RRWP) = [4, 5, 3, 5, 4, 4]
```

The base bins per dataset are:
- **QM9**: `[4, 5, 3, 5]` — atom type, degree, charge, num hydrogens
- **ZINC**: `[9, 6, 3, 4, 2]` — atom type, degree, charge, num hydrogens, is_in_ring

`create_config_with_rw` appends `[num_bins] * len(k_values)` to these.

### 2. Instantiate the encoder

```python
from graph_hdc.hypernet.rrwp_hypernet import RRWPHyperNet

# Optional: scan the dataset to prune the codebook to observed feature combos
from graph_hdc.datasets.utils import scan_node_features_with_rw
observed = scan_node_features_with_rw("qm9", rw_config)

net = RRWPHyperNet(config, observed_node_features=observed)
```

### 3. Augment data and encode

RRWP columns must be appended to `data.x` before calling `forward`:

```python
from graph_hdc.utils.rw_features import augment_data_with_rw

# data.x has shape [N, 4] (base QM9 features)
data = augment_data_with_rw(
    data,
    k_values=rw_config.k_values,
    num_bins=rw_config.num_bins,
)
# data.x now has shape [N, 6] (base + 2 RRWP bin columns)

result = net.forward(data, normalize=True)
# result["node_terms"]      — RRWP-enriched order-0 embedding [B, D]
# result["edge_terms"]      — structural (base-only) [B, D]
# result["graph_embedding"] — structural (base-only) [B, D]
```

### 4. Decode nodes from an embedding

```python
# Dot-product decoding (returns rounded counts per codebook entry)
counts = net.decode_order_zero(result["node_terms"])

# Iterative subtraction (more robust for noisy/generated embeddings)
node_tuples = net.decode_order_zero_iterative(result["node_terms"][0])
# → [(0, 2, 1, 0, 2, 1), (1, 3, 1, 1, 0, 3), ...]
#    ^^^^^^^^^^^^^^^^     base features + RRWP bins
```

## Quantile Binning for ZINC

Uniform binning works poorly for ZINC because return probability distributions are highly skewed (odd-k values are near-degenerate on molecular graphs). Precomputed quantile boundaries are available:

```python
from graph_hdc.utils.rw_features import get_zinc_rw_boundaries

boundaries = get_zinc_rw_boundaries(num_bins=4)  # available: 3, 4, 5, 6, 7

rw_config = RWConfig(
    enabled=True,
    k_values=(6,),
    num_bins=4,
    bin_boundaries=boundaries,
)
```

These boundaries were computed from 5,000 ZINC training molecules (~116k atoms) and provide equal-frequency bins per k-value.

## Limitations

- **Edge/graph decoding not supported** — `RRWPHyperNet` raises `NotImplementedError` for `decode_order_one`, `decode_graph`, etc. Use the base `HyperNet` for those.
- **Codebook size** — adding *R* RRWP dimensions with *B* bins each multiplies the codebook size by B^R (before pruning). Pruning to observed features is strongly recommended for large configs.
- **Computation cost** — `compute_rw_return_probabilities` builds a dense adjacency matrix and uses `torch.linalg.matrix_power`, which is O(n³) per k-value. This is fine for small molecules but expensive for large graphs.
