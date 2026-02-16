# IDEAS: HDC-Guided Sampling for FlowEdgeDecoder

This document describes four research ideas for leveraging the HyperNet (HDC) encoder
more effectively during the FlowEdgeDecoder's discrete flow matching sampling process.
Each section is self-contained with motivation, mechanism, key code references, and
implementation notes.

## Table of Contents

1. [Unbinding-Based Edge Probability Modulation](#1-unbinding-based-edge-probability-modulation)
2. [Gradient-Based Rate Matrix Correction (Soft HDC)](#2-gradient-based-rate-matrix-correction-soft-hdc)
3. [Delta Guidance from Incremental Encoding](#3-delta-guidance-from-incremental-encoding)
4. [Self-Play with Hindsight Relabeling](#4-self-play-with-hindsight-relabeling)

---

## Background

The FlowEdgeDecoder (`graph_hdc/models/flow_edge_decoder.py`) generates molecular edges
via discrete flow matching conditioned on HDC vectors `[order_0 | order_N]`. Currently
the HDC vector is used only as a **passive conditioning input**: it is projected through
an MLP into a global feature vector `y` (and optionally cross-attended). The HyperNet
encoder's rich algebraic structure (binding, unbinding, bundling, cosine probing) is
never exploited during sampling.

### Key Code Locations

| Component | File | Key Lines |
|-----------|------|-----------|
| FlowEdgeDecoder model | `graph_hdc/models/flow_edge_decoder.py` | Full file |
| Sampling loop | `flow_edge_decoder.py` | `sample()` L1091-1252 |
| Sampling step | `flow_edge_decoder.py` | `_sample_step()` L980-1089 |
| Existing HDC guidance | `flow_edge_decoder.py` | `sample_with_hdc_guidance()` L1528-1764 |
| HyperNet encoder | `graph_hdc/hypernet/encoder.py` | `forward()` L302-368 |
| Node codebooks | `encoder.py` | `_build_codebooks()` L208-241 |
| Edge codebook | `encoder.py` | `edges_codebook` L238-241 |
| Order-0 decode | `encoder.py` | `decode_order_zero_iterative()` L379-454 |
| Order-1 decode (edges) | `encoder.py` | `decode_order_one_no_node_terms()` L621-664 |
| Encode edge multiset | `encoder.py` | `encode_edge_multiset()` L505-547 |
| Bind / unbind | `graph_hdc/utils/helpers.py` | `unbind()` L188-195 |
| scatter_hd | `graph_hdc/utils/helpers.py` | `scatter_hd()` L67-149 |
| cartesian_bind_tensor | `graph_hdc/utils/helpers.py` | `cartesian_bind_tensor()` L152-185 |
| Streaming experiment | `experiments/decoding/train_flow_edge_decoder_streaming.py` | Full file |
| Mixed streaming loader | `graph_hdc/datasets/mixed_streaming.py` | Full file |

### HDC Encoding Structure

The HyperNet (`encoder.py:302-368`) uses HRR (Holographic Reduced Representations):

- **Bind** = circular convolution (differentiable via FFT). Used for composing features.
- **Bundle** = element-wise sum. Used for aggregating sets/multisets.
- **Unbind** = bind with inverse: `torchhd.bind(composite, factor.inverse())`.
- **Cosine similarity** = natural "contains?" query against codebooks.

The conditioning vector is `[order_0 (D dims) | order_N (D dims)]` where:

- **order_0**: `scatter_hd(node_hv, batch, op="bundle")` -- bag-of-atoms.
  Each node HV = `multibind(HV_atom, HV_degree, HV_charge, HV_Hs, HV_ring)`.
  Encodes **what** atoms exist and their counts. Does not encode connectivity.
- **edge_terms** (layer 0 of message passing): For each node,
  `bind(node_hv, bundle(neighbor_hvs))`. Encodes 1-hop structure.
- **order_N** (graph_embedding): Multi-layer message passing, then
  `multibundle(node_hv_stack)` and `scatter_hd(bundle)`. Encodes global topology
  up to depth N (typically 3-6).

### Codebook Structure

The HyperNet maintains pre-computed codebooks (`encoder.py:208-241`):

- `nodes_codebook`: Shape `[num_node_types, D]`. Cartesian bind of per-feature
  codebooks. Each row is the HV for a unique node type tuple
  `(atom_type, degree, charge, Hs, ring)`.
- `edges_codebook`: Shape `[num_node_types^2, D]`. Cartesian bind of
  `(nodes_codebook, nodes_codebook)`. Each row = `bind(hv_src, hv_dst)`.
- `nodes_indexer` / `edges_indexer`: `TupleIndexer` mapping between integer
  indices and feature tuples.

These codebooks enable O(1) lookup of any node or edge HV by feature tuple.

---

## 1. Unbinding-Based Edge Probability Modulation

### Motivation

The `edge_terms` component of the HDC vector directly encodes which edges exist
in the target graph. For each node pair (i, j), the product `bind(hv_i, hv_j)`
appears in `edge_terms` proportionally to the edge count. By probing `edge_terms`
with each possible edge HV via cosine similarity, we can extract a per-edge
"should this edge exist?" signal and use it to modulate the flow model's
predicted edge probabilities.

This is O(N^2 * D) per step vs O(K * forward_pass) for the current K-candidate
approach, while providing richer per-edge guidance.

### Mechanism

Given the target `edge_terms` (extracted from the conditioning vector), for each
pair of nodes (i, j) with known feature tuples `t_i` and `t_j`:

1. Look up the edge HV: `hv_edge = edges_codebook[edges_indexer.get_idx((t_i, t_j))]`
   or equivalently `hv_edge = bind(nodes_codebook[idx_i], nodes_codebook[idx_j])`.
2. Compute similarity: `sim_ij = cos(edge_terms, hv_edge)`.
3. **High similarity** means this edge type likely exists in the target graph.
   Boost `pred_E[b, i, j, 1:]` (bond classes). **Low similarity** means this edge
   likely does not exist. Boost `pred_E[b, i, j, 0]` (no-bond class).
4. Apply as a multiplicative or additive modifier to `pred_E` before computing
   the rate matrix.

### Refinement: Iterative Residual Unbinding

After identifying high-confidence edges, subtract their HVs from a residual copy
of `edge_terms` (exactly as `decode_order_one_no_node_terms` does at
`encoder.py:621-664`). The remaining residual can be probed for remaining edges
with higher precision. This can be applied incrementally during sampling -- as
edges crystallize at later timesteps, subtract their contributions and re-probe
for the remaining structure.

### Implementation Sketch

```python
def compute_edge_unbinding_prior(
    self,
    edge_terms: Tensor,       # (bs, D) -- edge_terms from HDC vector
    node_features: Tensor,     # (bs, n, 5) -- raw integer node features
    node_mask: Tensor,         # (bs, n)
    hypernet: HyperNet,
) -> Tensor:
    """
    Compute per-edge prior from HDC edge_terms via cosine probing.

    Returns:
        edge_prior: (bs, n, n) -- similarity score per edge pair.
                    Positive = edge likely exists; near-zero = unlikely.
    """
    bs, n = node_mask.shape
    D = edge_terms.shape[-1]
    device = edge_terms.device

    edge_prior = torch.zeros(bs, n, n, device=device)

    for b in range(bs):
        n_valid = node_mask[b].sum().int().item()
        if n_valid < 2:
            continue

        # Get node feature tuples for valid nodes
        raw_feats = node_features[b, :n_valid].long()  # (n_valid, 5)

        # Look up node HVs from codebook
        node_indices = []
        for i in range(n_valid):
            tup = tuple(raw_feats[i].tolist())
            idx = hypernet.nodes_indexer.get_idx(tup)
            node_indices.append(idx)

        node_hvs = hypernet.nodes_codebook[node_indices]  # (n_valid, D)

        # Compute all pairwise edge HVs: bind(hv_i, hv_j)
        # Shape: (n_valid, n_valid, D)
        hv_i = node_hvs.unsqueeze(1).expand(-1, n_valid, -1)
        hv_j = node_hvs.unsqueeze(0).expand(n_valid, -1, -1)
        edge_hvs = torchhd.bind(hv_i.reshape(-1, D), hv_j.reshape(-1, D))
        edge_hvs = edge_hvs.reshape(n_valid, n_valid, D)

        # Cosine similarity of each edge HV against edge_terms
        et = edge_terms[b]  # (D,)
        sims = torch.nn.functional.cosine_similarity(
            edge_hvs, et.unsqueeze(0).unsqueeze(0).expand_as(edge_hvs), dim=-1
        )  # (n_valid, n_valid)

        edge_prior[b, :n_valid, :n_valid] = sims

    return edge_prior
```

**Integration into sampling**: In `_sample_step()`, after computing `pred_E`:

```python
# Modulate predicted edge probabilities with HDC prior
if edge_prior is not None:
    # edge_prior: (bs, n, n), values in [-1, 1]
    # Convert to bond-existence weight
    bond_weight = (edge_prior.unsqueeze(-1) * strength).exp()  # (bs, n, n, 1)

    # Apply: boost bond classes where prior is positive
    pred_E_modified = pred_E.clone()
    pred_E_modified[:, :, :, 1:] *= bond_weight  # boost bonds
    pred_E_modified[:, :, :, 0] *= (1.0 / bond_weight.squeeze(-1))  # suppress no-bond
    pred_E_modified = pred_E_modified / pred_E_modified.sum(-1, keepdim=True)
```

### Key Considerations

- **Batch-vectorized implementation**: The per-batch loop above can be vectorized
  if all samples in a batch have the same number of valid nodes (pad and mask).
- **When to apply**: Most useful in mid-to-late sampling (t > 0.3) when the flow
  model's predictions are starting to crystallize. At very early timesteps the
  signal may be noisy.
- **Strength schedule**: Use a time-dependent strength that increases with t:
  `strength = gamma * t` so early steps are weakly guided and late steps are
  strongly guided.
- **Computational cost**: O(N^2 * D) per step. For N=20 atoms and D=256, this is
  ~1M multiply-adds -- negligible compared to a transformer forward pass.

### Extracting edge_terms from the Conditioning Vector

The conditioning vector is `[order_0 | order_N]`. The `edge_terms` are NOT directly
stored in this vector -- they are a separate output of `HyperNet.forward()`.

**Option A**: Modify the preprocessing to store `edge_terms` alongside `hdc_vector`.
In `preprocess_for_flow_edge_decoder()` (`flow_edge_decoder.py:2046-2138`), the
HyperNet forward pass at line 2111 already returns `edge_terms`. Save them:

```python
hdc_out = hypernet.forward(data_for_hdc)
order_n = hdc_out["graph_embedding"]
edge_terms = hdc_out["edge_terms"]  # <-- save this too
```

Store as `new_data.edge_terms = edge_terms.cpu()`.

**Option B**: Re-derive `edge_terms` at sampling time by running HyperNet on the
fixed node features with the target edges. This requires knowing the target graph,
which is available during training but not during generation. Option A is preferred.

---

## 2. Gradient-Based Rate Matrix Correction (Soft HDC)

### Motivation

The current `sample_with_hdc_guidance()` (`flow_edge_decoder.py:1528-1764`) samples
K candidate edge matrices, encodes each with HyperNet, picks the best by cosine
distance, and adds an R^HDC rate matrix term. This is:

- **Wasteful**: K candidates explore almost nothing of the O(5^(N^2)) edge space.
- **All-or-nothing**: R^HDC pushes toward one specific discrete candidate, not a
  distribution.
- **Expensive**: K * S HyperNet forward passes per molecule.

A gradient-based approach computes `d(HDC_distance) / d(edge_logits)` in a single
backward pass, providing **per-edge, per-class** guidance.

### Key Enabler: Soft HDC Encoding

The HyperNet encoder uses sparse `edge_index` for message passing
(`encoder.py:346-347`):

```python
messages = node_hv_stack[layer_index][dsts]
aggregated = scatter_hd(messages, srcs, dim_size=node_dim, op="bundle")
```

This can be replaced with a **soft adjacency matrix** multiplication:

```python
soft_A = pred_E[:, :, :, 1:].sum(dim=-1)  # probability of any bond, (bs, n, n)
soft_messages = soft_A @ node_hv_stack[layer_index]  # (bs, n, D) -- weighted neighbor sum
```

Since bundle = sum and matrix multiplication is a weighted sum, this is an exact
relaxation for the bundling operation. The bind step (`torchhd.bind`) is circular
convolution for HRR, which is differentiable via FFT.

### Mechanism

```python
def compute_soft_hdc_gradient(
    self,
    pred_E_logits: Tensor,   # (bs, n, n, de) -- raw logits from model
    node_hvs: Tensor,         # (bs, n, D) -- pre-computed node HVs
    target_order_n: Tensor,   # (bs, D) -- target order_N embedding
    node_mask: Tensor,        # (bs, n)
    depth: int,               # message passing depth
    tau: float = 0.5,         # Gumbel-softmax temperature
) -> Tensor:
    """
    Compute gradient of HDC distance w.r.t. edge logits via soft encoding.

    Returns:
        grad: (bs, n, n, de) -- gradient of HDC distance w.r.t. edge logits
    """
    logits = pred_E_logits.detach().clone().requires_grad_(True)

    # Soft edge probabilities via Gumbel-softmax (or plain softmax)
    soft_E = F.softmax(logits / tau, dim=-1)  # (bs, n, n, de)

    # Soft adjacency: probability of any bond existing
    soft_A = soft_E[:, :, :, 1:].sum(dim=-1)  # (bs, n, n)

    # Mask invalid edges
    edge_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
    soft_A = soft_A * edge_mask.float()

    # Soft HDC encoding via matrix multiplication message passing
    hv_stack = [node_hvs]  # layer 0 = node HVs
    for layer in range(depth):
        prev_hv = hv_stack[-1]  # (bs, n, D)
        # Soft message passing: weighted sum of neighbor HVs
        aggregated = torch.bmm(soft_A, prev_hv)  # (bs, n, D)
        # Bind with own HV (circular convolution, differentiable)
        hr = torchhd.bind(prev_hv.reshape(-1, D), aggregated.reshape(-1, D))
        hr = hr.reshape(bs, n, D)
        # Normalize
        hr = hr / (hr.norm(dim=-1, keepdim=True) + 1e-8)
        hv_stack.append(hr)

    # Readout: bundle across layers, then across nodes
    stacked = torch.stack(hv_stack, dim=2)          # (bs, n, depth+1, D)
    node_emb = torchhd.multibundle(stacked)          # (bs, n, D)
    # Mask and sum
    node_emb = node_emb * node_mask.unsqueeze(-1).float()
    graph_emb = node_emb.sum(dim=1)                  # (bs, D)

    # Cosine distance to target
    graph_norm = F.normalize(graph_emb, dim=-1)
    target_norm = F.normalize(target_order_n, dim=-1)
    hdc_dist = 1.0 - (graph_norm * target_norm).sum(dim=-1).mean()

    # Backward pass
    grad = torch.autograd.grad(hdc_dist, logits)[0]  # (bs, n, n, de)
    return grad
```

### Integration into Rate Matrix

In the sampling loop, after the standard forward pass:

```python
# Compute gradient-based HDC guidance
grad = self.compute_soft_hdc_gradient(
    pred.E, node_hvs, target_order_n, node_mask, depth=hypernet.depth
)

# Construct R^HDC from gradient
# Negative gradient = direction of decreasing distance = desirable transitions
R_hdc = F.relu(-grad) * gamma  # only keep directions that reduce distance
R_hdc = R_hdc / (Z_t_E * pt_at_Et).unsqueeze(-1).clamp(min=1e-8)

# Zero out self-transitions (diagonal in edge-class space)
E_t_label = E_t.argmax(dim=-1)
R_hdc.scatter_(-1, E_t_label.unsqueeze(-1), 0.0)

# Add to total rate matrix
R_t_E = R_t_E + R_hdc
```

### Alternative: Dual-Channel Blending

Instead of adding a separate R^HDC term, blend the gradient signal into the
model's prediction before rate matrix computation:

```python
# HDC-informed target distribution from negative gradient
E_1_hdc = F.softmax(-grad / temperature, dim=-1)

# Blend with model prediction
alpha = gamma * (1 - t)  # more HDC influence early, less later
E_1_mixed = (1 - alpha) * pred_E + alpha * E_1_hdc

# Use E_1_mixed in rate matrix computation (replaces pred_E)
R_t_X, R_t_E = self.rate_matrix_designer.compute_rate_matrices(
    t, node_mask, X_t, E_t, pred_X, E_1_mixed
)
```

This is the most principled approach: all three rate matrix components (R\*, R^DB,
R^TG) benefit from HDC information.

### Key Considerations

- **torch.enable_grad() during sampling**: The `sample()` method uses
  `@torch.no_grad()`. The gradient computation needs a `torch.enable_grad()`
  context for the soft HDC encoding section only.
- **Soft encoding quality**: The soft encoding approximates the discrete encoding.
  The approximation is good when `pred_E` is close to one-hot (late timesteps)
  and poor when it is uniform (early timesteps). This naturally matches the
  desired behavior: gradient guidance is most reliable when the model is confident.
- **Gumbel-softmax temperature**: Lower tau makes soft edges sharper (closer to
  discrete) but the gradient becomes noisier. tau=0.5 is a reasonable starting
  point.
- **Gamma calibration**: The gradient magnitude depends on the HDC distance and
  the temperature. Normalize the gradient by its L2 norm and scale by gamma to
  decouple the strength from these factors.
- **Cost**: One backward pass through the soft HDC encoder per step. This is much
  cheaper than a GraphTransformer forward pass since the soft encoder only involves
  matrix multiplications and element-wise operations.
- **Node HV pre-computation**: `node_hvs` (the layer-0 node hypervectors) should
  be computed once before sampling starts using `hypernet.encode_properties()`.
  They do not change since nodes are fixed. Store as `(bs, n_max, D)` padded tensor.

### Relationship to Existing Methods

This replaces the discrete K-candidate approach in `sample_with_hdc_guidance()`
(`flow_edge_decoder.py:1528-1764`) with a continuous gradient signal. The existing
`_compute_Rhdc()` method (`flow_edge_decoder.py:1437-1478`) provides the template
for how R^HDC integrates into the rate matrix.

---

## 3. Delta Guidance from Incremental Encoding

### Motivation

Rather than the crude "sample K, pick best" strategy, we can compute a
**directional signal** by comparing the current partial graph's HDC encoding
with the target, and use the *difference vector* to determine which edges should
be added or removed.

The key algebraic insight: adding edge (i, j) to a graph approximately shifts
its HDC encoding by `+bind(hv_i, hv_j)`. Therefore, if
`delta = target_order_N - current_order_N`, checking the alignment of each
possible edge HV with `delta` tells us whether that edge *should* exist in
the target.

### Mechanism

At each sampling step t:

1. **Encode current state**: Run HyperNet on the current edge state E_t to get
   `current_order_N`. This requires converting E_t from dense one-hot
   `(bs, n, n, de)` to sparse `edge_index` format -- use `dense_to_pyg()`.

2. **Compute delta**: `delta = target_order_N - current_order_N` (shape `(bs, D)`).
   This vector points "toward" the target in HDC space.

3. **Per-edge benefit score**: For each node pair (i, j), compute:
   `benefit_ij = cos(delta, bind(hv_i, hv_j))`
   This measures how much adding/keeping edge (i, j) would reduce the distance
   to the target.

4. **Modulate edge probabilities**: Use `benefit_ij` to adjust `pred_E`:
   - Positive benefit (adding this edge helps): boost bond probabilities.
   - Negative benefit (this edge should not exist): boost no-bond probability.

### Implementation Sketch

```python
def compute_delta_guidance(
    self,
    E_t: Tensor,               # (bs, n, n, de) current one-hot edges
    node_features: Tensor,      # (bs, n, 5) raw integer features
    node_mask: Tensor,          # (bs, n)
    target_order_n: Tensor,     # (bs, D) target order_N
    hypernet: HyperNet,
) -> Tensor:
    """
    Compute per-edge benefit score via HDC delta guidance.

    Returns:
        benefit: (bs, n, n) -- positive means "add/keep this edge"
    """
    bs, n, _, de = E_t.shape
    device = E_t.device

    # 1. Encode current state with HyperNet
    E_t_label = E_t.argmax(dim=-1)  # (bs, n, n)
    current_order_n = self._encode_candidates_to_order_n(
        E_t_label.unsqueeze(0),  # (1, bs, n, n) -- K=1 candidate
        node_features, node_mask, hypernet
    ).squeeze(0)  # (bs, D)

    # 2. Compute delta
    delta = target_order_n - current_order_n  # (bs, D)

    # 3. Per-edge benefit via cosine similarity with edge HVs
    D = delta.shape[-1]
    benefit = torch.zeros(bs, n, n, device=device)

    for b in range(bs):
        n_valid = node_mask[b].sum().int().item()
        if n_valid < 2:
            continue

        raw_feats = node_features[b, :n_valid].long()
        node_indices = []
        for i in range(n_valid):
            tup = tuple(raw_feats[i].tolist())
            node_indices.append(hypernet.nodes_indexer.get_idx(tup))

        node_hvs = hypernet.nodes_codebook[node_indices]  # (n_valid, D)

        # Pairwise edge HVs
        hv_i = node_hvs.unsqueeze(1).expand(-1, n_valid, -1)
        hv_j = node_hvs.unsqueeze(0).expand(n_valid, -1, -1)
        edge_hvs = torchhd.bind(
            hv_i.reshape(-1, D), hv_j.reshape(-1, D)
        ).reshape(n_valid, n_valid, D)

        # Cosine similarity with delta
        d = delta[b]  # (D,)
        sims = F.cosine_similarity(
            edge_hvs, d.unsqueeze(0).unsqueeze(0).expand_as(edge_hvs), dim=-1
        )
        benefit[b, :n_valid, :n_valid] = sims

    return benefit
```

**Integration**: Same pattern as Idea 1 -- use `benefit` to modulate `pred_E`.

### Comparison with Idea 1 (Unbinding Probes)

| Aspect | Idea 1 (Unbinding) | Idea 3 (Delta) |
|--------|-------------------|----------------|
| Signal source | Raw `edge_terms` | `target - current` difference |
| Adapts to current state | No (static prior) | Yes (re-encodes at each step) |
| Cost per step | O(N^2 * D) | O(N^2 * D + HyperNet_forward) |
| Best for | Static edge presence prior | Dynamic course-correction |

**Recommendation**: Use Idea 1 as a cheap static prior throughout sampling, and
Idea 3 periodically (every 5-10 steps) for dynamic correction when the cost of
a HyperNet forward is justified.

### Key Considerations

- **HyperNet encoding cost**: One HyperNet forward per step per batch. For a
  batch of 32 molecules with 20 atoms, this is ~75K multiply-adds -- negligible
  vs the transformer. The bottleneck is `Batch.from_data_list()` (CPU-bound).
  Pre-allocate batch structures to amortize.
- **Frequency**: Apply delta guidance every K steps (e.g., K=5) to reduce the
  overhead of re-encoding. Between checkpoints, fall back to the static
  unbinding prior from Idea 1.
- **Symmetry**: `bind(hv_i, hv_j)` may not equal `bind(hv_j, hv_i)` for HRR.
  Average both directions: `edge_hv = 0.5 * (bind(hv_i, hv_j) + bind(hv_j, hv_i))`.
- **Late-stage refinement**: Delta guidance is most valuable in the last 20-30%
  of sampling steps when the graph is nearly formed and fine structural details
  matter. The existing `start_time` parameter in `sample()` supports this.

---

## 4. Self-Play with Hindsight Relabeling

### Motivation

The FlowEdgeDecoder is trained on graphs from ZINC and BRICS fragment
combinations. At inference time, it generates graphs that may differ
systematically from the training distribution (e.g., different error patterns,
partial structures it has never seen). Training on the model's own outputs --
re-labeled with their true HDC vectors -- closes this distribution gap.

This is analogous to DAgger (Dataset Aggregation) in imitation learning: augment
the training set with the learner's own trajectories labeled by the expert (here,
the deterministic HDC encoder).

### Mechanism

Periodically during training:

1. **Sample**: Generate a batch of molecules using the current model checkpoint
   (run `sample()` with the current weights).
2. **Encode**: Run HyperNet on each generated molecule to compute its *actual*
   HDC vector `[order_0 | order_N]`.
3. **Relabel**: Create training pairs `(generated_graph, actual_hdc_vector)`.
   Even if the generated graph is wrong (doesn't match the conditioning HDC),
   the pair `(graph, HyperNet(graph))` is a valid training example.
4. **Inject**: Add these self-play examples to the training data stream.

### Implementation within the Streaming Framework

The streaming experiment (`train_flow_edge_decoder_streaming.py`) already uses
a `MixedStreamingDataLoader` with multiple `StreamingSource` instances (fragments
at 90%, small molecules at 10%). Self-play examples are naturally a third source.

#### Step 1: Self-Play Data Generator

Create a new streaming source that generates data from the model's own outputs.

```python
class SelfPlayStreamingDataset:
    """
    Streaming dataset that generates training data from the model's own outputs.

    Periodically samples molecules from the current model checkpoint, re-encodes
    them with HyperNet, and yields the (graph, HDC_vector) pairs as training data.
    """

    def __init__(
        self,
        model_checkpoint_path: str,
        hypernet_checkpoint_path: str,
        buffer_size: int = 1000,
        num_source_samples: int = 100,
        sample_steps: int = 50,
        refresh_interval: int = 500,  # regenerate every N batches consumed
    ):
        self.model_checkpoint_path = model_checkpoint_path
        self.hypernet_checkpoint_path = hypernet_checkpoint_path
        self.buffer_size = buffer_size
        self.num_source_samples = num_source_samples
        self.sample_steps = sample_steps
        self.refresh_interval = refresh_interval
        self._buffer = []
        self._consumed = 0

    def _generate_batch(self):
        """Sample from current model, re-encode with HyperNet."""
        model = FlowEdgeDecoder.load(self.model_checkpoint_path)
        hypernet = HyperNet.load(self.hypernet_checkpoint_path)

        # Need source HDC vectors + node features to condition on.
        # Use random HDC vectors from the training distribution,
        # or sample from previously seen valid molecules.
        # ... (see details below)

        samples = model.sample(hdc_vectors, node_features, node_mask,
                               sample_steps=self.sample_steps)

        # Re-encode each sample with HyperNet
        for data in samples:
            data_for_hdc = data.clone()
            data_for_hdc.batch = torch.zeros(data.x.size(0), dtype=torch.long)
            data_for_hdc = hypernet.encode_properties(data_for_hdc)
            out = hypernet.forward(data_for_hdc)
            order_0 = scatter_hd(data_for_hdc.node_hv, data_for_hdc.batch, op="bundle")
            order_n = out["graph_embedding"]
            hdc_vector = torch.cat([order_0, order_n], dim=-1)

            # Create training example with the ACTUAL HDC vector (not the conditioning one)
            new_data = preprocess_for_flow_edge_decoder(data, hypernet)
            if new_data is not None:
                new_data.hdc_vector = hdc_vector  # overwrite with actual
                self._buffer.append(new_data)
```

#### Step 2: Integration with MixedStreamingDataLoader

Add as a third `StreamingSource` in `load_train_data` hook:

```python
sources = [
    StreamingSource(name="fragments", dataset=fragment_dataset, weight=0.85),
    StreamingSource(name="small_molecules", dataset=small_mol_dataset, weight=0.10),
    StreamingSource(name="self_play", dataset=self_play_dataset, weight=0.05),
]
```

Start with a small weight (5%) to avoid destabilizing training.

#### Step 3: Periodic Model Checkpoint Updates

The self-play source needs access to the current model weights. Two approaches:

**A. Shared checkpoint file**: The training loop saves a checkpoint every N epochs.
The self-play worker loads the latest checkpoint when refreshing its buffer.
This is the simplest approach and naturally stale (worker uses weights from
N epochs ago), which actually helps stability.

**B. Copy-on-write**: After each epoch, copy the current model weights to a
shared location. The self-play worker picks up the new weights on its next
refresh cycle. This requires coordination but provides fresher samples.

Approach A is recommended for simplicity. The staleness is actually beneficial --
it prevents the feedback loop from becoming too tight (which can cause mode
collapse).

### Key Considerations

- **Distribution shift prevention**: Self-play examples should be a small fraction
  (5-10%) of the training data. Too much self-play data can cause the model to
  overfit to its own error patterns (mode collapse).
- **Quality filtering**: Optionally filter self-play examples by chemical validity
  (RDKit sanitization) or HDC distance (only keep samples whose generated HDC is
  within some threshold of the conditioning HDC). This ensures the model learns
  from reasonable outputs, not garbage.
- **Warm start**: Don't start self-play until the model has trained for some
  initial period (e.g., 100 epochs) to ensure the generated samples are
  non-trivial.
- **Staleness is a feature**: Using model weights from N epochs ago for generation
  provides implicit regularization and prevents tight feedback loops.
- **What conditioning to use for generation**: The self-play generator needs
  source HDC vectors and node features to condition generation. Options:
  - **Replay buffer**: Store HDC vectors from the training data and randomly
    sample them.
  - **Random perturbation**: Take real HDC vectors and add small noise to
    explore nearby regions of HDC space.
  - **Cross-source**: Use HDC vectors from fragment-generated molecules
    (available from the fragment streaming source).
- **Computational cost**: Generating samples is expensive (100 sampling steps
  per molecule). Run generation in a background process with low priority.
  The streaming architecture already supports this via worker processes.

### Relationship to Other Ideas

Self-play is orthogonal to the other three ideas and can be combined with any
of them. It addresses the **training side** (improving the model's predictions)
while Ideas 1-3 address the **inference side** (better using HDC during sampling).
Together they provide a complete loop:

1. Train with self-play to make predictions more HDC-consistent.
2. During sampling, use unbinding probes (Idea 1) for static edge priors.
3. Periodically apply delta guidance (Idea 3) for dynamic correction.
4. Optionally use soft HDC gradients (Idea 2) for per-edge rate matrix correction.

---

## Summary

| Idea | Type | Cost per Step | Signal Quality | Complexity |
|------|------|---------------|----------------|------------|
| 1. Unbinding Probes | Inference | O(N^2 * D) | Per-edge static prior | Low |
| 2. Soft HDC Gradient | Inference | 1 backward pass | Per-edge, per-class, adaptive | Medium-High |
| 3. Delta Guidance | Inference | O(N^2 * D + HyperNet) | Per-edge, adaptive | Medium |
| 4. Self-Play | Training | Background generation | Distributional improvement | Medium |

### Recommended Implementation Order

1. **Idea 1** (Unbinding Probes) -- simplest, provides immediate value as a
   static prior. Requires storing `edge_terms` in the data.
2. **Idea 3** (Delta Guidance) -- builds on Idea 1 infrastructure, adds dynamic
   adaptation. Apply every 5-10 steps.
3. **Idea 4** (Self-Play) -- independent training improvement. Can be developed
   in parallel with inference-side ideas.
4. **Idea 2** (Soft HDC Gradient) -- most sophisticated. Requires implementing
   a differentiable soft HDC encoder. Should be attempted after validating that
   HDC guidance improves sampling quality (via Ideas 1 and 3).
