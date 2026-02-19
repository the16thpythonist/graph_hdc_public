# HDC-Guided Discrete Flow Matching via TAG

This document describes a principled approach to guiding the FlowEdgeDecoder's
discrete flow matching sampling process using the HDC encoder as a predictor,
based on the **Discrete Guidance** framework (Nisonoff, Xiong, Allenspach &
Listgarten, UC Berkeley).

**Paper reference:** `dfm_guidance.md` in this repository.
**Source code:** https://github.com/hnisonoff/discrete_guidance

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Background: Discrete Guidance Framework](#2-background-discrete-guidance-framework)
3. [Our Setting: FlowEdgeDecoder + HDC Encoder](#3-our-setting-flowedgedecoder--hdc-encoder)
4. [Derivation: TAG with HDC Cosine Similarity](#4-derivation-tag-with-hdc-cosine-similarity)
5. [Critical Review Findings](#5-critical-review-findings)
6. [Refined Implementation Plan](#6-refined-implementation-plan)
7. [Evaluation Protocol](#7-evaluation-protocol)
8. [Appendix: Full Mathematical Derivation](#8-appendix-full-mathematical-derivation)

---

## 1. Motivation

The FlowEdgeDecoder reconstructs molecular graphs (edges) from HDC vectors using
a DeFoG-based discrete flow model (DFM). Currently, HDC information enters the
generation process only as **conditioning** (via MLP projection and cross-attention).
The model learns to map HDC vectors to edge distributions, but at inference time
the sampling trajectory is not explicitly steered toward high HDC similarity.

Existing approaches to improve reconstruction:
- **Best-of-N sampling**: Generate N independent samples, pick the one with
  highest HDC similarity. Simple but requires N forward passes (typically N=256).
- **`sample_with_soft_hdc_guidance()`**: Ad-hoc gradient-based guidance that
  either blends gradients into predictions or adds them to rate matrices.
  Works in practice but lacks theoretical grounding.

**The idea**: Use the Discrete Guidance framework to derive a *principled*
rate-matrix modulation that steers the CTMC sampling process toward states with
high HDC similarity to the target vector. Specifically, use the
**Taylor-Approximated Guidance (TAG)** variant for computational efficiency, with
the HDC encoder as a time-independent predictor.

---

## 2. Background: Discrete Guidance Framework

### 2.1 Discrete Flow Models as CTMCs

Discrete flow models (DFM, Campbell et al. 2024) define a generative process via
Continuous-Time Markov Chains (CTMCs). The dynamics are specified by rate matrices
R_t(x, x̃) relating to transition probabilities:

```
p(x_{t+dt} = x̃ | x_t = x) = δ_{x,x̃} + R_t(x, x̃) · dt
```

A denoising model p_θ(x_1 | x_t) is trained to predict clean data from noisy
data, which induces the rate matrices used for sampling. Starting from noise at
t=0, one integrates forward to t=1 to generate samples.

### 2.2 Key Insight: Tractability in Continuous Time

In a D-dimensional discrete state space where each dimension has cardinality S,
there are S^D possible states. Naively, conditioning via Bayes' theorem requires
summing over all S^D states — intractable.

However, in continuous-time formulations with independent per-dimension noising,
**only one dimension can change at any instant**. This reduces the sum to
D × (S-1) + 1 terms, making guidance tractable.

### 2.3 Predictor Guidance (Paper Eq. 2)

The conditional (guided) rates are obtained by modulating unconditional rates
with a likelihood ratio:

```
R_t(x, x̃ | y) = [p(y | x̃, t) / p(y | x, t)]^γ  ·  R_t(x, x̃)     for x̃ ≠ x
```

where:
- `R_t(x, x̃)` = unconditional rate (from the trained DFM)
- `p(y | ·, t)` = time-dependent predictor of property y
- `γ` = guidance strength (γ=0: unconditional, γ=1: exact conditioning, γ>1: amplified)

The diagonal (self-transition) rate is recomputed for conservation:

```
R_t(x, x | y) = -Σ_{x' ≠ x} R_t(x, x' | y)
```

### 2.4 Taylor-Approximated Guidance (TAG, Paper Eq. 4)

Computing exact guidance requires D × (S-1) forward passes of the predictor per
sampling step (one for each possible next state). TAG reduces this to O(1) by
using a first-order Taylor expansion:

```
log [p(y | x̃, t) / p(y | x, t)]  ≈  (x̃ - x)^T  ∇_x log p(y | x, t)
```

This requires only one forward pass and one backward pass of the predictor. Since
x̃ and x differ in only one dimension (one-hot encoded), the dot product reduces
to a simple index lookup into the gradient vector.

### 2.5 Predictor-Free Guidance (Paper Eq. 3)

Alternatively, one can blend conditional and unconditional rates without a
separate predictor:

```
R_t^(γ)(x, x̃ | y) = R_t(x, x̃ | y)^γ  ·  R_t(x, x̃)^(1-γ)
```

This requires both a conditional and unconditional model but no separate
predictor. We do not pursue this variant here since we want to leverage the
HDC encoder as an external predictor.

---

## 3. Our Setting: FlowEdgeDecoder + HDC Encoder

### 3.1 The Generative Model

The `FlowEdgeDecoder` is a DeFoG-based discrete flow model that:
- Keeps **node features fixed** (24-dim one-hot: atom_type, degree, charge, num_Hs, is_in_ring)
- Denoises **edges only** (5 classes: no_edge, single, double, triple, aromatic)
- Conditions on a pre-computed **HDC vector** (512-dim: [order_0 | order_N])
- Uses a masking flow: `p(E_t | E_1) = t · δ(E_t, E_1) + (1-t) · p_0(E_t)`

At sampling time, rate matrices `R_t_E` are computed by `RateMatrixDesigner`
from the transformer's predicted edge distributions. Euler integration with
configurable stochasticity (η) and time distortion (polydec) drives the CTMC.

### 3.2 The HDC Encoder as Predictor

The HyperNet encoder deterministically maps a molecular graph to a hypervector
via VSA message passing (bind + bundle + scatter). For guidance, we use the
**soft HDC encoder** (`_soft_hdc_encode`), which replaces discrete graph
operations with differentiable relaxations:

- Discrete adjacency → soft adjacency via softmax over edge logits
- `scatter_hd(bundle)` → `torch.bmm(soft_A, node_hvs)` (weighted sum)
- `torchhd.bind` → `torch.fft.ifft(fft(a) * fft(b)).real` (circular convolution)
- L2 normalization after each message-passing layer

This makes the encoding fully differentiable w.r.t. edge logits, enabling
gradient computation for TAG.

### 3.3 Cosine Similarity as Likelihood

We define the predictor likelihood as a Boltzmann distribution over cosine
similarity:

```
p(y | x, t)  ∝  exp( cos_sim(HDC_encode(x), hdc_target) / τ )
```

where `y = hdc_target` is the target HDC vector, `x` is the current graph state,
and `τ` is a temperature parameter controlling sharpness. This is an
energy-based model where the energy is the negative cosine similarity.

---

## 4. Derivation: TAG with HDC Cosine Similarity

Starting from the paper's predictor guidance (Eq. 2) and applying the TAG
approximation (Eq. 4) with our Boltzmann-style HDC likelihood, we derive the
concrete formulas for our setting.

### 4.1 The Log-Likelihood Ratio

The guidance requires computing the ratio `p(y|x̃,t) / p(y|x,t)` for each
possible next state x̃. Taking the log:

```
log [p(y|x̃,t) / p(y|x,t)] = [cos_sim(HDC(x̃), target) - cos_sim(HDC(x), target)] / τ
```

### 4.2 TAG Approximation

Define `f(x) = cos_sim(HDC_encode(x), hdc_target) / τ`. The first-order Taylor
expansion around the current state x gives:

```
f(x̃) - f(x)  ≈  (x̃ - x)^T · ∇_x f(x)
```

Since our state is a one-hot edge matrix and x̃ differs from x in exactly one
edge position (i,j) changing from class e to class e':

```
(x̃ - x)^T · ∇_x f(x)  =  ∇f[i,j,e'] - ∇f[i,j,e]
```

where `∇f = ∇_{edge_logits} [cos_sim(soft_HDC_encode(softmax(logits/τ)), target) / τ]`.

### 4.3 The Guided Rate Matrices

Combining with the paper's Eq. 2:

```
R_t^guided(x, x̃ | y) = exp(γ · [∇f[i,j,e'] - ∇f[i,j,e]]) · R_t(x, x̃)
```

In tensor form for all edges simultaneously:

```python
grad = ∇_{logits} [cos_sim(soft_HDC_encode(softmax(logits/τ)), target) / τ]
grad_at_current = grad.gather(-1, E_t_label.unsqueeze(-1))  # (bs, n, n, 1)
log_ratio = grad_at_current - grad                           # (bs, n, n, de)
R_guided = R_unconditional * exp(γ_t * log_ratio)
```

**Note on sign**: Since `loss = 1 - cos_sim` and `grad = ∇ loss`, the gradient
points toward *increasing* loss. Therefore `log_ratio = grad_at_current - grad`
gives positive values for transitions that *decrease* the loss (improve
similarity), which are the transitions we want to amplify.

### 4.4 Conservation

After modulating off-diagonal rates, the diagonal is recomputed:

```
R_t^guided(x, x | y) = -Σ_{x' ≠ x} R_t^guided(x, x' | y)
```

In practice this is handled by the existing Euler step code: zero the diagonal,
compute `stay_prob = 1 - Σ(jump_probs)`, clamp, and normalize.

---

## 5. Critical Review Findings

A three-agent review team (mathematical, implementation, ML/experimental) was
convened to critically assess this plan. Below are the key findings.

### 5.1 Theoretical Soundness (Math Review)

**Validated:**
- The Boltzmann likelihood is valid for likelihood ratios. The conditional
  independence assumption (`x_t ⊥ y | x_{t+dt}`) holds because the noising
  process is independent of the target HDC vector.
- The soft HDC encoder is C∞ smooth. The Taylor approximation is mathematically
  valid. Accuracy degrades with message-passing depth due to composed nonlinear
  operations (curvature grows as ~2^depth).
- Rate non-negativity is preserved: `exp(·)` is always positive, so
  `R_guided = exp(·) · R_unconditional ≥ 0` when `R_unconditional ≥ 0`.

**Concern — time-independence is the biggest gap (HIGH severity):**
- The paper requires a time-conditioned predictor `p^φ(y|x,t)` trained on noisy
  data at each noise level. Our time-independent HDC encoder produces
  **meaningless gradients at early timesteps** (t near 0) when the state is
  mostly noise.
- A linear ramp schedule (`γ_t = γ_max · t`) partially compensates but still
  allows non-trivial guidance on random data at early times.
- **Recommendation**: Use a threshold + ramp schedule: `γ = 0` for `t < t_min`
  (e.g., 0.3-0.5), then ramp to `γ_max`. Apply in distorted time to match the
  polydec schedule.

**Concern — Taylor accuracy vs depth (LOW-MEDIUM severity):**
- For depth=1, the TAG approximation is accurate. For depth=3 (common), the
  Hessian norm grows, making second-order terms significant.
- **Recommendation**: Use depth 1-2 for the guidance encoder, independent of
  the encoding depth used for the target vector.

### 5.2 Implementation Feasibility (Code Review)

**Validated:**
- Gradient flow through `_soft_hdc_encode` is unbroken. All operations (softmax,
  bmm, FFT bind, L2 norm) are differentiable. Node HVs are fixed constants.
- Memory overhead is manageable (~80MB for bs=64, n=50, D=512, depth=3).
- Compute overhead is modest (~10-20% per step).
- Node masking is correctly handled (edge_mask zeros out invalid positions).

**Concern — gradient symmetry (MEDIUM severity):**
- The `triu + transpose` symmetrization of `soft_A` inside `enable_grad()` means
  only upper-triangle `logits_g` entries get gradients. Lower-triangle entries
  have zero gradient.
- **Fix**: Explicitly symmetrize: `grad = (grad + grad.transpose(1,2)) / 2`.

**Concern — injection point:**
- Rate modulation should be applied **after** `compute_rate_matrices()` returns
  `R_t_E` but **before** the Euler step converts rates to probabilities. This
  is clean and requires no changes to `RateMatrixDesigner`.

**Concern — `_compute_Rhdc()` is not reusable:**
- The existing helper follows an additive pattern with `Z_t_E` / `pt_at_Et`
  normalization. TAG uses multiplicative modulation. A new helper
  `_apply_tag_modulation()` should be written.

### 5.3 ML/Experimental Concerns (ML Review)

**Concern — guidance collapse (HIGH risk):**
- Strong guidance (high γ) can push the model toward chemically invalid graphs
  because the HDC encoder encodes topology, not valence constraints.
- **Recommendation**: Start with γ_max in [0.5, 2.0]. Track validity rate vs
  HDC distance as a Pareto curve.

**Concern — HDC gradient signal quality (MODERATE-HIGH risk):**
- The cosine similarity landscape in HDC space is noisy. Gradient magnitude
  may be diluted across O(n²·5) edge logits for larger molecules.
- **Recommendation**: Run gradient diagnostics first (log gradient norms per
  timestep) before committing to full implementation.

**Concern — double HDC signal (MODERATE risk):**
- HDC enters the model three times: MLP conditioning, cross-attention, and now
  guidance. These can conflict if guidance disagrees with the model's learned
  mapping.
- **Recommendation**: Include an ablation that disables conditioning (zero HDC
  input) to isolate the guidance effect.

**Concern — stochasticity (η):**
- The paper finds η > 0 helps with guidance. Current default is η=0. Guidance
  with η=0 produces a single deterministic trajectory.
- **Recommendation**: Test η in {0, 0.5, 1.0, 5.0}. Start with η=1.0 when
  using guidance.

**Insight — use pred_E at early steps:**
- At early timesteps, the model's predicted edge distribution (argmax of pred_E)
  is more informative than the actual noisy state E_t. The soft HDC encoder
  should operate on `pred_E` logits (which it already does), not on E_t.

---

## 6. Refined Implementation Plan

### Step 0: Gradient Diagnostics

Before implementing TAG, assess the quality of the HDC gradient signal using the
existing `sample_with_soft_hdc_guidance()` infrastructure.

**What to measure** (per timestep, averaged over a batch of ~50 molecules):
- Gradient L2 norm: `grad.norm(dim=-1).mean()`
- Gradient sparsity: fraction of near-zero entries
- Directional consistency: cosine similarity between gradients at successive steps
- HDC similarity of the current state to target

**What to look for:**
- Gradients should be noisy/small at early t (confirming time-independence issue)
  and become structured/larger at late t.
- If gradients are uniformly noisy at ALL timesteps, TAG will not provide useful
  signal and we should reconsider the approach.

### Step 1: Implement `sample_with_tag_guidance()`

Create a **new standalone method** (not a mode in the existing soft guidance).

**Key design decisions:**
- Operates on `pred_E` logits from the transformer (not E_t)
- Uses reduced depth (1-2) for the soft HDC encoder
- Applies gradient symmetrization
- Uses threshold + ramp schedule for γ
- Modulates rate matrices multiplicatively via log-likelihood ratios

**Method signature:**
```python
def sample_with_tag_guidance(
    self,
    hdc_vectors: Tensor,          # (num_samples, hdc_dim)
    node_features: Tensor,        # (num_samples, n_max, num_node_classes)
    node_mask: Tensor,            # (num_samples, n_max)
    raw_node_features: Tensor,    # (num_samples, n_max, num_raw_features)
    hypernet: HyperNet,           # for codebook access
    gamma_max: float = 1.0,       # max guidance strength
    tau: float = 0.5,             # softmax temperature for soft adjacency
    guidance_depth: int = 1,      # message-passing depth for soft HDC
    t_min: float = 0.3,           # guidance starts after this time
    schedule: str = "threshold_ramp",  # gamma schedule type
    eta: float = 1.0,             # CTMC stochasticity
    sample_steps: int = 100,
    **kwargs
) -> List[Data]:
```

**Per-step logic (pseudocode):**
```python
for t_int in range(sample_steps):
    t, s = compute_time_pair(t_int, sample_steps)

    # 1. Transformer forward pass
    pred = self.forward(noisy_data, extra_data, node_mask)
    pred_E_logits = pred.E  # raw logits before softmax
    pred_E = softmax(pred_E_logits, dim=-1)

    # 2. Compute unconditional rate matrices
    R_t_E = rate_matrix_designer.compute_rate_matrices(t, ..., pred_E)

    # 3. Compute gamma for this timestep
    gamma_t = compute_gamma(t, gamma_max, t_min, schedule)

    # 4. TAG modulation (only if gamma_t > 0)
    if gamma_t > 0:
        with torch.enable_grad():
            logits_g = pred_E_logits.detach().clone().requires_grad_(True)
            soft_probs = softmax(logits_g / tau, dim=-1)
            soft_A = soft_probs[:, :, :, 1:].sum(dim=-1)
            soft_A = triu(soft_A, diagonal=1) + triu(soft_A, diagonal=1).T
            soft_A = soft_A * edge_mask

            graph_emb = _soft_hdc_encode(
                soft_A, node_hvs, node_mask,
                depth=guidance_depth, normalize=True
            )
            loss = (1 - cos_sim(graph_emb, target_order_n)).sum()
            grad = autograd.grad(loss, logits_g)[0]

        # Symmetrize gradient
        grad = (grad + grad.transpose(1, 2)) / 2.0

        # Apply TAG modulation
        R_t_E = _apply_tag_modulation(R_t_E, grad, E_t_label, gamma_t)

    # 5. Euler step + sample (unchanged from standard _sample_step)
    ...
```

**Helper: `_apply_tag_modulation()`**
```python
def _apply_tag_modulation(self, R_t_E, grad, E_t_label, gamma):
    """Multiplicative TAG modulation of rate matrices."""
    # grad points toward increasing loss (decreasing similarity)
    # log_ratio > 0 for transitions that IMPROVE similarity
    grad_at_current = grad.gather(-1, E_t_label.unsqueeze(-1))
    log_ratio = grad_at_current - grad  # (bs, n, n, de)

    # Multiplicative modulation
    R_guided = R_t_E * torch.exp(gamma * log_ratio)

    # Zero out diagonal (will be recomputed as stay probability)
    R_guided.scatter_(-1, E_t_label.unsqueeze(-1), 0.0)

    return R_guided
```

**Helper: `compute_gamma()`**
```python
def compute_gamma(t_frac, gamma_max, t_min, schedule):
    """Compute time-dependent guidance strength."""
    if schedule == "threshold_ramp":
        if t_frac < t_min:
            return 0.0
        return gamma_max * (t_frac - t_min) / (1.0 - t_min)
    elif schedule == "constant":
        return gamma_max if t_frac >= t_min else 0.0
    elif schedule == "sigmoid":
        return gamma_max * torch.sigmoid(10 * (t_frac - t_min)).item()
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
```

### Step 2: Refinements

**A. Reduced guidance depth:**
Use `guidance_depth=1` (or at most 2) for the soft HDC encoder in the guidance
computation, regardless of the HyperNet's actual encoding depth. This:
- Improves Taylor approximation accuracy (lower curvature)
- Reduces backward pass compute cost
- Still captures 1-hop structural information

**B. pred_E-based encoding at early steps:**
The soft HDC encoder operates on `pred_E_logits` (the model's predictions), not
on the actual noisy state `E_t`. This is important because at early timesteps
the model's predictions are already more informative than the noisy state. The
proposed implementation already does this naturally since we compute gradients
w.r.t. `pred_E_logits`.

**C. Gradient clamping:**
To prevent extreme rate modulation, clamp the log-ratio:
```python
log_ratio = torch.clamp(log_ratio, min=-5.0, max=5.0)
```
This bounds the multiplicative factor to [exp(-5), exp(5)] ≈ [0.007, 148.4],
preventing any single transition from being amplified or suppressed by more
than ~150x.

### Step 3: Systematic Evaluation

See [Section 7](#7-evaluation-protocol) for the full evaluation protocol.

### Step 4: Comparison

Benchmark TAG guidance against all existing methods with matched compute budgets:
- **Plain sampling** (baseline)
- **Best-of-N** (N=256, current production method)
- **Soft HDC guidance** (existing, "blend" mode)
- **TAG guidance** (proposed)
- **TAG + best-of-N** (TAG with N=16, for comparable total compute)

---

## 7. Evaluation Protocol

### 7.1 Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| HDC cosine distance | 1 - cos_sim(HDC(generated), target) | Lower is better |
| Exact SMILES match | Canonical SMILES equality | Higher is better |
| Valid molecule rate | Fraction of chemically valid outputs | Higher is better |
| Edge accuracy | Per-edge bond type classification accuracy | Higher is better |
| Tanimoto similarity | Morgan fingerprint similarity to ground truth | Higher is better |

### 7.2 Hyperparameter Grid

```
gamma_max:  [0.5, 1.0, 2.0]
tau:        [0.25, 0.5, 1.0]
eta:        [0, 0.5, 1.0]
t_min:      [0.2, 0.3, 0.5]
```

Initial sweep: fix eta=1.0, t_min=0.3, sweep gamma_max × tau (9 configs).
Then refine eta and t_min around best config.

### 7.3 Test Set

Use ~100 molecules spanning a range of sizes (5-30 atoms) from the validation
set. Generate 10 samples per molecule per configuration. Report mean ± std of
all metrics.

### 7.4 Ablations

1. **Guidance only (no conditioning):** Set `hdc_vectors = zeros` to disable
   MLP/cross-attention conditioning. Measures whether guidance alone can steer
   generation.
2. **Depth comparison:** guidance_depth in {1, 2, 3} to validate the
   reduced-depth recommendation.
3. **Schedule comparison:** threshold_ramp vs constant vs sigmoid.
4. **With/without gradient clamping.**

---

## 8. Appendix: Full Mathematical Derivation

### A.1 From the Paper's Framework to Our Setting

**Starting point**: The paper's Equation 2 for predictor guidance:

```
R_t(x, x̃ | y) = [p(y | x̃, t) / p(y | x, t)]^γ · R_t(x, x̃)     (Eq. 2)
```

**Our variables:**
- `x` = edge matrix of the molecular graph, represented as one-hot tensors
  of shape (n, n, S) where S=5 (edge classes)
- `y` = `hdc_target`, the target HDC vector (order_N component, shape (D,))
- `R_t(x, x̃)` = unconditional rate matrix from the FlowEdgeDecoder's
  `RateMatrixDesigner`

### A.2 Defining the Predictor Likelihood

We define the predictor as a Boltzmann distribution:

```
p(y | x, t) ∝ exp(E(x, y) / τ)
```

where the energy function is the cosine similarity:

```
E(x, y) = cos_sim(HDC_encode(x), y) = <HDC(x), y> / (||HDC(x)|| · ||y||)
```

This is not a normalized probability distribution, but the Discrete Guidance
framework only requires the **ratio** `p(y|x̃,t) / p(y|x,t)`, in which the
normalization constant cancels:

```
p(y | x̃, t) / p(y | x, t) = exp([E(x̃, y) - E(x, y)] / τ)
                             = exp([cos_sim(HDC(x̃), y) - cos_sim(HDC(x), y)] / τ)
```

### A.3 Validity of the Conditional Independence Assumption

The paper requires `x_t ⊥ y | x_{t+dt}` (Appendix C.4, Eq. 17-18), meaning
the noising process is independent of the guide property. In our case:
- The noising process corrupts edges via the masking flow:
  `p(E_t | E_1) = t · δ(E_t, E_1) + (1-t) · p_0(E_t)`
- The target HDC vector `y` is a deterministic function of the clean graph E_1
- The noising at time t does not depend on y

Therefore `p(x_t | x_{t+dt}, y) = p(x_t | x_{t+dt})`, and the conditional
independence assumption holds. This allows simplification of the likelihood ratio
from `p(y | x_{t+dt}, x_t)` to `p(y | x_{t+dt})` (the paper's Eq. 19).

### A.4 Applying the TAG Approximation

Define `f(x) = log p(y|x,t) = cos_sim(HDC(x), y) / τ + const`.

The TAG approximation (paper Eq. 4):

```
log [p(y|x̃,t) / p(y|x,t)] = f(x̃) - f(x) ≈ (x̃ - x)^T · ∇_x f(x)
```

**Why this works for one-hot inputs:**

The edge state x is represented as a one-hot tensor of shape (n, n, S). The
neural network (soft HDC encoder) treats this as a real-valued input, making it
a continuous function on R^{n×n×S}. Although we only evaluate at discrete
one-hot vectors, the function is differentiable everywhere.

For a single edge (i,j) transitioning from class e to class e':
```
x̃ - x = one_hot(e') - one_hot(e)    [at position (i,j)]
```

Therefore:
```
(x̃ - x)^T · ∇_x f(x) = ∇f[i,j,e'] - ∇f[i,j,e]
```

### A.5 The Gradient Computation

In practice, we compute `∇_x f(x)` as:

```python
# f(x) = cos_sim(soft_HDC_encode(softmax(x/τ)), target) / τ
# We differentiate the loss = 1 - cos_sim w.r.t. logits

logits_g = pred_E_logits.detach().clone().requires_grad_(True)
soft_probs = F.softmax(logits_g / tau, dim=-1)           # (bs, n, n, 5)
soft_A = soft_probs[:, :, :, 1:].sum(dim=-1)             # (bs, n, n)
soft_A = symmetrize(soft_A)                                # enforce undirected
graph_emb = soft_hdc_encode(soft_A, node_hvs, depth=1)   # (bs, D)
loss = (1 - F.cosine_similarity(graph_emb, target, dim=-1)).sum()
grad = torch.autograd.grad(loss, logits_g)[0]             # (bs, n, n, 5)
```

Since `loss = 1 - cos_sim`, the gradient ∇loss points UPHILL (toward worse
similarity). For the log-ratio, we need the direction that IMPROVES similarity:

```python
# Transitions with log_ratio > 0 improve similarity (should be amplified)
grad_at_current = grad.gather(-1, E_t_label.unsqueeze(-1))  # (bs, n, n, 1)
log_ratio = grad_at_current - grad                           # (bs, n, n, 5)
```

This is correct because:
- `grad[i,j,e']` = rate of loss increase when edge (i,j) moves toward class e'
- `grad_at_current` = rate of loss increase at the current class e
- `grad_at_current - grad[i,j,e']` > 0 when class e' has lower loss than e
  (i.e., transitioning to e' would improve similarity)

### A.6 The Guided Rates

Substituting into Eq. 2:

```
R_t^guided[i,j,e'] = exp(γ_t · (grad[i,j,e_current] - grad[i,j,e'])) · R_t[i,j,e']
```

where `γ_t` follows the threshold + ramp schedule:

```
γ_t = 0                                      if t < t_min
γ_t = γ_max · (t - t_min) / (1 - t_min)     if t ≥ t_min
```

### A.7 Smoothness and Approximation Quality

The soft HDC encode is a composition of smooth operations:
1. softmax: C∞
2. linear projection (bmm with fixed node_hvs): C∞
3. FFT circular convolution: C∞ (FFT is a unitary linear transform)
4. L2 normalization: C∞ (away from zero, guaranteed by the eps term)
5. Cosine similarity: C∞ (for non-zero vectors)

The Hessian norm (curvature) of the composed function grows approximately as
O(2^depth) due to the nested multiplicative structure of circular convolution.
For depth=1, the second-order Taylor remainder is small. For depth=3, it can
be significant, justifying the recommendation to use reduced depth for guidance.

### A.8 Rate Matrix Properties

**Non-negativity**: For off-diagonal rates (x̃ ≠ x):
```
R_t^guided = exp(·) · R_t ≥ 0
```
since exp(·) > 0 always and R_t ≥ 0 for off-diagonal entries.

**Conservation**: The diagonal rate is recomputed:
```
R_t^guided(x, x) = -Σ_{x' ≠ x} R_t^guided(x, x')
```
ensuring `Σ_{x'} R_t^guided(x, x') = 0`, which is required for a valid CTMC.

**Tractability**: In our setting, D = n(n-1)/2 edge positions (undirected) and
S = 5 edge classes. Only D·(S-1) = 2n(n-1) off-diagonal rates are non-zero per
state. For n=12 atoms, this is 264 rates — trivially manageable.
