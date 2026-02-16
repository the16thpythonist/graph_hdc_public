# Global Structure Encoding for HyperNet

## Overview

This document describes two training-free enhancements to the HyperNet HDC encoder that capture **global graph structure** beyond local message passing. These address the limitation that the current 3-4 hop message passing only captures local neighborhood information.

**Target file**: `graph_hdc/hypernet/encoder.py` (HyperNet class)

**Key insight**: Local message passing is bounded by 1-WL expressiveness. Adding global features (random walks, spectral, centrality) can exceed this bound while remaining training-free.

---

## Proposal 1: Random Walk Positional Encoding

### Motivation

The current HyperNet encodes node features and local neighborhood structure, but two nodes in different global positions (e.g., ring center vs chain end) may have identical local neighborhoods. Random walk return probabilities encode where a node sits in the global graph topology.

### Core Idea

For each node, compute the probability that a random walker starting at that node returns to it after k steps. This creates a "structural fingerprint" that differs based on:
- Ring membership (walkers return at cycle length)
- Chain position (end nodes vs middle nodes)
- Bridge nodes (connecting different substructures)
- Global connectivity patterns

### Mathematical Definition

Given:
- Adjacency matrix `A` of shape `[n, n]`
- Degree matrix `D = diag(sum(A, axis=1))`
- Transition matrix `P = D^(-1) @ A` (row-stochastic)

The k-step return probability for node i is:
```
RW_k(i) = [P^k]_{ii}   # Diagonal element of P^k
```

The random walk positional encoding for node i is the vector:
```
rw_pe(i) = [RW_1(i), RW_2(i), ..., RW_K(i)]
```

Where K is typically 8-16 steps.

### What Each Walk Length Captures

| Walk Length k | Structural Information |
|---------------|------------------------|
| 2 | Local density, degree-related |
| 3 | Triangle (3-cycle) participation |
| 4 | 4-cycle membership, local clustering |
| 5 | 5-membered rings (furan, cyclopentane) |
| 6 | 6-membered rings (benzene, cyclohexane) |
| 7-8 | Larger rings, fused ring systems |
| 10+ | Global connectivity, component structure |

### Implementation

#### Step 1: Compute Random Walk PE

```python
import numpy as np
import torch

def compute_random_walk_pe(adj_matrix: np.ndarray, k_steps: int = 8) -> np.ndarray:
    """
    Compute random walk return probabilities for all nodes.

    Parameters
    ----------
    adj_matrix : np.ndarray
        Adjacency matrix of shape [n, n]
    k_steps : int
        Number of walk lengths to compute (default: 8)

    Returns
    -------
    np.ndarray
        Random walk PE of shape [n, k_steps] where entry [i, k] is
        the probability of returning to node i after k+1 steps.
    """
    n = adj_matrix.shape[0]

    # Compute degree and handle isolated nodes
    degrees = adj_matrix.sum(axis=1)
    degrees[degrees == 0] = 1  # Avoid division by zero

    # Transition matrix: P[i,j] = prob of going from i to j
    D_inv = np.diag(1.0 / degrees)
    P = D_inv @ adj_matrix

    # Compute powers of P and extract diagonals
    rw_pe = np.zeros((n, k_steps))
    Pk = np.eye(n)

    for k in range(k_steps):
        Pk = Pk @ P
        rw_pe[:, k] = np.diag(Pk)

    return rw_pe
```

#### Step 2: Encode RW PE into Hypervectors

```python
import torchhd

class EnhancedHyperNet(HyperNet):
    def __init__(self, config, rw_steps: int = 8):
        super().__init__(config)
        self.rw_steps = rw_steps

        # Codebook for walk lengths (one HV per step)
        self.rw_step_codebook = torchhd.random(
            rw_steps, self.hv_dim,
            vsa=self.vsa.value,
            dtype=self._dtype
        )

        # Codebook for discretized probabilities (100 bins)
        self.rw_prob_codebook = torchhd.random(
            100, self.hv_dim,
            vsa=self.vsa.value,
            dtype=self._dtype
        )

    def encode_rw_pe(self, node_hv: torch.Tensor, rw_probs: np.ndarray) -> torch.Tensor:
        """
        Bind random walk information into node hypervector.

        Parameters
        ----------
        node_hv : torch.Tensor
            Node hypervector of shape [hv_dim]
        rw_probs : np.ndarray
            Return probabilities of shape [k_steps]

        Returns
        -------
        torch.Tensor
            Enhanced node hypervector with RW PE bound in
        """
        result = node_hv.clone()

        for k, prob in enumerate(rw_probs):
            # Discretize probability to 0-99
            prob_idx = min(int(prob * 99), 99)

            # Bind step marker with probability marker
            rw_hv = torchhd.bind(
                self.rw_step_codebook[k],
                self.rw_prob_codebook[prob_idx]
            )

            # Bind into node HV
            result = torchhd.bind(result, rw_hv)

        return result
```

#### Step 3: Integration into Forward Pass

```python
from torch_geometric.utils import to_dense_adj

def forward(self, data, *, bidirectional=False, normalize=None):
    """Enhanced forward with random walk PE."""

    # Compute adjacency matrix
    adj = to_dense_adj(
        data.edge_index,
        max_num_nodes=data.num_nodes
    )[0].cpu().numpy()

    # Compute random walk PE for all nodes
    rw_pe = compute_random_walk_pe(adj, k_steps=self.rw_steps)

    # Encode node properties (existing)
    data = self.encode_properties(data)

    # Enhance node HVs with random walk PE
    enhanced_node_hvs = []
    for i in range(data.num_nodes):
        enhanced_hv = self.encode_rw_pe(data.node_hv[i], rw_pe[i])
        enhanced_node_hvs.append(enhanced_hv)
    data.node_hv = torch.stack(enhanced_node_hvs)

    # Continue with existing message passing...
    # (rest of original forward method)
```

### Complexity Analysis

- **Time**: O(k × n²) for matrix powers, but can use sparse matrix multiplication for O(k × m) where m = edges
- **Space**: O(n × k) for storing RW PE
- **For molecules**: n < 100, k = 8 → negligible overhead

### Alternative: Sparse Implementation

For larger graphs, avoid dense matrix powers:

```python
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import matrix_power

def compute_random_walk_pe_sparse(edge_index, num_nodes, k_steps=8):
    """Sparse version for larger graphs."""
    # Build sparse adjacency
    row, col = edge_index.numpy()
    data = np.ones(len(row))
    A = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    # Normalize to transition matrix
    degrees = np.array(A.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1
    D_inv = csr_matrix(np.diag(1.0 / degrees))
    P = D_inv @ A

    # Compute diagonals of P^k
    rw_pe = np.zeros((num_nodes, k_steps))
    Pk = csr_matrix(np.eye(num_nodes))

    for k in range(k_steps):
        Pk = Pk @ P
        rw_pe[:, k] = Pk.diagonal()

    return rw_pe
```

---

## Proposal 2: Centrality-Weighted Bundling

### Motivation

The current graph readout is:
```python
graph_hv = scatter_hd(node_hvs, batch_index, op="bundle")  # = sum
```

This treats all nodes equally, but structurally important nodes (ring centers, functional group cores, bridge atoms) should contribute more to the graph representation.

### Core Idea

Replace flat bundling with weighted bundling:
```python
graph_hv = sum(centrality[i] * node_hv[i] for i in nodes)
```

Where `centrality[i]` measures the structural importance of node i.

### Centrality Measures

#### Option 1: PageRank (Recommended)

PageRank is a damped eigenvector centrality that:
- Gives high scores to nodes connected to high-score nodes
- Handles disconnected components gracefully
- Converges quickly (~20 iterations)

```python
def compute_pagerank(adj_matrix: np.ndarray, damping: float = 0.85,
                     iterations: int = 20) -> np.ndarray:
    """
    Compute PageRank centrality scores.

    Parameters
    ----------
    adj_matrix : np.ndarray
        Adjacency matrix of shape [n, n]
    damping : float
        Damping factor (default: 0.85)
    iterations : int
        Number of power iterations (default: 20)

    Returns
    -------
    np.ndarray
        Centrality scores of shape [n], summing to 1
    """
    n = adj_matrix.shape[0]

    # Compute out-degree
    out_degree = adj_matrix.sum(axis=1)
    out_degree[out_degree == 0] = 1  # Avoid division by zero

    # Transition matrix (column-stochastic for PageRank)
    P = (adj_matrix / out_degree[:, None]).T

    # Power iteration
    pr = np.ones(n) / n
    for _ in range(iterations):
        pr = damping * (P @ pr) + (1 - damping) / n

    return pr
```

#### Option 2: Eigenvector Centrality

Simpler than PageRank, but may not handle all graph structures well:

```python
def compute_eigenvector_centrality(adj_matrix: np.ndarray,
                                   iterations: int = 20) -> np.ndarray:
    """
    Compute eigenvector centrality via power iteration.

    The principal eigenvector of the adjacency matrix.
    """
    n = adj_matrix.shape[0]
    x = np.ones(n) / np.sqrt(n)

    for _ in range(iterations):
        x = adj_matrix @ x
        norm = np.linalg.norm(x)
        if norm > 0:
            x = x / norm

    # Normalize to sum to 1
    x = np.abs(x)  # Ensure positive
    return x / x.sum()
```

#### Option 3: Degree Centrality (Simplest)

Already partially captured by node features, but can still be useful:

```python
def compute_degree_centrality(adj_matrix: np.ndarray) -> np.ndarray:
    """Simple degree-based centrality."""
    degrees = adj_matrix.sum(axis=1)
    return degrees / degrees.sum()
```

### Implementation

```python
class EnhancedHyperNet(HyperNet):
    def __init__(self, config, centrality_method: str = 'pagerank'):
        super().__init__(config)
        self.centrality_method = centrality_method

    def compute_centrality(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Compute centrality scores based on configured method."""
        if self.centrality_method == 'pagerank':
            return compute_pagerank(adj_matrix)
        elif self.centrality_method == 'eigenvector':
            return compute_eigenvector_centrality(adj_matrix)
        elif self.centrality_method == 'degree':
            return compute_degree_centrality(adj_matrix)
        else:
            # Uniform (original behavior)
            n = adj_matrix.shape[0]
            return np.ones(n) / n

    def weighted_bundle(self, node_hvs: torch.Tensor,
                        weights: np.ndarray) -> torch.Tensor:
        """
        Bundle node HVs with centrality weighting.

        Parameters
        ----------
        node_hvs : torch.Tensor
            Node hypervectors of shape [n, hv_dim]
        weights : np.ndarray
            Centrality weights of shape [n], should sum to 1

        Returns
        -------
        torch.Tensor
            Weighted graph hypervector of shape [hv_dim]
        """
        weights_tensor = torch.from_numpy(weights).to(
            device=node_hvs.device,
            dtype=node_hvs.dtype
        ).unsqueeze(-1)

        # Scale by number of nodes to maintain magnitude
        n = node_hvs.shape[0]
        weighted_sum = (node_hvs * weights_tensor * n).sum(dim=0)

        return weighted_sum
```

### Integration into Forward Pass

Replace the final bundling step:

```python
def forward(self, data, **kwargs):
    # ... existing message passing code ...

    # Instead of:
    # graph_embedding = scatter_hd(node_hv, data.batch, op="bundle")

    # Compute centrality
    adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
    centrality = self.compute_centrality(adj.cpu().numpy())

    # Weighted bundling
    graph_embedding = self.weighted_bundle(node_hv, centrality)

    return {"graph_embedding": graph_embedding, ...}
```

### Handling Batched Graphs

For batched processing, compute centrality per graph:

```python
def forward_batched(self, batch):
    """Handle PyG Batch with multiple graphs."""
    graph_embeddings = []

    # Process each graph in the batch
    for i in range(batch.num_graphs):
        # Extract subgraph
        mask = batch.batch == i
        node_hvs_i = node_hv[mask]

        # Get adjacency for this graph
        edge_mask = mask[batch.edge_index[0]] & mask[batch.edge_index[1]]
        edge_index_i = batch.edge_index[:, edge_mask]
        # Reindex to 0-based
        edge_index_i = edge_index_i - edge_index_i.min()

        adj_i = to_dense_adj(edge_index_i, max_num_nodes=mask.sum())[0]
        centrality_i = self.compute_centrality(adj_i.cpu().numpy())

        graph_hv_i = self.weighted_bundle(node_hvs_i, centrality_i)
        graph_embeddings.append(graph_hv_i)

    return torch.stack(graph_embeddings)
```

---

## Combined Implementation

Here's a complete enhanced HyperNet combining both proposals:

```python
import numpy as np
import torch
import torchhd
from torch_geometric.utils import to_dense_adj

class GlobalStructureHyperNet(HyperNet):
    """
    HyperNet enhanced with global structure encoding.

    Additions over base HyperNet:
    1. Random Walk Positional Encoding - encodes multi-scale structure
    2. Centrality-Weighted Bundling - emphasizes important nodes

    Both are training-free and maintain HDC principles.
    """

    def __init__(self, config,
                 rw_steps: int = 8,
                 centrality_method: str = 'pagerank',
                 use_rw_pe: bool = True,
                 use_centrality_weighting: bool = True):
        super().__init__(config)

        self.rw_steps = rw_steps
        self.centrality_method = centrality_method
        self.use_rw_pe = use_rw_pe
        self.use_centrality_weighting = use_centrality_weighting

        if use_rw_pe:
            # Random walk codebooks
            self.rw_step_codebook = torchhd.random(
                rw_steps, self.hv_dim,
                vsa=self.vsa.value,
                dtype=self._dtype
            )
            self.rw_prob_codebook = torchhd.random(
                100, self.hv_dim,
                vsa=self.vsa.value,
                dtype=self._dtype
            )

    def compute_random_walk_pe(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Compute random walk return probabilities."""
        n = adj_matrix.shape[0]
        degrees = adj_matrix.sum(axis=1)
        degrees[degrees == 0] = 1

        D_inv = np.diag(1.0 / degrees)
        P = D_inv @ adj_matrix

        rw_pe = np.zeros((n, self.rw_steps))
        Pk = np.eye(n)

        for k in range(self.rw_steps):
            Pk = Pk @ P
            rw_pe[:, k] = np.diag(Pk)

        return rw_pe

    def compute_centrality(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Compute node centrality scores."""
        n = adj_matrix.shape[0]

        if self.centrality_method == 'pagerank':
            out_degree = adj_matrix.sum(axis=1)
            out_degree[out_degree == 0] = 1
            P = (adj_matrix / out_degree[:, None]).T

            pr = np.ones(n) / n
            for _ in range(20):
                pr = 0.85 * (P @ pr) + 0.15 / n
            return pr

        elif self.centrality_method == 'eigenvector':
            x = np.ones(n) / np.sqrt(n)
            for _ in range(20):
                x = adj_matrix @ x
                norm = np.linalg.norm(x)
                if norm > 0:
                    x = x / norm
            x = np.abs(x)
            return x / x.sum()

        else:  # uniform
            return np.ones(n) / n

    def encode_rw_pe(self, node_hvs: torch.Tensor,
                     rw_pe: np.ndarray) -> torch.Tensor:
        """Bind random walk PE into node hypervectors."""
        n = node_hvs.shape[0]
        enhanced = node_hvs.clone()

        for i in range(n):
            for k in range(self.rw_steps):
                prob_idx = min(int(rw_pe[i, k] * 99), 99)
                rw_hv = torchhd.bind(
                    self.rw_step_codebook[k],
                    self.rw_prob_codebook[prob_idx]
                )
                enhanced[i] = torchhd.bind(enhanced[i], rw_hv)

        return enhanced

    def weighted_bundle(self, node_hvs: torch.Tensor,
                        weights: np.ndarray) -> torch.Tensor:
        """Centrality-weighted bundling."""
        weights_tensor = torch.from_numpy(weights).to(
            device=node_hvs.device,
            dtype=node_hvs.dtype
        ).unsqueeze(-1)

        n = node_hvs.shape[0]
        return (node_hvs * weights_tensor * n).sum(dim=0)

    def forward(self, data, *, bidirectional=False, normalize=None):
        """
        Enhanced forward pass with global structure encoding.
        """
        if normalize is None:
            normalize = self.normalize

        # Get adjacency matrix
        adj = to_dense_adj(
            data.edge_index,
            max_num_nodes=data.num_nodes
        )[0].cpu().numpy()

        # Encode node properties (existing)
        data = self.encode_properties(data)

        # Enhancement 1: Random Walk PE
        if self.use_rw_pe:
            rw_pe = self.compute_random_walk_pe(adj)
            data.node_hv = self.encode_rw_pe(data.node_hv, rw_pe)

        # Message passing (existing code)
        edge_index = data.edge_index
        if bidirectional:
            edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        srcs, dsts = edge_index

        if data.node_hv.device != edge_index.device:
            data.node_hv = data.node_hv.to(edge_index.device)

        node_dim = data.x.size(0)
        node_hv_stack = data.node_hv.new_zeros(
            size=(self.depth + 1, node_dim, self.hv_dim)
        )
        node_hv_stack[0] = data.node_hv

        edge_terms = None
        for layer_index in range(self.depth):
            messages = node_hv_stack[layer_index][dsts]
            aggregated = scatter_hd(messages, srcs, dim_size=node_dim, op="bundle")
            prev_hv = node_hv_stack[layer_index].clone()
            hr = torchhd.bind(prev_hv, aggregated)

            if layer_index == 0:
                edge_terms = hr.clone()

            if normalize:
                hr_norm = hr.norm(dim=-1, keepdim=True)
                node_hv_stack[layer_index + 1] = hr / (hr_norm + 1e-8)
            else:
                node_hv_stack[layer_index + 1] = hr

        node_hv_stack = node_hv_stack.transpose(0, 1)
        node_hv = torchhd.multibundle(node_hv_stack)

        # Enhancement 2: Centrality-Weighted Bundling
        if self.use_centrality_weighting:
            centrality = self.compute_centrality(adj)
            graph_embedding = self.weighted_bundle(node_hv, centrality)
        else:
            graph_embedding = scatter_hd(node_hv, data.batch, op="bundle")

        edge_terms = scatter_hd(edge_terms, data.batch, op="bundle")

        return {
            "graph_embedding": graph_embedding,
            "edge_terms": edge_terms,
        }
```

---

## Testing

### Unit Test for Random Walk PE

```python
def test_random_walk_pe():
    """Test that RW PE distinguishes ring from chain."""
    import numpy as np

    # 6-node ring
    ring_adj = np.array([
        [0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0],
    ])

    # 6-node chain
    chain_adj = np.array([
        [0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0],
    ])

    ring_pe = compute_random_walk_pe(ring_adj, k_steps=8)
    chain_pe = compute_random_walk_pe(chain_adj, k_steps=8)

    # Ring nodes should have higher return prob at k=6 (cycle length)
    assert ring_pe[0, 5] > chain_pe[0, 5], "Ring should have higher 6-step return"

    # All ring nodes should have same PE (symmetric)
    assert np.allclose(ring_pe[0], ring_pe[3]), "Ring nodes should be equivalent"

    # Chain ends should differ from chain middle
    assert not np.allclose(chain_pe[0], chain_pe[2]), "Chain ends differ from middle"

    print("Random Walk PE tests passed!")
```

### Unit Test for Centrality

```python
def test_centrality_weighting():
    """Test that centrality correctly identifies important nodes."""
    import numpy as np

    # Star graph: center connected to 4 leaves
    star_adj = np.array([
        [0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ])

    pr = compute_pagerank(star_adj)

    # Center (node 0) should have highest centrality
    assert pr[0] == pr.max(), "Star center should have highest PageRank"

    # All leaves should have equal centrality
    assert np.allclose(pr[1], pr[2]), "Leaves should have equal PageRank"

    print("Centrality weighting tests passed!")
```

---

## Expected Benefits

1. **Better discrimination**: Molecules with same local structure but different global topology get different embeddings

2. **Improved reconstruction**: Global features help the decoder distinguish similar local patterns

3. **Training-free**: No additional learnable parameters, maintains HDC philosophy

4. **Modest overhead**: O(k × n²) for RW PE, O(n²) for centrality - negligible for molecular graphs

5. **Exceeds 1-WL**: The combination of local message passing + global features can distinguish graphs that 1-WL cannot

---

## References

- GPS (General Powerful Scalable Graph Transformer) - Rampasek et al. 2022
- Graphormer - Ying et al. 2021
- Random Walk Positional Encodings - Dwivedi et al. 2022
- PageRank - Page et al. 1999
- Weisfeiler-Lehman expressiveness bounds - Xu et al. 2019, Morris et al. 2019
