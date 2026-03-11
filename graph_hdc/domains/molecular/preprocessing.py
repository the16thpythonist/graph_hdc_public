"""
Molecular-specific constants and preprocessing for the FlowEdgeDecoder.

This module contains ZINC/molecular constants (atom types, bond types, feature bins)
and preprocessing functions that were previously hardcoded in the FlowEdgeDecoder.
These are domain-specific and should only be imported by molecular domain code.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from rdkit import Chem
from torch import Tensor
from torch_geometric.data import Data
from tqdm.auto import tqdm

from graph_hdc.models.flow_edge_decoder import (
    FlowEdgeDecoderConfig,
    extend_feature_bins,
    raw_features_to_onehot,
)


# =============================================================================
# ZINC Atom Types
# =============================================================================

ZINC_ATOM_TYPES = ["Br", "C", "Cl", "F", "I", "N", "O", "P", "S"]
ZINC_ATOM_TO_IDX = {atom: idx for idx, atom in enumerate(ZINC_ATOM_TYPES)}
ZINC_IDX_TO_ATOM = {idx: atom for atom, idx in ZINC_ATOM_TO_IDX.items()}


# =============================================================================
# ZINC Feature Dimensions
# =============================================================================

NUM_ATOM_CLASSES = 9
NUM_DEGREE_CLASSES = 6
NUM_CHARGE_CLASSES = 3
NUM_HS_CLASSES = 4
NUM_RING_CLASSES = 2

ZINC_FEATURE_BINS = [
    NUM_ATOM_CLASSES,
    NUM_DEGREE_CLASSES,
    NUM_CHARGE_CLASSES,
    NUM_HS_CLASSES,
    NUM_RING_CLASSES,
]

ZINC_FEATURE_DIM = sum(ZINC_FEATURE_BINS)  # 24

# Legacy aliases for backward compatibility of imports
NODE_FEATURE_BINS = ZINC_FEATURE_BINS
NODE_FEATURE_DIM = ZINC_FEATURE_DIM


# =============================================================================
# Bond Types
# =============================================================================

BOND_TYPES = ["no_edge", "single", "double", "triple", "aromatic"]
BOND_TYPE_TO_IDX = {
    None: 0,  # No edge
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}
NUM_BOND_CLASSES = 5

# Legacy alias
NUM_EDGE_CLASSES = NUM_BOND_CLASSES


# =============================================================================
# ZINC Preset Configuration
# =============================================================================

ZINC_EDGE_DECODER_CONFIG = FlowEdgeDecoderConfig(
    feature_bins=ZINC_FEATURE_BINS,
    num_edge_classes=NUM_BOND_CLASSES,
    hdc_dim=512,
    n_layers=8,
    hidden_dim=384,
    max_nodes=50,
)


# =============================================================================
# Preprocessing Functions
# =============================================================================


def get_bond_type_idx(bond) -> int:
    """Convert RDKit bond to 5-class index."""
    if bond is None:
        return 0  # No edge
    return BOND_TYPE_TO_IDX.get(bond.GetBondType(), 0)


def preprocess_for_flow_edge_decoder(
    data: Data,
    hypernet,
    device: torch.device = None,
) -> Optional[Data]:
    """
    Preprocess a PyG Data object for FlowEdgeDecoder training (molecular graphs).

    Adds/replaces:
    - x: One-hot encoding (24-dim base, or extended with RW bins when enabled)
    - edge_attr: 5-class bond type one-hot encoding
    - hdc_vector: Pre-computed HDC embedding
    - original_x: Raw feature indices for HDC-guided sampling

    Args:
        data: Original PyG Data object with molecular features
              data.x should have shape (n, 5) with columns:
              [atom_type, degree, charge, num_hs, is_in_ring]
        hypernet: HyperNet encoder for computing HDC embeddings
        device: Device for HDC computation

    Returns:
        Preprocessed Data object, or None if molecule has no edges
    """
    device = device or torch.device("cpu")

    # Skip molecules with no edges (single atoms)
    if data.edge_index.numel() == 0:
        return None

    # 1. Build node features — base 5-dim features, optionally extended
    #    with binned RW return probabilities when the encoder uses them.
    rw_config = getattr(hypernet, "rw_config", None)
    feature_bins = extend_feature_bins(ZINC_FEATURE_BINS, rw_config)

    raw_feats = data.x.clone()  # (n, 5) base features
    if rw_config is not None and rw_config.enabled:
        from graph_hdc.utils.rw_features import (
            bin_rw_probabilities,
            compute_rw_return_probabilities,
        )
        rw_probs = compute_rw_return_probabilities(
            data.edge_index, data.x.size(0), rw_config.k_values,
        )
        rw_binned = bin_rw_probabilities(rw_probs, rw_config.num_bins, bin_boundaries=rw_config.bin_boundaries, k_values=rw_config.k_values, clip_range=rw_config.clip_range)
        raw_feats = torch.cat([raw_feats, rw_binned], dim=-1)

    x_onehot = raw_features_to_onehot(raw_feats, feature_bins=feature_bins)

    # 2. Create 5-class edge attributes from SMILES
    mol = Chem.MolFromSmiles(data.smiles)
    if mol is None:
        return None

    edge_index = data.edge_index

    # Build adjacency with bond types
    edge_attr_list = []
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        bond = mol.GetBondBetweenAtoms(src, dst)
        bond_type = get_bond_type_idx(bond)
        edge_attr_list.append(bond_type)

    edge_attr = torch.tensor(edge_attr_list, dtype=torch.long)
    edge_attr_onehot = F.one_hot(edge_attr, num_classes=NUM_BOND_CLASSES).float()

    # 3. Compute HDC embedding: concatenate order-0 and order-N
    data_for_hdc = data.clone().to(device)
    # Add batch attribute for single graph (required by HyperNet)
    if data_for_hdc.batch is None:
        data_for_hdc.batch = torch.zeros(data_for_hdc.x.size(0), dtype=torch.long, device=device)

    # Augment with RW features if the encoder was built with them
    if hasattr(hypernet, "rw_config") and hypernet.rw_config.enabled:
        from graph_hdc.utils.rw_features import augment_data_with_rw

        data_for_hdc = augment_data_with_rw(
            data_for_hdc,
            k_values=hypernet.rw_config.k_values,
            num_bins=hypernet.rw_config.num_bins,
            bin_boundaries=hypernet.rw_config.bin_boundaries,
        )

    with torch.no_grad():
        # forward() handles encode_properties internally and returns:
        # - node_terms: order-0 (bundled node HVs, RRWP-enriched for RRWPHyperNet)
        # - graph_embedding: order-N (message passing result)
        hdc_out = hypernet.forward(data_for_hdc)
        order_zero = hdc_out["node_terms"]
        order_n = hdc_out["graph_embedding"]

        # Concatenate [order_0 | order_N]
        hdc_concat = torch.cat([order_zero, order_n], dim=-1).cpu().float()
        # Convert from HRRTensor subclass to regular tensor
        hdc_vector = torch.Tensor(hdc_concat.tolist())
        if hdc_vector.dim() == 1:
            hdc_vector = hdc_vector.unsqueeze(0)  # Ensure (1, 2*hdc_dim) for proper batching

    # 4. Create new Data object
    new_data = Data(
        x=x_onehot,  # one-hot: base 24-dim, or extended when RW enabled
        edge_index=edge_index.clone(),
        edge_attr=edge_attr_onehot,
        hdc_vector=hdc_vector,
        smiles=data.smiles,
        # Keep original raw features for HDC-guided sampling
        original_x=data.x.clone(),
    )

    # Copy other attributes
    if hasattr(data, "logp"):
        new_data.logp = data.logp.clone()
    if hasattr(data, "qed"):
        new_data.qed = data.qed.clone()

    return new_data


def preprocess_dataset(
    dataset,
    hypernet,
    device: torch.device = None,
    show_progress: bool = True,
) -> List[Data]:
    """
    Preprocess entire dataset for FlowEdgeDecoder (molecular graphs).

    Args:
        dataset: PyG dataset or list of Data objects
        hypernet: HyperNet encoder
        device: Device for HDC computation
        show_progress: Whether to show progress bar

    Returns:
        List of preprocessed Data objects
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hypernet = hypernet.to(device)
    hypernet.eval()

    processed = []
    iterator = dataset
    if show_progress:
        iterator = tqdm(iterator, desc="Preprocessing molecules")

    for data in iterator:
        new_data = preprocess_for_flow_edge_decoder(data, hypernet, device)
        if new_data is not None:
            processed.append(new_data)

    return processed
