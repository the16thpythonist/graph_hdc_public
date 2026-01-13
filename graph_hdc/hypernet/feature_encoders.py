"""
Feature encoders for hyperdimensional computing.

Each encoder maps discrete feature values to hypervectors using codebooks
generated via TorchHD primitives (random, level, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torchhd

from graph_hdc.utils.helpers import TupleIndexer


class AbstractFeatureEncoder(ABC):
    """Base class for all feature encoders."""

    def __init__(
        self,
        dim: int,
        vsa: str,
        device: torch.device = torch.device("cpu"),
        seed: int | None = None,
        num_categories: int | None = None,
        idx_offset: int = 0,
        dtype: str = "float32",
    ):
        self.dim = dim
        self.vsa = vsa
        self.device = device
        self.num_categories = num_categories
        self.idx_offset = idx_offset
        self.dtype = dtype
        if seed is not None:
            torch.manual_seed(seed)
        self.codebook = self.generate_codebook()

    @abstractmethod
    def generate_codebook(self) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def normalize(value: Any) -> Any:
        return value

    @abstractmethod
    def encode(self, value: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, hv: torch.Tensor) -> Any:
        pass

    def get_codebook(self) -> torch.Tensor:
        return self.codebook

    @abstractmethod
    def decode_index(self, idx: int) -> torch.Tensor:
        pass


class CombinatoricIntegerEncoder(AbstractFeatureEncoder):
    """Encodes multi-dimensional feature tuples via an indexer."""

    def __init__(
        self,
        dim: int,
        num_categories: int,
        indexer: TupleIndexer = None,
        vsa: str = "MAP",
        device: torch.device = torch.device("cpu"),
        seed: int | None = None,
        idx_offset: int = 0,
        dtype: str = "float32",
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            vsa=vsa,
            device=device,
            seed=seed,
            num_categories=num_categories,
            idx_offset=idx_offset,
            dtype=dtype,
        )
        self.indexer = indexer if indexer else TupleIndexer([28, 6])

    def generate_codebook(self) -> torch.Tensor:
        dtype = torch.float64 if self.dtype == "float64" else torch.float32
        cb = torchhd.random(self.num_categories, self.dim, vsa=self.vsa, device="cpu", dtype=dtype)
        return cb.to(self.device)

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode feature tuples into hypervectors."""
        if data.shape[-1] == 1:
            raise ValueError(f"Expected last dim>1, got {data.shape[-1]}")
        tup = data.squeeze(-1).long() - self.idx_offset
        tup = list(map(tuple, tup.tolist()))
        idxs = self.indexer.get_idxs(tup)
        idxs_tens = torch.tensor(idxs, dtype=torch.long, device=self.device)
        return self.codebook[idxs_tens]

    def decode(self, hv: torch.Tensor) -> torch.LongTensor:
        """Decode hypervector back to feature tuples."""
        if hv.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim={self.dim}, got {hv.shape[-1]}")
        sims = torchhd.cosine_similarity(hv, self.codebook)
        idx = sims.argmax(dim=-1)
        idx += self.idx_offset
        idx = [int(i.item()) for i in idx.squeeze()]
        res = torch.tensor(self.indexer.get_tuples(idx))
        return res.float()

    def decode_index(self, label: int) -> torch.LongTensor:
        """Given a codebook index, return the feature tuple."""
        return torch.tensor([self.indexer.get_tuple(label)], dtype=torch.long, device=self.device)
