"""
Type definitions for the HDC encoding system.
"""

import enum
from dataclasses import dataclass

from torchhd import FHRRTensor, HRRTensor, MAPTensor, VSATensor


class VSAModel(enum.Enum):
    """
    Supported Vector Symbolic Architecture models.

    - MAP: Multiply-Add-Permute with bipolar {±1}^D vectors
    - HRR: Holographic Reduced Representation with real continuous vectors
    - FHRR: Fourier HRR with complex phasor vectors
    """

    MAP = ("MAP", MAPTensor)
    HRR = ("HRR", HRRTensor)
    FHRR = ("FHRR", FHRRTensor)

    def __new__(cls, value: str, t_class: VSATensor) -> "VSAModel":
        obj = object.__new__(cls)
        obj._value_ = value
        obj._vsa_type_ = t_class
        return obj

    @classmethod
    def is_supported(cls, vsa_type: VSATensor) -> bool:
        return any(vsa.tensor_class == vsa_type for vsa in cls)

    @property
    def tensor_class(self) -> VSATensor:
        return self._vsa_type_


@dataclass(frozen=True)
class Feat:
    """
    Hashable node feature representation (discrete indices).

    Attributes:
        atom_type: Index of atom type (e.g., C=0, N=1, O=2, F=3 for QM9)
        degree_idx: Degree minus one, i.e., 0 -> degree 1, 4 -> degree 5
        formal_charge_idx: Encoded as 0, 1, 2 for charges [0, +1, -1]
        explicit_hs: Total explicit hydrogens (0..4)
        is_in_ring: Whether atom is in a ring (optional, used for ZINC)
    """

    atom_type: int
    degree_idx: int
    formal_charge_idx: int
    explicit_hs: int
    is_in_ring: bool | None = None

    @property
    def target_degree(self) -> int:
        """Final/desired node degree (degree index + 1)."""
        return self.degree_idx + 1

    def to_tuple(self) -> tuple:
        """Return feature tuple (atom_type, degree_idx, formal_charge_idx, explicit_hs[, is_in_ring])."""
        res = [self.atom_type, self.degree_idx, self.formal_charge_idx, self.explicit_hs]
        if self.is_in_ring is not None:
            res.append(int(self.is_in_ring))
        return tuple(res)

    @staticmethod
    def from_tuple(t: tuple) -> "Feat":
        """
        Construct a Feat from a tuple of length 4 or 5.

        Args:
            t: Tuple (atom_type, degree_idx, formal_charge_idx, explicit_hs[, is_in_ring])

        Returns:
            Feat instance

        Raises:
            ValueError: If tuple length is not 4 or 5
        """
        if len(t) == 4:
            a, d, c, h = t
            return Feat(int(a), int(d), int(c), int(h))

        a, d, c, h, r = t
        return Feat(int(a), int(d), int(c), int(h), bool(r) if r is not None else None)
