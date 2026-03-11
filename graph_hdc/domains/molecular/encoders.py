"""
Molecular-specific feature encoders.

These handle the non-trivial mappings used in molecular graph features.
"""

from __future__ import annotations


class DegreeEncoder:
    """Encodes atom degree using the existing ``max(0, degree - 1)`` convention.

    This matches the encoding in ``qm9_smiles.mol_to_data()`` and
    ``zinc_smiles.mol_to_data()``.  Degree 0 and degree 1 both map to
    index 0 (they are aliased).

    Parameters
    ----------
    max_degree : int
        Maximum *original* degree. ``num_bins = max_degree`` because
        encoded values range from 0 to ``max_degree - 1``.
    """

    def __init__(self, max_degree: int) -> None:
        self.max_degree = max_degree

    @property
    def num_bins(self) -> int:
        return self.max_degree

    def encode(self, degree: int) -> int:
        return min(max(0, degree - 1), self.max_degree - 1)

    def decode(self, index: int) -> int:
        if not (0 <= index < self.max_degree):
            raise ValueError(
                f"Index {index} out of range [0, {self.max_degree})"
            )
        return index + 1

    def __repr__(self) -> str:
        return f"DegreeEncoder(max_degree={self.max_degree})"


class FormalChargeEncoder:
    """Encodes formal charge as {0: 0, +1: 1, -1: 2}.

    Matches the existing ``FORMAL_CHARGE_IDX_TO_VAL`` mapping in
    ``graph_hdc/utils/chem.py``.
    """

    _CHARGE_TO_IDX: dict[int, int] = {0: 0, 1: 1, -1: 2}
    _IDX_TO_CHARGE: dict[int, int] = {0: 0, 1: 1, 2: -1}

    @property
    def num_bins(self) -> int:
        return 3

    def encode(self, charge: int) -> int:
        charge = int(charge)
        if charge not in self._CHARGE_TO_IDX:
            raise ValueError(
                f"Unknown formal charge {charge}. "
                f"Expected one of {list(self._CHARGE_TO_IDX.keys())}"
            )
        return self._CHARGE_TO_IDX[charge]

    def decode(self, index: int) -> int:
        if index not in self._IDX_TO_CHARGE:
            raise ValueError(f"Index {index} out of range [0, 3)")
        return self._IDX_TO_CHARGE[index]

    def __repr__(self) -> str:
        return "FormalChargeEncoder()"
