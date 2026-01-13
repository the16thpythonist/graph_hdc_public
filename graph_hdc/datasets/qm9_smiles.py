"""
QM9 SMILES Dataset.

PyG InMemoryDataset for QM9 molecular graphs from SMILES.

Node features: [atom_type, degree-1, formal_charge, total_Hs]
- Atom types: 4 (C, N, O, F)
- Degrees: 5 values (0-4)
- Formal charges: 3 values (0, 1, 2 for 0, +1, -1)
- Total Hs: 5 values (0-4)

Combinatorial space: 4 * 5 * 3 * 5 = 300 possible node types
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import QED, Crippen
from torch_geometric.data import Data, InMemoryDataset
from tqdm.auto import tqdm

from graph_hdc.utils.helpers import ROOT

DATASET_PATH = ROOT / "data"

QM9_ATOM_TO_IDX: dict[str, int] = {"C": 0, "N": 1, "O": 2, "F": 3}
QM9_IDX_TO_ATOM: dict[int, str] = {v: k for k, v in QM9_ATOM_TO_IDX.items()}


def iter_smiles(fp: Path):
    """Yield connected SMILES strings, skipping disconnected molecules and optional header."""
    with fp.open() as fh:
        for i, line in enumerate(fh):
            if i == 0 and line.strip().lower() == "smiles":
                continue
            if line := line.strip().split()[0]:
                # Skip disconnected molecules (contain '.')
                if "." in line:
                    continue
                yield line


def mol_to_data(mol: Chem.Mol) -> Data:
    """Convert RDKit molecule to PyG Data with QM9 features."""
    x = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in QM9_ATOM_TO_IDX:
            raise ValueError(f"Unexpected atom '{sym}' for QM9.")
        x.append([
            float(QM9_ATOM_TO_IDX[sym]),
            float(max(0, atom.GetDegree() - 1)),
            float(atom.GetFormalCharge() if atom.GetFormalCharge() >= 0 else 2),
            float(atom.GetTotalNumHs()),
        ])

    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]

    logp = Crippen.MolLogP(mol)
    qed = QED.qed(mol)

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        smiles=Chem.MolToSmiles(mol, canonical=True),
        logp=torch.tensor([float(logp)], dtype=torch.float32),
        qed=torch.tensor([float(qed)], dtype=torch.float32),
    )


class QM9Smiles(InMemoryDataset):
    """
    QM9 SMILES dataset.

    Reads <split>_smile.txt from root/raw/ and caches processed data.

    Parameters
    ----------
    root : Path
        Dataset root directory
    split : str
        One of {"train", "valid", "test"}
    """

    def __init__(
        self,
        root: str | Path = DATASET_PATH / "QM9Smiles",
        split: str = "train",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        enc_suffix: str = "",
    ) -> None:
        self.split = split.lower()
        self.enc_suffix = enc_suffix
        assert self.split in {"train", "valid", "test"}
        super().__init__(root, transform, pre_transform, pre_filter)

        with open(self.processed_paths[0], "rb") as f:
            self.data, self.slices = torch.load(f, map_location="cpu", weights_only=False)

    @property
    def raw_file_names(self) -> list[str]:
        return [f"{self.split}_smile.txt"]

    @property
    def processed_file_names(self) -> list[str]:
        suffix = f"_{self.enc_suffix}" if self.enc_suffix else ""
        return [f"data_{self.split}{suffix}.pt"]

    def download(self):
        pass

    def process(self):
        data_list: list[Data] = []

        src = Path(self.raw_paths[0])

        for s in tqdm(iter_smiles(src), desc=f"QM9Smiles[{self.split}]"):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            data = mol_to_data(mol)
            if self.pre_filter and not self.pre_filter(data):
                continue
            if self.pre_transform:
                data = self.pre_transform(data)
            data_list.append(data)

        data, slices = self.collate(data_list)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
