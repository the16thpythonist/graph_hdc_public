#!/usr/bin/env python
"""Quick check: do ZINC molecules contain charged atoms?"""
from collections import Counter

from rdkit import Chem
from tqdm.auto import tqdm

from graph_hdc.datasets.utils import get_split


def main():
    charged_mols = 0
    total_mols = 0
    charge_counter: Counter = Counter()  # (symbol, charge) -> count
    charged_smiles: list[str] = []

    for split in ("train", "valid", "test"):
        ds = get_split(split, dataset="zinc")
        for data in tqdm(ds, desc=f"zinc/{split}"):
            smiles = data.smiles
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            total_mols += 1
            has_charge = False
            for atom in mol.GetAtoms():
                fc = atom.GetFormalCharge()
                if fc != 0:
                    has_charge = True
                    charge_counter[(atom.GetSymbol(), fc)] += 1
            if has_charge:
                charged_mols += 1
                if len(charged_smiles) < 20:
                    charged_smiles.append(smiles)

    print(f"\nTotal molecules: {total_mols}")
    print(f"Molecules with charged atoms: {charged_mols} ({100 * charged_mols / total_mols:.2f}%)")
    print(f"\nCharged atom types (symbol, charge) -> count:")
    for (sym, chg), cnt in charge_counter.most_common():
        print(f"  {sym:>2s} {chg:+d}: {cnt}")

    if charged_smiles:
        print(f"\nExample charged molecules (first {len(charged_smiles)}):")
        for s in charged_smiles:
            print(f"  {s}")


if __name__ == "__main__":
    main()
