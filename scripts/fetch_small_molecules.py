#!/usr/bin/env python
"""Fetch small organic molecules (3-6 heavy atoms) from PubChem.

Enumerates molecular formulas for 3-6 heavy atoms using the ZINC-supported
atom set {C, N, O, F, S, Cl, Br}, queries PubChem's PUG REST API for each
formula, validates results with RDKit, and appends new molecules to
``data/small_molecules.csv``.

Usage::

    python scripts/fetch_small_molecules.py [--max-per-formula 200] [--csv data/small_molecules.csv]
"""

import argparse
import csv
import io
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import List, Optional, Set, Tuple

from rdkit import Chem, RDLogger

# Silence RDKit warnings during validation
RDLogger.logger().setLevel(RDLogger.ERROR)

# Supported atom types (matching ZINC_ATOM_TYPES in flow_edge_decoder.py)
ZINC_ATOM_TYPES = {"Br", "C", "Cl", "F", "I", "N", "O", "P", "S"}

# Atom types to use for formula enumeration (skip P, I — extremely rare in
# small organic molecules, and including them would explode the formula space)
ENUM_ATOMS = [
    ("C", "C"),
    ("N", "N"),
    ("O", "O"),
    ("F", "F"),
    ("S", "S"),
    ("Cl", "Cl"),
    ("Br", "Br"),
]

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def enumerate_formulas(min_atoms: int = 3, max_atoms: int = 6) -> List[str]:
    """Enumerate molecular formulas for *min_atoms*–*max_atoms* heavy atoms.

    At least one carbon is required (organic molecules).  Returns Hill-order
    formulas suitable for PubChem's ``fastformula`` endpoint.
    """
    formulas = []
    atom_symbols = [sym for sym, _ in ENUM_ATOMS]

    for n in range(min_atoms, max_atoms + 1):
        for combo in _compositions(n, len(atom_symbols)):
            counts = dict(zip(atom_symbols, combo))
            if counts.get("C", 0) < 1:
                continue  # need at least one carbon
            formula = _counts_to_formula(counts)
            formulas.append(formula)

    return formulas


def _compositions(n: int, k: int):
    """Yield all *k*-tuples of non-negative ints summing to *n*."""
    if k == 1:
        yield (n,)
        return
    for first in range(n + 1):
        for rest in _compositions(n - first, k - 1):
            yield (first,) + rest


def _counts_to_formula(counts: dict) -> str:
    """Convert atom counts to Hill-order molecular formula string."""
    parts = []
    # Hill order: C first, then H, then alphabetical
    for sym in sorted(counts.keys(), key=lambda s: (s != "C", s)):
        c = counts[sym]
        if c == 0:
            continue
        parts.append(f"{sym}{c}" if c > 1 else sym)
    return "".join(parts)


def query_pubchem(formula: str, max_records: int = 200) -> List[str]:
    """Query PubChem for canonical SMILES matching a molecular formula.

    Uses the PUG REST ``fastformula`` endpoint.  Returns a list of SMILES
    strings.  Retries on HTTP 503 (server busy) with exponential backoff.
    """
    url = (
        f"{PUBCHEM_BASE}/compound/fastformula/{formula}"
        f"/property/CanonicalSMILES,HeavyAtomCount/CSV"
        f"?MaxRecords={max_records}"
    )

    text = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=60) as resp:
                text = resp.read().decode("utf-8")
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []  # no compounds for this formula
            if e.code in (503, 504) and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            return []
        except (urllib.error.URLError, TimeoutError):
            if attempt < 3:
                time.sleep(2 ** attempt)
                continue
            return []

    if text is None:
        return []

    # Parse CSV response — PubChem returns varying column names
    # (e.g. "ConnectivitySMILES", "SMILES") depending on the endpoint
    smiles_list = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        smiles = ""
        for col in ("CanonicalSMILES", "ConnectivitySMILES", "SMILES",
                     "IsomericSMILES"):
            smiles = row.get(col, "").strip()
            if smiles:
                break
        if smiles:
            smiles_list.append(smiles)

    return smiles_list


def validate_molecule(smiles: str, min_atoms: int = 3, max_atoms: int = 6) -> Optional[str]:
    """Validate a SMILES string for ZINC compatibility.

    Returns the canonical SMILES if valid, else ``None``.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Check heavy atom count
    n = mol.GetNumHeavyAtoms()
    if n < min_atoms or n > max_atoms:
        return None

    # Check all atoms in ZINC set
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ZINC_ATOM_TYPES:
            return None

    # Check connectivity
    canonical = Chem.MolToSmiles(mol)
    if "." in canonical:
        return None

    # Sanitization check
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    return canonical


def load_existing_smiles(csv_path: Path) -> Set[str]:
    """Load existing SMILES from the CSV to avoid duplicates."""
    existing = set()
    if not csv_path.exists():
        return existing
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row.get("smiles", "").strip()
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    existing.add(Chem.MolToSmiles(mol))
    return existing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default="data/small_molecules.csv",
        help="Path to small molecules CSV (default: data/small_molecules.csv)",
    )
    parser.add_argument(
        "--max-per-formula",
        type=int,
        default=200,
        help="Max PubChem results per formula (default: 200)",
    )
    parser.add_argument(
        "--min-atoms",
        type=int,
        default=3,
        help="Minimum heavy atom count (default: 3)",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=6,
        help="Maximum heavy atom count (default: 6)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel PubChem query threads (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results but don't write to CSV",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)

    print(f"Loading existing SMILES from {csv_path}...", flush=True)
    existing = load_existing_smiles(csv_path)
    print(f"  {len(existing)} existing molecules", flush=True)

    print(f"\nEnumerating formulas for {args.min_atoms}-{args.max_atoms} heavy atoms...", flush=True)
    formulas = enumerate_formulas(args.min_atoms, args.max_atoms)
    print(f"  {len(formulas)} formulas to query", flush=True)

    # Thread-safe collection
    lock = Lock()
    new_molecules = []  # (canonical_smiles, heavy_atom_count)
    seen = set(existing)
    stats = {"fetched": 0, "valid": 0, "done": 0, "errors": 0}

    def process_formula(formula):
        """Query + validate one formula (runs in thread)."""
        smiles_list = query_pubchem(formula, args.max_per_formula)

        local_new = []
        local_valid = 0
        for smiles in smiles_list:
            canonical = validate_molecule(smiles, args.min_atoms, args.max_atoms)
            if canonical is None:
                continue
            local_valid += 1
            mol = Chem.MolFromSmiles(canonical)
            n_atoms = mol.GetNumHeavyAtoms()
            local_new.append((canonical, n_atoms))

        with lock:
            stats["fetched"] += len(smiles_list)
            stats["valid"] += local_valid
            stats["done"] += 1
            for canonical, n_atoms in local_new:
                if canonical not in seen:
                    seen.add(canonical)
                    new_molecules.append((canonical, n_atoms))

            if stats["done"] % 20 == 0 or stats["done"] == len(formulas):
                print(
                    f"  [{stats['done']}/{len(formulas)}] "
                    f"fetched={stats['fetched']} valid={stats['valid']} "
                    f"new={len(new_molecules)} errors={stats['errors']}",
                    flush=True,
                )

        return formula, len(smiles_list)

    # Parallel query
    print(f"\nQuerying PubChem with {args.workers} threads...", flush=True)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_formula, f): f for f in formulas}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                with lock:
                    stats["errors"] += 1

    elapsed = time.time() - t0

    # Summary by size
    print(f"\n--- Summary (took {elapsed:.0f}s) ---", flush=True)
    print(f"Total fetched from PubChem: {stats['fetched']}")
    print(f"Total valid (ZINC-compatible): {stats['valid']}")
    print(f"New unique molecules: {len(new_molecules)}")

    size_counts = {}
    for _, n in new_molecules:
        size_counts[n] = size_counts.get(n, 0) + 1

    for size in sorted(size_counts):
        print(f"  {size} atoms: {size_counts[size]} molecules")

    # Show examples
    if new_molecules:
        print("\nExamples:")
        for smiles, n in new_molecules[:10]:
            print(f"  {smiles} ({n} atoms)")

    # Write to CSV
    if not args.dry_run and new_molecules:
        print(f"\nAppending {len(new_molecules)} molecules to {csv_path}...")
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for smiles, _ in new_molecules:
                writer.writerow([smiles, "pubchem"])
        print("Done!")
    elif args.dry_run:
        print("\n(Dry run — not writing to CSV)")
    else:
        print("\nNo new molecules to add.")


if __name__ == "__main__":
    main()
