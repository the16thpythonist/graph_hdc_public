from __future__ import annotations

import inspect
import sys


# Keyword patterns that MUST be present in post_compute_encodings.
REQUIRED = [
    "node_terms[i].clone()",
    "edge_terms[i].clone()",
    "graph_terms[i].clone()",
    "d.clone()",
]


def main() -> int:
    try:
        from graph_hdc.datasets.utils import post_compute_encodings
    except Exception as e:
        print(f"FAIL: couldn't import post_compute_encodings: {e}")
        return 2

    # Unwrap any decorators (post_compute_encodings is @torch.no_grad-wrapped)
    # so we report the real source file, not the decorator's.
    target = inspect.unwrap(post_compute_encodings)
    src = inspect.getsource(target)
    missing = [kw for kw in REQUIRED if kw not in src]

    print(f"Inspecting: {inspect.getfile(target)}")
    print()
    for kw in REQUIRED:
        present = kw in src
        print(f"  [{'OK' if present else 'MISSING'}] {kw}")
    print()

    if missing:
        print(f"FAIL: {len(missing)}/{len(REQUIRED)} required clone(s) missing.")
        return 1

    print("PASS: all expected .clone() calls are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
