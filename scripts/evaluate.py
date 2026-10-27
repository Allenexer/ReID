#!/usr/bin/env python
"""Compute mAP/CMC from a distance matrix and a dataset manifest.

This file is the shared CLI entry point. The evaluation algorithm is TODO
for member A; see ``src/reid/metrics.py``.

Intended contract for every baseline:

1. Save a ``query x gallery`` distance matrix as ``.npy``.
2. Keep the manifest used for the run.
3. Run this script to produce ``metrics.json``.

Example::

    python -m scripts.evaluate \
        --manifest data/processed/manifests/market1501.json \
        --distmat outputs/20261027-1700-transreid-market/distmat.npy \
        --output outputs/20261027-1700-transreid-market/metrics.json \
        --run-id 20261027-1700-transreid-market \
        --baseline transreid

The output is the only source of numbers allowed into the report.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--distmat", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="default: print to stdout")
    parser.add_argument("--max-rank", type=int, default=50)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--baseline", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument(
        "--include-same-camera",
        action="store_true",
        help="do not drop same-id same-camera gallery entries",
    )
    args = parser.parse_args()

    print(f"manifest={args.manifest} distmat={args.distmat}")
    # TODO(A): load the manifest and distance matrix, validate shapes,
    # call src.reid.metrics.evaluate_rank, and write metrics.json to
    # `args.output` (or print it when omitted).
    raise SystemExit(
        "TODO(A): evaluation is not implemented yet; see src/reid/metrics.py"
    )


if __name__ == "__main__":
    raise SystemExit(main())
