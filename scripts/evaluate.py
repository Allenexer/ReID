#!/usr/bin/env python
"""Compute mAP/CMC from a distance matrix and a dataset manifest.

Contract for every baseline:

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
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reid.manifest import DatasetManifest  # noqa: E402
from src.reid.metrics import evaluate_rank  # noqa: E402


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

    manifest = DatasetManifest.load(args.manifest)
    distmat = np.load(args.distmat)
    if distmat.shape != (len(manifest.query), len(manifest.gallery)):
        raise SystemExit(
            f"distmat shape {distmat.shape} does not match manifest "
            f"({len(manifest.query)} query x {len(manifest.gallery)} gallery)"
        )

    result = evaluate_rank(
        distmat=distmat,
        q_pids=np.asarray([s.pid for s in manifest.query]),
        g_pids=np.asarray([s.pid for s in manifest.gallery]),
        q_camids=np.asarray([s.camid for s in manifest.query]),
        g_camids=np.asarray([s.camid for s in manifest.gallery]),
        max_rank=args.max_rank,
        same_camera_exclude=not args.include_same_camera,
    )

    payload = {
        "run_id": args.run_id,
        "baseline": args.baseline,
        "dataset": args.dataset or manifest.name,
        "protocol": manifest.protocol,
        "manifest": str(args.manifest),
        "distmat": str(args.distmat),
        "distmat_shape": list(distmat.shape),
        "max_rank": args.max_rank,
        **result.as_dict(),
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"metrics written to {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
