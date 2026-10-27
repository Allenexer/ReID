#!/usr/bin/env python
"""Parse an upstream baseline log into ``metrics.json``.

TransReID / CLIP-ReID print mAP and CMC at the end of their test log but do
not save a distance matrix. Use this script to turn that log into the same
``metrics.json`` contract as ``scripts/evaluate.py``. If a ``.npy`` distance
matrix is available, prefer ``scripts/evaluate.py``.

Example::

    python -m scripts.log_to_metrics \
        --log outputs/20261027-transreid-market/test_log.txt \
        --output outputs/20261027-transreid-market/metrics.json \
        --run-id 20261027-transreid-market \
        --baseline transreid \
        --dataset market1501

The output is the only source of numbers allowed into the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reid.logparse import build_metrics_payload  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="default: print to stdout")
    parser.add_argument(
        "--run-id",
        default="",
        help="default: name of the log's parent directory",
    )
    parser.add_argument("--baseline", default="")
    parser.add_argument("--dataset", default="")
    args = parser.parse_args()

    try:
        log_text = args.log.read_text(encoding="utf-8", errors="replace")
        payload = build_metrics_payload(
            log_text,
            log_path=str(args.log),
            run_id=args.run_id or args.log.parent.name,
            baseline=args.baseline,
            dataset=args.dataset,
        )
    except FileNotFoundError:
        parser.error(f"log file not found: {args.log}")
    except ValueError as exc:
        parser.error(str(exc))

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
