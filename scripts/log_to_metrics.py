#!/usr/bin/env python
"""Parse an upstream baseline log into ``metrics.json``.

This file is the shared CLI entry point. The parser and payload builder
are TODO for member B; see ``src/reid/logparse.py``.

TransReID / CLIP-ReID print mAP and CMC at the end of their test log but
do not save a distance matrix. Use this script to turn that log into the
same ``metrics.json`` contract as ``scripts/evaluate.py``.

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
from pathlib import Path


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

    print(f"log={args.log} output={args.output}")
    # TODO(B): read args.log, call src.reid.logparse.build_metrics_payload,
    # and write JSON to args.output (or print it when omitted).
    raise SystemExit(
        "TODO(B): log-to-metrics conversion is not implemented yet; "
        "see src/reid/logparse.py"
    )


if __name__ == "__main__":
    raise SystemExit(main())
