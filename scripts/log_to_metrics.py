#!/usr/bin/env python
"""Convert an upstream baseline log into metrics.json."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reid.logparse import metrics_from_log  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", choices=("transreid", "clipreid"), default="transreid")
    parser.add_argument("--dataset", choices=("market1501", "ccvid"), default="market1501")
    parser.add_argument("--num-query", type=int)
    parser.add_argument("--num-gallery", type=int)
    args = parser.parse_args()

    log_path = (ROOT / args.log).resolve()
    output = (ROOT / args.output).resolve()
    metrics = metrics_from_log(
        log_path,
        model=args.model,
        dataset=args.dataset,
        num_query=args.num_query,
        num_gallery=args.num_gallery,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"已写入：{output}")


if __name__ == "__main__":
    main()
