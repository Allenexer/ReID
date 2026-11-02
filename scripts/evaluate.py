#!/usr/bin/env python
"""读取成功的基线归档并执行统一评测。"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reid.artifacts import evaluate_run  # noqa: E402
from src.reid.metrics import PROTOCOLS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, choices=PROTOCOLS)
    parser.add_argument("--distance", required=True, choices=("native", "cosine"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        metrics = evaluate_run(args.run_dir, args.protocol, args.distance, args.output_dir)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, "评测失败：{}\n".format(error))
    print(
        "评测完成：mAP={:.6f}，Rank-1={:.6f}，有效 query={}，跳过={}，产物={}".format(
            metrics["mAP"],
            metrics["rank1"],
            metrics["num_valid_query"],
            metrics["num_skipped_query"],
            args.output_dir.absolute(),
        )
    )


if __name__ == "__main__":
    main()
