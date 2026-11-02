#!/usr/bin/env python
"""校验原始数据，生成索引、CSV 和验证报告。"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reid.manifest import build_manifest, save_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("market1501", "ccvid", "move"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args()
    try:
        manifest = build_manifest(args.dataset, args.root, args.labels)
        save_manifest(manifest, args.manifest, verify_images=args.verify_images)
    except (ValueError, OSError) as error:
        parser.exit(1, "数据检查失败：{}\n".format(error))
    print("数据检查通过：{}".format(args.manifest.absolute()))


if __name__ == "__main__":
    main()
