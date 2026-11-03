#!/usr/bin/env python
"""TransReID Market1501 实验入口。"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reid.transreid import load_run_config, run_experiment  # noqa: E402


def main() -> None:
    """解析参数并运行实验。"""
    parser = argparse.ArgumentParser(description="运行 TransReID 基线并导出实验产物")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/transreid/market1501.yml")
    )
    parser.add_argument("--output-dir", type=Path, help="新的输出目录")
    parser.add_argument("--batch-size", type=int, help="覆盖测试 batch size")
    parser.add_argument("--seed", type=int, help="覆盖随机种子")
    parser.add_argument("--manifest", type=Path, help="可选的 Market1501 原始 manifest")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置，不运行推理")
    args = parser.parse_args()

    started = datetime.now()
    output_dir = args.output_dir or Path(
        "outputs", f"{started:%Y%m%d-%H%M}-transreid-market-{started:%S%f}"
    )
    config_path = (ROOT / args.config).resolve()
    manifest = (ROOT / args.manifest).resolve() if args.manifest else None
    cfg = load_run_config(
        ROOT,
        config_path,
        (ROOT / output_dir).resolve(),
        args.batch_size,
        args.seed,
        manifest,
    )
    if args.dry_run:
        print("路径和配置检查通过；未运行 GPU，未创建输出目录。")
        print(cfg.dump())
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.DEVICE_ID)
    print(f"实验输出目录：{cfg.OUTPUT_DIR}", flush=True)
    metrics = run_experiment(ROOT, cfg)
    print(f"运行成功：mAP={metrics['mAP']:.6f}，Rank-1={metrics['rank1']:.6f}")
    print(f"产物已保存：{cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
