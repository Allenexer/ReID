#!/usr/bin/env python
"""CLIP-ReID 实验入口。"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reid.clipreid import load_run_config, run_experiment  # noqa: E402


def main() -> None:
    """解析参数并运行实验。"""
    parser = argparse.ArgumentParser(description="运行 Market1501 CLIP-ReID 基线并导出实验产物")
    parser.add_argument("--config", type=Path, default=Path("configs/clipreid/market1501.yml"))
    parser.add_argument("--output-dir", type=Path, help="新的输出目录；相对路径基于项目根目录")
    parser.add_argument("--batch-size", type=int, help="覆盖配置中的推理 batch size")
    parser.add_argument("--seed", type=int, help="覆盖配置中的随机种子")
    parser.add_argument("--dry-run", action="store_true", help="只检查路径和配置，不推理或写入")
    args = parser.parse_args()
    output_dir = args.output_dir or Path(
        "outputs", datetime.now().strftime("%Y%m%d-%H%M-clipreid-market-%S%f")
    )
    cfg = load_run_config(ROOT, ROOT / args.config, ROOT / output_dir, args.batch_size, args.seed)
    if args.dry_run:
        print("路径和配置检查通过；未运行 GPU，未创建输出目录。")
        print(cfg.dump())
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    print(f"实验输出目录：{cfg.OUTPUT_DIR}", flush=True)
    metrics = run_experiment(ROOT, cfg)
    print(f"运行成功：mAP={metrics['mAP']:.6f}, Rank-1={metrics['rank1']:.6f}")
    print(f"产物已保存：{cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
