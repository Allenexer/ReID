"""CLIP-ReID 推理与实验归档。"""

from __future__ import annotations

import ast
import csv
import importlib.metadata
import json
import logging
import platform
import random
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from yacs.config import CfgNode

Record = tuple[str, int, int, int]


def create_run_dir(output_dir: Path) -> Path:
    """创建运行目录，已存在时抛出 FileExistsError。"""
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def export_results(
    output_dir: Path,
    *,
    data_root: Path,
    records: list[Record],
    num_query: int,
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    distmat: np.ndarray,
    cmc: np.ndarray,
    mean_ap: float,
) -> dict[str, object]:
    """按 query/gallery 顺序保存特征、距离、指标和样本清单。"""
    num_gallery = len(records) - num_query
    if num_query <= 0 or num_gallery <= 0:
        raise ValueError("query 和 gallery 都必须非空")
    if (
        query_features.ndim != 2
        or gallery_features.ndim != 2
        or query_features.shape[0] != num_query
        or gallery_features.shape[0] != num_gallery
        or query_features.shape[1] != gallery_features.shape[1]
        or distmat.shape != (num_query, num_gallery)
    ):
        raise ValueError("特征、距离矩阵与样本清单的维度不一致")
    if not all(np.isfinite(array).all() for array in (query_features, gallery_features, distmat)):
        raise ValueError("特征或距离矩阵存在 NaN/Inf")
    if cmc.ndim != 1 or len(cmc) == 0 or not np.isfinite(cmc).all():
        raise ValueError("CMC 必须是非空有限数值数组")
    if not 0 <= mean_ap <= 1 or np.any((cmc < 0) | (cmc > 1)):
        raise ValueError("mAP 和 CMC 必须位于 0～1")

    rows = [
        {
            "path": Path(path).resolve().relative_to(data_root.resolve()).as_posix(),
            "pid": int(pid),
            "camid": int(camid),
            "viewid": int(viewid),
        }
        for path, pid, camid, viewid in records
    ]
    for split, split_rows in (("query", rows[:num_query]), ("gallery", rows[num_query:])):
        with (output_dir / f"{split}.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["index", "path", "pid", "camid", "viewid"])
            writer.writeheader()
            writer.writerows({"index": index, **row} for index, row in enumerate(split_rows))

    for name, array in (
        ("query_features", query_features),
        ("gallery_features", gallery_features),
        ("distmat", distmat),
    ):
        with (output_dir / f"{name}.npy").open("xb") as array_stream:
            np.save(array_stream, array, allow_pickle=False)

    metrics: dict[str, object] = {
        "schema_version": 1,
        "model": "clipreid",
        "dataset": "market1501",
        "evaluation_source": "clipreid_upstream",
        "protocol": "market1501",
        "distance_metric": "squared_euclidean",
        "feature_normalization": "l2",
        "re_ranking": False,
        "metric_scale": "0-1",
        "num_query": num_query,
        "num_gallery": num_gallery,
        "mAP": float(mean_ap),
        **{
            f"rank{rank}": float(cmc[rank - 1]) if len(cmc) >= rank else None for rank in (1, 5, 10)
        },
    }
    with (output_dir / "metrics.json").open("x", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return metrics


def load_run_config(
    root: Path,
    config_path: Path,
    output_dir: Path,
    batch_size: int | None = None,
    seed: int | None = None,
) -> CfgNode:
    """合并配置，以 CLIP-ReID 目录为基准解析数据和权重路径。"""
    upstream = root / "third_party" / "CLIP-ReID"
    if not (upstream / "test_clipreid.py").is_file():
        raise FileNotFoundError("请先运行 bash scripts/setup_third_party.sh")
    sys.path.insert(0, str(upstream))
    defaults = import_module("config").cfg

    cfg = defaults.clone()
    cfg.merge_from_file(str(config_path))
    if batch_size is not None:
        cfg.TEST.IMS_PER_BATCH = batch_size
    if seed is not None:
        cfg.SOLVER.SEED = seed
    if cfg.TEST.IMS_PER_BATCH <= 0 or not 0 <= cfg.SOLVER.SEED < 2**32:
        raise ValueError("batch-size 必须为正数，seed 必须位于 0～2**32-1")
    if cfg.DATASETS.NAMES != "market1501":
        raise ValueError("当前入口只支持 Market1501")
    if cfg.TEST.RE_RANKING or cfg.TEST.FEAT_NORM != "yes":
        raise ValueError("当前入口要求 TEST.RE_RANKING=False、TEST.FEAT_NORM='yes'")
    if not cfg.TEST.WEIGHT:
        raise ValueError("TEST.WEIGHT 不能为空")
    cfg.DATASETS.ROOT_DIR = str((upstream / cfg.DATASETS.ROOT_DIR).resolve())
    cfg.TEST.WEIGHT = str((upstream / cfg.TEST.WEIGHT).resolve())
    cfg.OUTPUT_DIR = str(output_dir.resolve())
    dataset = Path(cfg.DATASETS.ROOT_DIR) / "Market-1501-v15.09.15"
    for split in ("bounding_box_train", "query", "bounding_box_test"):
        if not (dataset / split).is_dir():
            raise FileNotFoundError(dataset / split)
    if not Path(cfg.TEST.WEIGHT).is_file():
        raise FileNotFoundError(cfg.TEST.WEIGHT)
    cfg.freeze()
    return cfg


def environment_snapshot() -> str:
    """记录 Python、CUDA、GPU 和依赖版本。"""
    import torch

    cuda_available = torch.cuda.is_available()
    lines = [
        f"Python: {sys.version}",
        f"Executable: {sys.executable}",
        f"Platform: {platform.platform()}",
        f"PyTorch: {torch.__version__}",
        f"CUDA build: {torch.version.cuda}",
        f"CUDA available: {cuda_available}",
    ]
    if cuda_available:
        lines.extend(
            f"GPU {index}: {torch.cuda.get_device_name(index)}"
            for index in range(torch.cuda.device_count())
        )
    lines.append("\nPython packages:")
    lines.extend(
        sorted(
            f"{item.metadata['Name']}=={item.version}"
            for item in importlib.metadata.distributions()
        )
    )
    return "\n".join(lines) + "\n"


def git_snapshot(root: Path) -> str:
    """记录项目与上游的 commit 和工作区状态。"""
    lines = []
    for repo in (root, root / "third_party" / "CLIP-ReID"):
        lines.append(f"Repository: {repo}")
        for args in (("rev-parse", "HEAD"), ("status", "--short")):
            result = subprocess.run(
                ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
            )
            lines.append(f"git {' '.join(args)} (exit {result.returncode}):")
            lines.append((result.stdout if result.returncode == 0 else result.stderr).strip())
    return "\n".join(lines) + "\n"


def infer(cfg: CfgNode, output_dir: Path) -> dict[str, object]:
    """调用上游模型和评测器，导出检索结果。"""
    import torch
    from torch.utils.data import SequentialSampler

    make_dataloader = import_module("datasets.make_dataloader_clipreid").make_dataloader
    make_model = import_module("model.make_model_clipreid").make_model
    setup_logger = import_module("utils.logger").setup_logger
    evaluator_type = import_module("utils.metrics").R1_mAP_eval

    if not torch.cuda.is_available():
        raise RuntimeError("需要可用的 CUDA 版 PyTorch；请在能访问 GPU 的 WSL 终端运行")
    random.seed(cfg.SOLVER.SEED)
    np.random.seed(cfg.SOLVER.SEED)
    torch.manual_seed(cfg.SOLVER.SEED)
    torch.cuda.manual_seed_all(cfg.SOLVER.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logger = setup_logger("transreid", str(output_dir), if_train=False)
    logger.info("Running with config:\n%s", cfg)
    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    if not isinstance(val_loader.sampler, SequentialSampler):
        raise ValueError("导出顺序清单要求验证集使用 SequentialSampler")
    records = list(val_loader.dataset.dataset)
    logger.info("query=%d, gallery=%d", num_query, len(records) - num_query)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to("cuda").eval()
    evaluator = evaluator_type(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    position = 0
    with torch.inference_mode():
        for images, pids, camids, cam_labels, view_labels, names in val_loader:
            expected = [
                (Path(path).name, int(pid), int(camid), int(viewid))
                for path, pid, camid, viewid in records[position : position + len(names)]
            ]
            actual = [
                (name, int(pid), int(camid), int(viewid))
                for name, pid, camid, viewid in zip(names, pids, camids, view_labels, strict=True)
            ]
            if actual != expected:
                raise ValueError("实际推理顺序与样本清单不一致")
            features = model(
                images.to("cuda"),
                cam_label=cam_labels.to("cuda") if cfg.MODEL.SIE_CAMERA else None,
                view_label=view_labels.to("cuda") if cfg.MODEL.SIE_VIEW else None,
            )
            evaluator.update((features, pids, camids))
            position += len(names)
    if position != len(records):
        raise ValueError("验证集未被完整遍历")
    cmc, mean_ap, distmat, _, _, query_features, gallery_features = evaluator.compute()
    metrics = export_results(
        output_dir,
        data_root=Path(cfg.DATASETS.ROOT_DIR),
        records=records,
        num_query=num_query,
        query_features=query_features.cpu().numpy(),
        gallery_features=gallery_features.cpu().numpy(),
        distmat=distmat,
        cmc=cmc,
        mean_ap=mean_ap,
    )
    logger = logging.getLogger("transreid.test")
    logger.info("Validation Results")
    logger.info("mAP: %.1f%%", mean_ap * 100)
    for rank in (1, 5, 10):
        rank_value = metrics[f"rank{rank}"]
        if isinstance(rank_value, float):
            logger.info("CMC curve, Rank-%d: %.1f%%", rank, rank_value * 100)
    logger.info("完整精度指标及原始数组已保存到 %s", output_dir)
    return metrics


def run_experiment(root: Path, cfg: CfgNode) -> dict[str, object]:
    """归档配置、环境、版本及推理结果，记录运行状态。"""
    output_dir = create_run_dir(Path(cfg.OUTPUT_DIR))
    status = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}

    def save_status() -> None:
        (output_dir / "status.json").write_text(
            json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    save_status()
    try:
        (output_dir / "config.yaml").write_text(dump_config(cfg), encoding="utf-8")
        (output_dir / "git.txt").write_text(git_snapshot(root), encoding="utf-8")
        (output_dir / "env.txt").write_text(environment_snapshot(), encoding="utf-8")
        metrics = infer(cfg, output_dir)
    except BaseException:
        status["status"] = "failed"
        (output_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    else:
        status["status"] = "success"
        return metrics
    finally:
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_status()


def dump_config(cfg: CfgNode) -> str:
    """避免 YACS 将 '0'、'None' 等字符串转为其他类型。"""
    saved = cfg.clone()
    saved.defrost()

    def protect_strings(node: dict) -> None:
        for key, value in node.items():
            if isinstance(value, dict):
                protect_strings(value)
            elif isinstance(value, str):
                try:
                    ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    continue
                node[key] = repr(value)

    protect_strings(saved)
    return saved.dump()
