"""TransReID 推理、结果导出和实验归档。"""

from __future__ import annotations

import ast
import csv
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from .adapters.market1501 import load_market1501

if TYPE_CHECKING:
    from yacs.config import CfgNode

Record = tuple[str, int, int, int]


def create_run_dir(output_dir: Path) -> Path:
    """创建运行目录，已存在时拒绝覆盖。"""
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
    inference_seconds: float | None = None,
) -> dict[str, object]:
    """按 TransReID 的实际推理顺序保存标准运行归档。"""
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
    if inference_seconds is not None and (
        not np.isfinite(inference_seconds) or inference_seconds < 0
    ):
        raise ValueError("推理耗时须为非负有限数值")

    fields = ["index", "path", "pid", "camid", "viewid"]
    rows = [
        {
            "path": Path(path).resolve().relative_to(data_root.resolve()).as_posix(),
            "pid": int(pid),
            "camid": int(camid),
            "viewid": int(viewid),
        }
        for path, pid, camid, viewid in records
    ]
    if any(row["viewid"] != 1 for row in rows):
        raise ValueError("TransReID 的 viewid 必须为 1")

    for split, split_rows in (("query", rows[:num_query]), ("gallery", rows[num_query:])):
        with (output_dir / f"{split}.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({"index": index, **row} for index, row in enumerate(split_rows))

    for name, array in (
        ("query_features", query_features),
        ("gallery_features", gallery_features),
        ("distmat", distmat),
    ):
        with (output_dir / f"{name}.npy").open("xb") as stream:
            np.save(stream, np.asarray(array, dtype=np.float32), allow_pickle=False)

    metrics: dict[str, object] = {
        "schema_version": 1,
        "model": "transreid",
        "dataset": "market1501",
        "evaluation_source": "transreid_upstream",
        "protocol": "market1501",
        "distance_metric": "squared_euclidean",
        "feature_normalization": "l2",
        "re_ranking": False,
        "metric_scale": "0-1",
        "num_query": num_query,
        "num_gallery": num_gallery,
        "mAP": float(mean_ap),
        **{
            f"rank{rank}": float(cmc[rank - 1]) if len(cmc) >= rank else None
            for rank in (1, 5, 10)
        },
    }
    if inference_seconds is not None:
        metrics["inference_seconds"] = float(inference_seconds)
    with (output_dir / "metrics.json").open("x", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return metrics


def _resolve_input_path(value: str, root: Path, upstream: Path) -> Path:
    """兼容项目根目录配置和上游目录相对配置。"""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    project_path = (root / candidate).resolve()
    upstream_path = (upstream / candidate).resolve()
    if project_path.exists() or not upstream_path.exists():
        return project_path
    return upstream_path


def load_run_config(
    root: Path,
    config_path: Path,
    output_dir: Path,
    batch_size: int | None = None,
    seed: int | None = None,
    manifest: Path | None = None,
) -> CfgNode:
    """加载项目配置，并将数据、权重和输出路径固化为绝对路径。"""
    upstream = root / "third_party" / "TransReID"
    if not (upstream / "test.py").is_file():
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
    if str(cfg.DATASETS.NAMES).lower() != "market1501":
        raise ValueError("TransReID 入口当前只支持 market1501")
    if cfg.TEST.RE_RANKING or cfg.TEST.FEAT_NORM != "yes":
        raise ValueError("当前归档要求 TEST.RE_RANKING=False、TEST.FEAT_NORM='yes'")
    if cfg.MODEL.PRETRAIN_CHOICE != "none":
        raise ValueError("推理配置必须设置 MODEL.PRETRAIN_CHOICE='none'")
    if not cfg.TEST.WEIGHT:
        raise ValueError("TEST.WEIGHT 不能为空")

    cfg.defrost()
    cfg.DATASETS.ROOT_DIR = str(
        _resolve_input_path(str(cfg.DATASETS.ROOT_DIR), root, upstream)
    )
    cfg.TEST.WEIGHT = str(_resolve_input_path(str(cfg.TEST.WEIGHT), root, upstream))
    cfg.OUTPUT_DIR = str(output_dir.resolve())
    if manifest is not None:
        cfg.DATASETS.MANIFEST = str(manifest.resolve())
    cfg.freeze()

    data_root = Path(cfg.DATASETS.ROOT_DIR)
    load_market1501(data_root)
    if not Path(cfg.TEST.WEIGHT).is_file():
        raise FileNotFoundError(cfg.TEST.WEIGHT)
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
    """记录项目和 TransReID 上游的 commit 及工作区状态。"""
    lines = []
    for repo in (root, root / "third_party" / "TransReID"):
        lines.append(f"Repository: {repo}")
        for args in (("rev-parse", "HEAD"), ("status", "--short")):
            result = subprocess.run(
                ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
            )
            lines.append(f"git {' '.join(args)} (exit {result.returncode}):")
            lines.append((result.stdout if result.returncode == 0 else result.stderr).strip())
    return "\n".join(lines) + "\n"


def _market_loader(cfg: CfgNode):
    """用 A 的原始清单构造验证 loader，避开上游训练集强依赖和路径假设。"""
    import torch
    import torchvision.transforms as transforms
    from datasets.bases import ImageDataset
    from torch.utils.data import DataLoader

    data_root = Path(cfg.DATASETS.ROOT_DIR)
    splits = load_market1501(data_root)
    train = splits["train"]
    query = splits["query"]
    gallery = splits["gallery"]
    records: list[Record] = [
        (str((data_root / row.path).resolve()), row.pid, row.camid, 1)
        for row in [*query, *gallery]
    ]
    upstream_records = [
        (path.replace("\\", "/"), pid, camid, 1) for path, pid, camid, _ in records
    ]
    dataset = ImageDataset(upstream_records, transforms.Compose([
        transforms.Resize(cfg.INPUT.SIZE_TEST),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ]))

    def collate(batch):
        images, pids, camids, viewids, names = zip(*batch)
        camid_tensor = torch.tensor(camids, dtype=torch.int64)
        return (
            torch.stack(images, dim=0),
            pids,
            camids,
            camid_tensor,
            torch.tensor(viewids, dtype=torch.int64),
            names,
        )

    loader = DataLoader(
        dataset,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=collate,
    )
    num_classes = len({row.pid for row in train})
    camera_num = len({row.camid for row in train})
    return loader, len(query), num_classes, camera_num, 1, records


def infer(cfg: CfgNode, output_dir: Path) -> dict[str, object]:
    """调用 TransReID 模型并导出完整特征、距离矩阵和指标。"""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("需要可用的 CUDA 版 PyTorch；请在 Linux GPU 或 Colab 环境运行")
    random.seed(cfg.SOLVER.SEED)
    np.random.seed(cfg.SOLVER.SEED)
    torch.manual_seed(cfg.SOLVER.SEED)
    torch.cuda.manual_seed_all(cfg.SOLVER.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logger = import_module("utils.logger").setup_logger(
        "transreid", str(output_dir), if_train=False
    )
    logger.info("Running with config:\n%s", cfg)
    loader, num_query, num_classes, camera_num, view_num, records = _market_loader(cfg)
    logger.info("query=%d, gallery=%d", num_query, len(records) - num_query)

    model = import_module("model").make_model(
        cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num
    )
    model.load_param(cfg.TEST.WEIGHT)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to("cuda").eval()

    evaluator = import_module("utils.metrics").R1_mAP_eval(
        num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM
    )
    evaluator.reset()
    position = 0
    started = perf_counter()
    expected_records = records
    with torch.inference_mode():
        for images, pids, camids, camids_batch, viewids, names in loader:
            expected = [
                (Path(path).name, pid, camid, viewid)
                for path, pid, camid, viewid in expected_records[
                    position : position + len(names)
                ]
            ]
            actual = [
                (name, int(pid), int(camid), int(viewid))
                for name, pid, camid, viewid in zip(
                    names, pids, camids, viewids
                )
            ]
            if actual != expected:
                raise ValueError("实际推理顺序与 Market1501 清单不一致")
            features = model(
                images.to("cuda"),
                cam_label=camids_batch.to("cuda"),
                view_label=viewids.to("cuda"),
            )
            evaluator.update((features.cpu(), pids, camids))
            position += len(names)
    if position != len(records):
        raise ValueError("验证集未被完整遍历")

    cmc, mean_ap, distmat, _, _, query_features, gallery_features = evaluator.compute()
    inference_seconds = perf_counter() - started
    metrics = export_results(
        output_dir,
        data_root=Path(cfg.DATASETS.ROOT_DIR),
        records=records,
        num_query=num_query,
        query_features=query_features.numpy(),
        gallery_features=gallery_features.numpy(),
        distmat=distmat,
        cmc=cmc,
        mean_ap=mean_ap,
        inference_seconds=inference_seconds,
    )
    logger.info("Validation Results")
    logger.info("mAP: %.1f%%", mean_ap * 100)
    for rank in (1, 5, 10):
        value = metrics[f"rank{rank}"]
        if isinstance(value, float):
            logger.info("CMC curve, Rank-%d: %.1f%%", rank, value * 100)
    logger.info("完整精度指标及原始数组已保存到 %s", output_dir)
    logger.info("推理及评测耗时：%.3f 秒", inference_seconds)
    return metrics


def run_experiment(root: Path, cfg: CfgNode) -> dict[str, object]:
    """归档配置、环境、版本及推理状态。"""
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
    """导出同时兼容 YACS 和普通 YAML 读取的配置。"""
    import yaml

    def to_plain(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: to_plain(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [to_plain(item) for item in value]
        if isinstance(value, str):
            try:
                decoded = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                decoded = value
            if not isinstance(decoded, str) and decoded != value:
                return repr(value)
        return value

    return yaml.safe_dump(
        to_plain(cfg),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
