"""读取 B/C 归档并生成可追溯的 A 统一评测产物；导入不加载模型。"""

import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .features import cosine_distance, validate_features
from .manifest import Manifest, load_manifest, read_samples
from .metrics import PROTOCOLS, EvaluationResult, evaluate_rank
from .storage import new_directory, read_json, sha256, write_json

RUN_FILES = (
    "status.json",
    "metrics.json",
    "config.yaml",
    "env.txt",
    "git.txt",
    "distmat.npy",
    "query_features.npy",
    "gallery_features.npy",
    "query.csv",
    "gallery.csv",
)


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    manifest: Manifest
    distmat: np.ndarray
    query_features: np.ndarray
    gallery_features: np.ndarray
    metrics: Dict[str, Any]
    config: Dict[str, Any]
    hashes: Dict[str, str]
    source_manifest_check: Dict[str, Any]


def _metadata(metrics: Dict[str, Any], config: Dict[str, Any], path: Path) -> None:
    required = {
        "schema_version",
        "model",
        "dataset",
        "protocol",
        "evaluation_source",
        "metric_scale",
        "distance_metric",
        "feature_normalization",
        "re_ranking",
        "num_query",
        "num_gallery",
        "mAP",
        "rank1",
        "rank5",
        "rank10",
    }
    if not required <= metrics.keys() or metrics["schema_version"] != 1:
        raise ValueError("{}：缺少指标字段或 schema_version 不受支持".format(path))
    dataset, model = metrics["dataset"], metrics["model"]
    if dataset not in {"market1501", "ccvid"} or model not in {"clipreid", "transreid"}:
        raise ValueError("{}：不支持的 model/dataset；Move 正式协议尚未核实".format(path))
    allowed = {"market1501"} if dataset == "market1501" else set(PROTOCOLS[1:])
    if metrics["protocol"] not in allowed or metrics["metric_scale"] != "0-1":
        raise ValueError("{}：协议或指标标度矛盾".format(path))
    if metrics["distance_metric"] not in {"squared_euclidean", "euclidean", "cosine"}:
        raise ValueError("{}：未知 distance_metric".format(path))
    if (
        metrics["feature_normalization"] not in {"l2", "none"}
        or type(metrics["re_ranking"]) is not bool
    ):
        raise ValueError("{}：归一化或重排序元数据非法".format(path))
    for key in ("num_query", "num_gallery"):
        if type(metrics[key]) is not int or metrics[key] <= 0:
            raise ValueError("{}：{} 须为正整数".format(path, key))
    for key in ("mAP", "rank1", "rank5", "rank10"):
        value = metrics[key]
        unavailable = key.startswith("rank") and int(key[4:]) > min(50, metrics["num_gallery"])
        if unavailable:
            if value is not None:
                raise ValueError("{}：{} 应为 null".format(path, key))
        elif type(value) not in (float, int) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("{}：{} 须为 0～1 有限数值".format(path, key))
    try:
        datasets, test = config["DATASETS"], config["TEST"]
        if not isinstance(datasets, dict) or not isinstance(test, dict):
            raise ValueError("DATASETS/TEST 须为配置映射")
        if datasets["NAMES"] != dataset or not isinstance(datasets["ROOT_DIR"], str):
            raise ValueError("DATASETS.NAMES/ROOT_DIR 与指标矛盾")
        if test["FEAT_NORM"] not in ("yes", "no") or type(test["RE_RANKING"]) is not bool:
            raise ValueError("TEST.FEAT_NORM/RE_RANKING 的值或类型非法")
        norm = {"yes": "l2", "no": "none"}[test["FEAT_NORM"]]
        if norm != metrics["feature_normalization"] or test["RE_RANKING"] != metrics["re_ranking"]:
            raise ValueError("配置与归一化/重排序元数据矛盾")
        if not Path(datasets["ROOT_DIR"]).is_absolute():
            raise ValueError("归档 DATASETS.ROOT_DIR 须记录实际绝对路径")
        if model == "clipreid" and (
            norm != "l2"
            or metrics["re_ranking"]
            or metrics["distance_metric"] != "squared_euclidean"
        ):
            raise ValueError("C 归档须为 L2 特征、平方欧氏距离且不重排序")
        if dataset == "ccvid":
            frames = metrics["frames_per_sequence"]
            if type(frames) is not int or frames <= 0 or frames != datasets["NUM_FRAMES"]:
                raise ValueError("请求帧数与配置矛盾")
            if (
                metrics["sample_unit"] != "sequence"
                or metrics["frame_sampling"] != ("index" if frames == 1 else "uniform")
                or metrics["aggregation"] != "mean_then_l2"
                or type(datasets["FRAME_INDEX"]) is not int
                or datasets["FRAME_INDEX"] < 0
                or (frames > 1 and datasets["FRAME_INDEX"] != 0)
            ):
                raise ValueError("CCVID 抽帧/聚合元数据矛盾")
    except KeyError as error:
        raise ValueError("{}：配置或指标缺少字段 {}".format(path, error)) from error


def compare_manifest(run_manifest: Manifest, source: Manifest) -> Dict[str, Any]:
    """按显式根目录及逐行完整路径核对；viewid 保留生产者语义。"""
    if run_manifest.dataset != source.dataset:
        raise ValueError("运行清单与原始 manifest 数据集不一致")
    for split in ("query", "gallery"):
        actual, expected = run_manifest.splits[split], source.splits[split]
        if len(actual) != len(expected):
            raise ValueError("{}：原始 manifest 与运行清单规模不一致".format(split))
        for index, (row, reference) in enumerate(zip(actual, expected)):
            same = (run_manifest.data_root / row.path).resolve() == (
                source.data_root / reference.path
            ).resolve() and (row.pid, row.camid) == (reference.pid, reference.camid)
            if run_manifest.dataset == "ccvid":
                same = same and (
                    row.sequence,
                    row.clothes_label,
                    row.num_frames,
                    row.sampled_frames,
                    row.frame_paths,
                ) == (
                    reference.sequence,
                    reference.clothes_label,
                    reference.num_frames,
                    reference.sampled_frames,
                    reference.frame_paths,
                )
            if not same:
                raise ValueError(
                    "{}:{} 原始 manifest 与运行行序/标签/采样不一致".format(split, index + 2)
                )
    return {"status": "passed", "query_gallery_rows_checked": True}


def load_run(run_dir: Path) -> RunArtifacts:
    """仅接收 success 归档，NPY 禁用 pickle 且原数组使用只读 mmap。"""
    import yaml

    root = Path(run_dir).resolve()
    for name in RUN_FILES:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    hashes = {name: sha256(root / name) for name in RUN_FILES}
    if read_json(root / "status.json").get("status") != "success":
        raise ValueError("{}：仅接收 status=success 的运行".format(root / "status.json"))
    metrics = read_json(root / "metrics.json")
    try:
        config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError("{}：YAML 配置解析失败：{}".format(root / "config.yaml", error)) from error
    if not isinstance(config, dict):
        raise ValueError("{}：配置顶层须为映射".format(root / "config.yaml"))
    _metadata(metrics, config, root / "metrics.json")
    dataset = metrics["dataset"]
    splits = {
        split: read_samples(root / (split + ".csv"), dataset) for split in ("query", "gallery")
    }
    seen = set()
    for split, rows in splits.items():
        if len(rows) != metrics["num_" + split]:
            raise ValueError("{}：CSV 与 num_{} 不一致".format(root / (split + ".csv"), split))
        if dataset == "market1501" and [row.path for row in rows] != sorted(
            row.path for row in rows
        ):
            raise ValueError("{}：Market1501 清单须保留生产者 JPG 字典序".format(split))
        for row in rows:
            key = row.sequence if dataset == "ccvid" else row.path
            if key in seen:
                raise ValueError("{}：query/gallery 重复样本 {}".format(root, key))
            seen.add(key)
            expected_view = 0 if metrics["model"] == "clipreid" else 1
            if row.viewid != expected_view:
                raise ValueError("{}:{} viewid 与生产者约定不一致".format(split, row.index + 2))
            if dataset == "market1501":
                match = re.match(r"(-?\d+)_c(\d)", Path(row.path).name)
                if match is None or (int(match[1]), int(match[2]) - 1) != (row.pid, row.camid):
                    raise ValueError(
                        "{}:{} 文件路径与 pid/camid 不一致".format(split, row.index + 2)
                    )
            elif row.sampled_frames != min(row.num_frames, metrics["frames_per_sequence"]):
                raise ValueError("{}:{} 实际采样帧数不一致".format(split, row.index + 2))
            elif config["DATASETS"]["FRAME_INDEX"] >= row.num_frames:
                raise ValueError("{}:{} FRAME_INDEX 超出序列帧数".format(split, row.index + 2))
    arrays = {}
    for name in ("distmat", "query_features", "gallery_features"):
        path = root / (name + ".npy")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2 or array.dtype.kind not in "fiu" or min(array.shape) <= 0:
            raise ValueError("{}：须为非空实数二维 NPY，实际 {}".format(path, array.shape))
        if metrics["model"] == "clipreid" and array.dtype != np.float32:
            raise ValueError("{}：C 归档数组须为 float32".format(path))
        for index, row in enumerate(array):
            if not np.isfinite(row).all():
                raise ValueError("{}:{} 存在 NaN/Inf".format(path, index))
        arrays[name] = array
    qf, gf, distances = arrays["query_features"], arrays["gallery_features"], arrays["distmat"]
    nq, ng = validate_features(qf, gf)
    expected_shape = (len(splits["query"]), len(splits["gallery"]))
    if (nq, ng) != expected_shape or distances.shape != expected_shape:
        raise ValueError(
            "{}：数组维度 {} / {} 与清单 {} 不一致".format(
                root, (nq, ng), distances.shape, expected_shape
            )
        )
    if metrics["feature_normalization"] == "l2":
        for name, array in (("query_features", qf), ("gallery_features", gf)):
            if not np.allclose(np.linalg.norm(array, axis=1), 1.0, atol=1e-4, rtol=0):
                raise ValueError("{}：声明 L2 但特征范数不为 1".format(root / (name + ".npy")))
    if not metrics["re_ranking"]:
        # 同时核实数组和距离元数据，避免只改标签冒充距离统一。
        gallery = gf.astype(np.float64)
        for start in range(0, nq, 256):
            query = qf[start : start + 256].astype(np.float64)
            if metrics["distance_metric"] == "cosine":
                expected = cosine_distance(query, gallery)
            else:
                expected = (
                    (query * query).sum(1)[:, None]
                    + (gallery * gallery).sum(1)[None, :]
                    - 2 * query @ gallery.T
                )
                if metrics["distance_metric"] == "euclidean":
                    expected = np.sqrt(np.maximum(expected, 0))
            if not np.allclose(distances[start : start + 256], expected, atol=1e-4, rtol=1e-5):
                raise ValueError(
                    "{}：距离与特征/距离元数据不一致，query {} 附近".format(
                        root / "distmat.npy", start
                    )
                )
    manifest = Manifest(
        dataset,
        metrics["protocol"],
        "sequence" if dataset == "ccvid" else "image",
        Path(config["DATASETS"]["ROOT_DIR"]),
        splits,
        "producer_order",
    )
    source_check = {
        "status": "not_requested",
        "reason": "未配置 DATASETS.MANIFEST；离线仅校验归档内部契约",
    }
    if "MANIFEST" in config["DATASETS"]:
        manifest_path = Path(config["DATASETS"]["MANIFEST"])
        if not manifest_path.is_absolute():
            raise ValueError("DATASETS.MANIFEST 须为显式绝对路径")
        source_check = compare_manifest(manifest, load_manifest(manifest_path))
        source_check.update({"path": str(manifest_path), "sha256": sha256(manifest_path)})
    return RunArtifacts(root, manifest, distances, qf, gf, metrics, config, hashes, source_check)


def _evaluate(run: RunArtifacts, matrix: np.ndarray, protocol: str) -> EvaluationResult:
    query, gallery = run.manifest.splits["query"], run.manifest.splits["gallery"]
    return evaluate_rank(
        matrix,
        [row.pid for row in query],
        [row.pid for row in gallery],
        [row.camid for row in query],
        [row.camid for row in gallery],
        protocol,
        [row.clothes_label for row in query],
        [row.clothes_label for row in gallery],
    )


def _raw_metrics(result: EvaluationResult) -> Dict[str, Optional[float]]:
    return {
        "mAP": result.mAP,
        **{
            "rank{}".format(rank): float(result.cmc[rank - 1]) if rank <= len(result.cmc) else None
            for rank in (1, 5, 10)
        },
    }


def evaluate_run(run_dir: Path, protocol: str, distance: str, output_dir: Path) -> Dict[str, Any]:
    """校验后计算并原子发布；原 run 的数组、指标和配置始终只读。"""
    if protocol not in PROTOCOLS or distance not in {"native", "cosine"}:
        raise ValueError("非法 protocol/distance：{}/{}".format(protocol, distance))
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    run = load_run(run_dir)
    expected_dataset = "market1501" if protocol == "market1501" else "ccvid"
    if run.metrics["dataset"] != expected_dataset:
        raise ValueError("评测协议与运行数据集不一致")
    data_root = run.manifest.data_root.resolve()
    if data_root == output.resolve() or data_root in output.resolve().parents:
        raise ValueError("评测输出不得写入原始数据根目录")
    native_result = _evaluate(run, run.distmat, protocol)
    comparison = {"status": "not_comparable", "reason": "原始协议与请求协议不同"}
    if protocol == run.metrics["protocol"]:
        raw = _raw_metrics(native_result)
        differences = {
            key: abs(value - run.metrics[key]) for key, value in raw.items() if value is not None
        }
        comparison = {
            "status": "passed"
            if all(value < 0.001 for value in differences.values())
            else "failed",
            "upstream": {key: run.metrics[key] for key in raw},
            "unified_native": raw,
            "absolute_difference": differences,
            "absolute_tolerance_exclusive": 0.001,
        }
        if comparison["status"] == "failed":
            raise ValueError("原始指标同协议对拍失败：{}".format(differences))
    matrix = (
        run.distmat
        if distance == "native"
        else cosine_distance(run.query_features, run.gallery_features)
    )
    result = native_result if distance == "native" else _evaluate(run, matrix, protocol)
    metrics = {
        **run.metrics,
        **result.as_metrics(),
        "evaluation_source": "reid_unified",
        "protocol": protocol,
        "distance": distance,
        "distance_metric": run.metrics["distance_metric"] if distance == "native" else "cosine",
        "feature_normalization": run.metrics["feature_normalization"]
        if distance == "native"
        else "l2",
        "re_ranking": run.metrics["re_ranking"] if distance == "native" else False,
        "num_valid_query": result.num_valid_query,
        "num_skipped_query": result.num_skipped_query,
        "source_run": str(run.run_dir),
        "source_hashes": run.hashes,
        "source_verification": run.source_manifest_check,
        "cmc": [round(float(value), 6) for value in result.cmc],
    }
    verification = {
        "schema_version": 1,
        "source_run": str(run.run_dir),
        "source_hashes": run.hashes,
        "source_manifest": run.source_manifest_check,
        "same_matrix_comparison": comparison,
        "upstream_git": (run.run_dir / "git.txt").read_text(encoding="utf-8"),
        "upstream_environment": (run.run_dir / "env.txt").read_text(encoding="utf-8"),
        "evaluation_environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "sorting": "numpy.argsort(kind=quicksort)",
        "native": _raw_metrics(native_result),
        "evaluated": _raw_metrics(result),
    }
    if distance == "cosine":
        verification["native_cosine_order_different_queries"] = sum(
            not np.array_equal(
                np.argsort(left, kind="quicksort"), np.argsort(right, kind="quicksort")
            )
            for left, right in zip(run.distmat, matrix)
        )
        verification["native_cosine_absolute_difference"] = {
            key: abs(value - _raw_metrics(native_result)[key])
            for key, value in _raw_metrics(result).items()
            if value is not None
        }
    # 防止校验期间源产物被其他进程更改后仍发布 success。
    if run.hashes != {name: sha256(run.run_dir / name) for name in RUN_FILES}:
        raise ValueError("评测期间原运行产物发生变化")
    with new_directory(output) as temporary:
        if distance == "cosine":
            with (temporary / "distmat.npy").open("xb") as stream:
                np.save(stream, matrix, allow_pickle=False)
            metrics["derived_distmat_sha256"] = sha256(temporary / "distmat.npy")
            verification["derived_distmat_sha256"] = metrics["derived_distmat_sha256"]
        write_json(temporary / "metrics.json", metrics)
        write_json(temporary / "verification.json", verification)
        write_json(
            temporary / "status.json",
            {
                "status": "success",
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    return metrics
