"""与模型无关的数据清单、CSV 存储和严格校验。"""

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

from .storage import inside, read_json, relative_path, sha256, write_json

BASE_FIELDS = ("index", "path", "pid", "camid", "viewid")
CCVID_FIELDS = ("sequence", "clothes_label", "num_frames", "sampled_frames", "frame_paths")
EXPECTED = {
    "market1501": {
        "train": {"samples": 12936, "identities": 751},
        "query": {"samples": 3368, "identities": 750},
        "gallery": {"samples": 15913},
    },
    "ccvid": {
        "train": {"samples": 948, "identities": 75, "frames": 118613},
        "query": {"samples": 834, "identities": 151, "frames": 116799},
        "gallery": {"samples": 1074, "identities": 151, "frames": 112421},
    },
}


@dataclass(frozen=True)
class Sample:
    index: int
    path: str
    pid: int
    camid: Optional[int]
    viewid: int = 0
    sequence: Optional[str] = None
    clothes_label: Optional[str] = None
    num_frames: Optional[int] = None
    sampled_frames: Optional[int] = None
    frame_paths: Tuple[str, ...] = ()


@dataclass
class Manifest:
    dataset: str
    protocol: str
    sample_unit: str
    data_root: Path
    splits: Dict[str, List[Sample]]
    sorting: str
    sources: Dict[str, Any] = field(default_factory=dict)
    expected: Dict[str, Dict[str, int]] = field(default_factory=dict)
    schema_version: int = 1


@dataclass
class ValidationReport:
    errors: List[str]
    statistics: Dict[str, Dict[str, int]]
    verified_images: int = 0

    @property
    def valid(self) -> bool:
        return not self.errors

    def require_valid(self) -> None:
        if self.errors:
            raise ValueError("数据校验失败：\n" + "\n".join(self.errors))

    def as_dict(self) -> Dict[str, Any]:
        return {"valid": self.valid, **asdict(self)}


def build_manifest(dataset: str, root: Path, labels: Optional[Path] = None) -> Manifest:
    """读取官方划分；完整规模检查由 validate_manifest 执行。"""
    root = Path(root).resolve()
    sources = {"created_at": datetime.now(timezone.utc).isoformat(), "split_version": "official"}
    if dataset == "market1501":
        from .adapters.market1501 import load_market1501

        splits = load_market1501(root)
        protocol, unit, sorting = "market1501", "image", "jpg_path_lexicographic"
        sources["url"] = "https://www.kaggle.com/datasets/pengcw1/market-1501"
        sources["inventory_sha256"] = hashlib.sha256(
            "\n".join(
                "{}\t{}".format(row.path, inside(root, row.path, split).stat().st_size)
                for split, rows in splits.items()
                for row in rows
            ).encode("utf-8")
        ).hexdigest()
    elif dataset == "ccvid":
        # 使用 C 已实现的读取和抽帧规则，不另造序列采样器。
        from .adapters.ccvid import load_ccvid_split

        splits = {}
        for split in ("train", "query", "gallery"):
            splits[split] = [
                Sample(
                    index=index,
                    path=row.frame_path.relative_to(root).as_posix(),
                    pid=row.pid,
                    camid=row.camid,
                    sequence=row.sequence,
                    clothes_label=row.clothes_label,
                    num_frames=row.num_frames,
                    sampled_frames=len(row.frame_paths),
                    frame_paths=tuple(
                        path.relative_to(root).as_posix() for path in row.frame_paths
                    ),
                )
                for index, row in enumerate(load_ccvid_split(root, split))
            ]
        protocol, unit, sorting = "ccvid_general", "sequence", "split_file_order"
        sources["url"] = "https://github.com/guxinqian/Simple-CCReID"
        sources["split_sha256"] = {split: sha256(root / (split + ".txt")) for split in splits}
    elif dataset == "move":
        from .adapters.move import load_move

        if labels is None:
            raise ValueError("Move 必须提供 --labels 标准标签表")
        splits = load_move(root, Path(labels))
        protocol, unit, sorting = "pending", "image", "label_file_order"
        sources.update({"labels_sha256": sha256(Path(labels)), "split_version": "course_pending"})
    else:
        raise ValueError("不支持的数据集：{}".format(dataset))
    return Manifest(
        dataset, protocol, unit, root, splits, sorting, sources, EXPECTED.get(dataset, {})
    )


def validate_sample(row: Sample, dataset: str, location: str) -> None:
    relative_path(row.path, location)
    for name, value in (("index", row.index), ("pid", row.pid), ("viewid", row.viewid)):
        if type(value) is not int or value < 0:
            raise ValueError("{}：{} 须为非负整数".format(location, name))
    if row.camid is None and dataset != "move":
        raise ValueError("{}：缺少 camid".format(location))
    if row.camid is not None and (type(row.camid) is not int or row.camid < 0):
        raise ValueError("{}：camid 须为非负整数".format(location))
    if dataset == "market1501" and (row.pid > 1501 or row.camid > 5):
        raise ValueError("{}：Market1501 标签越界".format(location))
    if dataset != "ccvid":
        return
    if not row.sequence or not row.clothes_label or not row.clothes_label.strip():
        raise ValueError("{}：缺少 sequence/clothes_label".format(location))
    relative_path(row.sequence, location)
    match = re.fullmatch(r"(session[123])/(\d+)_(\d+)", row.sequence)
    if match is None or int(match[2]) != row.pid:
        raise ValueError("{}：序列路径与 pid 不一致".format(location))
    expected_camid = int(match[3]) + (12 if match[1] == "session3" else 0)
    if row.camid != expected_camid:
        raise ValueError("{}：序列与 camid 不一致".format(location))
    if (
        type(row.num_frames) is not int
        or type(row.sampled_frames) is not int
        or not 0 < row.sampled_frames <= row.num_frames
        or len(row.frame_paths) != row.sampled_frames
        or len(set(row.frame_paths)) != row.sampled_frames
        or row.path != row.frame_paths[0]
        or list(row.frame_paths) != sorted(row.frame_paths)
    ):
        raise ValueError("{}：序列帧数或实际采样不一致".format(location))
    for frame in row.frame_paths:
        relative_path(frame, location)
        if PurePosixPath(frame).parent.as_posix() != row.sequence:
            raise ValueError("{}：采样帧越过序列边界 {}".format(location, frame))


def validate_manifest(manifest: Manifest, verify_images: bool = False) -> ValidationReport:
    """检查规模、身份泄漏、全部路径；显式选择时全量解码图片。"""
    report = ValidationReport([], {})
    expected_splits = (
        {"query", "gallery"} if manifest.dataset == "move" else {"train", "query", "gallery"}
    )
    if manifest.dataset not in {"market1501", "ccvid", "move"}:
        report.errors.append("非法数据集：{}".format(manifest.dataset))
    expected_protocol = {"market1501": "market1501", "ccvid": "ccvid_general", "move": "pending"}
    if manifest.protocol != expected_protocol.get(manifest.dataset):
        report.errors.append("数据集与清单协议不一致")
    if manifest.schema_version != 1 or manifest.sample_unit != (
        "sequence" if manifest.dataset == "ccvid" else "image"
    ):
        report.errors.append("schema_version/sample_unit 不受支持")
    if set(manifest.splits) != expected_splits:
        report.errors.append("划分须为 {}".format(sorted(expected_splits)))
    seen = set()
    identities = {}
    for split, rows in manifest.splits.items():
        identities[split] = {row.pid for row in rows}
        statistics = {"samples": len(rows), "identities": len(identities[split])}
        if manifest.dataset == "ccvid":
            statistics["frames"] = sum(row.num_frames or 0 for row in rows)
        report.statistics[split] = statistics
        if not rows:
            report.errors.append("{}：空划分".format(split))
        for key, expected in manifest.expected.get(split, {}).items():
            if statistics[key] != expected:
                report.errors.append(
                    "{}：{} 期望 {}，实际 {}".format(split, key, expected, statistics[key])
                )
        for index, row in enumerate(rows):
            location = "{}:{}".format(split, index + 2)
            try:
                validate_sample(row, manifest.dataset, location)
                if row.index != index:
                    raise ValueError("{}：index 须连续，期望 {}".format(location, index))
                key = row.sequence if manifest.dataset == "ccvid" else row.path
                path = inside(manifest.data_root, row.path, location)
                resolved_key = inside(manifest.data_root, key, location)
                if resolved_key in seen:
                    raise ValueError("{}：重复样本或划分泄漏 {}".format(location, key))
                seen.add(resolved_key)
                if not path.is_file():
                    raise FileNotFoundError(path)
                paths = [path]
                if manifest.dataset == "ccvid":
                    directory = inside(manifest.data_root, row.sequence, location)
                    paths = sorted(directory.glob("*.jpg"))
                    if len(paths) != row.num_frames:
                        raise ValueError("{}：原序列帧数不一致".format(location))
                    for frame in row.frame_paths:
                        if not inside(manifest.data_root, frame, location).is_file():
                            raise FileNotFoundError(frame)
                for image in paths:
                    image.resolve().relative_to(manifest.data_root.resolve())
                    if verify_images:
                        from PIL import Image

                        with Image.open(image) as opened:
                            opened.load()
                        report.verified_images += 1
            except (ValueError, OSError) as error:
                report.errors.append("{}：{}".format(location, error))
    train = identities.get("train", set())
    if train & (identities.get("query", set()) | identities.get("gallery", set())):
        report.errors.append("train 与 query/gallery 身份泄漏")
    query_ids, gallery_ids = identities.get("query", set()), identities.get("gallery", set())
    if not query_ids or not query_ids <= gallery_ids:
        report.errors.append("query 身份须包含于 gallery 身份集合")
    if manifest.dataset == "ccvid" and query_ids != gallery_ids:
        report.errors.append("CCVID query/gallery 身份集合须一致")
    return report


def read_samples(path: Path, dataset: str) -> List[Sample]:
    """严格读取生产者 CSV，不排序、不重编号、不按 basename 匹配。"""
    fields = BASE_FIELDS + (CCVID_FIELDS if dataset == "ccvid" else ())
    rows = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(fields):
            raise ValueError("{}:1 CSV 表头须为 {}".format(path, fields))
        for number, values in enumerate(reader, 2):
            location = "{}:{}".format(path, number)
            try:
                if None in values or any(value is None for value in values.values()):
                    raise ValueError("CSV 列数不一致")
                frames = json.loads(values["frame_paths"]) if dataset == "ccvid" else []
                if not isinstance(frames, list) or any(not isinstance(p, str) for p in frames):
                    raise ValueError("frame_paths 须为 JSON 字符串列表")
                row = Sample(
                    index=int(values["index"]),
                    path=values["path"],
                    pid=int(values["pid"]),
                    camid=int(values["camid"]) if values["camid"] else None,
                    viewid=int(values["viewid"]),
                    sequence=values["sequence"] if dataset == "ccvid" else None,
                    clothes_label=values["clothes_label"] if dataset == "ccvid" else None,
                    num_frames=int(values["num_frames"]) if dataset == "ccvid" else None,
                    sampled_frames=int(values["sampled_frames"]) if dataset == "ccvid" else None,
                    frame_paths=tuple(frames),
                )
                validate_sample(row, dataset, location)
                if row.index != len(rows):
                    raise ValueError("index 须从 0 连续递增，期望 {}".format(len(rows)))
                rows.append(row)
            except (ValueError, TypeError) as error:
                raise ValueError("{}：{}".format(location, error)) from error
    if not rows:
        raise ValueError("{}：空划分".format(path))
    keys = [row.sequence if dataset == "ccvid" else row.path for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("{}：重复样本".format(path))
    return rows


def _write_samples(path: Path, rows: List[Sample], dataset: str) -> None:
    fields = BASE_FIELDS + (CCVID_FIELDS if dataset == "ccvid" else ())
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            values = asdict(row)
            values["frame_paths"] = json.dumps(row.frame_paths, ensure_ascii=False)
            writer.writerow({name: values[name] for name in fields})


def save_manifest(manifest: Manifest, path: Path, verify_images: bool = False) -> None:
    """校验后写 CSV、验证报告，最后发布索引；拒绝覆盖任何已有目标。"""
    path = Path(path).absolute()
    if (
        manifest.data_root.resolve() == path.resolve()
        or manifest.data_root.resolve() in path.resolve().parents
    ):
        raise ValueError("manifest 输出不得写入原始数据根目录")
    directory = path.with_suffix("")
    validation = path.with_suffix(".validation.json")
    if path.suffix != ".json" or directory == path:
        raise ValueError("manifest 索引须为 .json：{}".format(path))
    for target in (path, directory, validation):
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
    report = validate_manifest(manifest, verify_images)
    report.require_valid()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".manifest-", dir=str(path.parent)))
    published = []
    try:
        csv_dir = temporary / directory.name
        csv_dir.mkdir()
        splits = {}
        for split, rows in manifest.splits.items():
            csv_path = csv_dir / (split + ".csv")
            _write_samples(csv_path, rows, manifest.dataset)
            splits[split] = {
                "csv": "{}/{}.csv".format(directory.name, split),
                "sha256": sha256(csv_path),
                **report.statistics[split],
            }
        index = {
            "schema_version": 1,
            "dataset": manifest.dataset,
            "protocol": manifest.protocol,
            "sample_unit": manifest.sample_unit,
            "data_root": Path(os.path.relpath(manifest.data_root, path.parent)).as_posix(),
            "sorting": manifest.sorting,
            "sources": manifest.sources,
            "expected": manifest.expected,
            "splits": splits,
        }
        write_json(temporary / path.name, index)
        write_json(temporary / validation.name, report.as_dict())
        directory.mkdir()
        published.append(directory)
        os.replace(str(csv_dir), str(directory))
        for target in (validation, path):
            os.link(str(temporary / target.name), str(target))
            published.append(target)
    except BaseException:
        for target in reversed(published):
            if target.is_dir():
                shutil.rmtree(str(target))
            else:
                target.unlink()
        raise
    finally:
        shutil.rmtree(str(temporary))


def load_manifest(path: Path) -> Manifest:
    """读取索引并验证 CSV 指纹；不强制离线机器具有原图。"""
    path = Path(path).resolve()
    data = read_json(path)
    required = {
        "schema_version",
        "dataset",
        "protocol",
        "sample_unit",
        "data_root",
        "sorting",
        "sources",
        "expected",
        "splits",
    }
    if not required <= data.keys() or data["schema_version"] != 1:
        raise ValueError("{}：不支持的 manifest schema".format(path))
    if not isinstance(data["data_root"], str) or PurePosixPath(data["data_root"]).is_absolute():
        raise ValueError("{}：data_root 须为相对索引的路径".format(path))
    if "\\" in data["data_root"] or not data["data_root"]:
        raise ValueError("{}：data_root 须为非空 POSIX 路径".format(path))
    dataset = data["dataset"]
    allowed_splits = {"query", "gallery"} if dataset == "move" else {"train", "query", "gallery"}
    if dataset not in {"market1501", "ccvid", "move"} or set(data["splits"]) != allowed_splits:
        raise ValueError("{}：数据集或划分不合法".format(path))
    splits = {}
    for split, metadata in data["splits"].items():
        csv_path = inside(path.parent, metadata["csv"], str(path))
        if sha256(csv_path) != metadata["sha256"]:
            raise ValueError("{}：CSV SHA256 不一致".format(csv_path))
        splits[split] = read_samples(csv_path, data["dataset"])
        if len(splits[split]) != metadata["samples"]:
            raise ValueError("{}：CSV 规模不一致".format(csv_path))
    return Manifest(
        data["dataset"],
        data["protocol"],
        data["sample_unit"],
        (path.parent / data["data_root"]).resolve(),
        splits,
        data["sorting"],
        data["sources"],
        data["expected"],
    )
