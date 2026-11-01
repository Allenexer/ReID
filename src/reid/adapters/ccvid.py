"""CCVID 序列检索适配。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from yacs.config import CfgNode


@dataclass(frozen=True)
class CCVIDSample:
    """视频序列及所选帧。"""

    frame_paths: tuple[Path, ...]
    sequence: str
    pid: int
    camid: int
    clothes_label: str
    num_frames: int

    @property
    def frame_path(self) -> Path:
        """首个选中帧，用于兼容单帧清单。"""
        return self.frame_paths[0]

    def as_record(self) -> tuple[str, int, int, int]:
        """转换为 CLIP-ReID 图像记录。"""
        return str(self.frame_path), self.pid, self.camid, 0


def select_frame_indices(frame_count: int, num_frames: int, frame_index: int = 0) -> list[int]:
    """均匀选取不重复的帧索引，包含首尾。"""
    if frame_count <= 0 or num_frames <= 0 or not 0 <= frame_index < frame_count:
        raise ValueError("帧数须为正数，frame_index 须在序列范围内")
    if num_frames == 1:
        return [frame_index]
    if frame_index != 0:
        raise ValueError("均匀抽帧要求 FRAME_INDEX=0")
    return np.linspace(0, frame_count - 1, min(frame_count, num_frames), dtype=int).tolist()


def mean_pool_features(features: np.ndarray, counts: list[int]) -> np.ndarray:
    """按序列帧数分组求均值。"""
    if (
        features.ndim != 2
        or not counts
        or any(count <= 0 for count in counts)
        or sum(counts) != len(features)
    ):
        raise ValueError("序列帧数与特征行数不一致")
    offsets = np.cumsum([0, *counts])
    return np.stack(
        [
            features[start:end].mean(axis=0)
            for start, end in zip(offsets[:-1], offsets[1:], strict=True)
        ]
    )


def load_ccvid_split(
    root: Path, split: str, frame_index: int = 0, num_frames: int = 1
) -> list[CCVIDSample]:
    """按划分文件顺序选帧；session3 的 camid 偏移 12。"""
    if split not in {"train", "query", "gallery"} or frame_index < 0 or num_frames <= 0:
        raise ValueError("划分须为 train/query/gallery，frame_index 须非负，num_frames 须为正数")
    root = root.resolve()
    samples = []
    seen = set()
    split_path = root / f"{split}.txt"
    for line_number, line in enumerate(split_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 3:
            raise ValueError(f"{split_path}:{line_number} 应包含序列路径、身份和衣服标签")
        sequence, pid_text, clothes = fields
        match = re.fullmatch(r"(session[123])/(\d+)_(\d+)", sequence)
        if match is None or int(match[2]) != int(pid_text):
            raise ValueError(f"无效序列路径或身份：{sequence}, {pid_text}")
        if sequence in seen:
            raise ValueError(f"重复序列：{sequence}")
        seen.add(sequence)
        directory = (root / sequence).resolve()
        directory.relative_to(root)
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        frames = sorted(directory.glob("*.jpg"))
        if not frames:
            raise ValueError(f"序列没有 JPG：{sequence}")
        if frame_index >= len(frames):
            raise ValueError(f"frame_index={frame_index} 超出 {sequence} 的 {len(frames)} 帧")
        indices = select_frame_indices(len(frames), num_frames, frame_index)
        camid = int(match[3]) + (12 if match[1] == "session3" else 0)
        samples.append(
            CCVIDSample(
                frame_paths=tuple(frames[index] for index in indices),
                sequence=sequence,
                pid=int(pid_text),
                camid=camid,
                clothes_label=clothes,
                num_frames=len(frames),
            )
        )
    if not samples:
        raise ValueError(f"空划分：{split_path}")
    return samples


def make_ccvid_loader(cfg: CfgNode) -> tuple[DataLoader, int, list[CCVIDSample]]:
    """按 query、gallery、序列、帧的顺序构造验证集。"""
    from torch.utils.data import DataLoader
    from torchvision import transforms as T

    image_dataset = import_module("datasets.bases").ImageDataset
    collate = import_module("datasets.make_dataloader_clipreid").val_collate_fn
    root = Path(cfg.DATASETS.ROOT_DIR)
    query = load_ccvid_split(root, "query", cfg.DATASETS.FRAME_INDEX, cfg.DATASETS.NUM_FRAMES)
    gallery = load_ccvid_split(root, "gallery", cfg.DATASETS.FRAME_INDEX, cfg.DATASETS.NUM_FRAMES)
    samples = query + gallery
    transform = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )
    records = [
        (str(path), sample.pid, sample.camid, 0)
        for sample in samples
        for path in sample.frame_paths
    ]
    dataset = image_dataset(records, transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=collate,
    )
    return loader, len(query), samples
