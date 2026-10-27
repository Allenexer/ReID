"""Dataset manifests for person re-identification.

A manifest is the stable interface between data preparation and every
baseline/evaluator: ordered lists of ``(relative path, pid, camid)`` for
the query, gallery and (optional) train splits.

Paths are POSIX paths relative to ``DatasetManifest.root`` so a manifest
can be regenerated on another machine. The file itself is generated (not
committed); see ``docs/DATA.md`` for the commands.

Only the standard library is used here, so this module runs before the
ML environment is installed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Market1501 filename examples:
#   0002_c1s1_000451_03.jpg
#   -1_c1s1_000000_00.jpg   (junk)
#   0000_c3s1_000000_00.jpg (distractor)
MARKET_FILENAME_RE = re.compile(
    r"^(?P<pid>-?\d+)_c(?P<cam>\d+)s(?P<seq>\d+)_"
    r"(?P<frame>\d+)_(?P<det>\d+)\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Sample:
    """One image in a split."""

    path: str  # POSIX path relative to DatasetManifest.root
    pid: int
    camid: int

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "pid": self.pid, "camid": self.camid}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sample":
        return cls(
            path=str(data["path"]),
            pid=int(data["pid"]),
            camid=int(data["camid"]),
        )


@dataclass
class DatasetManifest:
    """Query/gallery/train splits plus provenance."""

    name: str
    root: str
    query: list[Sample]
    gallery: list[Sample]
    train: list[Sample] = field(default_factory=list)
    protocol: str = ""
    notes: str = ""

    def stats(self) -> dict[str, Any]:
        def split_stats(samples: list[Sample]) -> dict[str, Any]:
            return {
                "images": len(samples),
                "pids": len({s.pid for s in samples}),
                "camids": sorted({s.camid for s in samples}),
            }

        return {
            "name": self.name,
            "root": self.root,
            "protocol": self.protocol,
            "train": split_stats(self.train),
            "query": split_stats(self.query),
            "gallery": split_stats(self.gallery),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root": self.root,
            "protocol": self.protocol,
            "notes": self.notes,
            "train": [s.to_dict() for s in self.train],
            "query": [s.to_dict() for s in self.query],
            "gallery": [s.to_dict() for s in self.gallery],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        path.write_text(text + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=str(data["name"]),
            root=str(data["root"]),
            query=[Sample.from_dict(x) for x in data["query"]],
            gallery=[Sample.from_dict(x) for x in data["gallery"]],
            train=[Sample.from_dict(x) for x in data.get("train", [])],
            protocol=str(data.get("protocol", "")),
            notes=str(data.get("notes", "")),
        )


def _resolve_market_dir(root: str | Path) -> Path:
    """Accept either the dataset dir itself or a common parent."""
    root = Path(root)
    candidates = [root, root / "market1501", root / "Market-1501-v15.09.15"]
    for candidate in candidates:
        if (candidate / "query").is_dir() and (candidate / "bounding_box_test").is_dir():
            return candidate
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Cannot find Market1501 under {root}. Expected query/ and "
        f"bounding_box_test/ in one of: {tried}"
    )


def _parse_market_split(split_dir: Path, dataset_dir: Path) -> list[Sample]:
    if not split_dir.is_dir():
        return []

    samples: list[Sample] = []
    for image_path in sorted(split_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        match = MARKET_FILENAME_RE.match(image_path.name)
        if match is None:
            continue
        pid = int(match.group("pid"))
        if pid == -1:  # junk; keep pid 0 distractors like the upstream code
            continue
        samples.append(
            Sample(
                path=image_path.relative_to(dataset_dir).as_posix(),
                pid=pid,
                camid=int(match.group("cam")),
            )
        )
    return samples


def build_market1501_manifest(
    root: str | Path, name: str = "market1501"
) -> DatasetManifest:
    """Build a manifest from the official Market1501 directory layout."""
    dataset_dir = _resolve_market_dir(root)
    return DatasetManifest(
        name=name,
        root=dataset_dir.resolve().as_posix(),
        query=_parse_market_split(dataset_dir / "query", dataset_dir),
        gallery=_parse_market_split(dataset_dir / "bounding_box_test", dataset_dir),
        train=_parse_market_split(dataset_dir / "bounding_box_train", dataset_dir),
        protocol="market1501",
        notes=(
            "Official split; junk (-1) removed, distractors (0) kept in "
            "gallery to match the upstream evaluation protocol."
        ),
    )


def verify_manifest(manifest: DatasetManifest) -> list[str]:
    """Return human-readable issues; an empty list means no known problem."""
    issues: list[str] = []
    if not manifest.query:
        issues.append("query split is empty")
    if not manifest.gallery:
        issues.append("gallery split is empty")
    if issues:
        return issues

    gallery_cams: dict[int, set[int]] = {}
    for sample in manifest.gallery:
        gallery_cams.setdefault(sample.pid, set()).add(sample.camid)

    missing = sorted(
        {
            sample.pid
            for sample in manifest.query
            if sample.pid not in gallery_cams
            or gallery_cams[sample.pid] <= {sample.camid}
        }
    )
    if missing:
        issues.append(
            f"{len(missing)} query pid(s) have no cross-camera gallery match, "
            f"e.g. {missing[:5]}"
        )
    return issues
