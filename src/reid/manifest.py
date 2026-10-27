"""Dataset manifests for person re-identification.

A manifest is the stable interface between data preparation and every
baseline/evaluator: ordered lists of ``(relative path, pid, camid)`` for
the query, gallery and (optional) train splits.

The JSON schema and the ``Sample`` / ``DatasetManifest`` containers are
shared contracts. Dataset-specific parsing and validation are left to
member A; see the TODO notes in the two module-level functions.

Paths are POSIX paths relative to ``DatasetManifest.root`` so a manifest
can be regenerated on another machine. The file itself is generated (not
committed); see ``docs/DATA.md`` for the intended commands.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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


def build_market1501_manifest(
    root: str | Path, name: str = "market1501"
) -> DatasetManifest:
    """Build a manifest from the official Market1501 directory layout.

    TODO(A): implement this function. Requirements:

    - ``root`` may be the dataset directory itself or a common parent
      containing ``market1501/`` or ``Market-1501-v15.09.15/``.
    - Read ``bounding_box_train/``, ``query/`` and ``bounding_box_test/``.
    - Parse the filename format
      ``{pid}_c{cam}s{seq}_{frame}_{det}.jpg``.
    - Drop junk samples with ``pid == -1``.
    - Keep distractor samples with ``pid == 0`` in the gallery, matching
      the upstream evaluation protocol.
    - Store POSIX paths relative to the resolved dataset directory.
    """
    raise NotImplementedError("A: implement build_market1501_manifest")


def verify_manifest(manifest: DatasetManifest) -> list[str]:
    """Return human-readable issues; an empty list means no known problem.

    TODO(A): implement this function. At minimum check:

    - ``query`` and ``gallery`` are not empty;
    - every query identity has at least one gallery match from a
      different camera.
    """
    raise NotImplementedError("A: implement verify_manifest")
