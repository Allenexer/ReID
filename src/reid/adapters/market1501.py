"""Market1501 原始身份清单，不修改数据目录。"""

import re
from pathlib import Path
from typing import Dict, List

from ..manifest import Sample
from ..storage import inside, sha256


def load_market1501(root: Path) -> Dict[str, List[Sample]]:
    candidates = [root / name for name in ("market1501", "Market-1501-v15.09.15")]
    existing = [path for path in candidates if path.is_dir()]
    if not existing:
        raise FileNotFoundError("{} 下缺少 market1501/ 或 Market-1501-v15.09.15/".format(root))
    if len(existing) == 2 and existing[0].resolve() != existing[1].resolve():
        inventories = [
            {
                path.relative_to(folder).as_posix(): path
                for name in ("bounding_box_train", "query", "bounding_box_test")
                for path in (folder / name).glob("*.jpg")
            }
            for folder in existing
        ]
        if inventories[0].keys() != inventories[1].keys() or any(
            sha256(inventories[0][name]) != sha256(inventories[1][name]) for name in inventories[0]
        ):
            raise ValueError("{}：两个 Market1501 目录有歧义，数据不同".format(root))
    # 两份内容相同时选官方目录，便于与 C 的固定入口交接。
    directory = next(
        (path for path in existing if path.name == "Market-1501-v15.09.15"), existing[0]
    )
    splits = {}
    for split, name in (
        ("train", "bounding_box_train"),
        ("query", "query"),
        ("gallery", "bounding_box_test"),
    ):
        source = directory / name
        if not source.is_dir():
            raise FileNotFoundError(source)
        rows = []
        for image in sorted(source.glob("*.jpg")):
            # 与固定上游一致解析标签前缀；保留来源包中带 .jpg.jpg 的真实路径。
            match = re.match(r"(-?\d+)_c(\d)", image.name)
            if match is None:
                raise ValueError("{}：非法 Market1501 文件名".format(image))
            pid, camid = int(match[1]), int(match[2])
            if pid == -1:
                continue
            if not 0 <= pid <= 1501 or not 1 <= camid <= 6:
                raise ValueError("{}：pid/camid 越界".format(image))
            path = image.relative_to(root).as_posix()
            inside(root, path, str(image))
            rows.append(Sample(len(rows), path, pid, camid - 1))
        if not rows:
            raise ValueError("{}：空划分".format(source))
        splits[split] = rows
    return splits
