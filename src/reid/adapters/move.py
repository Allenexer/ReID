"""仅接收课程划分的 Move 标准表，不构造训练集和伪相机。"""

import csv
from pathlib import Path
from typing import Dict, List

from ..manifest import Sample, validate_sample
from ..storage import inside


def load_move(root: Path, labels: Path) -> Dict[str, List[Sample]]:
    splits = {"query": [], "gallery": []}
    with labels.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["split", "path", "pid", "camid"]:
            raise ValueError("{}:1 标签列须为 split,path,pid,camid".format(labels))
        for number, values in enumerate(reader, 2):
            location = "{}:{}".format(labels, number)
            try:
                if None in values or any(value is None for value in values.values()):
                    raise ValueError("标签列数不一致")
                split = values["split"]
                if split not in splits:
                    raise ValueError("Move 只接受 query/gallery")
                row = Sample(
                    len(splits[split]),
                    values["path"],
                    int(values["pid"]),
                    int(values["camid"]) if values["camid"] else None,
                )
                validate_sample(row, "move", location)
                inside(root, row.path, location)
                splits[split].append(row)
            except (ValueError, TypeError) as error:
                raise ValueError("{}：{}".format(location, error)) from error
    return splits
