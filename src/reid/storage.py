"""清单和评测产物共用的路径、指纹及发布操作。"""

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator


def relative_path(value: str, location: str) -> str:
    """拒绝绝对路径、父目录及非 POSIX 路径，不静默改写输入。"""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("{}：非法 POSIX 相对路径 {}".format(location, value))
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or value == ".":
        raise ValueError("{}：路径越界或未规范化 {}".format(location, value))
    return value


def inside(root: Path, value: str, location: str) -> Path:
    path = (root / relative_path(value, location)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise ValueError("{}：路径越界 {}".format(location, value)) from None
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except json.JSONDecodeError as error:
        raise ValueError("{}:{} JSON 解析失败：{}".format(path, error.lineno, error.msg)) from error
    if not isinstance(value, dict):
        raise ValueError("{}：JSON 顶层须为对象".format(path))
    return value


def write_json(path: Path, value: Dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


@contextmanager
def new_directory(target: Path) -> Iterator[Path]:
    """同文件系统临时写入；只在全部写完后发布，并拒绝覆盖已有目录。"""
    target = target.absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".{}-".format(target.name), dir=str(target.parent)))
    try:
        yield temporary
        # mkdir 原子占位，防止并发发布覆盖空目录；失败时不动其他进程的目标。
        target.mkdir()
        try:
            os.replace(str(temporary), str(target))
        except BaseException:
            target.rmdir()
            raise
    finally:
        if temporary.exists():
            shutil.rmtree(str(temporary))
