from pathlib import Path

import pytest

from src.reid.manifest import (
    DatasetManifest,
    build_market1501_manifest,
    verify_manifest,
)

FILES = {
    "bounding_box_train/0002_c1s1_000451_03.jpg": (2, 1),
    "bounding_box_train/0003_c2s1_000451_03.jpg": (3, 2),
    "query/0002_c1s1_000451_03.jpg": (2, 1),
    "bounding_box_test/0002_c2s1_000451_03.jpg": (2, 2),
    "bounding_box_test/0003_c1s1_000451_03.jpg": (3, 1),
    "bounding_box_test/0000_c1s1_000000_00.jpg": (0, 1),  # distractor
    "bounding_box_test/-1_c1s1_000000_00.jpg": (-1, 1),  # junk
}


def _make_dataset(root: Path) -> Path:
    for name in FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return root


def test_market1501_parser_and_filtering(tmp_path):
    dataset_dir = _make_dataset(tmp_path / "market1501")
    manifest = build_market1501_manifest(dataset_dir)

    assert len(manifest.train) == 2
    assert len(manifest.query) == 1
    # junk is removed, distractor pid 0 is kept
    assert len(manifest.gallery) == 3
    assert {s.pid for s in manifest.gallery} == {0, 2, 3}
    assert manifest.query[0].path == "query/0002_c1s1_000451_03.jpg"
    assert manifest.query[0].pid == 2
    assert manifest.query[0].camid == 1
    assert manifest.stats()["query"] == {"images": 1, "pids": 1, "camids": [1]}


def test_parser_accepts_nested_market_dir(tmp_path):
    nested = _make_dataset(tmp_path / "Market-1501-v15.09.15")
    manifest = build_market1501_manifest(tmp_path)
    assert manifest.root == nested.resolve().as_posix()


def test_manifest_roundtrip(tmp_path):
    dataset_dir = _make_dataset(tmp_path / "market1501")
    manifest = build_market1501_manifest(dataset_dir)

    out = tmp_path / "manifests" / "market1501.json"
    manifest.save(out)
    loaded = DatasetManifest.load(out)

    assert loaded.name == manifest.name
    assert loaded.root == manifest.root
    assert loaded.query == manifest.query
    assert loaded.gallery == manifest.gallery
    assert loaded.train == manifest.train


def test_verify_manifest_reports_missing_cross_camera_match(tmp_path):
    root = tmp_path / "data"
    for name in [
        "query/0001_c1s1_000001_00.jpg",
        "bounding_box_test/0001_c1s1_000002_00.jpg",
    ]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    manifest = build_market1501_manifest(root)
    issues = verify_manifest(manifest)
    assert any("no cross-camera gallery match" in issue for issue in issues)


def test_verify_manifest_accepts_healthy_split(tmp_path):
    dataset_dir = _make_dataset(tmp_path / "market1501")
    manifest = build_market1501_manifest(dataset_dir)
    assert verify_manifest(manifest) == []


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Cannot find Market1501"):
        build_market1501_manifest(tmp_path)
