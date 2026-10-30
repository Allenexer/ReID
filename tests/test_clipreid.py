"""CLIP-ReID 结果导出测试。"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.reid import clipreid
from src.reid.clipreid import create_run_dir, export_results


def sample_results(tmp_path: Path) -> dict:
    root = tmp_path / "data"
    return {
        "data_root": root,
        "records": [
            (str(root / "query" / "same.jpg"), 9, 0, 0),
            (str(root / "gallery" / "same.jpg"), 9, 1, 0),
            (str(root / "gallery" / "other.jpg"), 0, 2, 0),
        ],
        "num_query": 1,
        "query_features": np.array([[1.0, 0.0]], dtype=np.float32),
        "gallery_features": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "distmat": np.array([[0.0, 2.0]], dtype=np.float32),
        "cmc": np.ones(2, dtype=np.float32),
        "mean_ap": 0.897123456789,
    }


def test_exports_preserve_paths_order_and_metric_precision(tmp_path):
    output = create_run_dir(tmp_path / "run")
    metrics = export_results(output, **sample_results(tmp_path))

    with (output / "query.csv").open(newline="", encoding="utf-8") as stream:
        query = list(csv.DictReader(stream))
    with (output / "gallery.csv").open(newline="", encoding="utf-8") as stream:
        gallery = list(csv.DictReader(stream))

    assert query[0]["path"] == "query/same.jpg"
    assert [row["path"] for row in gallery] == ["gallery/same.jpg", "gallery/other.jpg"]
    assert [row["pid"] for row in gallery] == ["9", "0"]
    assert [row["camid"] for row in gallery] == ["1", "2"]
    assert [row["index"] for row in gallery] == ["0", "1"]
    np.testing.assert_array_equal(np.load(output / "distmat.npy"), [[0.0, 2.0]])
    np.testing.assert_array_equal(np.load(output / "query_features.npy"), [[1.0, 0.0]])
    np.testing.assert_array_equal(
        np.load(output / "gallery_features.npy"), [[1.0, 0.0], [0.0, 1.0]]
    )
    saved = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert saved == metrics
    assert saved["mAP"] == 0.897123456789
    assert saved["rank1"] == 1.0
    assert saved["rank5"] is None
    assert saved["distance_metric"] == "squared_euclidean"
    assert saved["evaluation_source"] == "clipreid_upstream"


@pytest.mark.parametrize("invalid", ["distance_shape", "feature_count", "nonfinite"])
def test_invalid_arrays_are_rejected_before_export(tmp_path, invalid):
    output = create_run_dir(tmp_path / "run")
    results = sample_results(tmp_path)
    if invalid == "distance_shape":
        results["distmat"] = np.zeros((2, 1))
    elif invalid == "feature_count":
        results["gallery_features"] = np.zeros((1, 2))
    else:
        results["distmat"][0, 0] = np.nan
    with pytest.raises(ValueError):
        export_results(output, **results)
    assert list(output.iterdir()) == []


def test_run_directory_never_overwrites_existing_results(tmp_path):
    output = create_run_dir(tmp_path / "run")
    marker = output / "existing.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        create_run_dir(output)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_saved_config_can_be_reloaded_by_yacs(tmp_path, monkeypatch):
    cfg_type = pytest.importorskip("yacs.config").CfgNode
    cfg = cfg_type(
        {
            "OUTPUT_DIR": str(tmp_path / "run"),
            "MODEL": {"DEVICE_ID": "0", "TRANSFORMER_TYPE": "None"},
        }
    )
    monkeypatch.setattr(clipreid, "environment_snapshot", lambda: "test environment")
    monkeypatch.setattr(clipreid, "git_snapshot", lambda root: "test version")
    monkeypatch.setattr(clipreid, "infer", lambda config, output: {})
    clipreid.run_experiment(tmp_path, cfg)

    restored = cfg.clone()
    restored.merge_from_file(str(tmp_path / "run" / "config.yaml"))
    assert restored.MODEL.DEVICE_ID == "0"
    assert restored.MODEL.TRANSFORMER_TYPE == "None"
    status = json.loads((tmp_path / "run" / "status.json").read_text())
    assert status["status"] == "success"


def test_failed_inference_preserves_error_and_metadata(tmp_path, monkeypatch):
    cfg_type = pytest.importorskip("yacs.config").CfgNode
    cfg = cfg_type({"OUTPUT_DIR": str(tmp_path / "failed-run")})
    monkeypatch.setattr(clipreid, "environment_snapshot", lambda: "test environment")
    monkeypatch.setattr(clipreid, "git_snapshot", lambda root: "test version")

    def fail(config, output):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(clipreid, "infer", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        clipreid.run_experiment(tmp_path, cfg)
    output = tmp_path / "failed-run"
    assert (output / "config.yaml").is_file()
    assert (output / "env.txt").is_file()
    assert (output / "git.txt").is_file()
    assert "inference failed" in (output / "error.txt").read_text()
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    assert not (output / "metrics.json").exists()
