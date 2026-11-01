"""CLIP-ReID 结果导出测试。"""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

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


def test_ccvid_exports_preserve_sequence_and_clothes(tmp_path):
    output = create_run_dir(tmp_path / "ccvid-run")
    metrics = export_results(
        output,
        **sample_results(tmp_path),
        dataset="ccvid",
        extra_columns={
            "sequence": ["session1/009_01", "session3/009_01", "session1/010_01"],
            "clothes_label": ["u0_l0_s0_c0_a0", "u1_l0_s0_c0_a0", "u0_l0_s0_c0_a0"],
            "num_frames": [30, 40, 50],
        },
    )
    with (output / "gallery.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert metrics["dataset"] == "ccvid"
    assert metrics["protocol"] == "ccvid_general"
    assert metrics["sample_unit"] == "sequence"
    assert rows[0]["sequence"] == "session3/009_01"
    assert rows[0]["clothes_label"] == "u1_l0_s0_c0_a0"
    assert rows[0]["num_frames"] == "40"


@pytest.mark.parametrize("columns", [{"clothes_label": ["one"]}, {"pid": [1, 2, 3]}])
def test_invalid_extra_columns_are_rejected_before_export(tmp_path, columns):
    output = create_run_dir(tmp_path / "run")
    with pytest.raises(ValueError):
        export_results(output, **sample_results(tmp_path), extra_columns=columns)
    assert not list(output.iterdir())


def test_multiframe_exports_record_sampling_aggregation_and_time(tmp_path):
    output = create_run_dir(tmp_path / "run")
    metrics = export_results(
        output,
        **sample_results(tmp_path),
        dataset="ccvid",
        frames_per_sequence=8,
        inference_seconds=12.5,
        extra_columns={
            "sampled_frames": [2, 2, 2],
            "frame_paths": ['["a.jpg", "b.jpg"]'] * 3,
        },
    )
    assert metrics["frames_per_sequence"] == 8
    assert metrics["frame_sampling"] == "uniform"
    assert metrics["aggregation"] == "mean_then_l2"
    assert metrics["inference_seconds"] == 12.5
    with (output / "query.csv").open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["sampled_frames"] == "2"
    assert json.loads(row["frame_paths"]) == ["a.jpg", "b.jpg"]


@pytest.fixture
def config_fixture(tmp_path, monkeypatch):
    cfg_type = pytest.importorskip("yacs.config").CfgNode
    upstream = tmp_path / "third_party" / "CLIP-ReID"
    upstream.mkdir(parents=True)
    (upstream / "test_clipreid.py").touch()
    data = tmp_path / "data"
    data.mkdir()
    (data / "query.txt").touch()
    (data / "gallery.txt").touch()
    weight = tmp_path / "model.pth"
    weight.touch()
    defaults = cfg_type(
        {
            "MODEL": {"SIE_CAMERA": False, "SIE_VIEW": False},
            "DATASETS": {"NAMES": "ccvid", "ROOT_DIR": str(data)},
            "TEST": {
                "IMS_PER_BATCH": 8,
                "RE_RANKING": False,
                "FEAT_NORM": "yes",
                "WEIGHT": str(weight),
            },
            "SOLVER": {"SEED": 1234},
            "OUTPUT_DIR": "",
        }
    )
    monkeypatch.setattr(clipreid, "import_module", lambda name: SimpleNamespace(cfg=defaults))
    config = tmp_path / "config.yml"
    config.write_text("MODEL:\n  CHECKPOINT_CLASSES: 751\n", encoding="utf-8")
    return tmp_path, config, defaults


def test_frames_override_preserves_test_only_checkpoint_classes(config_fixture):
    root, config, _ = config_fixture
    cfg = clipreid.load_run_config(root, config, root / "run", frames=8)
    assert cfg.DATASETS.NUM_FRAMES == 8
    assert cfg.MODEL.CHECKPOINT_CLASSES == 751
    assert not (root / "run").exists()
    assert not (root / "data" / "train.txt").exists()


@pytest.mark.parametrize(
    "frames,dataset,frame_index",
    [(0, "ccvid", 0), (-1, "ccvid", 0), (4, "market1501", 0), (4, "ccvid", 1)],
)
def test_invalid_frames_configuration_is_rejected(config_fixture, frames, dataset, frame_index):
    root, config, defaults = config_fixture
    defaults.DATASETS.NAMES = dataset
    config.write_text(
        f"MODEL:\n  CHECKPOINT_CLASSES: 751\nDATASETS:\n  FRAME_INDEX: {frame_index}\n"
    )
    with pytest.raises(ValueError):
        clipreid.load_run_config(root, config, root / "run", frames=frames)


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
