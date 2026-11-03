"""TransReID 结果归档测试。"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.reid.transreid import create_run_dir, dump_config, export_results


def sample_results(tmp_path: Path) -> dict:
    root = tmp_path / "data"
    return {
        "data_root": root,
        "records": [
            (str(root / "query" / "0001_c1.jpg"), 1, 0, 1),
            (str(root / "gallery" / "0001_c2.jpg"), 1, 1, 1),
            (str(root / "gallery" / "0002_c1.jpg"), 2, 0, 1),
        ],
        "num_query": 1,
        "query_features": np.array([[1.0, 0.0]], dtype=np.float32),
        "gallery_features": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "distmat": np.array([[0.0, 2.0]], dtype=np.float32),
        "cmc": np.ones(2, dtype=np.float32),
        "mean_ap": 0.87123456789,
    }


def test_exports_transreid_schema_and_viewid(tmp_path):
    output = create_run_dir(tmp_path / "run")
    metrics = export_results(output, **sample_results(tmp_path))
    saved = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert saved == metrics
    assert saved["model"] == "transreid"
    assert saved["evaluation_source"] == "transreid_upstream"
    assert saved["distance_metric"] == "squared_euclidean"
    assert (output / "query.csv").is_file()
    assert np.load(output / "query_features.npy").dtype == np.float32


def test_transreid_rejects_other_viewid(tmp_path):
    output = create_run_dir(tmp_path / "run")
    results = sample_results(tmp_path)
    results["records"][0] = (*results["records"][0][:3], 0)
    with pytest.raises(ValueError, match="viewid"):
        export_results(output, **results)
    assert not list(output.iterdir())


def test_run_directory_does_not_overwrite(tmp_path):
    output = create_run_dir(tmp_path / "run")
    marker = output / "marker"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        create_run_dir(output)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_dump_config_quotes_yaml_boolean_strings(tmp_path):
    yacs = pytest.importorskip("yacs.config")
    cfg = yacs.CfgNode({"TEST": {"FEAT_NORM": "yes"}, "MODEL": {"DEVICE_ID": "0"}})
    dumped = dump_config(cfg)
    assert "FEAT_NORM: 'yes'" in dumped
    assert "DEVICE_ID: '''0'''" in dumped

    restored = yacs.CfgNode({"TEST": {"FEAT_NORM": "yes"}, "MODEL": {"DEVICE_ID": "0"}})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(dumped, encoding="utf-8")
    restored.merge_from_file(str(config_path))
    assert restored.TEST.FEAT_NORM == "yes"
    assert restored.MODEL.DEVICE_ID == "0"
