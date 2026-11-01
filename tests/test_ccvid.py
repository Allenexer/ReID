"""CCVID 序列读取测试。"""

from pathlib import Path

import numpy as np
import pytest

from src.reid.adapters.ccvid import load_ccvid_split, mean_pool_features, select_frame_indices


def make_sequence(root: Path, sequence: str) -> None:
    directory = root / sequence
    directory.mkdir(parents=True)
    (directory / "00002.jpg").touch()
    (directory / "00001.jpg").touch()


def test_preserves_split_order_identity_and_clothes_without_train(tmp_path):
    make_sequence(tmp_path, "session2/148_03")
    make_sequence(tmp_path, "session1/001_01")
    (tmp_path / "query.txt").write_bytes(
        b"session2/148_03\t148\tu0_l1_s0_c0_a0\r\nsession1/001_01\t001\tu0_l0_s0_c0_a0\r\n"
    )
    samples = load_ccvid_split(tmp_path, "query")
    assert [sample.pid for sample in samples] == [148, 1]
    assert [sample.camid for sample in samples] == [3, 1]
    assert [sample.frame_path.name for sample in samples] == ["00001.jpg", "00001.jpg"]
    assert samples[0].sequence == "session2/148_03"
    assert samples[0].clothes_label == "u0_l1_s0_c0_a0"
    assert samples[0].as_record() == (str(samples[0].frame_path), 148, 3, 0)
    assert not (tmp_path / "train.txt").exists()


def test_session3_camera_offset_and_frame_index(tmp_path):
    make_sequence(tmp_path, "session3/001_02")
    (tmp_path / "gallery.txt").write_text("session3/001_02 001 u1_l0_s0_c0_a0\n")
    sample = load_ccvid_split(tmp_path, "gallery", frame_index=1)[0]
    assert sample.camid == 14
    assert sample.frame_path.name == "00002.jpg"


@pytest.mark.parametrize(
    "line",
    [
        "session1/001_01 001",
        "../session1/001_01 001 clothes",
        "session4/001_01 001 clothes",
        "session1/001_01 002 clothes",
    ],
)
def test_invalid_records_are_rejected(tmp_path, line):
    (tmp_path / "query.txt").write_text(line + "\n")
    with pytest.raises(ValueError):
        load_ccvid_split(tmp_path, "query")


def test_missing_sequence_is_reported(tmp_path):
    (tmp_path / "query.txt").write_text("session1/001_01 001 clothes\n")
    with pytest.raises(FileNotFoundError):
        load_ccvid_split(tmp_path, "query")


def test_empty_sequence_is_rejected(tmp_path):
    (tmp_path / "session1" / "001_01").mkdir(parents=True)
    (tmp_path / "query.txt").write_text("session1/001_01 001 clothes\n")
    with pytest.raises(ValueError, match="JPG"):
        load_ccvid_split(tmp_path, "query")


@pytest.mark.parametrize("frame_index", [-1, 2])
def test_out_of_range_frame_index_is_rejected(tmp_path, frame_index):
    make_sequence(tmp_path, "session1/001_01")
    (tmp_path / "query.txt").write_text("session1/001_01 001 clothes\n")
    with pytest.raises(ValueError):
        load_ccvid_split(tmp_path, "query", frame_index=frame_index)


def test_duplicate_sequence_is_rejected(tmp_path):
    make_sequence(tmp_path, "session1/001_01")
    (tmp_path / "query.txt").write_text("session1/001_01 001 clothes\n" * 2)
    with pytest.raises(ValueError, match="重复"):
        load_ccvid_split(tmp_path, "query")


def test_uniform_sampling_includes_endpoints():
    assert select_frame_indices(10, 4) == [0, 3, 6, 9]
    assert select_frame_indices(10, 1, frame_index=2) == [2]


def test_short_sequences_use_all_frames_without_duplicates():
    assert select_frame_indices(3, 8) == [0, 1, 2]
    assert select_frame_indices(1, 8) == [0]


@pytest.mark.parametrize("frame_count,num_frames,frame_index", [(0, 4, 0), (10, 0, 0), (10, 4, 1)])
def test_invalid_sampling_is_rejected(frame_count, num_frames, frame_index):
    with pytest.raises(ValueError):
        select_frame_indices(frame_count, num_frames, frame_index)


def test_reader_retains_selected_frame_paths(tmp_path):
    make_sequence(tmp_path, "session1/001_01")
    (tmp_path / "query.txt").write_text("session1/001_01 001 clothes\n")
    sample = load_ccvid_split(tmp_path, "query", num_frames=8)[0]
    assert [path.name for path in sample.frame_paths] == ["00001.jpg", "00002.jpg"]
    assert sample.frame_path == sample.frame_paths[0]
    assert sample.num_frames == 2


def test_mean_pooling_keeps_sequence_boundaries_across_batches():
    batches = [
        np.array([[1, 0], [3, 2], [2, 1]], dtype=np.float32),
        np.array([[4, 3], [6, 5]], dtype=np.float32),
    ]
    pooled = mean_pool_features(np.concatenate(batches), [2, 3])
    np.testing.assert_array_equal(pooled, [[2, 1], [4, 3]])
    assert pooled.dtype == np.float32


def test_one_frame_pooling_preserves_features():
    features = np.array([[1.25, 3.5], [0.0, 2.0]], dtype=np.float32)
    np.testing.assert_array_equal(mean_pool_features(features, [1, 1]), features)


@pytest.mark.parametrize("counts", [[2, 2], [0, 3], []])
def test_pooling_rejects_misaligned_counts(counts):
    with pytest.raises(ValueError):
        mean_pool_features(np.zeros((3, 2), dtype=np.float32), counts)
