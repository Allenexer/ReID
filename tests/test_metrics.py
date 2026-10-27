import numpy as np
import pytest

from src.reid.metrics import evaluate_rank


def test_perfect_ranking():
    distmat = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    result = evaluate_rank(
        distmat,
        q_pids=np.array([1, 2]),
        g_pids=np.array([1, 2, 3]),
        q_camids=np.array([1, 1]),
        g_camids=np.array([2, 2, 3]),
        max_rank=3,
    )
    assert result.mAP == pytest.approx(1.0)
    assert result.cmc == pytest.approx((1.0, 1.0, 1.0))
    assert result.num_valid_query == 2


def test_hand_computed_ap_and_cmc():
    # Gallery order: negative, positive, positive, negative.
    # AP = (1/2 + 2/3) / 2; CMC = [0, 1, 1, 1].
    result = evaluate_rank(
        np.array([[0.1, 0.2, 0.3, 0.4]]),
        q_pids=[1],
        g_pids=[2, 1, 1, 3],
        q_camids=[1],
        g_camids=[2, 2, 2, 3],
        max_rank=4,
    )
    assert result.mAP == pytest.approx((0.5 + 2 / 3) / 2)
    assert result.cmc == pytest.approx((0.0, 1.0, 1.0, 1.0))


def test_same_camera_match_is_excluded():
    distmat = np.array([[0.05, 0.1, 0.2, 0.3, 0.4]])
    kwargs = dict(
        q_pids=[1],
        g_pids=[1, 2, 1, 1, 3],
        q_camids=[1],
        g_camids=[1, 2, 2, 2, 3],
        max_rank=4,
    )

    filtered = evaluate_rank(distmat, **kwargs)
    unfiltered = evaluate_rank(
        distmat, same_camera_exclude=False, **kwargs
    )

    assert filtered.mAP == pytest.approx((0.5 + 2 / 3) / 2)
    assert filtered.cmc == pytest.approx((0.0, 1.0, 1.0, 1.0))
    assert unfiltered.mAP == pytest.approx((1.0 + 2 / 3 + 3 / 4) / 3)
    assert unfiltered.cmc == pytest.approx((1.0, 1.0, 1.0, 1.0))


def test_cmc_saturates_when_filtered_gallery_is_short():
    # After dropping four same-camera entries only one true match remains.
    result = evaluate_rank(
        np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
        q_pids=[1],
        g_pids=[1, 1, 1, 1, 1],
        q_camids=[1],
        g_camids=[1, 1, 1, 1, 2],
        max_rank=5,
    )
    assert result.mAP == pytest.approx(1.0)
    assert result.cmc == pytest.approx((1.0, 1.0, 1.0, 1.0, 1.0))


def test_query_without_valid_match_is_skipped():
    # q1's only true match is same-camera, so q1 is skipped; q0 is scored.
    result = evaluate_rank(
        np.array([[0.1, 0.5, 0.6], [0.5, 0.1, 0.6]]),
        q_pids=[1, 2],
        g_pids=[1, 2, 3],
        q_camids=[1, 1],
        g_camids=[2, 1, 3],
        max_rank=3,
    )
    assert result.num_query == 2
    assert result.num_valid_query == 1
    assert result.mAP == pytest.approx(1.0)
    assert result.cmc == pytest.approx((1.0, 1.0, 1.0))


def test_all_queries_invalid_raises():
    with pytest.raises(ValueError, match="no query has a valid gallery match"):
        evaluate_rank(
            np.array([[0.1, 0.2]]),
            q_pids=[1],
            g_pids=[1, 2],
            q_camids=[1],
            g_camids=[1, 2],
            max_rank=2,
        )


def test_shape_validation():
    with pytest.raises(ValueError, match="2-D"):
        evaluate_rank(
            np.array([0.1, 0.2]),
            q_pids=[1],
            g_pids=[1, 2],
            q_camids=[1],
            g_camids=[2, 2],
        )

    with pytest.raises(ValueError, match="query ids/cameras"):
        evaluate_rank(
            np.zeros((2, 2)),
            q_pids=[1],
            g_pids=[1, 2],
            q_camids=[1, 1],
            g_camids=[2, 2],
        )
