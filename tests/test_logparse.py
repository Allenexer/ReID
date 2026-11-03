"""上游日志解析测试。"""

import pytest

from src.reid.logparse import metrics_from_log, parse_metrics


def test_parse_percent_metrics_uses_last_evaluation():
    text = """
    mAP: 80.0%
    CMC curve, Rank-1: 90.0%
    CMC curve, Rank-5: 95.0%
    CMC curve, Rank-10: 97.0%
    mAP: 87.1%
    CMC curve, Rank-1: 94.6%
    CMC curve, Rank-5: 98.1%
    CMC curve, Rank-10: 99.0%
    """
    assert parse_metrics(text) == {
        "mAP": 0.871,
        "rank1": 0.946,
        "rank5": 0.981,
        "rank10": 0.99,
    }


def test_parse_fraction_metrics():
    assert parse_metrics(
        "mAP: 0.871\nRank-1: 0.946\nRank-5: 0.981\nRank-10: 0.990\n"
    )["mAP"] == pytest.approx(0.871)


def test_log_metrics_include_contract_metadata(tmp_path):
    log = tmp_path / "test_log.txt"
    log.write_text("mAP: 87.1%\nRank-1: 94.6%\nRank-5: 98.1%\nRank-10: 99.0%\n")
    result = metrics_from_log(log, num_query=3368, num_gallery=15913)
    assert result["model"] == "transreid"
    assert result["num_query"] == 3368
    assert result["rank1"] == pytest.approx(0.946)


def test_missing_metric_is_rejected():
    with pytest.raises(ValueError, match="rank10"):
        parse_metrics("mAP: 87.1%\nRank-1: 94.6%\nRank-5: 98.1%\n")
