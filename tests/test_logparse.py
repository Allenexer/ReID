import pytest

from src.reid.logparse import build_metrics_payload, parse_metrics

SAMPLE_LOG = """
2026-01-22 09:22:49,939 transreid.test INFO: Validation Results
2026-01-22 09:22:49,939 transreid.test INFO: mAP: 54.6%
2026-01-22 09:22:49,939 transreid.test INFO: CMC curve, Rank-1  :70.4%
2026-01-22 09:22:49,939 transreid.test INFO: CMC curve, Rank-5  :78.4%
2026-01-22 09:22:49,939 transreid.test INFO: CMC curve, Rank-10 :82.2%
"""


def test_parse_sample_log():
    metrics = parse_metrics(SAMPLE_LOG)
    assert metrics == {
        "mAP": 0.546,
        "rank_1": 0.704,
        "rank_5": 0.784,
        "rank_10": 0.822,
    }


def test_last_occurrence_wins():
    log = "mAP: 50.0%\nmAP: 54.6%\nRank-1 :70.4%\n"
    assert parse_metrics(log)["mAP"] == 0.546


def test_missing_map_raises():
    with pytest.raises(ValueError, match="no 'mAP"):
        parse_metrics("Rank-1 :70.4%\n")


def test_build_metrics_payload():
    payload = build_metrics_payload(
        SAMPLE_LOG,
        log_path="test_log.txt",
        run_id="20261027-transreid-market",
        baseline="transreid",
        dataset="market1501",
    )
    assert payload == {
        "run_id": "20261027-transreid-market",
        "baseline": "transreid",
        "dataset": "market1501",
        "source": "upstream_log",
        "log": "test_log.txt",
        "mAP": 0.546,
        "rank_1": 0.704,
        "rank_5": 0.784,
        "rank_10": 0.822,
    }
