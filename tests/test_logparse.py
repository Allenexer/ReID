import pytest


@pytest.mark.skip(reason="B: implement parse_metrics first")
def test_log_parser_todo():
    """B should cover: a sample TransReID/CLIP-ReID log, last occurrence
    wins, missing mAP raises ValueError, and the metrics.json payload
    metadata."""
