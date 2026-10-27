import pytest


@pytest.mark.skip(reason="A: implement evaluate_rank first")
def test_metrics_todo():
    """A should cover: perfect ranking, hand-computed AP/CMC,
    same-camera exclusion, CMC saturation after filtering, queries
    without a valid match, and input validation."""
