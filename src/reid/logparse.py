"""Parse mAP/CMC from upstream TransReID / CLIP-ReID logs.

Both baselines print one line per metric, for example::

    transreid.test INFO: mAP: 54.6%
    transreid.test INFO: CMC curve, Rank-1  :70.4%
    transreid.test INFO: CMC curve, Rank-5  :78.4%
    transreid.test INFO: CMC curve, Rank-10 :82.2%

The runner uses this module to turn a log into ``metrics.json`` so numbers
are never copied by hand.
"""

from __future__ import annotations

import re

_PATTERNS = {
    "mAP": re.compile(r"\bmAP:\s*([0-9]+(?:\.[0-9]+)?)%"),
    "rank_1": re.compile(r"\bRank-1\s*:\s*([0-9]+(?:\.[0-9]+)?)%"),
    "rank_5": re.compile(r"\bRank-5\s*:\s*([0-9]+(?:\.[0-9]+)?)%"),
    "rank_10": re.compile(r"\bRank-10\s*:\s*([0-9]+(?:\.[0-9]+)?)%"),
}


def parse_metrics(log_text: str) -> dict[str, float]:
    """Return metrics as fractions (54.6% -> 0.546).

    Raises:
        ValueError: if no mAP line is present.
    """
    result: dict[str, float] = {}
    for name, pattern in _PATTERNS.items():
        matches = pattern.findall(log_text)
        if matches:
            result[name] = round(float(matches[-1]) / 100.0, 6)
    if "mAP" not in result:
        raise ValueError("no 'mAP: <number>%' line found in the log")
    return result


def build_metrics_payload(
    log_text: str,
    *,
    log_path: str = "",
    run_id: str = "",
    baseline: str = "",
    dataset: str = "",
) -> dict[str, object]:
    """Build the ``metrics.json`` payload from an upstream log."""
    metrics = parse_metrics(log_text)
    return {
        "run_id": run_id,
        "baseline": baseline,
        "dataset": dataset,
        "source": "upstream_log",
        "log": log_path,
        **metrics,
    }
