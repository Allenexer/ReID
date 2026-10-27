"""Parse mAP/CMC from upstream TransReID / CLIP-ReID logs.

TODO(B): implement both functions below.

Both baselines print one line per metric, for example::

    transreid.test INFO: mAP: 54.6%
    transreid.test INFO: CMC curve, Rank-1  :70.4%
    transreid.test INFO: CMC curve, Rank-5  :78.4%
    transreid.test INFO: CMC curve, Rank-10 :82.2%

The parser turns such a log into ``metrics.json`` so numbers are never
copied by hand.
"""

from __future__ import annotations


def parse_metrics(log_text: str) -> dict[str, float]:
    """Return metrics as fractions (54.6% -> 0.546).

    TODO(B): implement. Requirements:

    - recognise mAP, Rank-1, Rank-5 and Rank-10;
    - use the last occurrence of each metric (a test log may contain
      multiple validation blocks);
    - round to 6 decimals;
    - raise ``ValueError`` if no mAP line is found.
    """
    raise NotImplementedError("B: implement parse_metrics")


def build_metrics_payload(
    log_text: str,
    *,
    log_path: str = "",
    run_id: str = "",
    baseline: str = "",
    dataset: str = "",
) -> dict[str, object]:
    """Build the ``metrics.json`` payload from an upstream log.

    TODO(B): implement using :func:`parse_metrics`. The payload should
    contain ``run_id``, ``baseline``, ``dataset``, ``source`` (set to
    ``"upstream_log"``), ``log`` and the parsed metric fields.
    """
    raise NotImplementedError("B: implement build_metrics_payload")
