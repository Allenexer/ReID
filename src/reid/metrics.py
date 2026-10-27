"""Standard Re-ID evaluation: mAP and CMC.

Protocol contract (member A implements the algorithm):

- for each query, gallery samples with the same identity AND the same
  camera are removed before ranking (the usual Market1501/DukeMTMC
  protocol);
- queries left without any true match are skipped, matching the
  reference implementations used by TransReID/CLIP-ReID;
- the evaluator accepts a ``query x gallery`` distance matrix so the same
  code can score any baseline, including ones that only dump features.

``RankMetrics`` and ``evaluate_rank`` are the shared output/interface
contract; the function body is TODO for member A.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankMetrics:
    """Result of one evaluation run."""

    mAP: float
    cmc: tuple[float, ...]
    num_query: int
    num_valid_query: int
    num_gallery: int

    def rank(self, k: int) -> float:
        if k < 1 or k > len(self.cmc):
            raise ValueError(f"rank must be in [1, {len(self.cmc)}], got {k}")
        return self.cmc[k - 1]

    def as_dict(self, ranks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float | int]:
        result: dict[str, float | int] = {
            "mAP": round(float(self.mAP), 6),
            "num_query": self.num_query,
            "num_valid_query": self.num_valid_query,
            "num_gallery": self.num_gallery,
        }
        for k in ranks:
            if k <= len(self.cmc):
                result[f"rank_{k}"] = round(float(self.cmc[k - 1]), 6)
        return result


def evaluate_rank(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = 50,
    same_camera_exclude: bool = True,
) -> RankMetrics:
    """Compute mAP and CMC from a distance matrix.

    Args:
        distmat: ``(num_query, num_gallery)`` distances; smaller is closer.
        q_pids/g_pids: identity ids for queries and gallery.
        q_camids/g_camids: camera ids for queries and gallery.
        max_rank: length of the returned CMC curve.
        same_camera_exclude: drop same-id same-camera gallery entries.

    TODO(A): implement. Requirements:

    - validate shapes and reject empty inputs;
    - sort each query's gallery by distance (``np.argsort``, stable);
    - if ``same_camera_exclude``, drop same-id same-camera entries;
    - AP = sum(precision at each true match * match) / number of matches;
    - CMC = cumulative hit indicator, saturated after the last gallery
      entry when filtering shortens the list;
    - skip queries without a valid match; raise ``ValueError`` if no
      query is valid;
    - return ``RankMetrics`` with the mean over valid queries.
    """
    raise NotImplementedError("A: implement evaluate_rank")
