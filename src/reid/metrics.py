"""Standard Re-ID evaluation: mAP and CMC.

Protocol: for each query, gallery samples with the same identity AND the
same camera are removed before ranking (the usual Market1501/DukeMTMC
protocol). Queries left without any true match are skipped, matching the
reference implementations used by TransReID/CLIP-ReID.

The evaluator accepts a ``query x gallery`` distance matrix so the same
code can score any baseline, including ones that only dump features.
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
    """
    distmat = np.asarray(distmat)
    q_pids = np.asarray(q_pids, dtype=np.int64)
    g_pids = np.asarray(g_pids, dtype=np.int64)
    q_camids = np.asarray(q_camids, dtype=np.int64)
    g_camids = np.asarray(g_camids, dtype=np.int64)

    if distmat.ndim != 2:
        raise ValueError(f"distmat must be 2-D, got shape {distmat.shape}")
    num_q, num_g = distmat.shape
    if q_pids.shape != (num_q,) or q_camids.shape != (num_q,):
        raise ValueError("query ids/cameras do not match distmat rows")
    if g_pids.shape != (num_g,) or g_camids.shape != (num_g,):
        raise ValueError("gallery ids/cameras do not match distmat columns")
    if num_q == 0 or num_g == 0:
        raise ValueError("empty query or gallery")
    if max_rank < 1:
        raise ValueError("max_rank must be positive")

    max_rank = min(max_rank, num_g)
    indices = np.argsort(distmat, axis=1, kind="stable")
    matches = g_pids[indices] == q_pids[:, None]

    all_cmc = np.zeros((num_q, max_rank), dtype=np.float64)
    all_ap = np.zeros(num_q, dtype=np.float64)
    valid = np.zeros(num_q, dtype=bool)

    for qi in range(num_q):
        order = indices[qi]
        if same_camera_exclude:
            keep = ~(
                (g_pids[order] == q_pids[qi])
                & (g_camids[order] == q_camids[qi])
            )
        else:
            keep = np.ones(num_g, dtype=bool)

        raw_cmc = matches[qi][keep]
        if not raw_cmc.any():
            continue

        valid[qi] = True
        hits = (np.cumsum(raw_cmc, dtype=np.int64) > 0).astype(np.float64)
        n = min(len(hits), max_rank)
        all_cmc[qi, :n] = hits[:n]
        if n < max_rank:  # CMC saturates at the last observed rank
            all_cmc[qi, n:] = hits[n - 1]

        precision = np.cumsum(raw_cmc, dtype=np.float64) / np.arange(
            1, len(raw_cmc) + 1, dtype=np.float64
        )
        all_ap[qi] = float(np.sum(precision * raw_cmc) / raw_cmc.sum())

    num_valid = int(valid.sum())
    if num_valid == 0:
        raise ValueError(
            "no query has a valid gallery match after same-camera filtering"
        )

    cmc = all_cmc[valid].mean(axis=0)
    return RankMetrics(
        mAP=float(all_ap[valid].mean()),
        cmc=tuple(float(x) for x in cmc),
        num_query=num_q,
        num_valid_query=num_valid,
        num_gallery=num_g,
    )
