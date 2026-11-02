"""仅依赖 NumPy 的统一 mAP/CMC 评测。"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

PROTOCOLS = ("market1501", "ccvid_general", "ccvid_cc", "ccvid_sc")


@dataclass(frozen=True)
class EvaluationResult:
    cmc: np.ndarray
    mAP: float
    num_valid_query: int
    num_skipped_query: int

    def as_metrics(self) -> Dict[str, Optional[float]]:
        """仅在 JSON 边界舍入，计算始终保留原精度。"""
        return {
            "mAP": round(self.mAP, 6),
            **{
                "rank{}".format(rank): (
                    round(float(self.cmc[rank - 1]), 6) if rank <= len(self.cmc) else None
                )
                for rank in (1, 5, 10)
            },
        }


def _labels(values: Sequence[int], size: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (size,) or array.dtype.kind not in "iu" or np.any(array < 0):
        raise ValueError("{} 须为非负整数向量，期望 ({},)，实际 {}".format(name, size, array.shape))
    return array


def evaluate_rank(
    distmat: np.ndarray,
    q_pids: Sequence[int],
    g_pids: Sequence[int],
    q_camids: Sequence[int],
    g_camids: Sequence[int],
    protocol: str,
    q_clothes: Optional[Sequence[str]] = None,
    g_clothes: Optional[Sequence[str]] = None,
    max_rank: int = 50,
) -> EvaluationResult:
    """按 query 排序；CC/SC 过滤只作用于同一身份的衣服标签。"""
    if protocol not in PROTOCOLS:
        raise ValueError("非法评测协议：{}".format(protocol))
    if not isinstance(max_rank, int) or isinstance(max_rank, bool) or max_rank <= 0:
        raise ValueError("max_rank 须为正整数")
    matrix = np.asarray(distmat)
    if matrix.ndim != 2 or matrix.dtype.kind not in "fiu" or min(matrix.shape) <= 0:
        raise ValueError("distmat 须为非空实数二维数组，实际 {}".format(matrix.shape))
    num_query, num_gallery = matrix.shape
    qp = _labels(q_pids, num_query, "q_pids")
    gp = _labels(g_pids, num_gallery, "g_pids")
    qc = _labels(q_camids, num_query, "q_camids")
    gc = _labels(g_camids, num_gallery, "g_camids")
    clothes_query = clothes_gallery = None
    if protocol in {"ccvid_cc", "ccvid_sc"}:
        for values, size, name in (
            (q_clothes, num_query, "q_clothes"),
            (g_clothes, num_gallery, "g_clothes"),
        ):
            if (
                values is None
                or np.asarray(values).shape != (size,)
                or any(not isinstance(value, str) or not value.strip() for value in values)
            ):
                raise ValueError("{} 须包含 {} 个非空衣服标签".format(name, size))
        clothes_query, clothes_gallery = np.asarray(q_clothes), np.asarray(g_clothes)
    rank_size = min(max_rank, num_gallery)
    cmc_sum = np.zeros(rank_size, dtype=np.float64)
    ap_sum = 0.0
    valid = 0
    for index, row in enumerate(matrix):
        if not np.isfinite(row).all():
            raise ValueError("distmat query {} 存在 NaN/Inf".format(index))
        order = np.argsort(row, kind="quicksort")
        same_pid = gp[order] == qp[index]
        remove = same_pid & (gc[order] == qc[index])
        if clothes_query is not None:
            same_clothes = clothes_gallery[order] == clothes_query[index]
            remove |= same_pid & (same_clothes if protocol == "ccvid_cc" else ~same_clothes)
        matches = same_pid[~remove]
        positions = np.flatnonzero(matches)
        if not len(positions):
            continue
        valid += 1
        ap_sum += float(np.mean(np.arange(1, len(positions) + 1) / (positions + 1)))
        # 过滤后不足 max_rank 的 query，以末值 1 补齐，避免不同长度的 CMC 无法聚合。
        cmc_sum[positions[0] :] += 1
    if not valid:
        raise ValueError("协议 {} 下无有效 query".format(protocol))
    return EvaluationResult(cmc_sum / valid, ap_sum / valid, valid, num_query - valid)
