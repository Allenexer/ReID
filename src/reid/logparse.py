"""解析 TransReID/CLIP-ReID 的文本评测日志。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_METRIC_PATTERNS = {
    "mAP": re.compile(r"\bmAP\s*:\s*([0-9]+(?:\.[0-9]+)?)(\s*%)?", re.IGNORECASE),
    "rank1": re.compile(
        r"\bRank[-_ ]?1\s*:\s*([0-9]+(?:\.[0-9]+)?)(\s*%)?", re.IGNORECASE
    ),
    "rank5": re.compile(
        r"\bRank[-_ ]?5\s*:\s*([0-9]+(?:\.[0-9]+)?)(\s*%)?", re.IGNORECASE
    ),
    "rank10": re.compile(
        r"\bRank[-_ ]?10\s*:\s*([0-9]+(?:\.[0-9]+)?)(\s*%)?", re.IGNORECASE
    ),
}


def _normalise(value: str, percent: str | None) -> float:
    number = float(value)
    if percent or number > 1:
        number /= 100
    if not 0 <= number <= 1:
        raise ValueError(f"指标超出 0～1 范围：{value}{percent or ''}")
    return number


def parse_metrics(text: str) -> dict[str, float]:
    """从日志中提取最后一组 mAP、Rank-1/5/10，统一为 0～1。"""
    parsed: dict[str, float] = {}
    for name, pattern in _METRIC_PATTERNS.items():
        matches = list(pattern.finditer(text))
        if matches:
            match = matches[-1]
            parsed[name] = _normalise(match.group(1), match.group(2))
    missing = sorted(set(_METRIC_PATTERNS) - set(parsed))
    if missing:
        raise ValueError("日志缺少指标：" + ", ".join(missing))
    return parsed


def metrics_from_log(
    log_path: Path,
    *,
    model: str = "transreid",
    dataset: str = "market1501",
    num_query: int | None = None,
    num_gallery: int | None = None,
) -> dict[str, Any]:
    """读取日志并生成可供报告使用的标准指标 JSON。"""
    if model not in {"transreid", "clipreid"}:
        raise ValueError("model 必须是 transreid 或 clipreid")
    if dataset not in {"market1501", "ccvid"}:
        raise ValueError("dataset 必须是 market1501 或 ccvid")
    if (num_query is None) != (num_gallery is None):
        raise ValueError("num-query 和 num-gallery 必须同时提供")
    values = parse_metrics(log_path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "schema_version": 1,
        "model": model,
        "dataset": dataset,
        "evaluation_source": f"{model}_log",
        "protocol": "market1501" if dataset == "market1501" else "ccvid_general",
        "distance_metric": "squared_euclidean",
        "feature_normalization": "l2",
        "re_ranking": False,
        "metric_scale": "0-1",
        **values,
    }
    if num_query is not None and num_gallery is not None:
        if num_query <= 0 or num_gallery <= 0:
            raise ValueError("样本数量必须为正整数")
        result.update({"num_query": num_query, "num_gallery": num_gallery})
    return result
