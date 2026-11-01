"""独立于模型的特征距离计算。"""

from typing import Tuple

import numpy as np


def validate_features(query_features: np.ndarray, gallery_features: np.ndarray) -> Tuple[int, int]:
    """检查特征维度与每行有限值、非零范数。"""
    query, gallery = np.asarray(query_features), np.asarray(gallery_features)
    if (
        query.ndim != 2
        or gallery.ndim != 2
        or min(query.shape) <= 0
        or min(gallery.shape) <= 0
        or query.shape[1] != gallery.shape[1]
    ):
        raise ValueError("特征维度不一致：query {}，gallery {}".format(query.shape, gallery.shape))
    for name, array in (("query_features", query), ("gallery_features", gallery)):
        if array.dtype.kind not in "fiu":
            raise ValueError("{} 须为实数数组".format(name))
        for start in range(0, len(array), 256):
            block = array[start : start + 256].astype(np.float64)
            if not np.isfinite(block).all():
                raise ValueError("{} 第 {} 行附近存在 NaN/Inf".format(name, start))
            norms = np.linalg.norm(block, axis=1)
            if not np.isfinite(norms).all() or np.any(norms == 0):
                raise ValueError("{} 第 {} 行附近存在零或非有限范数".format(name, start))
    return len(query), len(gallery)


def cosine_distance(query_features: np.ndarray, gallery_features: np.ndarray) -> np.ndarray:
    """重新 L2 后分块计算 1-q·g，输出 float32 的 Q×G 距离。"""
    num_query, num_gallery = validate_features(query_features, gallery_features)
    gallery = np.asarray(gallery_features, dtype=np.float64)
    gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    distances = np.empty((num_query, num_gallery), dtype=np.float32)
    for start in range(0, num_query, 256):
        query = np.asarray(query_features[start : start + 256], dtype=np.float64)
        query = query / np.linalg.norm(query, axis=1, keepdims=True)
        distances[start : start + 256] = 1.0 - np.clip(query @ gallery.T, -1.0, 1.0)
    return distances
