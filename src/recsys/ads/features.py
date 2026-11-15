"""广告批次特征：复用推荐排序的 ``FeatureContext``，不新建特征口径。

* 物品类目 4 列（一级/二级类目、item_type、upload_type）；
* 物品数值 5 列（author_freq / caption_tokens / upload_days / is_ad / upload_valid，已标准化）；
* 用户数值 2 列（训练交互数、唯一物品数的 log1p）；
* 行为序列：按 ``(user, request_timestamp)`` 取严格早于请求的最近 ``max_history`` 个物品。

``history`` 只对 DIN/DIEN 有意义；DCNv2 与 DeepCTR 模型会忽略它。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from recsys.ranking.features import FeatureContext


@dataclass
class AdBatch:
    users: np.ndarray
    items: np.ndarray
    item_cat: np.ndarray
    item_numeric: np.ndarray
    user_numeric: np.ndarray
    history: np.ndarray
    labels: np.ndarray
    groups: np.ndarray

    def __len__(self) -> int:
        return int(self.items.size)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "users": self.users,
            "items": self.items,
            "item_cat": self.item_cat,
            "item_numeric": self.item_numeric,
            "user_numeric": self.user_numeric,
            "history": self.history,
            "labels": self.labels,
            "groups": self.groups,
        }


def row_timestamps(view: Any) -> np.ndarray:
    """把请求级时间戳展开成行级（每个请求的候选共享同一时间戳）。"""
    return np.repeat(np.asarray(view.timestamps), np.asarray(view.groups))


def build_ad_batch(
    view: Any,
    context: FeatureContext,
    histories: dict[tuple[int, int], np.ndarray],
    *,
    rows: np.ndarray | None = None,
    max_history: int = 20,
) -> AdBatch:
    """构造一批广告候选的模型输入；``rows`` 缺省为全部候选行。"""
    row_index = np.arange(len(view)) if rows is None else np.asarray(rows, dtype=np.int64)
    users = np.asarray(view.users)[row_index].astype(np.int64)
    items = np.asarray(view.items)[row_index].astype(np.int64)
    labels = np.asarray(view.labels)[row_index].astype(np.float32)
    item_cat, item_numeric = context.item_arrays(items)
    user_numeric = context.user_array(users)
    timestamps = row_timestamps(view)[row_index]
    history = np.zeros((users.size, max_history), dtype=np.int64)
    if histories:
        for position, (user, timestamp) in enumerate(zip(users.tolist(), timestamps.tolist())):
            tokens = histories.get((int(user), int(timestamp)))
            if tokens is not None:
                history[position] = tokens
    groups = (
        np.asarray(view.groups, dtype=np.int32)
        if rows is None
        else np.zeros(0, dtype=np.int32)
    )
    return AdBatch(
        users=users,
        items=items,
        item_cat=item_cat,
        item_numeric=item_numeric,
        user_numeric=user_numeric,
        history=history,
        labels=labels,
        groups=groups,
    )


def build_click_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """广告 CTR 口径的 AUC / LogLoss（复用推荐侧指标实现）。"""
    from recsys.evaluation.metrics import log_loss, roc_auc

    labels_array = np.asarray(labels, dtype=np.float64)
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
    return {
        "rows": int(labels_array.size),
        "clicks": int(labels_array.sum()),
        "ctr": float(labels_array.mean()) if labels_array.size else 0.0,
        "predicted_ctr": float(clipped.mean()) if clipped.size else 0.0,
        "AUC": float(roc_auc(clipped.tolist(), labels_array.tolist())),
        "LogLoss": float(log_loss(clipped.tolist(), labels_array.tolist())),
    }


__all__ = ["AdBatch", "build_ad_batch", "build_click_metrics", "row_timestamps"]
