"""排序模型的统一特征视图（LightGBM / DIN / DIEN / DCNv2 / Ultra-HSTU 共用）。

所有特征都来自项目已有的物品特征表与训练交互统计，不引入新的数据管线：

* 物品数值特征 5 列：author_freq(log1p)、caption_tokens(log1p)、
  upload_days(年)、is_ad、upload_valid；
* 物品类目编码 4 列：一级/二级类目、item_type、upload_type（-1 → 0 未知）；
* 用户数值特征 2 列：训练交互数(log1p)、训练唯一物品数(log1p)；
* LightGBM 仍使用原有 9 列 ``kuasar_ranking.ITEM_FEATURE_COLUMNS``，保持既有
  MVP 的特征口径不变。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from recsys.data.schema import PreparedData
from recsys.data.kuasar_ranking import ITEM_FEATURE_COLUMNS

CATEGORICAL_COLUMNS = (
    "category_first_code",
    "category_second_code",
    "item_type_code",
    "upload_type_code",
)
CATEGORICAL_CARDINALITIES = (38, 294, 4, 30)
NUMERIC_ITEM_COLUMNS = ("author_freq", "caption_tokens", "upload_days", "is_ad", "upload_valid")
NUMERIC_USER_COLUMNS = ("train_count_log1p", "train_items_log1p")


@dataclass
class FeatureContext:
    """缓存全量特征矩阵，避免每个候选批次重复计算。"""

    data: PreparedData
    item_cat: np.ndarray | None = None
    item_numeric: np.ndarray | None = None
    user_numeric: np.ndarray | None = None
    _lgbm_cache: dict[str, np.ndarray] = field(default_factory=dict)

    def _ensure_item(self) -> None:
        if self.item_numeric is not None and self.item_cat is not None:
            return
        features = self.data.item_features
        if features is None:
            raise ValueError("数据集没有 item_features，无法构建排序特征")
        num_items = self.data.num_items
        cat = np.zeros((num_items, len(CATEGORICAL_COLUMNS)), dtype=np.int64)
        for column, name in enumerate(CATEGORICAL_COLUMNS):
            values = np.asarray(features.column(name)).astype(np.int64)
            cardinality = CATEGORICAL_CARDINALITIES[column]
            cat[:, column] = np.where((values >= 0) & (values < cardinality), values, 0)

        numeric = np.zeros((num_items, len(NUMERIC_ITEM_COLUMNS)), dtype=np.float32)
        author = np.asarray(features.column("author_freq"), dtype=np.float32)
        numeric[:, 0] = np.log1p(np.clip(author, 0, None))
        caption = np.asarray(features.column("caption_tokens"), dtype=np.float32)
        numeric[:, 1] = np.log1p(np.clip(caption, 0, None))
        upload = np.asarray(features.column("upload_days"), dtype=np.float32)
        numeric[:, 2] = np.clip(upload, 0, None) / 365.25
        numeric[:, 3] = np.asarray(features.column("is_ad"), dtype=np.float32)
        numeric[:, 4] = np.asarray(features.column("upload_valid"), dtype=np.float32)
        mean = numeric.mean(axis=0, keepdims=True)
        std = numeric.std(axis=0, keepdims=True) + 1e-6
        self.item_cat = cat
        self.item_numeric = (numeric - mean) / std

    def _ensure_user(self) -> None:
        if self.user_numeric is not None:
            return
        counts = np.zeros(self.data.num_users, dtype=np.float64)
        unique: list[set[int]] = [set() for _ in range(self.data.num_users)]
        for row in self.data.train_interactions:
            counts[row.user_idx] += 1
            unique[row.user_idx].add(row.item_idx)
        unique_counts = np.asarray([len(items) for items in unique], dtype=np.float64)
        self.user_numeric = np.stack(
            [np.log1p(counts), np.log1p(unique_counts)], axis=1
        ).astype(np.float32)

    def item_arrays(self, item_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_item()
        items = np.asarray(item_indices, dtype=np.int64)
        return self.item_cat[items], self.item_numeric[items]

    def user_array(self, user_indices: np.ndarray) -> np.ndarray:
        self._ensure_user()
        return self.user_numeric[np.asarray(user_indices, dtype=np.int64)]

    def lightgbm_features(self, item_indices: np.ndarray) -> dict[str, np.ndarray]:
        """与原 MVP 完全一致的 9 列特征字典（列名 → 数组）。"""
        if self.data.item_features is None:
            raise ValueError("数据集没有 item_features，无法构建排序特征")
        items = np.asarray(item_indices, dtype=np.int64)
        output: dict[str, np.ndarray] = {}
        for name in ITEM_FEATURE_COLUMNS:
            cached = self._lgbm_cache.get(name)
            if cached is None:
                cached = np.asarray(self.data.item_features.column(name))
                self._lgbm_cache[name] = cached
            output[name] = cached[items].astype(np.float32)
        return output


__all__ = [
    "CATEGORICAL_CARDINALITIES",
    "CATEGORICAL_COLUMNS",
    "FeatureContext",
    "NUMERIC_ITEM_COLUMNS",
    "NUMERIC_USER_COLUMNS",
]
