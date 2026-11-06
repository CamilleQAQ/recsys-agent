"""适配层公共协议与默认实现。

召回适配器承诺四件事（命名对齐 PROJECT_MASTER §5.1）：

1. ``recommend`` / ``retrieve``：给待评测（或给定）用户返回 Top-K item_idx；
2. ``encode_users`` / ``encode_items``：导出向量，交给 FAISS 等检索器；
3. ``evaluate``：用项目统一指标算 Recall/NDCG/HitRate/Precision + Coverage；
4. ``save`` / ``load``：checkpoint 读写由各模型自己实现。

排序适配器（``RankingAdapter``）走另一套：``fit`` / ``predict`` / ``evaluate``（AUC、LogLoss）。

任何「词表对不上」「checkpoint 缺失」之类的问题都抛 ``AdapterError``，
方便上层脚本统一处理，而不是让 KeyError/形状错误冒到调用方。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from recsys.data.schema import Interaction, PreparedData
from recsys.evaluation.evaluator import evaluate_recommendations
from recsys.evaluation.metrics import (
    catalog_coverage,
    compute_ranking_model_metrics,
)


class AdapterError(RuntimeError):
    """适配层无法完成请求（缺 checkpoint、词表不一致、模型不支持该操作等）。"""


def evaluation_users(data: PreparedData, split: str) -> list[int]:
    """返回该切分里真正有留出交互的用户（升序）。"""
    targets = _targets(data, split)
    return sorted({row.user_idx for row in targets})


def _targets(data: PreparedData, split: str) -> Sequence[Interaction]:
    if split == "validation":
        return data.validation_interactions
    if split == "test":
        return data.test_interactions
    raise AdapterError(f"split 必须是 'validation' 或 'test'，收到 {split!r}")


class RecommenderAdapter(ABC):
    """推荐模型的最小统一接口。"""

    model_id: str = ""
    model_type: str = ""

    @abstractmethod
    def recommend(
        self,
        data: PreparedData,
        *,
        split: str = "test",
        k: int = 20,
        **kwargs: object,
    ) -> dict[int, list[int]]:
        """给该切分的每个评测用户返回 Top-K item_idx。"""

    def retrieve(
        self,
        data: PreparedData,
        *,
        split: str = "test",
        k: int = 20,
        **kwargs: object,
    ) -> dict[int, list[int]]:
        """召回语义的入口（PROJECT_MASTER §5.1 的 ``retrieve``）。

        与 ``recommend`` 是同一件事：给每个用户从全量物品里取 Top-K。
        单独留一个名字，是为了让后续多路召回融合按「通道」调用而不必关心实现差异。
        """
        return self.recommend(data, split=split, k=k, **kwargs)

    def encode_users(
        self, user_ids: Sequence[int], *, batch_size: int = 1024
    ) -> np.ndarray:
        """生成用户向量；不支持向量化表示的模型抛 AdapterError。"""
        raise AdapterError(f"{type(self).__name__} 不提供用户向量（encode_users）")

    def encode_items(
        self, item_ids: Sequence[int], *, batch_size: int = 1024
    ) -> np.ndarray:
        """生成物品向量；不支持向量化表示的模型抛 AdapterError。"""
        raise AdapterError(f"{type(self).__name__} 不提供物品向量（encode_items）")

    def evaluate(
        self,
        data: PreparedData,
        *,
        split: str = "test",
        ks: Sequence[int] = (10, 20),
        **kwargs: object,
    ) -> dict[str, float]:
        """统一评估入口：一次打分，多切点出指标 + 全局 Coverage。"""
        if not ks:
            raise AdapterError("ks 不能为空")
        recommendations = self.recommend(data, split=split, k=max(ks), **kwargs)
        metrics = evaluate_recommendations(recommendations, _targets(data, split), ks)
        metrics["Coverage"] = catalog_coverage(recommendations, data.num_items)
        return metrics

    def save(self, path: Path) -> None:
        """默认不支持保存，子类按需覆盖。"""
        raise AdapterError(f"{type(self).__name__} 不支持 save()")

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"<{type(self).__name__} model_id={self.model_id!r}>"


class RankingAdapter(ABC):
    """排序/CTR 模型的最小统一接口（PROJECT_MASTER §5.1 的排序侧）。

    与召回适配器的差别：输入是「候选 + 特征」，输出是每个候选的分数，
    因此没有 encode_*；评估也换成 AUC/LogLoss（``compute_ranking_model_metrics``）。
    """

    model_id: str = ""
    feature_names: tuple[str, ...] = ()

    @abstractmethod
    def predict(self, features: Mapping[str, np.ndarray] | np.ndarray) -> np.ndarray:
        """输出每个候选的分数（或点击概率），形状 ``[num_candidates]``。"""

    def fit(
        self,
        features: Mapping[str, np.ndarray] | np.ndarray,
        labels: np.ndarray,
        *,
        groups: np.ndarray | None = None,
    ) -> None:
        """默认不支持训练，子类按需覆盖；``groups`` 用于 LambdaRank 的 query 分组。"""
        raise AdapterError(f"{type(self).__name__} 不支持 fit()")

    def save(self, path: Path) -> None:
        raise AdapterError(f"{type(self).__name__} 不支持 save()")

    @classmethod
    def load(cls, path: Path) -> "RankingAdapter":
        raise AdapterError(f"{cls.__name__} 不支持 load()")

    def evaluate(
        self,
        features: Mapping[str, np.ndarray] | np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, float]:
        """排序模型统一评估：AUC + LogLoss。"""
        return compute_ranking_model_metrics(
            [float(score) for score in self.predict(features)],
            [int(label) for label in labels],
        )
