"""MIND 适配器：把 torch-rechub 的多兴趣模型接进统一评测接口。

MIND 与双塔的区别在于「一个用户」不是单个向量，而是 K 个兴趣向量；
服务时对每个候选取 ``max over K`` 个兴趣的相似度，这也是论文与工业界的标准做法。

实现约束：用户向量依赖**历史序列**，因此 ``encode_users`` 需要数据；
统一接口里它在没有 ``PreparedData`` 时无法工作，会明确报错，避免静默给错向量。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from recsys.adapters.base import AdapterError, RecommenderAdapter
from recsys.data.dataset import build_observed_histories
from recsys.data.schema import PreparedData, seen_items_for_split
from recsys.models.mind import load_mind_from_checkpoint, rank_users


class MindAdapter(RecommenderAdapter):
    """MIND（多兴趣召回）的统一接口实现。"""

    model_id = "mind"
    model_type = "retrieval"

    def __init__(self, model: torch.nn.Module, checkpoint: dict, device: torch.device) -> None:
        self.model = model
        self.checkpoint = checkpoint
        self.device = device

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "MindAdapter":
        if not path.exists():
            raise AdapterError(
                f"找不到 checkpoint: {path}。先运行 python -m recsys.training.train_mind。"
            )
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        expected = checkpoint.get("split_strategy")
        if expected is None:
            raise AdapterError("checkpoint 缺少 split_strategy，无法确认切分协议")
        model = load_mind_from_checkpoint(checkpoint, device)
        return cls(model, checkpoint, device)

    def _histories(self, data: PreparedData, split: str) -> dict[int, list[int]]:
        # 与 SASRec 一致：测试阶段的历史 = train + validation
        return build_observed_histories(data, split, include_validation_in_test_history=True)

    def recommend(
        self,
        data: PreparedData,
        *,
        split: str = "test",
        k: int = 20,
        **kwargs: object,
    ) -> dict[int, list[int]]:
        from recsys.evaluation.evaluator import get_evaluation_interactions, group_items_by_user

        targets = get_evaluation_interactions(data, split)
        users = sorted(group_items_by_user(targets))
        if not users:
            return {}
        histories = self._histories(data, split)
        config = self.checkpoint["model_config"]
        blocked = seen_items_for_split(data, split)
        routing_seed = int(kwargs.pop("routing_seed", 1234))
        return rank_users(
            self.model,
            users,
            histories,
            data.num_items,
            max_length=int(config["max_length"]),
            device=self.device,
            k=k,
            seen_items=blocked,
            routing_seed=routing_seed,
        )

    def encode_users(self, user_ids: Sequence[int], *, batch_size: int = 1024) -> np.ndarray:
        raise AdapterError(
            "MIND 的用户向量由历史序列动态生成，需要 PreparedData；"
            "请用 recommend()（内部会取每个候选在 K 个兴趣上的最大相似度）"
        )

    def encode_items(self, item_ids: Sequence[int], *, batch_size: int = 1024) -> np.ndarray:
        tokens = torch.as_tensor([int(item) + 1 for item in item_ids], dtype=torch.long, device=self.device)
        self.model.eval()
        self.model.mode = "item"
        with torch.no_grad():
            return self.model({"item_id": tokens}).float().cpu().numpy()

    def evaluate(
        self,
        data: PreparedData,
        *,
        split: str = "test",
        ks: Sequence[int] = (10, 20),
        **kwargs: object,
    ) -> dict[str, float]:
        from recsys.evaluation.evaluator import evaluate_recommendations, get_evaluation_interactions
        from recsys.evaluation.metrics import catalog_coverage

        recommendations = self.recommend(data, split=split, k=max(ks))
        metrics = evaluate_recommendations(
            recommendations, get_evaluation_interactions(data, split), ks
        )
        metrics["Coverage"] = catalog_coverage(recommendations, data.num_items)
        return metrics


__all__ = ["MindAdapter"]
