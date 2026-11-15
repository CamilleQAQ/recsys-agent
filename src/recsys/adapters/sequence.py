"""BERT4Rec / HSTU 的统一序列适配器。

两者都实现 ``max_seq_len`` + ``score_all_items``，因此可以复用 SASRec 已经
验证过的全量排序与 seen-item 过滤实现，不再复制一份评测逻辑。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from recsys.adapters.base import AdapterError, RecommenderAdapter, evaluation_users
from recsys.data.loader import resolve_checkpoint_dataset, split_strategy_for
from recsys.data.schema import PreparedData
from recsys.evaluation.sasrec_evaluator import generate_sasrec_recommendations
from recsys.models.bert4rec import BERT4Rec
from recsys.models.hstu import HSTU


class SequenceCheckpointAdapter(RecommenderAdapter):
    model_id = ""
    model_type = "sequence"
    model_class: type[nn.Module] = HSTU

    def __init__(self, model: nn.Module, checkpoint: dict, device: torch.device) -> None:
        self.model = model
        self.checkpoint = checkpoint
        self.device = device

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "SequenceCheckpointAdapter":
        if not path.exists():
            raise AdapterError(f"找不到 checkpoint: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        expected = split_strategy_for(resolve_checkpoint_dataset(checkpoint))
        if checkpoint.get("split_strategy") != expected:
            raise AdapterError(
                f"{cls.model_id} checkpoint 切分协议 "
                f"{checkpoint.get('split_strategy')!r} 与数据集要求 {expected!r} 不一致"
            )
        model = cls.model_class(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return cls(model, checkpoint, device)

    def _check_vocabulary(self, data: PreparedData) -> None:
        expected = int(self.checkpoint["model_config"].get("num_items", -1))
        if expected != data.num_items:
            raise AdapterError(
                f"{self.model_id}: checkpoint 物品数 {expected} 与数据 {data.num_items} 不一致"
            )

    def recommend(
        self,
        data: PreparedData,
        *,
        split: str = "test",
        k: int = 20,
        batch_size: int = 256,
        include_validation_in_test_history: bool = True,
        **_: object,
    ) -> dict[int, list[int]]:
        self._check_vocabulary(data)
        if not evaluation_users(data, split):
            raise AdapterError(f"{split} 切分里没有可评测用户")
        return generate_sasrec_recommendations(
            model=self.model,
            prepared_data=data,
            split=split,
            k=k,
            batch_size=batch_size,
            device=self.device,
            include_validation_in_test_history=include_validation_in_test_history,
        )

    def encode_items(self, item_ids: Sequence[int], *, batch_size: int = 1024) -> np.ndarray:
        if not item_ids:
            raise AdapterError("encode_items: id 列表为空")
        with torch.no_grad():
            tokens = torch.tensor([int(item) for item in item_ids], dtype=torch.long, device=self.device)
            return self.model.encode_items(tokens).float().cpu().numpy()

    def encode_users(self, user_ids: Sequence[int], *, batch_size: int = 1024) -> np.ndarray:
        raise AdapterError(
            f"{self.model_id} 的用户表示依赖输入序列，没有稳定用户向量；请用 recommend()"
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint, path)


class Bert4RecAdapter(SequenceCheckpointAdapter):
    model_id = "bert4rec"
    model_class = BERT4Rec


class HSTUAdapter(SequenceCheckpointAdapter):
    model_id = "hstu"
    model_class = HSTU


__all__ = ["Bert4RecAdapter", "HSTUAdapter", "SequenceCheckpointAdapter"]
