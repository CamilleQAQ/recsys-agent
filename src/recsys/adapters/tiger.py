"""TIGER 适配器：Semantic ID 生成式召回接入统一接口。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from recsys.adapters.base import AdapterError, RecommenderAdapter, evaluation_users
from recsys.data.loader import resolve_checkpoint_dataset, split_strategy_for
from recsys.data.schema import PreparedData
from recsys.evaluation.tiger_evaluator import generate_tiger_recommendations
from recsys.models.rqvae import SemanticIDIndex
from recsys.models.tiger import CODE_OFFSET, TigerTransformer


class TigerAdapter(RecommenderAdapter):
    model_id = "tiger"
    model_type = "generative"

    def __init__(self, model: TigerTransformer, checkpoint: dict, device: torch.device) -> None:
        self.model = model
        self.checkpoint = checkpoint
        self.device = device
        codes = checkpoint.get("codes")
        if codes is None:
            raise AdapterError("TIGER checkpoint 缺少 codes（Semantic ID 码表）")
        self.codes = codes.cpu().numpy().astype(np.int32)
        self.sid_index = SemanticIDIndex(self.codes)

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "TigerAdapter":
        if not path.exists():
            raise AdapterError(f"找不到 checkpoint: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        expected = split_strategy_for(resolve_checkpoint_dataset(checkpoint))
        if checkpoint.get("split_strategy") != expected:
            raise AdapterError(
                f"TIGER checkpoint 切分协议 {checkpoint.get('split_strategy')!r} "
                f"与数据集要求 {expected!r} 不一致"
            )
        model = TigerTransformer(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return cls(model, checkpoint, device)

    def recommend(
        self,
        data: PreparedData,
        *,
        split: str = "test",
        k: int = 20,
        beam_size: int = 16,
        max_users: int | None = None,
        include_validation_in_test_history: bool = True,
        **_: object,
    ) -> dict[int, list[int]]:
        if self.codes.shape[0] != data.num_items:
            raise AdapterError(
                f"TIGER 码表物品数 {self.codes.shape[0]} 与数据 {data.num_items} 不一致"
            )
        if not evaluation_users(data, split):
            raise AdapterError(f"{split} 切分里没有可评测用户")
        return generate_tiger_recommendations(
            self.model,
            data,
            split,
            self.codes,
            self.sid_index,
            k=k,
            beam_size=beam_size,
            max_users=max_users,
            include_validation_in_test_history=include_validation_in_test_history,
        )

    def encode_items(self, item_ids: Sequence[int], *, batch_size: int = 1024) -> np.ndarray:
        """物品表示 = 其 Semantic ID 各级码字 embedding 的均值。"""
        if not item_ids:
            raise AdapterError("encode_items: id 列表为空")
        with torch.no_grad():
            output = []
            for item in item_ids:
                codes = self.codes[int(item)]
                tokens = torch.tensor(
                    [int(code) + CODE_OFFSET for code in codes], dtype=torch.long, device=self.device
                )
                output.append(self.model.item_embedding(tokens).mean(dim=0))
            return torch.stack(output).float().cpu().numpy()

    def encode_users(self, user_ids: Sequence[int], *, batch_size: int = 1024) -> np.ndarray:
        raise AdapterError("TIGER 的用户表示依赖历史 Semantic ID 序列；请用 recommend()")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint, path)


__all__ = ["TigerAdapter"]
