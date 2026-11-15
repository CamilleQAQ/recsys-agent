"""LightGCN 适配器：把图传播后的用户/物品表示接进统一召回接口。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from recsys.adapters.base import AdapterError, RecommenderAdapter, evaluation_users
from recsys.data.loader import resolve_checkpoint_dataset, split_strategy_for
from recsys.data.schema import PreparedData, seen_items_for_split
from recsys.evaluation.evaluator import brute_force_top_k
from recsys.models.lightgcn import LightGCN, build_normalized_adjacency


class LightGCNAdapter(RecommenderAdapter):
    model_id = "lightgcn"
    model_type = "graph"

    def __init__(self, model: LightGCN, checkpoint: dict, device: torch.device) -> None:
        self.model = model
        self.checkpoint = checkpoint
        self.device = device
        graph = checkpoint.get("graph") or {}
        if "users" not in graph or "items" not in graph:
            raise AdapterError("LightGCN checkpoint 缺少 graph.users/graph.items，无法重建图")
        self.adjacency = build_normalized_adjacency(
            graph["users"].cpu().numpy(),
            graph["items"].cpu().numpy(),
            int(checkpoint["model_config"]["num_users"]),
            int(checkpoint["model_config"]["num_items"]),
            device,
        )

    @classmethod
    def load(cls, checkpoint_path: Path, device: torch.device) -> "LightGCNAdapter":
        if not checkpoint_path.exists():
            raise AdapterError(f"找不到 checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        expected = split_strategy_for(resolve_checkpoint_dataset(checkpoint))
        if checkpoint.get("split_strategy") != expected:
            raise AdapterError(
                f"LightGCN checkpoint 切分协议 {checkpoint.get('split_strategy')!r} "
                f"与数据集要求 {expected!r} 不一致"
            )
        model = LightGCN(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return cls(model, checkpoint, device)

    def _check_vocabulary(self, data: PreparedData) -> None:
        config = self.checkpoint["model_config"]
        if int(config["num_users"]) != data.num_users:
            raise AdapterError("LightGCN checkpoint 用户词表与数据不一致")
        if int(config["num_items"]) != data.num_items:
            raise AdapterError("LightGCN checkpoint 物品词表与数据不一致")

    @torch.no_grad()
    def _embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        return self.model.split_embeddings(self.adjacency)

    def recommend(
        self,
        data: PreparedData,
        *,
        split: str = "test",
        k: int = 20,
        **_: object,
    ) -> dict[int, list[int]]:
        self._check_vocabulary(data)
        users = evaluation_users(data, split)
        if not users:
            raise AdapterError(f"{split} 切分里没有可评测用户")
        user_embeddings, item_embeddings = self._embeddings()
        user_tensor = torch.tensor(users, dtype=torch.long, device=self.device)
        return brute_force_top_k(
            user_indices=users,
            user_embeddings=user_embeddings[user_tensor],
            item_embeddings=item_embeddings,
            train_items_by_user=seen_items_for_split(data, split),
            k=k,
        )

    def encode_users(self, user_ids: Sequence[int], *, batch_size: int = 1024) -> np.ndarray:
        if not user_ids:
            raise AdapterError("encode_users: id 列表为空")
        user_embeddings, _ = self._embeddings()
        indices = torch.tensor([int(value) for value in user_ids], dtype=torch.long)
        return user_embeddings[indices].detach().cpu().numpy().astype(np.float32)

    def encode_items(self, item_ids: Sequence[int], *, batch_size: int = 1024) -> np.ndarray:
        if not item_ids:
            raise AdapterError("encode_items: id 列表为空")
        _, item_embeddings = self._embeddings()
        indices = torch.tensor([int(value) for value in item_ids], dtype=torch.long)
        return item_embeddings[indices].detach().cpu().numpy().astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint, path)


__all__ = ["LightGCNAdapter"]
