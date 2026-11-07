"""Two-Tower 召回模型：ID 查表（可选拼接侧特征），训练时用 in-batch negatives。

两种配置：

* **ID-only**：塔就是一个 embedding 表（``tower_type="embedding"``）或「表 + MLP」；
* **ID + 侧特征**：物品/用户表旁边再拼上类目 embedding 与数值特征，一起过 MLP。
  侧特征以 buffer 形式挂在模型上，因此 ``encode_items(ids)`` 可以独立推理，
  checkpoint 也自带特征，不依赖外部特征文件。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

#: 侧特征 embedding 的维度（每个类目字段一个）。
DEFAULT_FEATURE_DIM = 16


class Tower(nn.Module):
    """ID embedding（+ 可选侧特征）-> MLP -> L2 归一化向量。

    ``tower_type="embedding"`` 且没有侧特征时等价于「直接学习输出向量」；
    一旦提供侧特征，必须经过 MLP 才能把不同来源的向量融合起来。
    """

    def __init__(
        self,
        num_entities: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        tower_type: str = "mlp",
        cat_cardinalities: list[int] | None = None,
        numeric_count: int = 0,
        feature_dim: int = DEFAULT_FEATURE_DIM,
    ) -> None:
        super().__init__()
        if tower_type not in {"embedding", "mlp"}:
            raise ValueError("tower_type must be 'embedding' or 'mlp'")
        cat_cardinalities = list(cat_cardinalities or [])
        self.tower_type = tower_type
        self.cat_cardinalities = cat_cardinalities
        self.numeric_count = int(numeric_count)
        self.has_features = bool(cat_cardinalities) or self.numeric_count > 0
        self.feature_dim = feature_dim

        if tower_type == "embedding" and not self.has_features:
            self.embedding = nn.Embedding(num_entities, output_dim)
            self.mlp = nn.Identity()
            self.cat_embeddings = nn.ModuleList()
        else:
            self.embedding = nn.Embedding(num_entities, embedding_dim)
            input_dim = embedding_dim + feature_dim * len(cat_cardinalities) + self.numeric_count
            self.cat_embeddings = nn.ModuleList(
                [nn.Embedding(cardinality, feature_dim) for cardinality in cat_cardinalities]
            )
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(
        self,
        entity_ids: torch.Tensor,
        cat_features: torch.Tensor | None = None,
        numeric_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        entity_embeddings = self.embedding(entity_ids)
        parts = [entity_embeddings]
        if self.has_features:
            if self.cat_embeddings:
                if cat_features is None:
                    raise ValueError("cat_features 不能为空（模型配置了类目特征）")
                parts.extend(
                    table(cat_features[:, column])
                    for column, table in enumerate(self.cat_embeddings)
                )
            if self.numeric_count:
                if numeric_features is None:
                    raise ValueError("numeric_features 不能为空（模型配置了数值特征）")
                parts.append(numeric_features)
        tower_vectors = self.mlp(torch.cat(parts, dim=1) if len(parts) > 1 else parts[0])
        return F.normalize(tower_vectors, p=2, dim=1)


class TwoTowerModel(nn.Module):
    """用户塔与物品塔各自独立，但归一化后的输出共享同一个向量空间。"""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
        temperature: float = 0.2,
        tower_type: str = "mlp",
        item_cat_cardinalities: list[int] | None = None,
        item_numeric_count: int = 0,
        user_cat_cardinalities: list[int] | None = None,
        user_numeric_count: int = 0,
        feature_dim: int = DEFAULT_FEATURE_DIM,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.item_cat_cardinalities = list(item_cat_cardinalities or [])
        self.item_numeric_count = int(item_numeric_count)
        self.user_cat_cardinalities = list(user_cat_cardinalities or [])
        self.user_numeric_count = int(user_numeric_count)
        self.uses_item_features = bool(self.item_cat_cardinalities) or self.item_numeric_count > 0
        self.uses_user_features = bool(self.user_cat_cardinalities) or self.user_numeric_count > 0

        self.user_tower = Tower(
            num_users,
            embedding_dim,
            hidden_dim,
            output_dim,
            tower_type,
            self.user_cat_cardinalities,
            self.user_numeric_count,
            feature_dim,
        )
        self.item_tower = Tower(
            num_items,
            embedding_dim,
            hidden_dim,
            output_dim,
            tower_type,
            self.item_cat_cardinalities,
            self.item_numeric_count,
            feature_dim,
        )
        self.temperature = temperature
        self.tower_type = tower_type

        # 侧特征随模型一起保存：推理时只需要 item_ids，不需要额外特征文件。
        if self.uses_item_features:
            self.register_buffer(
                "item_cat_features",
                torch.zeros(num_items, len(self.item_cat_cardinalities), dtype=torch.long),
            )
            self.register_buffer(
                "item_numeric_features",
                torch.zeros(num_items, self.item_numeric_count, dtype=torch.float32),
            )
        if self.uses_user_features:
            self.register_buffer(
                "user_cat_features",
                torch.zeros(num_users, len(self.user_cat_cardinalities), dtype=torch.long),
            )
            self.register_buffer(
                "user_numeric_features",
                torch.zeros(num_users, self.user_numeric_count, dtype=torch.float32),
            )

    def _user_inputs(self, user_ids: torch.Tensor):
        if not self.uses_user_features:
            return None, None
        cat = self.user_cat_features[user_ids] if self.user_cat_cardinalities else None
        numeric = self.user_numeric_features[user_ids] if self.user_numeric_count else None
        return cat, numeric

    def _item_inputs(self, item_ids: torch.Tensor):
        if not self.uses_item_features:
            return None, None
        cat = self.item_cat_features[item_ids] if self.item_cat_cardinalities else None
        numeric = self.item_numeric_features[item_ids] if self.item_numeric_count else None
        return cat, numeric

    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        cat, numeric = self._user_inputs(user_ids)
        return self.user_tower(user_ids, cat, numeric)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        cat, numeric = self._item_inputs(item_ids)
        return self.item_tower(item_ids, cat, numeric)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_embeddings = self.encode_users(user_ids)
        item_embeddings = self.encode_items(item_ids)
        # 第 i 行是「用户 i 与本批次内每一个物品」的相似度；对角线是正样本，
        # 其余位置天然成为 in-batch 负样本。
        logits = (user_embeddings @ item_embeddings.T) / self.temperature
        return logits
