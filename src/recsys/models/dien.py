"""DIEN（深度兴趣演化网络）的项目适配实现。

上游参考：``third_party/DeepCTR-Torch/deepctr_torch/models/dien.py``。
核心是「兴趣抽取层（GRU）+ 兴趣演化层（AUGRU）」：AUGRU 用候选物品算出的
注意力缩放 GRU 的更新门，让兴趣随候选方向演化。这里按论文重写 AUGRU cell，
不依赖 DeepCTR 的 feature-column 体系，Phase 3 广告排序可直接复用。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AUGRUCell(nn.Module):
    """单步 AUGRU：``z' = a * z``，其余与标准 GRU 相同。"""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.weight_ih = nn.Linear(input_dim, 3 * hidden_dim)
        self.weight_hh = nn.Linear(hidden_dim, 3 * hidden_dim)

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor, attention: torch.Tensor
    ) -> torch.Tensor:
        ih = self.weight_ih(x)
        hh = self.weight_hh(hidden)
        reset = torch.sigmoid(ih[:, : self.hidden_dim] + hh[:, : self.hidden_dim])
        update = torch.sigmoid(
            ih[:, self.hidden_dim : 2 * self.hidden_dim]
            + hh[:, self.hidden_dim : 2 * self.hidden_dim]
        )
        candidate = torch.tanh(
            ih[:, 2 * self.hidden_dim :]
            + reset * hh[:, 2 * self.hidden_dim :]
        )
        update = update * attention.unsqueeze(1)
        return (1.0 - update) * candidate + update * hidden


class DIEN(nn.Module):
    def __init__(
        self,
        num_items: int,
        *,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        attention_dim: int = 32,
        cat_cardinalities: tuple[int, ...] = (38, 294, 4, 30),
        numeric_count: int = 5,
        user_numeric_count: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, embedding_dim) for cardinality in cat_cardinalities]
        )
        self.numeric_projection = nn.Linear(numeric_count, embedding_dim)
        self.augru = AUGRUCell(embedding_dim, embedding_dim)
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 4, attention_dim),
            nn.SiLU(),
            nn.Linear(attention_dim, 1),
        )
        self.output = nn.Sequential(
            nn.Linear(embedding_dim * 4 + user_numeric_count, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def candidate_vector(
        self,
        candidates: torch.Tensor,
        item_cat: torch.Tensor,
        item_numeric: torch.Tensor,
    ) -> torch.Tensor:
        vector = self.item_embedding(candidates + 1)
        for column, table in enumerate(self.cat_embeddings):
            vector = vector + table(item_cat[:, column])
        return vector + self.numeric_projection(item_numeric)

    def forward(
        self,
        *,
        history: torch.Tensor,
        candidates: torch.Tensor,
        item_cat: torch.Tensor,
        item_numeric: torch.Tensor,
        user_numeric: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.item_embedding(history)  # [B, H, D]
        candidate = self.candidate_vector(candidates, item_cat, item_numeric)
        hidden = torch.zeros_like(candidate)
        for position in range(history.shape[1]):
            current = embeddings[:, position]
            attention_input = torch.cat(
                [hidden, candidate, hidden * candidate, hidden - candidate], dim=1
            )
            attention = torch.sigmoid(self.attention(attention_input)).squeeze(1)
            updated = self.augru(current, hidden, attention)
            mask = history[:, position].ne(0).unsqueeze(1)
            hidden = torch.where(mask, updated, hidden)
        features = torch.cat(
            [hidden, candidate, hidden * candidate, hidden - candidate, user_numeric],
            dim=1,
        )
        return self.output(features).squeeze(1)


__all__ = ["AUGRUCell", "DIEN"]
