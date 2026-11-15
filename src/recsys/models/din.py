"""DIN（深度兴趣网络）的项目适配实现。

上游参考：``third_party/FuxiCTR/model_zoo/DIN/src/DIN.py`` 与
``third_party/DeepCTR-Torch/deepctr_torch/models/din.py``。上游要求
VarLenSparseFeat + feature columns 的数据格式；本项目直接接收
``history item tokens + candidate item + 物品特征``，核心计算保持一致：

1. 对用户历史物品做 embedding；
2. 用候选物品作为 query 计算历史注意力权重（候选/历史/差/积四元组）；
3. 加权池化得到「局部激活兴趣」；
4. 与候选表示、用户特征拼接后过 MLP 输出 logit。

底层实现不依赖 FuxiCTR/DeepCTR，Phase 3 广告场景可直接复用本模块。
"""

from __future__ import annotations

import torch
from torch import nn


class DIN(nn.Module):
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
        vector = vector + self.numeric_projection(item_numeric)
        return vector

    def forward(
        self,
        *,
        history: torch.Tensor,
        candidates: torch.Tensor,
        item_cat: torch.Tensor,
        item_numeric: torch.Tensor,
        user_numeric: torch.Tensor,
    ) -> torch.Tensor:
        history_vectors = self.item_embedding(history)  # [B, H, D]
        candidate = self.candidate_vector(candidates, item_cat, item_numeric)  # [B, D]
        query = candidate.unsqueeze(1).expand_as(history_vectors)
        attention_input = torch.cat(
            [query, history_vectors, query - history_vectors, query * history_vectors],
            dim=-1,
        )
        scores = self.attention(attention_input).squeeze(-1)  # [B, H]
        padding = history.eq(0)
        scores = scores.masked_fill(padding, float("-inf"))
        # 全部是 padding 时退化为 0 向量（冷启动用户），避免 NaN。
        all_padding = padding.all(dim=1, keepdim=True)
        safe_scores = torch.softmax(
            torch.where(all_padding, torch.zeros_like(scores), scores), dim=1
        )
        weighted = (safe_scores.unsqueeze(-1) * history_vectors).sum(dim=1)
        weighted = weighted.masked_fill(all_padding, 0.0)
        features = torch.cat(
            [candidate, weighted, candidate * weighted, candidate - weighted, user_numeric],
            dim=1,
        )
        return self.output(features).squeeze(1)


__all__ = ["DIN"]
