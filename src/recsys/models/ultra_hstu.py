"""Ultra-HSTU 的项目适配：HSTU 序列主干 + 候选排序头。

上游参考：``third_party/TorchEasyRec/tzrec/models/ultra_hstu.py``。
上游把 HSTU 转导堆叠接在 CTR/时长预估头上；本项目复用
``models.hstu.HSTU`` 的因果编码器（相对位置偏置 + pointwise gating），
取序列末位表示，与候选物品 embedding、类目/数值特征拼接后过 MLP 出分。
"""

from __future__ import annotations

import torch
from torch import nn

from recsys.models.hstu import HSTU


class UltraHSTU(nn.Module):
    def __init__(
        self,
        num_items: int,
        *,
        max_seq_len: int = 50,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        cat_cardinalities: tuple[int, ...] = (38, 294, 4, 30),
        numeric_count: int = 5,
        user_numeric_count: int = 2,
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.max_seq_len = int(max_seq_len)
        self.encoder = HSTU(
            num_items=num_items,
            max_seq_len=max_seq_len,
            hidden_dim=embedding_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, embedding_dim) for cardinality in cat_cardinalities]
        )
        self.numeric_projection = nn.Linear(numeric_count, embedding_dim)
        feature_dim = embedding_dim * 3 + user_numeric_count
        self.output = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        *,
        history: torch.Tensor,
        candidates: torch.Tensor,
        item_cat: torch.Tensor,
        item_numeric: torch.Tensor,
        user_numeric: torch.Tensor,
    ) -> torch.Tensor:
        representation = self.encoder.sequence_representation(history)
        candidate = self.encoder.item_embedding(candidates + 1)
        for column, table in enumerate(self.cat_embeddings):
            candidate = candidate + table(item_cat[:, column])
        candidate = candidate + self.numeric_projection(item_numeric)
        features = torch.cat(
            [representation, candidate, representation * candidate, user_numeric],
            dim=1,
        )
        return self.output(features).squeeze(1)


__all__ = ["UltraHSTU"]
