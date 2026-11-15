"""DCNv2（Deep & Cross Network v2）的项目适配实现。

上游参考：``third_party/FuxiCTR/model_zoo/DCNv2/src/DCNv2.py``。
核心是矩阵式 cross layer：``x_{l+1} = x_0 * (W_l x_l + b_l) + x_l``，
与 DNN 并行后拼接输出。本项目直接接收「类目编码 + 数值特征」批次，
不依赖 FuxiCTR 的 dataset_config，排序和广告 CTR 可共用。
"""

from __future__ import annotations

import torch
from torch import nn


class CrossNetV2(nn.Module):
    def __init__(self, input_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        for layer in self.layers:
            x = x0 * layer(x) + x
        return x


class DCNv2(nn.Module):
    def __init__(
        self,
        *,
        cat_cardinalities: tuple[int, ...] = (38, 294, 4, 30),
        numeric_count: int = 5,
        user_numeric_count: int = 2,
        embedding_dim: int = 8,
        cross_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, embedding_dim) for cardinality in cat_cardinalities]
        )
        input_dim = embedding_dim * len(cat_cardinalities) + numeric_count + user_numeric_count
        self.cross = CrossNetV2(input_dim, cross_layers)
        self.deep = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
        )
        self.output = nn.Linear(input_dim + hidden_dim // 2, 1)

    def _input(
        self,
        item_cat: torch.Tensor,
        item_numeric: torch.Tensor,
        user_numeric: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = [table(item_cat[:, column]) for column, table in enumerate(self.cat_embeddings)]
        return torch.cat([*embeddings, item_numeric, user_numeric], dim=1)

    def forward(
        self,
        *,
        item_cat: torch.Tensor,
        item_numeric: torch.Tensor,
        user_numeric: torch.Tensor,
        **_ignored,
    ) -> torch.Tensor:
        inputs = self._input(item_cat, item_numeric, user_numeric)
        cross = self.cross(inputs)
        deep = self.deep(inputs)
        return self.output(torch.cat([cross, deep], dim=1)).squeeze(1)


__all__ = ["CrossNetV2", "DCNv2"]
