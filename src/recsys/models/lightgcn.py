"""LightGCN：用户-物品二部图上的无特征线性传播（本项目适配实现）。

上游参考：``third_party/RecBole/recbole/model/general_recommender/lightgcn.py``
与 ``third_party/MMRec/src/models/lightgcn.py``。两者内部结构依赖各自框架
（RecBole 的 AbstractRecommender、MMRec 的 torch_geometric），无法在
Python 3.14 / 纯 torch 主环境直接实例化；这里按论文与上游前向逻辑重写一个
最小可运行版本，数据、训练和评测仍走本项目的统一契约。

核心公式（与论文一致）::

    e_u^(k+1) = sum_{i in N(u)} e_i^(k) / sqrt(|N(u)| |N(i)|)
    e_i^(k+1) = sum_{u in N(i)} e_u^(k) / sqrt(|N(u)| |N(i)|)
    e = mean(e^0, e^1, ..., e^K)

没有自连接、没有非线性、没有可学习传播权重；只有第 0 层 embedding 是参数。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

DEFAULT_EMBEDDING_DIM = 64
DEFAULT_LAYERS = 2


def build_normalized_adjacency(
    user_indices: np.ndarray,
    item_indices: np.ndarray,
    num_users: int,
    num_items: int,
    device: torch.device,
) -> torch.Tensor:
    """把训练交互转成对称归一化的稀疏邻接矩阵（用户在前、物品在后）。

    返回 ``[num_users + num_items, num_users + num_items]`` 的 COO 稀疏张量。
    同一 (user, item) 重复出现只保留一次边（隐式反馈场景重复观看不改变图结构）。
    """
    if len(user_indices) != len(item_indices):
        raise ValueError("user_indices 与 item_indices 长度必须一致")
    users = np.asarray(user_indices, dtype=np.int64)
    items = np.asarray(item_indices, dtype=np.int64) + int(num_users)
    rows = np.concatenate([users, items])
    columns = np.concatenate([items, users])
    if rows.size == 0:
        raise ValueError("训练集没有交互，无法构图")

    # 去重边，避免重复曝光把权重放大。
    pairs = np.unique(np.stack([rows, columns], axis=1), axis=0)
    rows, columns = pairs[:, 0], pairs[:, 1]

    degree = np.bincount(rows, minlength=num_users + num_items).astype(np.float64)
    degree[degree == 0] = 1.0
    norm = (1.0 / np.sqrt(degree))[rows] * (1.0 / np.sqrt(degree))[columns]
    indices = torch.from_numpy(np.stack([rows, columns], axis=0)).to(device)
    values = torch.from_numpy(norm.astype(np.float32)).to(device)
    size = (num_users + num_items, num_users + num_items)
    return torch.sparse_coo_tensor(indices, values, size=size).coalesce()


class LightGCN(nn.Module):
    """只有一层可学习 embedding 的图协同过滤模型。"""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        num_layers: int = DEFAULT_LAYERS,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim 必须为正")
        if num_layers < 0:
            raise ValueError("num_layers 不能为负")
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)
        self.num_layers = int(num_layers)
        self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)

    def propagate(self, adjacency: torch.Tensor) -> torch.Tensor:
        """返回所有节点的最终表示 ``[num_users + num_items, dim]``。"""
        if adjacency.shape[0] != self.num_users + self.num_items:
            raise ValueError("邻接矩阵尺寸与模型词表不一致")
        hidden = self.embedding.weight
        collected = [hidden]
        for _ in range(self.num_layers):
            hidden = torch.sparse.mm(adjacency, hidden)
            collected.append(hidden)
        return torch.stack(collected, dim=0).mean(dim=0)

    def split_embeddings(self, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        all_embeddings = self.propagate(adjacency)
        return (
            all_embeddings[: self.num_users],
            all_embeddings[self.num_users :],
        )

    def encode_users(
        self, user_ids: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        return self.split_embeddings(adjacency)[0][user_ids]

    def encode_items(
        self, item_ids: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        return self.split_embeddings(adjacency)[1][item_ids]


def bpr_loss(
    user_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    *,
    reg_weight: float = 1e-4,
) -> torch.Tensor:
    """标准 BPR pairwise loss + L2 正则（与 RecBole 的 BPR loss 口径一致）。"""
    positive_scores = (user_embeddings * positive_embeddings).sum(dim=1)
    negative_scores = (user_embeddings * negative_embeddings).sum(dim=1)
    ranking_loss = -torch.nn.functional.logsigmoid(positive_scores - negative_scores).mean()
    regularization = reg_weight * (
        user_embeddings.pow(2).sum()
        + positive_embeddings.pow(2).sum()
        + negative_embeddings.pow(2).sum()
    ) / max(1, user_embeddings.shape[0])
    return ranking_loss + regularization


def edges_from_interactions(
    interactions: Sequence[object],
) -> tuple[np.ndarray, np.ndarray]:
    """从交互序列里抽 (user_idx, item_idx) 两列（不物化 dataclass 列表）。"""
    users = np.fromiter((row.user_idx for row in interactions), dtype=np.int64)
    items = np.fromiter((row.item_idx for row in interactions), dtype=np.int64)
    return users, items


__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_LAYERS",
    "LightGCN",
    "bpr_loss",
    "build_normalized_adjacency",
    "edges_from_interactions",
]
