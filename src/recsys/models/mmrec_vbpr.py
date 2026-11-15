"""MMRec 多模态模型族的本项目适配：VBPR 风格的内容增强协同过滤。

上游参考：``third_party/MMRec/src/models/vbpr.py``（视觉贝叶斯个性化排序）。
上游依赖 torch_geometric/torch_scatter 与预提取的视觉/文本特征；当前环境
（Python 3.14）无法安装这些依赖，本地 KuaiSAR 快照也没有封面/文本 embedding。
因此这里做**最小可运行适配**：

* 物品内容向量来自 ``models.rqvae.build_content_vectors``（类目/类型/作者/日期）；
* VBPR 打分 = 用户 ID 向量·物品 ID 向量 + 用户内容偏好·内容投影；
* BPR 目标训练，与 MMRec 的 VBPR 公式一致，只是“模态特征”换成元数据向量。

明确限制：这不是真正的视觉/文本多模态模型；要跑上游 LATTICE/VBPR 原实现，
需要 torch_geometric + torch_scatter + 视觉特征文件（见 docs/model-zoo.md）。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from recsys.models.rqvae import build_content_vectors


class MMRecVBPR(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        content_dim: int = 32,
        embedding_dim: int = 64,
        content_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)
        self.content_dim = int(content_dim)
        self.content_embed_dim = int(content_embed_dim)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_content = nn.Embedding(num_users, content_embed_dim)
        self.content_projection = nn.Linear(content_dim, content_embed_dim, bias=False)
        self.item_bias = nn.Embedding(num_items, 1)
        # 内容矩阵随模型保存：推理时只给 item_id 就能算完整分数。
        self.register_buffer("item_content", torch.zeros(num_items, content_dim))
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.user_content.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

    def set_content(self, content: np.ndarray) -> None:
        if content.shape[0] != self.num_items:
            raise ValueError("内容向量行数与物品数不一致")
        self.item_content.copy_(torch.from_numpy(content.astype(np.float32)))

    def _projected_content(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.content_projection(self.item_content[item_ids])

    def score(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        collaborative = (self.user_embedding(user_ids) * self.item_embedding(item_ids)).sum(dim=1)
        content = (self.user_content(user_ids) * self._projected_content(item_ids)).sum(dim=1)
        return collaborative + content + self.item_bias(item_ids).squeeze(1)

    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.user_embedding(user_ids), self.user_content(user_ids)], dim=1)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.item_embedding(item_ids), self._projected_content(item_ids)], dim=1
        )


def vbpr_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    *,
    reg_weight: float = 1e-5,
    users: torch.Tensor | None = None,
    positive_items: torch.Tensor | None = None,
    negative_items: torch.Tensor | None = None,
    model: MMRecVBPR | None = None,
) -> torch.Tensor:
    loss = -torch.nn.functional.logsigmoid(positive_scores - negative_scores).mean()
    if model is not None and users is not None and positive_items is not None:
        regularization = (
            model.user_embedding(users).pow(2).sum()
            + model.item_embedding(positive_items).pow(2).sum()
            + model.item_embedding(negative_items).pow(2).sum()
        ) / max(1, users.numel())
        loss = loss + reg_weight * regularization
    return loss


def build_content(item_features, seed: int = 0, dim: int = 32) -> np.ndarray:
    return build_content_vectors(item_features, seed=seed, dim=dim)


__all__ = ["MMRecVBPR", "build_content", "vbpr_loss"]
