"""HSTU 的 PyTorch 适配实现（层次化序列转导单元，序列推荐）。

上游参考：``third_party/TorchEasyRec/tzrec/models/hstu.py`` 与
``third_party/generative-recommenders`` 的 HSTU 研究实现。上游依赖 tzrec /
fbgemm 运行时，本项目按论文结构重写核心部分：

* 因果自注意力 + **相对位置/时间偏置**（每个 head 一组可学习标量）；
* 注意力输出经过 **pointwise gating**（SiLU 门控），对应 HSTU 的
  "pointwise projected attention"；
* 前馈层与残差连接；
* 用序列末位表示对全量物品打分。

时间戳在本项目的数据契约里是请求级时间；为保持与 SASRec/BERT4Rec 相同的
输入契约，序列模型只使用物品序列，相对偏置按**位置距离**分桶（可视为
时间间隔的代理）。上游的绝对时间桶在 ``models/ultra_hstu.py`` 的排序头里
按候选位次复用。
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_NUM_BUCKETS = 32


class RelativeAttentionBias(nn.Module):
    """把相对位置距离映射为每个 head 一个标量偏置。"""

    def __init__(self, num_heads: int, num_buckets: int = DEFAULT_NUM_BUCKETS) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.num_buckets = int(num_buckets)
        self.table = nn.Embedding(num_buckets, num_heads)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        distance = (positions[None, :] - positions[:, None]).abs()
        # 0..num_buckets-1 分桶，超出部分归入最后一个桶（对数尺度更贴近长序列）。
        buckets = torch.clamp(
            torch.log2(distance.float() + 1.0).long(), max=self.num_buckets - 1
        )
        bias = self.table(buckets)  # [L, L, H]
        return bias.permute(2, 0, 1)  # [H, L, L]


class HSTUBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        num_buckets: int = DEFAULT_NUM_BUCKETS,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.relative_bias = RelativeAttentionBias(num_heads, num_buckets)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        seq_len = hidden.shape[1]
        normalized = self.attention_norm(hidden)
        causal = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=hidden.device),
            diagonal=1,
        )
        bias = self.relative_bias(seq_len, hidden.device)  # [H, L, L]
        bias = bias.masked_fill(causal.unsqueeze(0), float("-inf"))
        batch_heads = hidden.shape[0] * self.attention.num_heads
        attention_mask = bias.unsqueeze(0).expand(hidden.shape[0], -1, -1, -1)
        attention_mask = attention_mask.reshape(batch_heads, seq_len, seq_len)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        # HSTU 的 pointwise gating：SiLU 门控逐元素调制注意力输出。
        hidden = hidden + self.dropout(attended * F.silu(self.gate(attended)))
        hidden = hidden * (~padding_mask).unsqueeze(-1)
        hidden = hidden + self.dropout(self.feed_forward(self.feed_forward_norm(hidden)))
        return hidden * (~padding_mask).unsqueeze(-1)


class HSTU(nn.Module):
    """因果 HSTU 编码器；对外暴露与 SASRec 相同的序列打分接口。"""

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        num_buckets: int = DEFAULT_NUM_BUCKETS,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        self.num_items = int(num_items)
        self.max_seq_len = int(max_seq_len)
        self.hidden_dim = int(hidden_dim)
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                HSTUBlock(hidden_dim, num_heads, dropout, num_buckets)
                for _ in range(num_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def forward(self, item_sequence: torch.Tensor) -> torch.Tensor:
        if item_sequence.shape[1] > self.max_seq_len:
            raise ValueError("序列长度超过 max_seq_len")
        padding_mask = item_sequence.eq(0)
        hidden = self.item_embedding(item_sequence) * math.sqrt(self.hidden_dim)
        positions = torch.arange(item_sequence.shape[1], device=item_sequence.device)
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        hidden = self.input_dropout(hidden) * (~padding_mask).unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden, padding_mask)
        return self.output_norm(hidden) * (~padding_mask).unsqueeze(-1)

    def sequence_representation(self, item_sequence: torch.Tensor) -> torch.Tensor:
        hidden = self.forward(item_sequence)
        valid = item_sequence.ne(0).sum(dim=1)
        batch = torch.arange(item_sequence.shape[0], device=item_sequence.device)
        # 历史可能为空（严格早于请求的过滤后没有交互）；返回零向量，
        # 而不是让排序/推理在冷启动用户上直接报错。
        positions = torch.clamp(valid - 1, min=0)
        representation = hidden[batch, positions]
        return representation * valid.gt(0).unsqueeze(1).to(representation.dtype)

    def score_all_items(self, item_sequence: torch.Tensor) -> torch.Tensor:
        representation = self.sequence_representation(item_sequence)
        return representation @ self.item_embedding.weight[1:].T

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_embedding(item_ids + 1)


__all__ = ["DEFAULT_NUM_BUCKETS", "HSTU", "HSTUBlock", "RelativeAttentionBias"]
