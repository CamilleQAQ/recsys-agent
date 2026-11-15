"""BERT4Rec：双向 Transformer + 掩码物品预测（本项目 PyTorch 适配实现）。

上游参考：``third_party/BERT4Rec/modeling.py``（TF1.12 + tf.contrib）。
上游无法与当前 Python 3.14 / PyTorch 主环境共存，因此这里按论文与上游结构
重写：双向 TransformerEncoder、Cloze 掩码目标、权重共享的输出层。

token 空间与 SASRec 一致：0 = padding，1..num_items = 物品，
``num_items + 1`` = 掩码 token（MASK）。
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from recsys.data.schema import Interaction
from recsys.data.dataset import build_user_sequences

PAD_TOKEN = 0
MASK_TOKEN_OFFSET = 1  # MASK = num_items + 1


class BERT4RecBlock(nn.Module):
    """Pre-LN 的 Transformer 块（自注意力 + FFN），不使用因果掩码。"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden = hidden + self.dropout(attended)
        hidden = hidden * (~padding_mask).unsqueeze(-1)
        hidden = hidden + self.dropout(self.feed_forward(self.feed_forward_norm(hidden)))
        return hidden * (~padding_mask).unsqueeze(-1)


class BERT4Rec(nn.Module):
    """双向序列编码器；训练时预测被 MASK 的位置，推理时在序列末尾查询 MASK。"""

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        self.num_items = int(num_items)
        self.max_seq_len = int(max_seq_len)
        self.hidden_dim = int(hidden_dim)
        self.mask_token = int(num_items) + MASK_TOKEN_OFFSET
        # 权重共享：输入 embedding 同时作为输出层（与上游一致）。
        self.item_embedding = nn.Embedding(num_items + 2, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [BERT4RecBlock(hidden_dim, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.max_seq_len:
            raise ValueError("序列长度超过 max_seq_len")
        padding_mask = tokens.eq(PAD_TOKEN)
        hidden = self.item_embedding(tokens) * math.sqrt(self.hidden_dim)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        hidden = self.input_dropout(hidden) * (~padding_mask).unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden, padding_mask)
        return self.output_norm(hidden)

    def score_all_items(self, item_sequence: torch.Tensor) -> torch.Tensor:
        """在历史末尾追加 MASK，用该位置的双向表示给全部物品打分。"""
        batch_size = item_sequence.shape[0]
        query = torch.full(
            (batch_size, self.max_seq_len),
            PAD_TOKEN,
            dtype=torch.long,
            device=item_sequence.device,
        )
        query_positions = torch.zeros(batch_size, dtype=torch.long, device=item_sequence.device)
        for row in range(batch_size):
            valid = item_sequence[row][item_sequence[row].ne(PAD_TOKEN)]
            # 留一个位置给末尾 MASK；超长时丢掉最旧的物品。
            kept = valid[-(self.max_seq_len - 1) :]
            query[row, : kept.numel()] = kept
            query[row, kept.numel()] = self.mask_token
            query_positions[row] = kept.numel()
        hidden = self.forward(query)
        representation = hidden[
            torch.arange(batch_size, device=item_sequence.device), query_positions
        ]
        return representation @ self.item_embedding.weight[1 : self.num_items + 1].T

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """物品表示（token 整体 +1）；与输入 embedding 共享。"""
        return self.item_embedding(item_ids + 1)


class Bert4RecTrainDataset(Dataset):
    """每个「用户 + 目标位置」一条样本；目标位置总是被 MASK，其余位置按 15% 随机 MASK。"""

    def __init__(
        self,
        train_interactions: Sequence[Interaction],
        num_items: int,
        max_seq_len: int,
        seed: int = 42,
        mask_prob: float = 0.15,
    ) -> None:
        sequences = build_user_sequences(train_interactions)
        self.sequences = {
            user: np.asarray(items, dtype=np.int64) + 1
            for user, items in sequences.items()
            if len(items) >= 1
        }
        self.users = np.asarray(sorted(self.sequences), dtype=np.int64)
        counts = np.asarray(
            [len(self.sequences[int(user)]) for user in self.users], dtype=np.int64
        )
        self.offsets = np.concatenate(([0], np.cumsum(counts)))
        self.num_items = int(num_items)
        self.max_seq_len = int(max_seq_len)
        self.mask_token = int(num_items) + MASK_TOKEN_OFFSET
        self.mask_prob = float(mask_prob)
        self.seed = int(seed)
        if self.offsets[-1] == 0:
            raise ValueError("没有可用的训练序列")

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user_position = int(np.searchsorted(self.offsets, index, side="right") - 1)
        user = int(self.users[user_position])
        sequence = self.sequences[user]
        target_position = index - int(self.offsets[user_position])
        window = sequence[max(0, target_position - self.max_seq_len + 1) : target_position + 1]
        tokens = np.zeros(self.max_seq_len, dtype=np.int64)
        labels = np.zeros(self.max_seq_len, dtype=np.int64)
        tokens[: len(window)] = window
        rng = random.Random(self.seed * 1_000_003 + index)
        for position in range(len(window)):
            # 目标位置（window 末尾）必须被 MASK；上下文按 mask_prob 随机 MASK。
            if position == len(window) - 1 or rng.random() < self.mask_prob:
                labels[position] = int(window[position])
                tokens[position] = self.mask_token
        return {
            "user_idx": torch.tensor(user, dtype=torch.long),
            "input_sequence": torch.from_numpy(tokens),
            "labels": torch.from_numpy(labels),
        }


__all__ = [
    "BERT4Rec",
    "Bert4RecTrainDataset",
    "MASK_TOKEN_OFFSET",
    "PAD_TOKEN",
]
