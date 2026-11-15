"""TIGER：基于 Semantic ID 的生成式检索（本项目适配实现）。

上游参考：``third_party/genrec/genrec/models/tiger.py``（RQ-VAE 语义 ID +
Transformer 编码历史 + 自回归生成目标 SID + trie 约束解码）。上游依赖 gin /
wandb / 独立数据格式；这里按同一思路重写：

* 历史：用户最近若干物品的 Semantic ID 码字按顺序拉平；
* 编码器：双向 Transformer 编码历史；
* 解码器：以 BOS 起，自回归生成目标物品的 L 个码字；
* 推理：trie 约束的 beam search，只扩展在 Semantic ID 索引里出现过前缀的码字；
* 输出：SID → 物品列表，按累计 log-prob 排序并过滤已曝光物品。

因为 Semantic ID 是层级离散码，生成式召回天然只能产出“码本里存在的物品”，
这一点与上游 TIGER 一致。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from recsys.data.dataset import build_user_sequences
from recsys.data.schema import Interaction
from recsys.models.rqvae import SemanticIDIndex

PAD_TOKEN = 0
EOS_TOKEN = 1
CODE_OFFSET = 2
DEFAULT_MAX_HISTORY_ITEMS = 20


def sid_token(code: int) -> int:
    return int(code) + CODE_OFFSET


class TigerTransformer(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        num_levels: int,
        *,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        max_history_tokens: int = 60,
    ) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.num_levels = int(num_levels)
        self.max_history_tokens = int(max_history_tokens)
        self.bos_token = CODE_OFFSET + self.codebook_size
        self.vocab_size = self.bos_token + 1
        self.hidden_dim = int(hidden_dim)
        self.item_embedding = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_history_tokens, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_blocks)
        self.output = nn.Linear(hidden_dim, self.vocab_size)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def encode_history(self, history_tokens: torch.Tensor) -> torch.Tensor:
        padding = history_tokens.eq(PAD_TOKEN)
        positions = torch.arange(history_tokens.shape[1], device=history_tokens.device)
        hidden = self.item_embedding(history_tokens) * math.sqrt(self.hidden_dim)
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        hidden = hidden * (~padding).unsqueeze(-1)
        return self.encoder(hidden, src_key_padding_mask=padding)

    def decode(
        self, decoder_tokens: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        positions = torch.arange(decoder_tokens.shape[1], device=decoder_tokens.device)
        hidden = self.item_embedding(decoder_tokens) * math.sqrt(self.hidden_dim)
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        causal = torch.triu(
            torch.ones(
                (decoder_tokens.shape[1], decoder_tokens.shape[1]),
                dtype=torch.bool,
                device=decoder_tokens.device,
            ),
            diagonal=1,
        )
        hidden = self.decoder(hidden, memory, tgt_mask=causal)
        return self.output(hidden)

    def forward(
        self, history_tokens: torch.Tensor, target_codes: torch.Tensor
    ) -> torch.Tensor:
        """

        ``target_codes``: ``[B, L]`` 原始码字（0..codebook_size-1）。
        返回每个位置对 codebook_size 个码字的 logits ``[B, L, C]``。
        """
        memory = self.encode_history(history_tokens)
        decoder_input = torch.cat(
            [
                torch.full(
                    (target_codes.shape[0], 1),
                    self.bos_token,
                    dtype=torch.long,
                    device=target_codes.device,
                ),
                target_codes[:, :-1] + CODE_OFFSET,
            ],
            dim=1,
        )
        logits = self.decode(decoder_input, memory)
        return logits[:, :, CODE_OFFSET : CODE_OFFSET + self.codebook_size]

    @torch.no_grad()
    def generate_user(
        self,
        history_tokens: torch.Tensor,
        sid_index: SemanticIDIndex,
        *,
        k: int,
        beam_size: int = 32,
        blocked_items: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """单个用户的 trie 约束 beam search；返回 ``[(item_idx, logprob), ...]``。"""
        if history_tokens.numel() == 0:
            return []
        memory = self.encode_history(history_tokens.unsqueeze(0))
        beams: list[tuple[tuple[int, ...], float, list[int]]] = [((), 0.0, [self.bos_token])]
        blocked_items = blocked_items or set()
        for level in range(self.num_levels):
            input_ids = torch.tensor(
                [beam[2] for beam in beams], dtype=torch.long, device=history_tokens.device
            )
            logits = self.decode(input_ids, memory.expand(input_ids.shape[0], -1, -1))
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1).cpu()
            candidates: list[tuple[tuple[int, ...], float, list[int]]] = []
            for row, (prefix, score, tokens) in enumerate(beams):
                for code in sid_index.valid_next(prefix, level):
                    candidates.append(
                        (
                            prefix + (int(code),),
                            score + float(log_probs[row, CODE_OFFSET + int(code)]),
                            tokens + [CODE_OFFSET + int(code)],
                        )
                    )
            if not candidates:
                break
            candidates.sort(key=lambda item: -item[1])
            beams = candidates[:beam_size]
        results: dict[int, float] = {}
        for sid, score, _ in beams:
            for item in sid_index.items_for(sid):
                if item in blocked_items:
                    continue
                if item not in results or score > results[item]:
                    results[item] = score
        ranked = sorted(results.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:k]


class TigerTrainDataset(Dataset):
    """每个「用户 + 目标位置」一条样本；历史是之前物品的 SID 码字序列。"""

    def __init__(
        self,
        interactions: Sequence[Interaction],
        codes: np.ndarray,
        num_levels: int,
        max_history_items: int = DEFAULT_MAX_HISTORY_ITEMS,
        max_history_tokens: int = 60,
    ) -> None:
        self.sequences = build_user_sequences(interactions)
        self.sequences = {
            user: np.asarray(items, dtype=np.int64)
            for user, items in self.sequences.items()
            if len(items) >= 2
        }
        self.codes = np.asarray(codes, dtype=np.int64)
        self.num_levels = int(num_levels)
        self.max_history_items = int(max_history_items)
        self.max_history_tokens = int(max_history_tokens)
        self.users = np.asarray(sorted(self.sequences), dtype=np.int64)
        counts = np.asarray(
            [len(self.sequences[int(user)]) - 1 for user in self.users], dtype=np.int64
        )
        self.offsets = np.concatenate(([0], np.cumsum(counts)))
        if self.offsets[-1] == 0:
            raise ValueError("没有可用的 TIGER 训练样本")

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def _flat_tokens(self, items: np.ndarray) -> list[int]:
        tokens: list[int] = []
        for item in items[-self.max_history_items :]:
            tokens.extend(int(code) + CODE_OFFSET for code in self.codes[int(item)])
        return tokens[-self.max_history_tokens :]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user_position = int(np.searchsorted(self.offsets, index, side="right") - 1)
        user = int(self.users[user_position])
        sequence = self.sequences[user]
        target_position = index - int(self.offsets[user_position]) + 1
        history = self._flat_tokens(sequence[:target_position])
        target = [int(code) for code in self.codes[int(sequence[target_position])]]
        history_tokens = np.zeros(self.max_history_tokens, dtype=np.int64)
        history_tokens[: len(history)] = history
        return {
            "user_idx": torch.tensor(user, dtype=torch.long),
            "history_tokens": torch.from_numpy(history_tokens),
            "target_codes": torch.tensor(target, dtype=torch.long),
        }


__all__ = [
    "CODE_OFFSET",
    "DEFAULT_MAX_HISTORY_ITEMS",
    "EOS_TOKEN",
    "PAD_TOKEN",
    "TigerTrainDataset",
    "TigerTransformer",
    "sid_token",
]
