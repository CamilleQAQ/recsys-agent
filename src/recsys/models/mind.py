"""MIND（多兴趣召回）的模型封装：把 torch-rechub 的实现接进本项目。

**为什么引入 torch-rechub**：MIND 的参考实现只在 TorchEasyRec（依赖 torchrec/fbgemm，
只有 Linux wheel）里；torch-rechub 是纯 PyTorch + MIT 的实现，本机可直接跑。
我们只借它的模型本体，数据、训练循环、评测仍走本项目的统一契约。

MIND 的核心：用户历史序列 → 胶囊网络 + 动态路由 → K 个兴趣向量；
物品塔与历史序列**共享同一张 embedding 表**（`shared_with="item_id"`）。

注意 token 空间：本项目把 `item_idx` 整体 +1，token 0 留给 padding，
vocab_size 相应 +1。用户 id 同样 +1。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from torch import nn

from recsys.data.schema import PreparedData
from recsys.data.dataset import MindTrainDataset, pad_item_sequence

DEFAULT_EMBED_DIM = 64
DEFAULT_MAX_LENGTH = 20
DEFAULT_INTEREST_NUM = 4
DEFAULT_TEMPERATURE = 0.1

#: 评测/验证阶段固定胶囊路由的随机初始化，保证同一 checkpoint 可复现。
#: 训练阶段保持随机（等价于对路由做随机初始化增强），不受此常量影响。
EVAL_ROUTING_SEED = 1234


@contextmanager
def deterministic_routing(seed: int = EVAL_ROUTING_SEED):
    """临时固定 torch 全局 RNG，退出时恢复原状态（不污染训练随机流）。

    torch-rechub 的 ``CapsuleNetwork`` 用 ``torch.randn`` 初始化动态路由的
    对数权重，因此**同一 checkpoint 每次前向得到的兴趣向量都不同**。
    评测必须固定这个随机源，否则指标不可复现、验证集选模也会被噪声影响。
    """
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)


def build_features(num_users: int, num_items: int, embed_dim: int):
    """创建 torch-rechub 的三个特征：用户、物品、历史序列（与物品共享 embedding）。"""
    from torch_rechub.basic.features import SequenceFeature, SparseFeature

    user = SparseFeature("user_id", vocab_size=num_users + 1, embed_dim=embed_dim)
    item = SparseFeature("item_id", vocab_size=num_items + 1, embed_dim=embed_dim)
    history = SequenceFeature(
        "hist_item",
        vocab_size=num_items + 1,
        embed_dim=embed_dim,
        pooling="concat",
        shared_with="item_id",
    )
    return user, item, history


def build_mind_model(
    num_users: int,
    num_items: int,
    *,
    embed_dim: int = DEFAULT_EMBED_DIM,
    max_length: int = DEFAULT_MAX_LENGTH,
    interest_num: int = DEFAULT_INTEREST_NUM,
    temperature: float = DEFAULT_TEMPERATURE,
) -> nn.Module:
    """构造 torch-rechub 的 MIND；返回的模型用 ``mode`` 属性切换训练/推理形态。"""
    from torch_rechub.models.matching import MIND

    user, item, history = build_features(num_users, num_items, embed_dim)
    model = MIND(
        [user],
        [history],
        [item],
        [item],
        max_length=max_length,
        temperature=temperature,
        interest_num=interest_num,
    )
    return model


def build_batch(
    user_indices: np.ndarray | torch.Tensor,
    histories: np.ndarray,
    item_indices: np.ndarray | torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """按 torch-rechub 的输入格式组装一个 batch（user_id / hist_item / item_id）。"""
    users = torch.as_tensor(user_indices, dtype=torch.long, device=device) + 1
    hist = torch.as_tensor(histories, dtype=torch.long, device=device)
    items = torch.as_tensor(item_indices, dtype=torch.long, device=device).view(-1, 1)
    return {"user_id": users, "hist_item": hist, "item_id": items}


@torch.no_grad()
def encode_item_matrix(model: nn.Module, num_items: int, device: torch.device) -> np.ndarray:
    """物品 embedding 矩阵 [num_items, D]；token 从 1 开始。"""
    model.eval()
    model.mode = "item"
    tokens = torch.arange(1, num_items + 1, dtype=torch.long, device=device)
    embeddings = model({"item_id": tokens}).float().cpu().numpy()
    return np.ascontiguousarray(embeddings)


def score_users(
    model: nn.Module,
    user_indices: list[int],
    histories: dict[int, list[int]],
    num_items: int,
    *,
    routing_seed: int = EVAL_ROUTING_SEED,
    **kwargs,
):
    """评测入口：固定路由随机种子后调用实现，保证同一 checkpoint 指标可复现。"""
    with deterministic_routing(routing_seed):
        return _score_users_impl(model, user_indices, histories, num_items, **kwargs)


@torch.no_grad()
def _score_users_impl(
    model: nn.Module,
    user_indices: list[int],
    histories: dict[int, list[int]],
    num_items: int,
    *,
    max_length: int,
    device: torch.device,
    user_batch_size: int = 64,
    item_chunk_size: int = 65_536,
) -> tuple[np.ndarray, np.ndarray]:
    """给每个用户算「对全部物品的分数」，分数 = max over K 个兴趣。

    返回 ``(scores [num_users, num_items], attention [num_users, K])``；
    attention 是同批物品里的最大相似度，仅用于排查，不参与 Top-K。

    实现上是「用户批 × 物品块」的矩阵乘：``[B*K, D] @ [D, chunk]``，
    比逐用户循环快一个数量级；分块是为了避免 ``[B, K, num_items]`` 撑爆内存。
    """
    model.eval()
    item_embeddings = encode_item_matrix(model, num_items, device).astype(np.float32)
    model.mode = "user"  # encode_item_matrix 会把 mode 切成 "item"，这里切回来
    scores = np.empty((len(user_indices), num_items), dtype=np.float32)
    attention = np.empty((len(user_indices), int(getattr(model, "interest_num", 1))), dtype=np.float32)

    for start in range(0, len(user_indices), user_batch_size):
        batch_users = user_indices[start : start + user_batch_size]
        hist = np.stack(
            [
                np.asarray(pad_item_sequence(histories.get(user, []), max_length), dtype=np.int64)
                for user in batch_users
            ]
        )
        batch = {
            "user_id": torch.as_tensor(batch_users, dtype=torch.long, device=device) + 1,
            "hist_item": torch.as_tensor(hist, dtype=torch.long, device=device),
        }
        interests = model.user_tower(batch, interest_mask=None)  # [B, K, D]
        interests = interests.float().cpu().numpy()
        flat_interests = interests.reshape(-1, interests.shape[-1])  # [B*K, D]
        batch_scores = np.empty((len(batch_users), num_items), dtype=np.float32)
        for chunk_start in range(0, num_items, item_chunk_size):
            chunk_end = min(chunk_start + item_chunk_size, num_items)
            chunk = item_embeddings[chunk_start:chunk_end]  # [C, D]
            # [B*K, C] -> [B, K, C] -> max over interests
            chunk_scores = (flat_interests @ chunk.T).reshape(
                len(batch_users), interests.shape[1], chunk_end - chunk_start
            )
            batch_scores[:, chunk_start:chunk_end] = chunk_scores.max(axis=1)
        scores[start : start + len(batch_users)] = batch_scores
        attention[start : start + len(batch_users)] = interests.max(axis=2)
    return scores, attention


def rank_users(
    model: nn.Module,
    user_indices: list[int],
    histories: dict[int, list[int]],
    num_items: int,
    *,
    routing_seed: int = EVAL_ROUTING_SEED,
    **kwargs,
):
    """评测入口：固定路由随机种子后调用实现，保证同一 checkpoint 指标可复现。"""
    with deterministic_routing(routing_seed):
        return _rank_users_impl(model, user_indices, histories, num_items, **kwargs)


@torch.no_grad()
def _rank_users_impl(
    model: nn.Module,
    user_indices: list[int],
    histories: dict[int, list[int]],
    num_items: int,
    *,
    max_length: int,
    device: torch.device,
    k: int,
    seen_items: Any | None = None,
    user_batch_size: int = 64,
    item_chunk_size: int = 65_536,
) -> dict[int, list[int]]:
    """逐用户批算分并直接取 Top-K，不物化 ``[users, items]`` 全量分数。

    与 ``score_users`` 的区别只在内存：打分矩阵是 ``[batch, num_items]``
    （64 × 454,717 × 4B ≈ 116 MB），而不是 ``[全部用户, num_items]``（3.6 GB）。
    评测和验证都走这个入口，避免多进程并行时把内存打爆。
    """
    model.eval()
    item_embeddings = encode_item_matrix(model, num_items, device).astype(np.float32)
    model.mode = "user"
    recommendations: dict[int, list[int]] = {}
    k = min(k, num_items)

    for start in range(0, len(user_indices), user_batch_size):
        batch_users = user_indices[start : start + user_batch_size]
        hist = np.stack(
            [
                np.asarray(pad_item_sequence(histories.get(user, []), max_length), dtype=np.int64)
                for user in batch_users
            ]
        )
        batch = {
            "user_id": torch.as_tensor(batch_users, dtype=torch.long, device=device) + 1,
            "hist_item": torch.as_tensor(hist, dtype=torch.long, device=device),
        }
        interests = model.user_tower(batch, interest_mask=None).float().cpu().numpy()
        flat_interests = interests.reshape(-1, interests.shape[-1])
        batch_scores = np.empty((len(batch_users), num_items), dtype=np.float32)
        for chunk_start in range(0, num_items, item_chunk_size):
            chunk_end = min(chunk_start + item_chunk_size, num_items)
            chunk = item_embeddings[chunk_start:chunk_end]
            chunk_scores = (flat_interests @ chunk.T).reshape(
                len(batch_users), interests.shape[1], chunk_end - chunk_start
            )
            batch_scores[:, chunk_start:chunk_end] = chunk_scores.max(axis=1)

        for row, user in enumerate(batch_users):
            row_scores = batch_scores[row]
            blocked = seen_items.get(user, set()) if seen_items is not None else set()
            if blocked:
                row_scores[list(blocked)] = -np.inf
            top = np.argpartition(-row_scores, k - 1)[:k]
            top = top[np.argsort(-row_scores[top])]
            recommendations[int(user)] = [int(item) for item in top]
        del batch_scores
    return recommendations


def load_mind_from_checkpoint(
    checkpoint: dict[str, Any], device: torch.device
) -> nn.Module:
    """按 checkpoint 里保存的模型配置重建 MIND 并加载权重。"""
    config = checkpoint["model_config"]
    model = build_mind_model(
        config["num_users"],
        config["num_items"],
        embed_dim=config["embed_dim"],
        max_length=config["max_length"],
        interest_num=config["interest_num"],
        temperature=config["temperature"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def default_model_config(
    num_users: int,
    num_items: int,
    *,
    embed_dim: int = DEFAULT_EMBED_DIM,
    max_length: int = DEFAULT_MAX_LENGTH,
    interest_num: int = DEFAULT_INTEREST_NUM,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict[str, Any]:
    return {
        "num_users": int(num_users),
        "num_items": int(num_items),
        "embed_dim": int(embed_dim),
        "max_length": int(max_length),
        "interest_num": int(interest_num),
        "temperature": float(temperature),
    }


def build_mind_train_dataset(
    prepared_data: PreparedData,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    seed: int = 42,
) -> MindTrainDataset:
    return MindTrainDataset(
        prepared_data.train_interactions,
        num_items=prepared_data.num_items,
        max_seq_len=max_length,
        seed=seed,
    )
