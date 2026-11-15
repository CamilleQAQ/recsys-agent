"""多路召回：每路都用真实模型推理，并保留通道内名次与模型分数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from recsys.adapters.base import AdapterError, RecommenderAdapter
from recsys.data.dataset import build_evaluation_examples
from recsys.data.schema import PreparedData, seen_items_for_split


@dataclass(frozen=True)
class Candidate:
    item: int
    score: float | None
    rank: int


def _top_k(scores: np.ndarray, seen: set[int], k: int) -> list[Candidate]:
    row = np.asarray(scores, dtype=np.float64).copy()
    if seen:
        row[list(seen)] = -np.inf
    k = min(k, row.size)
    if k <= 0:
        return []
    top = np.argpartition(-row, k - 1)[:k]
    top = top[np.argsort(-row[top])]
    return [
        Candidate(int(item), float(row[item]), rank + 1)
        for rank, item in enumerate(top)
    ]


def _popularity_channel(
    data: PreparedData, users: Sequence[int], split: str, k: int
) -> dict[int, list[Candidate]]:
    counts = np.zeros(data.num_items, dtype=np.int64)
    for row in data.train_interactions:
        counts[row.item_idx] += 1
    seen = seen_items_for_split(data, split)
    return {
        int(user): _top_k(counts, seen.get(int(user), set()), k)
        for user in users
    }


def _embedding_channel(
    adapter: RecommenderAdapter,
    data: PreparedData,
    users: Sequence[int],
    split: str,
    k: int,
) -> dict[int, list[Candidate]]:
    user_vectors = adapter.encode_users([int(user) for user in users])
    item_vectors = adapter.encode_items(list(range(data.num_items)))
    seen = seen_items_for_split(data, split)
    output: dict[int, list[Candidate]] = {}
    batch = 64
    for start in range(0, len(users), batch):
        block_users = users[start : start + batch]
        scores = user_vectors[start : start + batch] @ item_vectors.T
        for row, user in enumerate(block_users):
            output[int(user)] = _top_k(
                scores[row], seen.get(int(user), set()), k
            )
    return output


def _mind_channel(
    adapter: RecommenderAdapter,
    data: PreparedData,
    users: Sequence[int],
    split: str,
    k: int,
    *,
    include_validation_in_test_history: bool = True,
) -> dict[int, list[Candidate]]:
    from recsys.data.dataset import build_observed_histories
    from recsys.models.mind import score_users

    config = adapter.checkpoint["model_config"]
    histories = build_observed_histories(
        data,
        split,
        include_validation_in_test_history=include_validation_in_test_history,
    )
    scores, _ = score_users(
        adapter.model,
        [int(user) for user in users],
        histories,
        data.num_items,
        max_length=int(config["max_length"]),
        device=adapter.device,
        routing_seed=1234,
    )
    seen = seen_items_for_split(data, split)
    return {
        int(user): _top_k(scores[row], seen.get(int(user), set()), k)
        for row, user in enumerate(users)
    }


def _sequence_channel(
    adapter: RecommenderAdapter,
    data: PreparedData,
    users: Sequence[int],
    split: str,
    k: int,
    *,
    include_validation_in_test_history: bool = True,
) -> dict[int, list[Candidate]]:
    model = adapter.model
    max_seq_len = int(adapter.checkpoint["model_config"]["max_seq_len"])
    examples = build_evaluation_examples(
        data,
        split=split,
        max_seq_len=max_seq_len,
        include_validation_in_test_history=include_validation_in_test_history,
    )
    # 同一用户可能有多个留出目标，历史相同；取第一条即可。
    first_by_user: dict[int, object] = {}
    for example in examples:
        first_by_user.setdefault(example.user_idx, example)
    ordered = [first_by_user[int(user)] for user in users if int(user) in first_by_user]
    seen = seen_items_for_split(data, split)
    output: dict[int, list[Candidate]] = {}
    model.eval()
    with torch.no_grad():
        batch_size = 64
        for start in range(0, len(ordered), batch_size):
            block = ordered[start : start + batch_size]
            sequences = torch.tensor(
                [example.input_sequence for example in block],
                dtype=torch.long,
                device=adapter.device,
            )
            scores = model.score_all_items(sequences).cpu().numpy()
            for row, example in enumerate(block):
                output[int(example.user_idx)] = _top_k(
                    scores[row], seen.get(int(example.user_idx), set()), k
                )
    for user in users:
        output.setdefault(int(user), [])
    return output


def _tiger_channel(
    adapter: RecommenderAdapter,
    data: PreparedData,
    users: Sequence[int],
    split: str,
    k: int,
    *,
    include_validation_in_test_history: bool = True,
) -> dict[int, list[Candidate]]:
    from recsys.evaluation.tiger_evaluator import generate_tiger_scored

    scored = generate_tiger_scored(
        adapter.model,
        data,
        split,
        adapter.codes,
        adapter.sid_index,
        users=[int(user) for user in users],
        k=k,
        include_validation_in_test_history=include_validation_in_test_history,
    )
    return {
        int(user): [
            Candidate(int(item), float(score), rank + 1)
            for rank, (item, score) in enumerate(rows)
        ]
        for user, rows in scored.items()
    }


def run_channel(
    adapter: RecommenderAdapter,
    data: PreparedData,
    users: Sequence[int],
    *,
    split: str,
    k: int,
    include_validation_in_test_history: bool = True,
) -> dict[int, list[Candidate]]:
    """按模型类型选择真实打分路径；无法打分时退回官方 ``recommend``。"""
    model_id = adapter.model_id
    if model_id == "popularity":
        return _popularity_channel(data, users, split, k)
    if model_id == "mind":
        return _mind_channel(
            adapter,
            data,
            users,
            split,
            k,
            include_validation_in_test_history=include_validation_in_test_history,
        )
    if model_id in {"sasrec", "bert4rec", "hstu"}:
        return _sequence_channel(
            adapter,
            data,
            users,
            split,
            k,
            include_validation_in_test_history=include_validation_in_test_history,
        )
    if model_id == "tiger":
        return _tiger_channel(
            adapter,
            data,
            users,
            split,
            k,
            include_validation_in_test_history=include_validation_in_test_history,
        )
    try:
        return _embedding_channel(adapter, data, users, split, k)
    except AdapterError:
        recommendations = adapter.recommend(data, split=split, k=k)
        return {
            int(user): [
                Candidate(int(item), None, rank + 1)
                for rank, item in enumerate(recommendations.get(int(user), []))
            ]
            for user in users
        }


__all__ = ["Candidate", "run_channel"]
