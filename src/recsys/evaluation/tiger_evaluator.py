"""TIGER 的全量召回评测：按用户历史做 trie 约束生成。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from recsys.data.dataset import build_observed_histories
from recsys.data.schema import PreparedData, seen_items_for_split
from recsys.evaluation.evaluator import evaluate_recommendations, group_items_by_user
from recsys.evaluation.metrics import catalog_coverage
from recsys.models.rqvae import SemanticIDIndex
from recsys.models.tiger import CODE_OFFSET, TigerTransformer


def _history_tokens(
    model: TigerTransformer,
    history: Sequence[int],
    codes: np.ndarray,
    max_history_items: int,
) -> torch.Tensor:
    tokens: list[int] = []
    for item in list(history)[-max_history_items:]:
        tokens.extend(int(code) + CODE_OFFSET for code in codes[int(item)])
    tokens = tokens[-model.max_history_tokens :]
    return torch.tensor(tokens, dtype=torch.long, device=next(model.parameters()).device)


def generate_tiger_scored(
    model: TigerTransformer,
    data: PreparedData,
    split: str,
    codes: np.ndarray,
    sid_index: SemanticIDIndex,
    *,
    users: Sequence[int],
    k: int = 20,
    beam_size: int = 16,
    max_history_items: int = 20,
    include_validation_in_test_history: bool = True,
) -> dict[int, list[tuple[int, float]]]:
    """与 ``generate_tiger_recommendations`` 相同，但保留生成 log-prob。"""
    model.eval()
    histories = build_observed_histories(
        data,
        split,
        include_validation_in_test_history=include_validation_in_test_history,
    )
    blocked = seen_items_for_split(data, split)
    output: dict[int, list[tuple[int, float]]] = {}
    with torch.no_grad():
        for user in users:
            tokens = _history_tokens(
                model, histories.get(int(user), []), codes, max_history_items
            )
            if tokens.numel() == 0:
                output[int(user)] = []
                continue
            output[int(user)] = model.generate_user(
                tokens,
                sid_index,
                k=k,
                beam_size=beam_size,
                blocked_items=set(blocked.get(int(user), set())),
            )
    return output


@torch.no_grad()
def generate_tiger_recommendations(
    model: TigerTransformer,
    data: PreparedData,
    split: str,
    codes: np.ndarray,
    sid_index: SemanticIDIndex,
    *,
    k: int = 20,
    beam_size: int = 16,
    max_history_items: int = 20,
    max_users: int | None = None,
    routing: str = "beam",
    include_validation_in_test_history: bool = True,
) -> dict[int, list[int]]:
    model.eval()
    histories = build_observed_histories(
        data,
        split,
        include_validation_in_test_history=include_validation_in_test_history,
    )
    targets = data.validation_interactions if split == "validation" else data.test_interactions
    users = sorted(group_items_by_user(targets))
    if max_users is not None:
        users = users[:max_users]
    blocked = seen_items_for_split(data, split)
    recommendations: dict[int, list[int]] = {}
    for user in users:
        history = histories.get(user, [])
        tokens = _history_tokens(model, history, codes, max_history_items)
        if tokens.numel() == 0:
            recommendations[user] = []
            continue
        ranked = model.generate_user(
            tokens,
            sid_index,
            k=k,
            beam_size=beam_size,
            blocked_items=set(blocked.get(user, set())),
        )
        recommendations[user] = [item for item, _ in ranked]
    return recommendations


def evaluate_tiger_model(
    model: TigerTransformer,
    data: PreparedData,
    split: str,
    codes: np.ndarray,
    sid_index: SemanticIDIndex,
    *,
    ks: Sequence[int] = (10, 20),
    beam_size: int = 16,
    max_users: int | None = None,
    include_validation_in_test_history: bool = True,
) -> dict[str, float]:
    recommendations = generate_tiger_recommendations(
        model,
        data,
        split,
        codes,
        sid_index,
        k=max(ks),
        beam_size=beam_size,
        max_users=max_users,
        include_validation_in_test_history=include_validation_in_test_history,
    )
    targets = data.validation_interactions if split == "validation" else data.test_interactions
    relevant_users = set(group_items_by_user(targets))
    recommendations = {user: items for user, items in recommendations.items() if user in relevant_users}
    metrics = evaluate_recommendations(recommendations, targets, ks)
    metrics["Coverage"] = catalog_coverage(recommendations, data.num_items)
    return metrics


__all__ = [
    "evaluate_tiger_model",
    "generate_tiger_recommendations",
    "generate_tiger_scored",
]
