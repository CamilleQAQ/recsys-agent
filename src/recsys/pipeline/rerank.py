"""类目约束与多样性重排（保留原 MVP 的类目配额语义，并可选作者多样性）。"""

from __future__ import annotations

import numpy as np

from recsys.pipeline.ranking import RankedCandidate


def diversify(
    candidates: list[RankedCandidate],
    *,
    categories: np.ndarray,
    authors: np.ndarray | None,
    k: int,
    category_cap: int = 5,
    author_cap: int = 0,
) -> tuple[list[RankedCandidate], int]:
    """首遍每类目最多 ``category_cap``、每作者最多 ``author_cap``；不足按序补齐。

    未知类目（-1）与未知作者不占配额，与既有 ``category_rerank`` 行为一致。
    返回 ``(选中候选, 补位条数)``；每条候选保留原始分数与来源。
    """
    picked: list[RankedCandidate] = []
    category_used: dict[int, int] = {}
    author_used: dict[int, int] = {}
    for candidate in candidates:
        if len(picked) >= k:
            break
        category = int(categories[candidate.item]) if categories.size else -1
        if category >= 0:
            if category_used.get(category, 0) >= category_cap:
                continue
            category_used[category] = category_used.get(category, 0) + 1
        if authors is not None and author_cap > 0:
            author = int(authors[candidate.item])
            if author >= 0:
                if author_used.get(author, 0) >= author_cap:
                    continue
                author_used[author] = author_used.get(author, 0) + 1
        picked.append(candidate)

    relaxed = 0
    if len(picked) < k:
        chosen = {candidate.item for candidate in picked}
        for candidate in candidates:
            if len(picked) >= k:
                break
            if candidate.item in chosen:
                continue
            picked.append(candidate)
            chosen.add(candidate.item)
            relaxed += 1
    return picked, relaxed


def category_rerank(
    ranked_items: list[int],
    ranked_scores: list[float],
    categories: np.ndarray,
    k: int = 20,
    cap: int = 5,
) -> tuple[list[int], int]:
    """兼容旧 ``recsys_mvp.category_rerank`` 的纯 item/score 接口。"""
    candidates = [
        RankedCandidate(item=item, score=score, rank=rank + 1, sources={})
        for rank, (item, score) in enumerate(zip(ranked_items, ranked_scores))
    ]
    picked, relaxed = diversify(
        candidates, categories=categories, authors=None, k=k, category_cap=cap
    )
    return [candidate.item for candidate in picked], relaxed


__all__ = ["category_rerank", "diversify"]
