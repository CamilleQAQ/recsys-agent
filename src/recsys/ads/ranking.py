"""广告候选排序：同一曝光请求内按 CTR 降序（平局按 item_id 升序）。

这是业务排序层，不绑定模型：输入是模型给出的 pCTR 与候选物品，输出排名、
分数和（可选的）真实点击标签，供 Ads Agent 的 ``rank_ads`` 工具与后续竞价使用。
"""

from __future__ import annotations

from typing import Any

import numpy as np


def rank_candidates(
    items: np.ndarray,
    scores: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    advertiser_ids: np.ndarray | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    items = np.asarray(items, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if items.size != scores.size:
        raise ValueError("items 与 scores 长度不一致")
    if labels is not None and len(labels) != items.size:
        raise ValueError("labels 与 items 长度不一致")
    if advertiser_ids is not None and len(advertiser_ids) != items.size:
        raise ValueError("advertiser_ids 与 items 长度不一致")
    order = np.lexsort((np.arange(items.size), items, -scores))
    limit = max(0, int(top_k))
    output: list[dict[str, Any]] = []
    for rank, position in enumerate(order[:limit] if limit else order, start=1):
        entry = {
            "rank": rank,
            "item": int(items[position]),
            "ctr": round(float(scores[position]), 6),
        }
        if labels is not None:
            entry["click_label"] = int(np.asarray(labels)[position])
        if advertiser_ids is not None:
            entry["advertiser"] = int(np.asarray(advertiser_ids)[position])
        output.append(entry)
    return output


def summarize_rankings(rankings: list[dict[str, Any]]) -> dict[str, Any]:
    if not rankings:
        return {"requests": 0, "candidates": 0, "top1_ctr_mean": 0.0}
    top1 = [record["ranking"][0]["ctr"] for record in rankings if record.get("ranking")]
    candidates = [record["candidate_count"] for record in rankings]
    hits = [
        sum(1 for entry in record["ranking"] if entry.get("click_label") == 1)
        for record in rankings
    ]
    return {
        "requests": len(rankings),
        "candidates": int(sum(candidates)),
        "mean_candidates_per_request": round(float(np.mean(candidates)), 3),
        "top1_ctr_mean": round(float(np.mean(top1)), 6) if top1 else 0.0,
        "top_k_clicks": int(sum(hits)),
        "requests_with_multiple_candidates": int(sum(1 for value in candidates if value > 1)),
    }


__all__ = ["rank_candidates", "summarize_rankings"]
