"""kuairec_small 测试集效果分析：命中名次、随机基线、按历史长度分层。

只读已锁定的 checkpoint 与测试切分，不训练、不调参，因此不产生新的"选择性"结论。
输出 ``effect_analysis.json``；同目录 report.md 的「效果」一节引用它。

    python experiments/2026-11-13-kuairec-small-formal/analyze_effect.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from recsys.data.loader import prepare_dataset
from recsys.data.schema import seen_items_for_split
from recsys.evaluation.evaluator import (
    generate_recommendations,
    get_evaluation_interactions,
    group_items_by_user,
    load_model,
)
from recsys.evaluation.sasrec_evaluator import (
    generate_sasrec_recommendations,
    load_sasrec_model,
)
from recsys.experiments.compare import load_mf_model
from recsys.models.popularity import (
    generate_popularity_recommendations,
    rank_items_by_popularity,
)
from recsys.training.trainer import resolve_device

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT.parents[1] / "artifacts" / "kuairec_small"
DATA = ROOT.parents[1] / "data"
SPLIT = "test"
K = 20


def rank_of_target(recommendations: dict[int, list[int]], targets: dict[int, int]) -> dict[int, int | None]:
    """目标物品在 top-K 里的名次（1 起）；未进 top-K 记 None。"""
    ranks: dict[int, int | None] = {}
    for user_idx, target in targets.items():
        items = recommendations.get(user_idx, [])
        ranks[user_idx] = items.index(target) + 1 if target in items else None
    return ranks


def metrics_from_ranks(ranks: dict[int, int | None]) -> dict[str, float]:
    total = len(ranks)
    hit10 = sum(1 for r in ranks.values() if r and r <= 10)
    hit20 = sum(1 for r in ranks.values() if r and r <= 20)
    ndcg10 = sum(1 / math.log2(r + 1) for r in ranks.values() if r and r <= 10)
    ndcg20 = sum(1 / math.log2(r + 1) for r in ranks.values() if r and r <= 20)
    return {
        "Recall@10": hit10 / total,
        "Recall@20": hit20 / total,
        "NDCG@10": ndcg10 / total,
        "NDCG@20": ndcg20 / total,
        "mean_hit_rank": (
            sum(r for r in ranks.values() if r) / hit20 if hit20 else 0.0
        ),
    }


def random_baseline(candidates: dict[int, int]) -> dict[str, float]:
    """每个用户在 C 个候选里均匀随机取 top-20 时的期望指标（精确值）。"""
    recall10 = recall20 = ndcg10 = ndcg20 = 0.0
    for c in candidates.values():
        top10, top20 = min(10, c), min(20, c)
        recall10 += top10 / c
        recall20 += top20 / c
        ndcg10 += sum(1 / math.log2(r + 1) for r in range(1, top10 + 1)) / c
        ndcg20 += sum(1 / math.log2(r + 1) for r in range(1, top20 + 1)) / c
    n = len(candidates)
    return {
        "Recall@10": recall10 / n,
        "Recall@20": recall20 / n,
        "NDCG@10": ndcg10 / n,
        "NDCG@20": ndcg20 / n,
    }


def bucket_histogram(ranks: dict[int, int | None]) -> dict[str, int]:
    buckets = {"1": 0, "2-5": 0, "6-10": 0, "11-20": 0, "miss": 0}
    for rank in ranks.values():
        if rank is None:
            buckets["miss"] += 1
        elif rank == 1:
            buckets["1"] += 1
        elif rank <= 5:
            buckets["2-5"] += 1
        elif rank <= 10:
            buckets["6-10"] += 1
        else:
            buckets["11-20"] += 1
    return buckets


def main() -> None:
    device = resolve_device("auto")
    prepared = prepare_dataset("kuairec_small", DATA)
    evaluation = get_evaluation_interactions(prepared, SPLIT)
    targets = {row.user_idx: row.item_idx for row in evaluation}
    users = sorted(targets)
    seen = seen_items_for_split(prepared, SPLIT)
    candidates = {u: prepared.num_items - len(seen.get(u, set())) for u in users}
    history_length = {u: len(seen.get(u, set())) for u in users}

    print(f"users={len(users)} items={prepared.num_items}")
    print(
        "candidates per user: min %d median %d max %d"
        % (
            min(candidates.values()),
            sorted(candidates.values())[len(candidates) // 2],
            max(candidates.values()),
        )
    )

    popularity_ranking = rank_items_by_popularity(
        prepared.train_interactions, prepared.num_items
    )
    recs: dict[str, dict[int, list[int]]] = {
        "Popularity": generate_popularity_recommendations(
            user_indices=users,
            popularity_ranking=popularity_ranking,
            train_items_by_user=seen,
            k=K,
        )
    }

    for name, path in (
        ("MF", ARTIFACTS / "matrix_factorization.pt"),
        ("Two-Tower", ARTIFACTS / "two_tower.pt"),
    ):
        model, _ = (
            load_mf_model(path, device)
            if name == "MF"
            else load_model(path, device)
        )
        recs[name] = generate_recommendations(
            model, users, prepared.num_items, seen, K, 512, device
        )

    sasrec_model, _ = load_sasrec_model(ARTIFACTS / "sasrec.pt", device)
    recs["SASRec"] = generate_sasrec_recommendations(
        sasrec_model, prepared, SPLIT, K, 512, device
    )

    results: dict[str, object] = {
        "split": SPLIT,
        "users": len(users),
        "num_items": prepared.num_items,
        "candidates": {
            "min": min(candidates.values()),
            "median": sorted(candidates.values())[len(candidates) // 2],
            "max": max(candidates.values()),
            "mean": sum(candidates.values()) / len(candidates),
        },
        "random_baseline": random_baseline(candidates),
        "models": {},
    }

    # 历史长度三分位（按训练前缀曝光数），看模型是否只在重度用户上有效。
    ordered = sorted(users, key=lambda u: history_length[u])
    terciles = {
        "light": ordered[: len(ordered) // 3],
        "mid": ordered[len(ordered) // 3 : 2 * len(ordered) // 3],
        "heavy": ordered[2 * len(ordered) // 3 :],
    }

    for name, recommendations in recs.items():
        ranks = rank_of_target(recommendations, targets)
        metrics = metrics_from_ranks(ranks)
        by_bucket = {
            label: metrics_from_ranks({u: ranks[u] for u in members})
            for label, members in terciles.items()
        }
        results["models"][name] = {
            "metrics": metrics,
            "hit_rank_histogram": bucket_histogram(ranks),
            "by_history_tercile": by_bucket,
        }
        print(
            f"{name:<11} R@20={metrics['Recall@20']:.6f} N@20={metrics['NDCG@20']:.6f} "
            f"mean_hit_rank={metrics['mean_hit_rank']:.2f} "
            f"hist={bucket_histogram(ranks)}"
        )

    out = ROOT / "effect_analysis.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
