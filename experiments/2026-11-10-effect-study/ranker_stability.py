"""排序器"稳定器"的机制检验：它跨种子输出的相似度 vs 融合的相似度。

如果 R5 的输出在三个种子上高度重合（而融合的输出不重合），说明它主要在输出一个
与候选无关的先验（物品热度 / 全局分数），这解释了 §5 的方差收窄。

    python ranker_stability.py
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_data, targets_by_user  # noqa: E402
from run_seed_study import popularity_channel  # noqa: E402

OUT = ROOT / "ranker_stability.json"
SEEDS = (42, 43, 44)


def load_stage(seed: int, stage: str) -> dict[int, list[int]]:
    path = ROOT / "seed_candidates" / f"seed{seed}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        blob = json.load(handle)
    return {int(user): [int(item) for item in items] for user, items in blob[stage].items()}


def jaccard_at(items_a: list[int], items_b: list[int], k: int = 20) -> float:
    a, b = set(items_a[:k]), set(items_b[:k])
    return len(a & b) / len(a | b) if a | b else 0.0


def main() -> int:
    data = load_data()
    ground_truth = targets_by_user(data, "test")
    users = sorted(ground_truth)

    popularity = popularity_channel(data, users, k=100)
    popularity_items = {user: [c.item for c in rows] for user, rows in popularity.items()}

    rows = []
    for seed in SEEDS:
        ranked = load_stage(seed, "ranked")
        fusion = load_stage(seed, "fusion")
        rows.append({
            "seed": seed,
            "ranked_vs_popularity_jaccard20": round(float(np.mean([
                jaccard_at(ranked[u], popularity_items[u]) for u in users
            ])), 4),
            "fusion_vs_popularity_jaccard20": round(float(np.mean([
                jaccard_at(fusion[u], popularity_items[u]) for u in users
            ])), 4),
        })

    pairs = []
    for i, left in enumerate(SEEDS):
        for right in SEEDS[i + 1:]:
            ranked_l, ranked_r = load_stage(left, "ranked"), load_stage(right, "ranked")
            fusion_l, fusion_r = load_stage(left, "fusion"), load_stage(right, "fusion")
            pairs.append({
                "pair": f"{left}-{right}",
                "ranked_jaccard20": round(float(np.mean([
                    jaccard_at(ranked_l[u], ranked_r[u]) for u in users
                ])), 4),
                "fusion_jaccard20": round(float(np.mean([
                    jaccard_at(fusion_l[u], fusion_r[u]) for u in users
                ])), 4),
            })

    payload = {
        "metric": "Top-20 集合的 Jaccard（每个用户先算，再对 2,000 用户平均）",
        "vs_popularity": rows,
        "cross_seed": pairs,
        "reading": (
            "ranked 与热度榜的重合度、以及 ranked 跨种子的重合度，如果都明显高于 fusion，"
            "说明排序器主要在输出一个与候选无关的先验，这正是它方差小的原因。"
        ),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("| 种子 | ranked vs 热度榜 | fusion vs 热度榜 |")
    print("| --- | ---: | ---: |")
    for row in rows:
        print(f"| {row['seed']} | {row['ranked_vs_popularity_jaccard20']} | "
              f"{row['fusion_vs_popularity_jaccard20']} |")
    print("\n| 种子对 | ranked 跨种子 | fusion 跨种子 |")
    print("| --- | ---: | ---: |")
    for pair in pairs:
        print(f"| {pair['pair']} | {pair['ranked_jaccard20']} | {pair['fusion_jaccard20']} |")
    print(f"\n写出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
