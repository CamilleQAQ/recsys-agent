"""出厂排序器 R1 的跨种子复验。

§2 的"现行精排显著差于不排序"只用 seed 42 的候选池验过。这里把同一份出厂权重
（``artifacts/kuasar_small/mvp/lgbm_ranker.txt``）放到三个种子的融合候选上，
检查这个结论是不是只在某一批候选上成立。

    python run_legacy_ranker_replication.py
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO, ROOT, load_data, per_user_metrics, targets_by_user  # noqa: E402
from run_fusion_rerank_ablation import rerank_final  # noqa: E402

from recsys.adapters.ranking import LightGBMRankerAdapter  # noqa: E402
from recsys.data.kuasar_ranking import ITEM_FEATURE_COLUMNS  # noqa: E402

OUT = ROOT / "legacy_ranker_replication.json"
LEGACY = REPO / "artifacts" / "kuasar_small" / "mvp" / "lgbm_ranker.txt"
SEEDS = (42, 43, 44)


def load_fusion(seed: int) -> dict[int, list[int]]:
    path = ROOT / "seed_candidates" / f"seed{seed}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        blob = json.load(handle)
    return {int(user): [int(item) for item in items] for user, items in blob["fusion"].items()}


def measure(recommendations: dict[int, list[int]], ground_truth: dict) -> dict:
    metrics = per_user_metrics(recommendations, ground_truth, ks=(20, 100))
    return {
        "Recall@20": round(float(metrics["Recall@20"].mean()), 6),
        "NDCG@20": round(float(metrics["NDCG@20"].mean()), 6),
        "hits@20": int(sum(len(set(recommendations[u][:20]) & ground_truth[u])
                          for u in recommendations)),
    }


def main() -> int:
    data = load_data()
    ground_truth = targets_by_user(data, "test")
    ranker = LightGBMRankerAdapter.load(LEGACY)
    features = {
        name: np.asarray(data.item_features.column(name), dtype=np.float32)
        for name in ITEM_FEATURE_COLUMNS
    }

    rows = []
    for seed in SEEDS:
        fusion = load_fusion(seed)
        users = sorted(fusion)
        ranked: dict[int, list[int]] = {}
        for user in users:
            items = np.asarray(fusion[user], dtype=np.int64)
            matrix = np.stack([features[name][items] for name in sorted(features)], axis=1)
            scores = ranker.predict(matrix)
            order = np.lexsort((np.arange(items.size), -scores))
            ranked[user] = [int(items[i]) for i in order]

        pool_rows = {u: [{"item": item, "fusion_rank": i + 1} for i, item in enumerate(fusion[u])]
                     for u in users}
        fused_metrics = measure(fusion, ground_truth)
        legacy_metrics = measure(ranked, ground_truth)
        fused_rerank = measure(rerank_final(data, pool_rows, 5), ground_truth)
        legacy_rerank = measure(
            rerank_final(data, {u: [{"item": item, "fusion_rank": i + 1} for i, item in enumerate(ranked[u])]
                                for u in users}, 5),
            ground_truth,
        )
        rows.append({
            "seed": seed,
            "pool_hits": int(sum(len(set(fusion[u]) & ground_truth[u]) for u in users)),
            "fusion": fused_metrics,
            "legacy_ranker": legacy_metrics,
            "fusion_plus_rerank": fused_rerank,
            "legacy_plus_rerank": legacy_rerank,
            "legacy_minus_fusion_hits": legacy_metrics["hits@20"] - fused_metrics["hits@20"],
        })
        print(f"seed {seed}: 池内 {rows[-1]['pool_hits']} 命中 | 融合 {fused_metrics['hits@20']} | "
              f"出厂精排 {legacy_metrics['hits@20']} | 融合+重排 {fused_rerank['hits@20']} | "
              f"精排+重排 {legacy_rerank['hits@20']}", flush=True)

    payload = {
        "ranker": str(LEGACY),
        "note": "同一份出厂 LightGBM 权重（2,418 请求 / 9,997 行 / 20 轮 / 9 列物品特征），只换候选池。",
        "per_seed": rows,
        "summary": {
            "seeds_where_legacy_worse_than_fusion": sum(1 for r in rows if r["legacy_minus_fusion_hits"] < 0),
            "seeds": len(rows),
            "mean_delta_hits": round(float(np.mean([r["legacy_minus_fusion_hits"] for r in rows])), 2),
            "mean_delta_hits_with_rerank": round(float(np.mean([
                r["legacy_plus_rerank"]["hits@20"] - r["fusion_plus_rerank"]["hits@20"] for r in rows
            ])), 2),
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    print(f"写出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
