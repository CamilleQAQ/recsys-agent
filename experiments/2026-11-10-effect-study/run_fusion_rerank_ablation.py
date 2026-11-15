"""E1 排序上界 + E2 融合策略 × 重排 cap 联合消融。

E1：同一批融合候选上做 oracle 排序（命中目标全部排最前），给出 Recall@20 的天花板，
    用来判断「排序器还值不值得修」。
E2：5 种融合 × 4 档类目 cap，全部在同一批 2000 用户候选上，成对 bootstrap 对比默认配置
    （round_robin + cap 5）。

只读 experiments/2026-11-10-formal-e2e/recommendation 的 recall_*.json，不训练、不推理。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    BASELINE_DIR,
    ROOT,
    load_data,
    paired_bootstrap,
    per_user_metrics,
    targets_by_user,
)

from recsys.pipeline.config import FusionConfig  # noqa: E402
from recsys.pipeline.fusion import fuse_channels  # noqa: E402
from recsys.pipeline.recall import Candidate  # noqa: E402
from recsys.pipeline.rerank import diversify  # noqa: E402
from recsys.pipeline.ranking import RankedCandidate  # noqa: E402

OUT = ROOT / "fusion_rerank_ablation.json"
CAPS = (3, 5, 10, 100)


def load_channel(name: str) -> dict[int, list[Candidate]]:
    blob = json.loads((BASELINE_DIR / f"recall_{name}.json").read_text(encoding="utf-8"))
    return {
        int(user): [
            Candidate(item=int(row["item_idx"]), score=float(row["score"]), rank=int(row["rank"]))
            for row in rows
        ]
        for user, rows in blob.items()
    }


def zscore_sum_fusion(channels: dict[str, dict[int, list[Candidate]]], limit: int = 100):
    """每路分数先按用户做 z-score，再相加；不同模型的分数量纲因此可比。"""
    names = list(channels)
    users = sorted({user for mapping in channels.values() for user in mapping})
    fused: dict[int, list[dict]] = {}
    for user in users:
        combined: dict[int, float] = {}
        best_rank: dict[int, int] = {}
        for name in names:
            rows = channels[name].get(user, [])
            if not rows:
                continue
            scores = np.asarray([row.score for row in rows], dtype=np.float64)
            if scores.std() > 1e-12:
                scores = (scores - scores.mean()) / scores.std()
            else:
                scores = np.zeros_like(scores)
            for row, value in zip(rows, scores, strict=True):
                combined[row.item] = combined.get(row.item, 0.0) + float(value)
                best_rank[row.item] = min(best_rank.get(row.item, 10**6), row.rank)
        ordered = sorted(combined, key=lambda item: (-combined[item], best_rank[item]))[:limit]
        fused[user] = [
            {"item": item, "fusion_rank": rank + 1} for rank, item in enumerate(ordered)
        ]
    return fused


def rerank_final(data, pool: dict[int, list[dict]], cap: int, k: int = 20) -> dict[int, list[int]]:
    categories = np.asarray(data.item_features.column("category_first_code"))
    output: dict[int, list[int]] = {}
    for user, rows in pool.items():
        candidates = [
            RankedCandidate(item=row["item"], score=0.0, rank=row["fusion_rank"])
            for row in rows
        ]
        if cap >= k:
            output[user] = [candidate.item for candidate in candidates[:k]]
            continue
        picked, _ = diversify(candidates, categories=categories, authors=None, k=k, category_cap=cap)
        output[user] = [candidate.item for candidate in picked]
    return output


def measure(label: str, recommendations: dict[int, list[int]], ground_truth: dict) -> dict:
    metrics = per_user_metrics(recommendations, ground_truth, ks=(20, 100))
    return {
        "arm": label,
        "metrics": {key: round(float(value.mean()), 6) for key, value in metrics.items()},
        "hit_targets_at20": sum(
            len(set(recommendations[u][:20]) & ground_truth[u]) for u in recommendations
        ),
        "_per_user": metrics,
    }


def main() -> int:
    data = load_data()
    ground_truth = targets_by_user(data, "test")
    users = sorted(ground_truth)
    channels = {"two_tower": load_channel("two_tower"), "mind": load_channel("mind")}

    # ---------------------------------------------------------------- E1 排序上界
    pool = fuse_channels(channels, FusionConfig(mode="round_robin", limit=100))
    pool_hits = {
        user: len({c.item for c in pool[user][:100]} & ground_truth[user]) for user in users
    }
    oracle = {
        "users": len(users),
        "pool_hits_total": int(sum(pool_hits.values())),
        "oracle_hits_at20": int(sum(min(20, hits) for hits in pool_hits.values())),
        "oracle_recall20": round(float(np.mean([
            min(20, pool_hits[user]) / len(ground_truth[user]) for user in users
        ])), 6),
        "oracle_recall100": round(float(np.mean([
            pool_hits[user] / len(ground_truth[user]) for user in users
        ])), 6),
        "users_capped_by_k20": int(sum(1 for hits in pool_hits.values() if hits > 20)),
        "note": "上界 = 把池内命中全部排到最前；每用户最多贡献 20 个命中，因此 k=20 是硬约束。",
    }

    # ---------------------------------------------------------------- E2 融合 × 重排
    fusions: dict[str, dict[int, list[dict]]] = {}
    fusions["round_robin"] = {
        user: [{"item": c.item, "fusion_rank": c.fusion_rank} for c in pool[user]] for user in users
    }
    for label, mode in (("rrf_k60", "rrf"), ("quota_30_70", "quota"), ("quota_50_50", "quota")):
        quota = (
            {"two_tower": 30, "mind": 70} if label == "quota_30_70"
            else {"two_tower": 50, "mind": 50} if label == "quota_50_50" else {}
        )
        fused = fuse_channels(
            channels, FusionConfig(mode=mode, limit=100, rrf_k=60, quota=quota)
        )
        fusions[label] = {
            user: [{"item": c.item, "fusion_rank": c.fusion_rank} for c in fused[user]]
            for user in users
        }
    fusions["zscore_sum"] = zscore_sum_fusion(channels)

    arms: dict[str, dict] = {}
    for fusion_name, rows in fusions.items():
        pool_recs = {user: [row["item"] for row in rows[user]] for user in users}
        arms[f"{fusion_name}__fusion"] = measure(f"{fusion_name}__fusion", pool_recs, ground_truth)
        for cap in CAPS:
            label = f"{fusion_name}__cap{cap if cap < 100 else 'none'}"
            arms[label] = measure(label, rerank_final(data, rows, cap), ground_truth)

    baseline = arms["round_robin__cap5"]["_per_user"]
    comparisons = {}
    for label, payload in arms.items():
        if label == "round_robin__cap5":
            continue
        comparisons[label] = {
            metric: paired_bootstrap(payload["_per_user"][metric], baseline[metric])
            for metric in ("Recall@20", "NDCG@20")
        }

    payload = {
        "protocol": {
            "dataset": "kuasar_small test (2000 users / 4275 targets)",
            "candidates": str(BASELINE_DIR / "recall_{two_tower,mind}.json"),
            "baseline": "round_robin + category_cap=5（现行链路的默认后处理，不含精排）",
            "k": {"recall": 100, "final": 20},
        },
        "oracle": oracle,
        "arms": {key: {k: v for k, v in value.items() if k != "_per_user"}
                 for key, value in arms.items()},
        "comparisons_vs_default": comparisons,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(oracle, ensure_ascii=False, indent=2))
    print()
    for name in fusions:
        row = arms[f"{name}__fusion"]
        print(f"{name:14s} fusion Recall@20={row['metrics']['Recall@20']:.6f} "
              f"hits={row['hit_targets_at20']}")
        for cap in CAPS:
            label = f"{name}__cap{cap if cap < 100 else 'none'}"
            payload_row = arms[label]
            delta = comparisons.get(label, {}).get("Recall@20")
            tail = (
                f"  Δ vs 默认 {delta['delta']:+.6f} [{delta['ci_low']:+.6f}, {delta['ci_high']:+.6f}]"
                if delta else "  （默认配置）"
            )
            print(f"   cap={cap:<4} Recall@20={payload_row['metrics']['Recall@20']:.6f} "
                  f"hits={payload_row['hit_targets_at20']:2d}{tail}")
    print(f"\n写出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
