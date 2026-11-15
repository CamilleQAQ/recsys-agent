"""E2 的跨种子复验：融合策略在三个种子的候选上各做一遍。

E2 原来只用了 seed 42 的候选池；这里用 ``seed_candidates/seed*.json.gz`` 复算
round_robin / RRF / 配额，检查「RRF 比轮流多 2 个命中」到底是不是稳定结论。

注意：候选文件只存了物品序列（没存分数），所以分数 z-score 相加这一组无法复验。
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_data, per_user_metrics, targets_by_user  # noqa: E402
from run_fusion_rerank_ablation import rerank_final  # noqa: E402

from recsys.pipeline.config import FusionConfig  # noqa: E402
from recsys.pipeline.fusion import fuse_channels  # noqa: E402
from recsys.pipeline.recall import Candidate  # noqa: E402

OUT = ROOT / "fusion_replication.json"
CAPS = (3, 5, 10, 100)


def load_seed_channels(seed: int) -> dict[str, dict[int, list[Candidate]]]:
    path = ROOT / "seed_candidates" / f"seed{seed}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        blob = json.load(handle)
    return {
        name: {
            int(user): [Candidate(item=int(item), score=None, rank=index + 1)
                        for index, item in enumerate(items)]
            for user, items in blob[name].items()
        }
        for name in ("two_tower", "mind")
    }


def measure(label, recommendations, ground_truth):
    metrics = per_user_metrics(recommendations, ground_truth, ks=(20, 100))
    return {
        "arm": label,
        "Recall@20": round(float(metrics["Recall@20"].mean()), 6),
        "Recall@100": round(float(metrics["Recall@100"].mean()), 6),
        "hits@20": int(sum(len(set(recommendations[u][:20]) & ground_truth[u])
                          for u in recommendations)),
    }


def main() -> int:
    data = load_data()
    ground_truth = targets_by_user(data, "test")
    users = sorted(ground_truth)

    per_seed: dict[str, dict] = {}
    for seed in (42, 43, 44):
        channels = load_seed_channels(seed)
        fusions = {
            "round_robin": fuse_channels(channels, FusionConfig(mode="round_robin", limit=100)),
            "rrf_k60": fuse_channels(channels, FusionConfig(mode="rrf", limit=100, rrf_k=60)),
            "quota_30_70": fuse_channels(
                channels, FusionConfig(mode="quota", limit=100, quota={"two_tower": 30, "mind": 70})
            ),
        }
        seed_result: dict[str, dict] = {}
        for name, fused in fusions.items():
            rows = {u: [{"item": c.item, "fusion_rank": c.fusion_rank} for c in fused[u]]
                    for u in users}
            pool = {u: [row["item"] for row in rows[u]] for u in users}
            seed_result[f"{name}__fusion"] = measure(f"{name}__fusion", pool, ground_truth)
            for cap in CAPS:
                label = f"{name}__cap{cap if cap < 100 else 'none'}"
                seed_result[label] = measure(label, rerank_final(data, rows, cap), ground_truth)
        per_seed[str(seed)] = seed_result
        best = seed_result["rrf_k60__capnone"]
        base = seed_result["round_robin__capnone"]
        print(f"seed {seed}: round_robin {base['hits@20']} 命中 / "
              f"RRF {best['hits@20']} 命中 / 配额 {seed_result['quota_30_70__capnone']['hits@20']} 命中",
              flush=True)

    deltas = []
    for seed, result in per_seed.items():
        deltas.append({
            "seed": int(seed),
            "rrf_minus_rr_hits": result["rrf_k60__capnone"]["hits@20"]
            - result["round_robin__capnone"]["hits@20"],
            "rrf_minus_rr_recall20": round(
                result["rrf_k60__capnone"]["Recall@20"]
                - result["round_robin__capnone"]["Recall@20"], 6),
            "cap5_minus_none_hits": result["round_robin__cap5"]["hits@20"]
            - result["round_robin__capnone"]["hits@20"],
        })

    payload = {
        "note": "只复验 rank/位置类融合；候选文件没有分数，z-score 相加无法复验。",
        "per_seed": per_seed,
        "deltas": deltas,
        "summary": {
            "rrf_beats_rr_seeds": sum(1 for row in deltas if row["rrf_minus_rr_hits"] > 0),
            "seeds": len(deltas),
            "mean_rrf_minus_rr_hits": round(float(np.mean([r["rrf_minus_rr_hits"] for r in deltas])), 2),
            "mean_cap5_minus_none_hits": round(float(np.mean([r["cap5_minus_none_hits"] for r in deltas])), 2),
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n| 种子 | round_robin 命中 | RRF 命中 | 配额 30/70 命中 | RRF−轮流 | cap5−无cap |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in deltas:
        result = per_seed[str(row["seed"])]
        print(f"| {row['seed']} | {result['round_robin__capnone']['hits@20']} | "
              f"{result['rrf_k60__capnone']['hits@20']} | "
              f"{result['quota_30_70__capnone']['hits@20']} | "
              f"{row['rrf_minus_rr_hits']:+d} | {row['cap5_minus_none_hits']:+d} |")
    print(f"\n{payload['summary']}")
    print(f"写出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
