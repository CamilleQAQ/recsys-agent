"""Top-20 零命中的固定小样本诊断。

在固定用户（默认 128 个验证用户、seed 42）上跑一次默认双路推荐链路，
逐目标记录「最先丢失的环节」，并补充目标物品热度与通道候选热度的对比。

    python scripts/diagnose_top20.py --users 128 --seed 42

产物：
    artifacts/kuasar_small/top20_diagnosis/  （完整链路工件）
    artifacts/kuasar_small/top20_diagnosis/diagnosis.json
    docs/top20-diagnosis.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recsys.pipeline.chain import RecommendationPipeline  # noqa: E402
from recsys.pipeline.config import load_pipeline_config  # noqa: E402


def item_popularity(data) -> np.ndarray:
    counts = np.zeros(data.num_items, dtype=np.int64)
    for row in data.train_interactions:
        counts[row.item_idx] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "pipeline" / "kuasar_small_default.toml"
    )
    parser.add_argument("--users", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "kuasar_small" / "top20_diagnosis",
    )
    args = parser.parse_args(argv)

    config = load_pipeline_config(args.config)
    from dataclasses import replace

    config = replace(
        config,
        run=replace(config.run, users=args.users, seed=args.seed, split=args.split),
    )
    pipeline = RecommendationPipeline(config)
    result = pipeline.run(write=True, output_dir=args.output)
    data = pipeline.data
    popularity = item_popularity(data)
    diagnostics = result.diagnostics
    target_details = diagnostics["target_details"]

    target_popularity = [int(popularity[data.mappings.item_to_index[row["item"]]]) for row in target_details]
    channel_popularity: dict[str, list[int]] = {name: [] for name in result.channels}
    for name, mapping in result.channels.items():
        for user in result.users:
            for candidate in mapping.get(user, []):
                channel_popularity[name].append(int(popularity[candidate.item]))

    def percentile(values: list[int], q: float) -> float:
        return float(np.percentile(values, q)) if values else 0.0

    summary = {
        "split": result.split,
        "users": len(result.users),
        "targets": diagnostics["targets"],
        "targets_by_stage": diagnostics["targets_by_stage"],
        "channel_hits_at_k_recall": diagnostics["channel_hits_at_k"],
        "union_hits_at_k_recall": diagnostics["union_hits_at_k_recall"],
        "metrics": result.metrics,
        "target_popularity": {
            "mean": round(float(np.mean(target_popularity)), 3) if target_popularity else 0.0,
            "median": percentile(target_popularity, 50),
            "p90": percentile(target_popularity, 90),
            "zero_share": round(
                float(np.mean([value == 0 for value in target_popularity])), 4
            )
            if target_popularity
            else 0.0,
        },
        "channel_candidate_popularity": {
            name: {
                "median": percentile(values, 50),
                "p90": percentile(values, 90),
                "mean": round(float(np.mean(values)), 2) if values else 0.0,
            }
            for name, values in channel_popularity.items()
        },
        "fusion_limit": config.fusion.limit,
        "k_recall": config.run.k_recall,
        "k_final": config.run.k_final,
        "relaxations": result.relaxations,
        "targets_lost_at_fusion_examples": [
            row
            for row in target_details
            if row["stage"]
            in {"in_union_not_in_fusion_pool", "in_fusion_pool_not_top20"}
        ][:10],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "diagnosis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
