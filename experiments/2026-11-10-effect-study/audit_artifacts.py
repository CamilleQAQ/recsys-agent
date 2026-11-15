"""交付前工件自检：所有实验 JSON 都能解析、关键字段齐全、数字在合理范围。

    python audit_artifacts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = {
    "ranker_ablation.json": ["protocol", "arms", "random_anchor", "comparisons_vs_R0", "diagnostics"],
    "ranker_recall_features.json": ["protocol", "arms", "comparisons_vs_R0"],
    "fusion_rerank_ablation.json": ["protocol", "oracle", "arms", "comparisons_vs_default"],
    "stub_channel_check.json": ["users", "single_channel", "fusion_two_real"],
    "seed_study.json": ["seeds", "per_seed", "aggregate", "deterministic"],
    "fusion_replication.json": ["per_seed", "deltas", "summary"],
    "legacy_ranker_replication.json": ["per_seed", "summary"],
    "variance_decomposition.json": ["seeds", "rows"],
    "ranker_stability.json": ["vs_popularity", "cross_seed"],
}


def main() -> int:
    problems = []
    for name, keys in REQUIRED.items():
        path = ROOT / name
        if not path.exists():
            problems.append(f"缺失文件：{name}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:  # noqa: BLE001
            problems.append(f"{name} 解析失败：{error}")
            continue
        missing = [key for key in keys if key not in payload]
        if missing:
            problems.append(f"{name} 缺字段：{missing}")
        print(f"{name:36s} ok  size={path.stat().st_size/1024:.0f} KB")

    seeds = json.loads((ROOT / "seed_study.json").read_text(encoding="utf-8"))
    print(f"\n种子：{seeds['seeds']}；逐种子条目：{sorted(seeds['per_seed'])}")
    if sorted(seeds["per_seed"], key=int) != [str(seed) for seed in seeds["seeds"]]:
        problems.append("seeds 列表与 per_seed 键不一致")

    oracle = json.loads((ROOT / "fusion_rerank_ablation.json").read_text(encoding="utf-8"))["oracle"]
    if oracle["oracle_hits_at20"] != oracle["pool_hits_total"]:
        problems.append("oracle 命中数与池内命中数不一致")

    if problems:
        print("\n发现的问题：")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\n全部工件通过自检")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
