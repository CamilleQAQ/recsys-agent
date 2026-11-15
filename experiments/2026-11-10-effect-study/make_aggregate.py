"""从 seed_study.json 的逐种子指标重算 aggregate（训练/评估被中断时补这一步）。

    python make_aggregate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT  # noqa: E402

STAGES = ["two_tower", "mind", "fusion", "ranked", "final"]


def main() -> int:
    path = ROOT / "seed_study.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_seed = payload.get("per_seed") or {}
    seeds = sorted(per_seed, key=int)
    if not seeds:
        raise SystemExit("seed_study.json 里还没有任何种子")

    aggregate = {}
    for stage in STAGES:
        values = [per_seed[seed]["metrics"][stage] for seed in seeds]
        aggregate[stage] = {}
        for metric in values[0]:
            array = np.asarray([row[metric] for row in values], dtype=np.float64)
            aggregate[stage][metric] = {
                "mean": round(float(array.mean()), 6),
                "std": round(float(array.std(ddof=1)), 6) if array.size > 1 else 0.0,
                "per_seed": [round(float(value), 6) for value in array],
                "spread_ratio": round(float(array.std(ddof=1) / array.mean()), 4)
                if array.size > 1 and array.mean() else None,
            }

    payload["seeds"] = [int(seed) for seed in seeds]
    payload["aggregate"] = aggregate
    payload["note"] = (
        "跨种子 std 只反映训练随机性；用户抽样不确定性另见成对 bootstrap。"
        "所有种子共用同一份测试目标与同一候选预算。"
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for stage in STAGES:
        row = aggregate[stage]["Recall@20"]
        print(f"{stage:10s} {row['mean']:.6f} ± {row['std']:.6f}  逐种子={row['per_seed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
