"""方差来源分解：换种子的波动 vs 换测试用户的波动。

对每个阶段：
* ``seed_std``：各种子均值之间的标准差（训练随机性）；
* ``user_se``：把种子平均后的逐用户指标当作一个样本，其标准误（测试用户抽样）；
* ``ratio``：seed_std / user_se —— 换一个种子带来的不确定性，相当于换多少倍规模的测试集。

    python variance_decomposition.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT  # noqa: E402

STAGES = ["two_tower", "mind", "fusion", "ranked", "final"]
OUT = ROOT / "variance_decomposition.json"


def main() -> int:
    seed_study = json.loads((ROOT / "seed_study.json").read_text(encoding="utf-8"))
    seeds = sorted(seed_study["per_seed"], key=int)
    arrays: dict[str, list[np.ndarray]] = {stage: [] for stage in STAGES}
    for seed in seeds:
        with np.load(ROOT / "peruser" / f"seed{seed}.npz") as blob:
            for stage in STAGES:
                arrays[stage].append(blob[f"{stage}__Recall@20"])

    rows = []
    for stage in STAGES:
        stack = np.stack(arrays[stage], axis=0)  # [seeds, users]
        seed_means = stack.mean(axis=1)
        seed_std = float(seed_means.std(ddof=1)) if len(seeds) > 1 else 0.0
        averaged = stack.mean(axis=0)
        user_se = float(averaged.std(ddof=1) / np.sqrt(averaged.size))
        rows.append({
            "stage": stage,
            "seeds": len(seeds),
            "mean": round(float(seed_means.mean()), 6),
            "seed_std": round(seed_std, 6),
            "user_se": round(user_se, 6),
            "ratio": round(seed_std / user_se, 2) if user_se else None,
            "seed_means": [round(float(value), 6) for value in seed_means],
        })

    payload = {
        "seeds": [int(seed) for seed in seeds],
        "rows": rows,
        "reading": (
            "ratio = seed_std / user_se。ratio > 1 表示「换一个训练种子」比「换同样规模的测试用户」"
            "带来更大的不确定性；也就是说，单次实验的结论可复现性不比换测试集更强。"
        ),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("| 阶段 | 种子数 | Recall@20 | 跨种子 std | 用户抽样 SE | ratio |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(f"| {row['stage']} | {row['seeds']} | {row['mean']:.6f} | "
              f"{row['seed_std']:.6f} | {row['user_se']:.6f} | {row['ratio']}× |")
    print(f"\n写出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
