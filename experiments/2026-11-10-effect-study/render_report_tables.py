"""把 seed_study.json 渲染成报告 §5 的 markdown 表，避免手抄出错。

    python render_report_tables.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT  # noqa: E402

STAGES = ["two_tower", "mind", "fusion", "ranked", "final"]
HEADERS = ["双塔", "MIND", "融合", "+排序器 R5", "+类目重排"]


def main() -> int:
    payload = json.loads((ROOT / "seed_study.json").read_text(encoding="utf-8"))
    seeds = sorted(payload["per_seed"], key=int)
    rows = []
    for seed in seeds:
        stages = payload["per_seed"][seed]["stages"]
        rows.append([stages[stage]["hit_targets_at20"] for stage in STAGES])
    array = np.asarray(rows, dtype=np.float64)

    lines = ["| 种子 | " + " | ".join(HEADERS) + " |",
             "| --- | " + " | ".join(["---:"] * len(HEADERS)) + " |"]
    for seed, values in zip(seeds, rows, strict=True):
        cells = [f"**{value}**" if value == max(row[i] for row in rows) else str(value)
                 for i, value in enumerate(values)]
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")
    means = array.mean(axis=0)
    stds = array.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(means)
    lines.append("| 均值 ± 标准差 | " + " | ".join(
        f"{mean:.1f} ± {std:.1f}" for mean, std in zip(means, stds, strict=True)
    ) + " |")

    recall = []
    for stage in STAGES:
        values = [payload["per_seed"][seed]["metrics"][stage]["Recall@20"] for seed in seeds]
        recall.append(f"{np.mean(values):.6f} ± {np.std(values, ddof=1):.6f}" if len(seeds) > 1
                      else f"{np.mean(values):.6f}")
    lines.append("| Recall@20 | " + " | ".join(recall) + " |")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
