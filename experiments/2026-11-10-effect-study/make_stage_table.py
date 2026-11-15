"""逐阶段增量表：把多种子结果压成「每一步贡献多少」。

对相邻阶段在同一批用户上做成对 bootstrap（先按种子平均逐用户指标，再重采样用户），
输出 markdown 表格到 stage_table.md 和结构化数据到 stage_table.json。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, paired_bootstrap  # noqa: E402

STAGES = ["popularity", "itemcf", "two_tower", "mind", "fusion", "ranked", "final"]
LABELS = {
    "popularity": "Popularity（全局热度）",
    "itemcf": "ItemCF（共现余弦）",
    "two_tower": "Two-Tower（多 seed）",
    "mind": "MIND（多 seed）",
    "fusion": "轮流融合（两路 top-100）",
    "ranked": "+ 排序器 R5",
    "final": "+ 类目重排",
}
CHAIN = ["fusion", "ranked", "final"]


def load_per_user() -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted((ROOT / "peruser").glob("*.npz")):
        with np.load(path) as blob:
            output[path.stem] = {key: blob[key] for key in blob.files}
    return output


def mean_over_seeds(per_user: dict[str, dict[str, np.ndarray]], stage: str, metric: str):
    arrays = [
        payload[f"{stage}__{metric}"]
        for payload in per_user.values()
        if f"{stage}__{metric}" in payload
    ]
    if not arrays:
        return None
    return np.mean(np.stack(arrays, axis=0), axis=0)


def main() -> int:
    seed_study = json.loads((ROOT / "seed_study.json").read_text(encoding="utf-8"))
    aggregate = seed_study["aggregate"]
    deterministic = seed_study["deterministic"]
    per_user = load_per_user()

    rows = []
    for stage in STAGES:
        if stage in deterministic:
            metrics = deterministic[stage]["metrics"]
            hits = deterministic[stage]["hit_targets_at20"]
            seeds = "确定性"
            std = 0.0
        else:
            block = aggregate[stage]["Recall@20"]
            metrics = {key: value["mean"] for key, value in aggregate[stage].items()}
            hits = round(np.mean([
                entry["stages"][stage]["hit_targets_at20"]
                for entry in seed_study["per_seed"].values()
            ]))
            seeds = f"{len(block['per_seed'])}"
            std = block["std"]
        rows.append({
            "stage": stage,
            "label": LABELS[stage],
            "recall20": metrics["Recall@20"],
            "recall20_std": std,
            "ndcg20": metrics["NDCG@20"],
            "recall100": metrics.get("Recall@100"),
            "hits20": hits,
            "seeds": seeds,
        })

    deltas = []
    # 前四行是并列的单路基线，不是链式阶段；只对「单路 → 融合 → 排序 → 重排」算增量。
    pairs = [
        ("popularity", "fusion"),
        ("itemcf", "fusion"),
        ("two_tower", "fusion"),
        ("mind", "fusion"),
        ("fusion", "ranked"),
        ("ranked", "final"),
    ]
    for previous, current in pairs:
        before = mean_over_seeds(per_user, previous, "Recall@20")
        after = mean_over_seeds(per_user, current, "Recall@20")
        if before is None or after is None:
            deltas.append({"from": previous, "to": current, "delta": None, "note": "缺逐用户数据"})
            continue
        stats = paired_bootstrap(after, before)
        deltas.append({"from": previous, "to": current, **stats})

    lines = [
        "前四行是**并列的单路基线**（互不包含，不是链式阶段）；从「轮流融合」开始才是链路阶段。",
        "",
        "| 组 | 种子数 | Recall@20 | NDCG@20 | Recall@100 | Top-20 命中 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        recall100 = "—" if row["recall100"] is None else f"{row['recall100']:.6f}"
        std = f" ± {row['recall20_std']:.6f}" if row["recall20_std"] else ""
        lines.append(
            f"| {row['label']} | {row['seeds']} | {row['recall20']:.6f}{std} | "
            f"{row['ndcg20']:.6f} | {recall100} | {row['hits20']} |"
        )

    lines += ["", "| 这一步 | Recall@20 变化 | 95% 区间 | 跨零 |", "| --- | ---: | --- | --- |"]
    for delta in deltas:
        if delta["delta"] is None:
            lines.append(f"| {LABELS[delta['from']]} → {LABELS[delta['to']]} | — | — | {delta['note']} |")
            continue
        sign = "+" if delta["delta"] >= 0 else ""
        lines.append(
            f"| {LABELS[delta['from']]} → {LABELS[delta['to']]} | "
            f"{sign}{delta['delta']:.6f} | [{delta['ci_low']:+.6f}, {delta['ci_high']:+.6f}] | "
            f"{'是' if delta['crosses_zero'] else '否'} |"
        )

    text = "\n".join(lines) + "\n"
    (ROOT / "stage_table.md").write_text(text, encoding="utf-8")
    (ROOT / "stage_table.json").write_text(
        json.dumps({"rows": rows, "deltas": deltas}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
