"""把两条确定性基线（Popularity / ItemCF）的逐用户指标单独落盘。

阶段增量表需要和神经通道做同口径成对比较，而这两个基线不依赖种子，
所以单独算一次写进 ``peruser/deterministic.npz``。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_data, per_user_metrics, targets_by_user  # noqa: E402
from run_seed_study import itemcf_channel, popularity_channel  # noqa: E402


def main() -> int:
    data = load_data()
    ground_truth = targets_by_user(data, "test")
    users = sorted(ground_truth)
    payload = {}
    for name, channel in (("popularity", popularity_channel), ("itemcf", itemcf_channel)):
        candidates = channel(data, users)
        metrics = per_user_metrics(
            {user: [c.item for c in rows] for user, rows in candidates.items()},
            ground_truth,
            ks=(20, 100),
        )
        for metric, values in metrics.items():
            payload[f"{name}__{metric}"] = values
        print(f"{name}: Recall@20={metrics['Recall@20'].mean():.6f}", flush=True)
    (ROOT / "peruser").mkdir(exist_ok=True)
    np.savez_compressed(ROOT / "peruser" / "deterministic.npz", **payload)
    print("写出 peruser/deterministic.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
