"""把「桩模型」当第三路接进链路，量化后果。

回答「为什么不能拿 sasrec / bert4rec / lightgcn 这些适配器去做对照」：
它们的 checkpoint 是 1 epoch / 1–2 batch 的桩，单独召回 Recall@20≈0，
但接进融合后会占掉一部分候选名额，把真模型的名次挤下去。

用法：python run_stub_channel_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO, ROOT, load_data, per_user_metrics, targets_by_user  # noqa: E402
from run_seed_study import mind_topk  # noqa: E402

from recsys.adapters.registry import load_adapter  # noqa: E402
from recsys.pipeline.config import FusionConfig  # noqa: E402
from recsys.pipeline.fusion import fuse_channels  # noqa: E402
from recsys.pipeline.recall import run_channel  # noqa: E402

ARTIFACTS = REPO / "artifacts" / "kuasar_small"
USERS = 512
OUT = ROOT / "stub_channel_check.json"


def hit_counts(channels: dict, users: list[int], ground_truth: dict) -> dict:
    metrics = per_user_metrics(
        {user: [c.item for c in channels[user]] for user in users}, ground_truth, ks=(20, 100)
    )
    return {
        "Recall@20": round(float(metrics["Recall@20"].mean()), 6),
        "Recall@100": round(float(metrics["Recall@100"].mean()), 6),
        "hits@20": sum(
            len({c.item for c in channels[user][:20]} & ground_truth[user]) for user in users
        ),
    }


def main() -> int:
    data = load_data()
    ground_truth = targets_by_user(data, "test")
    users = sorted(ground_truth)[:USERS]

    two_tower = run_channel(
        load_adapter("two_tower", ARTIFACTS / "two_tower_features.pt", device_name="cpu"),
        data, users, split="test", k=100,
    )
    mind = mind_topk(ARTIFACTS / "mind20.pt", data, users, k=100)

    stub_name = "sasrec"
    stub_adapter = load_adapter(stub_name, ARTIFACTS / f"{stub_name}.pt", device_name="cpu")
    stub = run_channel(stub_adapter, data, users, split="test", k=100)
    del stub_adapter
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 桩模型有多少候选和真模型重叠、独占了多少名额
    overlap = np.mean([
        len({c.item for c in stub[u]} & ({c.item for c in two_tower[u]} | {c.item for c in mind[u]}))
        for u in users
    ])

    real_two = fuse_channels({"two_tower": two_tower, "mind": mind},
                             FusionConfig(mode="round_robin", limit=100))
    real_three = fuse_channels({"two_tower": two_tower, "mind": mind, stub_name: stub},
                               FusionConfig(mode="round_robin", limit=100))

    payload = {
        "users": USERS,
        "single_channel": {
            "two_tower": hit_counts(two_tower, users, ground_truth),
            "mind": hit_counts(mind, users, ground_truth),
            stub_name: hit_counts(stub, users, ground_truth),
        },
        "fusion_two_real": hit_counts(real_two, users, ground_truth),
        f"fusion_with_{stub_name}": hit_counts(real_three, users, ground_truth),
        "stub_overlap_with_real_candidates": round(float(overlap), 2),
        "note": (
            f"{stub_name}.pt 的训练预算是 1 epoch / 2 batch，验证 Recall@20=0；"
            "它不是对手，是空转的通道，但融合按名次轮流取，它会稳定占掉约三分之一的候选名额。"
        ),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
