"""按训练集活跃度分群评测（Phase 3 收敛实验用）。

分组规则在**看测试指标之前**固定：

* ``low``：训练集正交互数 < 100
* ``mid``：100 ≤ 训练集正交互数 < 300
* ``high``：≥ 300

边界取训练集长度的约 1/3、2/3 分位（93 / 307）后取整，保证三组用户数接近。
历史长度只代表活跃度，不能当作「兴趣多样性」的证据。

用法：

    python scripts/eval_groups.py --model mind --checkpoint artifacts/kuasar_small/mind_p3_A42.pt \
        --split test --ks 10 20 --output artifacts/_eval/A42_test.json

同一份脚本对双塔基线同样适用（``--model two_tower``）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.adapters.registry import load_adapter  # noqa: E402
from recsys.data.loader import prepare_dataset  # noqa: E402
from recsys.evaluation.evaluator import (  # noqa: E402
    evaluate_recommendations,
    get_evaluation_interactions,
    group_items_by_user,
)
from recsys.evaluation.metrics import catalog_coverage  # noqa: E402

#: 分组边界（训练集正交互数），在看测试集结果之前固定。
GROUP_BOUNDARIES = {"low": (0, 100), "mid": (100, 300), "high": (300, 10**9)}


def train_positive_counts(data) -> np.ndarray:
    counts = np.zeros(data.num_users, dtype=np.int64)
    if hasattr(data.train_interactions, "_users"):
        np.add.at(counts, np.asarray(data.train_interactions._users), 1)
    else:
        for row in data.train_interactions:
            counts[row.user_idx] += 1
    return counts


def group_of(count: int) -> str:
    for name, (low, high) in GROUP_BOUNDARIES.items():
        if low <= count < high:
            return name
    return "high"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--routing-seed",
        type=int,
        default=1234,
        help="MIND 胶囊路由的随机初始化种子（评测确定性；其它模型忽略）",
    )
    args = parser.parse_args(argv)

    data = prepare_dataset("kuasar_small", args.data_dir)
    adapter = load_adapter(args.model, args.checkpoint, device_name="cpu")
    recommendations = adapter.recommend(
        data, split=args.split, k=max(args.ks), routing_seed=args.routing_seed
    )
    targets = get_evaluation_interactions(data, args.split)
    target_by_user = group_items_by_user(targets)

    counts = train_positive_counts(data)
    groups = {user: group_of(int(counts[user])) for user in target_by_user}
    overall = evaluate_recommendations(recommendations, targets, ks=args.ks)
    overall["Coverage"] = catalog_coverage(recommendations, data.num_items)

    result = {
        "model": args.model,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "split": args.split,
        "ks": args.ks,
        "group_boundaries": {name: [low, high] for name, (low, high) in GROUP_BOUNDARIES.items()},
        "overall": {key: round(float(value), 6) for key, value in overall.items()},
        "groups": {},
    }
    for name in GROUP_BOUNDARIES:
        users = [user for user, group in groups.items() if group == name]
        group_recs = {user: recommendations[user] for user in users if user in recommendations}
        group_targets = [row for row in targets if row.user_idx in set(users)]
        result["groups"][name] = {
            "users": len(users),
            "targets": len(group_targets),
            "metrics": (
                {
                    key: round(float(value), 6)
                    for key, value in evaluate_recommendations(
                        group_recs, group_targets, ks=args.ks
                    ).items()
                }
                if users
                else {}
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["overall"], ensure_ascii=False))
    for name, payload in result["groups"].items():
        print(f"{name}: users={payload['users']} targets={payload['targets']} {payload['metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
