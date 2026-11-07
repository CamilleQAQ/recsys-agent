"""统一推理/评估命令行入口。

    python -m recsys.adapters list
    python -m recsys.adapters evaluate --model two_tower \\
        --checkpoint artifacts/kuairec/two_tower.pt --data-dir data --split test

``evaluate`` 与 ``recsys.evaluation.*`` 里的模型专用脚本共用同一套数据准备和指标实现，
区别只在于「换模型」时不再需要记不同的脚本名和参数。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from recsys.adapters.base import AdapterError, RecommenderAdapter
from recsys.adapters.registry import adapter_ids, describe, load_adapter
from recsys.config import DEFAULT_DATASET, DEFAULT_DATA_DIR
from recsys.data.loader import (
    SUPPORTED_DATASETS,
    prepare_dataset,
    resolve_checkpoint_dataset,
    resolve_positive_threshold,
)
from recsys.data.schema import PreparedData
from recsys.utils.provenance import describe_provenance


def resolve_dataset_and_threshold(
    adapter: RecommenderAdapter,
    dataset: str,
    positive_threshold: float | None,
) -> tuple[str, float]:
    """有 checkpoint 就用 checkpoint 里记录的数据集与阈值，避免评测错数据。"""
    checkpoint = getattr(adapter, "checkpoint", None)
    if checkpoint is None:
        return dataset, resolve_positive_threshold(dataset, positive_threshold)
    resolved = resolve_checkpoint_dataset(checkpoint)
    threshold = resolve_positive_threshold(
        resolved, checkpoint.get("positive_threshold", positive_threshold)
    )
    return resolved, threshold


def run_evaluation(
    adapter: RecommenderAdapter,
    data: PreparedData,
    split: str,
    ks: list[int],
    **kwargs: object,
) -> dict[str, float]:
    """跑一次评分并打印指标；返回指标字典方便测试/复用。"""
    metrics = adapter.evaluate(data, split=split, ks=tuple(ks), **kwargs)
    print(
        f"model={adapter.model_id} split={split} "
        f"users={len({row.user_idx for row in (data.test_interactions if split == 'test' else data.validation_interactions)})} "
        f"items={data.num_items}"
    )
    for k in ks:
        for name in ("Recall", "NDCG", "HitRate", "Precision"):
            key = f"{name}@{k}"
            print(f"{key}: {metrics[key]:.6f}")
    if "Coverage" in metrics:
        print(f"Coverage: {metrics['Coverage']:.6f}")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="列出已接入统一接口的模型")

    evaluate = subparsers.add_parser("evaluate", help="用统一接口评测一个模型")
    evaluate.add_argument("--model", choices=adapter_ids(), required=True)
    evaluate.add_argument("--checkpoint", type=Path, default=None)
    evaluate.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET)
    evaluate.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    evaluate.add_argument(
        "--positive-threshold",
        type=float,
        default=None,
        help="仅对无 checkpoint 的模型生效；有 checkpoint 时以 checkpoint 记录为准",
    )
    evaluate.add_argument("--split", choices=["validation", "test"], default="test")
    evaluate.add_argument("--ks", type=int, nargs="+", default=[10, 20])
    evaluate.add_argument("--device", default="auto")

    recommend = subparsers.add_parser(
        "recommend", help="加载 checkpoint，给指定用户出 Top-K（独立推理，不跑全量评测）"
    )
    recommend.add_argument("--model", choices=adapter_ids(), required=True)
    recommend.add_argument("--checkpoint", type=Path, required=True)
    recommend.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET)
    recommend.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    recommend.add_argument("--positive-threshold", type=float, default=None)
    recommend.add_argument(
        "--users",
        required=True,
        help="原始 user_id，逗号分隔（不是内部索引），例如 5,17,1024",
    )
    recommend.add_argument("--k", type=int, default=10)
    recommend.add_argument(
        "--history",
        choices=["test", "validation", "none"],
        default="test",
        help="用哪个切分之前的历史过滤已看过的物品；none 表示不过滤",
    )
    recommend.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def _history_for_users(
    data: PreparedData, user_indices: set[int], split: str
) -> dict[int, set[int]]:
    """按时间口径取「目标时刻之前」的历史：test = train + validation，validation = train。"""
    history: dict[int, set[int]] = {user: set() for user in user_indices}
    sequences = [data.train_interactions]
    if split == "test":
        sequences.append(data.validation_interactions)
    for interactions in sequences:
        for interaction in interactions:
            if interaction.user_idx in history:
                history[interaction.user_idx].add(interaction.item_idx)
    return history


def run_recommend(adapter: RecommenderAdapter, data: PreparedData, args: argparse.Namespace) -> int:
    """独立推理：原始 user_id → Top-K 原始 item_id。"""
    raw_ids = [int(part) for part in args.users.split(",") if part.strip()]
    user_indices: list[int] = []
    missing: list[int] = []
    for raw_id in raw_ids:
        try:
            user_indices.append(int(data.mappings.user_to_index[raw_id]))
        except KeyError:
            missing.append(raw_id)
    if missing:
        print(f"[warn] 词表里没有这些 user_id：{missing}", file=sys.stderr)
    if not user_indices:
        print("[error] 没有可推理的用户", file=sys.stderr)
        return 2

    history = (
        _history_for_users(data, set(user_indices), args.history)
        if args.history != "none"
        else {user: set() for user in user_indices}
    )
    try:
        user_vectors = adapter.encode_users(user_indices)
        item_vectors = adapter.encode_items(list(range(data.num_items)))
        scores = user_vectors @ item_vectors.T
        index_to_item = data.mappings.index_to_item

        for row, user_idx in enumerate(user_indices):
            blocked = sorted(history[user_idx])
            if blocked:
                scores[row, blocked] = float("-inf")
            top = scores[row].argsort()[::-1][: args.k]
            item_ids = [int(index_to_item[int(item)]) for item in top]
            print(f"user {int(data.mappings.index_to_user[user_idx])}: {item_ids}")
    except AdapterError:
        # 序列类模型（如 MIND）的用户向量依赖历史，不能脱离数据单独编码；
        # 退回到统一的 recommend()，用该切分的历史做推理。
        recommendations = adapter.recommend(data, split=args.history if args.history != "none" else "test", k=args.k)
        for user_idx in user_indices:
            raw_id = int(data.mappings.index_to_user[user_idx])
            items = recommendations.get(user_idx)
            if items is None:
                print(f"[warn] 用户 {raw_id} 不在 {args.history} 切分的评测用户里", file=sys.stderr)
                continue
            print(f"user {raw_id}: {[int(data.mappings.index_to_item[int(item)]) for item in items]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.command == "list":
        print(f"{'model_id':<24}{'adapter':<32}{'type':<14}checkpoint")
        for item in describe():
            print(
                f"{item['model_id']:<24}{item['adapter']:<32}"
                f"{item['model_type']:<14}{item['needs_checkpoint']}"
            )
        return 0

    try:
        adapter = load_adapter(args.model, args.checkpoint, args.device)
        provenance = getattr(adapter, "checkpoint", {}).get("provenance")
        if provenance:
            # 让「这次数字是哪份数据/哪份代码」和指标一起打印出来。
            print(f"provenance: {describe_provenance(provenance)}")
        dataset, threshold = resolve_dataset_and_threshold(
            adapter, args.dataset, args.positive_threshold
        )
        data = prepare_dataset(dataset, args.data_dir, positive_threshold=threshold)
        if args.command == "recommend":
            return run_recommend(adapter, data, args)
        run_evaluation(adapter, data, args.split, list(args.ks))
    except AdapterError as error:
        print(f"[adapter error] {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
