"""推荐链路命令行入口。

    python -m recsys.pipeline list
    python -m recsys.pipeline run --config configs/pipeline/kuasar_small_default.toml
    python -m recsys.pipeline evaluate --artifacts artifacts/kuasar_small/pipeline --split validation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from recsys.adapters.ranking_registry import describe_rankers, ranker_ids
from recsys.adapters.registry import adapter_ids, describe
from recsys.config import DEFAULT_DATA_DIR
from recsys.data.loader import prepare_dataset
from recsys.evaluation.evaluator import evaluate_recommendations, group_items_by_user
from recsys.evaluation.metrics import catalog_coverage
from recsys.pipeline.chain import RecommendationPipeline, load_result, select_users
from recsys.pipeline.config import load_pipeline_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="列出召回与排序模型")

    run = subparsers.add_parser("run", help="按配置跑完整推荐链路")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--users", type=int, default=None)
    run.add_argument("--user-ids", default=None, help="原始 user_id，逗号分隔")
    run.add_argument("--split", choices=["validation", "test"], default=None)
    run.add_argument("--output", type=Path, default=None)
    run.add_argument("--no-write", action="store_true")
    run.add_argument("--device", default=None)

    evaluate = subparsers.add_parser("evaluate", help="对已落盘的最终结果重新计算指标")
    evaluate.add_argument("--artifacts", type=Path, required=True)
    evaluate.add_argument("--config", type=Path, default=None)
    evaluate.add_argument("--dataset", default=None)
    evaluate.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    evaluate.add_argument("--split", choices=["validation", "test"], default=None)
    evaluate.add_argument("--ks", type=int, nargs="+", default=[10, 20])
    return parser.parse_args(argv)


def command_run(args: argparse.Namespace) -> int:
    config = load_pipeline_config(args.config)
    if args.device:
        from dataclasses import replace

        config = replace(config, run=replace(config.run, device=args.device))
    if args.users is not None:
        from dataclasses import replace

        config = replace(config, run=replace(config.run, users=args.users))
    if args.user_ids:
        from dataclasses import replace

        ids = tuple(int(part) for part in args.user_ids.split(",") if part.strip())
        config = replace(config, run=replace(config.run, user_ids=ids))
    pipeline = RecommendationPipeline(config)
    split = args.split or config.run.split
    result = pipeline.run(split=split, write=not args.no_write, output_dir=args.output)
    print(json.dumps(result.summary(), indent=2, ensure_ascii=False))
    if result.output_dir:
        print(f"artifacts: {result.output_dir}")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    payload = load_result(args.artifacts)
    final = payload.get("final")
    if not final:
        print(f"[error] {args.artifacts}/final.json 不存在", file=sys.stderr)
        return 2
    run_meta = payload.get("run_meta", {})
    dataset = args.dataset or run_meta.get("config", {}).get("run", {}).get("dataset")
    split = args.split or run_meta.get("config", {}).get("run", {}).get("split", "validation")
    if not dataset:
        print("[error] 无法确定 dataset，请传 --dataset", file=sys.stderr)
        return 2
    data = prepare_dataset(dataset, args.data_dir)
    raw_item_to_index = data.mappings.item_to_index
    recommendations: dict[int, list[int]] = {}
    for user_text, row in final.items():
        user = int(row["user"])
        user_idx = data.mappings.user_to_index[user]
        recommendations[user_idx] = [
            int(raw_item_to_index[int(item["item"])]) for item in row["items"]
        ]
    interactions = data.validation_interactions if split == "validation" else data.test_interactions
    # 只评测该工件实际覆盖的用户，避免把未运行的用户算成 0 命中。
    users = set(recommendations)
    targets = [row for row in interactions if row.user_idx in users]
    metrics = evaluate_recommendations(recommendations, targets, tuple(args.ks))
    metrics["Coverage"] = catalog_coverage(recommendations, data.num_items)
    print(
        json.dumps(
            {
                "dataset": dataset,
                "split": split,
                "users": len(recommendations),
                "targets": len(targets),
                "metrics": {key: round(float(value), 6) for key, value in metrics.items()},
                "targets_per_user": round(len(targets) / max(1, len(recommendations)), 3),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "list":
        print("召回模型：")
        for row in describe():
            print(
                f"  {row['model_id']:<22}{row['adapter']:<30}"
                f"{row['model_type']:<12}checkpoint={row['needs_checkpoint']}"
            )
        print("排序模型：")
        for row in describe_rankers():
            print(f"  {row['model_id']:<22}{row['adapter']}")
        return 0
    if args.command == "run":
        return command_run(args)
    if args.command == "evaluate":
        return command_evaluate(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
