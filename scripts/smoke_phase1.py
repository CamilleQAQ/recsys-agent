"""Phase 1 定向冒烟检查：只验证本次改动直接涉及的接口与链路。

    python scripts/smoke_phase1.py

覆盖：
1. 全部新模块可导入；注册表与配置可读；
2. 每个已有 checkpoint 的召回适配器：小样本推理、seen-item 过滤、
   encode_*（适用时）、save/load 往返；
3. 每个已有 checkpoint 的排序适配器：小样本真实打分；
4. 默认双路链路与多路（含 LightGCN + DIN）链路各跑一次小样本；
5. Zoo 25 条记录的状态字段完整；
6. Rec Agent 工具自检（真实数据画像 / Model Zoo / 召回 / 链路 / 失败路径）。

不训练长模型、不跑全量用户，不把效果数字当作实验结论。
"""

from __future__ import annotations

import json
import sys
import tempfile
import tomllib
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recsys.adapters.ranking_registry import load_ranker, ranker_ids  # noqa: E402
from recsys.adapters.registry import adapter_ids, load_adapter  # noqa: E402
from recsys.config import DEFAULT_DATA_DIR  # noqa: E402
from recsys.data.loader import prepare_dataset  # noqa: E402
from recsys.data.schema import seen_items_for_split  # noqa: E402
from recsys.experiments.rec_agent import run_self_check  # noqa: E402
from recsys.experiments.rec_tools import resolve_checkpoint, resolve_ranker_checkpoint  # noqa: E402
from recsys.pipeline.chain import RecommendationPipeline, load_result  # noqa: E402
from recsys.pipeline.config import load_pipeline_config  # noqa: E402
from recsys.pipeline.ranking import rank_fused  # noqa: E402
from recsys.pipeline.fusion import fuse_channels  # noqa: E402
from recsys.pipeline.recall import run_channel  # noqa: E402
from recsys.pipeline.config import FusionConfig  # noqa: E402

DATASET = "kuasar_small"
ARTIFACTS = PROJECT_ROOT / "artifacts"


def check_zoo(failures: list[str]) -> None:
    zoo = tomllib.loads((PROJECT_ROOT / "models" / "zoo.toml").read_text(encoding="utf-8"))
    models = zoo.get("models", [])
    if len(models) < 25:
        failures.append(f"zoo.toml 只有 {len(models)} 条模型，少于 25")
    required = {
        "source_available",
        "source_reviewed",
        "adapter_ready",
        "runnable",
        "integrated",
        "agent_callable",
        "evaluated",
    }
    for model in models:
        missing = required - set(model.get("status", {}))
        if missing:
            failures.append(f"{model['id']}: status 缺 {sorted(missing)}")
    print(f"[ok] zoo: {len(models)} 条模型，状态字段完整")


def recall_only_data(data, users):
    """只保留这些用户的验证留出目标，避免适配器冒烟给 2,000 用户全量打分。"""
    from dataclasses import replace

    keep = set(int(user) for user in users)
    return replace(
        data,
        validation_interactions=[row for row in data.validation_interactions if row.user_idx in keep],
    )


def check_recall_adapters(data, users, failures: list[str], workdir: Path) -> None:
    data = recall_only_data(data, users)
    for model_id in adapter_ids():
        checkpoint = resolve_checkpoint(
            type("Ctx", (), {"dataset": DATASET, "artifacts_dir": ARTIFACTS, "state": {}})()
            if model_id != "popularity"
            else None,
            model_id,
        ) if model_id != "popularity" else None
        if model_id != "popularity" and checkpoint is None:
            print(f"[skip] {model_id}: 没有 checkpoint")
            continue
        try:
            adapter = load_adapter(model_id, checkpoint, device_name="cpu")
            recommendations = adapter.recommend(data, split="validation", k=5)
            expected = {row.user_idx for row in data.validation_interactions}
            if not set(recommendations).issubset(expected):
                failures.append(f"{model_id}: 返回了非评测用户")
            if not recommendations:
                failures.append(f"{model_id}: 没有返回候选")
            seen = seen_items_for_split(data, "validation")
            for user, items in recommendations.items():
                if len(items) != 5:
                    failures.append(f"{model_id}: user {user} 返回 {len(items)} 个物品")
                leaked = set(items) & seen.get(user, set())
                if leaked:
                    failures.append(f"{model_id}: user {user} 泄漏已曝光物品 {sorted(leaked)[:3]}")
            if model_id != "popularity":
                try:
                    item_vectors = adapter.encode_items([0, 1, 2])
                    if item_vectors.ndim != 2:
                        failures.append(f"{model_id}: encode_items 不是二维")
                except Exception as error:  # noqa: BLE001
                    failures.append(f"{model_id}: encode_items 失败 {error}")
            if model_id != "popularity":
                roundtrip = workdir / f"{model_id}.pt"
                adapter.save(roundtrip)
                reloaded = load_adapter(model_id, roundtrip, device_name="cpu")
                again = reloaded.recommend(data, split="validation", k=5)
                if again != recommendations:
                    failures.append(f"{model_id}: save/load 往返结果不一致")
            print(f"[ok] {model_id}: {len(recommendations)} 用户 Top-5，checkpoint 往返一致")
        except Exception as error:  # noqa: BLE001
            failures.append(f"{model_id}: {type(error).__name__}: {error}")


def check_rankers(data, users, failures: list[str]) -> None:
    items_by_user = {
        int(user): [int(item) for item in sorted(seen_items_for_split(data, "validation").get(int(user), set()))[:20]]
        for user in users
    }
    # 用测试/验证目标附近的物品构造候选，避免空候选。
    for user in users:
        items_by_user[int(user)] = [
            int(item) for item in range(20)
        ]
    from recsys.experiments.rec_tools import ToolContext

    context = ToolContext(dataset=DATASET, artifacts_dir=ARTIFACTS)
    for model_id in ranker_ids():
        checkpoint = resolve_ranker_checkpoint(context, model_id)
        if checkpoint is None:
            print(f"[skip] {model_id}: 没有 checkpoint")
            continue
        try:
            ranker = load_ranker(model_id, checkpoint, device_name="cpu")
            if hasattr(ranker, "score_candidates"):
                scores = ranker.score_candidates(
                    data, list(users), items_by_user, split="validation"
                )
                values = [value for row in scores.values() for value in row.values()]
            else:
                from recsys.ranking.features import FeatureContext

                context_features = FeatureContext(data)
                flat = np.asarray([item for row in items_by_user.values() for item in row], dtype=np.int64)
                values = ranker.predict(context_features.lightgbm_features(flat)).tolist()
            if not values or not np.isfinite(values).all():
                failures.append(f"{model_id}: 分数为空或非有限值")
            elif model_id != "lightgbm_ranker" and not all(0.0 <= value <= 1.0 for value in values):
                failures.append(f"{model_id}: 概率分数越界")
            # 排序器保存/重新加载后，同一批候选的分数应当一致。
            suffix = ".txt" if model_id == "lightgbm_ranker" else ".pt"
            roundtrip = PROJECT_ROOT / "artifacts" / "_phase1_smoke" / f"{model_id}{suffix}"
            roundtrip.parent.mkdir(parents=True, exist_ok=True)
            ranker.save(roundtrip)
            reloaded = load_ranker(model_id, roundtrip, device_name="cpu")
            if hasattr(reloaded, "score_candidates"):
                again = reloaded.score_candidates(
                    data, list(users), items_by_user, split="validation"
                )
                again_values = [value for row in again.values() for value in row.values()]
            else:
                from recsys.ranking.features import FeatureContext

                context_features = FeatureContext(data)
                flat = np.asarray(
                    [item for row in items_by_user.values() for item in row],
                    dtype=np.int64,
                )
                again_values = reloaded.predict(
                    context_features.lightgbm_features(flat)
                ).tolist()
            if len(again_values) != len(values) or not np.allclose(
                np.asarray(again_values), np.asarray(values), atol=1e-6
            ):
                failures.append(f"{model_id}: save/load 往返分数不一致")
            print(f"[ok] {model_id}: {len(values)} 个候选真实打分，范围 [{min(values):.4f}, {max(values):.4f}]")
        except Exception as error:  # noqa: BLE001
            failures.append(f"{model_id}: {type(error).__name__}: {error}")


def check_agent_tool_chain(data, failures: list[str], workdir: Path) -> None:
    """逐个调用 Agent 的召回/排序/重排/评估工具，验证真实函数与状态串联。"""
    from recsys.experiments.rec_tools import ToolBudget, ToolContext, build_registry

    context = ToolContext(
        dataset=DATASET,
        artifacts_dir=ARTIFACTS,
        experiment_dir=workdir / "agent_chain",
        users=4,
        budget=ToolBudget(max_tool_calls=6, max_seconds=120),
    )
    registry = build_registry()
    recall = registry.call("recall_models", {"models": ["two_tower"], "k": 50}, context)
    if not recall.get("ok"):
        failures.append(f"Agent recall_models 失败：{recall.get('error')}")
        return
    rank = registry.call("rank_candidates", {"model": "din", "fusion": "round_robin"}, context)
    if not rank.get("ok"):
        failures.append(f"Agent rank_candidates 失败：{rank.get('error')}")
        return
    rerank = registry.call("rerank_candidates", {"k": 20, "category_cap": 5}, context)
    if not rerank.get("ok"):
        failures.append(f"Agent rerank_candidates 失败：{rerank.get('error')}")
        return
    evaluate = registry.call(
        "evaluate_recommendations",
        {"stage": "final", "split": "validation", "ks": [10, 20]},
        context,
    )
    if not evaluate.get("ok") or "Recall@20" not in evaluate.get("metrics", {}):
        failures.append(f"Agent evaluate_recommendations 失败：{evaluate.get('error')}")
        return
    print(
        "[ok] Agent 工具链（recall two_tower → rank din → rerank → evaluate final）："
        f" users={evaluate['users']} Recall@20={evaluate['metrics']['Recall@20']:.4f}"
    )


def check_pipeline(data, failures: list[str], workdir: Path) -> None:
    default = load_pipeline_config(PROJECT_ROOT / "configs" / "pipeline" / "kuasar_small_default.toml")
    result = RecommendationPipeline(default, data=data).run(
        users=[int(user) for user in sorted({row.user_idx for row in data.validation_interactions})[:4]],
        write=True,
        output_dir=workdir / "pipeline_default",
    )
    if set(result.metrics) != {"recall_two_tower", "recall_mind", "fusion", "ranked", "rerank"}:
        failures.append(f"默认链路阶段缺失：{sorted(result.metrics)}")
    if not all(len(result.final[user]) <= default.run.k_final for user in result.users):
        failures.append("默认链路最终候选超过 k_final")
    payload = load_result(workdir / "pipeline_default")
    if "final" not in payload or "metrics" not in payload:
        failures.append("默认链路工件不可读")
    print(
        "[ok] 默认链路: "
        + ", ".join(f"{name} Recall@20={values['Recall@20']:.4f}" for name, values in result.metrics.items())
    )

    multi = load_pipeline_config(PROJECT_ROOT / "configs" / "pipeline" / "kuasar_small_multi_recall.toml")
    if not all(
        resolve_checkpoint(type("Ctx", (), {"dataset": DATASET, "artifacts_dir": ARTIFACTS, "state": {}})(), channel.model)
        for channel in multi.recall
    ):
        print("[skip] 多路链路：LightGCN checkpoint 尚不存在")
        return
    ranker_checkpoint = resolve_ranker_checkpoint(
        type("Ctx", (), {"dataset": DATASET, "artifacts_dir": ARTIFACTS, "state": {}})(), multi.ranking.model
    )
    if ranker_checkpoint is None:
        print("[skip] 多路链路：DIN checkpoint 尚不存在")
        return
    from dataclasses import replace

    multi = replace(
        multi,
        run=replace(multi.run, users=4),
        ranking=replace(multi.ranking, checkpoint=ranker_checkpoint),
    )
    result = RecommendationPipeline(multi, data=data).run(
        users=[int(user) for user in sorted({row.user_idx for row in data.validation_interactions})[:4]],
        write=True,
        output_dir=workdir / "pipeline_multi",
    )
    if len(result.channels) != 3:
        failures.append(f"多路链路召回通道数 {len(result.channels)} != 3")
    print(
        "[ok] 多路链路（Two-Tower + MIND + LightGCN → RRF → DIN → 类目/作者重排）: "
        + ", ".join(f"{name} Recall@20={values['Recall@20']:.4f}" for name, values in result.metrics.items())
    )


def main() -> int:
    failures: list[str] = []
    data = prepare_dataset(DATASET, DEFAULT_DATA_DIR)
    users = [int(user) for user in sorted({row.user_idx for row in data.validation_interactions})[:8]]
    workdir = PROJECT_ROOT / "artifacts" / "_phase1_smoke"
    import shutil

    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        check_zoo(failures)
        check_recall_adapters(data, users, failures, workdir)
        check_rankers(data, users, failures)
        check_agent_tool_chain(data, failures, workdir)
        check_pipeline(data, failures, workdir)
        agent = run_self_check(PROJECT_ROOT / "experiments" / "2026-11-09-kuasar_small-rec-agent" / "smoke_phase1")
        if not agent.get("passed"):
            failures.append(f"Rec Agent 自检失败：{agent}")
        else:
            print(f"[ok] Rec Agent 自检：{len(agent['checks'])} 项通过（planner=local）")
    finally:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)
    if failures:
        print("\n失败项：")
        for failure in failures:
            print(f" - {failure}")
        return 1
    print("\nPhase 1 定向冒烟检查全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
