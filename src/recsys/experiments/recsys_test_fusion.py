"""kuasar_small 独立测试集上的固定融合配额验证。

只做召回融合：复用 two_tower_features.pt 与 mind20.pt，按冻结配额
（50/50、30/70）各评测一次；不训练、不跑 LightGBM、不重排、不调用 LLM。

    python -m recsys.experiments.recsys_test_fusion --fixture-check `
      --config configs/experiments/kuasar_small_test_fusion.toml
    python -m recsys.experiments.recsys_test_fusion `
      --config configs/experiments/kuasar_small_test_fusion.toml `
      --output-root experiments/2026-11-09-kuasar_small-test-fusion
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from recsys.config import DEFAULT_DATA_DIR  # noqa: E402
from recsys.data.dataset import build_observed_histories  # noqa: E402
from recsys.data.loader import prepare_dataset  # noqa: E402
from recsys.data.schema import seen_items_for_split  # noqa: E402
from recsys.evaluation.evaluator import (  # noqa: E402
    evaluate_recommendations,
    group_items_by_user,
)
from recsys.evaluation.metrics import catalog_coverage  # noqa: E402
from recsys.experiments import recsys_mvp  # noqa: E402
from recsys.experiments.fusion_quota import fuse_by_quota  # noqa: E402
from recsys.utils.provenance import build_provenance  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "kuasar_small_test_fusion.toml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "2026-11-09-kuasar_small-test-fusion"
SCHEMA_VERSION = 1


class TestFusionError(RuntimeError):
    """配置、缓存口径或时间边界不满足。"""


class StageTimeout(TestFusionError):
    """整条命令超过墙钟上限；已完成的阶段保留。"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _load_config(path: Path) -> dict[str, Any]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    experiment = payload.get("experiment") or {}
    quotas = payload.get("quotas") or {}
    checkpoints = payload.get("checkpoints") or {}
    output = payload.get("output") or {}
    if experiment.get("dataset") != "kuasar_small":
        raise TestFusionError("只允许 kuasar_small")
    if experiment.get("split") != "test":
        raise TestFusionError("本入口只允许 test split")
    if int(experiment.get("max_users", 0)) > 128 or int(experiment.get("max_users", 0)) <= 0:
        raise TestFusionError("max_users 必须在 1..128")
    if int(experiment.get("k_recall", 0)) != 100:
        raise TestFusionError("k_recall 必须为 100")
    if int(experiment.get("threads", 0)) <= 0 or int(experiment.get("threads", 0)) > 4:
        raise TestFusionError("threads 必须在 1..4")
    if float(experiment.get("max_wall_seconds", 0)) > 300 or float(experiment.get("max_wall_seconds", 0)) <= 0:
        raise TestFusionError("max_wall_seconds 必须在 (0, 300]")
    labels = [str(value) for value in quotas.get("labels") or []]
    values = [[int(v) for v in pair] for pair in (quotas.get("values") or [])]
    if labels != ["50/50", "30/70"] or values != [[50, 50], [30, 70]]:
        raise TestFusionError("冻结配额必须是 50/50 与 30/70，且顺序固定")
    for pair in values:
        if len(pair) != 2 or sum(pair) != 100:
            raise TestFusionError("每个配额必须由两个整数组成且总和为 100")
    two_tower = (PROJECT_ROOT / str(checkpoints.get("two_tower", ""))).resolve()
    mind = (PROJECT_ROOT / str(checkpoints.get("mind", ""))).resolve()
    for checkpoint_path in (two_tower, mind):
        if not checkpoint_path.is_file():
            raise TestFusionError(f"checkpoint 不存在：{checkpoint_path}")
    if int(checkpoints.get("routing_seed", -1)) != 1234:
        raise TestFusionError("MIND 路由 seed 必须为 1234")
    return {
        "path": path,
        "raw": payload,
        "experiment": experiment,
        "labels": labels,
        "values": values,
        "two_tower": two_tower,
        "mind": mind,
        "routing_seed": 1234,
        "output_root": PROJECT_ROOT / str(output.get("root", "experiments/2026-11-09-kuasar_small-test-fusion")),
    }


def _make_run_dir(output_root: Path) -> Path:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = _now().strftime("%Y%m%dT%H%M%SZ")
    candidate = output_root / f"run-{stamp}-{os.getpid()}"
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"run-{stamp}-{os.getpid()}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _data_dir() -> Path:
    path = Path(DEFAULT_DATA_DIR)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _select_sample(data, experiment: Mapping[str, Any]) -> dict[str, Any]:
    split = str(experiment["split"])
    count = int(experiment["max_users"])
    seed = int(experiment["seed"])
    interactions = data.test_interactions
    candidates = sorted({row.user_idx for row in interactions})
    rng = np.random.default_rng(seed)
    chosen = np.sort(
        rng.choice(np.asarray(candidates), size=min(count, len(candidates)), replace=False).astype(np.int64)
    )
    selected = set(int(value) for value in chosen.tolist())
    targets = [row for row in interactions if row.user_idx in selected]
    user_ids = [int(data.mappings.index_to_user[int(user)]) for user in chosen]
    pairs = sorted(
        (int(data.mappings.index_to_user[row.user_idx]), int(data.mappings.index_to_item[row.item_idx]))
        for row in targets
    )
    return {
        "split": split,
        "seed": seed,
        "max_users": count,
        "actual_users": int(len(chosen)),
        "target_count": int(len(targets)),
        "selection_rule": "sorted(test split users); numpy.default_rng(seed).choice(max_users, replace=False); sort selected indices",
        "user_indices": [int(value) for value in chosen.tolist()],
        "user_ids": user_ids,
        "target_pairs": [{"user": user, "item": item} for user, item in pairs],
        "user_set_sha256": _json_sha(user_ids),
        "target_pairs_sha256": _json_sha(pairs),
        "targets": targets,
        "users": chosen,
    }


def _visible_latest_times(data) -> dict[int, float]:
    latest: dict[int, float] = {}
    for row in list(data.train_interactions) + list(data.validation_interactions):
        latest[row.user_idx] = max(latest.get(row.user_idx, float("-inf")), row.timestamp)
    return latest


def _boundary_checks(data, sample: Mapping[str, Any]) -> dict[str, Any]:
    histories = build_observed_histories(data, "test")
    seen = seen_items_for_split(data, "test")
    latest = _visible_latest_times(data)
    target_items = 0
    history_hits = 0
    seen_hits = 0
    time_violations = 0
    for row in sample["targets"]:
        target_items += 1
        if row.item_idx in histories.get(row.user_idx, []):
            history_hits += 1
        if row.item_idx in seen.get(row.user_idx, set()):
            seen_hits += 1
        if latest.get(row.user_idx, float("-inf")) >= row.timestamp:
            time_violations += 1
    if history_hits or seen_hits or time_violations:
        raise TestFusionError(
            f"测试目标边界检查失败：history={history_hits}, seen={seen_hits}, time={time_violations}"
        )
    return {
        "status": "pass",
        "users_checked": int(len(sample["users"])),
        "targets_checked": target_items,
        "targets_in_history": history_hits,
        "targets_in_seen_filter": seen_hits,
        "time_violations": time_violations,
        "rule": "每个测试目标物品不在 train+validation 历史、不在 test 已见过滤集合，且目标时间晚于可见历史",
    }


def _validate_recall(payload: Mapping[str, Any], users: np.ndarray, seen: Any, k: int, name: str) -> None:
    expected = {int(value) for value in users.tolist()}
    if set(int(key) for key in payload) != expected:
        raise TestFusionError(f"{name} 缓存用户集合不匹配")
    for user in users:
        rows = payload[int(user)]
        if not rows or len(rows) > k:
            raise TestFusionError(f"{name} 用户 {user} 候选数越界")
        items = [int(row["item"]) for row in rows]
        if len(items) != len(set(items)):
            raise TestFusionError(f"{name} 用户 {user} 候选重复")
        if [int(row["rank"]) for row in rows] != list(range(1, len(rows) + 1)):
            raise TestFusionError(f"{name} 用户 {user} 名次不连续")
        leaked = set(items) & seen.get(int(user), set())
        if leaked:
            raise TestFusionError(f"{name} 用户 {user} 含已见物品：{sorted(leaked)[:3]}")


def _validate_cache_meta(
    meta: Mapping[str, Any],
    sample: Mapping[str, Any],
    k: int,
    labels: list[str],
    two_tower_sha256: str,
    mind_sha256: str,
    routing_seed: int,
) -> None:
    if meta.get("split") != "test":
        raise TestFusionError("测试缓存 split 不是 test")
    if meta.get("user_set_sha256") != sample["user_set_sha256"]:
        raise TestFusionError("测试缓存用户集合与冻结样本不一致")
    if int(meta.get("k_recall", 0)) != k:
        raise TestFusionError("测试缓存 Top-K 与冻结配置不一致")
    if list(meta.get("quotas", [])) != labels:
        raise TestFusionError("测试缓存配额与冻结配置不一致")
    if meta.get("two_tower_checkpoint_sha256") != two_tower_sha256:
        raise TestFusionError("测试缓存双塔权重哈希与当前配置不一致")
    if meta.get("mind_checkpoint_sha256") != mind_sha256:
        raise TestFusionError("测试缓存 MIND 权重哈希与当前配置不一致")
    if int(meta.get("routing_seed", -1)) != routing_seed:
        raise TestFusionError("测试缓存 MIND 路由 seed 与当前配置不一致")


def _stage(state: dict[str, Any], run_dir: Path, name: str) -> None:
    state["current_stage"] = name
    state["updated_at"] = _iso(_now())
    _write_json(run_dir / "run_state.json", state)


def _fixture_check(config: Mapping[str, Any]) -> dict[str, Any]:
    data = prepare_dataset("kuasar_small", _data_dir())
    sample = _select_sample(data, config["experiment"])
    # 小夹具只取冻结样本前 8 人；正式入口仍会检查全部 128 人。
    fixture = dict(sample)
    fixture["users"] = sample["users"][:8]
    fixture["targets"] = [row for row in sample["targets"] if row.user_idx in set(fixture["users"].tolist())]
    result = _boundary_checks(data, fixture)
    result.update(
        {
            "sample_users": sample["actual_users"],
            "sample_targets": sample["target_count"],
            "fixture_users": int(len(fixture["users"])),
            "fixture_targets": int(len(fixture["targets"])),
            "user_set_sha256": sample["user_set_sha256"],
            "target_pairs_sha256": sample["target_pairs_sha256"],
        }
    )
    return result


def run(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    started_wall = _now()
    started = time.perf_counter()
    deadline = started + float(config["experiment"]["max_wall_seconds"])
    experiment = config["experiment"]
    k = int(experiment["k_recall"])
    threads = int(experiment["threads"])
    torch.set_num_threads(threads)
    run_dir = _make_run_dir(output_root)
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "split": "test",
        "started_at": _iso(started_wall),
        "current_stage": "init",
        "completed_stages": [],
        "last_completed_stage": None,
        "elapsed_s": 0.0,
    }
    _write_json(run_dir / "run_state.json", state)
    shutil.copyfile(config["path"], run_dir / "frozen_config.toml")
    _write_json(
        run_dir / "frozen_config.json",
        {
            "source": _rel(config["path"]),
            "sha256": _sha256(config["path"]),
            "split": "test",
            "seed": int(experiment["seed"]),
            "max_users": int(experiment["max_users"]),
            "k_recall": k,
            "quotas": config["labels"],
            "quota_values": config["values"],
            "max_wall_seconds": float(experiment["max_wall_seconds"]),
            "threads": threads,
        },
    )
    timings: dict[str, float] = {}
    stage_times: list[dict[str, Any]] = []

    def check_deadline(stage: str) -> None:
        if time.perf_counter() > deadline:
            raise StageTimeout(f"超过 {experiment['max_wall_seconds']} 秒墙钟上限，最后阶段：{stage}")

    def finish_stage(name: str, started_at: float) -> None:
        elapsed = round(time.perf_counter() - started_at, 3)
        timings[name] = elapsed
        stage_times.append({"stage": name, "elapsed_s": elapsed, "status": "completed"})
        state["completed_stages"].append(name)
        state["last_completed_stage"] = name
        state["elapsed_s"] = round(time.perf_counter() - started, 3)
        _stage(state, run_dir, "between_stages")

    try:
        _stage(state, run_dir, "load_data")
        check_deadline("load_data")
        stage = time.perf_counter()
        data = prepare_dataset("kuasar_small", _data_dir())
        provenance = build_provenance(data, dataset="kuasar_small", positive_threshold=1.0)
        finish_stage("load_data", stage)

        _stage(state, run_dir, "freeze_sample")
        check_deadline("freeze_sample")
        stage = time.perf_counter()
        sample = _select_sample(data, experiment)
        sample_meta = {
            "split": sample["split"],
            "seed": sample["seed"],
            "max_users": sample["max_users"],
            "actual_users": sample["actual_users"],
            "target_count": sample["target_count"],
            "selection_rule": sample["selection_rule"],
            "user_ids": sample["user_ids"],
            "user_set_sha256": sample["user_set_sha256"],
            "target_pairs_sha256": sample["target_pairs_sha256"],
            "target_pairs": sample["target_pairs"],
        }
        cache_dir = run_dir / "cache" / "test"
        if cache_dir.exists():
            raise TestFusionError(f"测试缓存目录已存在，拒绝覆盖：{cache_dir}")
        cache_dir.mkdir(parents=True)
        _write_json(cache_dir / "users.json", sample_meta)
        _write_json(run_dir / "frozen_users.json", sample_meta)
        _write_json(
            run_dir / "frozen_fingerprints.json",
            {
                "dataset": "kuasar_small",
                "split": "test",
                "split_strategy": data.split_strategy,
                "split_sha256": provenance["data"]["split_sha256"],
                "id_mapping_sha256": provenance["id_mapping"]["sha256"],
                "users": int(data.num_users),
                "items": int(data.num_items),
                "two_tower_checkpoint": {
                    "path": _rel(config["two_tower"]),
                    "sha256": _sha256(config["two_tower"]),
                },
                "mind_checkpoint": {
                    "path": _rel(config["mind"]),
                    "sha256": _sha256(config["mind"]),
                },
                "routing_seed": int(config["routing_seed"]),
                "k_recall": k,
            },
        )
        finish_stage("freeze_sample", stage)

        _stage(state, run_dir, "boundary_check")
        check_deadline("boundary_check")
        stage = time.perf_counter()
        boundary = _boundary_checks(data, sample)
        finish_stage("boundary_check", stage)

        cache_meta = {
            "schema_version": SCHEMA_VERSION,
            "dataset": "kuasar_small",
            "split": "test",
            "seed": int(experiment["seed"]),
            "max_users": int(experiment["max_users"]),
            "k_recall": k,
            "quotas": config["labels"],
            "user_set_sha256": sample["user_set_sha256"],
            "target_pairs_sha256": sample["target_pairs_sha256"],
            "created_at": _iso(_now()),
            "two_tower_checkpoint_sha256": _sha256(config["two_tower"]),
            "mind_checkpoint_sha256": _sha256(config["mind"]),
            "routing_seed": int(config["routing_seed"]),
        }
        _write_json(cache_dir / "cache_meta.json", cache_meta)

        _stage(state, run_dir, "recall_two_tower")
        check_deadline("recall_two_tower")
        stage = time.perf_counter()
        two_tower = recsys_mvp.recall_two_tower(
            data, sample["users"], k, split="test", checkpoint=config["two_tower"]
        )
        timings["recall_two_tower_s"] = round(time.perf_counter() - stage, 3)
        _write_json(cache_dir / "recall_two_tower.json", {str(user): rows for user, rows in two_tower.items()})
        finish_stage("recall_two_tower", stage)

        _stage(state, run_dir, "recall_mind")
        check_deadline("recall_mind")
        stage = time.perf_counter()
        mind = recsys_mvp.recall_mind(
            data, sample["users"], k, split="test", checkpoint=config["mind"]
        )
        timings["recall_mind_s"] = round(time.perf_counter() - stage, 3)
        _write_json(cache_dir / "recall_mind.json", {str(user): rows for user, rows in mind.items()})
        finish_stage("recall_mind", stage)

        _stage(state, run_dir, "validate_cache")
        check_deadline("validate_cache")
        stage = time.perf_counter()
        seen = seen_items_for_split(data, "test")
        _validate_recall(two_tower, sample["users"], seen, k, "two_tower")
        _validate_recall(mind, sample["users"], seen, k, "mind")
        _validate_cache_meta(
            cache_meta,
            sample,
            k,
            config["labels"],
            _sha256(config["two_tower"]),
            _sha256(config["mind"]),
            int(config["routing_seed"]),
        )
        _write_json(
            cache_dir / "cache_manifest.json",
            {
                "cache_meta": cache_meta,
                "files": {
                    "recall_two_tower": {"sha256": _sha256(cache_dir / "recall_two_tower.json"), "rows": int(sum(len(v) for v in two_tower.values()))},
                    "recall_mind": {"sha256": _sha256(cache_dir / "recall_mind.json"), "rows": int(sum(len(v) for v in mind.values()))},
                },
            },
        )
        finish_stage("validate_cache", stage)

        targets = group_items_by_user(sample["targets"])
        metrics_by_quota: dict[str, Any] = {}
        outputs: dict[str, Any] = {}
        for label, (two_tower_quota, mind_quota) in zip(config["labels"], config["values"], strict=True):
            stage_name = f"evaluate_{label.replace('/', '_')}"
            _stage(state, run_dir, stage_name)
            check_deadline(stage_name)
            stage = time.perf_counter()
            fused: dict[int, list[dict]] = {}
            stats: dict[str, dict[str, int]] = {}
            for user in sample["users"]:
                pool, user_stats = fuse_by_quota(
                    two_tower[int(user)], mind[int(user)], two_tower_quota, mind_quota, budget=k
                )
                fused[int(user)] = pool
                stats[str(int(user))] = user_stats
            recommendations = {user: [row["item"] for row in pool] for user, pool in fused.items()}
            metric = evaluate_recommendations(recommendations, sample["targets"], ks=(k,))
            hits = sum(
                len({row["item"] for row in fused[int(user)]} & targets.get(int(user), set()))
                for user in sample["users"]
            )
            coverage = catalog_coverage(recommendations, data.num_items)
            elapsed = round(time.perf_counter() - stage, 3)
            timings[stage_name] = elapsed
            result = {
                "quota": {"two_tower": two_tower_quota, "mind": mind_quota, "label": label},
                "split": "test",
                "users": int(len(sample["users"])),
                "targets": int(len(sample["targets"])),
                "hits": int(hits),
                "recall_at_100": round(float(metric[f"Recall@{k}"]), 6),
                "coverage": round(float(coverage), 6),
                "candidates_total": int(sum(row["candidates"] for row in stats.values())),
                "candidates_mean": round(float(np.mean([row["candidates"] for row in stats.values()])), 3),
                "two_tower_used_total": int(sum(row["two_tower_used"] for row in stats.values())),
                "mind_used_total": int(sum(row["mind_used"] for row in stats.values())),
                "fills_total": int(sum(row["fills"] for row in stats.values())),
                "elapsed_s": elapsed,
                "user_set_sha256": sample["user_set_sha256"],
                "target_pairs_sha256": sample["target_pairs_sha256"],
            }
            metrics_by_quota[label] = result
            output_dir = run_dir / "evaluations" / label.replace("/", "_")
            _write_json(
                output_dir / "fusion.json",
                {
                    "quota": result["quota"],
                    "split": "test",
                    "users": result["users"],
                    "user_set_sha256": sample["user_set_sha256"],
                    "target_pairs_sha256": sample["target_pairs_sha256"],
                    "stats_by_user": stats,
                    "pool": {str(user): pool for user, pool in fused.items()},
                },
            )
            _write_json(output_dir / "metrics.json", result)
            outputs[label] = result
            finish_stage(stage_name, stage)

        delta = {
            "label": "30/70 - 50/50",
            "hits": int(outputs["30/70"]["hits"] - outputs["50/50"]["hits"]),
            "recall_at_100": round(outputs["30/70"]["recall_at_100"] - outputs["50/50"]["recall_at_100"], 6),
            "coverage": round(outputs["30/70"]["coverage"] - outputs["50/50"]["coverage"], 6),
        }
        total_s = round(time.perf_counter() - started, 3)
        if total_s > float(experiment["max_wall_seconds"]):
            raise StageTimeout(f"超过 {experiment['max_wall_seconds']} 秒墙钟上限，最后阶段：evaluate")
        budget = {
            "max_users": int(experiment["max_users"]),
            "actual_users": int(sample["actual_users"]),
            "formal_commands": 1,
            "two_tower_recalls": 1,
            "mind_recalls": 1,
            "quota_evaluations": len(config["labels"]),
            "training_runs": 0,
            "lightgbm_runs": 0,
            "rerank_runs": 0,
            "llm_calls": 0,
            "threads": threads,
            "max_wall_seconds": float(experiment["max_wall_seconds"]),
            "within_wall_clock": total_s <= float(experiment["max_wall_seconds"]),
        }
        result = {
            "schema_version": SCHEMA_VERSION,
            "phase": "Phase 13 kuasar_small Test Fusion",
            "status": "completed",
            "split": "test",
            "dataset": "kuasar_small",
            "sample": sample_meta,
            "boundary": boundary,
            "fingerprints": json.loads((run_dir / "frozen_fingerprints.json").read_text(encoding="utf-8")),
            "cache": {
                "dir": _rel(cache_dir),
                "meta": _rel(cache_dir / "cache_meta.json"),
                "two_tower": _rel(cache_dir / "recall_two_tower.json"),
                "mind": _rel(cache_dir / "recall_mind.json"),
                "files": json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))["files"],
            },
            "quotas": metrics_by_quota,
            "delta": delta,
            "timings": {**timings, "total_s": total_s},
            "stage_times": stage_times,
            "budget": budget,
            "run": {
                "output_dir": _rel(run_dir),
                "started_at": _iso(started_wall),
                "finished_at": _iso(_now()),
                "elapsed_s": total_s,
                "max_wall_seconds": float(experiment["max_wall_seconds"]),
                "threads": threads,
                "command": " ".join(sys.argv),
            },
        }
        _write_json(run_dir / "metrics.json", result)
        _write_json(
            run_dir / "run_meta.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "completed",
                "split": "test",
                "dataset": "kuasar_small",
                "started_at": result["run"]["started_at"],
                "finished_at": result["run"]["finished_at"],
                "elapsed_s": total_s,
                "command": result["run"]["command"],
                "config": _rel(config["path"]),
                "config_sha256": _sha256(config["path"]),
                "sample": sample_meta,
                "boundary": boundary,
                "fingerprints": result["fingerprints"],
                "cache": result["cache"],
                "timings": result["timings"],
                "budget": budget,
                "last_completed_stage": state["last_completed_stage"],
            },
        )
        report = render_report(result)
        (run_dir / "report.md").write_text(report, encoding="utf-8")
        state.update({"status": "completed", "elapsed_s": total_s, "current_stage": "done", "finished_at": _iso(_now())})
        _write_json(run_dir / "run_state.json", state)
        return {"status": "completed", "run_dir": run_dir, "metrics": result, "report": report}

    except StageTimeout as error:
        total_s = round(time.perf_counter() - started, 3)
        state.update(
            {
                "status": "timeout",
                "error": str(error),
                "elapsed_s": total_s,
                "finished_at": _iso(_now()),
            }
        )
        _write_json(run_dir / "run_state.json", state)
        _write_json(
            run_dir / "run_meta.json",
            {
                "status": "timeout",
                "split": "test",
                "elapsed_s": total_s,
                "last_completed_stage": state["last_completed_stage"],
                "error": str(error),
                "timings": timings,
                "stage_times": stage_times,
            },
        )
        (run_dir / "report.md").write_text(render_timeout_report(run_dir, state, timings), encoding="utf-8")
        raise
    except Exception as error:  # noqa: BLE001
        total_s = round(time.perf_counter() - started, 3)
        state.update({"status": "failed", "error": f"{type(error).__name__}: {error}", "elapsed_s": total_s})
        _write_json(run_dir / "run_state.json", state)
        _write_json(
            run_dir / "run_meta.json",
            {
                "status": "failed",
                "split": "test",
                "elapsed_s": total_s,
                "last_completed_stage": state["last_completed_stage"],
                "error": f"{type(error).__name__}: {error}",
                "timings": timings,
                "stage_times": stage_times,
            },
        )
        raise


def render_timeout_report(run_dir: Path, state: Mapping[str, Any], timings: Mapping[str, float]) -> str:
    return f"""# kuasar_small 独立测试集融合验证 · 超时

- 状态：**timeout**
- 最后完成阶段：`{state.get('last_completed_stage')}`
- 当前阶段：`{state.get('current_stage')}`
- 已用时间：{state.get('elapsed_s')} s
- 错误：{state.get('error')}
- 已记录耗时：{json.dumps(dict(timings), ensure_ascii=False)}

未完成评测；不自动重试，也不扩大样本。运行目录：`{_rel(run_dir)}`。
"""


def render_report(result: Mapping[str, Any]) -> str:
    sample = result["sample"]
    a = result["quotas"]["50/50"]
    b = result["quotas"]["30/70"]
    delta = result["delta"]
    boundary = result["boundary"]
    budget = result["budget"]
    timings = result["timings"]
    return f"""# 2026-11-09 · kuasar_small 独立测试集融合验证

> 只验证召回融合：复用现有双塔与 MIND 权重，在冻结的 128 个测试用户上各召回一次，按固定配额评测。没有训练、LightGBM 排序、重排或 LLM 调用。

## 结论

- **50/50**：命中 {a['hits']}；Recall@100 **{a['recall_at_100']:.6f}**；Coverage **{a['coverage']:.6f}**。
- **30/70**：命中 {b['hits']}；Recall@100 **{b['recall_at_100']:.6f}**；Coverage **{b['coverage']:.6f}**。
- **30/70 − 50/50**：命中 {delta['hits']:+d}；Recall@100 {delta['recall_at_100']:+.6f}；Coverage {delta['coverage']:+.6f}。
- 结论只覆盖这批独立测试用户与这两个配额。

## 1. 冻结样本

- 数据集 / 切分：`kuasar_small` / `test`；时间切分 `{result['fingerprints']['split_strategy']}`。
- 抽样：seed `{sample['seed']}`，最多 `{sample['max_users']}` 人；实际 **{sample['actual_users']}** 人 / **{sample['target_count']}** 个目标。
- 用户集合 sha256：`{sample['user_set_sha256']}`。
- 目标对 sha256：`{sample['target_pairs_sha256']}`。
- 规则：`{sample['selection_rule']}`。

## 2. 时间边界与权重

- 边界检查：**{boundary['status']}**；检查 {boundary['targets_checked']} 个目标，目标进入历史 {boundary['targets_in_history']}、进入已见过滤 {boundary['targets_in_seen_filter']}、时间违规 {boundary['time_violations']}。
- 双塔权重：`{result['fingerprints']['two_tower_checkpoint']['path']}`，sha256 `{result['fingerprints']['two_tower_checkpoint']['sha256']}`。
- MIND 权重：`{result['fingerprints']['mind_checkpoint']['path']}`，sha256 `{result['fingerprints']['mind_checkpoint']['sha256']}`。
- MIND 路由 seed：`{result['fingerprints']['routing_seed']}`；Top-K：`{result['fingerprints']['k_recall']}`。

## 3. 测试缓存

- 独立目录：`{result['cache']['dir']}`。
- 双塔缓存：`{result['cache']['two_tower']}`（{result['cache']['files']['recall_two_tower']['rows']} 行）。
- MIND 缓存：`{result['cache']['mind']}`（{result['cache']['files']['recall_mind']['rows']} 行）。
- 两组配额读取同一份缓存；缓存 `split=test`，用户集合哈希与冻结样本一致。

## 4. 指标

| 配额（双塔 / MIND） | 用户 | 目标 | 命中 | Recall@100 | Coverage | 候选均值 | 双塔占用 | MIND 占用 | 补位 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50/50 | {a['users']} | {a['targets']} | {a['hits']} | {a['recall_at_100']:.6f} | {a['coverage']:.6f} | {a['candidates_mean']} | {a['two_tower_used_total']} | {a['mind_used_total']} | {a['fills_total']} |
| 30/70 | {b['users']} | {b['targets']} | {b['hits']} | {b['recall_at_100']:.6f} | {b['coverage']:.6f} | {b['candidates_mean']} | {b['two_tower_used_total']} | {b['mind_used_total']} | {b['fills_total']} |
| 差值 | — | — | {delta['hits']:+d} | {delta['recall_at_100']:+.6f} | {delta['coverage']:+.6f} | — | — | — | — |

## 5. 预算与耗时

| 项 | 实际 |
| --- | --- |
| 正式命令 | {budget['formal_commands']} 次 |
| 双塔召回 / MIND 召回 | {budget['two_tower_recalls']} / {budget['mind_recalls']} 次 |
| 配额评测 | {budget['quota_evaluations']} 次 |
| 训练 / LightGBM / 重排 / LLM | 0 / 0 / 0 / 0 |
| 线程 / 墙钟上限 | {budget['threads']} / {budget['max_wall_seconds']:.0f} s |
| 实际总耗时 | {timings['total_s']:.3f} s（{budget['within_wall_clock']}） |

阶段耗时：

| 阶段 | 秒 |
| --- | ---: |
| 数据加载 | {timings.get('load_data', 0):.3f} |
| 冻结样本 | {timings.get('freeze_sample', 0):.3f} |
| 边界检查 | {timings.get('boundary_check', 0):.3f} |
| 双塔召回 | {timings.get('recall_two_tower_s', 0):.3f} |
| MIND 召回 | {timings.get('recall_mind_s', 0):.3f} |
| 缓存校验 | {timings.get('validate_cache', 0):.3f} |
| 50/50 评测 | {timings.get('evaluate_50_50', 0):.3f} |
| 30/70 评测 | {timings.get('evaluate_30_70', 0):.3f} |

## 6. 验收与边界

- 独立测试用户、测试目标、缓存 `split=test`，且两组读取同一缓存。
- 用户集合、目标集合、配额和指标口径在召回前冻结；未按命中筛选用户或更换 seed。
- 只记录融合召回效果，不包含排序/重排；完整 Phase 13 的跨场景与 RAG 验证仍待办。
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kuasar_small 独立测试集固定配额验证")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--fixture-check", action="store_true", help="只做小样本时间边界夹具检查，不召回")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = _load_config(args.config)
    except Exception as error:  # noqa: BLE001
        print(json.dumps({"status": "blocked", "reason": str(error)}, ensure_ascii=False, indent=2))
        return 2
    if args.fixture_check:
        try:
            result = _fixture_check(config)
        except Exception as error:  # noqa: BLE001
            print(json.dumps({"status": "fail", "reason": str(error)}, ensure_ascii=False, indent=2))
            return 1
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    output_root = (args.output_root or config["output_root"]).resolve()
    try:
        result = run(config, output_root)
    except StageTimeout as error:
        print(f"[timeout] {error}")
        return 2
    except Exception as error:  # noqa: BLE001
        print(f"[failed] {type(error).__name__}: {error}")
        return 1
    metrics = result["metrics"]
    print(f"运行目录：{result['run_dir']}")
    print(json.dumps(metrics["quotas"], ensure_ascii=False, indent=2))
    print(json.dumps(metrics["delta"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
