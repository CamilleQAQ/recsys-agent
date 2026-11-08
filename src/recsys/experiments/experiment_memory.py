"""Phase 12：实验记忆 MVP。

把「历史优先」从只读 Phase 11 的 ``runs/*.json`` 扩展为可写的本地 JSON 索引：

* 评测成功后自动保存实际指标、输入指纹、评测配置、候选工件路径与来源；
* 实验键只由数据集/切分/目标/固定用户/两路召回哈希/配额/预算/规则/指标口径决定，
  日期、目录名、耗时都不参与；
* 写入先落盘 run record，再原子替换 ``index.json``；同一个键只保留一条；
* 读取时重新校验身份、成功状态、指标记录和候选工件可读性，不静默复用；
* 失败、``partial``、缺少工件的结果不会发布为可复用记录。

目录布局（``--memory-dir`` 指定）::

    memory/
    ├── index.json
    └── runs/<实验键>.json

索引和 run record 都包含 identity 与 evidence；候选明细仍留在
``evaluate_fusion_quota`` 的 ``fusion.json`` 中，不复制、不改写。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recsys.experiments.experiment_history import (
    FUSION_RULE_VERSION,
    METRIC_DEFINITION,
    search_experiment_history,
    validate_result_artifact,
)
from recsys.experiments.fusion_quota import (
    ALLOWED_QUOTAS,
    TOTAL_BUDGET,
    load_context,
    validate_quota,
)

MEMORY_SCHEMA_VERSION = 1
INDEX_FILENAME = "index.json"
RECORD_DIRNAME = "runs"

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOTS = (
    Path("artifacts/kuasar_small/quota_agent/cache_eval"),
    Path("artifacts/kuasar_small/llm_tool_call/cache_eval"),
    Path("artifacts/kuasar_small/langgraph_tool_call/cache_eval"),
)

METRIC_KEYS = (
    "users",
    "targets",
    "candidates_total",
    "candidates_mean",
    "two_tower_used_total",
    "mind_used_total",
    "fills_total",
    "hits",
    "recall_at_100",
    "coverage",
    "diagnostic_union_hits_at_100",
)


class MemoryError(RuntimeError):
    """索引、run record 或工件不可用。"""


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, payload: Any) -> None:
    """临时文件 + ``os.replace``：崩溃时不会留下半截索引。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def targets_sha256(context: Any) -> str:
    """验证目标的稳定指纹（按用户排序后哈希，不依赖行顺序/目录）。"""
    payload = {
        str(int(user)): sorted(int(item) for item in context.targets.get(int(user), set()))
        for user in context.users
    }
    return _sha256_text(_canonical(payload))


def build_identity(context: Any, quota: tuple[int, int]) -> dict:
    """稳定实验身份：不含运行日期、目录名和耗时。"""
    fingerprints = context.fingerprints
    checkpoints = fingerprints.get("checkpoints") or {}
    return {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "dataset": fingerprints.get("dataset"),
        "users": fingerprints.get("users"),
        "items": fingerprints.get("items"),
        "split_strategy": fingerprints.get("split_strategy"),
        "split_sha256": fingerprints.get("split_sha256"),
        "id_mapping_sha256": fingerprints.get("id_mapping_sha256"),
        "routing_seed": fingerprints.get("routing_seed"),
        "k_recall": fingerprints.get("k_recall"),
        "checkpoints": {
            "two_tower": (checkpoints.get("two_tower") or {}).get("sha256"),
            "mind": (checkpoints.get("mind") or {}).get("sha256"),
        },
        "users_file_sha256": fingerprints.get("users_file_sha256"),
        "recall_two_tower_sha256": fingerprints.get("recall_two_tower_sha256"),
        "recall_mind_sha256": fingerprints.get("recall_mind_sha256"),
        "targets_sha256": targets_sha256(context),
        "fixed_users": int(len(context.users)),
        "validation_targets": int(len(context.target_interactions)),
        "quota": {"two_tower": int(quota[0]), "mind": int(quota[1])},
        "candidate_budget": TOTAL_BUDGET,
        "fusion_rule_version": FUSION_RULE_VERSION,
        "metric_definition": METRIC_DEFINITION,
    }


def identity_key(identity: dict) -> str:
    return _sha256_text(_canonical(identity))


def _metric_view(result: dict) -> dict:
    metrics = {
        "quota": result.get("quota"),
        **{key: result[key] for key in METRIC_KEYS if key in result},
    }
    return metrics


def _index_path(memory_dir: Path) -> Path:
    return Path(memory_dir) / INDEX_FILENAME


def _record_path(memory_dir: Path, key: str) -> Path:
    return Path(memory_dir) / RECORD_DIRNAME / f"{key}.json"


def load_index(memory_dir: Path) -> tuple[dict, Path]:
    path = _index_path(memory_dir)
    if not path.exists():
        return {"schema_version": MEMORY_SCHEMA_VERSION, "entries": {}}, path
    try:
        data = _read_json(path)
    except (OSError, json.JSONDecodeError) as error:
        raise MemoryError(f"实验索引无法读取：{error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
        raise MemoryError("实验索引结构不合法：缺少 entries")
    if data.get("schema_version") != MEMORY_SCHEMA_VERSION:
        raise MemoryError(f"实验索引版本不兼容：{data.get('schema_version')}")
    return data, path


def _entry_reasons(entry: dict, *, key: str, identity: dict, context: Any) -> list[str]:
    reasons: list[str] = []
    if not isinstance(entry, dict):
        return ["索引记录不是 JSON 对象"]
    if entry.get("key") != key:
        reasons.append("索引记录 key 与当前实验身份不一致")
    if entry.get("status") != "success":
        reasons.append(f"索引记录状态不可复用：{entry.get('status')}")
    if entry.get("identity") != identity:
        reasons.append("索引记录 identity 与当前输入/规则不一致")
    evaluation = entry.get("evaluation")
    if not isinstance(evaluation, dict):
        reasons.append("索引记录缺少评测结果")
        return reasons
    for field in ("users", "targets", "candidates_total", "candidates_mean", "hits", "recall_at_100"):
        if field not in evaluation:
            reasons.append(f"评测结果缺少指标 {field}")
    if reasons:
        return reasons
    if int(evaluation["users"]) != len(context.users):
        reasons.append("评测结果用户数与当前缓存不一致")
    if int(evaluation["targets"]) != len(context.target_interactions):
        reasons.append("评测结果目标数与当前验证集不一致")
    if int(evaluation["candidates_total"]) != len(context.users) * TOTAL_BUDGET:
        reasons.append("候选总数与每用户 100 的预算不一致")
    if float(evaluation["candidates_mean"]) != float(TOTAL_BUDGET):
        reasons.append("每用户候选预算不是 100")
    metrics = entry.get("metrics") or {}
    for field in ("hits", "recall_at_100"):
        if metrics.get(field) != evaluation.get(field):
            reasons.append(f"索引 metrics.{field} 与评测结果不一致")

    artifact_reasons, _ = validate_result_artifact(evaluation, context.fingerprints)
    reasons.extend(artifact_reasons)

    evidence = entry.get("evidence") or {}
    record_value = evidence.get("record")
    if record_value:
        record_path = Path(record_value)
        if not record_path.is_absolute():
            record_path = REPO_ROOT / record_path
        if not record_path.exists():
            reasons.append(f"run record 不存在：{record_value}")
        else:
            try:
                record = _read_json(record_path)
            except (OSError, json.JSONDecodeError) as error:
                reasons.append(f"run record 无法读取：{error}")
            else:
                if record.get("key") != key:
                    reasons.append("run record 的 key 与索引不一致")
    reasons = list(dict.fromkeys(reasons))
    return reasons


def lookup_index_entry(memory_dir: Path, context: Any, quota: tuple[int, int]) -> dict:
    """按当前实验身份读取索引；只有身份/状态/指标/工件全部通过才 hit。"""
    quota = validate_quota(*quota)
    identity = build_identity(context, quota)
    key = identity_key(identity)
    result: dict[str, Any] = {
        "status": "miss",
        "hit": False,
        "key": key,
        "identity": identity,
        "memory_dir": str(memory_dir),
        "index": str(_index_path(memory_dir)),
        "reasons": [],
    }
    try:
        index, index_path = load_index(memory_dir)
    except MemoryError as error:
        result["reason"] = str(error)
        result["reasons"] = [str(error)]
        return result
    entry = index["entries"].get(key)
    if entry is None:
        result["reason"] = "本地实验索引里没有该实验键"
        return result
    reasons = _entry_reasons(entry, key=key, identity=identity, context=context)
    if reasons:
        result["reason"] = "本地实验索引记录不兼容：" + "；".join(reasons[:3])
        result["reasons"] = reasons
        result["entry"] = entry
        return result
    result.update(
        status="hit",
        hit=True,
        reason="命中本地实验索引中可复用的成功记录",
        entry=entry,
        index_path=str(index_path),
    )
    return result


def save_evaluation(
    memory_dir: Path,
    *,
    cache_dir: Path,
    quota: tuple[int, int],
    result: dict,
    source: dict | None = None,
    experiment_dir: Path | None = None,
) -> dict:
    """评测成功后写入 run record，再原子更新索引；返回写入记录与索引状态。"""
    quota = validate_quota(*quota)
    memory_dir = Path(memory_dir)
    context = load_context(cache_dir)
    identity = build_identity(context, quota)
    key = identity_key(identity)

    result_quota = result.get("quota") or {}
    if (result_quota.get("two_tower"), result_quota.get("mind")) != quota:
        raise MemoryError(f"评测结果配额与目标不一致：{result_quota} vs {quota}")
    for field in ("users", "targets", "candidates_total", "candidates_mean", "hits", "recall_at_100"):
        if field not in result:
            raise MemoryError(f"评测结果缺少指标 {field}")
    artifact_reasons, fusion_path = validate_result_artifact(result, context.fingerprints)
    if artifact_reasons:
        raise MemoryError("候选工件核验失败：" + "；".join(artifact_reasons))
    if fusion_path is None or not fusion_path.exists():
        raise MemoryError("候选工件路径不可读")

    record_path = _record_path(memory_dir, key)
    index_path = _index_path(memory_dir)
    metrics = _metric_view(result)
    evaluation = json.loads(json.dumps(result, ensure_ascii=False))
    evidence = {
        "fusion": str(fusion_path),
        "fusion_sha256": _sha256_file(fusion_path),
        "metrics": str(fusion_path.parent / "metrics.json")
        if (fusion_path.parent / "metrics.json").exists()
        else None,
        "cache": result.get("evidence", {}).get("cache"),
        "record": str(record_path),
        "index": str(index_path),
    }
    record = {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "key": key,
        "status": "success",
        "identity": identity,
        "metrics": metrics,
        "evaluation": evaluation,
        "evidence": evidence,
        "source": {
            "tool": "evaluate_fusion_quota",
            "workflow": "history_first_agent",
            "policy": (source or {}).get("policy"),
            "cache_dir": str(cache_dir),
            "cache_dir_source": (source or {}).get("cache_dir_source"),
            "experiment_dir": str(experiment_dir) if experiment_dir else None,
            "created_at": _utc_now(),
        },
        "created_at": _utc_now(),
    }

    # 1) 先写 run record；2) 再原子替换索引；同键覆盖，保证只保留一条可复用记录。
    _atomic_write_json(record_path, record)
    index, index_path = load_index(memory_dir)
    entries = dict(index["entries"])
    entries[key] = record
    index_payload = {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "updated_at": _utc_now(),
        "entries": entries,
    }
    _atomic_write_json(index_path, index_payload)
    return {
        "key": key,
        "identity": identity,
        "record": record,
        "record_path": str(record_path),
        "index_path": str(index_path),
        "entry_count": len(entries),
    }


def _artifact_quota_dirs() -> list[dict]:
    rows: list[dict] = []
    pattern = re.compile(r"^(\d+)_(\d+)$")
    for relative in ARTIFACT_ROOTS:
        root = REPO_ROOT / relative
        if not root.exists():
            continue
        for path in sorted(root.iterdir()):
            match = pattern.match(path.name)
            if match and path.is_dir():
                rows.append(
                    {
                        "quota": [int(match.group(1)), int(match.group(2))],
                        "path": str(relative / path.name),
                    }
                )
    return rows


def audit_quotas(cache_dir: Path, history_root: Path, memory_dir: Path | None) -> dict:
    """为 Phase 12 选一个真正未入库的配额，并留下可核对的证据。"""
    context = load_context(cache_dir)
    history_root = Path(history_root)
    old_runs: list[dict] = []
    runs_dir = history_root / "runs"
    if runs_dir.exists():
        for path in sorted(runs_dir.glob("*.json")):
            try:
                payload = _read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            quota = payload.get("quota") or {}
            pair = (quota.get("two_tower"), quota.get("mind"))
            old_runs.append(
                {
                    "run": str(path),
                    "quota": list(pair) if all(value is not None for value in pair) else None,
                }
            )
    old_quotas = {
        tuple(row["quota"]) for row in old_runs if row["quota"] and all(isinstance(v, int) for v in row["quota"])
    }

    index_entries: list[dict] = []
    index_quotas: set[tuple[int, int]] = set()
    if memory_dir is not None:
        try:
            index, _ = load_index(memory_dir)
        except MemoryError as error:
            index_entries.append({"error": str(error)})
        else:
            for key, entry in index["entries"].items():
                identity = entry.get("identity") or {}
                quota = identity.get("quota") or {}
                pair = (quota.get("two_tower"), quota.get("mind"))
                index_entries.append(
                    {
                        "key": key,
                        "quota": list(pair) if all(value is not None for value in pair) else None,
                        "status": entry.get("status"),
                    }
                )
                if all(isinstance(value, int) for value in pair):
                    index_quotas.add((int(pair[0]), int(pair[1])))

    artifact_rows = _artifact_quota_dirs()
    artifact_quotas = {tuple(row["quota"]) for row in artifact_rows}
    registered = old_quotas | index_quotas
    unregistered = [quota for quota in ALLOWED_QUOTAS if quota not in registered]
    clean = [quota for quota in unregistered if quota not in artifact_quotas]
    selected = clean[0] if clean else (unregistered[0] if unregistered else None)

    probe = None
    if selected is not None:
        probe_result = search_experiment_history(
            selected[0],
            selected[1],
            cache_dir=cache_dir,
            history_root=history_root,
            memory_dir=memory_dir,
        )
        probe = {
            "status": probe_result.get("status"),
            "hit": probe_result.get("hit"),
            "reason": probe_result.get("reason"),
            "key": probe_result.get("memory_key"),
        }

    return {
        "dataset": "kuasar_small",
        "cache_dir": str(cache_dir),
        "history_root": str(history_root),
        "memory_dir": str(memory_dir) if memory_dir else None,
        "allowed_quotas": [list(quota) for quota in ALLOWED_QUOTAS],
        "phase11_runs": old_runs,
        "index_entries": index_entries,
        "candidate_artifact_quotas": [row["quota"] for row in artifact_rows],
        "candidate_artifacts": artifact_rows,
        "registered_quotas": [list(quota) for quota in sorted(registered)],
        "unregistered_quotas": [list(quota) for quota in unregistered],
        "unregistered_without_artifact": [list(quota) for quota in clean],
        "selected_quota": list(selected) if selected else None,
        "selected_probe": probe,
        "why": (
            f"{selected[0]}/{selected[1]} 没有 Phase 11 runs JSON、没有本地索引条目，"
            "也没有候选工件目录，因此是真正未入库的配额"
            if selected
            else "所有允许配额都已有记录，停止，避免重复评测"
        ),
    }


def _copy_memory(memory_dir: Path, target: Path) -> Path:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(memory_dir, target)
    return target


def _mutated_index_lookup(
    memory_dir: Path,
    target: Path,
    context: Any,
    quota: tuple[int, int],
    mutate,
) -> dict:
    target = _copy_memory(memory_dir, target)
    index, index_path = load_index(target)
    key = identity_key(build_identity(context, quota))
    entry = index["entries"].get(key)
    if entry is None:
        raise MemoryError("安全检查找不到有效索引记录")
    mutate(entry)
    _atomic_write_json(index_path, index)
    return lookup_index_entry(target, context, quota)


def run_memory_safety_checks(
    memory_dir: Path,
    *,
    cache_dir: Path,
    quota: tuple[int, int],
    work_dir: Path,
) -> dict:
    """缓存指纹变化 / 规则不匹配 / 工件缺失 / partial 均不得命中；不运行评测。"""
    memory_dir = Path(memory_dir)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    context = load_context(cache_dir)
    valid = lookup_index_entry(memory_dir, context, quota)

    changed_fingerprints = dict(context.fingerprints)
    changed_fingerprints["recall_mind_sha256"] = "0" * 64
    changed_context = replace(context, fingerprints=changed_fingerprints)
    fingerprint_changed = lookup_index_entry(memory_dir, changed_context, quota)

    def mutate_rule(entry: dict) -> None:
        entry.setdefault("identity", {})["fusion_rule_version"] = "fusion_quota_v2"

    rule_mismatch = _mutated_index_lookup(
        memory_dir, work_dir / "rule_mismatch", context, quota, mutate_rule
    )

    def mutate_artifact(entry: dict) -> None:
        missing = str(work_dir / "missing_fusion.json")
        entry.setdefault("evidence", {})["fusion"] = missing
        entry.setdefault("evaluation", {}).setdefault("evidence", {})["fusion"] = missing

    missing_artifact = _mutated_index_lookup(
        memory_dir, work_dir / "missing_artifact", context, quota, mutate_artifact
    )

    def mutate_status(entry: dict) -> None:
        entry["status"] = "partial"

    partial_status = _mutated_index_lookup(
        memory_dir, work_dir / "partial_status", context, quota, mutate_status
    )

    checks = {
        "valid_index_hit": bool(valid.get("hit")),
        "fingerprint_change_miss": not fingerprint_changed.get("hit"),
        "rule_mismatch_miss": not rule_mismatch.get("hit"),
        "missing_artifact_miss": not missing_artifact.get("hit"),
        "partial_status_miss": not partial_status.get("hit"),
    }
    return {
        "memory_dir": str(memory_dir),
        "cache_dir": str(cache_dir),
        "quota": list(quota),
        "key": valid.get("key"),
        "checks": checks,
        "details": {
            "valid": {"hit": valid.get("hit"), "reason": valid.get("reason")},
            "fingerprint_change": {
                "hit": fingerprint_changed.get("hit"),
                "reason": fingerprint_changed.get("reason"),
            },
            "rule_mismatch": {
                "hit": rule_mismatch.get("hit"),
                "reason": rule_mismatch.get("reason"),
            },
            "missing_artifact": {
                "hit": missing_artifact.get("hit"),
                "reason": missing_artifact.get("reason"),
            },
            "partial_status": {
                "hit": partial_status.get("hit"),
                "reason": partial_status.get("reason"),
            },
        },
        "passed": all(checks.values()),
    }
