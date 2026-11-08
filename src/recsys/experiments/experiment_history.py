"""按目标配额 + 当前缓存身份检索已落盘实验，供 Agent 的 ``search_experiment_history`` 工具使用。

设计原则：

* **不重跑评测**：命中时指标直接取自历史 ``runs/*.json``；
* **路径名不算证据**：必须逐项核对数据集/验证切分、固定用户与两路候选缓存的文件哈希、
  目标配额、每用户候选预算、融合规则与指标口径；
* **证据不足即 miss**：失败、``partial``、缺少工件或配置无法确认规则的记录一律拒绝并记录原因；
* **本地证据留在本地**：返回给工具的 evidence 路径只进日志，不进入模型消息。

默认历史目录是 Phase 11 的本地配额实验
``experiments/2026-11-08-kuasar_small-fusion-quota-agent``；旧记录没有规则版本字段时，
用同一实验目录的 ``config.toml`` + 可核查的融合工件确认兼容性，并在结果里标记
``legacy_record`` / ``inferred_from``。
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from recsys.experiments.fusion_quota import (
    DEFAULT_CACHE,
    TOTAL_BUDGET,
    QuotaError,
    load_context,
    validate_quota,
)

#: 仓库根目录：历史记录里的工件路径是相对仓库根目录保存的。
REPO_ROOT = Path(__file__).resolve().parents[3]

#: 默认只查 Phase 11 本地配额实验（不扫描整个 experiments 目录）。
DEFAULT_HISTORY_ROOT = Path("experiments/2026-11-08-kuasar_small-fusion-quota-agent")

#: 当前融合/指标口径版本；新记录应写入，旧记录可由配置与工件反推。
FUSION_RULE_VERSION = "fusion_quota_v1"
METRIC_DEFINITION = (
    "hits=(user,item) 命中对计数；"
    "recall_at_100=evaluate_recommendations(ks=(100,)) 的用户平均 Recall@100"
)

FINGERPRINT_KEYS = (
    "dataset",
    "users",
    "items",
    "split_strategy",
    "split_sha256",
    "id_mapping_sha256",
    "routing_seed",
    "k_recall",
    "users_file_sha256",
    "recall_two_tower_sha256",
    "recall_mind_sha256",
)

UNUSABLE_STATUSES = {"partial", "failed", "blocked", "running", "error"}


class HistoryError(QuotaError):
    """历史目录或运行记录不可用。"""


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_path(value: str | None) -> Path | None:
    """历史工件路径可能是相对仓库根目录的旧字符串。"""
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return REPO_ROOT / path


def _config_checks(config: dict, quota: tuple[int, int]) -> dict:
    """核对同一实验目录的配置：融合顺序、去重/补位、预算、目标与指标方向。"""
    task = config.get("task") or {}
    fusion = config.get("fusion_rule") or {}
    allowed_pairs = [list(pair) for pair in ((config.get("allowed_quotas") or {}).get("pairs") or [])]
    checks = {
        "dataset": task.get("dataset") == "kuasar_small",
        "fixed_users": task.get("fixed_users") == 128,
        "validation_targets": task.get("validation_targets") == 362,
        "candidate_budget": task.get("candidate_budget_per_user") == TOTAL_BUDGET,
        "fusion_order": list(fusion.get("order") or []) == ["two_tower", "mind"],
        "max_candidates": fusion.get("max_candidates") == TOTAL_BUDGET,
        "quota_consumption": "只有成功加入新物品才消耗该通道配额" in str(fusion.get("quota_consumption") or ""),
        "fill_rule": "补位" in str(fusion.get("fill_rule") or ""),
        "source_trace": "原始名次" in str(fusion.get("source_trace") or ""),
        "objective": (
            "命中目标数优先" in str(task.get("objective") or "")
            and "Recall@100" in str(task.get("objective") or "")
        ),
        "allowed_quota": list(quota) in allowed_pairs,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": [key for key, value in checks.items() if not value],
        "fusion_rule_version": FUSION_RULE_VERSION,
        "metric_definition": METRIC_DEFINITION,
        "source": "同一实验目录的 config.toml + 落盘融合工件",
    }


def _fingerprint_mismatches(recorded: Any, current: dict) -> list[str]:
    if not isinstance(recorded, dict):
        return ["缺少 evidence.fingerprints"]
    reasons: list[str] = []
    for key in FINGERPRINT_KEYS:
        if key not in recorded:
            reasons.append(f"缺少指纹 {key}")
        elif recorded.get(key) != current.get(key):
            reasons.append(f"指纹不一致 {key}")
    recorded_checkpoints = recorded.get("checkpoints") or {}
    current_checkpoints = current.get("checkpoints") or {}
    for name in ("two_tower", "mind"):
        recorded_sha = (recorded_checkpoints.get(name) or {}).get("sha256")
        current_sha = (current_checkpoints.get(name) or {}).get("sha256")
        if recorded_sha != current_sha:
            reasons.append(f"checkpoint 指纹不一致 {name}")
    return reasons


def _artifact_reasons(run: dict, recorded_fingerprints: Any) -> tuple[list[str], Path | None]:
    """核对融合工件与 run 记录：配额、预算、用户数、池大小、来源和逐用户统计。"""
    evidence = run.get("evidence") or {}
    fusion_path = _candidate_path(evidence.get("fusion"))
    if fusion_path is None or not fusion_path.exists():
        return [f"融合工件不存在：{evidence.get('fusion')}"], fusion_path
    try:
        artifact = _json(fusion_path)
    except (OSError, json.JSONDecodeError) as error:
        return [f"融合工件无法读取：{error}"], fusion_path

    reasons: list[str] = []
    if artifact.get("quota") != run.get("quota"):
        reasons.append("工件 quota 与 run 记录不一致")
    if artifact.get("budget") != TOTAL_BUDGET:
        reasons.append("工件候选预算不是 100")
    if artifact.get("users") != run.get("users"):
        reasons.append("工件用户数与 run 记录不一致")
    if artifact.get("fingerprints") != recorded_fingerprints:
        reasons.append("工件指纹与 run 记录不一致")

    pool = artifact.get("pool") or {}
    stats = artifact.get("stats_by_user") or {}
    if not isinstance(pool, dict) or len(pool) != int(run.get("users", -1)):
        reasons.append("工件候选池的用户数与 run 记录不一致")
    if not isinstance(stats, dict) or len(stats) != int(run.get("users", -1)):
        reasons.append("工件逐用户统计数与 run 记录不一致")
    if reasons:
        return reasons, fusion_path

    sizes: list[int] = []
    for user, rows in pool.items():
        if not isinstance(rows, list):
            reasons.append(f"用户 {user} 的候选池不是列表")
            break
        sizes.append(len(rows))
        items = [int(row.get("item")) for row in rows]
        if len(items) != len(set(items)):
            reasons.append(f"用户 {user} 的候选池有重复")
            break
        if any(not row.get("sources") for row in rows):
            reasons.append(f"用户 {user} 的候选缺少来源/原始名次")
            break
    else:
        if sum(sizes) != int(run.get("candidates_total", -1)):
            reasons.append("工件候选总数与 run 记录不一致")
        if any(size <= 0 or size > TOTAL_BUDGET for size in sizes):
            reasons.append(f"逐用户候选数不在 1..{TOTAL_BUDGET}")

    if not reasons:
        try:
            used_two = sum(int(value["two_tower_used"]) for value in stats.values())
            used_mind = sum(int(value["mind_used"]) for value in stats.values())
            fills = sum(int(value["fills"]) for value in stats.values())
            candidates = sum(int(value["candidates"]) for value in stats.values())
        except (KeyError, TypeError, ValueError) as error:
            reasons.append(f"工件逐用户统计字段缺失：{error}")
        else:
            if used_two != int(run.get("two_tower_used_total", -1)):
                reasons.append("工件双塔占用量与 run 记录不一致")
            if used_mind != int(run.get("mind_used_total", -1)):
                reasons.append("工件 MIND 占用量与 run 记录不一致")
            if fills != int(run.get("fills_total", -1)):
                reasons.append("工件补位数与 run 记录不一致")
            if candidates != int(run.get("candidates_total", -1)):
                reasons.append("工件候选数与 run 记录不一致")
    return reasons, fusion_path


def _validate_run(
    run: Any,
    *,
    path: Path,
    current_fingerprints: dict,
    quota: tuple[int, int],
    config_rule: dict,
) -> tuple[bool, list[str], dict]:
    """返回 (是否可复用, 拒绝原因, 本地匹配详情)。"""
    reasons: list[str] = []
    details: dict[str, Any] = {
        "run": str(path),
        "legacy_record": run.get("fusion_rule_version") is None if isinstance(run, dict) else True,
        "fusion_rule_version": run.get("fusion_rule_version") if isinstance(run, dict) else None,
    }
    if not isinstance(run, dict):
        return False, ["运行记录不是 JSON 对象"], details

    status = str(run.get("status") or "").lower()
    if status in UNUSABLE_STATUSES:
        reasons.append(f"运行状态不可复用：{status}")
    if run.get("error"):
        reasons.append(f"运行记录了错误：{run.get('error')}")

    recorded_quota = run.get("quota") or {}
    pair = (recorded_quota.get("two_tower"), recorded_quota.get("mind"))
    if pair != quota:
        reasons.append(f"目标配额不一致：记录 {pair}，期望 {quota}")

    for key in ("users", "targets", "candidates_total", "candidates_mean", "hits", "recall_at_100"):
        if key not in run:
            reasons.append(f"缺少指标 {key}")
    if reasons:
        return False, reasons, details

    if int(run["users"]) != 128 or int(run["targets"]) != 362:
        reasons.append(f"用户/目标口径不一致：{run['users']} / {run['targets']}")
    if abs(float(run["candidates_mean"]) - float(TOTAL_BUDGET)) > 1e-9:
        reasons.append(f"每用户候选预算不是 {TOTAL_BUDGET}：{run['candidates_mean']}")
    if int(run["candidates_total"]) != int(run["users"]) * TOTAL_BUDGET:
        reasons.append("候选总数与用户数 × 100 不一致")
    if not 0.0 <= float(run["recall_at_100"]) <= 1.0:
        reasons.append("Recall@100 不在 [0,1]")
    if int(run["hits"]) < 0 or int(run["hits"]) > int(run["targets"]):
        reasons.append("命中数不在合法范围")

    used_two = int(run.get("two_tower_used_total", -1))
    used_mind = int(run.get("mind_used_total", -1))
    fills = int(run.get("fills_total", -1))
    if used_two + used_mind + fills != int(run["candidates_total"]):
        reasons.append("占用量 + 补位数不等于候选总数")
    if used_two > quota[0] * int(run["users"]):
        reasons.append("双塔占用量超过配额上限")
    if used_mind > quota[1] * int(run["users"]):
        reasons.append("MIND 占用量超过配额上限")

    if run.get("fusion_rule_version") not in (None, FUSION_RULE_VERSION):
        reasons.append(f"融合规则版本不兼容：{run.get('fusion_rule_version')}")
    if run.get("metric_definition") not in (None, METRIC_DEFINITION):
        reasons.append("指标口径版本不兼容")
    if not config_rule.get("passed"):
        reasons.append(f"实验配置无法确认融合/指标口径：{config_rule.get('failed')}")
    details["rule_compatibility"] = {
        "legacy_record": details["legacy_record"],
        "inferred_from": config_rule.get("source"),
        "checks": config_rule.get("checks"),
        "fusion_rule_version": config_rule.get("fusion_rule_version"),
        "metric_definition": config_rule.get("metric_definition"),
    }

    recorded_fingerprints = (run.get("evidence") or {}).get("fingerprints")
    mismatches = _fingerprint_mismatches(recorded_fingerprints, current_fingerprints)
    reasons.extend(mismatches)

    artifact_reasons, fusion_path = _artifact_reasons(run, recorded_fingerprints)
    reasons.extend(artifact_reasons)
    details["artifact"] = str(fusion_path) if fusion_path else None
    details["fingerprints_verified"] = not mismatches

    # 去重保序，避免同一原因重复出现
    reasons = list(dict.fromkeys(reasons))
    return not reasons, reasons, details


def _load_run(
    path: Path,
    *,
    current_fingerprints: dict,
    quota: tuple[int, int],
    config_rule: dict,
) -> tuple[dict | None, list[str], dict]:
    try:
        run = _json(path)
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"运行记录无法读取：{error}"], {"run": str(path)}
    ok, reasons, details = _validate_run(
        run,
        path=path,
        current_fingerprints=current_fingerprints,
        quota=quota,
        config_rule=config_rule,
    )
    return (run if ok else None), reasons, details


def _metric_view(run: dict) -> dict:
    keys = (
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
    return {key: run[key] for key in keys if key in run}


def _baseline_view(run: dict) -> dict:
    return {"quota": run.get("quota"), **_metric_view(run)}


def search_experiment_history(
    two_tower_quota: int,
    mind_quota: int,
    *,
    cache_dir: Path = DEFAULT_CACHE,
    history_root: Path = DEFAULT_HISTORY_ROOT,
) -> dict:
    """查一组配额的历史结果；返回 ``hit``（含聚合指标/基准/证据）或 ``miss``（含原因）。"""
    quota = validate_quota(two_tower_quota, mind_quota)
    cache_dir = Path(cache_dir)
    history_root = Path(history_root)
    result: dict[str, Any] = {
        "tool": "search_experiment_history",
        "quota": {"two_tower": quota[0], "mind": quota[1]},
        "cache_dir": str(cache_dir),
        "history_root": str(history_root),
        "checked_runs": 0,
        "quota_candidates": 0,
        "rejected": [],
    }

    if not cache_dir.exists():
        raise HistoryError(f"缓存目录不存在：{cache_dir}")

    # 当前缓存身份：文件级 sha256 + 数据集/切分指纹，全部由 CLI 指定的目录计算。
    context = load_context(cache_dir)
    current = context.fingerprints
    result["current_fingerprints"] = current

    config_path = history_root / "config.toml"
    runs_dir = history_root / "runs"
    result["evidence"] = {
        "history_root": str(history_root),
        "config": str(config_path),
        "cache": str(cache_dir),
    }
    if not config_path.exists():
        result.update(
            status="miss",
            hit=False,
            reason="历史实验配置缺失，无法确认融合规则与指标口径",
        )
        return result
    try:
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        result.update(status="miss", hit=False, reason=f"历史实验配置无法读取：{error}")
        return result

    config_rule = _config_checks(config, quota)
    result["rule_compatibility"] = {
        "legacy_record": True,
        "inferred_from": config_rule["source"],
        "checks": config_rule["checks"],
        "fusion_rule_version": FUSION_RULE_VERSION,
        "metric_definition": METRIC_DEFINITION,
    }
    if not config_rule["passed"]:
        result.update(
            status="miss",
            hit=False,
            reason=f"历史实验配置与当前融合/指标口径不兼容：{config_rule['failed']}",
        )
        return result

    baseline_path = runs_dir / "50_50.json"
    baseline_run, baseline_reasons, _ = _load_run(
        baseline_path,
        current_fingerprints=current,
        quota=(50, 50),
        config_rule=config_rule,
    )
    if baseline_run is None:
        result.update(
            status="miss",
            hit=False,
            reason="基准 50/50 记录不可复用：" + "；".join(baseline_reasons[:3]),
        )
        result["rejected"].append({"run": str(baseline_path), "reasons": baseline_reasons})
        return result
    result["baseline"] = _baseline_view(baseline_run)
    result["evidence"]["baseline_run"] = str(baseline_path)
    result["evidence"]["baseline_fusion"] = (baseline_run.get("evidence") or {}).get("fusion")

    run_paths = sorted(runs_dir.glob("*.json")) if runs_dir.exists() else []
    result["checked_runs"] = len(run_paths)
    if not run_paths:
        result.update(
            status="miss",
            hit=False,
            reason="历史实验目录里没有落盘记录",
        )
        return result

    for path in run_paths:
        if path.name == baseline_path.name:
            continue
        try:
            candidate = _json(path)
        except (OSError, json.JSONDecodeError) as error:
            result["rejected"].append({"run": str(path), "reasons": [f"运行记录无法读取：{error}"]})
            continue
        recorded_quota = (candidate.get("quota") or {}) if isinstance(candidate, dict) else {}
        pair = (recorded_quota.get("two_tower"), recorded_quota.get("mind"))
        if pair != quota:
            continue
        result["quota_candidates"] += 1
        ok, reasons, details = _validate_run(
            candidate,
            path=path,
            current_fingerprints=current,
            quota=quota,
            config_rule=config_rule,
        )
        if not ok:
            result["rejected"].append({"run": str(path), "reasons": reasons})
            continue

        delta_hits = int(candidate["hits"]) - int(baseline_run["hits"])
        delta_recall = round(float(candidate["recall_at_100"]) - float(baseline_run["recall_at_100"]), 6)
        rule = details.get("rule_compatibility") or {}
        rule = {**rule, "record_rule_version": candidate.get("fusion_rule_version")}
        result.update(
            status="hit",
            hit=True,
            reason="找到与当前缓存指纹、目标配额、候选预算和融合/指标口径一致的成功记录",
            metrics=_metric_view(candidate),
            delta_vs_baseline={"hits": delta_hits, "recall_at_100": delta_recall},
            fingerprints_verified=True,
            rule_compatibility=rule,
            evidence={
                **result["evidence"],
                "run": str(path),
                "fusion": details.get("artifact"),
            },
        )
        return result

    if result["quota_candidates"] == 0:
        reason = f"历史实验目录里没有 {quota[0]}/{quota[1]} 的落盘记录"
    else:
        reason = (
            f"找到 {result['quota_candidates']} 条 {quota[0]}/{quota[1]} 记录，"
            "但没有一条同时满足缓存文件哈希、候选预算与融合/指标口径"
        )
    result.update(status="miss", hit=False, reason=reason)
    return result


def model_view(result: dict) -> dict:
    """只保留可发给模型的聚合字段：无路径、无哈希、无原始 ID/候选明细。"""
    view: dict[str, Any] = {
        "tool": result.get("tool"),
        "status": result.get("status"),
        "hit": bool(result.get("hit")),
        "quota": result.get("quota"),
        "reason": result.get("reason"),
    }
    for key in ("metrics", "baseline", "delta_vs_baseline"):
        if key in result:
            view[key] = result[key]
    rule = result.get("rule_compatibility") or {}
    if rule:
        view["rule_compatibility"] = {
            "legacy_record": rule.get("legacy_record"),
            "fusion_rule_version": rule.get("fusion_rule_version"),
        }
    return view
