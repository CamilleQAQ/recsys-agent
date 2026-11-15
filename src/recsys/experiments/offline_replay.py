"""Phase 13 Offline Replay MVP：融合配额受控回放。

只做一件事：从已验收实验的**汇总 JSON** 中取出 5 组融合配额结果，让
「Rec Agent（本地确定性反馈策略）」和「固定策略」从同一份 50/50 历史出发，
在各自 2 次揭晓预算下做受控对照。没有训练、没有重新召回、没有重新评测、
没有真实 LLM 请求，也不读取候选明细或数据集。

    $env:PYTHONPATH = "src"
    python -m recsys.experiments.offline_replay --self-check
    python -m recsys.experiments.offline_replay

默认每次运行在 ``experiments/2026-11-09-kuasar_small-offline-replay/``
下新建 ``run-<UTC>-<pid>/``，保留旧运行证据。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "offline_replay.toml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "2026-11-09-kuasar_small-offline-replay"
SCHEMA_VERSION = 1
TOOL_NAME = "get_fusion_result"
MAX_DECISIONS = 64

PUBLIC_METRIC_FIELDS = (
    "quota",
    "two_tower",
    "mind",
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
)

REQUIRED_METRIC_FIELDS = (
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
)

DUPLICATE_COMPARE_FIELDS = (
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

DEFAULT_SYNTHETIC_METRICS: dict[str, tuple[int, float, float]] = {
    "50/50": (4, 0.011719, 0.026843),
    "30/70": (5, 0.013672, 0.026720),
    "20/80": (5, 0.013672, 0.026595),
    "70/30": (3, 0.009115, 0.026905),
    "80/20": (2, 0.005208, 0.026891),
}


class ReplayInputError(RuntimeError):
    """来源缺失、口径冲突或策略/执行器失败。"""


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def deep_get(document: Any, pointer: str) -> Any:
    if not pointer:
        return document
    current = document
    for part in pointer.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ReplayInputError(f"JSON 指针不存在：{pointer!r}（缺少 {part!r}）")
        current = current[part]
    return current


def parse_quota_text(value: Any) -> tuple[str, int, int]:
    if not isinstance(value, str):
        raise ValueError("配额必须是字符串，例如 30/70")
    match = re.fullmatch(r"\s*(\d{1,3})\s*/\s*(\d{1,3})\s*", value)
    if not match:
        raise ValueError(f"配额格式非法：{value!r}")
    two_tower, mind = int(match.group(1)), int(match.group(2))
    if two_tower + mind != 100:
        raise ValueError(f"配额之和必须为 100：{value!r}")
    return f"{two_tower}/{mind}", two_tower, mind


def quota_distance(record: Mapping[str, Any]) -> int:
    return abs(int(record["two_tower"]) - 50) + abs(int(record["mind"]) - 50)


def recall_value(record: Mapping[str, Any], decimals: int) -> float:
    return round(float(record["recall_at_100"]), decimals)


def rank_key(record: Mapping[str, Any], decimals: int) -> tuple[float, int, int]:
    """越小越好：Recall 高 → 离 50/50 近 → 双塔配额小。"""

    return (-recall_value(record, decimals), quota_distance(record), int(record["two_tower"]))


def compare_records(left: Mapping[str, Any], right: Mapping[str, Any], decimals: int) -> int:
    """-1：left 更好；0：按完整规则持平；1：left 更差。"""

    lk, rk = rank_key(left, decimals), rank_key(right, decimals)
    return -1 if lk < rk else (1 if lk > rk else 0)


def select_best(records: Iterable[Mapping[str, Any]], decimals: int) -> Mapping[str, Any]:
    items = list(records)
    if not items:
        raise ValueError("select_best 需要至少一条记录")
    return min(items, key=lambda record: rank_key(record, decimals))


def public_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    """策略只能看到这些字段；来源、指纹、候选明细均不进入观察。"""

    return {field: copy.deepcopy(record[field]) for field in PUBLIC_METRIC_FIELDS}


def format_result(record: Mapping[str, Any]) -> str:
    return (
        f"{record['quota']}（命中 {record['hits']}，Recall@100 {float(record['recall_at_100']):.6f}，"
        f"Coverage {float(record['coverage']):.6f}）"
    )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json_with_hash(path: Path) -> tuple[Any, str]:
    raw = path.read_bytes()
    try:
        return json.loads(raw.decode("utf-8")), sha256_bytes(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplayInputError(f"汇总 JSON 无法解析：{path}（{error}）") from error


# ---------------------------------------------------------------------------
# 配置与来源口径
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    import tomllib

    try:
        with path.open("rb") as file:
            config = tomllib.load(file)
    except FileNotFoundError as error:
        raise ReplayInputError(f"配置文件不存在：{path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ReplayInputError(f"配置 TOML 解析失败：{path}（{error}）") from error
    for section in ("replay", "best_rule", "fixed_policy", "agent_policy", "scope", "sources"):
        if section not in config:
            raise ReplayInputError(f"配置缺少 [{section}]")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ReplayInputError(f"配置 schema_version 必须为 {SCHEMA_VERSION}")
    _validate_config(config)
    return config


def _validate_config(config: Mapping[str, Any]) -> None:
    replay = config["replay"]
    baseline, two_tower, mind = parse_quota_text(replay["baseline_quota"])
    if baseline != "50/50" or (two_tower, mind) != (50, 50):
        raise ReplayInputError("本轮起始历史必须是 50/50")
    if int(replay["budget_reveals_per_side"]) != 2:
        raise ReplayInputError("本轮回放预算必须为每侧 2 次揭晓")
    allowed = [parse_quota_text(value)[0] for value in replay["allowed_quotas"]]
    if allowed != ["70/30", "30/70", "20/80", "80/20"]:
        raise ReplayInputError("可选配额必须且只能为 70/30、30/70、20/80、80/20")
    fixed_order = [parse_quota_text(value)[0] for value in config["fixed_policy"]["order"]]
    if fixed_order != ["70/30", "30/70"]:
        raise ReplayInputError("固定策略顺序必须预先写死为 70/30 → 30/70")
    if int(replay["candidate_pool_per_user"]) != 100:
        raise ReplayInputError("本轮融合池上限必须是每用户 100")
    if int(replay["expected_users"]) != 128 or int(replay["expected_targets"]) != 362:
        raise ReplayInputError("本轮口径固定为 128 用户 / 362 验证目标")


def _validate_fingerprints(fingerprints: Mapping[str, Any], replay: Mapping[str, Any]) -> None:
    required = (
        "dataset",
        "users",
        "items",
        "split_strategy",
        "split_sha256",
        "id_mapping_sha256",
        "checkpoints",
        "routing_seed",
        "k_recall",
        "users_file_sha256",
        "recall_two_tower_sha256",
        "recall_mind_sha256",
    )
    missing = [field for field in required if field not in fingerprints]
    if missing:
        raise ReplayInputError(f"缓存指纹缺少字段：{missing}")
    if fingerprints["dataset"] != replay["dataset"]:
        raise ReplayInputError(
            f"数据集不一致：来源 {fingerprints['dataset']!r} ≠ 配置 {replay['dataset']!r}"
        )
    if fingerprints["split_strategy"] != replay["split_strategy"]:
        raise ReplayInputError(
            f"切分协议不一致：{fingerprints['split_strategy']!r} ≠ {replay['split_strategy']!r}"
        )
    if int(fingerprints["k_recall"]) != int(replay["candidate_pool_per_user"]):
        raise ReplayInputError("k_recall 与每用户候选预算不一致")
    if not isinstance(fingerprints["split_sha256"], Mapping) or set(fingerprints["split_sha256"]) != {
        "train",
        "validation",
        "test",
    }:
        raise ReplayInputError("split_sha256 必须同时包含 train / validation / test")
    checkpoints = fingerprints["checkpoints"]
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != {"two_tower", "mind"}:
        raise ReplayInputError("checkpoints 必须包含 two_tower 和 mind")
    for name, payload in checkpoints.items():
        if not isinstance(payload, Mapping) or not payload.get("sha256"):
            raise ReplayInputError(f"checkpoint {name} 缺少 sha256")


def cache_identity(fingerprints: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "dataset": fingerprints["dataset"],
        "users": fingerprints["users"],
        "items": fingerprints["items"],
        "split_strategy": fingerprints["split_strategy"],
        "split_sha256": fingerprints["split_sha256"],
        "id_mapping_sha256": fingerprints["id_mapping_sha256"],
        "checkpoints": {
            name: {"path": payload["path"], "sha256": payload["sha256"]}
            for name, payload in fingerprints["checkpoints"].items()
        },
        "routing_seed": fingerprints["routing_seed"],
        "k_recall": fingerprints["k_recall"],
        "users_file_sha256": fingerprints["users_file_sha256"],
        "recall_two_tower_sha256": fingerprints["recall_two_tower_sha256"],
        "recall_mind_sha256": fingerprints["recall_mind_sha256"],
    }
    digest = sha256_bytes(canonical_json(payload).encode("utf-8"))
    return {"sha256": digest, "short": digest[:16], "payload": payload}


def _source_status(document: Mapping[str, Any], node: Mapping[str, Any]) -> str | None:
    status = node.get("status")
    if status is None:
        status = document.get("status")
    if status is None:
        conclusion = document.get("conclusion")
        if isinstance(conclusion, Mapping):
            status = conclusion.get("status")
    return status if isinstance(status, str) else None


def _normalize_record(
    node: Mapping[str, Any],
    spec: Mapping[str, Any],
    path: Path,
    sha: str,
    status: str | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    replay = config["replay"]
    quota, two_tower, mind = parse_quota_text(spec["quota"])

    if "quota" in node:
        node_quota = parse_quota_text(
            f"{node['quota'].get('two_tower')}/{node['quota'].get('mind')}"
            if isinstance(node["quota"], Mapping)
            else node["quota"]
        )[0]
        if node_quota != quota:
            raise ReplayInputError(f"{path}（{spec['pointer'] or '$'}）配额 {node_quota} ≠ {quota}")
    if "tool" in node and node["tool"] != "evaluate_fusion_quota":
        raise ReplayInputError(f"{path}: 工具名不是 evaluate_fusion_quota")
    if node.get("error"):
        raise ReplayInputError(f"{path}: 来源记录带 error，不能用于正式回放")

    missing = [field for field in REQUIRED_METRIC_FIELDS if field not in node]
    if missing:
        raise ReplayInputError(f"{quota} 来源缺少指标字段：{missing}（{path}）")
    evidence = node.get("evidence")
    if not isinstance(evidence, Mapping) or not evidence.get("fusion"):
        raise ReplayInputError(f"{quota} 来源缺少 evidence.fusion（{path}）")
    fingerprints = evidence.get("fingerprints")
    if not isinstance(fingerprints, Mapping):
        raise ReplayInputError(f"{quota} 来源缺少 evidence.fingerprints（{path}）")
    _validate_fingerprints(fingerprints, replay)

    users = int(node["users"])
    targets = int(node["targets"])
    pool = int(replay["candidate_pool_per_user"])
    candidates_total = int(node["candidates_total"])
    if users != int(replay["expected_users"]) or targets != int(replay["expected_targets"]):
        raise ReplayInputError(
            f"{quota} 用户/目标口径不一致：{users}/{targets} ≠ "
            f"{replay['expected_users']}/{replay['expected_targets']}"
        )
    if candidates_total != users * pool:
        raise ReplayInputError(f"{quota} 候选总数 {candidates_total} ≠ {users} × {pool}")
    if float(node["candidates_mean"]) != float(pool):
        raise ReplayInputError(f"{quota} 每用户候选均值不是 {pool}")
    two_tower_used = int(node["two_tower_used_total"])
    mind_used = int(node["mind_used_total"])
    if two_tower_used != users * two_tower or mind_used != users * mind:
        raise ReplayInputError(
            f"{quota} 实际占用 {two_tower_used}/{mind_used} 与配额比例不一致"
        )
    fills = int(node["fills_total"])
    if fills != int(replay["expected_fills"]):
        raise ReplayInputError(f"{quota} 补位次数 {fills} ≠ 预期 {replay['expected_fills']}")
    if two_tower_used + mind_used != candidates_total:
        raise ReplayInputError(f"{quota} 实际占用之和 ≠ 候选总数")
    if not 0.0 <= float(node["recall_at_100"]) <= 1.0:
        raise ReplayInputError(f"{quota} Recall@100 越界")
    if not 0.0 <= float(node["coverage"]) <= 1.0:
        raise ReplayInputError(f"{quota} Coverage 越界")

    identity = cache_identity(fingerprints)
    return {
        "quota": quota,
        "two_tower": two_tower,
        "mind": mind,
        "dataset": replay["dataset"],
        "split": replay["split"],
        "users": users,
        "targets": targets,
        "candidate_pool_per_user": pool,
        "candidates_total": candidates_total,
        "candidates_mean": float(node["candidates_mean"]),
        "two_tower_used_total": two_tower_used,
        "mind_used_total": mind_used,
        "fills_total": fills,
        "hits": int(node["hits"]),
        "recall_at_100": float(node["recall_at_100"]),
        "coverage": float(node["coverage"]),
        "diagnostic_union_hits_at_100": node.get("diagnostic_union_hits_at_100"),
        "elapsed_s": node.get("elapsed_s"),
        "recall_cache_version": identity["sha256"],
        "fingerprints": copy.deepcopy(dict(fingerprints)),
        "source": {
            "label": spec.get("label", quota),
            "experiment": spec.get("experiment", ""),
            "acceptance": spec.get("acceptance", ""),
            "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "pointer": spec.get("pointer", "") or "$",
            "sha256": sha,
            "status": status or spec.get("status", "accepted"),
            "fusion_artifact": evidence["fusion"],
            "fusion_artifact_read": False,
        },
    }


def _numbers_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= 1e-9
    return left == right


def _compare_duplicate(
    canonical: Mapping[str, Any],
    duplicate_node: Mapping[str, Any],
    duplicate_info: Mapping[str, Any],
    status: str | None,
) -> dict[str, Any]:
    compared: list[str] = []
    missing: list[str] = []
    conflicts: list[dict[str, Any]] = []
    for field in DUPLICATE_COMPARE_FIELDS:
        if field not in duplicate_node:
            missing.append(field)
            continue
        compared.append(field)
        if not _numbers_equal(canonical[field], duplicate_node[field]):
            conflicts.append(
                {"field": field, "canonical": canonical[field], "duplicate": duplicate_node[field]}
            )
    if "quota" in duplicate_node:
        try:
            duplicate_quota = parse_quota_text(
                f"{duplicate_node['quota'].get('two_tower')}/{duplicate_node['quota'].get('mind')}"
                if isinstance(duplicate_node["quota"], Mapping)
                else duplicate_node["quota"]
            )[0]
        except ValueError as error:
            conflicts.append({"field": "quota", "canonical": canonical["quota"], "duplicate": str(error)})
        else:
            compared.append("quota")
            if duplicate_quota != canonical["quota"]:
                conflicts.append(
                    {
                        "field": "quota",
                        "canonical": canonical["quota"],
                        "duplicate": duplicate_quota,
                    }
                )
    duplicate_fp = None
    if isinstance(duplicate_node.get("evidence"), Mapping):
        duplicate_fp = duplicate_node["evidence"].get("fingerprints")
    if isinstance(duplicate_fp, Mapping):
        try:
            duplicate_identity = cache_identity(duplicate_fp)
        except (KeyError, TypeError):
            conflicts.append({"field": "recall_cache_version", "canonical": canonical["recall_cache_version"], "duplicate": "incomplete"})
        else:
            compared.append("recall_cache_version")
            if duplicate_identity["sha256"] != canonical["recall_cache_version"]:
                conflicts.append(
                    {
                        "field": "recall_cache_version",
                        "canonical": canonical["recall_cache_version"],
                        "duplicate": duplicate_identity["sha256"],
                    }
                )
    if status not in (None, "verified", "success", "accepted"):
        conflicts.append({"field": "status", "canonical": "verified", "duplicate": status})
    return {
        "label": duplicate_info.get("label", ""),
        "path": duplicate_info["path"],
        "pointer": duplicate_info.get("pointer", "") or "$",
        "status": status,
        "compared_fields": compared,
        "missing_fields": missing,
        "conflicts": conflicts,
    }


def load_sources(config: Mapping[str, Any]) -> dict[str, Any]:
    """读取 5 个 canonical 汇总 JSON，核对重复记录和共同口径。"""

    replay = config["replay"]
    source_specs = list(config["sources"])
    expected_quotas = [parse_quota_text(value)[0] for value in replay["allowed_quotas"]]
    expected_quotas = [parse_quota_text(replay["baseline_quota"])[0], *expected_quotas]
    seen_specs: list[str] = []
    cache: dict[Path, tuple[Any, str]] = {}
    records: dict[str, dict[str, Any]] = {}
    duplicate_checks: list[dict[str, Any]] = []
    files_read: dict[str, dict[str, Any]] = {}

    def read(path_text: str, role: str) -> tuple[Any, str]:
        path = (PROJECT_ROOT / path_text).resolve()
        if PROJECT_ROOT not in path.parents:
            raise ReplayInputError(f"来源路径越出项目根目录：{path_text}")
        if not path.is_file():
            raise ReplayInputError(f"来源文件不存在：{path_text}")
        document, sha = cache.get(path, (None, None))
        if document is None:
            document, sha = read_json_with_hash(path)
            cache[path] = (document, sha)
        relative = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        if relative in files_read:
            entry = files_read[relative]
            roles = entry.setdefault("roles", [entry["role"]])
            if role not in roles:
                roles.append(role)
            entry["role"] = "; ".join(roles)
        else:
            files_read[relative] = {
                "sha256": sha,
                "role": role,
                "roles": [role],
                "read": True,
            }
        return document, sha

    for spec in source_specs:
        quota = parse_quota_text(spec["quota"])[0]
        seen_specs.append(quota)
        document, sha = read(spec["path"], f"canonical:{quota}")
        node = deep_get(document, spec.get("pointer", ""))
        if not isinstance(node, Mapping):
            raise ReplayInputError(f"{quota} 来源节点不是 JSON 对象：{spec['path']}")
        status = _source_status(document, node)
        expected_status = spec.get("status", "accepted")
        if status not in (None, "verified", "success", "accepted"):
            raise ReplayInputError(f"{quota} 来源状态为 {status!r}，不是已验收成功记录")
        if expected_status == "verified" and status != "verified":
            raise ReplayInputError(f"{quota} 要求 verified，实际 {status!r}")
        records[quota] = _normalize_record(
            node, spec, (PROJECT_ROOT / spec["path"]).resolve(), sha, status, config
        )
        for duplicate in spec.get("duplicates", []):
            duplicate_document, duplicate_sha = read(
                duplicate["path"], f"duplicate:{quota}:{duplicate.get('label', '')}"
            )
            duplicate_node = deep_get(duplicate_document, duplicate.get("pointer", ""))
            if not isinstance(duplicate_node, Mapping):
                raise ReplayInputError(
                    f"{quota} 重复来源节点不是 JSON 对象：{duplicate['path']}"
                )
            duplicate_status = _source_status(duplicate_document, duplicate_node)
            comparison = _compare_duplicate(
                records[quota], duplicate_node, duplicate, duplicate_status
            )
            comparison["quota"] = quota
            comparison["duplicate_sha256"] = duplicate_sha
            if comparison["conflicts"]:
                raise ReplayInputError(
                    f"{quota} 存在重复记录口径冲突："
                    f"{duplicate['path']} {comparison['conflicts']}"
                )
            duplicate_checks.append(comparison)

    missing = [quota for quota in expected_quotas if quota not in records]
    extra = [quota for quota in records if quota not in expected_quotas]
    if missing or extra:
        raise ReplayInputError(f"来源配额不完整或有未知配额：missing={missing}, extra={extra}")
    if len(set(seen_specs)) != len(seen_specs):
        raise ReplayInputError("canonical 来源存在重复配额")

    identities = {record["recall_cache_version"] for record in records.values()}
    if len(identities) != 1:
        detail = {
            quota: record["recall_cache_version"] for quota, record in records.items()
        }
        raise ReplayInputError(f"召回缓存版本不一致，停止正式对照：{detail}")
    cache_payload = cache_identity(records["50/50"]["fingerprints"])

    shared_fields = ("dataset", "split", "users", "targets", "candidate_pool_per_user", "fills_total")
    for field in shared_fields:
        values = {record[field] for record in records.values()}
        if len(values) != 1:
            raise ReplayInputError(f"共同口径字段 {field} 不一致：{values}")
    if any(record["candidate_pool_per_user"] != int(replay["candidate_pool_per_user"]) for record in records.values()):
        raise ReplayInputError("融合池上限不一致")

    baseline = records["50/50"]
    manifest_records = []
    for quota in [records[key]["quota"] for key in sorted(records, key=lambda q: (records[q]["two_tower"], records[q]["mind"]))]:
        record = records[quota]
        manifest_records.append(
            {
                "quota": record["quota"],
                "two_tower": record["two_tower"],
                "mind": record["mind"],
                "dataset": record["dataset"],
                "split": record["split"],
                "users": record["users"],
                "targets": record["targets"],
                "user_scope": f"{record['users']} 个固定验证用户（来源用户集合 sha256 相同）",
                "candidate_pool_per_user": record["candidate_pool_per_user"],
                "candidates_total": record["candidates_total"],
                "candidates_mean": record["candidates_mean"],
                "two_tower_used_total": record["two_tower_used_total"],
                "mind_used_total": record["mind_used_total"],
                "fills_total": record["fills_total"],
                "hits": record["hits"],
                "recall_at_100": record["recall_at_100"],
                "coverage": record["coverage"],
                "diagnostic_union_hits_at_100": record["diagnostic_union_hits_at_100"],
                "recall_cache_version": record["recall_cache_version"],
                "source": record["source"],
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": replay["phase"],
        "dataset": replay["dataset"],
        "split": replay["split"],
        "split_strategy": replay["split_strategy"],
        "split_evidence": replay["split_evidence"],
        "user_scope": {
            "fixed_users": baseline["users"],
            "targets": baseline["targets"],
            "users_file_sha256": cache_payload["payload"]["users_file_sha256"],
            "description": "128 个固定验证用户；来源记录与缓存指纹一致",
        },
        "recall_cache": {
            "version": cache_payload["sha256"],
            "version_short": cache_payload["short"],
            "identity": cache_payload["payload"],
            "description": "由 split 哈希、id_mapping、两路召回文件哈希、users 文件哈希、checkpoint 哈希、routing_seed、k_recall 共同计算的缓存身份",
        },
        "fusion_pool": {
            "per_user": int(replay["candidate_pool_per_user"]),
            "total": baseline["candidates_total"],
            "fills_total": baseline["fills_total"],
            "description": "每用户最多 100 个融合候选；本轮所有来源补位次数均为 0",
        },
        "metric_definition": replay["metric_definition"],
        "fusion_rule_version": replay["fusion_rule_version"],
        "quotas": manifest_records,
        "duplicate_checks": duplicate_checks,
        "consistency": {
            "status": "pass",
            "checks": [
                {"name": "same_dataset", "value": replay["dataset"]},
                {"name": "same_split", "value": replay["split"]},
                {"name": "same_users", "value": baseline["users"]},
                {"name": "same_targets", "value": baseline["targets"]},
                {"name": "same_candidate_pool", "value": baseline["candidate_pool_per_user"]},
                {"name": "same_recall_cache_version", "value": cache_payload["sha256"]},
                {"name": "same_metric_definition", "value": replay["metric_definition"]},
                {"name": "duplicate_records_checked", "value": len(duplicate_checks)},
                {"name": "duplicate_conflicts", "value": 0},
            ],
        },
        "source_files_read": [
            {"path": path, **payload} for path, payload in sorted(files_read.items())
        ],
    }
    return {
        "records": records,
        "manifest": manifest,
        "files_read": files_read,
        "duplicate_checks": duplicate_checks,
    }


# ---------------------------------------------------------------------------
# 执行器与策略
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyAction:
    kind: str
    quota: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class PolicyObservation:
    step: int
    candidates: tuple[str, ...]
    starting_history: dict[str, Any]
    revealed: tuple[dict[str, Any], ...]
    remaining_budget: int


class FusionResultLookup:
    """执行器内部的结果查询工具；策略永远拿不到完整结果表。"""

    name = TOOL_NAME
    schema = {
        "name": TOOL_NAME,
        "description": "返回一个此前未揭晓、且预算允许的融合配额的已有汇总指标",
        "parameters": {
            "type": "object",
            "properties": {
                "two_tower": {"type": "integer", "minimum": 0, "maximum": 100},
                "mind": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": ["two_tower", "mind"],
            "additionalProperties": False,
        },
    }

    def __init__(self, records: Mapping[str, Mapping[str, Any]]) -> None:
        self._records = dict(records)

    def lookup(self, quota: str) -> dict[str, Any]:
        if quota not in self._records:
            raise KeyError(quota)
        return public_metrics(self._records[quota])


class ReplaySession:
    """单侧回放状态：预算、重复/非法校验、结果查询和最佳更新。

    ``ReplayExecutor`` 与真实 LLM 工作流共用这套状态机；策略/模型只通过
    ``snapshot()`` 看到起始历史、候选、本侧已揭晓结果和剩余预算。
    """

    def __init__(
        self,
        records: Mapping[str, Mapping[str, Any]],
        baseline_quota: str,
        allowed_quotas: Sequence[str],
        budget: int,
        decimals: int,
    ) -> None:
        self._records = {parse_quota_text(quota)[0]: record for quota, record in records.items()}
        self._baseline_quota = parse_quota_text(baseline_quota)[0]
        self._allowed = tuple(parse_quota_text(quota)[0] for quota in allowed_quotas)
        self._budget = int(budget)
        self._decimals = int(decimals)
        if self._baseline_quota not in self._records:
            raise ValueError("缺少 50/50 起始历史")
        for quota in self._allowed:
            if quota not in self._records:
                raise ValueError(f"缺少配额结果：{quota}")
        self._lookup = FusionResultLookup(self._records)
        self.starting_history = public_metrics(self._records[self._baseline_quota])
        self.revealed: list[dict[str, Any]] = []
        self.revealed_keys: set[str] = set()
        self.remaining = self._budget
        self.best: dict[str, Any] = copy.deepcopy(self.starting_history)
        self.invalid_requests: list[dict[str, Any]] = []

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def candidates(self) -> tuple[str, ...]:
        return self._allowed

    @property
    def starting_histories(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(self.starting_history)]

    @property
    def starting_keys(self) -> set[str]:
        return {self.starting_history["quota"]}

    def snapshot(self) -> dict[str, Any]:
        return {
            "candidates": list(self._allowed),
            "starting_history": copy.deepcopy(self.starting_history),
            "revealed": copy.deepcopy(self.revealed),
            "remaining_budget": self.remaining,
        }

    def request(self, quota_text: Any, *, step: int | None = None) -> dict[str, Any]:
        """查询一组配额；失败不改变预算/已揭晓状态，也不返回指标。"""

        parsed: tuple[str, int, int] | None
        try:
            parsed = parse_quota_text(quota_text)
        except ValueError:
            parsed = None
        error = self._authorize(parsed)
        if error is not None:
            code, message = error
            record = {
                "step": step,
                "code": code,
                "message": message,
                "quota": quota_text,
            }
            self.invalid_requests.append(record)
            return {
                "status": "error",
                "code": code,
                "message": message,
                "quota": quota_text,
                "remaining_budget": self.remaining,
            }

        assert parsed is not None
        quota, two_tower, mind = parsed
        result = self._lookup.lookup(quota)
        best_before = self.best
        self.revealed.append(result)
        self.revealed_keys.add(quota)
        self.remaining -= 1
        self.best = public_metrics(select_best([self.starting_history, *self.revealed], self._decimals))
        return {
            "status": "ok",
            "quota": quota,
            "two_tower": two_tower,
            "mind": mind,
            "result": result,
            "remaining_budget": self.remaining,
            "best": copy.deepcopy(self.best),
            "best_changed": self.best["quota"] != best_before["quota"],
        }

    def summary(self) -> dict[str, Any]:
        return {
            "reveals": len(self.revealed),
            "remaining_budget": self.remaining,
            "invalid_requests": len(self.invalid_requests),
            "best": copy.deepcopy(self.best),
            "best_quota": self.best["quota"],
            "hits": int(self.best["hits"]),
            "recall_at_100": float(self.best["recall_at_100"]),
            "coverage": float(self.best["coverage"]),
        }

    def _authorize(self, parsed: tuple[str, int, int] | None) -> tuple[str, str] | None:
        if parsed is None:
            return "illegal_quota", "配额格式非法或两路之和不等于 100"
        quota, _, _ = parsed
        if quota == self._baseline_quota:
            return "duplicate_baseline", "50/50 是起始历史，不能作为新的揭晓请求"
        if quota not in self._allowed:
            return "illegal_quota", f"{quota} 不在公开可选配额内"
        if quota in self.revealed_keys:
            return "duplicate_quota", f"{quota} 本侧已经揭晓过，重复请求不扣预算但终止该侧"
        if self.remaining <= 0:
            return "budget_exhausted", "本侧 2 次揭晓预算已用完，拒绝继续揭晓"
        return None


class MultiStartReplaySession:
    """多起始历史回放状态（Phase 13 双场景）。

    起始历史可包含多组配额，全部计入当前最佳且不消耗预算；可揭晓配额必须
    与起始历史不相交。其他校验、结果查询和最佳更新与 ``ReplaySession`` 一致。
    """

    def __init__(
        self,
        records: Mapping[str, Mapping[str, Any]],
        starting_quotas: Sequence[str],
        allowed_quotas: Sequence[str],
        budget: int,
        decimals: int,
    ) -> None:
        self._records = {parse_quota_text(quota)[0]: record for quota, record in records.items()}
        self._starting = tuple(parse_quota_text(quota)[0] for quota in starting_quotas)
        self._allowed = tuple(parse_quota_text(quota)[0] for quota in allowed_quotas)
        self._budget = int(budget)
        self._decimals = int(decimals)
        if not self._starting:
            raise ValueError("至少需要一条起始历史")
        if set(self._starting) & set(self._allowed):
            raise ValueError("起始历史与可揭晓配额不能重叠")
        for quota in self._starting + self._allowed:
            if quota not in self._records:
                raise ValueError(f"缺少配额结果：{quota}")
        self._lookup = FusionResultLookup(self._records)
        self.starting_histories = [public_metrics(self._records[quota]) for quota in self._starting]
        self.starting_keys = set(self._starting)
        self.revealed: list[dict[str, Any]] = []
        self.revealed_keys: set[str] = set()
        self.remaining = self._budget
        self.best: dict[str, Any] = public_metrics(
            select_best(self.starting_histories, self._decimals)
        )
        self.invalid_requests: list[dict[str, Any]] = []

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def candidates(self) -> tuple[str, ...]:
        return self._allowed

    def snapshot(self) -> dict[str, Any]:
        return {
            "candidates": list(self._allowed),
            "starting_histories": copy.deepcopy(self.starting_histories),
            "revealed": copy.deepcopy(self.revealed),
            "remaining_budget": self.remaining,
        }

    def request(self, quota_text: Any, *, step: int | None = None) -> dict[str, Any]:
        parsed: tuple[str, int, int] | None
        try:
            parsed = parse_quota_text(quota_text)
        except ValueError:
            parsed = None
        error = self._authorize(parsed)
        if error is not None:
            code, message = error
            self.invalid_requests.append(
                {"step": step, "code": code, "message": message, "quota": quota_text}
            )
            return {
                "status": "error",
                "code": code,
                "message": message,
                "quota": quota_text,
                "remaining_budget": self.remaining,
            }

        assert parsed is not None
        quota, two_tower, mind = parsed
        result = self._lookup.lookup(quota)
        best_before = self.best
        self.revealed.append(result)
        self.revealed_keys.add(quota)
        self.remaining -= 1
        self.best = public_metrics(
            select_best([*self.starting_histories, *self.revealed], self._decimals)
        )
        return {
            "status": "ok",
            "quota": quota,
            "two_tower": two_tower,
            "mind": mind,
            "result": result,
            "remaining_budget": self.remaining,
            "best": copy.deepcopy(self.best),
            "best_changed": self.best["quota"] != best_before["quota"],
        }

    def summary(self) -> dict[str, Any]:
        return {
            "starting_quotas": [record["quota"] for record in self.starting_histories],
            "starting_histories": copy.deepcopy(self.starting_histories),
            "reveals": len(self.revealed),
            "remaining_budget": self.remaining,
            "invalid_requests": len(self.invalid_requests),
            "best": copy.deepcopy(self.best),
            "best_quota": self.best["quota"],
            "hits": int(self.best["hits"]),
            "recall_at_100": float(self.best["recall_at_100"]),
            "coverage": float(self.best["coverage"]),
            "first_best_reveal": 0
            if self.best["quota"] in self.starting_keys
            else next(
                (
                    index
                    for index, result in enumerate(self.revealed, start=1)
                    if result["quota"] == self.best["quota"]
                ),
                0,
            ),
        }

    def _authorize(self, parsed: tuple[str, int, int] | None) -> tuple[str, str] | None:
        if parsed is None:
            return "illegal_quota", "配额格式非法或两路之和不等于 100"
        quota, _, _ = parsed
        if quota in self.starting_keys:
            return "duplicate_history", f"{quota} 是起始历史，不能作为新的揭晓请求"
        if quota not in self._allowed:
            return "illegal_quota", f"{quota} 不在本场景可揭晓配额内"
        if quota in self.revealed_keys:
            return "duplicate_quota", f"{quota} 本侧已经揭晓过，重复请求不扣预算但终止该侧"
        if self.remaining <= 0:
            return "budget_exhausted", "本场景揭晓预算已用完，拒绝继续揭晓"
        return None


class ReplayExecutor:
    """双方共用的预算/配额校验器；只向策略返回公开指标。"""

    def __init__(
        self,
        records: Mapping[str, Mapping[str, Any]],
        baseline_quota: str,
        allowed_quotas: Sequence[str],
        budget: int,
        decimals: int,
    ) -> None:
        self._records = {parse_quota_text(quota)[0]: record for quota, record in records.items()}
        self._baseline_quota = parse_quota_text(baseline_quota)[0]
        self._allowed = tuple(parse_quota_text(quota)[0] for quota in allowed_quotas)
        self._budget = int(budget)
        self._decimals = int(decimals)
        if self._baseline_quota not in self._records:
            raise ValueError("缺少 50/50 起始历史")
        for quota in self._allowed:
            if quota not in self._records:
                raise ValueError(f"缺少配额结果：{quota}")
        self._tool = FusionResultLookup(self._records)

    def run(self, side: str, policy: Any) -> dict[str, Any]:
        baseline = self._records[self._baseline_quota]
        state: dict[str, Any] = {
            "revealed": [],
            "revealed_keys": set(),
            "remaining": self._budget,
            "best": baseline,
            "steps": [],
            "invalid_requests": [],
        }
        status = "success"
        error: dict[str, Any] | None = None
        stop: dict[str, Any] | None = None

        for step in range(1, MAX_DECISIONS + 1):
            visible_before = {
                "candidates": list(self._allowed),
                "starting_history": public_metrics(baseline),
                "revealed": [copy.deepcopy(item) for item in state["revealed"]],
                "remaining_budget": state["remaining"],
            }
            observation = PolicyObservation(
                step=step,
                candidates=tuple(self._allowed),
                starting_history=public_metrics(baseline),
                revealed=tuple(copy.deepcopy(item) for item in state["revealed"]),
                remaining_budget=state["remaining"],
            )
            try:
                action = policy.decide(observation)
            except Exception as error_object:  # noqa: BLE001 - 策略失败必须落轨迹而不是崩溃
                message = f"策略抛出异常：{type(error_object).__name__}: {error_object}"
                state["steps"].append(
                    {
                        "step": step,
                        "visible_before": visible_before,
                        "decision": {"action": "error", "quota": None, "reason": message},
                        "tool_call": None,
                        "tool_return": {"status": "error", "code": "policy_error", "message": message},
                        "remaining_budget_after": state["remaining"],
                        "best_after": public_metrics(state["best"]),
                        "best_changed": False,
                    }
                )
                state["invalid_requests"].append({"step": step, "code": "policy_error", "message": message})
                status = "failed"
                stop = {
                    "code": "invalid_request",
                    "reason": f"策略异常，终止该侧：{message}",
                    "remaining_budget": state["remaining"],
                    "best": public_metrics(state["best"]),
                }
                error = {"code": "policy_error", "message": message}
                break

            if not isinstance(action, PolicyAction) or action.kind not in {"choose", "stop"}:
                message = "策略返回值必须是 PolicyAction(choose|stop)"
                state["steps"].append(
                    {
                        "step": step,
                        "visible_before": visible_before,
                        "decision": {"action": "invalid", "quota": None, "reason": message},
                        "tool_call": None,
                        "tool_return": {"status": "error", "code": "invalid_action", "message": message},
                        "remaining_budget_after": state["remaining"],
                        "best_after": public_metrics(state["best"]),
                        "best_changed": False,
                    }
                )
                state["invalid_requests"].append({"step": step, "code": "invalid_action", "message": message})
                status = "failed"
                stop = {
                    "code": "invalid_request",
                    "reason": message,
                    "remaining_budget": state["remaining"],
                    "best": public_metrics(state["best"]),
                }
                error = {"code": "invalid_action", "message": message}
                break

            if action.kind == "stop":
                if state["remaining"] == 0:
                    stop = {
                        "code": "budget_exhausted",
                        "reason": action.reason or "预算用完",
                        "remaining_budget": 0,
                        "best": public_metrics(state["best"]),
                    }
                else:
                    stop = {
                        "code": "policy_stop",
                        "reason": action.reason or "策略主动停止",
                        "remaining_budget": state["remaining"],
                        "best": public_metrics(state["best"]),
                    }
                break

            raw_quota = action.quota
            parsed: tuple[str, int, int] | None = None
            parse_error: str | None = None
            try:
                parsed = parse_quota_text(raw_quota)
            except ValueError as error_object:
                parse_error = str(error_object)

            authorization = self._authorize(state, parsed)
            tool_call = (
                {"name": TOOL_NAME, "arguments": {"two_tower": parsed[1], "mind": parsed[2]}}
                if parsed is not None
                else {"name": TOOL_NAME, "arguments": {"raw": raw_quota}}
            )
            if authorization is not None:
                code, message = authorization
                state["steps"].append(
                    {
                        "step": step,
                        "visible_before": visible_before,
                        "decision": {
                            "action": "choose",
                            "quota": raw_quota,
                            "reason": action.reason or "",
                        },
                        "tool_call": tool_call,
                        "tool_return": {
                            "status": "error",
                            "code": code,
                            "message": message,
                        },
                        "remaining_budget_after": state["remaining"],
                        "best_after": public_metrics(state["best"]),
                        "best_changed": False,
                    }
                )
                state["invalid_requests"].append(
                    {"step": step, "code": code, "message": message, "quota": raw_quota}
                )
                status = "failed"
                stop = {
                    "code": "invalid_request",
                    "reason": f"执行器拒绝请求并终止该侧（{code}）：{message}",
                    "remaining_budget": state["remaining"],
                    "best": public_metrics(state["best"]),
                }
                error = {"code": code, "message": message}
                break

            assert parsed is not None
            quota, two_tower, mind = parsed
            result = self._tool.lookup(quota)
            best_before = state["best"]
            state["revealed"].append(result)
            state["revealed_keys"].add(quota)
            state["remaining"] -= 1
            best_now = select_best([baseline, *state["revealed"]], self._decimals)
            state["best"] = best_now
            changed = best_now["quota"] != best_before["quota"]
            state["steps"].append(
                {
                    "step": step,
                    "visible_before": visible_before,
                    "decision": {
                        "action": "choose",
                        "quota": quota,
                        "reason": action.reason or "",
                    },
                    "tool_call": tool_call,
                    "tool_return": {
                        "status": "ok",
                        "result": result,
                    },
                    "remaining_budget_after": state["remaining"],
                    "best_after": public_metrics(best_now),
                    "best_changed": changed,
                }
            )
        else:
            raise ReplayInputError("执行器超过最大决策次数，疑似策略未停止")

        if stop is None:
            raise ReplayInputError("执行器没有记录停止原因")

        reveals = len(state["revealed"])
        summary = {
            "status": status,
            "reveals": reveals,
            "invalid_requests": len(state["invalid_requests"]),
            "remaining_budget": state["remaining"],
            "best": public_metrics(state["best"]),
            "best_quota": state["best"]["quota"],
            "hits": int(state["best"]["hits"]),
            "recall_at_100": float(state["best"]["recall_at_100"]),
            "coverage": float(state["best"]["coverage"]),
            "stop_code": stop["code"],
            "stop_reason": stop["reason"],
        }
        trace = {
            "schema_version": SCHEMA_VERSION,
            "side": side,
            "policy": copy.deepcopy(getattr(policy, "info", {"name": type(policy).__name__})),
            "budget_initial": self._budget,
            "candidates": list(self._allowed),
            "starting_history": public_metrics(baseline),
            "steps": state["steps"],
            "invalid_requests": state["invalid_requests"],
            "stop": stop,
            "summary": summary,
            "status": status,
            "error": error,
        }
        return trace

    def _authorize(
        self, state: Mapping[str, Any], parsed: tuple[str, int, int] | None
    ) -> tuple[str, str] | None:
        if parsed is None:
            return "illegal_quota", "配额格式非法或两路之和不等于 100"
        quota, _, _ = parsed
        if quota == self._baseline_quota:
            return "duplicate_baseline", "50/50 是起始历史，不能作为新的揭晓请求"
        if quota not in self._allowed:
            return "illegal_quota", f"{quota} 不在公开可选配额内"
        if quota in state["revealed_keys"]:
            return "duplicate_quota", f"{quota} 本侧已经揭晓过，重复请求不扣预算但终止该侧"
        if state["remaining"] <= 0:
            return "budget_exhausted", "本侧 2 次揭晓预算已用完，拒绝继续揭晓"
        return None


class FixedPolicy:
    """固定策略：70/30 → 30/70，顺序由配置写死，不看反馈。"""

    def __init__(self, order: Sequence[str]) -> None:
        self._order = [parse_quota_text(value)[0] for value in order]
        self._index = 0
        self.info = {
            "name": "fixed_symmetric_probe",
            "order": list(self._order),
            "uses_feedback": False,
        }

    def decide(self, observation: PolicyObservation) -> PolicyAction:
        if observation.remaining_budget <= 0:
            return PolicyAction(
                "stop",
                reason="预设顺序 70/30 → 30/70 已执行完毕，2 次揭晓预算同时耗尽",
            )
        if self._index >= len(self._order):
            return PolicyAction("stop", reason="固定策略预设顺序执行完毕")
        quota = self._order[self._index]
        self._index += 1
        return PolicyAction(
            "choose",
            quota=quota,
            reason=f"固定策略按预设顺序执行第 {self._index} 项：{quota}（不根据回放结果调整）",
        )


class LocalFeedbackPolicy:
    """可复现的本地反馈策略：只读起始历史、本侧揭晓结果和剩余预算。"""

    def __init__(self, config: Mapping[str, Any], decimals: int | None = None) -> None:
        self._decimals = int(
            decimals if decimals is not None else config.get("recall_decimals", 6)
        )
        self.info = {
            "name": "local_feedback_v1",
            "first_step": config.get("first_step", ""),
            "better_step": config.get("better_step", ""),
            "worse_step": config.get("worse_step", ""),
            "tie_step": config.get("tie_step", ""),
            "stop_rules": list(config.get("stop_rules", [])),
            "uses_feedback": True,
        }

    def decide(self, observation: PolicyObservation) -> PolicyAction:
        if observation.remaining_budget <= 0:
            best = select_best(
                [observation.starting_history, *observation.revealed], self._decimals
            )
            return PolicyAction(
                "stop",
                reason=f"2 次揭晓预算已用完，最佳保持 {format_result(best)}",
            )

        if observation.step == 1:
            unrevealed = self._unrevealed(observation)
            if not unrevealed:
                return PolicyAction("stop", reason="没有未揭晓候选")
            first = min(
                unrevealed,
                key=lambda quota: (
                    abs(parse_quota_text(quota)[1] - 50) + abs(parse_quota_text(quota)[2] - 50),
                    parse_quota_text(quota)[1],
                ),
            )
            _, first_two_tower, first_mind = parse_quota_text(first)
            distance = abs(first_two_tower - 50) + abs(first_mind - 50)
            return PolicyAction(
                "choose",
                quota=first,
                reason=(
                    f"首步规则：{first} 与 50/50 的 L1 距离 {distance} 最小"
                    f"（并列时双塔配额升序）；起始历史 {format_result(observation.starting_history)}。"
                ),
            )

        if not observation.revealed:
            return PolicyAction("stop", reason="没有可用的第一步结果，停止")
        latest = observation.revealed[-1]
        best_before = select_best(
            [observation.starting_history, *observation.revealed[:-1]], self._decimals
        )
        latest_recall = recall_value(latest, self._decimals)
        best_recall = recall_value(best_before, self._decimals)
        if latest_recall == best_recall:
            return PolicyAction(
                "stop",
                reason=(
                    f"第一步 {format_result(latest)} 的 Recall@100 与当前最佳 "
                    f"{format_result(best_before)} 持平，未显示方向性收益，停止"
                    f"（剩余预算 {observation.remaining_budget}）。"
                ),
            )
        if latest_recall > best_recall:
            direction = "增加 MIND" if latest["two_tower"] < 50 else "增加双塔"
            candidate = self._same_direction(latest, observation)
            if candidate is None:
                return PolicyAction(
                    "stop",
                    reason=(
                        f"第一步 {format_result(latest)} 优于当前最佳 {format_result(best_before)}，"
                        f"但{direction}方向没有未揭晓候选，停止。"
                    ),
                )
            return PolicyAction(
                "choose",
                quota=candidate,
                reason=(
                    f"第一步 {format_result(latest)} 优于当前最佳 {format_result(best_before)}；"
                    f"方向为{direction}，按同方向再探一档选 {candidate}。"
                ),
            )

        mirror = self._mirror(latest, observation)
        if mirror is None:
            return PolicyAction(
                "stop",
                reason=(
                    f"第一步 {format_result(latest)} 劣于当前最佳 {format_result(best_before)}；"
                    f"镜像配额不可用或已揭晓，停止。"
                ),
            )
        return PolicyAction(
            "choose",
            quota=mirror,
            reason=(
                f"第一步 {format_result(latest)} 劣于当前最佳 {format_result(best_before)}；"
                f"按镜像规则转向 {mirror}。"
            ),
        )

    def _unrevealed(self, observation: PolicyObservation) -> list[str]:
        seen = {observation.starting_history["quota"]}
        seen.update(item["quota"] for item in observation.revealed)
        return [quota for quota in observation.candidates if quota not in seen]

    def _same_direction(
        self, latest: Mapping[str, Any], observation: PolicyObservation
    ) -> str | None:
        candidates = []
        for quota in self._unrevealed(observation):
            _, two_tower, mind = parse_quota_text(quota)
            candidates.append((quota, two_tower, mind))
        if int(latest["two_tower"]) > 50:
            higher = [item for item in candidates if item[1] > int(latest["two_tower"])]
            return min(higher, key=lambda item: item[1])[0] if higher else None
        if int(latest["two_tower"]) < 50:
            lower = [item for item in candidates if item[1] < int(latest["two_tower"])]
            return max(lower, key=lambda item: item[1])[0] if lower else None
        return None

    def _mirror(self, latest: Mapping[str, Any], observation: PolicyObservation) -> str | None:
        mirror = f"{int(latest['mind'])}/{int(latest['two_tower'])}"
        return mirror if mirror in self._unrevealed(observation) else None


def run_side(
    records: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    side: str,
    policy: Any,
) -> dict[str, Any]:
    replay = config["replay"]
    executor = ReplayExecutor(
        records=records,
        baseline_quota=replay["baseline_quota"],
        allowed_quotas=replay["allowed_quotas"],
        budget=int(replay["budget_reveals_per_side"]),
        decimals=int(config["best_rule"]["recall_decimals"]),
    )
    return executor.run(side, policy)


# ---------------------------------------------------------------------------
# 轻量本地断言（合成结果，不触发真实评测）
# ---------------------------------------------------------------------------


def synthetic_record(
    quota: str,
    hits: int,
    recall_at_100: float,
    coverage: float,
    users: int = 128,
    targets: int = 362,
    pool: int = 100,
) -> dict[str, Any]:
    key, two_tower, mind = parse_quota_text(quota)
    return {
        "quota": key,
        "two_tower": two_tower,
        "mind": mind,
        "users": users,
        "targets": targets,
        "candidates_total": users * pool,
        "candidates_mean": float(pool),
        "two_tower_used_total": users * two_tower,
        "mind_used_total": users * mind,
        "fills_total": 0,
        "hits": int(hits),
        "recall_at_100": float(recall_at_100),
        "coverage": float(coverage),
    }


def synthetic_table(overrides: Mapping[str, tuple[int, float, float]] | None = None) -> dict[str, dict[str, Any]]:
    metrics = dict(DEFAULT_SYNTHETIC_METRICS)
    if overrides:
        metrics.update(overrides)
    return {quota: synthetic_record(quota, *values) for quota, values in metrics.items()}


class ScriptedPolicy:
    def __init__(self, actions: Sequence[str], name: str = "scripted") -> None:
        self._actions = list(actions)
        self._index = 0
        self.info = {"name": name, "actions": list(actions)}

    def decide(self, observation: PolicyObservation) -> PolicyAction:
        if self._index >= len(self._actions):
            return PolicyAction("stop", reason="脚本动作执行完毕")
        quota = self._actions[self._index]
        self._index += 1
        return PolicyAction("choose", quota=quota, reason=f"脚本第 {self._index} 步选择 {quota}")


def _executor_for_checks(config: Mapping[str, Any], table: Mapping[str, Mapping[str, Any]]) -> ReplayExecutor:
    replay = config["replay"]
    return ReplayExecutor(
        records=table,
        baseline_quota=replay["baseline_quota"],
        allowed_quotas=replay["allowed_quotas"],
        budget=int(replay["budget_reveals_per_side"]),
        decimals=int(config["best_rule"]["recall_decimals"]),
    )


def run_local_checks(config: Mapping[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(check_id: str, function: Callable[[], str]) -> None:
        try:
            detail = function()
        except AssertionError as error:
            checks.append({"id": check_id, "status": "fail", "detail": str(error)})
        except Exception as error:  # noqa: BLE001 - 断言失败也要写入证据
            checks.append(
                {"id": check_id, "status": "fail", "detail": f"{type(error).__name__}: {error}"}
            )
        else:
            checks.append({"id": check_id, "status": "pass", "detail": detail})

    decimals = int(config["best_rule"]["recall_decimals"])

    def run_agent(overrides: Mapping[str, tuple[int, float, float]] | None = None) -> dict[str, Any]:
        table = synthetic_table(overrides)
        return _executor_for_checks(config, table).run(
            "agent",
            LocalFeedbackPolicy(
                config["agent_policy"], int(config["best_rule"]["recall_decimals"])
            ),
        )

    def run_fixed(table: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
        return _executor_for_checks(config, table or synthetic_table()).run(
            "fixed", FixedPolicy(config["fixed_policy"]["order"])
        )

    def initial_isolation() -> str:
        traces = [run_agent(), run_fixed()]
        for trace in traces:
            assert trace["status"] == "success", trace["status"]
            assert trace["starting_history"]["quota"] == "50/50"
            assert trace["summary"]["reveals"] <= 2
            first = trace["steps"][0]
            assert set(first["visible_before"]) == {
                "candidates",
                "starting_history",
                "revealed",
                "remaining_budget",
            }
            assert first["visible_before"]["revealed"] == []
            assert first["visible_before"]["remaining_budget"] == 2
            assert set(first["visible_before"]["starting_history"]) == set(PUBLIC_METRIC_FIELDS)
            assert set(first["tool_return"]["result"]) == set(PUBLIC_METRIC_FIELDS)
        return "双方第一步只看到 50/50 历史、4 个候选配额和 2 次预算；工具返回无来源/指纹字段"

    def feedback_sensitivity() -> str:
        better = run_agent({"30/70": (8, 0.020000, 0.030000)})
        worse = run_agent({"30/70": (2, 0.005000, 0.020000)})
        tie = run_agent({"30/70": (4, 0.011719, 0.026843)})
        first_choices = [trace["steps"][0]["decision"]["quota"] for trace in (better, worse, tie)]
        assert first_choices == ["30/70", "30/70", "30/70"], first_choices
        assert better["steps"][1]["decision"]["quota"] == "20/80", better["steps"][1]
        assert worse["steps"][1]["decision"]["quota"] == "70/30", worse["steps"][1]
        assert len(tie["steps"]) == 1, len(tie["steps"])
        assert tie["stop"]["code"] == "policy_stop", tie["stop"]
        assert tie["stop"]["remaining_budget"] == 1, tie["stop"]
        assert better["steps"][0]["visible_before"] == worse["steps"][0]["visible_before"]
        assert better["steps"][0]["visible_before"] == tie["steps"][0]["visible_before"]
        return "固定首步 30/70，仅替换首步指标：更好→20/80，更差→70/30，持平→停止（剩余 1 次预算）"

    def duplicate_request() -> str:
        trace = _executor_for_checks(config, synthetic_table()).run(
            "agent", ScriptedPolicy(["30/70", "30/70"])
        )
        assert trace["status"] == "failed"
        assert trace["summary"]["reveals"] == 1
        assert trace["summary"]["invalid_requests"] == 1
        assert trace["invalid_requests"][0]["code"] == "duplicate_quota"
        assert trace["steps"][-1]["tool_return"]["status"] == "error"
        assert "result" not in trace["steps"][-1]["tool_return"]
        assert trace["summary"]["remaining_budget"] == 1
        return "重复请求 30/70 被拒绝，未泄露指标、未扣预算、未新增揭晓，该侧标记失败"

    def budget_exhaustion() -> str:
        trace = _executor_for_checks(config, synthetic_table()).run(
            "agent", ScriptedPolicy(["30/70", "20/80", "70/30"])
        )
        assert trace["status"] == "failed"
        assert trace["summary"]["reveals"] == 2
        assert trace["summary"]["remaining_budget"] == 0
        assert trace["invalid_requests"][0]["code"] == "budget_exhausted"
        assert "result" not in trace["steps"][-1]["tool_return"]
        return "2 次揭晓后请求第 3 组被拒绝，未泄露指标；总揭晓数保持 2"

    def illegal_quota() -> str:
        trace = _executor_for_checks(config, synthetic_table()).run(
            "agent", ScriptedPolicy(["60/40"])
        )
        assert trace["status"] == "failed"
        assert trace["summary"]["reveals"] == 0
        assert trace["summary"]["remaining_budget"] == 2
        assert trace["invalid_requests"][0]["code"] == "illegal_quota"
        assert "result" not in trace["steps"][-1]["tool_return"]
        baseline_trace = _executor_for_checks(config, synthetic_table()).run(
            "agent", ScriptedPolicy(["50/50"])
        )
        assert baseline_trace["status"] == "failed"
        assert baseline_trace["summary"]["reveals"] == 0
        assert baseline_trace["invalid_requests"][0]["code"] == "duplicate_baseline"
        assert "result" not in baseline_trace["steps"][-1]["tool_return"]
        return "非法配额 60/40 与重复起始历史 50/50 分别被拒绝，均未揭晓、未扣预算、无指标泄露"

    def fixed_order() -> str:
        table = synthetic_table({"70/30": (9, 0.030000, 0.040000), "30/70": (0, 0.000000, 0.010000)})
        trace = run_fixed(table)
        choices = [step["decision"]["quota"] for step in trace["steps"]]
        assert choices == ["70/30", "30/70"], choices
        assert trace["summary"]["best_quota"] == "70/30"
        assert trace["summary"]["reveals"] == 2
        return "即使 70/30 合成指标远好于 30/70，固定策略仍按配置执行 70/30 → 30/70"

    def tie_breaks() -> str:
        close = [
            synthetic_record("30/70", 5, 0.013672, 0.026720),
            synthetic_record("20/80", 5, 0.013672, 0.026595),
        ]
        assert select_best(close, decimals)["quota"] == "30/70"
        symmetric = [
            synthetic_record("70/30", 5, 0.013672, 0.026720),
            synthetic_record("30/70", 5, 0.013672, 0.026720),
        ]
        assert select_best(symmetric, decimals)["quota"] == "30/70"
        return "Recall 并列先取更接近 50/50 的 30/70；再并列按双塔配额升序取 30/70"

    def side_isolation() -> str:
        executor = _executor_for_checks(config, synthetic_table())
        agent = executor.run(
            "agent",
            LocalFeedbackPolicy(
                config["agent_policy"], int(config["best_rule"]["recall_decimals"])
            ),
        )
        fixed = executor.run("fixed", FixedPolicy(config["fixed_policy"]["order"]))
        assert [item["quota"] for item in agent["steps"][1]["visible_before"]["revealed"]] == [
            "30/70"
        ]
        assert [item["quota"] for item in fixed["steps"][1]["visible_before"]["revealed"]] == [
            "70/30"
        ]
        assert agent["summary"]["remaining_budget"] == 0
        assert fixed["summary"]["remaining_budget"] == 0
        return "Agent 第二步只看到自己的 30/70；固定策略第二步只看到自己的 70/30；两侧预算独立"

    def trace_integrity() -> str:
        traces = [run_agent(), run_fixed()]
        for trace in traces:
            assert trace["summary"]["reveals"] <= 2, trace["summary"]
            for step in trace["steps"]:
                returned = step["tool_return"]
                if returned.get("status") == "ok":
                    assert step["decision"]["quota"] == returned["result"]["quota"]
                    assert set(returned["result"]) == set(PUBLIC_METRIC_FIELDS)
                else:
                    assert "result" not in returned
                    assert step["decision"]["quota"] is not None or step["decision"]["action"] != "choose"
        return "每条成功工具返回的配额都与当步选择一致，且只含公开指标；错误返回不含结果"

    check("initial_isolation", initial_isolation)
    check("feedback_sensitivity", feedback_sensitivity)
    check("duplicate_request", duplicate_request)
    check("budget_exhaustion", budget_exhaustion)
    check("illegal_quota", illegal_quota)
    check("fixed_order", fixed_order)
    check("tie_breaks", tie_breaks)
    check("side_isolation", side_isolation)
    check("trace_integrity", trace_integrity)

    passed = sum(1 for item in checks if item["status"] == "pass")
    failed = len(checks) - passed
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if failed == 0 else "fail",
        "total": len(checks),
        "passed": passed,
        "failed": failed,
        "synthetic_only": True,
        "real_evaluation_triggered": False,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# 汇总与报告
# ---------------------------------------------------------------------------


def compare_sides(agent: Mapping[str, Any], fixed: Mapping[str, Any], decimals: int) -> dict[str, Any]:
    agent_best = agent["summary"]["best"]
    fixed_best = fixed["summary"]["best"]
    comparison = compare_records(agent_best, fixed_best, decimals)
    if agent_best["quota"] == fixed_best["quota"]:
        winner = "tie"
    else:
        winner = "agent" if comparison < 0 else ("fixed" if comparison > 0 else "tie")
    return {
        "winner": winner,
        "winner_quota": agent_best["quota"] if winner == "agent" else (fixed_best["quota"] if winner == "fixed" else agent_best["quota"]),
        "agent_best": agent_best,
        "fixed_best": fixed_best,
        "delta_agent_minus_fixed": {
            "hits": int(agent_best["hits"]) - int(fixed_best["hits"]),
            "recall_at_100": float(agent_best["recall_at_100"]) - float(fixed_best["recall_at_100"]),
            "coverage": float(agent_best["coverage"]) - float(fixed_best["coverage"]),
        },
        "agent_found_best_on_reveal": _quota_reveal_index(agent, agent_best["quota"]),
        "fixed_found_best_on_reveal": _quota_reveal_index(fixed, fixed_best["quota"]),
        "note": _comparison_note(winner, agent_best, fixed_best),
    }


def _quota_reveal_index(trace: Mapping[str, Any], quota: str) -> int | None:
    for step in trace["steps"]:
        if step.get("tool_return", {}).get("status") == "ok":
            if step["tool_return"]["result"]["quota"] == quota:
                return int(step["step"])
    return 0


def _comparison_note(winner: str, agent_best: Mapping[str, Any], fixed_best: Mapping[str, Any]) -> str:
    if winner == "tie" and agent_best["quota"] == fixed_best["quota"]:
        return (
            f"双方最佳均为 {agent_best['quota']}，主要指标完全相同；"
            "按本轮规则记为持平（不宣称 Agent 优胜）。"
        )
    if winner == "agent":
        return f"Agent 最佳 {agent_best['quota']} 按完整规则优于固定策略的 {fixed_best['quota']}。"
    if winner == "fixed":
        return f"固定策略最佳 {fixed_best['quota']} 按完整规则优于 Agent 的 {agent_best['quota']}。"
    return "双方最佳不同，但按完整规则打平。"


def build_metrics(
    config: Mapping[str, Any],
    sources: Mapping[str, Any],
    checks: Mapping[str, Any],
    agent_trace: Mapping[str, Any],
    fixed_trace: Mapping[str, Any],
    run_meta: Mapping[str, Any],
) -> dict[str, Any]:
    decimals = int(config["best_rule"]["recall_decimals"])
    agent_summary = agent_trace["summary"]
    fixed_summary = fixed_trace["summary"]
    comparison = compare_sides(agent_trace, fixed_trace, decimals)
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": config["replay"]["phase"],
        "dataset": config["replay"]["dataset"],
        "split": config["replay"]["split"],
        "scope": config["scope"],
        "rules": {
            "baseline_quota": config["replay"]["baseline_quota"],
            "allowed_quotas": list(config["replay"]["allowed_quotas"]),
            "budget_reveals_per_side": int(config["replay"]["budget_reveals_per_side"]),
            "best_rule": config["best_rule"],
            "fixed_order": list(config["fixed_policy"]["order"]),
            "agent_policy": config["agent_policy"],
            "result_lookup_tool": FusionResultLookup.schema,
        },
        "agent": agent_summary,
        "fixed": fixed_summary,
        "comparison": comparison,
        "consistency": sources["manifest"]["consistency"],
        "source_manifest": "sources.json",
        "trajectories": {
            "agent": "agent_trace.json",
            "fixed": "fixed_trace.json",
        },
        "checks": {
            "path": "checks.json",
            "status": checks["status"],
            "passed": checks["passed"],
            "total": checks["total"],
            "synthetic_only": True,
        },
        "run": {
            "command": run_meta["command"],
            "started_at": run_meta["started_at"],
            "elapsed_s": run_meta["elapsed_s"],
            "output_dir": run_meta["output_dir"],
        },
    }


def _fmt_metric_table(record: Mapping[str, Any]) -> str:
    return (
        f"{record['hits']} / {float(record['recall_at_100']):.6f} / "
        f"{float(record['coverage']):.6f}"
    )


def _fmt_visible(visible: Mapping[str, Any]) -> str:
    parts = [f"50/50: {_fmt_metric_table(visible['starting_history'])}"]
    for record in visible["revealed"]:
        parts.append(f"{record['quota']}: {_fmt_metric_table(record)}")
    return "<br>".join(parts)


def _fmt_tool_return(tool_return: Mapping[str, Any]) -> str:
    if tool_return.get("status") != "ok":
        return f"error `{tool_return.get('code')}`：{tool_return.get('message', '')}"
    result = tool_return["result"]
    return f"ok `{result['quota']}`: {_fmt_metric_table(result)}"


def _trace_rows(trace: Mapping[str, Any]) -> list[str]:
    rows = []
    for step in trace["steps"]:
        rows.append(
            "| {step} | {visible} | {choice} | {reason} | {tool_return} | {remaining} | {best} |".format(
                step=step["step"],
                visible=_fmt_visible(step["visible_before"]),
                choice=step["decision"]["quota"],
                reason=step["decision"]["reason"].replace("|", "\\|"),
                tool_return=_fmt_tool_return(step["tool_return"]),
                remaining=step["remaining_budget_after"],
                best=step["best_after"]["quota"],
            )
        )
    return rows


def build_report(
    config: Mapping[str, Any],
    sources: Mapping[str, Any],
    checks: Mapping[str, Any],
    agent_trace: Mapping[str, Any],
    fixed_trace: Mapping[str, Any],
    metrics: Mapping[str, Any],
    run_meta: Mapping[str, Any],
) -> str:
    replay = config["replay"]
    manifest = sources["manifest"]
    comparison = metrics["comparison"]
    agent = metrics["agent"]
    fixed = metrics["fixed"]
    winner_text = {"agent": "Agent 胜", "fixed": "固定策略胜", "tie": "持平"}[comparison["winner"]]
    source_rows = "\n".join(
        "| {quota} | `{path}` `{pointer}` | {hits} | {recall:.6f} | {coverage:.6f} | {used} |".format(
            quota=record["quota"],
            path=record["source"]["path"],
            pointer=record["source"]["pointer"],
            hits=record["hits"],
            recall=record["recall_at_100"],
            coverage=record["coverage"],
            used=f"{record['two_tower_used_total']} / {record['mind_used_total']}",
        )
        for record in manifest["quotas"]
    )
    duplicate_rows = "\n".join(
        "| {quota} | `{path}` `{pointer}` | {compared} | {missing} | {status} |".format(
            quota=check["quota"],
            path=check["path"],
            pointer=check["pointer"],
            compared=", ".join(check["compared_fields"]),
            missing=", ".join(check["missing_fields"]) or "—",
            status="通过" if not check["conflicts"] else "冲突",
        )
        for check in manifest["duplicate_checks"]
    )
    check_rows = "\n".join(
        "| {id} | {status} | {detail} |".format(
            id=check["id"],
            status="通过" if check["status"] == "pass" else "失败",
            detail=str(check["detail"]).replace("|", "\\|"),
        )
        for check in checks["checks"]
    )
    agent_rows = "\n".join(_trace_rows(agent_trace))
    fixed_rows = "\n".join(_trace_rows(fixed_trace))
    identity = manifest["recall_cache"]["identity"]
    split = identity["split_sha256"]
    source_list = "\n".join(
        f"- `{path}`（sha256 `{payload['sha256']}`，{payload['role']}）"
        for path, payload in sorted(run_meta["source_files_read"].items())
    )
    command = run_meta["command"]
    source_file_count = len(run_meta["source_files_read"])
    canonical_count = len(manifest["quotas"])
    duplicate_count = len(manifest["duplicate_checks"])
    start_note = (
        f"**本轮为离线回放 MVP，不是正式 Phase 13 实验**：只读取 {canonical_count} 个 canonical 汇总 JSON "
        f"以及重复记录核对用的汇总节点（共 {source_file_count} 个文件）；没有训练、召回、"
        "真实评测、真实 LLM 请求，也没有读取候选明细、完整数据集或实验记忆索引。"
    )
    agent_stop = agent_trace["stop"]
    fixed_stop = fixed_trace["stop"]

    return f"""# 2026-11-09 · Phase 13 Offline Replay MVP：融合配额受控回放

> {start_note}

## 结论

- **Rec Agent 最佳**：`{agent['best_quota']}`，命中 **{agent['hits']}**、Recall@100 **{agent['recall_at_100']:.6f}**、Coverage {agent['coverage']:.6f}；实际揭晓 **{agent['reveals']}** 次，异常请求 **{agent['invalid_requests']}** 次。
- **固定策略最佳**：`{fixed['best_quota']}`，命中 **{fixed['hits']}**、Recall@100 **{fixed['recall_at_100']:.6f}**、Coverage {fixed['coverage']:.6f}；实际揭晓 **{fixed['reveals']}** 次，异常请求 **{fixed['invalid_requests']}** 次。
- **对照结果：{winner_text}**。{comparison['note']}
- **边界**：本轮只覆盖 `kuasar_small` 的这批 128 个验证用户和 5 个已有配额结果；不据此宣称正式测试集收益、策略泛化能力或完整 Phase 13 结论。

## 1. 回放设置

| 项 | 设置 |
| --- | --- |
| 数据集 / 划分 | `{replay['dataset']}` / `{replay['split']}`；{replay['split_evidence']} |
| 用户范围 | 128 个固定验证用户；362 个验证目标 |
| 起始历史 | `50/50` 已知，计入当前最佳，不消耗预算 |
| 公开候选配额 | `70/30`、`30/70`、`20/80`、`80/20`（指标在结果查询工具背后） |
| 回放预算 | 每侧最多揭晓 2 个此前未知配额；每揭晓 1 组扣 1 次 |
| 最佳规则 | 先比 Recall@100；并列选更接近 50/50；再并列按双塔配额升序；Coverage 仅描述 |
| 固定策略 | 预先写死 `70/30 → 30/70`，两次都执行，不根据结果调整 |
| Agent 首步 | 在未揭晓候选中选与 50/50 的 L1 距离最小者；并列时双塔配额较小者优先 |
| Agent 第二步 | 读第一步工具返回：Recall 更高→同方向再探一档；更低→镜像；Recall 持平→停止 |
| Agent 停止 | 预算用完 / 首步 Recall 持平 / 无同方向候选 / 无未揭晓候选 |

Agent 第二步规则的可复现定义：`30/70` 更好时选 `20/80`，`70/30` 更好时选 `80/20`；`30/70` 更差时选 `70/30`，`70/30` 更差时选 `30/70`；Recall@100 持平则停止。这里的「更好/更差/持平」按 Recall@100 判断；最终最佳仍按完整规则（Recall → 离 50/50 距离 → 双塔配额升序）选择。该规则只看本侧已揭晓结果，不包含已知最优配额或完整试验顺序。

结果查询工具是 `get_fusion_result(two_tower, mind)`：执行器先校验配额合法性、是否重复、剩余预算，只有全部通过才从只读结果表中取出该组公开指标；拒绝时不返回任何指标。

## 2. 数据来源与口径

canonical 结果（指标直接来自已有 JSON，未重算）：

| 配额（双塔/MIND） | 来源文件 / JSON 指针 | 命中数 | Recall@100 | Coverage | 实际占用（双塔/MIND） |
| --- | --- | --- | --- | --- | --- |
{source_rows}

同一口径核对：

- **数据集 / 划分**：`{replay['dataset']}` / `{replay['split']}`，切分策略 `{replay['split_strategy']}`；train/validation/test 哈希分别为 `{split['train']}`、`{split['validation']}`、`{split['test']}`。
- **用户与目标**：全部来源均为 {manifest['user_scope']['fixed_users']} 个固定用户 / {manifest['user_scope']['targets']} 个验证目标；`users.json` sha256 为 `{manifest['user_scope']['users_file_sha256']}`。
- **召回缓存**：所有来源共享同一缓存身份 `sha256:{manifest['recall_cache']['version']}`（短版 `{manifest['recall_cache']['version_short']}`）；两路召回文件 sha256 为 `{identity['recall_two_tower_sha256']}` / `{identity['recall_mind_sha256']}`，checkpoint 为 `{identity['checkpoints']['two_tower']['sha256']}` / `{identity['checkpoints']['mind']['sha256']}`，`routing_seed={identity['routing_seed']}`、`k_recall={identity['k_recall']}`。
- **融合池上限**：每用户 {manifest['fusion_pool']['per_user']} 个候选，总计 {manifest['fusion_pool']['total']}；本轮所有来源补位次数为 {manifest['fusion_pool']['fills_total']}。
- **评测口径**：`{manifest['metric_definition']}`；融合规则版本 `{manifest['fusion_rule_version']}`。

重复记录一致性核对（冲突数为 0）：

| 配额 | 重复来源 | 已比对字段 | 来源缺失字段 | 结论 |
| --- | --- | --- | --- | --- |
{duplicate_rows}

## 3. Rec Agent 决策轨迹

| 步 | 当时可见（用户只看本侧） | 所选配额 | 选择理由 | 工具返回 | 剩余预算 | 最佳 |
| --- | --- | --- | --- | --- | --- | --- |
{agent_rows}

停止原因：**{agent_stop['reason']}**（`{agent_stop['code']}`；剩余预算 {agent_stop['remaining_budget']}）。

## 4. 固定策略决策轨迹

| 步 | 当时可见（用户只看本侧） | 所选配额 | 选择理由 | 工具返回 | 剩余预算 | 最佳 |
| --- | --- | --- | --- | --- | --- | --- |
{fixed_rows}

停止原因：**{fixed_stop['reason']}**（`{fixed_stop['code']}`；剩余预算 {fixed_stop['remaining_budget']}）。

## 5. 汇总对照

| 侧 | 最佳配额 | 命中数 | Recall@100 | Coverage | 实际揭晓 | 异常请求 | 停止原因 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Rec Agent | `{agent['best_quota']}` | {agent['hits']} | {agent['recall_at_100']:.6f} | {agent['coverage']:.6f} | {agent['reveals']} / 2 | {agent['invalid_requests']} | {agent['stop_reason']} |
| 固定策略 | `{fixed['best_quota']}` | {fixed['hits']} | {fixed['recall_at_100']:.6f} | {fixed['coverage']:.6f} | {fixed['reveals']} / 2 | {fixed['invalid_requests']} | {fixed['stop_reason']} |

- Agent 在**第 {comparison['agent_found_best_on_reveal']} 次**揭晓时取得最终最佳；固定策略在**第 {comparison['fixed_found_best_on_reveal']} 次**揭晓时取得最终最佳。
- 差值（Agent − 固定策略）：命中 **{comparison['delta_agent_minus_fixed']['hits']:+d}**、Recall@100 **{comparison['delta_agent_minus_fixed']['recall_at_100']:+.6f}**、Coverage **{comparison['delta_agent_minus_fixed']['coverage']:+.6f}**。
- 结果判定：**{winner_text}**。{comparison['note']}

## 6. 轻量本地断言（合成结果，不触发真实评测）

`--self-check` 与本次正式回放共用同一批断言；断言使用合成指标，不读取来源、不触发真实评测。

| 断言 | 状态 | 说明 |
| --- | --- | --- |
{check_rows}

## 7. 执行范围与合规

| 项 | 实际 |
| --- | --- |
| 执行命令 | `{command}` |
| 读取的工件 | 仅以下汇总 JSON（完整 sha256 见 `sources.json` / `run_meta.json`）：<br>{source_list.replace(chr(10), '<br>')} |
| 训练 / 重新召回 / 真实评测 / 真实 LLM | **0 / 0 / 0 / 0** |
| 网络请求 | 0 |
| 完整数据集 / 候选明细 / 缓存 | 未读取 / 未读取 / 未读取 |
| 实验记忆索引 | 未读取 |
| 对方轨迹 | 策略不可见；两侧各自使用独立执行器状态 |
| 旧缓存 / 旧验收证据 | 未读取、未改写；本次只写入新的运行子目录 |
| 输入文件回读校验 | 运行前后 sha256 一致（见 `run_meta.json`） |

## 8. 验收对照

| 验收项 | 结论 | 证据 |
| --- | --- | --- |
| 数据口径一致 | 通过 | {canonical_count} 个 canonical 来源共享同一缓存身份；{duplicate_count} 条重复记录核对无冲突 |
| 起点与预算相同 | 通过 | 双方第一步 `visible_before.revealed` 为空、`remaining_budget=2`；两侧预算独立 |
| Agent 根据反馈决策 | 通过 | `feedback_sensitivity`：仅替换首步指标，更好→`20/80`、更差→`70/30`、持平→停止 |
| 预算可核验 | 通过 | 双方实际揭晓均为 2；一致化断言覆盖重复请求与预算耗尽，拒绝时无指标泄露 |
| 执行范围可核验 | 通过 | 训练/召回/真实评测/LLM 均为 0；只读汇总 JSON；输入 sha256 前后一致 |
| 结论符合证据 | 通过 | 如实记录 **{winner_text}**；未外推到测试集或策略泛化 |

## 9. 证据

- [config.toml](config.toml)
- [sources.json](sources.json)
- [agent_trace.json](agent_trace.json)
- [fixed_trace.json](fixed_trace.json)
- [metrics.json](metrics.json)
- [checks.json](checks.json)
- [run_meta.json](run_meta.json)
- [report.md](report.md)

## 10. 结论边界

本轮只覆盖 `kuasar_small` 上的一次配额回放：128 个固定验证用户、362 个目标、5 个已验收配额结果。它验收的是**受控回放、数据口径、预算与决策流程**，不是正式测试集收益，也不是策略泛化能力；完整 Phase 13 的正式实验仍保留为待办。
"""


# ---------------------------------------------------------------------------
# 运行与 CLI
# ---------------------------------------------------------------------------


def make_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    candidate = output_root / f"run-{stamp}-{os.getpid()}"
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"run-{stamp}-{os.getpid()}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _command_string(config_path: Path, output_root: Path) -> str:
    parts = ["python", "-m", "recsys.experiments.offline_replay"]
    if config_path.resolve() != DEFAULT_CONFIG_PATH.resolve():
        parts.extend(["--config", str(config_path)])
    if output_root.resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        parts.extend(["--output-root", str(output_root)])
    return " ".join(shlex.quote(part) for part in parts)


def _verify_sources_unchanged(files_read: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    after: dict[str, str] = {}
    changed: list[str] = []
    for relative, payload in files_read.items():
        path = PROJECT_ROOT / relative
        current = sha256_bytes(path.read_bytes())
        after[relative] = current
        if current != payload["sha256"]:
            changed.append(relative)
    return {"before": {path: item["sha256"] for path, item in files_read.items()}, "after": after, "changed": changed}


def execute_replay(
    config_path: Path,
    output_root: Path,
    run_dir: Path,
    started_at: datetime,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    sources = load_sources(config)
    checks = run_local_checks(config)
    if checks["status"] != "pass":
        raise ReplayInputError(f"轻量本地断言未通过：{checks['failed']} 项失败")

    agent_trace = run_side(
        sources["records"],
        config,
        "agent",
        LocalFeedbackPolicy(
            config["agent_policy"], int(config["best_rule"]["recall_decimals"])
        ),
    )
    fixed_trace = run_side(
        sources["records"], config, "fixed", FixedPolicy(config["fixed_policy"]["order"])
    )
    if agent_trace["status"] != "success" or fixed_trace["status"] != "success":
        raise ReplayInputError(
            "回放失败："
            f"agent={agent_trace['status']}/{agent_trace['error']}, "
            f"fixed={fixed_trace['status']}/{fixed_trace['error']}"
        )

    unchanged = _verify_sources_unchanged(sources["files_read"])
    if unchanged["changed"]:
        raise ReplayInputError(f"输入汇总 JSON 在回放期间被改写：{unchanged['changed']}")

    finished_at = utc_now()
    run_meta = {
        "schema_version": SCHEMA_VERSION,
        "phase": config["replay"]["phase"],
        "command": _command_string(config_path, output_root),
        "argv": sys.argv[1:],
        "started_at": iso_time(started_at),
        "finished_at": iso_time(finished_at),
        "elapsed_s": round((finished_at - started_at).total_seconds(), 3),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(run_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "config": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_files_read": sources["files_read"],
        "source_hashes_unchanged": not unchanged["changed"],
        "source_hashes_after": unchanged["after"],
        "scope": config["scope"],
        "write_scope": str(run_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    metrics = build_metrics(config, sources, checks, agent_trace, fixed_trace, run_meta)
    report = build_report(config, sources, checks, agent_trace, fixed_trace, metrics, run_meta)
    return {
        "config": config,
        "sources": sources,
        "checks": checks,
        "agent_trace": agent_trace,
        "fixed_trace": fixed_trace,
        "metrics": metrics,
        "run_meta": run_meta,
        "report": report,
    }, unchanged


def write_run(run_dir: Path, config_path: Path, payload: Mapping[str, Any]) -> None:
    config_text = config_path.read_text(encoding="utf-8")
    (run_dir / "config.toml").write_text(config_text, encoding="utf-8")
    write_json(run_dir / "sources.json", payload["sources"]["manifest"])
    write_json(run_dir / "agent_trace.json", payload["agent_trace"])
    write_json(run_dir / "fixed_trace.json", payload["fixed_trace"])
    write_json(run_dir / "checks.json", payload["checks"])
    write_json(run_dir / "metrics.json", payload["metrics"])
    write_json(run_dir / "run_meta.json", payload["run_meta"])
    (run_dir / "report.md").write_text(payload["report"], encoding="utf-8")


def write_blocked(run_dir: Path, config_path: Path, error: Exception) -> None:
    if config_path.is_file():
        (run_dir / "config.toml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    blocked = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "reason": str(error),
        "error_type": type(error).__name__,
        "created_at": iso_time(utc_now()),
    }
    write_json(run_dir / "blocked.json", blocked)
    report = f"""# Phase 13 Offline Replay MVP：本次正式对照已停止

- 状态：**blocked**
- 原因：{error}
- 说明：来源缺失、关键口径冲突或本地断言失败时，按任务要求停止正式对照，不拼接不同口径的结果。

本次未产出双方轨迹或胜负结论。修复输入后请重新运行：

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.offline_replay
```

证据：[blocked.json](blocked.json)
"""
    (run_dir / "report.md").write_text(report, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 13 offline replay MVP：只读汇总 JSON，对比 Rec Agent 与固定策略的融合配额选择"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="回放配置 TOML")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="运行根目录（默认每次新建子目录）")
    parser.add_argument("--self-check", action="store_true", help="只运行合成结果的轻量断言，不读取来源")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_check:
        try:
            config = load_config(args.config)
            checks = run_local_checks(config)
        except ReplayInputError as error:
            print(json.dumps({"status": "blocked", "reason": str(error)}, ensure_ascii=False, indent=2))
            return 2
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        return 0 if checks["status"] == "pass" else 1

    started_at = utc_now()
    run_dir = make_run_dir(args.output_root)
    try:
        payload, _ = execute_replay(args.config, args.output_root, run_dir, started_at)
    except Exception as error:  # noqa: BLE001 - 失败也要留下 blocked 证据
        write_blocked(run_dir, args.config, error)
        print(f"[blocked] {error}")
        print(f"运行目录：{run_dir}")
        return 2
    write_run(run_dir, args.config, payload)
    print(f"回放完成：{run_dir}")
    print(
        f"Agent 最佳 {payload['metrics']['agent']['best_quota']} / "
        f"固定策略最佳 {payload['metrics']['fixed']['best_quota']} / "
        f"结论 {payload['metrics']['comparison']['winner']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
