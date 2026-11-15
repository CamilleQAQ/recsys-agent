"""Phase 4 共享实验平台。

复用 Phase 1–3 已有能力：

* 推荐历史继续走 :mod:`recsys.experiments.experiment_history`（Phase 11 JSON +
  实验记忆索引 + 指纹/工件校验），不重写、不改旧记录格式；
* 新增推荐 / 搜索 / 广告统一实验记录（JSONL 追加 + 每条记录独立 JSON），
  保留任务 ID、业务、配置、数据与切分、模型与权重来源、状态、结果、预算消耗和
  工件路径；
* 历史查询支持按业务、数据、切分、模型、任务筛选，并返回命中或拒绝复用的原因；
* 追加使用跨进程文件锁，避免两个进程覆盖彼此记录；记录只追加、不原地改写，
  更新用同一 ``record_id`` 的新 ``revision`` 表达；
* 验证集 / 测试集、真实运行 / 离线替身、广告模拟 / 真实业务数据分别标记；
* 失败和部分完成也入库供排查，但只有 ``validated`` 的成功记录才允许作为证据复用。
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from recsys.platform.locks import file_lock

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_EXPERIMENT_ROOT = Path("experiments/2026-11-10-kuasar_small-platform-phase4")
DEFAULT_STORE_DIR = DEFAULT_EXPERIMENT_ROOT / "shared"
DEFAULT_TASK_DIRNAME = "tasks"

SCHEMA_VERSION = 1
BUSINESSES: tuple[str, ...] = ("recommendation", "search", "ads")
ALL_BUSINESSES: tuple[str, ...] = (*BUSINESSES, "task")
STATUSES: tuple[str, ...] = ("success", "partial", "failed", "blocked")

STATUS_ALIASES = {
    "verified": "success",
    "completed": "success",
    "ok": "success",
    "error": "failed",
    "timeout": "failed",
    "budget_exhausted": "partial",
}

SPLIT_ROLES = ("train", "validation", "test", "not_applicable")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in STATUSES:
        return text
    return STATUS_ALIASES.get(text, text or "failed")


def jsonable(payload: Any) -> Any:
    """把任意结果转成可 JSON 序列化对象；不可序列化的字段降级为字符串。"""

    try:
        return json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return json.loads(json.dumps({"repr": repr(payload)}, ensure_ascii=False))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def display_path(value: str | Path) -> str:
    path = Path(value)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except (ValueError, OSError):
        return str(path)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_checks(artifacts: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for name, value in (artifacts or {}).items():
        item: dict[str, Any] = {"name": str(name), "path": str(value) if value else None}
        path = resolve_path(value) if value else None
        item["exists"] = bool(path and path.exists())
        if item["exists"] and path is not None and path.is_file():
            item["size"] = path.stat().st_size
            # 只对较小文件记录 sha256；大 checkpoint 只核对存在与大小，避免查询变慢。
            if item["size"] <= 16 * 1024 * 1024:
                try:
                    item["sha256"] = sha256_file(path)
                except OSError as error:
                    item["sha256_error"] = str(error)
        checks.append(item)
    return checks


def _config_hash(config: Mapping[str, Any] | None) -> str:
    import hashlib

    text = json.dumps(jsonable(dict(config or {})), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class SharedExperimentStore:
    """JSONL 追加式共享实验库。"""

    root: Path = DEFAULT_STORE_DIR
    filename: str = "records.jsonl"

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def records_path(self) -> Path:
        return self.root / self.filename

    @property
    def lock_path(self) -> Path:
        return self.root / f"{self.filename}.lock"

    # ------------------------------------------------------------------ 读

    def _read_lines(self) -> tuple[list[dict[str, Any]], int]:
        path = self.records_path
        if not path.exists():
            return [], 0
        records: list[dict[str, Any]] = []
        bad = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                # 另一个进程正在追加时可能读到半行；交给下一次查询。
                bad += 1
                continue
            if isinstance(payload, dict) and payload.get("record_id"):
                records.append(payload)
            else:
                bad += 1
        return records, bad

    def records(self) -> list[dict[str, Any]]:
        records, _ = self._read_lines()
        return records

    def latest(self) -> list[dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for record in self.records():
            record_id = str(record.get("record_id"))
            current = latest.get(record_id)
            key = (int(record.get("revision", 1)), str(record.get("created_at", "")))
            if current is None:
                latest[record_id] = record
                continue
            current_key = (int(current.get("revision", 1)), str(current.get("created_at", "")))
            if key >= current_key:
                latest[record_id] = record
        return sorted(latest.values(), key=lambda item: str(item.get("created_at", "")))

    def get(self, record_id: str) -> dict[str, Any] | None:
        for record in self.latest():
            if record.get("record_id") == record_id:
                return record
        return None

    def bad_lines(self) -> int:
        _, bad = self._read_lines()
        return bad

    # ------------------------------------------------------------------ 写

    def append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        payload = jsonable(dict(record))
        payload.setdefault("schema_version", SCHEMA_VERSION)
        payload.setdefault("created_at", utc_now())
        payload.setdefault("producer", "recsys.platform.shared_experiments")
        with file_lock(self.lock_path):
            existing, _ = self._read_lines()
            record_id = str(payload.get("record_id") or uuid.uuid4().hex)
            revisions = [
                int(item.get("revision", 1))
                for item in existing
                if item.get("record_id") == record_id
            ]
            if "revision" in payload and payload["revision"] is not None:
                revision = int(payload["revision"])
            else:
                revision = (max(revisions) + 1) if revisions else 1
            payload["record_id"] = record_id
            payload["revision"] = revision
            temp_line = json.dumps(payload, ensure_ascii=False) + "\n"
            with self.records_path.open("a", encoding="utf-8") as file:
                file.write(temp_line)
                file.flush()
                os.fsync(file.fileno())
        return payload

    def append_many(self, records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [self.append(record) for record in records]


# ---------------------------------------------------------------------- 记录


def build_record(
    *,
    task_id: str,
    business: str,
    status: Any,
    config: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
    model: Mapping[str, Any] | None = None,
    result: Mapping[str, Any] | None = None,
    budget: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    record_id: str | None = None,
    revision: int | None = None,
    producer: str = "recsys.platform",
) -> dict[str, Any]:
    """构造统一记录；``evidence`` 里可带 validated / reusable / driver / simulation。"""

    evidence_in = dict(evidence or {})
    artifact_checks = _artifact_checks(artifacts)
    all_artifacts_ok = all(item.get("exists") for item in artifact_checks) if artifact_checks else True
    validated = bool(evidence_in.get("validated", False))
    execution_mode = str(evidence_in.get("execution_mode") or "real_tools")
    decision_driver = str(evidence_in.get("decision_driver") or evidence_in.get("driver") or "local")
    reusable_default = bool(validated and normalize_status(status) == "success" and execution_mode == "real_tools")
    reusable = bool(evidence_in.get("reusable", reusable_default))
    if not all_artifacts_ok:
        reusable = False
    evidence_payload = {
        "validated": validated,
        "reusable": reusable,
        "execution_mode": execution_mode,
        "decision_driver": decision_driver,
        "data_kind": str(evidence_in.get("data_kind") or "real_data"),
        "simulation": jsonable(evidence_in.get("simulation") or {}),
        "warnings": list(evidence_in.get("warnings") or []),
        "checks": jsonable(evidence_in.get("checks") or {}),
        "notes": str(evidence_in.get("notes") or ""),
        "artifact_checks": artifact_checks,
        "artifacts_ok": all_artifacts_ok,
    }
    config_payload = jsonable(dict(config or {}))
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id or uuid.uuid4().hex,
        "revision": revision,
        "task_id": str(task_id),
        "business": str(business),
        "status": normalize_status(status),
        "config": config_payload,
        "config_sha256": _config_hash(config_payload),
        "data": jsonable(dict(data or {})),
        "model": jsonable(dict(model or {})),
        "result": jsonable(dict(result or {})),
        "budget": jsonable(dict(budget or {})),
        "artifacts": {str(key): str(value) for key, value in (artifacts or {}).items() if value},
        "evidence": evidence_payload,
        "producer": producer,
        "created_at": utc_now(),
    }
    return record


def record_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    result = record.get("result") or {}
    data = record.get("data") or {}
    model = record.get("model") or {}
    evidence = record.get("evidence") or {}
    metrics = result.get("metrics")
    if isinstance(metrics, Mapping):
        metrics = jsonable(metrics)
    return {
        "record_id": record.get("record_id"),
        "revision": record.get("revision"),
        "task_id": record.get("task_id"),
        "business": record.get("business"),
        "status": record.get("status"),
        "created_at": record.get("created_at"),
        "dataset": data.get("dataset"),
        "split": data.get("split"),
        "split_role": data.get("split_role"),
        "models": model.get("models"),
        "weights_source": model.get("weights_source"),
        "metrics": metrics,
        "summary": (result.get("summary") if isinstance(result, Mapping) else None),
        "decision_driver": evidence.get("decision_driver"),
        "execution_mode": evidence.get("execution_mode"),
        "validated": bool(evidence.get("validated")),
        "reusable": bool(evidence.get("reusable")),
        "warnings": list(evidence.get("warnings") or []),
        "simulation": evidence.get("simulation") or {},
        "artifacts": record.get("artifacts") or {},
        "artifacts_ok": bool(evidence.get("artifacts_ok", True)),
    }


# ---------------------------------------------------------------------- 查询


def _record_matches_model(record: Mapping[str, Any], model: str | None) -> bool:
    if not model:
        return True
    needle = str(model).lower()
    payload = record.get("model") or {}
    names = [str(item).lower() for item in payload.get("models") or []]
    if payload.get("weights_source") and needle in str(payload["weights_source"]).lower():
        return True
    if any(needle == name or needle in name for name in names):
        return True
    for key in ("ranker", "encoder", "models"):
        value = payload.get(key)
        if isinstance(value, str) and needle in value.lower():
            return True
    return False


def _record_text(record: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("business", "task_id", "status"):
        parts.append(str(record.get(key) or ""))
    config = record.get("config") or {}
    for key in ("task", "name", "description"):
        parts.append(str(config.get(key) or ""))
    result = record.get("result") or {}
    for key in ("summary", "final_answer"):
        parts.append(str(result.get(key) or ""))
    return "\n".join(parts).lower()


def reuse_reasons(
    record: Mapping[str, Any],
    request: Mapping[str, Any] | None = None,
) -> list[str]:
    """检查一条记录能否作为当前请求的成功证据复用；返回具体拒绝原因。"""

    request = dict(request or {})
    reasons: list[str] = []
    status = str(record.get("status") or "")
    evidence = record.get("evidence") or {}
    if status != "success":
        reasons.append(f"状态不是 success：{status or 'missing'}")
    if not evidence.get("validated"):
        reasons.append("记录未通过校验（validated=false）")
    if not evidence.get("reusable"):
        reasons.append("记录被标记为不可复用（reusable=false）")
    if str(evidence.get("execution_mode") or "real_tools") != "real_tools":
        reasons.append(f"执行方式不是真实工具：{evidence.get('execution_mode')}")

    business = request.get("business")
    if business and business != "any" and record.get("business") != business:
        reasons.append(f"业务不匹配：记录 {record.get('business')}，请求 {business}")
    task_id = request.get("task_id")
    if task_id and record.get("task_id") != task_id:
        reasons.append(f"任务不匹配：记录 {record.get('task_id')}，请求 {task_id}")

    data = record.get("data") or {}
    for key in ("dataset", "split", "split_role"):
        expected = request.get(key)
        if expected and str(data.get(key)) != str(expected):
            reasons.append(f"{key} 不匹配：记录 {data.get(key)}，请求 {expected}")

    if request.get("model") and not _record_matches_model(record, request.get("model")):
        reasons.append(f"模型不匹配：请求 {request.get('model')}")

    if record.get("business") == "search":
        config = record.get("config") or {}
        if request.get("require_search_query") and not (request.get("session_id") or request.get("tokens")):
            reasons.append("搜索历史复用需要会话 ID、查询 token 或任务 ID")
        for key in ("session_id", "tokens", "mode"):
            expected = request.get(key)
            if expected is not None and config.get(key) != expected:
                reasons.append(f"搜索 {key} 不匹配：记录 {config.get(key)}，请求 {expected}")

    config_sha = request.get("config_sha256") or request.get("config_hash")
    if config_sha and str(record.get("config_sha256")) != str(config_sha):
        reasons.append("实验配置哈希不匹配")

    driver = request.get("driver")
    if driver and str(evidence.get("decision_driver")) != str(driver):
        reasons.append(
            f"决策驱动不匹配：记录 {evidence.get('decision_driver')}，请求 {driver}"
        )

    if evidence.get("simulation"):
        simulation = evidence["simulation"]
        if isinstance(simulation, Mapping) and simulation.get("auction_money") and request.get("allow_simulated") is False:
            reasons.append("广告金额为模拟值，请求要求真实业务数据")

    artifacts = record.get("artifacts") or {}
    checks = {
        str(item.get("name")): item
        for item in (evidence.get("artifact_checks") or [])
        if isinstance(item, Mapping)
    }
    if request.get("require_artifacts", True):
        for name, value in artifacts.items():
            check = checks.get(str(name)) or {}
            path = resolve_path(value)
            if not path.exists():
                reasons.append(f"工件不存在：{name}（{value}）")
                continue
            recorded_size = check.get("size")
            if recorded_size is not None and (not path.is_file() or path.stat().st_size != int(recorded_size)):
                reasons.append(f"工件大小或类型变化：{name}")
            elif check.get("sha256"):
                try:
                    if sha256_file(path) != check["sha256"]:
                        reasons.append(f"工件 SHA-256 不匹配：{name}")
                except OSError:
                    reasons.append(f"无法校验工件 SHA-256：{name}")
    reasons = list(dict.fromkeys(reasons))
    return reasons


def reuse_warnings(record: Mapping[str, Any], request: Mapping[str, Any] | None = None) -> list[str]:
    request = dict(request or {})
    warnings: list[str] = []
    evidence = record.get("evidence") or {}
    simulation = evidence.get("simulation") or {}
    if isinstance(simulation, Mapping):
        if simulation.get("auction_money"):
            warnings.append("广告竞价金额来自模拟配置，不代表真实出价/预算")
        if simulation.get("synthetic_conversion") or simulation.get("conversion_labels") == "synthetic_fixture":
            warnings.append("转化标签来自 synthetic fixture，不是真实 CVR")
    if evidence.get("decision_driver") in {"scripted", "local"}:
        warnings.append(f"决策驱动为 {evidence.get('decision_driver')}（离线替身），业务工具真实执行")
    data = record.get("data") or {}
    if data.get("split_role") == "test":
        warnings.append("该记录在 test 切分上评估")
    if request.get("allow_simulated") and simulation:
        warnings.append("请求显式允许模拟结果")
    return warnings


def query_records(
    store: SharedExperimentStore | Path | str,
    *,
    business: str | None = None,
    task_id: str | None = None,
    dataset: str | None = None,
    split: str | None = None,
    split_role: str | None = None,
    model: str | None = None,
    status: str | None = None,
    reusable: bool | None = None,
    driver: str | None = None,
    text: str | None = None,
    limit: int = 50,
    request: Mapping[str, Any] | None = None,
    include_tasks: bool = False,
) -> dict[str, Any]:
    """统一历史查询；返回匹配记录和拒绝复用的具体原因。"""

    if isinstance(store, SharedExperimentStore):
        store_obj = store
    else:
        store_obj = SharedExperimentStore(resolve_path(store))
    records = store_obj.latest()
    if not include_tasks:
        records = [item for item in records if item.get("business") in BUSINESSES]

    base: list[dict[str, Any]] = []
    for record in records:
        if business and business != "any" and record.get("business") != business:
            continue
        if task_id and record.get("task_id") != task_id:
            continue
        data = record.get("data") or {}
        if dataset and str(data.get("dataset")) != str(dataset):
            continue
        if split and str(data.get("split")) != str(split):
            continue
        if split_role and str(data.get("split_role")) != str(split_role):
            continue
        if model and not _record_matches_model(record, model):
            continue
        if status and str(record.get("status")) != str(status):
            continue
        if driver and str((record.get("evidence") or {}).get("decision_driver")) != str(driver):
            continue
        if text and text.lower() not in _record_text(record):
            continue
        base.append(record)

    filters = {
        "business": business,
        "task_id": task_id,
        "dataset": dataset,
        "split": split,
        "split_role": split_role,
        "model": model,
        "status": status,
        "reusable": reusable,
        "driver": driver,
        "text": text,
    }
    matched: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    request_payload = dict(request or {})
    if business and business != "any":
        request_payload.setdefault("business", business)
    for record in base:
        if reusable is None:
            matched.append(record_summary(record))
            continue
        reasons = reuse_reasons(record, request_payload)
        if bool(not reasons) is bool(reusable):
            matched.append(
                {
                    **record_summary(record),
                    "reuse_warnings": reuse_warnings(record, request_payload),
                    "reuse_reasons": [],
                }
            )
        elif len(rejected) < 20:
            rejected.append(
                {
                    "record_id": record.get("record_id"),
                    "business": record.get("business"),
                    "task_id": record.get("task_id"),
                    "status": record.get("status"),
                    "reasons": reasons,
                }
            )
    matched.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    matched = matched[: max(0, int(limit))]
    return {
        "ok": True,
        "filters": filters,
        "records_path": str(store_obj.records_path),
        "candidates": len(base),
        "count": len(matched),
        "matched": matched,
        "rejected": rejected,
        "bad_lines": store_obj.bad_lines(),
    }


# ---------------------------------------------------------------------- 历史


def _legacy_history_result(
    quota: tuple[int, int] | list[int] | None,
    legacy: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not quota:
        return None
    legacy = dict(legacy or {})
    try:
        from recsys.experiments.fusion_quota import DEFAULT_CACHE, validate_quota
        from recsys.experiments.experiment_history import (
            DEFAULT_HISTORY_ROOT,
            search_experiment_history,
        )
    except Exception as error:  # noqa: BLE001 - 历史工具不可用时按 miss 记录
        return {
            "status": "miss",
            "hit": False,
            "source": "legacy_fusion_quota",
            "reason": f"推荐历史工具不可用：{type(error).__name__}: {error}",
            "reasons": [str(error)],
        }
    try:
        two, mind = validate_quota(int(quota[0]), int(quota[1]))
    except Exception as error:  # noqa: BLE001
        return {
            "status": "miss",
            "hit": False,
            "source": "legacy_fusion_quota",
            "reason": f"配额不合法：{error}",
            "reasons": [str(error)],
        }
    cache_dir = resolve_path(legacy.get("cache_dir") or DEFAULT_CACHE)
    history_root = resolve_path(legacy.get("history_root") or DEFAULT_HISTORY_ROOT)
    memory_dir = legacy.get("memory_dir")
    try:
        result = search_experiment_history(
            two,
            mind,
            cache_dir=cache_dir,
            history_root=history_root,
            memory_dir=resolve_path(memory_dir) if memory_dir else None,
        )
    except Exception as error:  # noqa: BLE001
        return {
            "status": "miss",
            "hit": False,
            "source": "legacy_fusion_quota",
            "reason": f"推荐历史查询失败：{type(error).__name__}: {error}",
            "reasons": [str(error)],
        }
    return {
        "status": result.get("status") or ("hit" if result.get("hit") else "miss"),
        "hit": bool(result.get("hit")),
        "source": f"legacy_fusion_quota:{result.get('source') or 'unknown'}",
        "reason": result.get("reason"),
        "reasons": list((result.get("rejected") or [{}])[0].get("reasons") or []) if result.get("rejected") else [],
        "quota": result.get("quota"),
        "metrics": result.get("metrics"),
        "delta_vs_baseline": result.get("delta_vs_baseline"),
        "fingerprints_verified": result.get("fingerprints_verified"),
        "rule_compatibility": result.get("rule_compatibility"),
        "memory_index": result.get("memory_index"),
        "evidence": result.get("evidence"),
        "warnings": (
            ["phase11_legacy_record"]
            if result.get("source") == "phase11_json"
            else (["memory_index_record"] if result.get("source") == "memory_index" else [])
        ),
    }


def lookup_history(
    *,
    business: str = "any",
    quota: tuple[int, int] | list[int] | None = None,
    task_id: str | None = None,
    dataset: str | None = None,
    split: str | None = None,
    model: str | None = None,
    store: SharedExperimentStore | Path | str | None = None,
    legacy: Mapping[str, Any] | None = None,
    request: Mapping[str, Any] | None = None,
    include_tasks: bool = False,
) -> dict[str, Any]:
    """先查共享库，再查推荐历史；返回 hit 或带原因的 miss。"""

    store_obj = (
        store
        if isinstance(store, SharedExperimentStore)
        else SharedExperimentStore(resolve_path(store) if store else REPO_ROOT / DEFAULT_STORE_DIR)
    )
    request_payload = dict(request or {})
    if business == "any" and not quota and any(request_payload.get(key) for key in ("session_id", "tokens", "mode")):
        business = "search"
    request_payload.setdefault("business", business)
    request_payload["require_search_query"] = not task_id
    if dataset:
        request_payload.setdefault("dataset", dataset)
    if split:
        request_payload.setdefault("split", split)
    if model:
        request_payload.setdefault("model", model)

    unified = (
        {
            "ok": True,
            "matched": [],
            "rejected": [],
            "candidates": 0,
            "records_path": str(store_obj.records_path),
        }
        if quota
        else query_records(
            store_obj,
            business=business,
            task_id=task_id,
            dataset=dataset,
            split=split,
            model=model,
            reusable=True,
            request=request_payload,
            include_tasks=include_tasks,
            limit=20,
        )
    )
    legacy_result = None
    if quota and business in ("recommendation", "any"):
        legacy_result = _legacy_history_result(quota, legacy)

    legacy_reasons = []
    if legacy_result and legacy_result.get("hit"):
        # 旧配额实验只验证了固定的双塔 / MIND 验证集口径。
        legacy_reasons = [
            f"旧配额记录的 {key} 不匹配：记录 {actual}，请求 {request_payload[key]}"
            for key, actual in (("dataset", "kuasar_small"), ("split", "validation"), ("split_role", "validation"))
            if request_payload.get(key) and request_payload[key] != actual
        ]
        requested_model = str(request_payload.get("model") or "").lower().replace("-", "_").replace(" ", "_")
        if requested_model and requested_model not in {"two_tower", "双塔", "mind"}:
            legacy_reasons.append(f"旧配额记录的模型不匹配：{request_payload['model']}")
        if task_id or any(request_payload.get(key) for key in ("task_id", "config_sha256", "config_hash", "driver")):
            legacy_reasons.append("旧配额记录无法核对请求指定的任务 ID、配置哈希或决策来源")

    if legacy_result and legacy_result.get("hit") and not legacy_reasons:
        return {
            "tool": "search_experiment_history",
            "business": "recommendation",
            "quota": legacy_result.get("quota"),
            "status": "hit",
            "hit": True,
            "source": legacy_result.get("source"),
            "reason": legacy_result.get("reason") or "推荐历史命中且通过校验",
            "metrics": legacy_result.get("metrics"),
            "delta_vs_baseline": legacy_result.get("delta_vs_baseline"),
            "checks": {
                "fingerprints_verified": legacy_result.get("fingerprints_verified"),
                "rule_compatibility": legacy_result.get("rule_compatibility"),
                "memory_index": legacy_result.get("memory_index"),
            },
            "warnings": legacy_result.get("warnings") or [],
            "reusable": True,
            "record_id": None,
            "legacy": legacy_result,
            "records": unified.get("matched") or [],
            "rejected": unified.get("rejected") or [],
            "evidence": legacy_result.get("evidence") or {},
        }

    if unified.get("matched"):
        record = unified["matched"][0]
        return {
            "tool": "search_experiment_history",
            "business": record.get("business"),
            "quota": quota,
            "status": "hit",
            "hit": True,
            "source": "shared_experiment_store",
            "reason": "共享实验库中存在校验通过且条件匹配的成功记录",
            "metrics": record.get("metrics"),
            "checks": {"reusable": True, "validated": record.get("validated")},
            "warnings": list(record.get("reuse_warnings") or []),
            "reusable": True,
            "record_id": record.get("record_id"),
            "task_id": record.get("task_id"),
            "records": unified.get("matched") or [],
            "rejected": unified.get("rejected") or [],
            "evidence": {"records_path": unified.get("records_path")},
        }

    reasons = legacy_reasons
    if legacy_result and not legacy_result.get("hit"):
        reasons.append(str(legacy_result.get("reason") or "推荐历史未命中"))
        reasons.extend(str(item) for item in legacy_result.get("reasons") or [])
    if unified.get("rejected"):
        for item in unified["rejected"][:3]:
            reasons.append(f"共享记录 {item.get('record_id')} 拒绝复用：" + "；".join(item.get("reasons") or []))
    elif unified.get("candidates") == 0:
        reasons.append("共享实验库中没有符合条件的记录")
    if not reasons:
        reasons.append("没有可复用记录")
    result = {
        "tool": "search_experiment_history",
        "business": business,
        "quota": quota,
        "status": "miss",
        "hit": False,
        "source": None,
        "reason": "；".join(list(dict.fromkeys(reasons))[:5]),
        "reasons": list(dict.fromkeys(reasons)),
        "metrics": None,
        "records": unified.get("matched") or [],
        "rejected": unified.get("rejected") or [],
        "legacy": legacy_result,
        "warnings": [],
        "reusable": False,
        "evidence": {
            "records_path": unified.get("records_path"),
            "legacy": (legacy_result or {}).get("evidence"),
        },
    }
    return result


def history_model_view(result: Mapping[str, Any]) -> dict[str, Any]:
    """给模型看的历史结果：不含本地路径、哈希和原始工件。"""

    view: dict[str, Any] = {
        "tool": result.get("tool"),
        "business": result.get("business"),
        "status": result.get("status"),
        "hit": bool(result.get("hit")),
        "source": result.get("source"),
        "reason": result.get("reason"),
        "reusable": bool(result.get("reusable")),
        "warnings": list(result.get("warnings") or []),
    }
    if result.get("quota"):
        view["quota"] = result.get("quota")
    if result.get("metrics"):
        view["metrics"] = result.get("metrics")
    if result.get("delta_vs_baseline"):
        view["delta_vs_baseline"] = result.get("delta_vs_baseline")
    checks = result.get("checks") or {}
    if checks:
        view["checks"] = {
            "fingerprints_verified": checks.get("fingerprints_verified"),
            "has_rule_compatibility": bool(checks.get("rule_compatibility")),
            "memory_index_used": bool(checks.get("memory_index")),
            "reusable": checks.get("reusable"),
        }
    if result.get("record_id"):
        view["record_id"] = result.get("record_id")
    if result.get("task_id"):
        view["task_id"] = result.get("task_id")
    return view


# ---------------------------------------------------------------------- 模型注册


def _rec_models() -> dict[str, Any]:
    import tomllib

    path = REPO_ROOT / "models" / "zoo.toml"
    if not path.exists():
        return {"ok": False, "error": f"缺少 {display_path(path)}", "models": []}
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        return {"ok": False, "error": str(error), "models": []}
    try:
        from recsys.experiments.rec_tools import CHECKPOINT_CANDIDATES
    except Exception:  # noqa: BLE001 - 只读注册信息，导入失败不影响返回
        CHECKPOINT_CANDIDATES = {}
    artifacts = REPO_ROOT / "artifacts" / "kuasar_small"
    models: list[dict[str, Any]] = []
    for item in payload.get("models") or []:
        model_id = str(item.get("id") or "")
        candidates = CHECKPOINT_CANDIDATES.get(model_id, (f"{model_id}.pt",))
        checkpoint = next((artifacts / name for name in candidates if (artifacts / name).exists()), None)
        models.append(
            {
                "id": model_id,
                "name": item.get("name"),
                "category": item.get("category"),
                "completeness": item.get("completeness"),
                "adapter": item.get("adapter"),
                "status": item.get("status") or {},
                "checkpoint": display_path(checkpoint) if checkpoint else None,
            }
        )
    return {
        "ok": True,
        "source": display_path(path),
        "version": payload.get("version"),
        "updated": payload.get("updated"),
        "count": len(models),
        "models": models,
    }


def _search_models() -> dict[str, Any]:
    def _load(relative: str) -> dict[str, Any] | None:
        path = REPO_ROOT / relative
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    bm25 = _load("artifacts/search/kuasar_bm25/manifest.json")
    vector = _load("artifacts/search/kuasar_vector/manifest.json")
    try:
        from recsys.search.tool import TOOL_NAMES
    except Exception:  # noqa: BLE001
        TOOL_NAMES = ("search_bm25", "search_vector", "search_hybrid", "search_rerank")
    index = (bm25 or {}).get("index") or {}
    vector_index = (vector or {}).get("index") or {}
    encoder = (vector or {}).get("encoder") or {}
    backend = (vector or {}).get("backend") or {}
    return {
        "ok": bool(bm25 or vector),
        "tools": list(TOOL_NAMES),
        "bm25": {
            "manifest": "artifacts/search/kuasar_bm25/manifest.json" if bm25 else None,
            "index": index.get("path") or "artifacts/search/kuasar_bm25/index.json",
            "num_docs": index.get("num_docs"),
            "vocab_size": index.get("vocab_size"),
            "generated_at": (bm25 or {}).get("generated_at"),
        },
        "vector": {
            "manifest": "artifacts/search/kuasar_vector/manifest.json" if vector else None,
            "num_items": vector_index.get("num_items"),
            "dim": vector_index.get("dim") or encoder.get("dim"),
            "encoder": encoder.get("encoder"),
            "encoder_requires_training": encoder.get("requires_training"),
            "faiss_active": backend.get("faiss_active"),
            "faiss_version": backend.get("faiss_version"),
            "generated_at": (vector or {}).get("generated_at"),
        },
        "note": "KuaiSAR 只有匿名整数 token；token_hash 编码不是训练好的语义向量。",
    }


def _ads_models() -> dict[str, Any]:
    try:
        from recsys.ads.registry import list_ad_models
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "error": f"{type(error).__name__}: {error}", "models": []}
    try:
        models = list_ad_models()
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "error": f"{type(error).__name__}: {error}", "models": []}
    return {
        "ok": True,
        "source": "recsys.ads.registry.list_ad_models",
        "count": len(models),
        "models": models,
        "note": "status=interface_test 表示只有固定初始化权重，不能表述为已训练模型。",
    }


def model_registry_view(business: str | None = None) -> dict[str, Any]:
    """统一查看推荐 / 搜索 / 广告的模型注册和版本信息。"""

    payload: dict[str, Any] = {"ok": True, "generated_at": utc_now()}
    if business in (None, "any", "recommendation"):
        payload["recommendation"] = _rec_models()
    if business in (None, "any", "search"):
        payload["search"] = _search_models()
    if business in (None, "any", "ads"):
        payload["ads"] = _ads_models()
    return payload


def registry_model_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    """给模型看的模型注册摘要（去掉路径）。"""

    view: dict[str, Any] = {"ok": payload.get("ok")}
    rec = payload.get("recommendation")
    if isinstance(rec, Mapping):
        view["recommendation"] = {
            "count": rec.get("count"),
            "updated": rec.get("updated"),
            "models": [
                {
                    "id": item.get("id"),
                    "category": item.get("category"),
                    "completeness": item.get("completeness"),
                    "checkpoint_ready": bool(item.get("checkpoint")),
                }
                for item in rec.get("models") or []
            ],
        }
    search = payload.get("search")
    if isinstance(search, Mapping):
        view["search"] = {
            "tools": search.get("tools"),
            "bm25": search.get("bm25"),
            "vector": search.get("vector"),
        }
    ads = payload.get("ads")
    if isinstance(ads, Mapping):
        view["ads"] = {
            "count": ads.get("count"),
            "models": [
                {
                    "name": item.get("name"),
                    "task": item.get("task"),
                    "status": item.get("status"),
                    "weight_kind": item.get("weight_kind"),
                    "usable": item.get("usable"),
                }
                for item in ads.get("models") or []
            ],
        }
    return view


# ---------------------------------------------------------------------- 便捷


def default_store() -> SharedExperimentStore:
    return SharedExperimentStore(REPO_ROOT / DEFAULT_STORE_DIR)


def record_business_result(
    *,
    task_id: str,
    business: str,
    status: Any,
    config: Mapping[str, Any],
    data: Mapping[str, Any],
    model: Mapping[str, Any],
    result: Mapping[str, Any],
    budget: Mapping[str, Any],
    artifacts: Mapping[str, str],
    evidence: Mapping[str, Any],
    store: SharedExperimentStore | Path | str | None = None,
    producer: str = "recsys.platform.adapters",
) -> dict[str, Any]:
    store_obj = (
        store
        if isinstance(store, SharedExperimentStore)
        else SharedExperimentStore(resolve_path(store) if store else REPO_ROOT / DEFAULT_STORE_DIR)
    )
    record = build_record(
        task_id=task_id,
        business=business,
        status=status,
        config=config,
        data=data,
        model=model,
        result=result,
        budget=budget,
        artifacts=artifacts,
        evidence=evidence,
        producer=producer,
    )
    return store_obj.append(record)


def stale_marker(path: Path, *, seconds: float) -> bool:  # pragma: no cover - 运维辅助
    if not path.exists():
        return True
    return (time.time() - path.stat().st_mtime) > float(seconds)


__all__ = [
    "ALL_BUSINESSES",
    "BUSINESSES",
    "DEFAULT_EXPERIMENT_ROOT",
    "DEFAULT_STORE_DIR",
    "REPO_ROOT",
    "SCHEMA_VERSION",
    "SharedExperimentStore",
    "build_record",
    "default_store",
    "display_path",
    "history_model_view",
    "lookup_history",
    "model_registry_view",
    "normalize_status",
    "query_records",
    "record_business_result",
    "record_summary",
    "registry_model_view",
    "resolve_path",
    "reuse_reasons",
    "reuse_warnings",
]
