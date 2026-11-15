"""Phase 5 全项目集成：三业务协作演示、工件恢复、缺工件失败用例。

一次 ``run_demo`` 用一个任务 ID 串起四层记录：

1. 总控分发（Research/Master Agent 的 ``trace.jsonl`` / ``decisions.json``）；
2. 三个子 Agent 的真实工具调用（各自 ``events.jsonl`` / ``trace.json``）；
3. 共享实验库里的三条业务记录 + 一条任务记录（同一 ``task_id``）；
4. 统一摘要 ``demo_summary.json`` / ``demo_summary.md`` / ``run.log``，
   包含预算消耗、工件来源（清单 + sha256）和输出路径。

``run_restore`` 在独立进程里只从清单加载工件并跑一次必要推理；
``run_missing_check`` 用缺工件的小用例证明不会触发自动训练、换模型或重建索引。

所有函数都不写 ``artifacts/`` 下的权重、索引或历史记录。
"""

from __future__ import annotations

import copy
import json
import time
import tomllib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from recsys.platform.artifacts import (
    DEFAULT_MANIFEST_PATH,
    ManifestError,
    display_path,
    get_entry,
    identity,
    load_manifest,
    resolve_path,
    verify_artifact_usage,
    verify_business_usage,
    verify_manifest,
)
from recsys.platform.budget import BudgetExceeded, TaskBudget
from recsys.platform.research_agent import ResearchAgent, ResearchConfig
from recsys.platform.shared_experiments import (
    SharedExperimentStore,
    query_records,
)
from recsys.utils.artifact_guard import (
    diff_snapshots,
    find_new_files,
    snapshot_paths,
)

DEFAULT_DEMO_CONFIG = Path("configs/platform/phase5_demo.toml")
BUSINESSES: tuple[str, ...] = ("recommendation", "search", "ads")
BUSINESS_LABELS = {"recommendation": "推荐", "search": "搜索", "ads": "广告"}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


@dataclass
class DemoConfig:
    path: Path
    raw: dict[str, Any]
    research: ResearchConfig
    demo: dict[str, Any] = field(default_factory=dict)
    restore: dict[str, Any] = field(default_factory=dict)
    manifest_path: Path = DEFAULT_MANIFEST_PATH

    @property
    def root(self) -> Path:
        return resolve_path(self.research.experiment_root)


def load_demo_config(
    path: str | Path | None = None, *, output_root: str | Path | None = None
) -> DemoConfig:
    config_path = resolve_path(path or DEFAULT_DEMO_CONFIG)
    if not config_path.is_file():
        raise FileNotFoundError(f"Phase 5 演示配置不存在：{config_path}")
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    research = ResearchConfig.load(config_path)
    research.subagent_intent.setdefault("recommendation", {})["artifact_policy"] = "existing_only"
    if output_root is not None:
        research.experiment_root = resolve_path(output_root)
        research.store_dir = research.experiment_root / "shared"
    demo = dict(raw.get("demo") or {})
    restore = dict(raw.get("restore") or {})
    manifest_path = Path(demo.get("manifest") or DEFAULT_MANIFEST_PATH)
    return DemoConfig(
        path=config_path,
        raw=raw,
        research=research,
        demo=demo,
        restore=restore,
        manifest_path=manifest_path,
    )


def _store(config: DemoConfig) -> SharedExperimentStore:
    return SharedExperimentStore(resolve_path(config.research.store_dir))


# ---------------------------------------------------------------------- 摘要


def _compact_metrics(business: str, record: Mapping[str, Any]) -> dict[str, Any]:
    metrics = (record.get("result") or {}).get("metrics") or {}
    model = record.get("model") or {}
    if business == "recommendation":
        pipeline = metrics.get("run_pipeline") or {}
        stages = (
            pipeline
            if any(name in pipeline for name in ("ranked", "rerank"))
            else (pipeline.get("metrics") or {})
        )
        return {
            "ranker": model.get("ranker"),
            "recall_models": model.get("models"),
            "weights_source": model.get("weights_source"),
            "ranker_checkpoint": model.get("checkpoint"),
            "stages": {
                name: stages.get(name) for name in ("ranked", "rerank") if stages.get(name)
            },
        }
    if business == "search":
        return {
            "passed": metrics.get("passed"),
            "tool_sequence": metrics.get("tool_sequence"),
            "return_counts": metrics.get("return_counts"),
            "status": metrics.get("status"),
        }
    if business == "ads":
        prediction = (metrics.get("predict_ad_ctr") or {}).get("metrics") or {}
        auction = (metrics.get("simulate_auction") or {}).get("summaries") or []
        auction_summary = (auction[0].get("summary") if auction else {}) or {}
        return {
            "predict": {
                key: prediction.get(key) for key in ("AUC", "LogLoss", "ctr") if key in prediction
            },
            "auction": {
                key: auction_summary.get(key)
                for key in ("requests", "winners", "clicks", "spend")
                if key in auction_summary
            },
            "budget": auction_summary.get("budget"),
        }
    return {}


def _failure_step(
    business: str,
    *,
    tool_record: Mapping[str, Any] | None,
    record: Mapping[str, Any] | None,
    usage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """定位失败发生在哪一步：分发 / 子 Agent 内部工具 / 记录写入 / 工件核对。"""

    if tool_record is None:
        return {"step": "dispatch", "detail": "总控没有为该业务留下工具调用记录（预算不足或未分发）"}
    if not tool_record.get("ok"):
        return {
            "step": "business_tool",
            "detail": tool_record.get("error") or (record or {}).get("result", {}).get("error"),
        }
    if record is None:
        return {"step": "record", "detail": "子 Agent 成功但没有写入共享实验记录"}
    if str(record.get("status")) != "success":
        return {
            "step": "business_result",
            "detail": (record.get("result") or {}).get("error")
            or f"业务状态 {record.get('status')}",
        }
    if usage is not None and not usage.get("ok"):
        return {
            "step": "artifact_verification",
            "detail": "本次运行实际加载的工件与清单不一致",
        }
    return {"step": "none", "detail": None}


def _business_entry(
    business: str,
    *,
    tool_result: Mapping[str, Any] | None,
    record: Mapping[str, Any] | None,
    usage: Mapping[str, Any] | None,
    task_dir: Path,
) -> dict[str, Any]:
    artifacts = {}
    if record is not None:
        artifacts = dict(record.get("artifacts") or {})
    elif tool_result is not None:
        artifacts = dict(tool_result.get("result_summary", {}).get("artifacts") or {})
    internal_tools = (record or {}).get("result", {}).get("tools_called") or []
    if not internal_tools:
        record_metrics = ((record or {}).get("result") or {}).get("metrics") or {}
        internal_tools = record_metrics.get("tool_sequence") or []
    if not internal_tools:
        internal_tools = (
            (tool_result or {}).get("result_summary", {}).get("tools_called") or []
        )
    entry = {
        "business": business,
        "label": BUSINESS_LABELS[business],
        "status": (record or {}).get("status") or (tool_result or {}).get("status") or "missing",
        "ok": bool((tool_result or {}).get("ok")),
        "dispatch_tool": (tool_result or {}).get("tool"),
        "record_id": (record or {}).get("record_id") or (tool_result or {}).get("record_id"),
        "internal_tools": internal_tools,
        "execution_dir": display_path(task_dir / f"business_{business}"),
        "artifacts": artifacts,
        "metrics": _compact_metrics(business, record or {}),
        "weights_source": ((record or {}).get("model") or {}).get("weights_source"),
        "error": (record or {}).get("result", {}).get("error")
        or (tool_result or {}).get("error"),
        "artifact_usage": usage or {},
    }
    entry["failure"] = _failure_step(
        business, tool_record=tool_result, record=record, usage=usage
    )
    return entry


# ---------------------------------------------------------------------- 演示


def _manifest_gate(
    config: DemoConfig, *, business: str | None = None
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """加载并校验工件清单；返回 (manifest, error)。清单失败时不跑业务。"""

    try:
        manifest = load_manifest(config.manifest_path)
    except ManifestError as error:
        return {}, {"ok": False, "error": str(error), "path": display_path(config.manifest_path)}
    check = verify_manifest(manifest, business=business)
    return manifest, (None if check["ok"] else {"ok": False, **check})


def run_demo(
    config_path: str | Path | None = None,
    *,
    task: str | None = None,
    planner: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """从统一入口跑一次三业务协作演示；返回统一摘要。"""

    config = load_demo_config(config_path, output_root=output_root)
    root = config.root
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest, manifest_error = _manifest_gate(config)
    demo_task = task or str(config.demo.get("task") or "Phase 5 三业务演示")
    effective_planner = str(planner or config.demo.get("planner") or "scripted")

    if manifest_error is not None:
        return _write_demo_outputs(
            config,
            root=root,
            task=demo_task,
            task_id=f"task-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}",
            summary={},
            records={},
            manifest=manifest,
            manifest_error=manifest_error,
            started=started,
            ran=False,
        )

    intent: dict[str, Any] = {"businesses": list(BUSINESSES)}
    if config.demo.get("history_first") is not None:
        intent["history_first"] = bool(config.demo["history_first"])
    summary = ResearchAgent(
        config.research, planner=effective_planner, store=_store(config)
    ).run(demo_task, intent=intent)

    task_id = str(summary.get("task_id"))
    task_dir = root / "tasks" / task_id
    store = _store(config)
    store_query = query_records(store, task_id=task_id, reusable=None, limit=50, include_tasks=True)
    records = {
        str(item.get("business")): store.get(item.get("record_id"))
        for item in store_query.get("matched") or []
        if item.get("business") in BUSINESSES
    }
    tool_by_business = {
        str(item.get("business")): item
        for item in summary.get("tool_results") or []
        if item.get("business") in BUSINESSES
    }

    usages: dict[str, Any] = {}
    for business in BUSINESSES:
        record = records.get(business)
        if record is None:
            continue
        try:
            usages[business] = verify_business_usage(business, record=record, manifest=manifest)
        except (ManifestError, KeyError, OSError, ValueError) as error:
            usages[business] = {"ok": False, "error": f"{type(error).__name__}: {error}"}

    return _write_demo_outputs(
        config,
        root=root,
        task=demo_task,
        task_id=task_id,
        summary=summary,
        records=records,
        manifest=manifest,
        manifest_error=None,
        started=started,
        ran=True,
        usages=usages,
        tool_by_business=tool_by_business,
        task_dir=task_dir,
        store_query=store_query,
    )


def _write_demo_outputs(
    config: DemoConfig,
    *,
    root: Path,
    task: str,
    task_id: str,
    summary: Mapping[str, Any],
    records: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_error: Mapping[str, Any] | None,
    started: float,
    ran: bool,
    usages: Mapping[str, Any] | None = None,
    tool_by_business: Mapping[str, Any] | None = None,
    task_dir: Path | None = None,
    store_query: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    usages = dict(usages or {})
    tool_by_business = dict(tool_by_business or {})
    task_dir = task_dir or (root / "tasks" / task_id)
    run_dir = root / "runs" / task_id
    run_dir.mkdir(parents=True, exist_ok=True)

    businesses: dict[str, Any] = {}
    if ran:
        for business in BUSINESSES:
            businesses[business] = _business_entry(
                business,
                tool_result=tool_by_business.get(business),
                record=records.get(business),
                usage=usages.get(business),
                task_dir=task_dir,
            )

    manifest_ok = not manifest_error
    all_records_success = bool(businesses) and all(
        str(item.get("status")) == "success" for item in businesses.values()
    )
    all_dispatched = bool(businesses) and all(item.get("record_id") for item in businesses.values())
    usage_ok = bool(businesses) and all(
        bool((usages.get(business) or {}).get("ok")) for business in businesses
    )
    rec_checks = (
        ((records.get("recommendation") or {}).get("evidence") or {}).get("checks") or {}
    )
    auto_train_disabled = bool(rec_checks.get("auto_train_disabled"))

    if not ran:
        status = "blocked"
    elif all_records_success and all_dispatched and usage_ok and auto_train_disabled:
        status = "verified"
    elif any(str(item.get("status")) == "success" for item in businesses.values()):
        status = "partial"
    else:
        status = "failed"

    budget = dict(summary.get("budget") or {})
    checks = {
        "manifest_verified": bool(manifest_ok),
        "all_businesses_present": set(businesses) == set(BUSINESSES),
        "all_records_success": all_records_success,
        "all_records_dispatched": all_dispatched,
        "artifacts_match_manifest": usage_ok,
        "auto_train_disabled": auto_train_disabled,
        "business_tools_real": all(
            item.get("dispatch_tool")
            == {
                "recommendation": "run_recommendation_experiment",
                "search": "run_search_task",
                "ads": "run_ads_experiment",
            }.get(business)
            for business, item in businesses.items()
        ),
        "new_experiments": summary.get("new_experiments"),
    }
    payload = {
        "schema": "recsys.platform.phase5.demo.v1",
        "demo": "three-business",
        "task_id": task_id,
        "task": task,
        "status": status,
        "verified": status == "verified",
        "planner": summary.get("planner"),
        "decision_source": summary.get("decision_source"),
        "llm_used": summary.get("llm_used"),
        "stop_reason": summary.get("stop_reason"),
        "stop_detail": summary.get("stop_detail"),
        "manifest": {
            "path": manifest.get("manifest_path") or display_path(config.manifest_path),
            "sha256": manifest.get("manifest_sha256"),
            "ok": bool(manifest_ok),
            "error": manifest_error,
            "artifacts": [
                identity(manifest, str(entry["id"]))
                for entry in manifest.get("artifacts") or []
            ]
            if manifest
            else [],
        },
        "businesses": businesses,
        "records": {
            business: (record or {}).get("record_id") for business, record in records.items()
        },
        "task_record_id": summary.get("task_record_id"),
        "budget": budget,
        "budget_refusals": summary.get("budget_refusals") or [],
        "checks": checks,
        "decisions": summary.get("decisions") or [],
        "tool_results": [
            {
                key: item.get(key)
                for key in (
                    "step",
                    "tool",
                    "business",
                    "ok",
                    "status",
                    "record_id",
                    "error",
                    "elapsed_s",
                )
            }
            for item in summary.get("tool_results") or []
        ],
        "outputs": {
            "root": display_path(root),
            "task_dir": display_path(task_dir),
            "run_dir": display_path(run_dir),
            "summary": display_path(run_dir / "demo_summary.json"),
            "report": display_path(run_dir / "demo_summary.md"),
            "log": display_path(run_dir / "run.log"),
            "artifacts_used": display_path(run_dir / "artifacts_used.json"),
            "shared_store": display_path(_store(config).records_path),
        },
        "notes": _notes(
            status,
            businesses,
            manifest_ok,
            auto_train_disabled,
            decision_source=summary.get("decision_source"),
        ),
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    if store_query is not None:
        payload["store_query"] = {
            "records_path": store_query.get("records_path"),
            "candidates": store_query.get("candidates"),
            "count": store_query.get("count"),
            "bad_lines": store_query.get("bad_lines"),
        }

    _write_json(run_dir / "demo_summary.json", payload)
    _write_json(
        run_dir / "artifacts_used.json",
        {
            "task_id": task_id,
            "manifest": payload["manifest"],
            "businesses": {
                business: item.get("artifact_usage") for business, item in businesses.items()
            },
        },
    )
    (run_dir / "demo_summary.md").write_text(_render_markdown(payload), encoding="utf-8")
    (run_dir / "run.log").write_text(_render_log(payload), encoding="utf-8")
    return payload


def _notes(
    status: str,
    businesses: Mapping[str, Any],
    manifest_ok: bool,
    auto_train_disabled: bool,
    decision_source: str | None = None,
) -> list[str]:
    notes: list[str] = []
    if not manifest_ok:
        notes.append("工件清单校验失败，演示未启动任何业务；不会训练、换模型或重建索引。")
        return notes
    # 决策来源必须如实标注：scripted 是离线替身，llm 是真实模型调用。
    if str(decision_source) == "llm":
        notes.append("决策由真实 LLM 给出（source=llm），业务工具是真实代码。")
    else:
        notes.append("决策由离线脚本替身给出（source=scripted_stand_in），业务工具是真实代码。")
    if auto_train_disabled:
        notes.append("推荐排序权重以 artifact_policy=existing_only 加载，运行记录确认未自动训练。")
    if status != "verified":
        pending = [
            name
            for name, item in businesses.items()
            if str(item.get("status")) != "success"
        ]
        if pending:
            notes.append(f"未完成业务：{', '.join(pending)}；已完成的业务结果保留。")
    notes.append("广告竞价金额与预算为模拟值，转化标签为 synthetic fixture，不代表真实效果。")
    return notes


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 5 三业务协作演示",
        "",
        f"- 任务 ID：`{payload.get('task_id')}`",
        f"- 状态：**{payload.get('status')}**（verified={payload.get('verified')}）",
        f"- 决策来源：`{payload.get('decision_source')}` / planner=`{payload.get('planner')}`",
        f"- 停止原因：`{payload.get('stop_reason')}`",
        f"- 任务记录：`{payload.get('task_record_id')}`",
        "",
        "## 三业务结果",
        "",
        "| 业务 | 状态 | 记录 ID | 内部工具 | 工件核对 | 失败步骤 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, item in (payload.get("businesses") or {}).items():
        tools = ", ".join(str(tool) for tool in item.get("internal_tools") or []) or "-"
        usage = item.get("artifact_usage") or {}
        failure = item.get("failure") or {}
        lines.append(
            f"| {item.get('label')} | {item.get('status')} | `{item.get('record_id')}` | "
            f"{tools} | {'通过' if usage.get('ok') else '未通过'} | "
            f"{failure.get('step')}{('：' + str(failure.get('detail'))) if failure.get('detail') else ''} |"
        )
    budget = payload.get("budget") or {}
    lines += [
        "",
        "## 预算（总控与子 Agent 共用）",
        "",
        f"- 上限：`{budget.get('policy')}`",
        f"- 用量：`{budget.get('counters')}`",
        f"- 剩余：`{budget.get('remaining')}`",
        "",
        "## 工件",
        "",
    ]
    for item in (payload.get("manifest") or {}).get("artifacts") or []:
        lines.append(
            f"- `{item.get('id')}`（{item.get('business')}/{item.get('role')}）："
            f"`{item.get('path')}` sha256=`{(item.get('sha256') or '')[:16]}…`"
        )
    lines += ["", "## 输出", ""]
    for key, value in (payload.get("outputs") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines += ["", "## 说明", ""]
    lines += [f"- {note}" for note in payload.get("notes") or []]
    lines.append("")
    return "\n".join(lines)


def _render_log(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 5 统一运行日志（无密钥；路径相对仓库根目录）",
        f"task_id={payload.get('task_id')} status={payload.get('status')} "
        f"decision_source={payload.get('decision_source')}",
    ]
    for item in payload.get("tool_results") or []:
        lines.append(
            f"step={item.get('step')} tool={item.get('tool')} business={item.get('business')} "
            f"ok={item.get('ok')} status={item.get('status')} record_id={item.get('record_id')} "
            f"elapsed_s={item.get('elapsed_s')} error={item.get('error')}"
        )
    for business, item in (payload.get("businesses") or {}).items():
        tools = ",".join(str(tool) for tool in item.get("internal_tools") or []) or "-"
        usage = item.get("artifact_usage") or {}
        lines.append(
            f"business={business} status={item.get('status')} record_id={item.get('record_id')} "
            f"internal_tools={tools} artifacts_ok={bool(usage.get('ok'))} "
            f"failure_step={(item.get('failure') or {}).get('step')}"
        )
    budget = payload.get("budget") or {}
    lines.append(f"budget_counters={budget.get('counters')} remaining={budget.get('remaining')}")
    for key, value in (payload.get("outputs") or {}).items():
        lines.append(f"output.{key}={value}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------- 单业务


def run_single_business(
    business: str,
    config_path: str | Path | None = None,
    *,
    task: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """从统一入口单独调用推荐 / 搜索 / 广告（仍走同一套业务适配层）。"""

    if business not in BUSINESSES:
        raise ValueError(f"未知业务 {business!r}；可选 {BUSINESSES}")
    from recsys.platform.adapters import run_business

    config = load_demo_config(config_path, output_root=output_root)
    root = config.root
    manifest, manifest_error = _manifest_gate(config, business=business)
    if manifest_error is not None:
        return {
            "ok": False,
            "business": business,
            "status": "blocked",
            "error": "该业务所需工件缺失或校验不通过，未执行",
            "manifest": manifest_error,
        }

    task_id = f"task-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    budget = TaskBudget(task_id, config.research.budget)
    scope = budget.child(business, config.research.policy_for(business))
    experiment_dir = root / "single" / f"{business}-{task_id}"
    try:
        result = run_business(
            business,
            task or f"Phase 5 单业务调用：{BUSINESS_LABELS[business]}",
            task_id=task_id,
            scope=scope,
            experiment_dir=experiment_dir,
            intent=config.research.intent_for(business),
            store=_store(config),
        )
    except BudgetExceeded as error:
        result = {
            "ok": False,
            "status": "failed",
            "error": error.reason,
            "budget_refusal": error.as_dict(),
            "budget": scope.snapshot(),
            "tools_called": [],
        }
    record = _store(config).get(str(result.get("record_id"))) if result.get("record_id") else None
    usage = (
        verify_business_usage(business, record=record, manifest=manifest, require_vector=False)
        if record is not None
        else {"ok": False, "error": "没有记录可核对"}
    )
    payload = {
        "schema": "recsys.platform.phase5.single_business.v1",
        "task_id": task_id,
        "business": business,
        "status": result.get("status"),
        "ok": bool(result.get("ok")) and bool(usage.get("ok")),
        "record_id": result.get("record_id"),
        "tools_called": result.get("tools_called"),
        "artifacts": result.get("artifacts"),
        "artifact_usage": usage,
        "scope_budget": result.get("budget"),
        "task_budget": budget.snapshot(),
        "experiment_dir": display_path(experiment_dir),
        "shared_store": display_path(_store(config).records_path),
        "error": result.get("error"),
        "budget_refusal": result.get("budget_refusal"),
        "elapsed_s": result.get("elapsed_s"),
    }
    _write_json(root / "single" / f"{business}-{task_id}" / "single_summary.json", payload)
    return payload


# ---------------------------------------------------------------------- 工件恢复


def _restore_rec(
    manifest: Mapping[str, Any], *, users: int, k_recall: int, k_final: int, output_dir: Path
) -> dict[str, Any]:
    from recsys.pipeline.chain import RecommendationPipeline
    from recsys.pipeline.config import (
        FusionConfig,
        PipelineConfig,
        RankingConfig,
        RecallChannelConfig,
        RerankConfig,
        RunConfig,
    )

    two_tower = resolve_path(get_entry(manifest, "rec_two_tower")["path"])
    mind = resolve_path(get_entry(manifest, "rec_mind")["path"])
    ranker = resolve_path(get_entry(manifest, "rec_lightgbm_ranker")["path"])
    config = PipelineConfig(
        run=RunConfig(
            dataset=str(manifest.get("dataset") or "kuasar_small"),
            artifacts_dir=Path("artifacts"),
            output_name=f"restore-{output_dir.name}",
            split="validation",
            users=int(users),
            k_recall=int(k_recall),
            k_final=int(k_final),
            seed=42,
        ),
        recall=(
            RecallChannelConfig(name="two_tower", model="two_tower", checkpoint=two_tower, k=k_recall),
            RecallChannelConfig(name="mind", model="mind", checkpoint=mind, k=k_recall),
        ),
        fusion=FusionConfig(mode="round_robin", limit=k_recall),
        ranking=RankingConfig(
            model="lightgbm_ranker",
            checkpoint=ranker,
            train_if_missing=False,
        ),
        rerank=RerankConfig(category_cap=5, author_cap=0, diversity=True),
    )
    pipeline = RecommendationPipeline(config)
    result = pipeline.run(split="validation", write=True, output_dir=output_dir)
    run_meta = json.loads((output_dir / "run_meta.json").read_text(encoding="utf-8"))
    checkpoints = run_meta.get("checkpoints") or {}
    expected = {
        "two_tower": get_entry(manifest, "rec_two_tower"),
        "mind": get_entry(manifest, "rec_mind"),
        "ranker": get_entry(manifest, "rec_lightgbm_ranker"),
    }
    checks = []
    for name, entry in expected.items():
        actual = checkpoints.get(name) or {}
        passed = str(actual.get("sha256") or "").lower() == str(entry.get("sha256")).lower()
        checks.append(
            {
                "checkpoint": name,
                "ok": passed,
                "path": actual.get("path"),
                "sha256": actual.get("sha256"),
                "manifest_sha256": entry.get("sha256"),
            }
        )
    sample_user = result.users[0] if result.users else None
    return {
        "ok": bool(result.final) and all(item["ok"] for item in checks),
        "users": len(result.users),
        "items_for_first_user": len(result.final.get(sample_user, [])) if sample_user is not None else 0,
        "checkpoints": checks,
        "runtime_s": result.timings.get("total_s"),
        "output_dir": display_path(output_dir),
        "metrics": result.metrics.get("rerank"),
    }


def _restore_search(
    manifest: Mapping[str, Any], *, session_id: str, top_k: int
) -> dict[str, Any]:
    from recsys.experiments import search_agent

    config = search_agent.load_config(resolve_path("configs/experiments/search_agent.toml"))
    # 清单是唯一事实来源：把配置里的索引路径改成清单登记的路径，再交给业务链路加载。
    config["search_agent"]["index_path"] = display_path(get_entry(manifest, "search_bm25_index")["path"])
    config["search_agent"]["sample_path"] = display_path(get_entry(manifest, "search_sample")["path"])
    config["search_backend"]["vector_index_dir"] = display_path(
        resolve_path(get_entry(manifest, "search_vector_payload")["path"]).parent
    )
    config["search_backend"]["vector_manifest"] = display_path(
        get_entry(manifest, "search_vector_manifest")["path"]
    )
    config["knowledge"] = {
        "enabled": False,
    }
    context = search_agent.load_context(config, tokens=None, session_id=session_id)
    tokens = list(context.query_tokens)
    hybrid = context.toolbox.run("search_hybrid", {"query_tokens": tokens, "top_k": top_k})
    rerank = context.toolbox.run(
        "search_rerank",
        {"query_tokens": tokens, "top_k": top_k},
        prior_queries=[tokens],
    )
    expected = get_entry(manifest, "search_bm25_index")
    vector = context.backend_identity()["vector"]
    vector_check = verify_artifact_usage(
        manifest, "search_vector_payload", vector["payload"] if vector["faiss_search_calls"] else {}
    )
    checks = [
        {
            "artifact": "search_bm25_index",
            "ok": str(context.index_sha256).lower() == str(expected.get("sha256")).lower(),
            "sha256": context.index_sha256,
            "manifest_sha256": expected.get("sha256"),
            "num_docs": context.index.num_docs,
            "vocab_size": context.index.vocab_size,
        }
    ]
    return {
        "ok": bool(hybrid.get("results")) and bool(rerank.get("results"))
        and checks[0]["ok"] and vector_check["ok"],
        "session_id": context.session_id,
        "query_tokens": tokens,
        "query_source": context.query_source,
        "hybrid_top": hybrid.get("results") or [],
        "rerank_top": rerank.get("results") or [],
        "fusion": hybrid.get("fusion"),
        "rerank": rerank.get("rerank"),
        "index": checks[0],
        "vector_index": vector_check,
    }


def _restore_ads(manifest: Mapping[str, Any], *, limit: int) -> dict[str, Any]:
    from recsys.ads.runtime import AdsRuntime
    from recsys.ads.service import predict_ad_ctr
    from recsys.utils.artifact_guard import sha256_file

    expected = get_entry(manifest, "ads_din_weights")
    checkpoint = resolve_path(expected["path"])
    runtime = AdsRuntime(
        dataset=str(manifest.get("dataset") or "kuasar_small"),
        checkpoints={"din": checkpoint},
    )
    prediction = predict_ad_ctr(runtime, "din", split="test", limit=limit)
    info = runtime.weight_info("din")
    actual_sha = sha256_file(checkpoint)
    weight_ok = (
        actual_sha.lower() == str(expected.get("sha256")).lower()
        and info.get("weight_kind") in {"interface_test", "fixed_init_interface_test"}
        and info.get("is_trained") is False
    )
    return {
        "ok": bool(prediction.get("rows")) and weight_ok,
        "weights": {
            "path": display_path(checkpoint),
            "sha256": actual_sha,
            "manifest_sha256": expected.get("sha256"),
            "weight_kind": info.get("weight_kind"),
            "is_trained": info.get("is_trained"),
            "ok": weight_ok,
        },
        "rows": prediction.get("rows"),
        "metrics": prediction.get("metrics"),
        "preview": prediction.get("preview"),
    }


def _restore_rag(manifest: Mapping[str, Any], *, query: str, top_k: int) -> dict[str, Any]:
    from recsys.platform.rag import RagIndex, retrieve_knowledge

    entry = get_entry(manifest, "rag_index")
    index_dir = resolve_path(entry["path"]).parent
    description = RagIndex(index_dir).describe(include_sources=False)
    result = retrieve_knowledge(query, top_k=top_k, index_dir=index_dir)
    hits = result.get("hits") or []
    citation = hits[0].get("citation") if hits else None
    cite_check = False
    if citation:
        try:
            RagIndex(index_dir).read_citation(citation)
            cite_check = True
        except Exception:  # noqa: BLE001 - 引用不可定位即失败
            cite_check = False
    return {
        "ok": bool(result.get("ok")) and bool(hits) and cite_check,
        "index": {
            "path": display_path(index_dir / "index.json"),
            "stats": description.get("stats"),
            "manifest_sha256": entry.get("sha256"),
        },
        "query": query,
        "hits": [
            {"citation": hit.get("citation"), "score": hit.get("score"), "path": hit.get("path")}
            for hit in hits
        ],
        "citation_resolvable": cite_check,
    }


def run_restore(
    config_path: str | Path | None = None,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """在独立进程里从清单加载工件并跑一次真实推理/查询。"""

    config = load_demo_config(config_path, output_root=output_root)
    root = config.root
    stamp = utc_stamp()
    out_dir = root / "restore" / f"restore-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    restore_cfg = config.restore

    manifest, manifest_error = _manifest_gate(config)
    checks: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "schema": "recsys.platform.phase5.restore.v1",
        "restore_id": f"restore-{stamp}",
        "process_id": __import__("os").getpid(),
        "manifest": {
            "path": display_path(config.manifest_path),
            "ok": manifest_error is None,
            "error": manifest_error,
        },
        "checks": checks,
        "output_dir": display_path(out_dir),
    }
    if manifest_error is not None:
        result["status"] = "blocked"
        result["ok"] = False
        _write_json(out_dir / "restore.json", result)
        return result

    runner = lambda name, fn, **kwargs: _run_restore_check(checks, name, fn, **kwargs)

    runner(
        "manifest",
        lambda: {"ok": True, "checked": len(manifest.get("artifacts") or [])},
    )
    runner(
        "recommendation",
        lambda: _restore_rec(
            manifest,
            users=int(restore_cfg.get("users", 2)),
            k_recall=int(restore_cfg.get("k_recall", 20)),
            k_final=int(restore_cfg.get("k_final", 5)),
            output_dir=out_dir / "recommendation",
        ),
    )
    runner(
        "search",
        lambda: _restore_search(
            manifest,
            session_id=str(restore_cfg.get("search_session_id") or "11208"),
            top_k=int(restore_cfg.get("search_top_k", 3)),
        ),
    )
    runner(
        "ads",
        lambda: _restore_ads(manifest, limit=int(restore_cfg.get("ads_limit", 32))),
    )
    runner(
        "rag",
        lambda: _restore_rag(
            manifest,
            query=str(restore_cfg.get("rag_query") or "实验记忆 指纹 复用"),
            top_k=int(restore_cfg.get("rag_top_k", 2)),
        ),
    )
    ok = all(item.get("ok") for item in checks)
    result.update(
        {
            "status": "verified" if ok else "failed",
            "ok": bool(ok),
            "notes": [
                "新进程只从清单登记的路径加载工件，不训练、不重建索引。",
                "推荐 run_meta 的 checkpoints、搜索的 index.sha256、广告的 weight.checkpoint "
                "都是业务链路自己写出的身份，已与清单逐项比对。",
            ],
        }
    )
    _write_json(out_dir / "restore.json", result)
    return result


def _run_restore_check(
    checks: list[dict[str, Any]],
    name: str,
    fn,
    **kwargs: Any,
) -> None:
    started = time.perf_counter()
    try:
        detail = fn(**kwargs)
        ok = bool(detail.pop("ok", True))
    except Exception as error:  # noqa: BLE001 - 单项失败不阻塞其他恢复检查
        detail = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        ok = False
    checks.append(
        {
            "check": name,
            "ok": ok,
            "elapsed_s": round(time.perf_counter() - started, 2),
            "detail": detail,
        }
    )


# ---------------------------------------------------------------------- 缺工件


def run_missing_check(
    config_path: str | Path | None = None,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """缺工件小用例：必须明确失败，且不产生训练/重建副作用。"""

    config = load_demo_config(config_path, output_root=output_root)
    root = config.root
    stamp = utc_stamp()
    out_dir = root / "missing" / f"missing-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    missing_root = Path("artifacts/phase5_missing_do_not_create")
    missing_ranker = missing_root / "ranker" / "lgbm_ranker.txt"
    missing_index = missing_root / "search" / "index.json"
    missing_rag = missing_root / "rag"
    missing_ads = missing_root / "ads"

    watched = [
        resolve_path("artifacts/kuasar_small/mvp/lgbm_ranker.txt"),
        resolve_path("artifacts/kuasar_small/two_tower_features.pt"),
        resolve_path("artifacts/search/kuasar_bm25/index.json"),
        resolve_path("artifacts/search/kuasar_vector/vector/ann.faiss"),
        resolve_path("artifacts/kuasar_small/ads/interface_weights/din.pt"),
        resolve_path(missing_ranker),
        resolve_path(missing_index),
        resolve_path(missing_rag),
        resolve_path(missing_ads),
    ]
    before = snapshot_paths(watched)
    started_ns = time.time_ns()
    checks: list[dict[str, Any]] = []

    # 1) 推荐：existing_only + 指定的缺失排序权重 → 明确失败，不训练。
    def check_rec() -> dict[str, Any]:
        from recsys.experiments.rec_tools import ToolBudget, ToolContext, build_registry

        context = ToolContext(
            artifacts_dir=resolve_path("artifacts"),
            experiment_dir=out_dir / "recommendation",
            users=2,
            budget=ToolBudget(max_tool_calls=3, max_train_calls=0),
        )
        registry = build_registry()
        result = registry.call(
            "run_pipeline",
            {
                "models": ["popularity"],
                "ranker": "lightgbm_ranker",
                "split": "validation",
                "k_recall": 20,
                "k_final": 5,
                "artifact_policy": "existing_only",
                "ranker_checkpoint": str(missing_ranker),
            },
            context,
        )
        trained = list((out_dir / "recommendation").rglob("*.txt")) if (
            out_dir / "recommendation"
        ).exists() else []
        return {
            "ok": bool(result.get("ok")) is False
            and str(missing_ranker) in str(result.get("error") or result.get("missing_artifact"))
            and result.get("auto_train_attempted") is False
            and not trained
            and not resolve_path(missing_ranker).exists(),
            "tool_ok": result.get("ok"),
            "error": result.get("error"),
            "missing_artifact": result.get("missing_artifact"),
            "artifact_policy": result.get("artifact_policy"),
            "auto_train_attempted": result.get("auto_train_attempted"),
            "training_files_written": [display_path(path) for path in trained],
        }

    _run_restore_check(checks, "recommendation_missing_ranker", check_rec)

    # 2) 推荐：默认候选路径也不存在时，错误必须列出检查过的文件。
    def check_rec_defaults() -> dict[str, Any]:
        from recsys.experiments.rec_tools import ToolBudget, ToolContext, build_registry

        context = ToolContext(
            artifacts_dir=out_dir / "empty_artifacts",
            experiment_dir=out_dir / "recommendation_defaults",
            users=2,
            budget=ToolBudget(max_tool_calls=3),
        )
        result = build_registry().call(
            "run_pipeline",
            {
                "models": ["popularity"],
                "ranker": "lightgbm_ranker",
                "artifact_policy": "existing_only",
            },
            context,
        )
        return {
            "ok": bool(result.get("ok")) is False
            and "existing_only" in str(result.get("error"))
            and bool(result.get("checked_paths")),
            "error": result.get("error"),
            "checked_paths": result.get("checked_paths"),
            "auto_train_attempted": result.get("auto_train_attempted"),
        }

    _run_restore_check(checks, "recommendation_missing_defaults", check_rec_defaults)

    # 3) 搜索：索引文件缺失 → Search Agent 直接 blocked，不重建索引。
    def check_search() -> dict[str, Any]:
        from recsys.experiments import search_agent

        search_config = copy.deepcopy(
            search_agent.load_config(resolve_path("configs/experiments/search_agent.toml"))
        )
        search_config["search_agent"]["index_path"] = str(missing_index)
        blocked: str | None = None
        try:
            search_agent.load_context(search_config, tokens=None, session_id="11208")
        except search_agent.AgentBlocked as error:
            blocked = str(error)
        return {
            "ok": blocked is not None
            and "phase5_missing_do_not_create" in blocked
            and not resolve_path(missing_index).exists()
            and not resolve_path(missing_index).parent.exists(),
            "blocked_reason": blocked,
            "index_created": resolve_path(missing_index).exists(),
        }

    _run_restore_check(checks, "search_missing_index", check_search)

    # 4) 广告：没有权重 → 明确不可用，不构建新权重。
    def check_ads() -> dict[str, Any]:
        from recsys.ads.models import AdModelUnavailable
        from recsys.ads.registry import model_status
        from recsys.ads.runtime import AdsRuntime

        runtime = AdsRuntime(artifacts_dir=resolve_path(missing_ads))
        status = model_status("din", artifacts_dir=resolve_path(missing_ads))
        error_text: str | None = None
        try:
            runtime.model("din")
        except AdModelUnavailable as error:
            error_text = str(error)
        return {
            "ok": status.get("status") == "unavailable"
            and error_text is not None
            and not resolve_path(missing_ads).exists(),
            "status": status.get("status"),
            "error": error_text,
            "artifacts_dir_created": resolve_path(missing_ads).exists(),
        }

    _run_restore_check(checks, "ads_missing_weights", check_ads)

    # 5) RAG：索引目录缺失 → 工具返回 ok=false，不重建索引。
    def check_rag() -> dict[str, Any]:
        from recsys.platform.rag import retrieve_knowledge

        result = retrieve_knowledge("实验记忆 指纹 复用", top_k=2, index_dir=missing_rag)
        return {
            "ok": bool(result.get("ok")) is False
            and "phase5_missing_do_not_create" in str(result.get("error"))
            and not resolve_path(missing_rag).exists(),
            "error": result.get("error"),
            "index_created": resolve_path(missing_rag).exists(),
        }

    _run_restore_check(checks, "rag_missing_index", check_rag)

    after = snapshot_paths(watched)
    changes = diff_snapshots(before, after)
    new_artifacts = find_new_files(
        resolve_path("artifacts"),
        since_ns=started_ns,
        suffixes=(".pt", ".txt", ".json", ".faiss", ".npz"),
    )
    ok = (
        all(item.get("ok") for item in checks)
        and not changes
        and not new_artifacts
        and not resolve_path(missing_root).exists()
    )
    result = {
        "schema": "recsys.platform.phase5.missing.v1",
        "case": "missing_artifact_no_train_no_rebuild",
        "status": "verified" if ok else "failed",
        "ok": bool(ok),
        "checks": checks,
        "artifact_changes": changes,
        "new_artifacts": new_artifacts,
        "missing_root": display_path(missing_root),
        "missing_root_created": resolve_path(missing_root).exists(),
        "output_dir": display_path(out_dir),
        "notes": [
            "推荐使用 artifact_policy=existing_only：缺失时在训练之前就返回失败。",
            "搜索 / RAG 缺失时直接 blocked/ok=false，没有重建索引的代码路径。",
            "广告缺失权重时 resolve_weight 返回 unavailable，不会构建新权重。",
        ],
    }
    _write_json(out_dir / "missing_check.json", result)
    return result


# ---------------------------------------------------------------------- 查询


def find_demo_run(
    task_id: str, config_path: str | Path | None = None, *, output_root: str | Path | None = None
) -> Path | None:
    config = load_demo_config(config_path, output_root=output_root)
    root = config.root
    direct = root / "runs" / task_id
    if (direct / "demo_summary.json").is_file():
        return direct
    runs = root / "runs"
    if not runs.exists():
        return None
    for candidate in sorted(runs.iterdir(), reverse=True):
        file = candidate / "demo_summary.json"
        if not file.is_file():
            continue
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(payload.get("task_id")) == str(task_id):
            return candidate
    return None


def report(
    task_id: str,
    config_path: str | Path | None = None,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """按任务 ID 查回统一摘要、三条业务记录与工件路径（不重跑）。"""

    config = load_demo_config(config_path, output_root=output_root)
    run_dir = find_demo_run(task_id, config_path, output_root=output_root)
    store = _store(config)
    store_query = query_records(
        store, task_id=task_id, reusable=None, limit=50, include_tasks=True
    )
    records = []
    for item in store_query.get("matched") or []:
        record = store.get(str(item.get("record_id"))) or {}
        records.append(
            {
                "record_id": item.get("record_id"),
                "business": item.get("business"),
                "status": item.get("status"),
                "models": (record.get("model") or {}).get("models"),
                "weights_source": (record.get("model") or {}).get("weights_source"),
                "artifacts": record.get("artifacts") or {},
                "validated": (record.get("evidence") or {}).get("validated"),
                "reusable": (record.get("evidence") or {}).get("reusable"),
                "warnings": (record.get("evidence") or {}).get("warnings") or [],
            }
        )
    payload: dict[str, Any] = {
        "task_id": task_id,
        "found": run_dir is not None,
        "run_dir": display_path(run_dir) if run_dir else None,
        "summary": None,
        "summary_path": display_path(run_dir / "demo_summary.json") if run_dir else None,
        "report_path": display_path(run_dir / "demo_summary.md") if run_dir else None,
        "log_path": display_path(run_dir / "run.log") if run_dir else None,
        "shared_store": display_path(store.records_path),
        "records_path": store_query.get("records_path"),
        "records": records,
        "business_records": {
            item["business"]: item["record_id"]
            for item in records
            if item.get("business") in BUSINESSES
        },
        "task_record_id": next(
            (item["record_id"] for item in records if item.get("business") == "task"), None
        ),
    }
    if run_dir is not None:
        payload["summary"] = json.loads(
            (run_dir / "demo_summary.json").read_text(encoding="utf-8")
        )
    return payload


__all__ = [
    "BUSINESSES",
    "DEFAULT_DEMO_CONFIG",
    "DemoConfig",
    "find_demo_run",
    "load_demo_config",
    "report",
    "run_demo",
    "run_missing_check",
    "run_restore",
    "run_single_business",
]
