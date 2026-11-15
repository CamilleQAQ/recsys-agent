"""三个业务 Agent 入口的薄适配。

* 推荐：``recsys.experiments.rec_agent``（LangGraph + 真实模型工具）；
* 搜索：``recsys.experiments.search_agent``（BM25 / FAISS / 混合 / 精排）；
* 广告：``recsys.ads.agent``（CTR / 排序 / 竞价 / 预算）。

适配层只做三件事：把预算收紧后传入、把任务参数翻译成已有的 ``PlannedCall``、
把返回结果写进共享实验平台。原有 CLI 与单独运行方式完全不变。

业务 Agent 的真实工具执行由原实现完成；适配层不会用缓存结果冒充实验，
也不会因为总控调用而写业务默认权重路径。
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Mapping

from recsys.platform.budget import BudgetExceeded
from recsys.platform.rag import display_path
from recsys.platform.shared_experiments import (
    SharedExperimentStore,
    record_business_result,
    resolve_path,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SEARCH_CONFIG = Path("configs/experiments/search_agent.toml")
SEARCH_MODES = ("bm25", "vector", "hybrid", "rerank")

DEFAULT_REC_USERS = 8
DEFAULT_REC_RECALL = ("popularity",)
DEFAULT_ADS_REQUESTS = 2
DEFAULT_ADS_LIMIT = 200


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _now() -> float:
    return time.perf_counter()


def _spend_dispatch(scope: Any, *, business: str, min_internal_tools: int) -> int:
    """扣一次实验 + 一次工具分发，并返回子 Agent 可用的内部工具上限。"""

    scope.check("experiment")
    scope.check("tool")
    scope.check("seconds")
    if scope.remaining("tool") < 1 + min_internal_tools:
        raise BudgetExceeded(
            "tool",
            f"{business} 业务至少需要 {1 + min_internal_tools} 次工具调用，"
            f"当前只剩 {int(scope.remaining('tool'))} 次",
            scope=business,
            snapshot=scope.snapshot(),
        )
    scope.spend("experiment")
    scope.spend("tool")
    return int(max(1, scope.remaining("tool")))


def _record(
    *,
    store: SharedExperimentStore | Path | str | None,
    task_id: str,
    business: str,
    status: str,
    config: Mapping[str, Any],
    data: Mapping[str, Any],
    model: Mapping[str, Any],
    result: Mapping[str, Any],
    artifacts: Mapping[str, str],
    scope: Any,
    driver: str,
    validated: bool,
    warnings: list[str],
    simulation: Mapping[str, Any] | None = None,
    checks: Mapping[str, Any] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    return record_business_result(
        task_id=task_id,
        business=business,
        status=status,
        config=config,
        data=data,
        model=model,
        result=result,
        budget=scope.snapshot(),
        artifacts=artifacts,
        evidence={
            "validated": bool(validated),
            "reusable": bool(validated and status == "success"),
            "execution_mode": "real_tools",
            "decision_driver": driver,
            "data_kind": "real_ads_data" if business == "ads" else "real_data",
            "simulation": dict(simulation or {}),
            "warnings": list(warnings),
            "checks": dict(checks or {}),
            "notes": notes,
        },
        store=store,
        producer=f"recsys.platform.adapters.{business}",
    )


def _artifact_dict(pairs: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, value in pairs.items():
        if not value:
            continue
        path = resolve_path(value)
        if path is not None and path.exists():
            out[str(name)] = display_path(path)
    return out


def _recommendation_artifact_usage(
    summary: Mapping[str, Any], artifacts: Mapping[str, str]
) -> dict[str, Any]:
    """从 run_pipeline 落盘的 ``run_meta.json`` 提取本次真实使用的权重身份。

    run_meta 的 ``checkpoints`` 由推荐链路自己写入（路径 + sha256），因此这里得到的
    是「这次运行实际加载的文件」，而不是配置里声明的候选。
    """

    used: dict[str, Any] = {}
    run_meta_path = artifacts.get("run_meta")
    if not run_meta_path:
        return used
    path = resolve_path(run_meta_path)
    if path is None or not path.is_file():
        return used
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return used
    for name, item in (payload.get("checkpoints") or {}).items():
        if not isinstance(item, Mapping):
            continue
        used[str(name)] = {
            "path": display_path(str(item.get("path"))),
            "sha256": item.get("sha256"),
        }
    used["run_meta"] = display_path(path)
    used["source"] = "run_meta.json#checkpoints"
    return used


# ---------------------------------------------------------------------- 推荐


def run_recommendation(
    task: str,
    *,
    task_id: str,
    scope: Any,
    experiment_dir: Path | str,
    intent: Mapping[str, Any] | None = None,
    store: SharedExperimentStore | Path | str | None = None,
) -> dict[str, Any]:
    """调用 Rec Agent 跑一次真实的小样本推荐链路，并写入共享实验库。"""

    intent = dict(intent or {})
    experiment_dir = resolve_path(experiment_dir) or Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    started = _now()
    users = max(1, _int(intent.get("users"), DEFAULT_REC_USERS))
    split = str(intent.get("split") or "validation")
    recall_models = list(intent.get("recall_models") or DEFAULT_REC_RECALL)
    ranker = str(intent.get("ranker") or "lightgbm_ranker")
    k_recall = _int(intent.get("k_recall"), 20)
    k_final = _int(intent.get("k_final"), 10)
    step = _int(intent.get("step"), 1)
    artifact_policy = str(intent.get("artifact_policy") or "existing_only")
    ranker_checkpoint = intent.get("ranker_checkpoint")
    expected_ranker_sha256 = intent.get("expected_ranker_sha256")

    min_tools = 2
    internal_cap = _spend_dispatch(scope, business="recommendation", min_internal_tools=min_tools)

    from recsys.experiments.rec_agent import AgentSettings, PlannedCall, run_agent

    settings = AgentSettings(
        users=users,
        split=split,
        max_tool_calls=min(internal_cap, 8),
        max_train_calls=0,
        max_seconds=float(min(120.0, scope.remaining("seconds"))),
    )
    pipeline_arguments: dict[str, Any] = {
        "models": recall_models,
        "ranker": ranker,
        "fusion": str(intent.get("fusion") or "round_robin"),
        "split": split,
        "k_recall": k_recall,
        "k_final": k_final,
        "output_name": f"platform_{task_id[-8:]}_s{step}",
        "artifact_policy": artifact_policy,
    }
    if ranker_checkpoint:
        pipeline_arguments["ranker_checkpoint"] = str(ranker_checkpoint)
    if expected_ranker_sha256:
        pipeline_arguments["expected_ranker_sha256"] = str(expected_ranker_sha256)
    plan = [
        PlannedCall("read_data_profile", {}, reason="先读取数据画像"),
        PlannedCall(
            "run_pipeline",
            pipeline_arguments,
            reason="执行一次真实小样本推荐链路（只加载已有工件）",
        ),
    ]
    status = "failed"
    error: str | None = None
    summary: dict[str, Any] = {}
    try:
        summary = run_agent(
            settings,
            task=task,
            plan=plan,
            experiment_dir=experiment_dir,
            force_local=True,
        )
        status = {
            "verified": "success",
            "partial": "partial",
            "failed": "failed",
            "blocked": "blocked",
        }.get(str(summary.get("status")), "failed")
    except Exception as exc:  # noqa: BLE001 - 失败也要入库排查
        error = f"{type(exc).__name__}: {exc}"
    elapsed = round(_now() - started, 2)

    tool_results = list(summary.get("tool_results") or [])
    internal_calls = len(tool_results)
    if internal_calls:
        try:
            scope.spend("tool", internal_calls)
        except BudgetExceeded as budget_error:  # pragma: no cover - 理论上内部上限已收紧
            error = (error + "；" if error else "") + budget_error.reason
            status = "partial"
    llm_requests = _int(summary.get("llm_requests"), 0)
    if llm_requests:
        try:
            scope.spend("llm", llm_requests)
        except BudgetExceeded as budget_error:  # pragma: no cover
            error = (error + "；" if error else "") + budget_error.reason
            status = "partial"

    metrics = (summary.get("metrics") or {}) if isinstance(summary, Mapping) else {}
    artifacts = _artifact_dict(
        {
            "summary": experiment_dir / "summary.json",
            "tool_results": experiment_dir / "tool_results.json",
            "events": experiment_dir / "events.jsonl",
            **(summary.get("artifacts") or {} if isinstance(summary, Mapping) else {}),
        }
    )
    used = _recommendation_artifact_usage(summary, artifacts)
    if status == "success" and not used.get("ranker", {}).get("sha256"):
        status = "partial"
        error = (error + "；" if error else "") + "推荐链路未报告排序权重身份，无法确认使用了已有权重"
    all_ok = bool(tool_results) and all(bool(item.get("ok")) for item in tool_results)
    validated = status == "success" and all_ok
    warnings = ["small_sample_interface_check"] if users <= 16 else []
    if error:
        warnings.append("adapter_error")
    record = _record(
        store=store,
        task_id=task_id,
        business="recommendation",
        status=status,
        config={
            "task": task,
            "step": step,
            "split": split,
            "users": users,
            "recall_models": recall_models,
            "ranker": ranker,
            "k_recall": k_recall,
            "k_final": k_final,
            "artifact_policy": artifact_policy,
            "ranker_checkpoint": used.get("ranker", {}).get("path"),
        },
        data={
            "dataset": "kuasar_small",
            "split": split,
            "split_role": split,
            "source": "existing cached recall + LightGBM ranker",
        },
        model={
            "models": recall_models,
            "ranker": ranker,
            "weights_source": "existing_checkpoints",
            "checkpoint": used.get("ranker", {}).get("path"),
            "used_artifacts": used,
        },
        result={
            "status": status,
            "agent_status": summary.get("status") if isinstance(summary, Mapping) else None,
            "metrics": metrics,
            "final_answer": summary.get("final_answer") if isinstance(summary, Mapping) else None,
            "tools_called": summary.get("tools_called") if isinstance(summary, Mapping) else [],
            "error": error,
        },
        artifacts=artifacts,
        scope=scope,
        driver="local",
        validated=validated,
        warnings=warnings,
        checks={
            "tools_ok": all_ok,
            "tool_calls": internal_calls,
            "small_sample": users <= 16,
            "artifact_policy": artifact_policy,
            "auto_train_disabled": True,
        },
        notes="推荐业务薄适配：复用 rec_agent 的确定性计划与真实工具。",
    )
    return {
        "ok": status == "success" and error is None,
        "business": "recommendation",
        "status": status,
        "agent_status": summary.get("status") if isinstance(summary, Mapping) else None,
        "driver": "local",
        "metrics": metrics,
        "tools_called": list(summary.get("tools_called") or []),
        "artifacts": artifacts,
        "used_artifacts": used,
        "record_id": record.get("record_id"),
        "warnings": warnings,
        "error": error,
        "budget": scope.snapshot(),
        "elapsed_s": elapsed,
        "final_answer": summary.get("final_answer") if isinstance(summary, Mapping) else None,
    }


# ---------------------------------------------------------------------- 搜索


def _load_search_config(path: Path | str | None = None) -> dict[str, Any]:
    from recsys.experiments.search_agent import load_config

    return copy.deepcopy(load_config(resolve_path(path) if path else SEARCH_CONFIG))


def run_search(
    task: str,
    *,
    task_id: str,
    scope: Any,
    experiment_dir: Path | str,
    intent: Mapping[str, Any] | None = None,
    store: SharedExperimentStore | Path | str | None = None,
) -> dict[str, Any]:
    """调用 Search Agent 做一次真实检索（脚本替身只负责决策，工具是真实代码）。"""

    intent = dict(intent or {})
    experiment_dir = resolve_path(experiment_dir) or Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    started = _now()
    step = _int(intent.get("step"), 1)
    mode = str(intent.get("mode") or "bm25").lower()
    if mode not in SEARCH_MODES:
        mode = "bm25"
    top_k = max(1, min(5, _int(intent.get("top_k"), 5)))
    tokens = intent.get("tokens")
    tokens = [int(item) for item in tokens] if isinstance(tokens, (list, tuple)) and tokens else None
    session_id = intent.get("session_id")
    session_id = str(session_id) if session_id is not None else None
    include_knowledge = bool(intent.get("include_knowledge"))
    planner = str(intent.get("planner") or "scripted")
    require_existing_index = bool(intent.get("require_existing_index", True))
    search_config_path = intent.get("search_config")

    from recsys.experiments import search_agent

    use_llm = planner == "llm" and search_agent.llm_config() is not None
    if use_llm:
        scope.check("llm")
    min_tools = 1 + int(mode == "rerank") + int(include_knowledge)
    internal_cap = _spend_dispatch(scope, business="search", min_internal_tools=min_tools)
    config = _load_search_config(search_config_path)
    config["llm"]["total_budget_s"] = min(
        float(config["llm"]["total_budget_s"]), scope.remaining("seconds")
    )
    config["search_agent"]["max_results"] = top_k
    config["search_agent"]["max_search_calls"] = min(
        _int(config["search_agent"].get("max_search_calls"), 2), max(1, internal_cap)
    )
    if include_knowledge:
        config.setdefault("knowledge", {})
        config["knowledge"] = {
            "enabled": True,
            "index_dir": str(intent.get("rag_index_dir") or "artifacts/rag/phase4"),
            "top_k": max(1, min(5, _int(intent.get("knowledge_top_k"), 2))),
        }
    output_root = experiment_dir / "search"
    status = "failed"
    error: str | None = None
    result: dict[str, Any] = {}
    driver = "scripted"
    try:
        if use_llm:
            driver = "llm"
            # 一轮最多发出一次工具调用，用请求上限约束子 Agent 的总工具消耗。
            config["search_agent"]["max_model_requests"] = min(
                _int(config["search_agent"].get("max_model_requests"), 3),
                int(scope.remaining("llm")),
                internal_cap,
            )
            result = search_agent.run(
                config,
                tokens=tokens,
                session_id=session_id,
                output_root=output_root,
                command=f"platform task {task_id}",
            )
        else:
            plan = f"knowledge_{mode}" if include_knowledge else search_agent.SEARCH_MODE_PLANS[mode]
            config["search_agent"]["max_model_requests"] = max(
                _int(config["search_agent"].get("max_model_requests"), 3), min_tools + 1
            )
            result = search_agent.run_scripted(
                config,
                plan=plan,
                tokens=tokens,
                session_id=session_id,
                output_root=output_root,
            )
        metrics = result.get("metrics") or {}
        status = {
            "verified": "success",
            "partial": "partial",
            "failed": "failed",
            "blocked": "blocked",
        }.get(str(metrics.get("status")), "failed")
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        metrics = {}

    summary = metrics.get("summary") or {}
    tool_sequence = list(summary.get("tool_sequence") or [])
    knowledge_calls = int(summary.get("knowledge_calls") or 0)
    if status == "success" and (f"search_{mode}" not in tool_sequence or (include_knowledge and not knowledge_calls)):
        status = "partial"
        error = f"搜索子 Agent 未完成指定模式或知识检索：{mode}"
    internal_calls = len(tool_sequence) + knowledge_calls
    if internal_calls:
        try:
            scope.spend("tool", internal_calls)
        except BudgetExceeded as budget_error:  # pragma: no cover
            error = (error + "；" if error else "") + budget_error.reason
            status = "partial"
    llm_requests = _int(metrics.get("budget", {}).get("request_count"), 0) if driver == "llm" else 0
    if llm_requests:
        try:
            scope.spend("llm", llm_requests)
        except BudgetExceeded as budget_error:  # pragma: no cover
            error = (error + "；" if error else "") + budget_error.reason
            status = "partial"

    run_dir = None
    if metrics.get("run", {}).get("output_dir"):
        run_dir = resolve_path(metrics["run"]["output_dir"])
    artifacts: dict[str, str] = {}
    if run_dir is not None:
        artifacts = _artifact_dict(
            {
                "trace": run_dir / "trace.json",
                "metrics": run_dir / "metrics.json",
                "run_meta": run_dir / "run_meta.json",
            }
        )
    elif result.get("run_meta"):
        artifacts = _artifact_dict(
            {
                "trace": result.get("trace_path") or "",
                "metrics": result.get("metrics_path") or "",
            }
        )
    checks = dict(metrics.get("checks") or {})
    passed = bool(metrics.get("passed"))
    validated = status == "success" and passed
    warnings: list[str] = []
    if driver == "scripted":
        warnings.append("scripted_decision_driver")
    if metrics.get("driver") == "scripted":
        warnings.append("offline_stand_in_decisions")
    index_identity = (metrics.get("index") or {}) if isinstance(metrics, Mapping) else {}
    query_identity = metrics.get("query") or {}
    vector = (metrics.get("search_backend") or {}).get("vector") or {}
    vector_payload = vector.get("payload") if vector.get("faiss_search_calls") else None
    elapsed = round(_now() - started, 2)
    record = _record(
        store=store,
        task_id=task_id,
        business="search",
        status=status,
        config={
            "task": task,
            "step": step,
            "mode": mode,
            "top_k": top_k,
            "session_id": query_identity.get("session_id", session_id),
            "tokens": query_identity.get("original_tokens") or tokens,
            "include_knowledge": include_knowledge,
            "planner": planner,
            "require_existing_index": require_existing_index,
            "scripted_plan": None if driver == "llm" else "search_agent.run_scripted",
        },
        data={
            "dataset": "kuasar_small",
            "split": "not_applicable",
            "split_role": "not_applicable",
            "source": "existing BM25 + FAISS search index",
            "index": {
                "path": index_identity.get("path"),
                "num_docs": index_identity.get("num_docs"),
                "vocab_size": index_identity.get("vocab_size"),
                "sha256": index_identity.get("sha256"),
            },
        },
        model={
            "models": tool_sequence or ([mode] if mode else []),
            "weights_source": "existing_search_index",
            "encoder": (metrics.get("search_backend") or {}).get("vector", {}).get("encoder")
            if isinstance(metrics.get("search_backend"), Mapping)
            else None,
            "used_artifacts": {
                "index": {
                    "path": index_identity.get("path"),
                    "sha256": index_identity.get("sha256"),
                    "num_docs": index_identity.get("num_docs"),
                    "vocab_size": index_identity.get("vocab_size"),
                },
                "source": "search_agent metrics#index",
                "vector_payload": vector_payload,
            },
        },
        result={
            "status": status,
            "agent_status": metrics.get("status"),
            "metrics": {
                "status": metrics.get("status"),
                "passed": passed,
                "return_counts": summary.get("return_counts"),
                "tool_sequence": tool_sequence,
                "first_tool": summary.get("first_tool"),
                "second_tool": summary.get("second_tool"),
                "knowledge_calls": knowledge_calls,
                "budget": metrics.get("budget"),
                "checks": checks,
            },
            "summary": summary.get("stop_summary") or summary.get("stop_code"),
            "error": error,
        },
        artifacts=artifacts,
        scope=scope,
        driver=driver,
        validated=validated,
        warnings=warnings,
        checks={
            "agent_passed": passed,
            "tool_calls": internal_calls,
            "require_existing_index": require_existing_index,
            **checks,
        },
        notes="搜索业务薄适配：脚本替身只提供决策，检索/向量/融合/精排是真实代码。",
    )
    return {
        "ok": status == "success" and error is None,
        "business": "search",
        "status": status,
        "agent_status": metrics.get("status"),
        "driver": driver,
        "metrics": {
            "return_counts": summary.get("return_counts"),
            "tool_sequence": tool_sequence,
            "first_result_count": summary.get("first_result_count"),
            "second_result_count": summary.get("second_result_count"),
            "knowledge_calls": knowledge_calls,
            "passed": passed,
        },
        "used_artifacts": {
            "index": {
                "path": index_identity.get("path"),
                "sha256": index_identity.get("sha256"),
                "num_docs": index_identity.get("num_docs"),
                "vocab_size": index_identity.get("vocab_size"),
            }
        },
        "tools_called": tool_sequence + (["retrieve_knowledge"] if knowledge_calls else []),
        "artifacts": artifacts,
        "record_id": record.get("record_id"),
        "warnings": warnings,
        "error": error,
        "budget": scope.snapshot(),
        "elapsed_s": elapsed,
        "final_answer": summary.get("stop_summary"),
    }


# ---------------------------------------------------------------------- 广告


def run_ads(
    task: str,
    *,
    task_id: str,
    scope: Any,
    experiment_dir: Path | str,
    intent: Mapping[str, Any] | None = None,
    store: SharedExperimentStore | Path | str | None = None,
) -> dict[str, Any]:
    """调用 Ads Agent 做一次真实 CTR 推理 / 排序 / 竞价模拟。"""

    intent = dict(intent or {})
    experiment_dir = resolve_path(experiment_dir) or Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    started = _now()
    step = _int(intent.get("step"), 1)
    from recsys.ads.agent import PlannedCall, detect_model, run_agent
    from recsys.ads.config import AgentConfig, load_agent_config
    from recsys.ads.tools import build_context

    agent_config = load_agent_config()
    model = str(intent.get("model") or detect_model(task) or agent_config.default_model)
    split = str(intent.get("split") or agent_config.default_split)
    requests = max(1, min(10, _int(intent.get("requests"), DEFAULT_ADS_REQUESTS)))
    limit = max(1, min(2000, _int(intent.get("limit"), DEFAULT_ADS_LIMIT)))
    simulate = bool(intent.get("simulate_auction", True))

    plan = [
        PlannedCall("list_ad_models", {}, reason="先确认六个广告模型与权重状态"),
        PlannedCall(
            "predict_ad_ctr",
            {"model": model, "split": split, "limit": limit},
            reason="真实 CTR 推理",
        ),
        PlannedCall(
            "rank_ads",
            {
                "model": model,
                "split": "train",
                "requests": requests,
                "min_candidates": 1,
                "top_k": 2,
            },
            reason="请求内广告排序",
        ),
    ]
    if simulate:
        plan.append(
            PlannedCall(
                "simulate_auction",
                {
                    "model": model,
                    "split": "train",
                    "requests": requests,
                    "min_candidates": 1,
                    "top_k": 2,
                },
                reason="二价竞价与预算扣减（模拟金额）",
            )
        )
    plan.append(PlannedCall("get_budget_state", {}, reason="读取最近一次预算状态"))
    internal_cap = _spend_dispatch(scope, business="ads", min_internal_tools=len(plan))

    context = build_context(
        experiment_dir=experiment_dir,
        max_tool_calls=min(internal_cap, 8),
        max_seconds=float(min(150.0, scope.remaining("seconds"))),
    )
    status = "failed"
    error: str | None = None
    summary: dict[str, Any] = {}
    try:
        summary = run_agent(
            context,
            task=task,
            experiment_dir=experiment_dir,
            agent_config=AgentConfig(
                max_tool_calls=min(internal_cap, 8),
                max_llm_requests=1,
                max_seconds=float(min(150.0, scope.remaining("seconds"))),
                default_model=model,
            ),
            plan=plan,
            planner="local",
        )
        status = {
            "verified": "success",
            "partial": "partial",
            "failed": "failed",
            "blocked": "blocked",
        }.get(str(summary.get("status")), "failed")
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    tool_results = list(summary.get("tool_results") or [])
    internal_calls = len(tool_results)
    if internal_calls:
        try:
            scope.spend("tool", internal_calls)
        except BudgetExceeded as budget_error:  # pragma: no cover
            error = (error + "；" if error else "") + budget_error.reason
            status = "partial"
    llm_requests = _int(summary.get("llm_requests"), 0)
    if llm_requests:
        try:
            scope.spend("llm", llm_requests)
        except BudgetExceeded as budget_error:  # pragma: no cover
            error = (error + "；" if error else "") + budget_error.reason
            status = "partial"

    metrics = summary.get("metrics") or {}
    weight = {}
    try:
        from recsys.ads.registry import model_status

        weight = model_status(model)
    except Exception:  # noqa: BLE001
        weight = {}
    artifacts = _artifact_dict(
        {
            "report": experiment_dir / "report.md",
            "trace": experiment_dir / "trace.json",
            "metrics": experiment_dir / "metrics.json",
            "run_meta": experiment_dir / "run_meta.json",
        }
    )
    all_ok = bool(tool_results) and all(bool(item.get("ok")) for item in tool_results)
    validated = status == "success" and all_ok
    warnings = ["simulated_auction_money", "interface_test_weights"]
    if weight.get("weight_kind") == "default" or weight.get("is_trained"):
        warnings = ["simulated_auction_money"]
    ad_budget = None
    auction_metrics = metrics.get("simulate_auction") or {}
    auction_summaries = auction_metrics.get("summaries") or []
    if auction_summaries:
        ad_budget = (auction_summaries[0].get("summary") or {}).get("budget")
    predicted_weight = ((metrics.get("predict_ad_ctr") or {}).get("weight") or {})
    used_artifacts = {
        "weights": {
            "path": predicted_weight.get("checkpoint") or weight.get("checkpoint"),
            "weight_kind": predicted_weight.get("weight_kind") or weight.get("weight_kind"),
            "is_trained": predicted_weight.get("is_trained"),
        },
        "source": "ads.agent metrics#predict_ad_ctr.weight",
    }
    elapsed = round(_now() - started, 2)
    record = _record(
        store=store,
        task_id=task_id,
        business="ads",
        status=status,
        config={
            "task": task,
            "step": step,
            "model": model,
            "split": split,
            "requests": requests,
            "limit": limit,
            "simulate_auction": simulate,
        },
        data={
            "dataset": "kuasar_small",
            "split": split,
            "split_role": split,
            "source": "KuaiSAR ad exposure/click; auction money simulated",
        },
        model={
            "models": [model],
            "weights_source": weight.get("weight_kind") or "interface_test",
            "checkpoint": weight.get("checkpoint"),
            "is_trained": weight.get("is_trained"),
            "used_artifacts": used_artifacts,
        },
        result={
            "status": status,
            "agent_status": summary.get("status"),
            "metrics": metrics,
            "final_answer": summary.get("final_answer"),
            "tools_called": summary.get("tools_called"),
            "error": error,
        },
        artifacts=artifacts,
        scope=scope,
        driver="local",
        validated=validated,
        warnings=warnings,
        simulation={
            "auction_money": True,
            "conversion_labels": "synthetic_fixture",
            "bid_source": "simulated_config",
            "ad_budget": ad_budget,
            "note": "KuaiSAR 没有真实出价/预算/转化标签；竞价金额是多任务接口模拟。",
        },
        checks={"tools_ok": all_ok, "tool_calls": internal_calls},
        notes="广告业务薄适配：复用 ads.agent + ads.tools，模拟金额单独标记。",
    )
    return {
        "ok": status == "success" and error is None,
        "business": "ads",
        "status": status,
        "agent_status": summary.get("status"),
        "driver": "local",
        "metrics": metrics,
        "tools_called": list(summary.get("tools_called") or []),
        "artifacts": artifacts,
        "used_artifacts": used_artifacts,
        "record_id": record.get("record_id"),
        "warnings": warnings,
        "error": error,
        "budget": scope.snapshot(),
        "elapsed_s": elapsed,
        "simulation": {
            "auction_money": True,
            "conversion_labels": "synthetic_fixture",
            "ad_budget": ad_budget,
        },
        "final_answer": summary.get("final_answer"),
    }


# ---------------------------------------------------------------------- 分发


def run_business(
    business: str,
    task: str,
    *,
    task_id: str,
    scope: Any,
    experiment_dir: Path | str,
    intent: Mapping[str, Any] | None = None,
    store: SharedExperimentStore | Path | str | None = None,
) -> dict[str, Any]:
    if business == "recommendation":
        return run_recommendation(
            task, task_id=task_id, scope=scope, experiment_dir=experiment_dir, intent=intent, store=store
        )
    if business == "search":
        return run_search(
            task, task_id=task_id, scope=scope, experiment_dir=experiment_dir, intent=intent, store=store
        )
    if business == "ads":
        return run_ads(
            task, task_id=task_id, scope=scope, experiment_dir=experiment_dir, intent=intent, store=store
        )
    raise ValueError(f"未知业务：{business}")


__all__ = [
    "SEARCH_CONFIG",
    "SEARCH_MODES",
    "run_ads",
    "run_business",
    "run_recommendation",
    "run_search",
]
