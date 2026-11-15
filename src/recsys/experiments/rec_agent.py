"""Rec Agent：在 LangGraph 上调用真实训练 / 召回 / 排序 / 重排 / 评估工具。

设计原则：

* 有 LLM 时，模型只能通过 ``tool_calls`` 选择注册表里的工具；程序校验参数、
  预算和重复调用，再把真实结果回传。
* 没有 LLM（或 ``--force-local``）时，用确定性计划器从同样的工具注册表执行，
  保证离线可复现；报告里明确标注 ``planner=local``。
* 任何工具失败都以 ``ok=false`` + 错误原因返回，Agent 不把失败写成成功。

常用入口：

    python -m recsys.experiments.rec_agent --self-check
    python -m recsys.experiments.rec_agent --task "用双塔和 MIND 召回，LightGBM 排序并评估" --users 16
    python -m recsys.experiments.rec_agent --train lightgcn --recall two_tower,mind,lightgcn --ranker lightgbm_ranker
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from recsys.experiments.llm_tool_call import (
    host_only,
    llm_config,
    load_dotenv,
    post_chat,
)
from recsys.experiments.rec_tools import (
    RANKER_MODELS,
    RECALL_MODELS,
    TRAINABLE_MODELS,
    ToolBudget,
    ToolBudgetExceeded,
    ToolCallError,
    ToolContext,
    ToolRegistry,
    build_registry,
    resolve_checkpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENT = PROJECT_ROOT / "experiments" / "2026-11-09-kuasar_small-rec-agent"


def run_research(task: str, **kwargs) -> dict:
    """Rec Agent's code-editing research, sharing the master's budget/memory."""
    from recsys.platform.autoresearch import run_research as research
    return research(task, business="recommendation", **kwargs)


# --------------------------------------------------------------------------- 计划


@dataclass(frozen=True)
class PlannedCall:
    tool: str
    arguments: dict
    reason: str = ""


@dataclass
class AgentSettings:
    dataset: str = "kuasar_small"
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    split: str = "validation"
    users: int = 32
    seed: int = 42
    device: str = "cpu"
    max_tool_calls: int = 10
    max_train_calls: int = 2
    max_seconds: float = 600.0
    max_epochs: int = 3
    max_batches: int = 20
    include_validation_in_test_history: bool = True


def _split_models(text: str | None) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def deterministic_plan(settings: AgentSettings, args: argparse.Namespace) -> list[PlannedCall]:
    """从 CLI 的显式参数生成工具序列；不使用任何模拟结果。"""
    calls: list[PlannedCall] = []
    train_models = _split_models(getattr(args, "train", None))
    for model in train_models:
        if model not in TRAINABLE_MODELS:
            raise ToolCallError(f"--train 不支持 {model!r}")
        calls.append(
            PlannedCall(
                "train_model",
                {
                    "model": model,
                    "epochs": int(getattr(args, "epochs", 1)),
                    "max_batches": int(getattr(args, "max_batches", 2)),
                },
                reason="CLI 指定先训练模型",
            )
        )
    recall_models = _split_models(getattr(args, "recall", None))
    ranker = getattr(args, "ranker", None) or "lightgbm_ranker"
    task = (getattr(args, "task", "") or "").lower()
    if recall_models:
        calls.append(
            PlannedCall(
                "run_pipeline",
                {
                    "models": recall_models,
                    "ranker": ranker,
                    "fusion": getattr(args, "fusion", "round_robin"),
                    "split": settings.split,
                    "k_recall": int(getattr(args, "k_recall", 100)),
                    "k_final": int(getattr(args, "k_final", 20)),
                    "category_cap": int(getattr(args, "category_cap", 5)),
                    "author_cap": int(getattr(args, "author_cap", 0)),
                    "include_validation_in_test_history": (
                        settings.include_validation_in_test_history
                    ),
                },
                reason="按指定召回与排序模型组合方案跑完整链路",
            )
        )
        return calls

    # 自由文本任务：按关键词选择工具，但不猜测不存在的模型。
    mentioned_recall = [model for model in RECALL_MODELS if model in task or model.replace("_", " ") in task]
    mentioned_rankers = [model for model in RANKER_MODELS if model in task]
    if "train" in task or "训练" in task:
        for model in mentioned_recall or mentioned_rankers:
            if model in TRAINABLE_MODELS:
                calls.append(
                    PlannedCall(
                        "train_model",
                        {"model": model, "epochs": 1, "max_batches": 2},
                        reason="任务要求训练该模型",
                    )
                )
    if "zoo" in task or "模型库" in task:
        calls.append(PlannedCall("query_model_zoo", {}, reason="任务要求查询 Model Zoo"))
    if "画像" in task or "profile" in task or "数据" in task:
        calls.append(PlannedCall("read_data_profile", {}, reason="先读取数据画像"))
    if not calls and not (mentioned_recall or mentioned_rankers):
        calls.append(PlannedCall("read_data_profile", {}, reason="默认先读取数据画像"))
        calls.append(PlannedCall("query_model_zoo", {}, reason="默认先查询 Model Zoo"))
    if mentioned_recall or mentioned_rankers or any(word in task for word in ("推荐", "recall", "召回", "评估", "evaluate")):
        calls.append(
            PlannedCall(
                "run_pipeline",
                {
                    "models": mentioned_recall or ["two_tower", "mind"],
                    "ranker": mentioned_rankers[0] if mentioned_rankers else "lightgbm_ranker",
                    "fusion": "round_robin",
                    "split": settings.split,
                    "k_recall": 100,
                    "k_final": 20,
                    "include_validation_in_test_history": (
                        settings.include_validation_in_test_history
                    ),
                },
                reason="执行一次可验证的完整推荐链路",
            )
        )
    if not calls:
        calls.append(
            PlannedCall(
                "run_pipeline",
                {
                    "models": ["two_tower", "mind"],
                    "ranker": "lightgbm_ranker",
                    "split": settings.split,
                },
                reason="默认组合：双塔 + MIND 召回，LightGBM 排序",
            )
        )
    return calls


# --------------------------------------------------------------------------- LangGraph 状态


class AgentState(TypedDict, total=False):
    status: str
    stage: str
    nodes: list[str]
    events: list[dict]
    task: str
    settings: Any
    context: Any
    registry: Any
    plan: list[dict]
    plan_index: int
    pending: dict | None
    pending_queue: list[dict]
    messages: list[dict]
    tool_results: list[dict]
    errors: list[str]
    final_answer: str
    summary: dict
    started: float
    deadline: float
    llm_used: bool
    llm_enabled: bool
    llm_requests: int
    llm_config: dict | None
    chat_fn: Any


ChatFn = Any


def _event(state: AgentState, payload: dict) -> None:
    state.setdefault("events", []).append(payload)
    context = state.get("context")
    if context is not None:
        context.log(payload)


def _enter(state: AgentState, node: str) -> AgentState:
    state = dict(state)  # type: ignore[assignment]
    state.setdefault("nodes", []).append(node)
    state["stage"] = node
    _event(state, {"event": "node_start", "node": node})
    return state  # type: ignore[return-value]


def _leave(state: AgentState, node: str) -> AgentState:
    _event(state, {"event": "node_end", "node": node, "status": state.get("status")})
    return state


def _remaining(state: AgentState) -> float:
    return float(state.get("deadline", 0)) - time.perf_counter()


# --------------------------------------------------------------------------- LLM 规划


SYSTEM_PROMPT = """你是推荐系统 Agent。你必须通过提供的工具完成真实任务：
1. 只能调用已注册工具，一次只调用一个；参数必须来自工具 schema。
2. 先读数据画像 / Model Zoo / 工件，再决定训练、召回、排序、重排或评估。
3. 不要编造指标；工具返回 ok=false 时必须把它当作失败并报告具体原因。
4. 训练预算有限：默认 1 epoch、少量 batch；不要请求超出预算的参数。
5. 完成时停止调用工具并输出简短中文结论（包含真实指标或失败原因）。
6. 召回和排序沿用任务的历史设置。可以省略 include_validation_in_test_history 参数。
"""


def _context_message(context: ToolContext) -> dict:
    data = context.data()
    return {
        "role": "user",
        "content": json.dumps(
            {
                "task": "见第一条用户消息",
                "dataset": context.dataset,
                "split": context.split,
                "include_validation_in_test_history": context.history_setting({}),
                "users": context.users,
                "counts": {
                    "users": data.num_users,
                    "items": data.num_items,
                    "train": len(data.train_interactions),
                    "validation": len(data.validation_interactions),
                },
                "available_checkpoints": {
                    model: str(path)
                    for model in RECALL_MODELS
                    if (path := resolve_checkpoint(context, model))
                },
                "budget": context.budget.as_dict(),
            },
            ensure_ascii=False,
        ),
    }


def _extract_tool_calls(response: dict) -> list[dict]:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message", {})
    calls = message.get("tool_calls") or []
    extracted: list[dict] = []
    for call in calls:
        function = call.get("function") or {}
        if (call.get("type") or "function") != "function":
            raise ToolCallError(f"不支持的 tool_call 类型 {call.get('type')}")
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError as error:
            raise ToolCallError(f"tool_call 参数不是合法 JSON：{error}") from error
        extracted.append(
            {
                "id": call.get("id") or f"call_{int(time.time() * 1000)}_{len(extracted)}",
                "name": function.get("name") or "",
                "arguments": arguments,
            }
        )
    return extracted


def _tool_message(call_id: str, result: dict) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(result, ensure_ascii=False)[:8000],
    }


def _assistant_tool_message(calls: list[dict], content: str = "") -> dict:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                },
            }
            for call in calls
        ],
    }


# --------------------------------------------------------------------------- 节点


def node_plan(
    state: AgentState,
) -> AgentState:
    state = _enter(state, "plan")
    if _remaining(state) <= 0:
        state["status"] = "partial"
        state["pending"] = None
        state["final_answer"] = "时间预算耗尽，停止执行。"
        _event(state, {"event": "stop", "reason": "deadline"})
        return _leave(state, "plan")
    context: ToolContext = state["context"]
    queued = list(state.get("pending_queue") or [])
    plan = state.get("plan") or []
    index = int(state.get("plan_index", 0))
    if context.budget.calls >= context.budget.max_tool_calls and (
        queued or index < len(plan) or state.get("llm_enabled")
    ):
        state["pending"] = None
        state["final_answer"] = "工具调用次数预算耗尽，停止执行。"
        state["status"] = "partial"
        return _leave(state, "plan")

    # 同一轮 LLM 返回多个 tool_calls 时，按顺序逐个执行。
    if queued:
        state["pending"] = queued.pop(0)
        state["pending_queue"] = queued
        _event(state, {"event": "plan_next", "source": "queue", **state["pending"]})
        return _leave(state, "plan")

    # 确定性计划优先（可复现的 CLI 序列）。
    if index < len(plan):
        state["plan_index"] = index + 1
        state["pending"] = plan[index]
        _event(state, {"event": "plan_next", "source": "local", **plan[index]})
        return _leave(state, "plan")

    if not state.get("llm_enabled"):
        state["pending"] = None
        _event(state, {"event": "plan_finish", "source": "local"})
        return _leave(state, "plan")

    state["llm_used"] = True
    state["llm_requests"] = int(state.get("llm_requests", 0)) + 1
    config = state.get("llm_config")
    chat_fn: ChatFn = state.get("chat_fn") or post_chat
    messages = list(state.get("messages") or [])
    if not messages:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state.get("task", "")},
            _context_message(context),
        ]
    try:
        response = chat_fn(
            config,
            messages,
            tools=state["registry"].schemas(),
            timeout=max(1.0, min(60.0, _remaining(state))),
        )
    except Exception as error:  # noqa: BLE001
        state["status"] = "partial"
        state["pending"] = None
        state["final_answer"] = f"LLM 请求失败：{type(error).__name__}: {error}"
        _event(state, {"event": "llm_error", "error": state["final_answer"]})
        return _leave(state, "plan")

    content = ((response.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    try:
        calls = _extract_tool_calls(response)
    except ToolCallError as error:
        state["errors"].append(str(error))
        state["pending"] = None
        state["final_answer"] = f"LLM 工具调用不合法：{error}"
        state["status"] = "failed"
        return _leave(state, "plan")

    if not calls:
        state["pending"] = None
        state["final_answer"] = content or "LLM 选择结束（无工具调用）。"
        messages.append({"role": "assistant", "content": content})
        state["messages"] = messages
        _event(state, {"event": "plan_finish", "source": "llm"})
        return _leave(state, "plan")

    for call in calls:
        try:
            state["registry"].validate(call["name"], call["arguments"])
        except ToolCallError as error:
            state["errors"].append(str(error))
            state["pending"] = None
            state["final_answer"] = f"LLM 工具参数不合法：{error}"
            state["status"] = "failed"
            return _leave(state, "plan")

    pending_calls = [
        {
            "tool": call["name"],
            "arguments": call["arguments"],
            "call_id": call["id"],
            "reason": "LLM 选择",
        }
        for call in calls
    ]
    state["pending"] = pending_calls[0]
    state["pending_queue"] = pending_calls[1:]
    state["messages"] = messages + [_assistant_tool_message(calls, content)]
    _event(
        state,
        {
            "event": "plan_next",
            "source": "llm",
            "tool": pending_calls[0]["tool"],
            "arguments": pending_calls[0]["arguments"],
            "queued": len(pending_calls) - 1,
        },
    )
    return _leave(state, "plan")


def node_execute(state: AgentState) -> AgentState:
    state = _enter(state, "execute")
    pending = state["pending"]
    context: ToolContext = state["context"]
    registry: ToolRegistry = state["registry"]
    if not pending:
        return _leave(state, "execute")
    # 重复调用保护：完全相同的 tool+args 只允许一次。
    signature = json.dumps(
        {"tool": pending["tool"], "arguments": pending["arguments"]},
        sort_keys=True,
        ensure_ascii=False,
    )
    seen = {item.get("signature") for item in state.get("tool_results", [])}
    if signature in seen:
        state["status"] = "partial"
        state["errors"].append(f"拒绝重复调用：{pending['tool']} {pending['arguments']}")
        state["pending"] = None
        state["final_answer"] = f"拒绝重复调用 {pending['tool']}（相同参数已执行过）。"
        return _leave(state, "execute")

    try:
        result = registry.call(pending["tool"], pending["arguments"], context)
    except (ToolBudgetExceeded, ToolCallError) as error:
        state["status"] = "failed"
        state["errors"].append(f"{pending['tool']}: {error}")
        result = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        context.log({"event": "tool_rejected", "tool": pending["tool"], "error": str(error)})
    record = {
        "signature": signature,
        "tool": pending["tool"],
        "arguments": pending["arguments"],
        "reason": pending.get("reason", ""),
        "ok": bool(result.get("ok")),
        "result": result,
    }
    state.setdefault("tool_results", []).append(record)
    if not result.get("ok"):
        state.setdefault("errors", []).append(
            f"{pending['tool']}: {result.get('error', 'unknown error')}"
        )
    if state.get("llm_used") and pending.get("call_id"):
        messages = list(state.get("messages") or [])
        messages.append(_tool_message(pending["call_id"], result))
        state["messages"] = messages
    _event(
        state,
        {
            "event": "tool_result",
            "tool": pending["tool"],
            "ok": bool(result.get("ok")),
            "elapsed_s": next(
                (item.get("elapsed_s") for item in reversed(context.executions) if item["tool"] == pending["tool"]),
                None,
            ),
        },
    )
    state["pending"] = None
    return _leave(state, "execute")


def node_finish(state: AgentState) -> AgentState:
    state = _enter(state, "finish")
    results = state.get("tool_results", [])
    failed = [item for item in results if not item["ok"]]
    if failed and not any(item["ok"] for item in results):
        status = "failed"
    elif failed:
        status = "partial"
    else:
        status = "verified"
    if state.get("status") in {"failed", "partial"} and results:
        status = state["status"]
    elif state.get("status") == "failed" and not results:
        status = "failed"
    elif state.get("status") == "partial" and not results:
        status = "partial"
    state["status"] = status
    context: ToolContext = state["context"]
    metrics = {}
    artifacts = {}
    diagnostics = {}
    for item in results:
        result = item.get("result") or {}
        if result.get("metrics"):
            metrics[item["tool"]] = result["metrics"]
        if result.get("artifacts"):
            artifacts.update(result["artifacts"])
        if result.get("diagnostics"):
            diagnostics[item["tool"]] = result["diagnostics"]
    summary = {
        "status": status,
        "task": state.get("task", ""),
        "dataset": context.dataset,
        "split": context.split,
        "planner": "llm" if state.get("llm_used") else "local",
        "llm_used": bool(state.get("llm_used")),
        "llm_available": bool(state.get("llm_config")),
        "llm_enabled": bool(state.get("llm_enabled")),
        "llm_requests": int(state.get("llm_requests", 0)),
        "tools_called": [item["tool"] for item in results],
        "tool_results": [
            {
                "tool": item["tool"],
                "arguments": item["arguments"],
                "ok": item["ok"],
                "result": item["result"],
            }
            for item in results
        ],
        "errors": state.get("errors", []),
        "metrics": metrics,
        "diagnostics": diagnostics,
        "artifacts": artifacts,
        "trained_checkpoints": dict(context.state.get("trained_checkpoints", {})),
        "budget": context.budget.as_dict(),
        "executions": context.executions,
        "final_answer": state.get("final_answer")
        or ("工具执行完成，结论见 metrics / artifacts。" if status != "failed" else "工具执行失败，见 errors。"),
        "elapsed_s": round(time.perf_counter() - float(state.get("started", time.perf_counter())), 2),
    }
    state["summary"] = summary
    if not state.get("final_answer"):
        state["final_answer"] = summary["final_answer"]
    _event(state, {"event": "finish", "status": status, "tools": summary["tools_called"]})
    return _leave(state, "finish")


# --------------------------------------------------------------------------- 图


def _route_after_plan(state: AgentState) -> str:
    return "execute" if state.get("pending") else "finish"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("plan", node_plan)
    graph.add_node("execute", node_execute)
    graph.add_node("finish", node_finish)
    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", _route_after_plan, {"execute": "execute", "finish": "finish"})
    graph.add_edge("execute", "plan")
    graph.add_edge("finish", END)
    return graph.compile()


# --------------------------------------------------------------------------- 运行


def run_agent(
    settings: AgentSettings,
    *,
    task: str,
    plan: list[PlannedCall] | None = None,
    experiment_dir: Path = DEFAULT_EXPERIMENT,
    force_local: bool = False,
    chat_fn: ChatFn | None = None,
) -> dict:
    load_dotenv()
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "events.jsonl").write_text("", encoding="utf-8")
    config = None if force_local else llm_config()
    if chat_fn is not None:
        config = config or {"base_url": "http://local", "api_key": "", "model": "scripted"}
    llm_available = config is not None

    context = ToolContext(
        dataset=settings.dataset,
        data_dir=settings.data_dir,
        artifacts_dir=settings.artifacts_dir,
        experiment_dir=experiment_dir,
        split=settings.split,
        users=settings.users,
        seed=settings.seed,
        device=settings.device,
        budget=ToolBudget(
            max_tool_calls=settings.max_tool_calls,
            max_train_calls=settings.max_train_calls,
            max_seconds=settings.max_seconds,
            max_epochs=settings.max_epochs,
            max_batches=settings.max_batches,
        ),
    )
    context.state["include_validation_in_test_history"] = (
        settings.include_validation_in_test_history
    )
    registry = build_registry()
    planned = plan if plan is not None else deterministic_plan(settings, argparse.Namespace())
    state: AgentState = {
        "status": "running",
        "stage": "start",
        "nodes": [],
        "events": [],
        "task": task,
        "settings": settings,
        "context": context,
        "registry": registry,
        "plan": [
            {"tool": call.tool, "arguments": call.arguments, "reason": call.reason}
            for call in planned
        ],
        "plan_index": 0,
        "pending": None,
        "pending_queue": [],
        "messages": [],
        "tool_results": [],
        "errors": [],
        "started": time.perf_counter(),
        "deadline": time.perf_counter() + settings.max_seconds,
        "llm_used": False,
        "llm_enabled": llm_available and not plan,
        "llm_requests": 0,
        "llm_config": config,
        "chat_fn": chat_fn,
    }
    context.log(
        {
            "event": "start",
            "task": task,
            "dataset": settings.dataset,
            "planner": "llm" if llm_available and not plan else "local",
            "llm_available": llm_available,
            "llm_host": host_only(config["base_url"]) if config and config.get("base_url") else None,
            "plan": state["plan"],
        }
    )
    graph = build_graph()
    final_state = graph.invoke(state)
    summary = final_state["summary"]
    (experiment_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (experiment_dir / "tool_results.json").write_text(
        json.dumps(summary["tool_results"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


# --------------------------------------------------------------------------- 自检


def run_self_check(experiment_dir: Path = DEFAULT_EXPERIMENT / "self_check") -> dict:
    """不依赖 LLM，验证工具能到达真实函数、返回真实结果并正确报告失败。"""
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    context = ToolContext(
        dataset="kuasar_small",
        users=4,
        experiment_dir=experiment_dir,
        budget=ToolBudget(max_tool_calls=12, max_train_calls=1, max_seconds=300),
    )
    registry = build_registry()
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    profile = registry.call("read_data_profile", {}, context)
    record(
        "read_data_profile_real",
        profile.get("ok") and profile["counts"]["users"] > 0,
        {key: profile[key] for key in ("dataset", "counts", "ranking_samples_available")},
    )
    zoo = registry.call("query_model_zoo", {}, context)
    record(
        "query_model_zoo_real",
        zoo.get("ok") and zoo["count"] >= 20,
        {"count": zoo["count"], "categories": zoo["categories"]},
    )
    artifacts = registry.call("list_artifacts", {}, context)
    record("list_artifacts_real", artifacts.get("ok"), {"files": len(artifacts.get("files", []))})

    try:
        registry.validate("train_model", {"model": "not_a_model"})
        record("reject_unknown_model", False, "未拒绝未知模型")
    except ToolCallError as error:
        record("reject_unknown_model", True, str(error))

    recall = registry.call("recall_models", {"models": ["popularity"], "k": 20}, context)
    record(
        "recall_real_inference",
        recall.get("ok") and recall["channel_summary"]["popularity"]["candidates"] > 0,
        recall.get("channel_summary"),
    )

    # 失败路径：把 artifacts 指到空目录，DIN 必然缺 checkpoint，工具必须返回 ok=false。
    empty_context = ToolContext(
        dataset="kuasar_small",
        users=4,
        artifacts_dir=experiment_dir / "empty_artifacts",
        experiment_dir=experiment_dir / "failure",
        budget=ToolBudget(max_tool_calls=4),
    )
    empty_context.state["channels"] = context.state["channels"]
    missing_rank = registry.call("rank_candidates", {"model": "din"}, empty_context)
    record(
        "missing_checkpoint_reported",
        missing_rank.get("ok") is False and "checkpoint" in missing_rank.get("error", ""),
        missing_rank,
    )

    # 真实端到端：popularity 单路召回 + LightGBM 排序（checkpoint 已存在时）。
    pipeline = registry.call(
        "run_pipeline",
        {
            "models": ["popularity"],
            "ranker": "lightgbm_ranker",
            "split": "validation",
            "k_recall": 20,
            "k_final": 10,
            "output_name": "self_check",
        },
        context,
    )
    record(
        "run_pipeline_real",
        pipeline.get("ok") and "rerank" in pipeline.get("metrics", {}),
        {
            "metrics": pipeline.get("metrics", {}).get("rerank"),
            "artifacts": pipeline.get("artifacts"),
        },
    )

    summary = {
        "passed": all(check["ok"] for check in checks),
        "checks": checks,
        "planner": "local (deterministic); LLM planner not used in self-check",
        "note": "自检不跑长训练；训练工具的真实执行由 --train din / --train matrix_factorization 单独验证。",
        "experiment_dir": str(experiment_dir),
    }
    (experiment_dir / "self_check.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


# --------------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--task", default="")
    parser.add_argument("--dataset", default="kuasar_small")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--users", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train", default=None, help="逗号分隔的模型 id，先训练")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--recall", default=None, help="逗号分隔的召回模型 id")
    parser.add_argument("--ranker", choices=list(RANKER_MODELS), default="lightgbm_ranker")
    parser.add_argument("--fusion", choices=["round_robin", "rrf", "quota"], default="round_robin")
    parser.add_argument("--k-recall", type=int, default=100)
    parser.add_argument("--k-final", type=int, default=20)
    parser.add_argument("--category-cap", type=int, default=5)
    parser.add_argument("--author-cap", type=int, default=0)
    parser.add_argument("--max-tool-calls", type=int, default=10)
    parser.add_argument("--max-train-calls", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument(
        "--no-validation-in-test-history",
        action="store_false",
        dest="include_validation_in_test_history",
        help="test 时只使用训练历史（默认 True：训练+验证）",
    )
    parser.set_defaults(include_validation_in_test_history=True)
    parser.add_argument("--force-local", action="store_true", help="忽略 LLM 配置，使用确定性计划")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="使用真实 LLM 作为计划器（需 .env 里配置 DSH_AGENT_LLM_*）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_check:
        summary = run_self_check(args.experiment_dir / "self_check")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if summary["passed"] else 1
    settings = AgentSettings(
        dataset=args.dataset,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        split=args.split,
        users=args.users,
        seed=args.seed,
        device=args.device,
        max_tool_calls=args.max_tool_calls,
        max_train_calls=args.max_train_calls,
        max_seconds=args.max_seconds,
        include_validation_in_test_history=args.include_validation_in_test_history,
    )
    plan = deterministic_plan(settings, args)
    if args.llm:
        load_dotenv()
        if llm_config() is not None and not args.force_local:
            plan = []  # 交给 LLM 规划；工具结果会回传给模型
        else:
            print("[warn] 未配置 LLM 或指定 --force-local，回退确定性计划", file=sys.stderr)
    task = args.task or "按 CLI 参数执行推荐任务"
    summary = run_agent(
        settings,
        task=task,
        plan=plan,
        experiment_dir=args.experiment_dir,
        force_local=args.force_local,
    )
    print(json.dumps({key: summary[key] for key in ("status", "planner", "tools_called", "metrics", "artifacts", "errors")}, indent=2, ensure_ascii=False))
    return 0 if summary["status"] in {"verified", "partial"} else 1


if __name__ == "__main__":
    sys.exit(main())
