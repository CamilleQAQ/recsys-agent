"""Ads Agent：LangGraph 上的广告预测 / 排序 / 竞价 / 预算工具调用。

复用 YieldAgent 的结构（工具调用方式、工具与业务分离、工具调用与结果记录）：

* 工具注册表与薄工具函数见 ``recsys.ads.tools``（对应
  ``yieldagent/agents/console/tools.py`` 的写法：工具只描述 schema 并调度业务）；
* 业务实现集中在 ``recsys.ads.service``（对应 YieldAgent 的 Connector，
  工具不直接碰模型/竞价细节）；
* 工具结果记录沿用 YieldAgent ``runtime.py`` 的处理方式：工具返回的结构化结果
  先在轨迹里记 ``tool_call`` / ``tool_result``，模型侧消息按同一 ``tool_call_id``
  回传；``coerce_payload`` / ``summarize_payload`` 由 YieldAgent 的
  ``_coerce`` / ``_summarize`` 改写而来（本项目没有安装 LangChain，因此不用它的
  ``@tool`` / ``create_react_agent``，改用项目已有的 LangGraph ``StateGraph`` +
  OpenAI 兼容 HTTP 客户端）。

两种驱动方式，报告里分别标注：

* ``planner=llm``：模型只通过原生 ``tool_calls`` 选择工具，程序校验参数、预算与
  重复调用，再把真实结果用同一 ``tool_call_id`` 回传；
* ``planner=local``：无 LLM 或 ``--planner local`` 时用确定性计划执行同一批工具，
  用于离线测试与可复现轨迹。

入口：

    python -m recsys.ads.agent --task "用 din 模型做 CTR 预测、排序、竞价和预算模拟"
    python -m recsys.ads.agent --planner local
    python -m recsys.ads.agent --self-check
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from recsys.ads import AD_MODEL_NAMES
from recsys.ads.config import AgentConfig, load_agent_config
from recsys.ads.models import AdModelUnavailable
from recsys.ads.service import list_models
from recsys.ads.tools import (
    AdsToolContext,
    build_context,
    build_registry,
)
from recsys.ads.tools import SPLITS
from recsys.experiments.llm_tool_call import (
    host_only,
    llm_config,
    load_dotenv,
    post_chat,
)
from recsys.experiments.rec_tools import ToolBudgetExceeded, ToolCallError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "2026-11-09-kuasar_small-ads-phase3"
DEFAULT_SELF_CHECK = PROJECT_ROOT / "experiments" / "2026-11-09-kuasar_small-ads-phase3" / "self_check.json"


def run_research(task: str, **kwargs) -> dict:
    """Ads Agent's code-editing research with existing prediction features."""
    from recsys.platform.autoresearch import run_research as research
    return research(task, business="ads", **kwargs)

SYSTEM_PROMPT = """你是广告系统 Agent。你必须通过工具完成真实任务，规则如下：
1. 只能调用已注册工具，一次只调用一个；参数必须符合工具 schema，禁止并行多个 tool_calls。
2. 六个广告模型是 deepfm / din / dien / dcnv2 / mmoe / ple。status=interface_test 表示只有
   固定初始化权重：只能证明接口可推理，绝不能说成已训练、已收敛或有效果。
3. CTR 预测用 predict_ad_ctr；广告排序用 rank_ads；竞价与预算扣减用 simulate_auction。
4. KuaiSAR 没有真实转化标签：不要用 click 冒充转化。CVR 只能通过 check_label_availability
   和 predict_multitask 说明其 synthetic 性质。
5. 预算不足时工具会返回 no_winner_reason=budget_exhausted_*；如实报告，不要编造赢家或扣费。
6. 数据规模小、权重未训练，指标只能作为接口检查；最多发起 4 次工具调用，不要重复相同参数，
   也不要只用文字回答而不调用工具。
7. 完成时停止调用工具，用中文简短总结真实数字。
"""


# --------------------------------------------------------------------------- YieldAgent 结果处理改写


def coerce_payload(content: Any) -> Any:
    """把工具消息恢复成对象（改写自 YieldAgent ``runtime._coerce``）。

    兼容 JSON（双引号）与 Python repr（单引号）两种字符串结果。
    """
    if isinstance(content, str):
        for parse in (json.loads, ast.literal_eval):
            try:
                return parse(content)
            except (ValueError, SyntaxError, TypeError):
                continue
        return content
    return content


def summarize_payload(content: Any, limit: int = 300) -> str:
    """工具结果摘要（改写自 YieldAgent ``runtime._summarize``）。"""
    text = content if isinstance(content, str) else json.dumps(content, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "…"


# --------------------------------------------------------------------------- 计划


@dataclass(frozen=True)
class PlannedCall:
    tool: str
    arguments: dict
    reason: str = ""


def detect_model(task: str) -> str | None:
    """从任务文本里找模型名（按首次出现位置）；找不到返回 None。"""
    lowered = task.lower()
    best: tuple[int, str] | None = None
    for name in AD_MODEL_NAMES:
        position = lowered.find(name)
        if position < 0:
            continue
        before = lowered[position - 1] if position else " "
        after_position = position + len(name)
        after = lowered[after_position] if after_position < len(lowered) else " "
        # 只把 ASCII 字母数字视为单词边界：中文紧邻模型名也算合法（“用ple和din对比”）。
        if (before.isascii() and before.isalnum()) or (after.isascii() and after.isalnum()):
            continue
        if best is None or position < best[0]:
            best = (position, name)
    return best[1] if best else None


def build_local_plan(
    task: str, agent: AgentConfig, *, model: str | None = None
) -> list[PlannedCall]:
    """确定性计划：同一批真实工具，覆盖预测、排序、竞价、预算不足。"""
    chosen = model or detect_model(task) or agent.default_model
    auction_split = agent.auction_split
    return [
        PlannedCall("list_ad_models", {}, reason="先了解六个模型与权重状态"),
        PlannedCall(
            "predict_ad_ctr",
            {"model": chosen, "split": agent.default_split, "limit": 500},
            reason="CTR 预测（真实推理）",
        ),
        PlannedCall(
            "check_label_availability", {}, reason="核对真实转化标签缺口"
        ),
        PlannedCall(
            "rank_ads",
            {
                "model": chosen,
                "split": auction_split,
                "requests": agent.default_requests,
                "min_candidates": agent.auction_min_candidates,
                "top_k": agent.top_k,
            },
            reason="多候选请求内的广告排序",
        ),
        PlannedCall(
            "simulate_auction",
            {
                "model": chosen,
                "split": auction_split,
                "requests": agent.default_requests,
                "min_candidates": agent.auction_min_candidates,
                "top_k": agent.top_k,
            },
            reason="二价竞价与预算扣减",
        ),
        PlannedCall("get_budget_state", {}, reason="读取正常预算剩余"),
        PlannedCall(
            "simulate_auction",
            {
                "model": chosen,
                "split": auction_split,
                "requests": agent.default_requests,
                "min_candidates": agent.auction_min_candidates,
                "top_k": agent.top_k,
                "total_budget": 0.0,
                "per_advertiser_budget": 0.0,
            },
            reason="验证预算不足时的行为",
        ),
    ]


# --------------------------------------------------------------------------- 状态


class AgentState(TypedDict, total=False):
    status: str
    stage: str
    nodes: list[str]
    events: list[dict]
    task: str
    agent_config: Any
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


# --------------------------------------------------------------------------- LLM 消息


def _context_message(context: AdsToolContext, agent: AgentConfig) -> dict:
    runtime = context.runtime
    schema = runtime.ad_schema()
    models = [
        {
            "name": item["name"],
            "task": item["task"],
            "needs_history": item["needs_history"],
            "status": item["status"],
            "usable": item["usable"],
        }
        for item in list_models(runtime)["models"]
    ]
    splits = {
        split: {
            "ad_candidates": schema["splits"][split]["ad_candidates"],
            "requests": schema["splits"][split]["requests_with_ads"],
            "requests_with_2plus_ads": schema["splits"][split]["requests_with_2plus_ads"],
            "click_rate": schema["splits"][split]["click_rate"],
        }
        for split in ("train", "validation", "test")
    }
    return {
        "role": "user",
        "content": json.dumps(
            {
                "dataset": runtime.dataset,
                "ad_samples": splits,
                "models": models,
                "auction": context.config.auction.as_dict(),
                "budget": context.config.budget.as_dict(),
                "budget_limits": {
                    "max_tool_calls": context.budget.max_tool_calls,
                    "max_llm_requests": agent.max_llm_requests,
                    "max_seconds": context.budget.max_seconds,
                },
                "label_gap": {
                    "click": "真实标签",
                    "conversion": "缺失；CVR 只在 synthetic 模拟样本上验证",
                },
                "data_scale_warning": (
                    "kuasar_small 只有 2,000 用户；test 仅 43 个广告候选、41 个请求，"
                    "多数请求只有 1 个广告。排序/竞价演示用 train 的多候选请求（278 个）。"
                ),
                "instruction": (
                    "直接调用工具，不要先长篇解释；最多 4 次工具调用。至少完成一次 "
                    "predict_ad_ctr、一次 rank_ads、一次 simulate_auction（split=train, "
                    "min_candidates=2），再用中文总结真实数字与标签限制。"
                ),
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


def _tool_message(call_id: str, result: dict) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(result, ensure_ascii=False)[:8000],
    }


# --------------------------------------------------------------------------- 节点


def node_prepare(state: AgentState) -> AgentState:
    state = _enter(state, "prepare")
    context: AdsToolContext = state["context"]
    registry = state["registry"]
    state["llm_enabled"] = bool(state.get("llm_enabled"))
    _event(
        state,
        {
            "event": "prepare",
            "dataset": context.runtime.dataset,
            "tools": list(registry.names()),
            "planner": "llm" if state.get("llm_enabled") else "local",
            "llm_available": bool(state.get("llm_config")),
        },
    )
    return _leave(state, "prepare")


def node_plan(state: AgentState) -> AgentState:
    state = _enter(state, "plan")
    if _remaining(state) <= 0:
        state["status"] = "partial"
        state["pending"] = None
        state["final_answer"] = "时间预算耗尽，停止执行。"
        _event(state, {"event": "stop", "reason": "deadline"})
        return _leave(state, "plan")
    context: AdsToolContext = state["context"]
    queued = list(state.get("pending_queue") or [])
    plan = state.get("plan") or []
    index = int(state.get("plan_index", 0))
    if context.budget.calls >= context.budget.max_tool_calls and (
        queued or index < len(plan) or state.get("llm_enabled")
    ):
        state["status"] = "partial"
        state["pending"] = None
        state["final_answer"] = "工具调用次数预算耗尽，停止执行。"
        _event(state, {"event": "stop", "reason": "tool_budget"})
        return _leave(state, "plan")

    if queued:
        state["pending"] = queued.pop(0)
        state["pending_queue"] = queued
        _event(state, {"event": "plan_next", "source": "queue", **state["pending"]})
        return _leave(state, "plan")

    if index < len(plan):
        state["plan_index"] = index + 1
        state["pending"] = plan[index]
        _event(state, {"event": "plan_next", "source": "local", **plan[index]})
        return _leave(state, "plan")

    if not state.get("llm_enabled"):
        state["pending"] = None
        _event(state, {"event": "plan_finish", "source": "local"})
        return _leave(state, "plan")

    agent_config: AgentConfig = state["agent_config"]
    if int(state.get("llm_requests", 0)) >= agent_config.max_llm_requests:
        state["pending"] = None
        state["status"] = "partial"
        state["final_answer"] = "模型请求次数预算耗尽，停止执行。"
        _event(state, {"event": "stop", "reason": "llm_budget"})
        return _leave(state, "plan")

    state["llm_used"] = True
    state["llm_requests"] = int(state.get("llm_requests", 0)) + 1
    config = state.get("llm_config")
    chat_fn = state.get("chat_fn") or post_chat
    messages = list(state.get("messages") or [])
    if not messages:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state.get("task", "")},
            _context_message(context, agent_config),
        ]
    try:
        response = chat_fn(
            config,
            messages,
            tools=state["registry"].schemas(),
            timeout=max(1.0, min(60.0, _remaining(state))),
        )
    except Exception as error:  # noqa: BLE001 - 模型失败必须落盘
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
        state["status"] = "failed"
        state["pending"] = None
        state["final_answer"] = f"LLM 工具调用不合法：{error}"
        return _leave(state, "plan")

    if not calls:
        state["pending"] = None
        state["final_answer"] = content or "LLM 选择结束（无工具调用）。"
        state["messages"] = messages + [{"role": "assistant", "content": content}]
        _event(state, {"event": "plan_finish", "source": "llm"})
        return _leave(state, "plan")

    for call in calls:
        try:
            state["registry"].validate(call["name"], call["arguments"])
        except ToolCallError as error:
            state["errors"].append(str(error))
            state["status"] = "failed"
            state["pending"] = None
            state["final_answer"] = f"LLM 工具参数不合法：{error}"
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
    pending = state.get("pending")
    context: AdsToolContext = state["context"]
    registry = state["registry"]
    if not pending:
        return _leave(state, "execute")
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
        _event(state, {"event": "tool_duplicate", "tool": pending["tool"]})
        return _leave(state, "execute")

    started = time.perf_counter()
    call_id = pending.get("call_id")
    _event(
        state,
        {
            "event": "tool_call",
            "tool": pending["tool"],
            "arguments": pending["arguments"],
            "tool_call_id": call_id,
            "reason": pending.get("reason", ""),
        },
    )
    try:
        result = registry.call(pending["tool"], pending["arguments"], context)
    except (ToolBudgetExceeded, ToolCallError) as error:
        state["status"] = "failed"
        state["errors"].append(f"{pending['tool']}: {error}")
        result = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        context.log(
            {"event": "tool_rejected", "tool": pending["tool"], "error": str(error)}
        )
        _event(
            state,
            {
                "event": "tool_result",
                "tool": pending["tool"],
                "tool_call_id": call_id,
                "ok": False,
                "error": str(error),
            },
        )
    elapsed = round(time.perf_counter() - started, 3)
    record = {
        "signature": signature,
        "tool": pending["tool"],
        "arguments": pending["arguments"],
        "reason": pending.get("reason", ""),
        "tool_call_id": call_id,
        "ok": bool(result.get("ok")),
        "error": result.get("error"),
        "elapsed_s": elapsed,
        "result": result,
    }
    state.setdefault("tool_results", []).append(record)
    if not result.get("ok"):
        state.setdefault("errors", []).append(
            f"{pending['tool']}: {result.get('error', 'unknown error')}"
        )
    if state.get("llm_used") and call_id:
        messages = list(state.get("messages") or [])
        messages.append(_tool_message(call_id, result))
        state["messages"] = messages
    _event(
        state,
        {
            "event": "tool_result",
            "tool": pending["tool"],
            "tool_call_id": call_id,
            "ok": bool(result.get("ok")),
            "elapsed_s": elapsed,
            "result_summary": summarize_payload(result),
        },
    )
    state["pending"] = None
    return _leave(state, "execute")


def _metrics_from_results(results: list[dict]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for item in results:
        result = item.get("result") or {}
        tool = item["tool"]
        entry = metrics.setdefault(tool, {"calls": 0})
        entry["calls"] = int(entry.get("calls", 0)) + 1
        if result.get("metrics"):
            entry.setdefault("metrics", result["metrics"])
        if result.get("summary"):
            entry.setdefault("summaries", []).append(
                {
                    "arguments": item["arguments"],
                    "run_id": result.get("run_id"),
                    "summary": result["summary"],
                    "budget_overrides": result.get("budget_overrides"),
                }
            )
        if result.get("synthetic"):
            entry["synthetic"] = True
        if result.get("weight"):
            entry.setdefault(
                "weight",
                {
                    "weight_kind": result["weight"].get("weight_kind"),
                    "is_trained": result["weight"].get("is_trained"),
                    "checkpoint": result["weight"].get("checkpoint"),
                },
            )
    return metrics


def _compose_final(summary: dict) -> str:
    lines = ["Ads Agent 执行完成。"]
    model = summary.get("selected_model") or "default"
    lines.append(f"使用模型：{model}（权重状态见 list_ad_models / 工具返回的 weight 字段）。")
    for item in summary.get("tool_results", []):
        result = item.get("result") or {}
        tool = item["tool"]
        if tool == "predict_ad_ctr" and result.get("metrics"):
            metrics = result["metrics"]
            lines.append(
                f"CTR 预测：{result.get('split')} {result.get('rows')} 行，"
                f"AUC={float(metrics.get('AUC', 0.0)):.4f}，"
                f"LogLoss={float(metrics.get('LogLoss', 0.0)):.4f}，"
                f"实际点击率={float(metrics.get('ctr', 0.0)):.4f}。"
            )
        elif tool == "rank_ads" and result.get("summary"):
            ranking = result["summary"]
            lines.append(
                f"广告排序：{ranking.get('requests')} 个请求 / {ranking.get('candidates')} 个候选，"
                f"Top1 平均 pCTR={ranking.get('top1_ctr_mean')}。"
            )
        elif tool == "simulate_auction" and result.get("summary"):
            auction = result["summary"]
            budget = auction.get("budget", {})
            overrides = result.get("budget_overrides") or {}
            label = (
                "（预算不足验证）"
                if overrides.get("total") == 0 or overrides.get("per_advertiser") == 0
                else ""
            )
            lines.append(
                f"竞价模拟{label}：{auction.get('requests')} 个请求，赢家 {auction.get('winners')}，"
                f"点击 {auction.get('clicks')}，模拟扣费 {auction.get('spend')}，"
                f"总预算剩余 {budget.get('remaining')}。"
            )
            if auction.get("no_winner_requests"):
                lines.append(
                    f"其中 {auction.get('no_winner_requests')} 个请求未成交，原因："
                    f"{', '.join(str(reason) for reason in auction.get('no_winner_reasons', [])[:3])}。"
                )
        elif tool == "check_label_availability":
            conversion = result.get("conversion_label") or {}
            lines.append(
                "标签核对：真实 click 可用；转化标签"
                + ("可用。" if conversion.get("available") else "缺失，CVR 仅在 synthetic 模拟样本上验证。")
            )
        elif tool == "predict_multitask":
            lines.append(
                f"多任务预测（{result.get('model')}）：synthetic 样本 {result.get('rows')} 行，"
                "CTR/CVR 两路输出均可用，不代表真实 CVR 效果。"
            )
    lines.append("说明：本轮没有训练；interface_test 权重只用于接口验证。")
    return "\n".join(lines)


def node_finish(state: AgentState) -> AgentState:
    state = _enter(state, "finish")
    results = state.get("tool_results", [])
    failed = [item for item in results if not item["ok"]]
    if not results:
        status = "incomplete"
    elif failed and not any(item["ok"] for item in results):
        status = "failed"
    elif failed:
        status = "partial"
    else:
        status = "verified"
    if state.get("status") in {"failed", "partial", "blocked"}:
        status = state["status"]
    state["status"] = status

    context: AdsToolContext = state["context"]
    agent_config: AgentConfig = state["agent_config"]
    prediction = context.state.get("last_prediction") or {}
    selected_model = (
        prediction.get("model")
        or context.state.get("last_multitask", {}).get("model")
        or next(
            (
                item["arguments"].get("model")
                for item in results
                if isinstance(item.get("arguments"), dict) and item["arguments"].get("model")
            ),
            None,
        )
    )
    metrics = _metrics_from_results(results)
    final_answer = state.get("final_answer")
    if status == "incomplete":
        final_answer = "本次未执行任何广告工具，任务未完成。"
    if not final_answer:
        final_answer = _compose_final({"selected_model": selected_model, "tool_results": results})

    summary = {
        "status": status,
        "task": state.get("task", ""),
        "dataset": context.runtime.dataset,
        "planner": "llm" if state.get("llm_used") else "local",
        "llm_used": bool(state.get("llm_used")),
        "llm_available": bool(state.get("llm_config")),
        "llm_enabled": bool(state.get("llm_enabled")),
        "llm_requests": int(state.get("llm_requests", 0)),
        "max_llm_requests": agent_config.max_llm_requests,
        "selected_model": selected_model,
        "tools_called": [item["tool"] for item in results],
        "tool_results": [
            {
                "tool": item["tool"],
                "arguments": item["arguments"],
                "tool_call_id": item.get("tool_call_id"),
                "ok": item["ok"],
                "error": item.get("error"),
                "elapsed_s": item.get("elapsed_s"),
                "result": item["result"],
            }
            for item in results
        ],
        "errors": state.get("errors", []),
        "metrics": metrics,
        "budget": context.budget.as_dict(),
        "weights": {
            name: context.runtime.weight_info(name) for name in AD_MODEL_NAMES
        },
        "final_answer": final_answer,
        "elapsed_s": round(time.perf_counter() - float(state.get("started", time.perf_counter())), 2),
    }
    state["summary"] = summary
    state["final_answer"] = final_answer
    _event(state, {"event": "finish", "status": status, "tools": summary["tools_called"]})
    return _leave(state, "finish")


# --------------------------------------------------------------------------- 图


def _route_after_plan(state: AgentState) -> str:
    return "execute" if state.get("pending") else "finish"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("prepare", node_prepare)
    graph.add_node("plan", node_plan)
    graph.add_node("execute", node_execute)
    graph.add_node("finish", node_finish)
    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "plan")
    graph.add_conditional_edges(
        "plan", _route_after_plan, {"execute": "execute", "finish": "finish"}
    )
    graph.add_edge("execute", "plan")
    graph.add_edge("finish", END)
    return graph.compile()


# --------------------------------------------------------------------------- 运行与落盘


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_report(path: Path, summary: dict, trace: dict) -> None:
    lines = [
        "# Ads Agent 运行报告",
        "",
        f"- 状态：**{summary['status']}**",
        f"- 计划器：`{summary['planner']}`（LLM 请求 {summary['llm_requests']} 次）",
        f"- 任务：{summary['task']}",
        f"- 数据集：`{summary['dataset']}`",
        f"- 选中模型：`{summary['selected_model']}`",
        f"- 工具顺序：{' → '.join(summary['tools_called'])}",
        f"- 耗时：{summary['elapsed_s']} s；工具预算：{summary['budget']}",
        "",
        "## 工具调用轨迹",
        "",
        "| # | 工具 | tool_call_id | ok | 耗时(s) | 参数 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for index, item in enumerate(summary["tool_results"], start=1):
        lines.append(
            f"| {index} | {item['tool']} | {item.get('tool_call_id') or '-'} | "
            f"{item['ok']} | {item.get('elapsed_s')} | "
            f"`{json.dumps(item['arguments'], ensure_ascii=False)}` |"
        )
    lines += ["", "## 真实输出摘要", "", "```json", json.dumps(summary["metrics"], ensure_ascii=False, indent=2)[:6000], "```", ""]
    if summary["errors"]:
        lines += ["## 错误", "", *[f"- {error}" for error in summary["errors"]], ""]
    lines += ["## 最终回复", "", summary["final_answer"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _trace_payload(summary: dict) -> dict:
    return {
        "planner": summary["planner"],
        "status": summary["status"],
        "task": summary["task"],
        "selected_model": summary["selected_model"],
        "llm": {
            "used": summary["llm_used"],
            "available": summary["llm_available"],
            "requests": summary["llm_requests"],
            "max_requests": summary["max_llm_requests"],
        },
        "budget": summary["budget"],
        "steps": [
            {
                "index": index,
                "tool": item["tool"],
                "tool_call_id": item.get("tool_call_id"),
                "arguments": item["arguments"],
                "ok": item["ok"],
                "error": item.get("error"),
                "elapsed_s": item.get("elapsed_s"),
                "result_summary": summarize_payload(item["result"]),
            }
            for index, item in enumerate(summary["tool_results"], start=1)
        ],
        "errors": summary["errors"],
        "final_answer": summary["final_answer"],
    }


def run_agent(
    context: AdsToolContext,
    *,
    task: str,
    experiment_dir: Path,
    agent_config: AgentConfig | None = None,
    plan: list[PlannedCall] | None = None,
    planner: str = "auto",
    force_local: bool = False,
    chat_fn: Any = None,
) -> dict:
    load_dotenv()
    agent_config = agent_config or load_agent_config()
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    context.experiment_dir = experiment_dir
    (experiment_dir / "events.jsonl").write_text("", encoding="utf-8")

    config = None if force_local or planner == "local" else llm_config()
    if chat_fn is not None and not force_local and planner != "local":
        config = config or {"base_url": "http://local", "api_key": "", "model": "scripted"}
    llm_available = config is not None
    llm_enabled = llm_available

    planned = plan if plan is not None else (
        build_local_plan(task, agent_config) if not llm_enabled else []
    )
    state: AgentState = {
        "status": "running",
        "stage": "start",
        "nodes": [],
        "events": [],
        "task": task,
        "agent_config": agent_config,
        "context": context,
        "registry": build_registry(),
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
        "deadline": time.perf_counter() + min(agent_config.max_seconds, context.budget.remaining),
        "llm_used": False,
        "llm_enabled": llm_enabled,
        "llm_requests": 0,
        "llm_config": config,
        "chat_fn": chat_fn,
    }
    graph = build_graph()
    try:
        final_state = graph.invoke(state)
    except Exception as error:  # noqa: BLE001 - 图内部异常也要落盘
        summary = {
            "status": "failed",
            "task": task,
            "dataset": context.runtime.dataset,
            "planner": "llm" if llm_enabled else "local",
            "llm_used": False,
            "llm_available": llm_available,
            "llm_enabled": llm_enabled,
            "llm_requests": 0,
            "max_llm_requests": agent_config.max_llm_requests,
            "selected_model": detect_model(task) or agent_config.default_model,
            "tools_called": [],
            "tool_results": [],
            "errors": [f"{type(error).__name__}: {error}"],
            "metrics": {},
            "budget": context.budget.as_dict(),
            "weights": {},
            "final_answer": f"Agent 执行异常：{type(error).__name__}: {error}",
            "elapsed_s": round(time.perf_counter() - float(state["started"]), 2),
        }
        final_state = {"summary": summary}

    summary = final_state.get("summary") or {}
    summary.setdefault("llm_available", llm_available)
    summary.setdefault("llm_enabled", llm_enabled)
    summary["messages_count"] = len(final_state.get("messages") or [])

    trace = _trace_payload(summary)
    _write_json(experiment_dir / "trace.json", trace)
    _write_json(experiment_dir / "metrics.json", summary.get("metrics", {}))
    _write_json(
        experiment_dir / "run_meta.json",
        {
            key: summary.get(key)
            for key in (
                "status",
                "task",
                "dataset",
                "planner",
                "llm_used",
                "llm_available",
                "llm_requests",
                "max_llm_requests",
                "selected_model",
                "tools_called",
                "budget",
                "errors",
                "elapsed_s",
                "weights",
            )
        },
    )
    _write_report(experiment_dir / "report.md", summary, trace)
    return summary


# --------------------------------------------------------------------------- 自检


class _ScriptedChat:
    """本地替身模型：按顺序返回预设响应，不联网（用于工具协议回归）。"""

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def __call__(self, config, messages, *, tools=None, timeout=None):
        self.calls.append(list(messages))
        if not self.responses:
            raise RuntimeError("scripted chat 没有更多响应")
        return self.responses.pop(0)


def _tool_call_response(name: str, arguments: dict, call_id: str) -> dict:
    return {
        "id": f"resp_{call_id}",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    ],
                }
            }
        ],
    }


def _final_response(text: str) -> dict:
    return {
        "id": "resp_final",
        "choices": [{"message": {"role": "assistant", "content": text}}],
    }


def run_self_check(output: Path = DEFAULT_SELF_CHECK) -> dict:
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    context = build_context(
        max_tool_calls=9,
        max_seconds=180.0,
    )
    registry = build_registry()

    models = registry.call("list_ad_models", {}, context)
    record(
        "list_ad_models",
        models.get("ok") and len(models.get("models", [])) == 6,
        {
            "models": [item["name"] for item in models.get("models", [])],
            "statuses": {item["name"]: item["status"] for item in models.get("models", [])},
        },
    )

    try:
        registry.validate("predict_ad_ctr", {"model": "not_a_model"})
        record("unknown_model_rejected", False, "非法模型没有被拒绝")
    except ToolCallError as error:
        record("unknown_model_rejected", True, str(error))

    prediction = registry.call(
        "predict_ad_ctr", {"model": "din", "split": "test", "limit": 500}, context
    )
    record(
        "predict_ad_ctr",
        prediction.get("ok") and prediction.get("rows", 0) > 0
        and "AUC" in prediction.get("metrics", {}),
        {
            "rows": prediction.get("rows"),
            "metrics": prediction.get("metrics"),
            "weight_kind": prediction.get("weight", {}).get("weight_kind"),
        },
    )

    ranking = registry.call(
        "rank_ads",
        {"model": "din", "split": "train", "requests": 2, "min_candidates": 2, "top_k": 3},
        context,
    )
    order_ok = all(
        all(
            record_item["ctr"] >= next_item["ctr"]
            for record_item, next_item in zip(ranking_record["ranking"], ranking_record["ranking"][1:])
        )
        for ranking_record in ranking.get("rankings", [])
    )
    record(
        "rank_ads",
        ranking.get("ok") and ranking.get("summary", {}).get("requests", 0) >= 1 and order_ok,
        {"summary": ranking.get("summary")},
    )

    auction = registry.call(
        "simulate_auction",
        {
            "model": "din",
            "split": "train",
            "requests": 2,
            "min_candidates": 2,
            "top_k": 3,
        },
        context,
    )
    auction_summary = auction.get("summary", {})
    record(
        "simulate_auction",
        auction.get("ok")
        and auction_summary.get("winners", 0) >= 1
        and auction_summary.get("spend", 0) > 0
        and abs(
            auction_summary.get("budget", {}).get("remaining", -1)
            - (
                auction_summary.get("budget", {}).get("total", 0)
                - auction_summary.get("budget", {}).get("spent", 0)
            )
        )
        < 1e-6,
        {"summary": auction_summary},
    )

    budget_state = registry.call("get_budget_state", {}, context)
    record(
        "get_budget_state",
        budget_state.get("ok")
        and budget_state.get("run_id") == context.state["last_auction"]["run_id"],
        {"run_id": budget_state.get("run_id"), "budget": budget_state.get("budget")},
    )

    zero = registry.call(
        "simulate_auction",
        {
            "model": "din",
            "split": "train",
            "requests": 2,
            "min_candidates": 2,
            "top_k": 3,
            "total_budget": 0.0,
            "per_advertiser_budget": 0.0,
        },
        context,
    )
    zero_summary = zero.get("summary", {})
    record(
        "budget_insufficient",
        zero.get("ok")
        and zero_summary.get("winners") == 0
        and zero_summary.get("spend") == 0
        and all(
            record_item.get("no_winner_reason")
            for record_item in zero.get("requests_detail", [])
        ),
        {
            "summary": {
                "winners": zero_summary.get("winners"),
                "spend": zero_summary.get("spend"),
                "reasons": zero_summary.get("no_winner_reasons"),
            }
        },
    )

    multitask = registry.call("predict_multitask", {"model": "mmoe"}, context)
    record(
        "predict_multitask",
        multitask.get("ok")
        and multitask.get("synthetic") is True
        and multitask.get("metrics", {}).get("cvr_task", {}).get("rows", 0) > 0,
        {
            "rows": multitask.get("rows"),
            "metrics": {
                key: value.get("rows")
                for key, value in multitask.get("metrics", {}).items()
            },
        },
    )

    labels = registry.call("check_label_availability", {}, context)
    record(
        "check_label_availability",
        labels.get("ok")
        and labels.get("conversion_label", {}).get("available") is False
        and labels.get("simulated_fixture", {}).get("synthetic") is True,
        {
            "conversion_available": labels.get("conversion_label", {}).get("available"),
            "cvr_mode": labels.get("conversion_label", {}).get("cvr_mode"),
        },
    )

    # ---- LLM 协议回归（本地替身，不联网）
    scripted = _ScriptedChat(
        [
            _tool_call_response(
                "rank_ads",
                {"model": "din", "split": "train", "requests": 1, "min_candidates": 2, "top_k": 3},
                "call_ads_1",
            ),
            _final_response("排序完成，结果见工具返回。"),
        ]
    )
    llm_context = build_context(max_tool_calls=3, max_seconds=120.0)
    llm_run = run_agent(
        llm_context,
        task="用 din 对广告候选排序",
        experiment_dir=Path(output).parent / "self_check_llm",
        agent_config=AgentConfig(max_tool_calls=3, max_llm_requests=3, max_seconds=120.0),
        planner="auto",
        chat_fn=scripted,
    )
    record(
        "llm_tool_roundtrip",
        llm_run["status"] == "verified"
        and llm_run["llm_requests"] == 2
        and llm_run["tool_results"]
        and llm_run["tool_results"][0]["tool_call_id"] == "call_ads_1"
        and llm_run["tool_results"][0]["tool"] == "rank_ads",
        {
            "status": llm_run["status"],
            "llm_requests": llm_run["llm_requests"],
            "tool_call_id": llm_run["tool_results"][0]["tool_call_id"] if llm_run["tool_results"] else None,
            "trace": str(Path(output).parent / "self_check_llm" / "trace.json"),
        },
    )

    unknown_script = _ScriptedChat(
        [_tool_call_response("delete_database", {}, "call_bad_1")]
    )
    unknown_context = build_context(max_tool_calls=2, max_seconds=60.0)
    unknown_run = run_agent(
        unknown_context,
        task="调用不存在的工具",
        experiment_dir=Path(output).parent / "self_check_llm_unknown",
        agent_config=AgentConfig(max_tool_calls=2, max_llm_requests=2, max_seconds=60.0),
        planner="auto",
        chat_fn=unknown_script,
    )
    record(
        "llm_unknown_tool_rejected",
        unknown_run["status"] == "failed"
        and not unknown_run["tool_results"]
        and any("未注册的工具" in error for error in unknown_run["errors"]),
        {"status": unknown_run["status"], "errors": unknown_run["errors"]},
    )

    duplicate_context = build_context(max_tool_calls=4, max_seconds=120.0)
    duplicate_plan = [
        PlannedCall("list_ad_models", {}),
        PlannedCall("list_ad_models", {}),
    ]
    duplicate_run = run_agent(
        duplicate_context,
        task="重复调用回归",
        experiment_dir=Path(output).parent / "self_check_duplicate",
        agent_config=AgentConfig(max_tool_calls=4, max_llm_requests=1, max_seconds=120.0),
        plan=duplicate_plan,
        planner="local",
    )
    record(
        "duplicate_call_rejected",
        duplicate_run["status"] == "partial"
        and any("重复调用" in error for error in duplicate_run["errors"]),
        {"status": duplicate_run["status"], "errors": duplicate_run["errors"]},
    )

    budget_context = build_context(max_tool_calls=1, max_seconds=120.0)
    budget_run = run_agent(
        budget_context,
        task="工具预算回归",
        experiment_dir=Path(output).parent / "self_check_budget",
        agent_config=AgentConfig(max_tool_calls=1, max_llm_requests=1, max_seconds=120.0),
        plan=[PlannedCall("list_ad_models", {}), PlannedCall("check_label_availability", {})],
        planner="local",
    )
    record(
        "tool_budget_enforced",
        budget_run["status"] == "partial"
        and budget_run["budget"]["calls"] == 1
        and len(budget_run["tool_results"]) == 1,
        {"status": budget_run["status"], "budget": budget_run["budget"]},
    )

    local_context = build_context(max_tool_calls=9, max_seconds=180.0)
    local_run = run_agent(
        local_context,
        task="用 din 模型完成广告预测、排序、竞价和预算模拟",
        experiment_dir=Path(output).parent / "self_check_local",
        agent_config=AgentConfig(max_tool_calls=9, max_llm_requests=1, max_seconds=180.0),
        planner="local",
    )
    run_dir = Path(output).parent / "self_check_local"
    record(
        "local_plan_e2e",
        local_run["status"] == "verified"
        and {"predict_ad_ctr", "rank_ads", "simulate_auction"}.issubset(set(local_run["tools_called"]))
        and local_run["selected_model"] == "din"
        and all(
            (run_dir / name).exists()
            for name in ("trace.json", "metrics.json", "run_meta.json", "report.md", "events.jsonl")
        ),
        {
            "status": local_run["status"],
            "tools": local_run["tools_called"],
            "model": local_run["selected_model"],
            "artifacts": [str(run_dir / name) for name in (
                "trace.json",
                "metrics.json",
                "run_meta.json",
                "report.md",
                "events.jsonl",
            )],
        },
    )

    passed = sum(1 for item in checks if item["passed"])
    report = {
        "passed": passed,
        "total": len(checks),
        "ok": passed == len(checks),
        "checks": checks,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(Path(output), report)
    return report


# --------------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="用 din 模型完成一次广告 CTR 预测、广告排序、竞价和预算模拟，并说明转化标签缺口",
    )
    parser.add_argument("--model", choices=AD_MODEL_NAMES, default=None)
    parser.add_argument("--split", choices=SPLITS, default=None)
    parser.add_argument("--planner", choices=("auto", "local", "llm"), default="auto")
    parser.add_argument("--dataset", default="kuasar_small")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--config", type=Path, default=Path("configs/ads/ads_agent.toml"))
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--self-check-output", type=Path, default=DEFAULT_SELF_CHECK)
    parser.add_argument("--json", action="store_true", help="把运行摘要打印为 JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv()
    if args.self_check:
        report = run_self_check(args.self_check_output)
        print(json.dumps(report, ensure_ascii=False, indent=2)[:8000])
        return 0 if report["ok"] else 1

    agent_config = load_agent_config(args.config)
    if args.model:
        agent_config = AgentConfig(
            **{
                **agent_config.as_dict(),
                "default_model": args.model,
            }
        )
    if args.split:
        agent_config = AgentConfig(
            **{**agent_config.as_dict(), "default_split": args.split}
        )
    if args.planner == "llm" and not llm_config():
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "error": (
                        "缺少 LLM 配置（DSH_AGENT_LLM_BASE_URL / "
                        "DSH_AGENT_LLM_API_KEY / DSH_AGENT_LLM_MODEL）；"
                        "可先用 --planner local 跑离线工具链。"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    context = build_context(
        dataset=args.dataset,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        max_tool_calls=agent_config.max_tool_calls,
        max_seconds=agent_config.max_seconds,
    )
    experiment_dir = args.experiment_dir or (
        Path(args.output_root) / f"run-{_utc_stamp()}-{__import__('os').getpid()}"
    )
    force_local = args.planner == "local"
    summary = run_agent(
        context,
        task=args.task,
        experiment_dir=experiment_dir,
        agent_config=agent_config,
        planner=args.planner,
        force_local=force_local,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "planner": summary["planner"],
                "selected_model": summary["selected_model"],
                "tools_called": summary["tools_called"],
                "llm_requests": summary["llm_requests"],
                "experiment_dir": str(experiment_dir),
                "final_answer": summary["final_answer"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.json:
        print(json.dumps(summary.get("metrics", {}), ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
