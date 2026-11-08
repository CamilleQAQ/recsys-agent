"""Phase 11.2：历史优先的 LangGraph Agent（先查历史，未命中才评测一次）。

正式入口（一次真实 LLM 工作流）：

    python -m recsys.experiments.history_first_agent `
      --experiment-dir experiments/2026-11-08-kuasar_small-history-first-agent `
      --quota 30/70

本地检查（不请求真实 LLM）：

    python -m recsys.experiments.history_first_agent --self-check
    python -m recsys.experiments.history_first_agent --check-cache --cache-dir <dir>
    python -m recsys.experiments.history_first_agent --regression-miss
    python -m recsys.experiments.history_first_agent --regression-timeout history
    python -m recsys.experiments.history_first_agent --regression-timeout evaluation

图节点：

``load_baseline`` → ``request_history`` → ``execute_history`` → ``send_history_result``
→（命中）``finish``
→（未命中且模型选择评测）``execute_eval`` → ``send_eval_result`` → ``finish``
→（未命中且模型停止）``finish``

程序守卫：

* 第一次请求只开放 ``search_experiment_history``；任何评测请求在历史查询之前都被拦截；
* 历史命中后评测工具不开放；未命中最多评测一次且必须同一目标配额；
* 最多 3 次模型请求、1 次历史查询、1 次配额评测、180 秒；
* 只有「实际历史查询结果 + 有效最终回复（若评测则同 ID 回传）」才写 ``verified``；
* 失败保留已有工具结果，状态为 ``partial`` / ``blocked`` / ``failed``，CLI 非零。

``--cache-dir`` 从 CLI 进入 LangGraph 状态，基准校验、历史匹配与融合评测都只使用该目录；
省略时用默认目录；显式目录缺失/哈希不符/口径不符直接报错，不回退默认目录。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph

from recsys.experiments.agent_tools import (
    TOOL_EVALUATE,
    TOOL_SEARCH_HISTORY,
    ToolCallError,
    ToolContext,
    ToolRegistry,
    build_registry,
)
from recsys.experiments.experiment_history import (
    DEFAULT_HISTORY_ROOT,
    model_view as history_model_view,
)
from recsys.experiments.fusion_quota import (
    DEFAULT_CACHE,
    QuotaError,
    load_context,
    validate_quota,
)
from recsys.experiments.llm_tool_call import (
    VerificationError,
    host_only,
    llm_config,
    load_dotenv,
    post_chat,
)

EXPERIMENT_DEFAULT = Path("experiments/2026-11-08-kuasar_small-history-first-agent")
BASELINE_RUN = Path("experiments/2026-11-08-kuasar_small-fusion-quota-agent/runs/50_50.json")
OUTPUT_ROOT = Path("artifacts/kuasar_small/history_first_agent")

MAX_MODEL_REQUESTS = 3
MAX_HISTORY_QUERIES = 1
MAX_EVALUATIONS = 1
BUDGET_SECONDS = 180.0
DEFAULT_TARGET_QUOTA = (30, 70)
LLM_ENV = ("DSH_AGENT_LLM_BASE_URL", "DSH_AGENT_LLM_API_KEY", "DSH_AGENT_LLM_MODEL")


class RequestFailed(RuntimeError):
    """模型请求未发出或失败；由节点决定 blocked/partial。"""


# --------------------------------------------------------------------------- 基准与缓存身份


def _artifact_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return Path(__file__).resolve().parents[3] / path


def _baseline_artifact_reasons(baseline: dict) -> list[str]:
    evidence = baseline.get("evidence") or {}
    path = _artifact_path(evidence.get("fusion"))
    if path is None or not path.exists():
        return [f"基准融合工件不存在：{evidence.get('fusion')}"]
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"基准融合工件无法读取：{error}"]
    reasons = []
    if artifact.get("quota") != baseline.get("quota"):
        reasons.append("基准工件 quota 不一致")
    if artifact.get("budget") != 100:
        reasons.append("基准工件预算不是 100")
    if artifact.get("users") != baseline.get("users"):
        reasons.append("基准工件用户数不一致")
    if artifact.get("fingerprints") != (evidence.get("fingerprints") or {}):
        reasons.append("基准工件指纹与记录不一致")
    return reasons


def _baseline_model_view(baseline: dict) -> dict:
    return {
        key: baseline.get(key)
        for key in (
            "quota",
            "users",
            "targets",
            "hits",
            "recall_at_100",
            "coverage",
            "candidates_total",
        )
    }


def load_baseline(cache_dir: Path) -> tuple[dict, dict]:
    """读取 50/50 基准，并用 CLI 指定目录的当前缓存身份核对可比性。"""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise VerificationError(f"缓存目录不存在：{cache_dir}")
    context = load_context(cache_dir)
    current = context.fingerprints
    if not BASELINE_RUN.exists():
        raise VerificationError(f"缺少基准文件：{BASELINE_RUN}（先跑 2.23 的本地 Agent）")
    baseline = json.loads(BASELINE_RUN.read_text(encoding="utf-8"))
    if (baseline.get("quota") or {}) != {"two_tower": 50, "mind": 50}:
        raise VerificationError("基准记录的配额不是 50/50")
    if str(baseline.get("status") or "").lower() in {"partial", "failed", "blocked", "error"}:
        raise VerificationError(f"基准记录状态不可复用：{baseline.get('status')}")
    recorded = (baseline.get("evidence") or {}).get("fingerprints") or {}
    keys = (
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
    mismatched = [key for key in keys if recorded.get(key) != current.get(key)]
    if mismatched:
        raise VerificationError(f"基准指纹与当前缓存不一致：{mismatched}")
    artifact_reasons = _baseline_artifact_reasons(baseline)
    if artifact_reasons:
        raise VerificationError("基准证据不可复用：" + "；".join(artifact_reasons))
    based_on = {
        "quota": baseline["quota"],
        "hits": baseline["hits"],
        "recall_at_100": baseline["recall_at_100"],
        "coverage": baseline["coverage"],
        "candidates_total": baseline["candidates_total"],
        "users": baseline["users"],
        "targets": baseline["targets"],
        "fingerprints_verified": True,
        "evidence": {
            "run": str(BASELINE_RUN),
            "fusion": (baseline.get("evidence") or {}).get("fusion"),
            "cache": str(cache_dir),
        },
    }
    return based_on, current


# --------------------------------------------------------------------------- 消息与解析


def base_messages(state: "AgentState") -> list[dict]:
    """只发送任务配置与聚合指标：无原始用户/物品 ID、候选明细、标签、路径或密钥。"""
    baseline = _baseline_model_view(state["baseline"])
    target = list(state["target_quota"])
    return [
        {
            "role": "system",
            "content": (
                "你是推荐系统研究 Agent。必须使用服务商原生 tool_calls，每轮最多一条，"
                "禁止并行多个 tool_calls；不要用普通文本代替工具调用；"
                "不要输出用户/物品原始 ID、验证标签、文件路径或密钥。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": "30/70 与 50/50 相比，哪个融合配额能保留更多验证目标？",
                    "dataset": "kuasar_small",
                    "fixed_users": 128,
                    "validation_targets": 362,
                    "candidate_budget_per_user": 100,
                    "target_quota": {
                        "two_tower_quota": target[0],
                        "mind_quota": target[1],
                    },
                    "quota_order": "双塔 / MIND",
                    "baseline_quota": baseline["quota"],
                    "baseline_aggregate_metrics": baseline,
                    "selection_rule": "命中目标数优先，Recall@100 次之；持平时保留基准",
                    "instruction": (
                        "第一步必须调用 search_experiment_history，查询上述目标配额是否已有"
                        "与当前缓存一致的可复用结果；禁止直接回答，也不要调用其他工具。"
                    ),
                },
                ensure_ascii=False,
            ),
        },
    ]


def extract_tool_call(
    response: dict,
    *,
    registry: ToolRegistry,
    allowed: tuple[str, ...],
    target_quota: tuple[int, int],
    stage: str,
) -> dict:
    """只接受注册过且当前阶段开放的单条原生 tool_call；参数只能是目标配额。"""
    message = (response.get("choices") or [{}])[0].get("message", {}) or {}
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        raise VerificationError(f"{stage} 响应里没有 tool_calls（普通文本/JSON 不计入验收）")
    if len(tool_calls) != 1:
        raise VerificationError(f"{stage} 的 tool_calls 数量为 {len(tool_calls)}，要求恰好 1 条")
    call = tool_calls[0]
    if (call.get("type") or "function") != "function":
        raise VerificationError(f"不支持的 tool call 类型：{call.get('type')}")
    name = (call.get("function") or {}).get("name")
    if name not in allowed:
        if name in registry.names:
            raise VerificationError(
                f"阶段 {stage} 不允许调用 {name}；必须先完成历史查询且历史未命中"
            )
        raise VerificationError(f"未注册的工具：{name}")
    if not call.get("id"):
        raise VerificationError("tool_call 缺少 id")
    raw_arguments = (call.get("function") or {}).get("arguments") or "{}"
    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        try:
            arguments = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError) as error:
            raise VerificationError(f"工具参数不是合法 JSON：{error}") from error
    try:
        validated = registry.validate(name, arguments)
    except ToolCallError as error:
        raise VerificationError(str(error)) from error
    quota = (validated["two_tower_quota"], validated["mind_quota"])
    if quota != target_quota:
        raise VerificationError(f"工具参数配额 {quota} 与任务目标 {target_quota} 不一致")
    return {
        "id": call["id"],
        "name": name,
        "arguments": validated,
        "quota": quota,
        "assistant_message": {
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": tool_calls,
        },
    }


# --------------------------------------------------------------------------- LangGraph 状态


class AgentState(TypedDict, total=False):
    status: str  # running / verified / partial / blocked / failed
    stage: str
    nodes: list[str]
    events: list[dict]
    started: float
    deadline: float
    config: dict
    cache_dir: str
    cache_dir_source: str
    history_root: str
    output_dir: str
    target_quota: list[int]
    baseline: dict
    fingerprints: dict
    request_count: int
    history_queries: int
    evaluation_count: int
    response_ids: dict
    call_ids: dict
    history_call_id: str
    history_arguments: dict
    history_assistant_message: dict
    history_result: dict
    eval_call_id: str
    eval_arguments: dict
    eval_assistant_message: dict
    eval_result: dict
    final_reply: str
    analysis_path: str
    decision: str
    stop_reason: str
    error: str
    conclusion: dict


ChatFn = Callable[..., dict]
BaselineFn = Callable[[Path], tuple[dict, dict]]


def _remaining(state: AgentState) -> float:
    return float(state.get("deadline", time.perf_counter())) - time.perf_counter()


def _event(state: AgentState, payload: dict) -> None:
    state.setdefault("events", []).append(payload)


def _enter(state: AgentState, node: str) -> AgentState:
    state = dict(state)  # type: ignore[assignment]
    state.setdefault("nodes", []).append(node)
    state["stage"] = node
    _event(state, {"event": "node_start", "node": node})
    return state  # type: ignore[return-value]


def _leave(state: AgentState, node: str) -> AgentState:
    _event(state, {"event": "node_end", "node": node, "status": state.get("status")})
    return state  # type: ignore[return-value]


def _tool_context(state: AgentState) -> ToolContext:
    """所有本地路径都来自程序状态，绝不从模型参数读取。"""
    return ToolContext(
        cache_dir=Path(state["cache_dir"]),
        output_dir=Path(state["output_dir"]),
        history_root=Path(state["history_root"]),
    )


def _request(
    state: AgentState,
    chat_fn: ChatFn,
    messages: list[dict],
    *,
    tools: list[dict] | None,
    label: str,
) -> dict:
    if _remaining(state) <= 0:
        raise RequestFailed("预算耗尽，未发起下一次模型请求")
    if int(state.get("request_count", 0)) >= MAX_MODEL_REQUESTS:
        raise RequestFailed(f"模型请求次数已达到上限 {MAX_MODEL_REQUESTS}")
    number = int(state.get("request_count", 0)) + 1
    timeout = min(30.0, max(1.0, _remaining(state)))
    state["request_count"] = number
    tool_names = [entry["function"]["name"] for entry in (tools or [])]
    _event(
        state,
        {
            "event": "llm_request",
            "request": number,
            "stage": label,
            "timeout_s": round(timeout, 1),
            "tools": tool_names,
        },
    )
    try:
        response = chat_fn(state["config"], messages, tools=tools or None, timeout=timeout)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, RuntimeError) as error:
        raise RequestFailed(f"第 {number} 次模型请求失败：{error}") from error
    response_id = response.get("id")
    state.setdefault("response_ids", {})[label] = response_id
    message = (response.get("choices") or [{}])[0].get("message", {}) or {}
    _event(
        state,
        {
            "event": "llm_response",
            "request": number,
            "stage": label,
            "response_id": response_id,
            "tool_calls": [
                {
                    "id": call.get("id"),
                    "type": call.get("type"),
                    "name": (call.get("function") or {}).get("name"),
                    "arguments": (call.get("function") or {}).get("arguments"),
                }
                for call in (message.get("tool_calls") or [])
            ],
            "content_preview": (message.get("content") or "")[:200],
        },
    )
    return response


def _fail_request(state: AgentState, node: str, error: Exception) -> AgentState:
    state["status"] = "partial" if state.get("history_result") else "blocked"
    state["error"] = str(error)
    _event(state, {"event": state["status"], "reason": str(error), "stage": node})
    return _leave(state, node)


def _set_stop(state: AgentState, reason: str) -> None:
    state["stop_reason"] = reason
    _event(state, {"event": "stop", "reason": reason, "decision": state.get("decision")})


# --------------------------------------------------------------------------- 节点


def node_load_baseline(state: AgentState, baseline_fn: BaselineFn) -> AgentState:
    state = _enter(state, "load_baseline")
    cache_dir = Path(state["cache_dir"])
    if state.get("baseline"):
        _event(
            state,
            {
                "event": "cache_identity",
                "cache_dir": str(cache_dir),
                "cache_dir_source": state.get("cache_dir_source"),
                "input_hashes": state.get("fingerprints") or {},
                "baseline": _baseline_model_view(state["baseline"]),
                "note": "基准由本地模拟注入",
            },
        )
        return _leave(state, "load_baseline")
    try:
        baseline, fingerprints = baseline_fn(cache_dir)
    except (VerificationError, QuotaError) as error:
        state["status"] = "blocked"
        state["error"] = str(error)
        _event(state, {"event": "blocked", "reason": str(error), "cache_dir": str(cache_dir)})
        return _leave(state, "load_baseline")
    state["baseline"] = baseline
    state["fingerprints"] = fingerprints
    _event(
        state,
        {
            "event": "cache_identity",
            "cache_dir": str(cache_dir),
            "cache_dir_source": state.get("cache_dir_source"),
            "input_hashes": fingerprints,
            "baseline": _baseline_model_view(baseline),
        },
    )
    _event(
        state,
        {
            "event": "baseline_loaded",
            "baseline": _baseline_model_view(baseline),
            "evidence": baseline.get("evidence"),
            "fingerprints_verified": True,
        },
    )
    return _leave(state, "load_baseline")


def node_request_history(
    state: AgentState, chat_fn: ChatFn, registry: ToolRegistry
) -> AgentState:
    state = _enter(state, "request_history")
    if state.get("history_queries", 0) >= MAX_HISTORY_QUERIES:
        state["status"] = "blocked"
        state["error"] = "历史查询已经执行过，拒绝重复查询"
        _event(state, {"event": "blocked", "reason": state["error"]})
        return _leave(state, "request_history")
    try:
        response = _request(
            state,
            chat_fn,
            base_messages(state),
            tools=registry.schemas((TOOL_SEARCH_HISTORY,)),
            label="history_request",
        )
    except RequestFailed as error:
        return _fail_request(state, "request_history", error)

    target = (int(state["target_quota"][0]), int(state["target_quota"][1]))
    try:
        parsed = extract_tool_call(
            response,
            registry=registry,
            allowed=(TOOL_SEARCH_HISTORY,),
            target_quota=target,
            stage="第 1 次请求",
        )
    except VerificationError as error:
        state["status"] = "blocked"
        state["error"] = str(error)
        _event(state, {"event": "blocked", "reason": str(error), "stage": "request_history"})
        return _leave(state, "request_history")

    state["history_call_id"] = parsed["id"]
    state["history_arguments"] = parsed["arguments"]
    state["history_assistant_message"] = parsed["assistant_message"]
    state.setdefault("call_ids", {})["history"] = parsed["id"]
    _event(
        state,
        {
            "event": "tool_call_accepted",
            "tool": TOOL_SEARCH_HISTORY,
            "tool_call_id": parsed["id"],
            "quota": list(parsed["quota"]),
            "request": 1,
        },
    )
    return _leave(state, "request_history")


def node_execute_history(state: AgentState, registry: ToolRegistry) -> AgentState:
    state = _enter(state, "execute_history")
    if state.get("history_queries", 0) >= MAX_HISTORY_QUERIES:
        state["status"] = "blocked"
        state["error"] = "历史查询最多 1 次，拒绝重复执行"
        _event(state, {"event": "blocked", "reason": state["error"]})
        return _leave(state, "execute_history")
    started = time.perf_counter()
    try:
        result = registry.execute(
            TOOL_SEARCH_HISTORY,
            dict(state["history_arguments"]),
            context=_tool_context(state),
        )
    except Exception as error:  # noqa: BLE001 - 工具失败进入失败分支
        state["status"] = "failed"
        state["error"] = f"历史工具执行失败：{error}"
        _event(
            state,
            {
                "event": "tool_failed",
                "tool": TOOL_SEARCH_HISTORY,
                "tool_call_id": state.get("history_call_id"),
                "error": str(error),
            },
        )
        return _leave(state, "execute_history")
    state["history_queries"] = int(state.get("history_queries", 0)) + 1
    state["history_result"] = result
    _event(
        state,
        {
            "event": "history_searched",
            "tool_call_id": state.get("history_call_id"),
            "status": result.get("status"),
            "hit": bool(result.get("hit")),
            "quota": result.get("quota"),
            "reason": result.get("reason"),
            "metrics": result.get("metrics"),
            "baseline": result.get("baseline"),
            "delta_vs_baseline": result.get("delta_vs_baseline"),
            "rule_compatibility": result.get("rule_compatibility"),
            "fingerprints_verified": result.get("fingerprints_verified"),
            "evidence": result.get("evidence"),
            "cache_dir": state.get("cache_dir"),
            "tool_elapsed_s": round(time.perf_counter() - started, 2),
        },
    )
    return _leave(state, "execute_history")


def _history_followup(state: AgentState) -> list[dict]:
    history = state["history_result"]
    if history.get("hit"):
        instruction = (
            "这是真实历史查询结果，status=hit。请引用历史指标并与 50/50 基准比较，"
            "用一句话说明为什么停止（不再执行融合评测）。不要再调用工具，"
            "不要输出 JSON 代码块。"
        )
    else:
        instruction = (
            "这是真实历史查询结果，status=miss，原因见上。你可以选择调用 "
            f"{TOOL_EVALUATE} 评测目标配额 {state['target_quota'][0]}/"
            f"{state['target_quota'][1]}（只能一次），或直接说明原因后停止。"
            "若调用工具，只允许一条 tool_call，参数必须是 two_tower_quota="
            f"{state['target_quota'][0]}、mind_quota={state['target_quota'][1]}。"
        )
    return base_messages(state) + [
        state["history_assistant_message"],
        {
            "role": "tool",
            "tool_call_id": state["history_call_id"],
            "content": json.dumps(history_model_view(history), ensure_ascii=False),
        },
        {"role": "user", "content": instruction},
    ]


def node_send_history_result(
    state: AgentState, chat_fn: ChatFn, registry: ToolRegistry
) -> AgentState:
    state = _enter(state, "send_history_result")
    history = state.get("history_result")
    if not history:
        state["status"] = "partial"
        state["error"] = "缺少历史查询结果，不能继续"
        _event(state, {"event": "partial", "reason": state["error"]})
        return _leave(state, "send_history_result")

    tools = [] if history.get("hit") else registry.schemas((TOOL_EVALUATE,))
    try:
        response = _request(
            state,
            chat_fn,
            _history_followup(state),
            tools=tools,
            label="history_followup",
        )
    except RequestFailed as error:
        return _fail_request(state, "send_history_result", error)

    message = (response.get("choices") or [{}])[0].get("message", {}) or {}
    content = (message.get("content") or "").strip()

    if history.get("hit"):
        if message.get("tool_calls"):
            state["status"] = "partial"
            state["error"] = "历史已命中，但模型仍请求调用工具，未得到最终结论"
            _event(state, {"event": "partial", "reason": state["error"]})
            return _leave(state, "send_history_result")
        if not content:
            state["status"] = "partial"
            state["error"] = "历史命中后最终回复为空"
            _event(state, {"event": "partial", "reason": state["error"]})
            return _leave(state, "send_history_result")
        state["final_reply"] = content
        state["analysis_path"] = "history_hit"
        state["decision"] = "reuse_history"
        _event(
            state,
            {
                "event": "history_reused",
                "decision": "reuse_history",
                "evaluation_count": int(state.get("evaluation_count", 0)),
                "final_reply": content,
            },
        )
        return _leave(state, "send_history_result")

    if not message.get("tool_calls"):
        if not content:
            state["status"] = "partial"
            state["error"] = "历史未命中，且模型既未评测也未给出停止理由"
            _event(state, {"event": "partial", "reason": state["error"]})
            return _leave(state, "send_history_result")
        state["final_reply"] = content
        state["analysis_path"] = "miss_stop"
        state["decision"] = "stopped_without_eval"
        _set_stop(state, content)
        return _leave(state, "send_history_result")

    target = (int(state["target_quota"][0]), int(state["target_quota"][1]))
    try:
        parsed = extract_tool_call(
            response,
            registry=registry,
            allowed=(TOOL_EVALUATE,),
            target_quota=target,
            stage="历史未命中后的第 2 次请求",
        )
    except VerificationError as error:
        state["status"] = "blocked"
        state["error"] = str(error)
        _event(state, {"event": "blocked", "reason": str(error), "stage": "send_history_result"})
        return _leave(state, "send_history_result")

    state["eval_call_id"] = parsed["id"]
    state["eval_arguments"] = parsed["arguments"]
    state["eval_assistant_message"] = parsed["assistant_message"]
    state.setdefault("call_ids", {})["evaluation"] = parsed["id"]
    _event(
        state,
        {
            "event": "tool_call_accepted",
            "tool": TOOL_EVALUATE,
            "tool_call_id": parsed["id"],
            "quota": list(parsed["quota"]),
            "request": 2,
        },
    )
    return _leave(state, "send_history_result")


def node_execute_eval(state: AgentState, registry: ToolRegistry) -> AgentState:
    state = _enter(state, "execute_eval")
    history = state.get("history_result") or {}
    if history.get("hit"):
        state["status"] = "blocked"
        state["error"] = "历史已命中，评测工具不开放"
    elif int(state.get("evaluation_count", 0)) >= MAX_EVALUATIONS:
        state["status"] = "blocked"
        state["error"] = f"评测最多 {MAX_EVALUATIONS} 次，拒绝重复执行"
    elif int(state.get("history_queries", 0)) != 1:
        state["status"] = "blocked"
        state["error"] = "必须先完成 1 次历史查询，才能评测"
    elif not state.get("eval_call_id"):
        state["status"] = "blocked"
        state["error"] = "缺少评估工具调用 ID，拒绝执行"
    if state.get("status") == "blocked":
        _event(state, {"event": "blocked", "reason": state["error"], "stage": "execute_eval"})
        return _leave(state, "execute_eval")

    started = time.perf_counter()
    try:
        result = registry.execute(
            TOOL_EVALUATE,
            dict(state["eval_arguments"]),
            context=_tool_context(state),
        )
    except Exception as error:  # noqa: BLE001 - 工具失败进入失败分支
        state["status"] = "failed"
        state["error"] = f"评测工具执行失败：{error}"
        _event(
            state,
            {
                "event": "tool_failed",
                "tool": TOOL_EVALUATE,
                "tool_call_id": state.get("eval_call_id"),
                "error": str(error),
            },
        )
        return _leave(state, "execute_eval")

    state["evaluation_count"] = int(state.get("evaluation_count", 0)) + 1
    state["eval_result"] = result
    state["decision"] = "evaluated"
    _event(
        state,
        {
            "event": "tool_executed",
            "tool": TOOL_EVALUATE,
            "tool_call_id": state.get("eval_call_id"),
            "quota": result.get("quota"),
            "result": {
                key: result.get(key)
                for key in (
                    "users",
                    "targets",
                    "candidates_total",
                    "candidates_mean",
                    "fills_total",
                    "hits",
                    "recall_at_100",
                    "coverage",
                )
            },
            "fingerprints_verified": result.get("evidence", {}).get("fingerprints"),
            "evidence": result.get("evidence"),
            "cache_dir": state.get("cache_dir"),
            "tool_elapsed_s": round(time.perf_counter() - started, 2),
        },
    )
    return _leave(state, "execute_eval")


def node_send_eval_result(state: AgentState, chat_fn: ChatFn) -> AgentState:
    state = _enter(state, "send_eval_result")
    result = state.get("eval_result")
    if not result:
        state["status"] = "partial"
        state["error"] = "缺少评测结果，不能回传"
        _event(state, {"event": "partial", "reason": state["error"]})
        return _leave(state, "send_eval_result")

    followup = _history_followup(state) + [
        state["eval_assistant_message"],
        {
            "role": "tool",
            "tool_call_id": state["eval_call_id"],
            "content": json.dumps(
                {
                    "quota": result["quota"],
                    "hits": result["hits"],
                    "recall_at_100": result["recall_at_100"],
                    "coverage": result["coverage"],
                    "candidates_total": result["candidates_total"],
                    "fills_total": result["fills_total"],
                },
                ensure_ascii=False,
            ),
        },
        {
            "role": "user",
            "content": (
                "这是真实融合评测结果。请与 50/50 基准比较，用一句话给出结论和停止理由；"
                "不要再调用工具，也不要输出 JSON 代码块。"
            ),
        },
    ]
    try:
        response = _request(
            state,
            chat_fn,
            followup,
            tools=None,
            label="evaluation_followup",
        )
    except RequestFailed as error:
        return _fail_request(state, "send_eval_result", error)

    message = (response.get("choices") or [{}])[0].get("message", {}) or {}
    content = (message.get("content") or "").strip()
    if message.get("tool_calls"):
        state["status"] = "partial"
        state["error"] = "评测结果回传后模型再次请求工具，未得到最终结论"
        _event(state, {"event": "partial", "reason": state["error"]})
        return _leave(state, "send_eval_result")
    if not content:
        state["status"] = "partial"
        state["error"] = "评测结果回传后最终回复为空"
        _event(state, {"event": "partial", "reason": state["error"]})
        return _leave(state, "send_eval_result")
    state["final_reply"] = content
    state["analysis_path"] = "miss_eval"
    state["decision"] = "evaluated"
    _set_stop(state, "历史未命中，评测同一目标配额一次后停止")
    return _leave(state, "send_eval_result")


def node_finish(state: AgentState) -> AgentState:
    state = _enter(state, "finish")
    baseline = state.get("baseline") or {}
    history = state.get("history_result") or {}
    evaluation = state.get("eval_result") or {}
    chosen: dict | None = None
    if history.get("hit"):
        chosen = history.get("metrics")
    elif evaluation:
        chosen = evaluation

    if chosen and baseline:
        chosen_hits = int(chosen["hits"])
        baseline_hits = int(baseline["hits"])
        better = chosen_hits > baseline_hits or (
            chosen_hits == baseline_hits
            and float(chosen["recall_at_100"]) > float(baseline["recall_at_100"])
        )
        state["conclusion"] = {
            "decision": state.get("decision"),
            "history_hit": bool(history.get("hit")),
            "baseline_quota": baseline["quota"],
            "chosen_quota": chosen.get("quota")
            or {"two_tower": state["target_quota"][0], "mind": state["target_quota"][1]},
            "chosen_source": "history" if history.get("hit") else "evaluation",
            "baseline_hits": baseline_hits,
            "chosen_hits": chosen_hits,
            "delta_hits": chosen_hits - baseline_hits,
            "delta_recall_at_100": round(
                float(chosen["recall_at_100"]) - float(baseline["recall_at_100"]), 6
            ),
            "baseline_recall_at_100": baseline["recall_at_100"],
            "chosen_recall_at_100": chosen["recall_at_100"],
            "winner": "chosen" if better else "baseline",
            "evaluation_count": int(state.get("evaluation_count", 0)),
            "final_model_reply": state.get("final_reply", ""),
            "stop_reason": state.get("stop_reason"),
        }
    else:
        state["conclusion"] = {
            "decision": state.get("decision"),
            "history_hit": bool(history.get("hit")),
            "baseline_quota": baseline.get("quota"),
            "baseline_hits": baseline.get("hits"),
            "chosen_quota": None,
            "chosen_hits": None,
            "delta_hits": None,
            "evaluation_count": int(state.get("evaluation_count", 0)),
            "winner": "not_evaluated",
            "final_model_reply": state.get("final_reply", ""),
            "stop_reason": state.get("stop_reason") or history.get("reason"),
        }

    if state.get("decision") == "reuse_history":
        state["stop_reason"] = "历史结果与当前输入和评估口径一致，直接复用，未执行融合评测"
        state["conclusion"]["stop_reason"] = state["stop_reason"]

    if state.get("status", "running") == "running":
        if not history or int(state.get("history_queries", 0)) != 1:
            state["status"] = "partial"
            state.setdefault("error", "缺少实际历史查询结果")
        elif not state.get("final_reply"):
            state["status"] = "partial"
            state.setdefault("error", "缺少有效最终回复")
        elif state.get("eval_call_id") and not evaluation:
            state["status"] = "partial"
            state.setdefault("error", "评测调用没有对应结果")
        elif history.get("hit") and int(state.get("evaluation_count", 0)) != 0:
            state["status"] = "partial"
            state.setdefault("error", "历史命中却执行了评测，违反工具顺序")
        else:
            state["status"] = "verified"

    if state.get("status") != "verified" and not state.get("stop_reason"):
        state["stop_reason"] = state.get("error") or "闭环未完成"
        if state.get("conclusion"):
            state["conclusion"]["stop_reason"] = state["stop_reason"]
    state["conclusion"]["decision"] = state.get("decision")
    state["conclusion"]["status"] = state.get("status")
    _event(
        state,
        {
            "event": "finish",
            "status": state.get("status"),
            "decision": state.get("decision"),
            "evaluation_count": int(state.get("evaluation_count", 0)),
            "stop_reason": state.get("stop_reason"),
            "error": state.get("error"),
        },
    )
    return _leave(state, "finish")


# --------------------------------------------------------------------------- 图


def _route(next_node: str):
    def route(state: AgentState) -> str:
        if state.get("status") in {"blocked", "partial", "failed"} or state.get("error"):
            return "finish"
        return next_node

    return route


def _route_after_history(state: AgentState) -> str:
    if state.get("status") in {"blocked", "partial", "failed"} or state.get("error"):
        return "finish"
    if state.get("eval_call_id"):
        return "execute_eval"
    return "finish"


def build_graph(
    *,
    chat_fn: ChatFn = post_chat,
    baseline_fn: BaselineFn = load_baseline,
    registry: ToolRegistry | None = None,
):
    registry = registry or build_registry()
    graph = StateGraph(AgentState)
    graph.add_node("load_baseline", lambda state: node_load_baseline(state, baseline_fn))
    graph.add_node(
        "request_history", lambda state: node_request_history(state, chat_fn, registry)
    )
    graph.add_node("execute_history", lambda state: node_execute_history(state, registry))
    graph.add_node(
        "send_history_result",
        lambda state: node_send_history_result(state, chat_fn, registry),
    )
    graph.add_node("execute_eval", lambda state: node_execute_eval(state, registry))
    graph.add_node(
        "send_eval_result", lambda state: node_send_eval_result(state, chat_fn)
    )
    graph.add_node("finish", node_finish)
    graph.set_entry_point("load_baseline")
    graph.add_conditional_edges("load_baseline", _route("request_history"))
    graph.add_conditional_edges("request_history", _route("execute_history"))
    graph.add_conditional_edges("execute_history", _route("send_history_result"))
    graph.add_conditional_edges(
        "send_history_result",
        _route_after_history,
        {"execute_eval": "execute_eval", "finish": "finish"},
    )
    graph.add_conditional_edges("execute_eval", _route("send_eval_result"))
    graph.add_conditional_edges("send_eval_result", _route("finish"))
    graph.add_edge("finish", END)
    return graph.compile()


# --------------------------------------------------------------------------- 状态与写盘


def _initial_state(
    *,
    config: dict,
    cache_dir: Path,
    cache_dir_source: str,
    history_root: Path,
    output_dir: Path,
    target_quota: tuple[int, int],
    started: float | None = None,
) -> AgentState:
    started = started if started is not None else time.perf_counter()
    return {
        "status": "running",
        "stage": "init",
        "nodes": [],
        "events": [
            {
                "event": "start",
                "cache_dir": str(cache_dir),
                "cache_dir_source": cache_dir_source,
                "history_root": str(history_root),
                "output_dir": str(output_dir),
                "target_quota": list(target_quota),
                "model": config.get("model"),
                "endpoint_host": host_only(config.get("base_url", "")),
            }
        ],
        "started": started,
        "deadline": started + BUDGET_SECONDS,
        "config": config,
        "cache_dir": str(cache_dir),
        "cache_dir_source": cache_dir_source,
        "history_root": str(history_root),
        "output_dir": str(output_dir),
        "target_quota": list(target_quota),
        "request_count": 0,
        "history_queries": 0,
        "evaluation_count": 0,
        "response_ids": {},
        "call_ids": {},
    }


def _finalize(state: AgentState, experiment_dir: Path) -> dict:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    events = state.get("events", [])
    (experiment_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events)
        + ("\n" if events else ""),
        encoding="utf-8",
    )
    history = state.get("history_result") or {}
    evaluation = state.get("eval_result") or {}
    baseline = state.get("baseline") or {}
    evidence = {
        "cache_dir": state.get("cache_dir"),
        "history_run": (history.get("evidence") or {}).get("run"),
        "history_fusion": (history.get("evidence") or {}).get("fusion"),
        "history_config": (history.get("evidence") or {}).get("config"),
        "baseline_run": (baseline.get("evidence") or {}).get("run") or str(BASELINE_RUN),
        "baseline_fusion": (baseline.get("evidence") or {}).get("fusion"),
        "evaluation_fusion": (evaluation.get("evidence") or {}).get("fusion"),
    }
    config = state.get("config") or {}
    summary = {
        "status": state.get("status", "blocked"),
        "decision": state.get("decision"),
        "stage": state.get("stage"),
        "graph_nodes": state.get("nodes", []),
        "request_count": int(state.get("request_count", 0)),
        "history_query_count": int(state.get("history_queries", 0)),
        "evaluation_count": int(state.get("evaluation_count", 0)),
        "tool_calls": {
            TOOL_SEARCH_HISTORY: int(state.get("history_queries", 0)),
            TOOL_EVALUATE: int(state.get("evaluation_count", 0)),
        },
        "response_ids": state.get("response_ids", {}),
        "call_ids": state.get("call_ids", {}),
        "target_quota": state.get("target_quota"),
        "baseline": baseline or None,
        "history": history or None,
        "evaluation": evaluation or None,
        "conclusion": state.get("conclusion"),
        "stop_reason": state.get("stop_reason"),
        "error": state.get("error"),
        "cache": {
            "dir": state.get("cache_dir"),
            "source": state.get("cache_dir_source"),
            "fingerprints": state.get("fingerprints"),
        },
        "model": {
            "name": config.get("model"),
            "endpoint_host": host_only(config.get("base_url", "")),
        },
        "budget": {
            "elapsed_s": round(
                time.perf_counter() - float(state.get("started", time.perf_counter())), 2
            ),
            "limit_seconds": BUDGET_SECONDS,
            "max_model_requests": MAX_MODEL_REQUESTS,
            "max_history_queries": MAX_HISTORY_QUERIES,
            "max_evaluations": MAX_EVALUATIONS,
        },
        "evidence": evidence,
    }
    (experiment_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (experiment_dir / "decision.json").write_text(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "evaluation_count": summary["evaluation_count"],
                "history_query_count": summary["history_query_count"],
                "target_quota": summary["target_quota"],
                "baseline": summary["baseline"],
                "history": summary["history"],
                "evaluation": summary["evaluation"],
                "conclusion": summary["conclusion"],
                "stop_reason": summary["stop_reason"],
                "evidence": summary["evidence"],
                "error": summary["error"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return summary


def run(
    experiment_dir: Path,
    *,
    cache_dir: Path = DEFAULT_CACHE,
    cache_dir_source: str = "default",
    history_root: Path = DEFAULT_HISTORY_ROOT,
    output_dir: Path = OUTPUT_ROOT,
    target_quota: tuple[int, int] = DEFAULT_TARGET_QUOTA,
) -> dict:
    target_quota = validate_quota(*target_quota)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "events.jsonl").write_text("", encoding="utf-8")
    started = time.perf_counter()
    load_dotenv()
    config = llm_config()

    if config is None:
        missing = [name for name in LLM_ENV if not os.environ.get(name)]
        state: AgentState = {
            "status": "blocked",
            "stage": "init",
            "nodes": [],
            "events": [
                {
                    "event": "blocked",
                    "reason": f"缺少真实 LLM 配置：{', '.join(missing)}",
                    "cache_dir": str(cache_dir),
                    "cache_dir_source": cache_dir_source,
                }
            ],
            "started": started,
            "deadline": started + BUDGET_SECONDS,
            "cache_dir": str(cache_dir),
            "cache_dir_source": cache_dir_source,
            "history_root": str(history_root),
            "output_dir": str(output_dir),
            "target_quota": list(target_quota),
            "request_count": 0,
            "history_queries": 0,
            "evaluation_count": 0,
            "response_ids": {},
            "call_ids": {},
            "error": f"缺少真实 LLM 配置：{', '.join(missing)}",
        }
        return _finalize(state, experiment_dir)

    state = _initial_state(
        config=config,
        cache_dir=cache_dir,
        cache_dir_source=cache_dir_source,
        history_root=history_root,
        output_dir=output_dir,
        target_quota=target_quota,
        started=started,
    )
    final_state = build_graph().invoke(state)
    summary = _finalize(final_state, experiment_dir)
    (experiment_dir / "baseline.json").write_text(
        json.dumps(summary.get("baseline") or {}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


# --------------------------------------------------------------------------- 本地检查


def check_cache(
    experiment_dir: Path,
    *,
    cache_dir: Path,
    cache_dir_source: str,
) -> dict:
    """只做基准 + 缓存身份核对，不请求模型、不查历史、不评测。"""
    experiment_dir.mkdir(parents=True, exist_ok=True)
    try:
        baseline, fingerprints = load_baseline(cache_dir)
    except (VerificationError, QuotaError) as error:
        result = {
            "status": "blocked",
            "cache_dir": str(cache_dir),
            "cache_dir_source": cache_dir_source,
            "error": str(error),
        }
    else:
        result = {
            "status": "ok",
            "cache_dir": str(cache_dir),
            "cache_dir_source": cache_dir_source,
            "input_hashes": fingerprints,
            "baseline": _baseline_model_view(baseline),
            "evidence": baseline.get("evidence"),
        }
    (experiment_dir / "cache_check.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _fake_eval_result(quota: tuple[int, int], *, cache_dir: Path, output_dir: Path) -> dict:
    return {
        "tool": TOOL_EVALUATE,
        "quota": {"two_tower": quota[0], "mind": quota[1]},
        "users": 128,
        "targets": 362,
        "candidates_total": 12800,
        "candidates_mean": 100.0,
        "two_tower_used_total": quota[0] * 128,
        "mind_used_total": quota[1] * 128,
        "fills_total": 0,
        "hits": 5,
        "recall_at_100": 0.013672,
        "coverage": 0.02672,
        "elapsed_s": 0.0,
        "diagnostic_union_hits_at_100": 9,
        "evidence": {
            "fusion": str(output_dir / "fake_fusion.json"),
            "cache": str(cache_dir),
            "fingerprints": {"simulated": True},
        },
    }


def _fake_call(
    call_id: str,
    name: str,
    arguments: dict,
    *,
    content: str = "",
) -> dict:
    return {
        "id": f"resp-{call_id}",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
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


def _fake_reply(content: str) -> dict:
    return {
        "id": f"resp-final-{abs(hash(content)) % 100000}",
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }


def _prepare_miss_history_root(experiment_dir: Path) -> Path:
    """复制配置与 50/50 基准，但不放 30/70 记录，保证真实历史查询返回 miss。"""
    root = experiment_dir / "regression_miss" / "history_miss"
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    source_config = DEFAULT_HISTORY_ROOT / "config.toml"
    if source_config.exists():
        shutil.copyfile(source_config, root / "config.toml")
    source_baseline = DEFAULT_HISTORY_ROOT / "runs" / "50_50.json"
    if source_baseline.exists():
        shutil.copyfile(source_baseline, runs / "50_50.json")
    return root


def run_regression_miss(
    experiment_dir: Path,
    *,
    cache_dir: Path,
    cache_dir_source: str,
    output_dir: Path,
    target_quota: tuple[int, int],
    real_eval: bool,
) -> dict:
    """本地模拟未命中：真实历史查询（临时目录无 30/70）→ fake/真实评测一次 → 最终回复。"""
    run_dir = experiment_dir / "regression_miss"
    history_root = _prepare_miss_history_root(experiment_dir)
    fake_config = {"model": "fake-model", "base_url": "http://127.0.0.1", "api_key": "fake"}
    calls = {"chat": 0, "eval": 0}

    script = [
        _fake_call(
            "call-sim-history",
            TOOL_SEARCH_HISTORY,
            {"two_tower_quota": target_quota[0], "mind_quota": target_quota[1]},
        ),
        _fake_call(
            "call-sim-eval",
            TOOL_EVALUATE,
            {"two_tower_quota": target_quota[0], "mind_quota": target_quota[1]},
        ),
        _fake_reply("本地模拟：历史未命中，评测 30/70 后停止。"),
    ]

    def fake_chat(config, messages, *, tools, timeout):
        calls["chat"] += 1
        if not script:
            raise AssertionError("本地模拟脚本已用尽")
        return script.pop(0)

    def fake_eval(arguments, context: ToolContext) -> dict:
        calls["eval"] += 1
        quota = (arguments["two_tower_quota"], arguments["mind_quota"])
        return _fake_eval_result(quota, cache_dir=context.cache_dir, output_dir=context.output_dir)

    registry = build_registry() if real_eval else build_registry(evaluate_execute=fake_eval)
    state = _initial_state(
        config=fake_config,
        cache_dir=cache_dir,
        cache_dir_source=cache_dir_source,
        history_root=history_root,
        output_dir=output_dir,
        target_quota=target_quota,
    )
    final_state = build_graph(
        chat_fn=fake_chat, baseline_fn=load_baseline, registry=registry
    ).invoke(state)
    summary = _finalize(final_state, run_dir)
    history = summary.get("history") or {}
    evaluation = summary.get("evaluation") or {}
    checks = {
        "status_verified": summary["status"] == "verified",
        "decision_evaluated": summary["decision"] == "evaluated",
        "history_query_once": summary["history_query_count"] == 1,
        "history_miss": history.get("status") == "miss",
        "evaluation_once": summary["evaluation_count"] == 1,
        "request_count_3": summary["request_count"] == 3,
        "history_call_id_returned": (summary.get("call_ids") or {}).get("history")
        == "call-sim-history",
        "eval_call_id_returned": (summary.get("call_ids") or {}).get("evaluation")
        == "call-sim-eval",
        "eval_result_kept": bool(evaluation.get("quota")),
        "tool_order_history_first": summary["history_query_count"] == 1
        and summary["evaluation_count"] == 1,
        "cache_dir_used_by_eval": evaluation.get("evidence", {}).get("cache")
        == str(cache_dir),
        "no_verified_on_partial_path": summary["status"] == "verified",
    }
    if real_eval:
        checks["real_eval_hits_5"] = evaluation.get("hits") == 5
        checks["real_eval_evidence_under_output"] = str(output_dir) in str(
            evaluation.get("evidence", {}).get("fusion")
        )
    result = {
        "mode": "local_miss_then_evaluate",
        "history_root": str(history_root),
        "cache_dir": str(cache_dir),
        "real_eval": real_eval,
        "status": summary["status"],
        "decision": summary["decision"],
        "request_count": summary["request_count"],
        "history_query_count": summary["history_query_count"],
        "evaluation_count": summary["evaluation_count"],
        "chat_calls": calls["chat"],
        "fake_eval_calls": calls["eval"],
        "history_reason": history.get("reason"),
        "checks": checks,
        "passed": all(checks.values()),
        "metrics_file": str(run_dir / "metrics.json"),
    }
    _write_json(run_dir / "regression_miss.json", result)
    return result


def run_regression_timeout(
    experiment_dir: Path,
    *,
    mode: str,
    cache_dir: Path,
    cache_dir_source: str,
    output_dir: Path,
    target_quota: tuple[int, int],
) -> dict:
    """工具成功后最终请求失败：历史命中后失败 / 评测成功后失败。"""
    run_dir = experiment_dir / f"regression_timeout_{mode}"
    fake_config = {"model": "fake-model", "base_url": "http://127.0.0.1", "api_key": "fake"}
    calls = {"chat": 0, "eval": 0}

    if mode == "history":
        history_root = DEFAULT_HISTORY_ROOT
        script = [
            _fake_call(
                "call-timeout-history",
                TOOL_SEARCH_HISTORY,
                {"two_tower_quota": target_quota[0], "mind_quota": target_quota[1]},
            ),
        ]
    else:
        history_root = _prepare_miss_history_root(experiment_dir)
        script = [
            _fake_call(
                "call-timeout-history",
                TOOL_SEARCH_HISTORY,
                {"two_tower_quota": target_quota[0], "mind_quota": target_quota[1]},
            ),
            _fake_call(
                "call-timeout-eval",
                TOOL_EVALUATE,
                {"two_tower_quota": target_quota[0], "mind_quota": target_quota[1]},
            ),
        ]

    def fake_chat(config, messages, *, tools, timeout):
        calls["chat"] += 1
        if script:
            return script.pop(0)
        raise TimeoutError(f"simulated timeout on final request ({mode})")

    def fake_eval(arguments, context: ToolContext) -> dict:
        calls["eval"] += 1
        quota = (arguments["two_tower_quota"], arguments["mind_quota"])
        return _fake_eval_result(quota, cache_dir=context.cache_dir, output_dir=context.output_dir)

    registry = build_registry(evaluate_execute=fake_eval)
    state = _initial_state(
        config=fake_config,
        cache_dir=cache_dir,
        cache_dir_source=cache_dir_source,
        history_root=history_root,
        output_dir=output_dir,
        target_quota=target_quota,
    )
    final_state = build_graph(
        chat_fn=fake_chat, baseline_fn=load_baseline, registry=registry
    ).invoke(state)
    summary = _finalize(final_state, run_dir)
    history = summary.get("history") or {}
    evaluation = summary.get("evaluation") or {}
    expected_requests = 2 if mode == "history" else 3
    checks = {
        "status_partial": summary["status"] == "partial",
        "simulated_cli_exit_2": (0 if summary["status"] == "verified" else 2) != 0,
        "request_count_expected": summary["request_count"] == expected_requests,
        "history_query_once": summary["history_query_count"] == 1,
        "not_verified": summary["status"] != "verified",
        "error_recorded": bool(summary.get("error")),
        "history_result_kept": bool(history.get("tool")),
        "tool_result_kept": bool(history.get("tool")) if mode == "history" else bool(evaluation.get("quota")),
    }
    if mode == "history":
        checks["history_hit"] = bool(history.get("hit"))
        checks["evaluation_count_0"] = summary["evaluation_count"] == 0
    else:
        checks["evaluation_count_1"] = summary["evaluation_count"] == 1
        checks["eval_result_kept"] = bool(evaluation.get("quota"))
    result = {
        "mode": f"final_request_timeout_after_{mode}",
        "cache_dir": str(cache_dir),
        "status": summary["status"],
        "simulated_cli_exit": 0 if summary["status"] == "verified" else 2,
        "request_count": summary["request_count"],
        "history_query_count": summary["history_query_count"],
        "evaluation_count": summary["evaluation_count"],
        "error": summary.get("error"),
        "checks": checks,
        "passed": all(checks.values()),
        "metrics_file": str(run_dir / "metrics.json"),
    }
    _write_json(run_dir / "regression_timeout.json", result)
    return result


def _synthetic_history(hit: bool, quota: tuple[int, int]) -> dict:
    base = {
        "tool": TOOL_SEARCH_HISTORY,
        "status": "hit" if hit else "miss",
        "hit": hit,
        "quota": {"two_tower": quota[0], "mind": quota[1]},
        "reason": "模拟命中" if hit else "模拟未命中：没有可比记录",
        "evidence": {"run": "local-sim"},
    }
    if hit:
        base["metrics"] = {
            "users": 128,
            "targets": 362,
            "candidates_total": 12800,
            "hits": 5,
            "recall_at_100": 0.013672,
            "coverage": 0.02672,
        }
        base["baseline"] = {
            "quota": {"two_tower": 50, "mind": 50},
            "hits": 4,
            "recall_at_100": 0.011719,
            "coverage": 0.026843,
            "candidates_total": 12800,
        }
        base["delta_vs_baseline"] = {"hits": 1, "recall_at_100": 0.001953}
    return base


def _synthetic_baseline() -> dict:
    return {
        "quota": {"two_tower": 50, "mind": 50},
        "hits": 4,
        "recall_at_100": 0.011719,
        "coverage": 0.026843,
        "candidates_total": 12800,
        "users": 128,
        "targets": 362,
        "fingerprints_verified": True,
        "evidence": {"run": "local-sim", "fusion": None},
    }


def _run_synthetic_case(
    *,
    script: list[dict],
    history_hit: bool,
    quota: tuple[int, int] = (30, 70),
    baseline: dict | None = None,
) -> dict:
    fake_config = {"model": "fake-model", "base_url": "http://127.0.0.1", "api_key": "fake"}
    calls = {"history": 0, "eval": 0}

    def fake_chat(config, messages, *, tools, timeout):
        if not script:
            raise AssertionError("自检脚本已用尽")
        return script.pop(0)

    def fake_history(arguments, context: ToolContext) -> dict:
        calls["history"] += 1
        return _synthetic_history(history_hit, quota)

    def fake_eval(arguments, context: ToolContext) -> dict:
        calls["eval"] += 1
        return _fake_eval_result(quota, cache_dir=context.cache_dir, output_dir=context.output_dir)

    registry = build_registry(
        history_execute=fake_history, evaluate_execute=fake_eval
    )
    state = _initial_state(
        config=fake_config,
        cache_dir=Path("synthetic-cache"),
        cache_dir_source="synthetic",
        history_root=Path("synthetic-history"),
        output_dir=Path("synthetic-output"),
        target_quota=quota,
    )
    state["baseline"] = baseline or _synthetic_baseline()
    state["fingerprints"] = {"synthetic": True}
    final_state = build_graph(
        chat_fn=fake_chat,
        baseline_fn=lambda cache_dir: (state["baseline"], state["fingerprints"]),
        registry=registry,
    ).invoke(state)
    final_state["_calls"] = calls
    return final_state


def run_self_check() -> dict:
    """合成响应验证工具顺序、预算与状态机；不联网、不读缓存。"""
    checks: list[str] = []
    history_call = _fake_call(
        "call-h", TOOL_SEARCH_HISTORY, {"two_tower_quota": 30, "mind_quota": 70}
    )
    eval_call = _fake_call(
        "call-e", TOOL_EVALUATE, {"two_tower_quota": 30, "mind_quota": 70}
    )
    final_reply = _fake_reply("停止：模拟结论。")

    # 1) 历史查询前请求评测 → blocked，评测 0 次
    case = _run_synthetic_case(script=[eval_call], history_hit=False)
    assert case["status"] == "blocked", case["status"]
    assert case["evaluation_count"] == 0
    assert case["_calls"]["eval"] == 0
    checks.append("查询前评测被拦截（blocked，拒绝执行）")

    # 2) 未知工具 → blocked
    unknown = _fake_call(
        "call-u", "train_model", {"two_tower_quota": 30, "mind_quota": 70}
    )
    case = _run_synthetic_case(script=[unknown], history_hit=False)
    assert case["status"] == "blocked"
    checks.append("未知工具被拒绝")

    # 3) 参数夹带路径/命令/脚本 → blocked（注册表只接受两个整数配额）
    injected = _fake_call(
        "call-i",
        TOOL_SEARCH_HISTORY,
        {
            "two_tower_quota": 30,
            "mind_quota": 70,
            "path": "artifacts/kuasar_small/mvp",
            "command": "python train.py",
            "script": "custom_eval.py",
        },
    )
    case = _run_synthetic_case(script=[injected], history_hit=False)
    assert case["status"] == "blocked"
    checks.append("路径/命令/自定义脚本参数被拒绝")

    # 4) 目标配额不一致 → blocked
    wrong = _fake_call(
        "call-w", TOOL_SEARCH_HISTORY, {"two_tower_quota": 50, "mind_quota": 50}
    )
    case = _run_synthetic_case(script=[wrong], history_hit=False)
    assert case["status"] == "blocked"
    checks.append("历史查询参数与任务目标不一致被拒绝")

    # 5) 命中 → reuse_history，评测 0 次，2 次请求
    case = _run_synthetic_case(script=[history_call, final_reply], history_hit=True)
    assert case["status"] == "verified", case.get("error")
    assert case["decision"] == "reuse_history"
    assert case["evaluation_count"] == 0
    assert case["history_queries"] == 1
    assert case["request_count"] == 2
    checks.append("历史命中：verified / reuse_history / 评测 0 次")

    # 6) 未命中 → 允许一次评测并同 ID 回传，3 次请求
    case = _run_synthetic_case(
        script=[history_call, eval_call, final_reply], history_hit=False
    )
    assert case["status"] == "verified", case.get("error")
    assert case["decision"] == "evaluated"
    assert case["history_queries"] == 1 and case["evaluation_count"] == 1
    assert case["request_count"] == 3
    assert case["call_ids"]["evaluation"] == "call-e"
    checks.append("未命中：一次评测 + 同 ID 回传 + 最终回复（3 次请求）")

    # 7) 未命中但模型说明原因停止 → stopped_without_eval
    case = _run_synthetic_case(
        script=[history_call, _fake_reply("历史未命中，停止。")], history_hit=False
    )
    assert case["status"] == "verified"
    assert case["decision"] == "stopped_without_eval"
    assert case["evaluation_count"] == 0
    checks.append("未命中时模型可说明原因停止（评测 0 次）")

    # 8) 未命中后返回多条 tool_calls → blocked，评测 0 次
    double = _fake_call(
        "call-e1", TOOL_EVALUATE, {"two_tower_quota": 30, "mind_quota": 70}
    )
    double["choices"][0]["message"]["tool_calls"].append(
        double["choices"][0]["message"]["tool_calls"][0]
    )
    case = _run_synthetic_case(script=[history_call, double], history_hit=False)
    assert case["status"] == "blocked"
    assert case["_calls"]["eval"] == 0
    checks.append("多条 tool_calls 被拒绝（未执行评测）")

    # 9) 命中后再次请求评测 → blocked，评测 0 次
    case = _run_synthetic_case(script=[history_call, eval_call], history_hit=True)
    assert case["status"] == "partial"
    assert case["evaluation_count"] == 0
    checks.append("历史命中后请求工具不落成功分支（partial，评测 0 次）")

    # 10) 直接复核 execute_eval 的守卫：命中后评测 / 重复评测
    registry = build_registry()
    guard_state = _initial_state(
        config={"model": "fake", "base_url": "http://127.0.0.1", "api_key": "fake"},
        cache_dir=Path("synthetic-cache"),
        cache_dir_source="synthetic",
        history_root=Path("synthetic-history"),
        output_dir=Path("synthetic-output"),
        target_quota=(30, 70),
    )
    guard_state.update(
        history_queries=1,
        history_result={"hit": True},
        eval_call_id="call-x",
        eval_arguments={"two_tower_quota": 30, "mind_quota": 70},
    )
    guarded = node_execute_eval(guard_state, registry)
    assert guarded["status"] == "blocked"
    checks.append("命中历史后评测守卫生效")

    guard_state.update(
        history_result={"hit": False},
        evaluation_count=1,
        status="running",
    )
    guarded = node_execute_eval(guard_state, registry)
    assert guarded["status"] == "blocked"
    checks.append("重复评测守卫生效")

    # 11) 模型请求次数上限
    capped = _initial_state(
        config={"model": "fake", "base_url": "http://127.0.0.1", "api_key": "fake"},
        cache_dir=Path("synthetic-cache"),
        cache_dir_source="synthetic",
        history_root=Path("synthetic-history"),
        output_dir=Path("synthetic-output"),
        target_quota=(30, 70),
    )
    capped["baseline"] = _synthetic_baseline()
    capped["fingerprints"] = {"synthetic": True}
    capped["request_count"] = MAX_MODEL_REQUESTS

    def never_called(*args, **kwargs):
        raise AssertionError("超预算仍然发起了模型请求")

    guarded = node_request_history(capped, never_called, registry)
    assert guarded["status"] == "blocked"
    assert guarded["request_count"] == MAX_MODEL_REQUESTS
    checks.append("模型请求次数上限生效（未发出额外请求）")

    return {"passed": True, "checks": checks}


# --------------------------------------------------------------------------- CLI


def parse_quota(text: str) -> tuple[int, int]:
    cleaned = text.strip().replace(",", "/").replace(":", "/")
    if cleaned.count("/") != 1:
        raise argparse.ArgumentTypeError(f"配额格式应为 30/70，收到：{text}")
    left, right = cleaned.split("/")
    try:
        quota = (int(left), int(right))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"配额必须是整数：{text}") from error
    return validate_quota(*quota)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT_DEFAULT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="缓存目录；省略时用 artifacts/kuasar_small/mvp（显式传入时绝不回退默认）",
    )
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--quota", type=parse_quota, default=DEFAULT_TARGET_QUOTA)
    parser.add_argument("--self-check", action="store_true", help="合成响应自检（不联网、不读缓存）")
    parser.add_argument("--check-cache", action="store_true", help="只核对基准与缓存身份（不请求模型）")
    parser.add_argument("--regression-miss", action="store_true", help="本地模拟未命中后评测一次")
    parser.add_argument(
        "--real-eval",
        action="store_true",
        help="与 --regression-miss 连用：执行真实融合评测（仍不请求模型）",
    )
    parser.add_argument(
        "--regression-timeout",
        choices=("history", "evaluation"),
        default=None,
        help="本地模拟工具成功后最终请求超时",
    )
    return parser.parse_args(argv)


def _cache_selection(args: argparse.Namespace) -> tuple[Path, str]:
    if args.cache_dir is None:
        return DEFAULT_CACHE, "default"
    return Path(args.cache_dir), "cli"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache_dir, cache_source = _cache_selection(args)
    try:
        if args.self_check:
            result = run_self_check()
            _write_json(args.experiment_dir / "self_check.json", result)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.check_cache:
            result = check_cache(
                args.experiment_dir, cache_dir=cache_dir, cache_dir_source=cache_source
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["status"] == "ok" else 2
        if args.regression_miss:
            result = run_regression_miss(
                args.experiment_dir,
                cache_dir=cache_dir,
                cache_dir_source=cache_source,
                output_dir=args.output_dir,
                target_quota=args.quota,
                real_eval=args.real_eval,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["passed"] else 1
        if args.regression_timeout:
            result = run_regression_timeout(
                args.experiment_dir,
                mode=args.regression_timeout,
                cache_dir=cache_dir,
                cache_dir_source=cache_source,
                output_dir=args.output_dir,
                target_quota=args.quota,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["passed"] else 1
        summary = run(
            args.experiment_dir,
            cache_dir=cache_dir,
            cache_dir_source=cache_source,
            history_root=args.history_root,
            output_dir=args.output_dir,
            target_quota=args.quota,
        )
    except QuotaError as error:
        print(f"[未验收] {error}", file=sys.stderr)
        return 2

    if summary["status"] != "verified":
        print(f"[{summary['status']}] {summary.get('error') or '闭环未完成'}", file=sys.stderr)
        return 2
    print(json.dumps(summary["conclusion"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
