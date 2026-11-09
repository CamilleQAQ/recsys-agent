"""Search 2 · Search Agent MVP：BM25 工具调用 + 一次反馈决策。

复用 Search 1 的 KuaiSAR 小样本索引与 ``BM25Index.search``，通过原生
tool_calls 让模型调用 ``search_bm25``：

1. 第一次必须原样使用 ``original_query_tokens``；
2. 收到结果后，模型可以主动停止，也可以从原查询里保留一部分 token 再查；
3. 最多 2 次 BM25 查询、3 次模型请求；用完查询预算后只允许最终回复；
4. 程序在执行工具前检查参数、重复、子集关系和预算；不自动重试。

真实工作流只启动一次。先跑本地 self-check（不联网）：

    $env:PYTHONPATH = "src"
    python -m recsys.experiments.search_agent --self-check
    python -m recsys.experiments.search_agent `
        --tokens "3538762,3181091,4061567,3525929" `
        --output-root experiments/2026-11-09-kuasar-search-agent
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import tomllib
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypedDict

from langgraph.graph import END, StateGraph

from recsys.experiments.llm_tool_call import (
    LLM_ENV,
    host_only,
    llm_config,
    load_dotenv,
    post_chat,
)
from recsys.search.bm25 import BM25Index
from recsys.search.tool import (
    TOOL_NAME,
    TOOL_SCHEMA,
    SearchBM25Tool,
    ToolRejected,
    normalize_top_k,
    normalize_tokens,
    validate_first_query,
    validate_second_query,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "search_agent.toml"
SCHEMA_VERSION = 1
TRACE_FORMAT = "recsys.search_agent.trace.v1"
MAX_ASSISTANT_PREVIEW = 300


class AgentBlocked(RuntimeError):
    """配置、索引、样本或查询来源不满足，真实工作流不应启动。"""


# ---------------------------------------------------------------------------
# 路径、哈希与错误
# ---------------------------------------------------------------------------


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_text(value: Any, limit: int = 600) -> str:
    text = str(value or "")
    return text[:limit] + ("…" if len(text) > limit else "")


def _safe_error(error: Any, config: Mapping[str, Any] | None = None) -> str:
    text = f"{type(error).__name__}: {error}" if not isinstance(error, str) else error
    key = str((config or {}).get("api_key") or "")
    if key:
        text = text.replace(key, "***")
    return _safe_text(text, 600)


# ---------------------------------------------------------------------------
# 配置与上下文
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        config = tomllib.load(file)
    for section in ("search_agent", "llm", "scope"):
        if section not in config:
            raise AgentBlocked(f"配置缺少 [{section}]")
    return config


def _session_by_id(sessions: Sequence[Mapping[str, Any]], session_id: str) -> Mapping[str, Any]:
    for session in sessions:
        if str(session.get("session_id")) == str(session_id):
            return session
    raise AgentBlocked(f"样本里找不到 session_id={session_id}")


def _session_by_tokens(
    sessions: Sequence[Mapping[str, Any]], tokens: Sequence[int]
) -> Mapping[str, Any]:
    for session in sessions:
        if [int(token) for token in session.get("keyword", [])] == list(tokens):
            return session
    raise AgentBlocked("查询 token 不属于 Search 1 样本中的任何一个会话；请使用已有 KuaiSAR 查询")


class SearchContext:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        index: BM25Index,
        tool: SearchBM25Tool,
        index_path: Path,
        sample_path: Path,
        index_sha256: str,
        sample_sha256: str,
        query_tokens: list[int],
        session_id: str,
        query_source: str,
        clicked_items_count: int,
        clicked_joined_count: int,
    ) -> None:
        self.config = config
        self.index = index
        self.tool = tool
        self.index_path = index_path
        self.sample_path = sample_path
        self.index_sha256 = index_sha256
        self.sample_sha256 = sample_sha256
        self.query_tokens = query_tokens
        self.session_id = session_id
        self.query_source = query_source
        self.clicked_items_count = clicked_items_count
        self.clicked_joined_count = clicked_joined_count

    def identity(self) -> dict[str, Any]:
        return {
            "path": _display_path(self.index_path),
            "sha256": self.index_sha256,
            "num_docs": self.index.num_docs,
            "vocab_size": self.index.vocab_size,
            "avgdl": round(self.index.avgdl, 6),
        }


def load_context(
    config: Mapping[str, Any],
    *,
    tokens: Sequence[int] | None,
    session_id: str | None,
) -> SearchContext:
    search_cfg = config["search_agent"]
    index_path = _resolve_path(search_cfg["index_path"])
    sample_path = _resolve_path(search_cfg["sample_path"])
    for path in (index_path, sample_path):
        if not path.exists():
            raise AgentBlocked(f"缺少 Search 1 产物：{_display_path(path)}")

    index_sha = _sha256_file(index_path)
    expected_index_sha = str(search_cfg.get("expected_index_sha256") or "")
    if expected_index_sha and index_sha != expected_index_sha:
        raise AgentBlocked(
            f"索引 sha256 与 Search 1 记录不一致：{index_sha} ≠ {expected_index_sha}"
        )
    sample_sha = _sha256_file(sample_path)
    expected_sample_sha = str(search_cfg.get("expected_sample_sha256") or "")
    if expected_sample_sha and sample_sha != expected_sample_sha:
        raise AgentBlocked(
            f"样本 sha256 与 Search 1 记录不一致：{sample_sha} ≠ {expected_sample_sha}"
        )

    index = BM25Index.load(index_path)
    expected_docs = int(search_cfg.get("expected_num_docs", index.num_docs))
    expected_vocab = int(search_cfg.get("expected_vocab_size", index.vocab_size))
    if index.num_docs != expected_docs or index.vocab_size != expected_vocab:
        raise AgentBlocked(
            "索引规模与配置不一致："
            f"docs={index.num_docs}/{expected_docs}, vocab={index.vocab_size}/{expected_vocab}"
        )

    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    sessions = sample.get("sessions") or []
    if not sessions:
        raise AgentBlocked("sample.json 里没有 sessions")

    explicit_tokens = list(tokens) if tokens is not None else None
    if session_id and explicit_tokens is not None:
        session = _session_by_id(sessions, session_id)
        if [int(token) for token in session.get("keyword", [])] != explicit_tokens:
            raise AgentBlocked("--tokens 与 --session-id 指向的查询不一致")
        query_source = "cli_tokens_verified_against_session"
    elif session_id:
        session = _session_by_id(sessions, session_id)
        explicit_tokens = [int(token) for token in session.get("keyword", [])]
        query_source = "sample_session"
    elif explicit_tokens is not None:
        session = _session_by_tokens(sessions, explicit_tokens)
        session_id = str(session.get("session_id"))
        query_source = "cli_tokens"
    else:
        session_id = str(search_cfg.get("default_session_id") or "")
        if not session_id:
            raise AgentBlocked("未提供 --tokens/--session-id，且配置没有 default_session_id")
        session = _session_by_id(sessions, session_id)
        explicit_tokens = [int(token) for token in session.get("keyword", [])]
        query_source = "config_default_session"

    if not explicit_tokens:
        raise AgentBlocked("查询 token 为空")
    try:
        query_tokens = list(normalize_tokens(explicit_tokens, max_tokens=int(search_cfg["max_query_tokens"])))
    except ToolRejected as error:
        raise AgentBlocked(f"查询 token 非法：{error.message}") from error

    tool = SearchBM25Tool(index, max_results=int(search_cfg["max_results"]))
    return SearchContext(
        config=dict(config),
        index=index,
        tool=tool,
        index_path=index_path,
        sample_path=sample_path,
        index_sha256=index_sha,
        sample_sha256=sample_sha,
        query_tokens=query_tokens,
        session_id=str(session_id),
        query_source=query_source,
        clicked_items_count=len(session.get("clicked_items") or []),
        clicked_joined_count=len(session.get("clicked_joined_items") or []),
    )


# ---------------------------------------------------------------------------
# 消息与工具调用解析
# ---------------------------------------------------------------------------


def _messages_first(context: SearchContext) -> list[dict[str, Any]]:
    system = (
        "你是短视频搜索 Agent。你必须在第一轮调用 search_bm25 工具。"
        "第一次 query_tokens 必须与 original_query_tokens 完全一致（顺序也不能变），"
        "不能增删改。禁止用普通文本代替工具调用，禁止并行多个 tool_calls。"
        "每次最多返回 5 条结果。"
    )
    payload = {
        "task": "在 KuaiSAR 小样本 BM25 索引中检索短视频。",
        "original_query_tokens": list(context.query_tokens),
        "session_id": context.session_id,
        "max_results_per_query": context.tool.max_results,
        "remaining_search_budget": int(context.config["search_agent"]["max_search_calls"]),
        "instruction": (
            "第一次必须原样使用 original_query_tokens 调用 search_bm25；"
            "只允许 1 个 tool_call，禁止并行。"
        ),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _followup_message(state: "AgentState", phase: str) -> dict[str, Any]:
    context: SearchContext = state["context"]
    remaining = _pending_search_budget(state)
    if phase == "feedback":
        instruction = (
            "第一次检索结果已经通过 tool 消息返回。"
            "你现在可以停止并说明原因，也可以从 original_query_tokens 中保留一部分 token 再查一次。"
            "如果继续：query_tokens 必须非空、与第一次不同、只能使用原查询 token，"
            "并且至少去掉一个 token；只允许 1 个 tool_call，禁止并行。"
            "如果停止：直接给出简短停止理由，不要调用工具。"
        )
    elif phase == "final":
        instruction = (
            "检索预算已经用完，不能再调用任何工具。"
            "请只输出一句最终结论或停止理由，不要调用工具，不要输出 JSON 代码块。"
        )
    else:
        raise ValueError(f"未知阶段：{phase}")
    payload = {
        "phase": phase,
        "original_query_tokens": list(context.query_tokens),
        "previous_queries": [list(query) for query in state.get("queries_used", [])],
        "remaining_search_budget": remaining,
        "max_results_per_query": context.tool.max_results,
        "instruction": instruction,
    }
    return {
        "role": "user",
        "content": json.dumps(payload, ensure_ascii=False),
    }


def _message(response: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return (response.get("choices") or [{}])[0].get("message") or {}
    except (AttributeError, IndexError, TypeError):
        return {}


def _message_content(response: Mapping[str, Any]) -> str:
    return str(_message(response).get("content") or "")


def _message_tool_calls(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    calls = _message(response).get("tool_calls") or []
    return calls if isinstance(calls, list) else []


def _finish_reason(response: Mapping[str, Any]) -> str:
    try:
        choice = (response.get("choices") or [{}])[0]
    except (AttributeError, IndexError, TypeError):
        return ""
    return str(choice.get("finish_reason") or "")


def _reply_status(response: Mapping[str, Any]) -> tuple[str, str, str]:
    if _message_tool_calls(response):
        return "tool_calls", "", _finish_reason(response)
    finish_reason = _finish_reason(response)
    content = _message_content(response).strip()
    if finish_reason != "stop":
        return ("truncated" if finish_reason == "length" else "non_stop"), content, finish_reason
    if not content:
        return "empty", "", finish_reason
    return "ok", content, finish_reason


def _sanitize_reply(text: str) -> tuple[str, bool]:
    """去掉模型最终回复里附带的逐条候选列表，只保留停止理由。

    候选只在本次工具返回与模型决策中传递，默认不落盘；模型若在最终回复里
    复述 item_id 列表，落盘前截断到列表之前。
    """
    stripped = str(text or "").strip()
    if not stripped:
        return "", False
    kept: list[str] = []
    for line in stripped.splitlines():
        lower = line.lower()
        if "item_id" in lower or "最终结果" in line or "最终候选" in line or "结果列表" in line:
            break
        kept.append(line)
    sanitized = "\n".join(kept).strip()
    if not sanitized:
        sanitized = "模型选择停止（候选明细未落盘）。"
    return sanitized, sanitized != stripped


def _remember_final_reply(state: "AgentState", content: str) -> str:
    sanitized, redacted = _sanitize_reply(content)
    state["final_reply"] = _safe_text(sanitized, 1000)
    state["final_reply_redacted"] = redacted
    state["final_reply_raw_chars"] = len(content)
    state["final_reply_raw_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return state["final_reply"]


def _response_summary(response: Mapping[str, Any]) -> dict[str, Any]:
    calls = []
    for call in _message_tool_calls(response):
        function = call.get("function") or {}
        calls.append(
            {
                "id": call.get("id"),
                "type": call.get("type"),
                "name": function.get("name"),
                "arguments_chars": len(str(function.get("arguments") or "")),
            }
        )
    content = _message_content(response)
    preview, _ = _sanitize_reply(content)
    return {
        "id": response.get("id"),
        "finish_reason": _finish_reason(response),
        "content_chars": len(content),
        "content_preview": _safe_text(preview, MAX_ASSISTANT_PREVIEW),
        "tool_calls": calls,
    }


def _assistant_message(response: Mapping[str, Any]) -> dict[str, Any]:
    message = _message(response)
    return {
        "role": "assistant",
        "content": str(message.get("content") or ""),
        "tool_calls": copy.deepcopy(message.get("tool_calls") or []),
    }


def _tool_message(call_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(result, ensure_ascii=False),
    }


def parse_single_search_call(response: Mapping[str, Any]) -> dict[str, Any]:
    calls = _message_tool_calls(response)
    if not calls:
        raise ToolRejected("no_tool_call", "响应里没有 tool_calls；普通文本不算检索")
    if len(calls) != 1:
        raise ToolRejected("multiple_tool_calls", f"tool_calls 数量为 {len(calls)}，要求恰好 1 条")
    call = calls[0]
    function = call.get("function") or {}
    if (call.get("type") or "function") != "function":
        raise ToolRejected("unsupported_tool_type", f"不支持的 tool call 类型：{call.get('type')}")
    if function.get("name") != TOOL_NAME:
        raise ToolRejected("unknown_tool", f"未知工具：{function.get('name')}")
    call_id = call.get("id")
    if not call_id:
        raise ToolRejected("missing_tool_call_id", "tool_call 缺少 id")
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError as error:
        raise ToolRejected("invalid_arguments_json", f"工具参数不是合法 JSON：{error}") from error
    if not isinstance(arguments, dict):
        raise ToolRejected("invalid_arguments", "工具参数必须是 JSON 对象")
    extra = set(arguments) - {"query_tokens", "top_k"}
    if extra:
        raise ToolRejected("unexpected_arguments", f"工具参数包含未知字段：{sorted(extra)}")
    query_tokens = normalize_tokens(arguments.get("query_tokens"))
    top_k = normalize_top_k(arguments.get("top_k"))
    return {
        "id": str(call_id),
        "name": TOOL_NAME,
        "query_tokens": list(query_tokens),
        "top_k": top_k,
        "arguments": {"query_tokens": list(query_tokens), "top_k": top_k},
    }


# ---------------------------------------------------------------------------
# LangGraph 状态
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    status: str  # running / verified / partial / incomplete / blocked / failed
    stage: str
    nodes: list[str]
    events: list[dict]
    started: float
    deadline: float
    config: dict
    context: Any
    llm_config: dict
    query_tokens: list[int]
    session_id: str
    request_count: int
    search_calls: int
    max_search_calls: int
    max_model_requests: int
    response_ids: dict[str, str]
    trace_steps: list[dict]
    dialogue: list[dict]
    queries_used: list[list[int]]
    tool_call_ids: list[str]
    tool_result_ids: list[str]
    tool_result_sent: bool
    route_after_feedback: str
    current_call_id: str
    current_query_tokens: list[int]
    current_top_k: int
    current_arguments: dict
    first_result_count: int
    second_result_count: int
    final_reply: str
    final_reply_redacted: bool
    final_reply_raw_chars: int
    final_reply_raw_sha256: str
    stop_code: str
    stop_reason: str
    error: str
    summary: dict
    checks: dict
    index_sha_after: str
    elapsed_s: float
    chat_fn: Any


ChatFn = Callable[..., dict]


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
    return state


def _append_step(state: AgentState, step: dict) -> None:
    steps = copy.deepcopy(state.get("trace_steps", []))
    steps.append(step)
    state["trace_steps"] = steps


def _update_last_step(state: AgentState, **updates: Any) -> None:
    steps = copy.deepcopy(state.get("trace_steps", []))
    if not steps:
        return
    steps[-1].update(updates)
    state["trace_steps"] = steps


def _pending_search_budget(state: AgentState) -> int:
    return max(0, int(state.get("max_search_calls", 2)) - int(state.get("search_calls", 0)))


def _visible(state: AgentState) -> dict[str, Any]:
    context: SearchContext = state["context"]
    return {
        "original_query_tokens": list(context.query_tokens),
        "previous_queries": [list(query) for query in state.get("queries_used", [])],
        "remaining_search_budget": _pending_search_budget(state),
        "max_results_per_query": context.tool.max_results,
    }


def _shutdown(
    state: AgentState,
    *,
    status: str,
    code: str,
    reason: str,
    error: str | None = None,
    decision_action: str = "error",
) -> AgentState:
    state["status"] = status
    state["stop_code"] = code
    state["stop_reason"] = reason
    state["error"] = error or reason
    steps = state.get("trace_steps", [])
    existing_decision = steps[-1].get("decision") if steps else None
    updates: dict[str, Any] = {
        "stop": {
            "code": code,
            "reason": reason,
            "status": status,
        }
    }
    if not existing_decision:
        updates["decision"] = {
            "action": decision_action,
            "reason": reason,
        }
    _update_last_step(state, **updates)
    _event(state, {"event": "stop", "status": status, "code": code, "reason": reason})
    return state


def _call_model(
    state: AgentState,
    *,
    phase: str,
    tools: list[dict] | None,
    messages: list[dict],
) -> tuple[AgentState, dict | None]:
    if state.get("status") != "running":
        return state, None
    max_requests = int(state.get("max_model_requests", 3))
    if int(state.get("request_count", 0)) >= max_requests:
        reason = f"模型请求次数已达到上限 {max_requests}，拒绝继续请求"
        status = "partial" if state.get("search_calls", 0) else "incomplete"
        _event(state, {"event": "rejected", "code": "request_budget_exhausted", "reason": reason})
        return _shutdown(
            state,
            status=status,
            code="request_budget_exhausted",
            reason=reason,
            error=reason,
        ), None
    remaining = float(state.get("deadline", time.perf_counter())) - time.perf_counter()
    if remaining <= 0:
        reason = "总时间预算耗尽，拒绝发起模型请求"
        status = "partial" if state.get("search_calls", 0) else "blocked"
        _event(state, {"event": "rejected", "code": "timeout", "reason": reason})
        return _shutdown(
            state,
            status=status,
            code="timeout",
            reason=reason,
            error=reason,
        ), None

    timeout = min(float(state["config"]["llm"]["request_timeout_s"]), max(1.0, remaining))
    request_index = int(state.get("request_count", 0)) + 1
    step_index = len(state.get("trace_steps", [])) + 1
    _append_step(
        state,
        {
            "request": step_index,
            "phase": phase,
            "visible": copy.deepcopy(_visible(state)),
            "tool_result_in_context": {
                "sent": bool(state.get("tool_result_sent")),
                "call_id": state.get("current_call_id") if state.get("tool_result_sent") else None,
                "count": (
                    int(state.get("second_result_count", 0))
                    if phase == "final"
                    else int(state.get("first_result_count", 0))
                )
                if state.get("tool_result_sent")
                else None,
            },
            "model_request": {
                "tools": bool(tools),
                "timeout_s": round(timeout, 2),
            },
            "response": None,
            "decision": None,
            "tool_call": None,
            "tool_return": None,
            "stop": None,
        },
    )
    state["request_count"] = request_index
    _event(
        state,
        {
            "event": "llm_request",
            "request": request_index,
            "phase": phase,
            "tools": bool(tools),
            "timeout_s": round(timeout, 2),
        },
    )
    try:
        response = state["chat_fn"](
            state["llm_config"],
            copy.deepcopy(messages),
            tools=tools,
            timeout=timeout,
        )
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, RuntimeError) as error:
        message = _safe_error(error, state.get("llm_config"))
        status = "partial" if state.get("search_calls", 0) else "incomplete"
        _update_last_step(state, response={"status": "error", "error": message})
        _event(state, {"event": "llm_error", "request": request_index, "error": message})
        return _shutdown(
            state,
            status=status,
            code="llm_request_failed",
            reason=f"第 {request_index} 次模型请求失败：{message}",
            error=message,
        ), None
    if not isinstance(response, dict):
        message = "模型返回不是 JSON 对象"
        status = "partial" if state.get("search_calls", 0) else "incomplete"
        _update_last_step(state, response={"status": "error", "error": message})
        return _shutdown(
            state,
            status=status,
            code="invalid_response",
            reason=message,
            error=message,
        ), None

    state.setdefault("response_ids", {})[phase] = str(response.get("id") or "")
    summary = _response_summary(response)
    _update_last_step(state, response=summary)
    _event(
        state,
        {
            "event": "llm_response",
            "request": request_index,
            "phase": phase,
            "response_id": summary.get("id"),
            "finish_reason": summary.get("finish_reason"),
            "tool_call_count": len(summary.get("tool_calls") or []),
            "content_chars": summary.get("content_chars", 0),
        },
    )
    return state, response


def _tool_return_update(
    state: AgentState,
    *,
    call_id: str,
    query_tokens: Sequence[int],
    top_k: int,
    result: Mapping[str, Any],
) -> None:
    _update_last_step(
        state,
        tool_return={
            "status": "ok",
            "call_id": call_id,
            "query_tokens": list(query_tokens),
            "top_k": top_k,
            "count": int(result["count"]),
            "remaining_search_budget": _pending_search_budget(state),
        },
    )


def node_request_first(state: AgentState, chat_fn: ChatFn) -> AgentState:
    state = _enter(state, "request_first")
    messages = _messages_first(state["context"])
    state["dialogue"] = messages
    state, response = _call_model(state, phase="first", tools=[TOOL_SCHEMA], messages=messages)
    if response is None:
        return _leave(state, "request_first")
    state["dialogue"] = [*state["dialogue"], _assistant_message(response)]
    try:
        parsed = parse_single_search_call(response)
        validate_first_query(parsed["query_tokens"], state["query_tokens"])
    except ToolRejected as error:
        _event(
            state,
            {"event": "tool_rejected", "code": error.code, "message": error.message},
        )
        return _leave(
            _shutdown(
                state,
                status="incomplete",
                code=f"invalid_first_request:{error.code}",
                reason=f"首轮请求未形成合法检索：{error.message}",
                error=error.message,
                decision_action="reject",
            ),
            "request_first",
        )
    state["current_call_id"] = parsed["id"]
    state["current_query_tokens"] = parsed["query_tokens"]
    state["current_top_k"] = parsed["top_k"]
    state["current_arguments"] = parsed["arguments"]
    state.setdefault("tool_call_ids", []).append(parsed["id"])
    _update_last_step(
        state,
        decision={
            "action": "search",
            "reason": "first_query_must_equal_original",
            "query_tokens": parsed["query_tokens"],
        },
        tool_call={
            "id": parsed["id"],
            "name": TOOL_NAME,
            "arguments": parsed["arguments"],
        },
    )
    _event(
        state,
        {
            "event": "tool_call_accepted",
            "request": state.get("request_count"),
            "tool_call_id": parsed["id"],
            "query_tokens": parsed["query_tokens"],
        },
    )
    return _leave(state, "request_first")


def node_execute_first(state: AgentState) -> AgentState:
    state = _enter(state, "execute_first")
    if int(state.get("search_calls", 0)) >= int(state.get("max_search_calls", 2)):
        _event(state, {"event": "rejected", "code": "search_budget_exhausted"})
        return _leave(
            _shutdown(
                state,
                status="partial",
                code="search_budget_exhausted",
                reason="第一次检索前预算已耗尽",
                decision_action="reject",
            ),
            "execute_first",
        )
    try:
        result = state["context"].tool.search(
            state["current_query_tokens"], state["current_top_k"]
        )
    except ToolRejected as error:
        _update_last_step(
            state,
            tool_return={"status": "rejected", "call_id": state["current_call_id"], "code": error.code},
        )
        return _leave(
            _shutdown(
                state,
                status="incomplete",
                code=f"first_tool_rejected:{error.code}",
                reason=f"第一次检索被拒绝：{error.message}",
                error=error.message,
                decision_action="reject",
            ),
            "execute_first",
        )
    except Exception as error:  # noqa: BLE001 - 工具异常进入失败分支
        message = _safe_error(error, state.get("llm_config"))
        _update_last_step(
            state,
            tool_return={"status": "error", "call_id": state["current_call_id"], "error": message},
        )
        return _leave(
            _shutdown(
                state,
                status="failed",
                code="tool_failed",
                reason=f"第一次检索执行失败：{message}",
                error=message,
            ),
            "execute_first",
        )

    state["search_calls"] = int(state.get("search_calls", 0)) + 1
    state["first_result_count"] = int(result["count"])
    state.setdefault("queries_used", []).append(list(state["current_query_tokens"]))
    result_with_budget = {**result, "remaining_search_budget": _pending_search_budget(state)}
    state["dialogue"] = [
        *state["dialogue"],
        _tool_message(state["current_call_id"], result_with_budget),
    ]
    state.setdefault("tool_result_ids", []).append(state["current_call_id"])
    _tool_return_update(
        state,
        call_id=state["current_call_id"],
        query_tokens=state["current_query_tokens"],
        top_k=state["current_top_k"],
        result=result,
    )
    _event(
        state,
        {
            "event": "tool_returned",
            "tool_call_id": state["current_call_id"],
            "query_tokens": state["current_query_tokens"],
            "count": int(result["count"]),
            "remaining_search_budget": _pending_search_budget(state),
        },
    )
    return _leave(state, "execute_first")


def node_request_feedback(state: AgentState, chat_fn: ChatFn) -> AgentState:
    state = _enter(state, "request_feedback")
    state["tool_result_sent"] = True
    _event(
        state,
        {
            "event": "tool_result_sent",
            "request": 2,
            "tool_call_id": state.get("current_call_id"),
            "count": state.get("first_result_count", 0),
        },
    )
    state["dialogue"] = [*state["dialogue"], _followup_message(state, "feedback")]
    state, response = _call_model(
        state,
        phase="feedback",
        tools=[TOOL_SCHEMA] if _pending_search_budget(state) > 0 else None,
        messages=state["dialogue"],
    )
    if response is None:
        return _leave(state, "request_feedback")
    state["dialogue"] = [*state["dialogue"], _assistant_message(response)]
    if not _message_tool_calls(response):
        reply_status, content, finish_reason = _reply_status(response)
        if reply_status == "ok":
            final_reply = _remember_final_reply(state, content)
            state["stop_code"] = "model_stop"
            state["stop_reason"] = final_reply
            state["status"] = "verified"
            _update_last_step(
                state,
                decision={"action": "stop", "reason": final_reply},
                stop={
                    "code": "model_stop",
                    "reason": final_reply,
                    "status": "verified",
                    "finish_reason": finish_reason,
                    "reply_status": "ok",
                },
            )
            _event(state, {"event": "model_stop", "request": state.get("request_count"), "reason": state["final_reply"]})
            state["route_after_feedback"] = "finish"
            return _leave(state, "request_feedback")
        code = (
            "reply_truncated"
            if reply_status == "truncated"
            else ("non_stop_finish_reason" if reply_status == "non_stop" else "empty_feedback")
        )
        reason = (
            f"第一次检索后的模型回复不可用（{reply_status}, finish_reason={finish_reason or 'missing'}）；"
            f"已保留 {state.get('search_calls', 0)} 次检索。"
        )
        return _leave(
            _shutdown(
                state,
                status="partial",
                code=code,
                reason=reason,
                error=reply_status,
                decision_action="stop",
            ),
            "request_feedback",
        )

    try:
        parsed = parse_single_search_call(response)
        validate_second_query(
            parsed["query_tokens"],
            state["query_tokens"],
            state.get("queries_used", []),
        )
    except ToolRejected as error:
        _update_last_step(
            state,
            tool_call={
                "name": TOOL_NAME,
                "arguments": {"raw": _safe_text(error.message, 200)},
            },
            decision={"action": "reject", "reason": error.message},
        )
        _event(state, {"event": "tool_rejected", "code": error.code, "message": error.message})
        return _leave(
            _shutdown(
                state,
                status="partial",
                code=f"invalid_second_request:{error.code}",
                reason=f"第二次检索被拒绝：{error.message}",
                error=error.message,
                decision_action="reject",
            ),
            "request_feedback",
        )

    state["current_call_id"] = parsed["id"]
    state["current_query_tokens"] = parsed["query_tokens"]
    state["current_top_k"] = parsed["top_k"]
    state["current_arguments"] = parsed["arguments"]
    state.setdefault("tool_call_ids", []).append(parsed["id"])
    preview = _safe_text(_message_content(response), MAX_ASSISTANT_PREVIEW)
    _update_last_step(
        state,
        decision={
            "action": "search",
            "reason": preview or "model_requested_second_query",
            "query_tokens": parsed["query_tokens"],
        },
        tool_call={
            "id": parsed["id"],
            "name": TOOL_NAME,
            "arguments": parsed["arguments"],
        },
    )
    _event(
        state,
        {
            "event": "tool_call_accepted",
            "request": state.get("request_count"),
            "tool_call_id": parsed["id"],
            "query_tokens": parsed["query_tokens"],
        },
    )
    state["route_after_feedback"] = "execute_second"
    return _leave(state, "request_feedback")


def node_execute_second(state: AgentState) -> AgentState:
    state = _enter(state, "execute_second")
    if int(state.get("search_calls", 0)) >= int(state.get("max_search_calls", 2)):
        _event(state, {"event": "rejected", "code": "search_budget_exhausted"})
        return _leave(
            _shutdown(
                state,
                status="partial",
                code="search_budget_exhausted",
                reason="第二次检索前预算已耗尽",
                decision_action="reject",
            ),
            "execute_second",
        )
    try:
        result = state["context"].tool.search(
            state["current_query_tokens"], state["current_top_k"]
        )
    except ToolRejected as error:
        _update_last_step(
            state,
            tool_return={"status": "rejected", "call_id": state["current_call_id"], "code": error.code},
        )
        return _leave(
            _shutdown(
                state,
                status="partial",
                code=f"second_tool_rejected:{error.code}",
                reason=f"第二次检索被拒绝：{error.message}",
                error=error.message,
                decision_action="reject",
            ),
            "execute_second",
        )
    except Exception as error:  # noqa: BLE001 - 工具异常进入失败分支
        message = _safe_error(error, state.get("llm_config"))
        _update_last_step(
            state,
            tool_return={"status": "error", "call_id": state["current_call_id"], "error": message},
        )
        return _leave(
            _shutdown(
                state,
                status="failed",
                code="second_tool_failed",
                reason=f"第二次检索执行失败：{message}",
                error=message,
            ),
            "execute_second",
        )

    state["search_calls"] = int(state.get("search_calls", 0)) + 1
    state["second_result_count"] = int(result["count"])
    state.setdefault("queries_used", []).append(list(state["current_query_tokens"]))
    result_with_budget = {**result, "remaining_search_budget": _pending_search_budget(state)}
    state["dialogue"] = [
        *state["dialogue"],
        _tool_message(state["current_call_id"], result_with_budget),
    ]
    state.setdefault("tool_result_ids", []).append(state["current_call_id"])
    _tool_return_update(
        state,
        call_id=state["current_call_id"],
        query_tokens=state["current_query_tokens"],
        top_k=state["current_top_k"],
        result=result,
    )
    _event(
        state,
        {
            "event": "tool_returned",
            "tool_call_id": state["current_call_id"],
            "query_tokens": state["current_query_tokens"],
            "count": int(result["count"]),
            "remaining_search_budget": _pending_search_budget(state),
        },
    )
    return _leave(state, "execute_second")


def node_request_final(state: AgentState, chat_fn: ChatFn) -> AgentState:
    state = _enter(state, "request_final")
    state["dialogue"] = [*state["dialogue"], _followup_message(state, "final")]
    state, response = _call_model(
        state,
        phase="final",
        tools=None,
        messages=state["dialogue"],
    )
    if response is None:
        return _leave(state, "request_final")
    state["dialogue"] = [*state["dialogue"], _assistant_message(response)]
    calls = _message_tool_calls(response)
    if calls:
        call = calls[0]
        function = call.get("function") or {}
        _update_last_step(
            state,
            tool_call={
                "name": function.get("name"),
                "id": call.get("id"),
                "arguments": {"raw": _safe_text(function.get("arguments"), 200)},
            },
            tool_return={
                "status": "rejected",
                "code": "final_tool_call_rejected",
                "message": "查询预算用尽后禁止工具调用",
            },
        )
        return _leave(
            _shutdown(
                state,
                status="partial",
                code="final_tool_call_rejected",
                reason="预算用尽后的最终回复再次请求工具，执行器拒绝且不检索",
                error="final response contained tool_calls",
                decision_action="stop",
            ),
            "request_final",
        )

    reply_status, content, finish_reason = _reply_status(response)
    if reply_status == "ok":
        final_reply = _remember_final_reply(state, content)
        state["stop_code"] = "budget_exhausted_final"
        state["stop_reason"] = final_reply
        state["status"] = "verified"
        _update_last_step(
            state,
            decision={"action": "stop", "reason": final_reply},
            stop={
                "code": "budget_exhausted_final",
                "reason": final_reply,
                "status": "verified",
                "finish_reason": finish_reason,
                "reply_status": "ok",
            },
        )
        _event(state, {"event": "model_final", "request": state.get("request_count"), "reason": state["final_reply"]})
        return _leave(state, "request_final")

    code = (
        "reply_truncated"
        if reply_status == "truncated"
        else ("non_stop_finish_reason" if reply_status == "non_stop" else "empty_final_reply")
    )
    reason = (
        f"最终回复不可用（{reply_status}, finish_reason={finish_reason or 'missing'}）；"
        f"已保留 {state.get('search_calls', 0)} 次检索。"
    )
    return _leave(
        _shutdown(
            state,
            status="partial",
            code=code,
            reason=reason,
            error=reply_status,
            decision_action="stop",
        ),
        "request_final",
    )


def node_finish(state: AgentState) -> AgentState:
    state = _enter(state, "finish")
    if state.get("status", "running") == "running":
        if (
            int(state.get("search_calls", 0)) >= 1
            and state.get("final_reply")
            and state.get("stop_code") in {"model_stop", "budget_exhausted_final"}
        ):
            state["status"] = "verified"
        elif int(state.get("search_calls", 0)) == 0:
            state["status"] = "incomplete"
            state.setdefault("stop_code", "no_successful_search")
            state.setdefault("stop_reason", "没有成功的 BM25 检索")
            state.setdefault("error", state["stop_reason"])
        else:
            state["status"] = "partial"
            state.setdefault("stop_code", "incomplete_workflow")
            state.setdefault("stop_reason", "工作流未形成合法最终回复")
            state.setdefault("error", state["stop_reason"])

    max_requests = int(state.get("max_model_requests", 3))
    max_searches = int(state.get("max_search_calls", 2))
    if int(state.get("request_count", 0)) > max_requests:
        state["status"] = "partial"
        state["stop_code"] = "request_budget_exceeded"
        state["stop_reason"] = f"模型请求次数超过 {max_requests}"
        state["error"] = state["stop_reason"]
    if int(state.get("search_calls", 0)) > max_searches:
        state["status"] = "partial"
        state["stop_code"] = "search_budget_exceeded"
        state["stop_reason"] = f"BM25 查询次数超过 {max_searches}"
        state["error"] = state["stop_reason"]

    context: SearchContext = state["context"]
    try:
        state["index_sha_after"] = _sha256_file(context.index_path)
    except OSError as error:
        state["index_sha_after"] = ""
        state["status"] = "partial"
        state["stop_code"] = "index_rehash_failed"
        state["stop_reason"] = f"运行后索引哈希失败：{error}"
        state["error"] = state["stop_reason"]
    if state.get("index_sha_after") != context.index_sha256:
        state["status"] = "partial"
        state["stop_code"] = "index_modified"
        state["stop_reason"] = "运行前后索引 sha256 不一致；原索引可能被改写"
        state["error"] = state["stop_reason"]

    search_calls = int(state.get("search_calls", 0))
    request_count = int(state.get("request_count", 0))
    queries_used = [list(query) for query in state.get("queries_used", [])]
    return_counts = [
        int(state.get("first_result_count", 0)),
    ]
    if search_calls >= 2:
        return_counts.append(int(state.get("second_result_count", 0)))
    second_query_valid = True
    if search_calls >= 2:
        try:
            validate_second_query(queries_used[1], state["query_tokens"], [queries_used[0]])
        except (ToolRejected, IndexError):
            second_query_valid = False
    tool_roundtrip = list(state.get("tool_call_ids", [])) == list(state.get("tool_result_ids", []))
    trace_has_results = any(
        isinstance(step.get("tool_return"), dict) and "results" in step["tool_return"]
        for step in state.get("trace_steps", [])
    )
    reply_has_item_id = any(
        "item_id" in str((step.get("decision") or {}).get("reason") or "").lower()
        for step in state.get("trace_steps", [])
    )
    checks = {
        "index_loaded": context.index.num_docs == int(state["config"]["search_agent"]["expected_num_docs"]),
        "index_unchanged": state.get("index_sha_after") == context.index_sha256,
        "native_tool_call": bool(state.get("tool_call_ids")),
        "tool_result_sent": bool(state.get("tool_result_sent")),
        "tool_call_id_roundtrip": tool_roundtrip,
        "return_counts_present": bool(return_counts) and all(count >= 0 for count in return_counts),
        "request_budget_ok": request_count <= max_requests,
        "search_budget_ok": search_calls <= max_searches,
        "second_query_rule_ok": second_query_valid,
        "final_reply_ok": state.get("status") == "verified" and bool(state.get("final_reply")),
        "no_candidate_dump": not trace_has_results and not reply_has_item_id,
    }
    state["checks"] = checks
    state["summary"] = {
        "status": state.get("status"),
        "request_count": request_count,
        "search_calls": search_calls,
        "max_model_requests": max_requests,
        "max_search_calls": max_searches,
        "stop_code": state.get("stop_code"),
        "stop_reason": state.get("stop_reason"),
        "final_reply": state.get("final_reply", ""),
        "final_reply_redacted": bool(state.get("final_reply_redacted")),
        "final_reply_raw_chars": int(state.get("final_reply_raw_chars", 0)),
        "final_reply_raw_sha256": state.get("final_reply_raw_sha256", ""),
        "queries_used": queries_used,
        "return_counts": return_counts,
        "index_sha256_before": context.index_sha256,
        "index_sha256_after": state.get("index_sha_after"),
        "elapsed_s": round(time.perf_counter() - float(state.get("started", time.perf_counter())), 3),
    }
    _update_last_step(
        state,
        stop={
            "code": state.get("stop_code"),
            "reason": state.get("stop_reason"),
            "status": state.get("status"),
        },
    )
    _event(
        state,
        {
            "event": "finish",
            "status": state.get("status"),
            "request_count": request_count,
            "search_calls": search_calls,
            "stop_code": state.get("stop_code"),
        },
    )
    return _leave(state, "finish")


def _route(next_node: str):
    def route(state: AgentState) -> str:
        if state.get("status") != "running":
            return "finish"
        return next_node

    return route


def _route_after_feedback(state: AgentState) -> str:
    if state.get("status") != "running":
        return "finish"
    return "execute_second" if state.get("route_after_feedback") == "execute_second" else "finish"


def _route_after_second(state: AgentState) -> str:
    if state.get("status") != "running":
        return "finish"
    return "request_final"


def build_graph(chat_fn: ChatFn):
    graph = StateGraph(AgentState)
    graph.add_node("request_first", lambda state: node_request_first(state, chat_fn))
    graph.add_node("execute_first", node_execute_first)
    graph.add_node("request_feedback", lambda state: node_request_feedback(state, chat_fn))
    graph.add_node("execute_second", node_execute_second)
    graph.add_node("request_final", lambda state: node_request_final(state, chat_fn))
    graph.add_node("finish", node_finish)
    graph.set_entry_point("request_first")
    graph.add_conditional_edges("request_first", _route("execute_first"))
    graph.add_conditional_edges("execute_first", _route("request_feedback"))
    graph.add_conditional_edges("request_feedback", _route_after_feedback)
    graph.add_conditional_edges("execute_second", _route_after_second)
    graph.add_edge("request_final", "finish")
    graph.add_edge("finish", END)
    return graph.compile()


def run_workflow(
    context: SearchContext,
    config: Mapping[str, Any],
    *,
    llm_cfg: Mapping[str, Any],
    chat_fn: ChatFn = post_chat,
    started_at: float | None = None,
) -> AgentState:
    start = started_at if started_at is not None else time.perf_counter()
    state: AgentState = {
        "status": "running",
        "stage": "init",
        "nodes": [],
        "events": [
            {
                "event": "start",
                "model": llm_cfg.get("model"),
                "endpoint_host": host_only(str(llm_cfg.get("base_url", ""))),
                "max_model_requests": int(config["search_agent"]["max_model_requests"]),
                "max_search_calls": int(config["search_agent"]["max_search_calls"]),
                "query_tokens": list(context.query_tokens),
                "session_id": context.session_id,
            }
        ],
        "started": start,
        "deadline": start + float(config["llm"]["total_budget_s"]),
        "config": dict(config),
        "context": context,
        "llm_config": dict(llm_cfg),
        "query_tokens": list(context.query_tokens),
        "session_id": context.session_id,
        "request_count": 0,
        "search_calls": 0,
        "max_search_calls": int(config["search_agent"]["max_search_calls"]),
        "max_model_requests": int(config["search_agent"]["max_model_requests"]),
        "response_ids": {},
        "trace_steps": [],
        "dialogue": [],
        "queries_used": [],
        "tool_call_ids": [],
        "tool_result_ids": [],
        "tool_result_sent": False,
        "route_after_feedback": "finish",
        "first_result_count": 0,
        "second_result_count": 0,
        "final_reply": "",
        "final_reply_redacted": False,
        "final_reply_raw_chars": 0,
        "final_reply_raw_sha256": "",
        "stop_code": "",
        "stop_reason": "",
        "error": "",
        "chat_fn": chat_fn,  # type: ignore[typeddict-unknown-key]
    }
    graph = build_graph(chat_fn)
    return graph.invoke(state)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 写盘
# ---------------------------------------------------------------------------


def _checks_for_blocked(state: AgentState) -> dict[str, bool]:
    return {
        "index_loaded": False,
        "index_unchanged": False,
        "native_tool_call": False,
        "tool_result_sent": False,
        "tool_call_id_roundtrip": False,
        "return_counts_present": False,
        "request_budget_ok": int(state.get("request_count", 0)) <= int(
            state.get("max_model_requests", 3)
        ),
        "search_budget_ok": int(state.get("search_calls", 0)) <= int(
            state.get("max_search_calls", 2)
        ),
        "second_query_rule_ok": True,
        "final_reply_ok": False,
        "no_candidate_dump": True,
    }


def _finalize(
    state: AgentState,
    *,
    run_dir: Path,
    config: Mapping[str, Any],
    context: SearchContext | None,
    command: str,
    started_wall: str,
    elapsed_s: float,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    checks = state.get("checks") or _checks_for_blocked(state)
    if context is not None:
        # blocked 也可能发生在索引已加载、只是缺 LLM 配置时；索引身份仍应可核对。
        checks["index_loaded"] = True
        checks["index_unchanged"] = state.get("index_sha_after") == context.index_sha256
    summary = state.get("summary") or {
        "status": state.get("status"),
        "request_count": int(state.get("request_count", 0)),
        "search_calls": int(state.get("search_calls", 0)),
        "max_model_requests": int(state.get("max_model_requests", 3)),
        "max_search_calls": int(state.get("max_search_calls", 2)),
        "stop_code": state.get("stop_code"),
        "stop_reason": state.get("stop_reason"),
        "final_reply": state.get("final_reply", ""),
        "final_reply_redacted": bool(state.get("final_reply_redacted")),
        "final_reply_raw_chars": int(state.get("final_reply_raw_chars", 0)),
        "final_reply_raw_sha256": state.get("final_reply_raw_sha256", ""),
        "queries_used": [list(query) for query in state.get("queries_used", [])],
        "return_counts": [],
        "index_sha256_before": context.index_sha256 if context else None,
        "index_sha256_after": state.get("index_sha_after"),
        "elapsed_s": round(elapsed_s, 3),
    }
    state["summary"] = summary
    state["checks"] = checks
    passed = state.get("status") == "verified" and all(bool(value) for value in checks.values())
    model = state.get("llm_config") or {}
    index_identity = context.identity() if context else None
    query_payload = (
        {
            "tokens": list(context.query_tokens),
            "session_id": context.session_id,
            "source": context.query_source,
            "clicked_items_count": context.clicked_items_count,
            "clicked_joined_count": context.clicked_joined_count,
        }
        if context
        else None
    )
    trace = {
        "format": TRACE_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "status": state.get("status"),
        "query": query_payload,
        "index": index_identity,
        "rules": {
            "tool": TOOL_NAME,
            "max_results_per_query": int(config["search_agent"]["max_results"]),
            "max_search_calls": int(config["search_agent"]["max_search_calls"]),
            "max_model_requests": int(config["search_agent"]["max_model_requests"]),
            "first_query": "must_equal_original",
            "second_query": "non_empty_true_subset_of_original_and_different",
            "candidates_persisted": False,
        },
        "steps": state.get("trace_steps", []),
        "checks": checks,
        "summary": summary,
        "candidates_persisted": False,
        "final_reply_redacted": bool(summary.get("final_reply_redacted")),
        "persistence_note": (
            "候选只在本次工具返回与模型决策中传递；落盘只保留数量、停止理由与哈希。"
        ),
        "error": state.get("error"),
    }
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "phase": config["search_agent"]["phase"],
        "status": state.get("status"),
        "passed": passed,
        "model": {
            "name": model.get("model"),
            "endpoint_host": host_only(str(model.get("base_url", ""))) if model else None,
        },
        "index": index_identity,
        "query": query_payload,
        "budget": {
            "request_count": int(state.get("request_count", 0)),
            "max_model_requests": int(config["search_agent"]["max_model_requests"]),
            "search_calls": int(state.get("search_calls", 0)),
            "max_search_calls": int(config["search_agent"]["max_search_calls"]),
            "elapsed_s": round(elapsed_s, 3),
            "total_budget_s": float(config["llm"]["total_budget_s"]),
        },
        "steps": state.get("trace_steps", []),
        "checks": checks,
        "summary": summary,
        "candidates_persisted": False,
        "final_reply_redacted": bool(summary.get("final_reply_redacted")),
        "persistence_note": (
            "候选只在本次工具返回与模型决策中传递；落盘只保留数量、停止理由与哈希。"
        ),
        "error": state.get("error"),
        "scope": dict(config.get("scope") or {}),
        "run": {
            "command": command,
            "started_at": started_wall,
            "elapsed_s": round(elapsed_s, 3),
            "output_dir": _display_path(run_dir),
        },
    }
    run_meta = {
        "format": "recsys.search_agent.run_meta.v1",
        "command": command,
        "started_at": started_wall,
        "elapsed_s": round(elapsed_s, 3),
        "status": state.get("status"),
        "passed": passed,
        "model": metrics["model"],
        "index": index_identity,
        "query": query_payload,
        "output_dir": _display_path(run_dir),
        "trace_file": _display_path(run_dir / "trace.json"),
        "metrics_file": _display_path(run_dir / "metrics.json"),
        "candidates_persisted": False,
    }
    (run_dir / "trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"metrics": metrics, "trace": trace, "run_meta": run_meta, "passed": passed}


def _blocked_state(reason: str, config: Mapping[str, Any]) -> AgentState:
    return {
        "status": "blocked",
        "stage": "preflight",
        "nodes": [],
        "events": [{"event": "blocked", "reason": reason}],
        "request_count": 0,
        "search_calls": 0,
        "max_search_calls": int(config["search_agent"]["max_search_calls"]),
        "max_model_requests": int(config["search_agent"]["max_model_requests"]),
        "trace_steps": [],
        "queries_used": [],
        "tool_call_ids": [],
        "tool_result_ids": [],
        "final_reply": "",
        "final_reply_redacted": False,
        "final_reply_raw_chars": 0,
        "final_reply_raw_sha256": "",
        "stop_code": "blocked",
        "stop_reason": reason,
        "error": reason,
    }


# ---------------------------------------------------------------------------
# 真实运行
# ---------------------------------------------------------------------------


def run(
    config: Mapping[str, Any],
    *,
    tokens: Sequence[int] | None,
    session_id: str | None,
    output_root: Path,
    command: str,
) -> dict[str, Any]:
    started_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    started = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"run-{run_id}-{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        context = load_context(config, tokens=tokens, session_id=session_id)
    except (AgentBlocked, OSError, ValueError, json.JSONDecodeError) as error:
        reason = f"加载索引/样本/查询失败：{_safe_error(error)}"
        state = _blocked_state(reason, config)
        state["elapsed_s"] = round(time.perf_counter() - started, 3)
        return _finalize(
            state,
            run_dir=run_dir,
            config=config,
            context=None,
            command=command,
            started_wall=started_wall,
            elapsed_s=time.perf_counter() - started,
        )

    load_dotenv()
    llm_cfg = llm_config()
    if llm_cfg is None:
        missing = [name for name in LLM_ENV if not os.environ.get(name)]
        reason = f"缺少真实 LLM 配置：{', '.join(missing)}"
        state = _blocked_state(reason, config)
        state["max_model_requests"] = int(config["search_agent"]["max_model_requests"])
        state["max_search_calls"] = int(config["search_agent"]["max_search_calls"])
        state["summary"] = {
            "status": "blocked",
            "request_count": 0,
            "search_calls": 0,
            "max_model_requests": int(config["search_agent"]["max_model_requests"]),
            "max_search_calls": int(config["search_agent"]["max_search_calls"]),
            "stop_code": "blocked",
            "stop_reason": reason,
            "final_reply": "",
            "final_reply_redacted": False,
            "final_reply_raw_chars": 0,
            "final_reply_raw_sha256": "",
            "queries_used": [],
            "return_counts": [],
            "index_sha256_before": context.index_sha256,
            "index_sha256_after": context.index_sha256,
            "elapsed_s": round(time.perf_counter() - started, 3),
        }
        state["index_sha_after"] = context.index_sha256
        return _finalize(
            state,
            run_dir=run_dir,
            config=config,
            context=context,
            command=command,
            started_wall=started_wall,
            elapsed_s=time.perf_counter() - started,
        )

    state = run_workflow(context, config, llm_cfg=llm_cfg, chat_fn=post_chat, started_at=started)
    state["elapsed_s"] = round(time.perf_counter() - started, 3)
    return _finalize(
        state,
        run_dir=run_dir,
        config=config,
        context=context,
        command=command,
        started_wall=started_wall,
        elapsed_s=time.perf_counter() - started,
    )


# ---------------------------------------------------------------------------
# 本地 self-check（不联网）
# ---------------------------------------------------------------------------


class ScriptedChat:
    def __init__(self, responses: Sequence[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, config, messages, *, tools, timeout):
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
                "timeout": timeout,
            }
        )
        index = len(self.calls) - 1
        if index >= len(self._responses):
            return _text_response("脚本动作已执行完毕，停止。")
        response = self._responses[index]
        if isinstance(response, Exception):
            raise response
        return copy.deepcopy(response)


def _tool_response(call_id: str, tokens: Sequence[int], top_k: int = 5) -> dict[str, Any]:
    return {
        "id": f"resp-{call_id}",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": TOOL_NAME,
                                "arguments": json.dumps(
                                    {"query_tokens": list(tokens), "top_k": top_k}
                                ),
                            },
                        }
                    ],
                },
            }
        ],
    }


def _text_response(text: str, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": f"resp-text-{abs(hash(text)) % 100000}",
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": text},
            }
        ],
    }


def _fake_run(
    context: SearchContext,
    config: Mapping[str, Any],
    responses: Sequence[Any],
) -> tuple[AgentState, ScriptedChat]:
    chat = ScriptedChat(responses)
    final = run_workflow(
        context,
        config,
        llm_cfg={"model": "fake-model", "base_url": "http://127.0.0.1", "api_key": "fake-key"},
        chat_fn=chat,
    )
    return final, chat


def _check(checks: list[dict[str, Any]], check_id: str, fn: Callable[[], str]) -> None:
    try:
        detail = fn()
    except Exception as error:  # noqa: BLE001 - 自检失败也写入报告
        checks.append({"id": check_id, "status": "fail", "detail": _safe_error(error)})
    else:
        checks.append({"id": check_id, "status": "pass", "detail": detail})


def run_self_check(config: Mapping[str, Any], *, output_root: Path) -> dict[str, Any]:
    context = load_context(config, tokens=None, session_id=None)
    checks: list[dict[str, Any]] = []

    def one_query_stop() -> str:
        final, chat = _fake_run(
            context,
            config,
            [
                _tool_response("call-first", context.query_tokens),
                _text_response("第一次检索已返回结果，选择停止。"),
            ],
        )
        assert final["status"] == "verified", final["status"]
        assert final["summary"]["search_calls"] == 1, final["summary"]
        assert final["summary"]["request_count"] == 2, final["summary"]
        assert final["summary"]["return_counts"] == [final["first_result_count"]]
        assert final["final_reply"]
        return "一次检索 + 模型主动停止 → verified；2 次请求 / 1 次检索"

    def two_query_then_final() -> str:
        second = list(context.query_tokens[:-1])
        assert second and second != list(context.query_tokens)
        final, _ = _fake_run(
            context,
            config,
            [
                _tool_response("call-first", context.query_tokens),
                _tool_response("call-second", second),
                _text_response("第二次检索已返回结果，预算用尽，停止。"),
            ],
        )
        assert final["status"] == "verified", final["status"]
        assert final["summary"]["search_calls"] == 2, final["summary"]
        assert final["summary"]["request_count"] == 3, final["summary"]
        assert final["summary"]["queries_used"][1] == second
        return "两次检索 + 最终回复 → verified；3 次请求 / 2 次检索"

    def first_query_mismatch() -> str:
        bad = list(context.query_tokens[:-1]) + [999999]
        final, _ = _fake_run(
            context,
            config,
            [_tool_response("call-bad-first", bad)],
        )
        assert final["status"] == "incomplete", final["status"]
        assert final["summary"]["search_calls"] == 0, final["summary"]
        assert "first_query_mismatch" in str(final.get("stop_code")), final.get("stop_code")
        return "首查不是原查询 → 不执行工具，incomplete"

    def second_not_subset() -> str:
        final, _ = _fake_run(
            context,
            config,
            [
                _tool_response("call-first", context.query_tokens),
                _tool_response("call-bad-second", [999999]),
            ],
        )
        assert final["status"] == "partial", final["status"]
        assert final["summary"]["search_calls"] == 1, final["summary"]
        assert "second_query_not_subset" in str(final.get("stop_code")), final.get("stop_code")
        return "第二次使用非原子 token → 拒绝，保留第一次检索"

    def second_duplicate() -> str:
        final, _ = _fake_run(
            context,
            config,
            [
                _tool_response("call-first", context.query_tokens),
                _tool_response("call-dup", context.query_tokens),
            ],
        )
        assert final["status"] == "partial", final["status"]
        assert final["summary"]["search_calls"] == 1, final["summary"]
        assert "duplicate_query" in str(final.get("stop_code")), final.get("stop_code")
        return "第二次与第一次相同 → 拒绝"

    def second_empty() -> str:
        final, _ = _fake_run(
            context,
            config,
            [
                _tool_response("call-first", context.query_tokens),
                _tool_response("call-empty", []),
            ],
        )
        assert final["status"] == "partial", final["status"]
        assert "empty_query" in str(final.get("stop_code")), final.get("stop_code")
        return "第二次查询为空 → 拒绝"

    def final_tool_call_rejected() -> str:
        second = list(context.query_tokens[:-1])
        final, _ = _fake_run(
            context,
            config,
            [
                _tool_response("call-first", context.query_tokens),
                _tool_response("call-second", second),
                _tool_response("call-final", second),
            ],
        )
        assert final["status"] == "partial", final["status"]
        assert final["summary"]["search_calls"] == 2, final["summary"]
        assert "final_tool_call_rejected" in str(final.get("stop_code")), final.get("stop_code")
        return "预算用尽后的最终回复再次调用工具 → 拒绝，保留两次检索"

    def truncated_stop_rejected() -> str:
        final, _ = _fake_run(
            context,
            config,
            [
                _tool_response("call-first", context.query_tokens),
                _text_response("这段停止说明被截断", finish_reason="length"),
            ],
        )
        assert final["status"] == "partial", final["status"]
        assert final["summary"]["search_calls"] == 1, final["summary"]
        assert "reply_truncated" in str(final.get("stop_code")), final.get("stop_code")
        return "停止说明 finish_reason=length → partial，保留检索"

    def tool_call_id_roundtrip() -> str:
        final, chat = _fake_run(
            context,
            config,
            [
                _tool_response("call-roundtrip", context.query_tokens),
                _text_response("停止。"),
            ],
        )
        assert final["summary"]["search_calls"] == 1
        second_messages = chat.calls[1]["messages"]
        tool_messages = [item for item in second_messages if item.get("role") == "tool"]
        assert len(tool_messages) == 1, tool_messages
        assert tool_messages[0]["tool_call_id"] == "call-roundtrip"
        payload = json.loads(tool_messages[0]["content"])
        assert payload["count"] == final["first_result_count"]
        return "第二次请求包含同一 tool_call_id 的 tool 消息，返回数量一致"

    def budget_limits() -> str:
        second = list(context.query_tokens[:-1])
        final, _ = _fake_run(
            context,
            config,
            [
                _tool_response("call-first", context.query_tokens),
                _tool_response("call-second", second),
                _text_response("预算用尽，停止。"),
            ],
        )
        assert final["summary"]["request_count"] <= final["summary"]["max_model_requests"]
        assert final["summary"]["search_calls"] <= final["summary"]["max_search_calls"]
        return "请求 3/3、检索 2/2，预算未超限"

    def no_candidate_dump() -> str:
        final, _ = _fake_run(
            context,
            config,
            [
                _tool_response("call-first", context.query_tokens),
                _text_response("停止。"),
            ],
        )
        assert final["checks"]["no_candidate_dump"] is True
        return "轨迹只保留数量，不保存候选明细"

    def candidate_reply_redacted() -> str:
        final, _ = _fake_run(
            context,
            config,
            [
                _tool_response("call-first", context.query_tokens),
                _text_response("相关性足够，停止。\n\n最终结果：\n1. item_id 123 — demo（9.9）"),
            ],
        )
        assert final["status"] == "verified", final["status"]
        assert final["final_reply_redacted"] is True
        assert "item_id" not in final["final_reply"].lower()
        assert final["checks"]["no_candidate_dump"] is True
        assert final["final_reply_raw_sha256"]
        return "最终回复里的逐条候选在落盘前被截断，只保留停止理由"

    _check(checks, "one_query_stop", one_query_stop)
    _check(checks, "two_query_then_final", two_query_then_final)
    _check(checks, "first_query_mismatch", first_query_mismatch)
    _check(checks, "second_not_subset", second_not_subset)
    _check(checks, "second_duplicate", second_duplicate)
    _check(checks, "second_empty", second_empty)
    _check(checks, "final_tool_call_rejected", final_tool_call_rejected)
    _check(checks, "truncated_stop_rejected", truncated_stop_rejected)
    _check(checks, "tool_call_id_roundtrip", tool_call_id_roundtrip)
    _check(checks, "budget_limits", budget_limits)
    _check(checks, "no_candidate_dump", no_candidate_dump)
    _check(checks, "candidate_reply_redacted", candidate_reply_redacted)

    result = {
        "format": "recsys.search_agent.self_check.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass" if all(item["status"] == "pass" for item in checks) else "fail",
        "passed": all(item["status"] == "pass" for item in checks),
        "total": len(checks),
        "checks": checks,
        "index": context.identity(),
        "query": {
            "session_id": context.session_id,
            "tokens": list(context.query_tokens),
            "source": context.query_source,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "self_check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_tokens(values: Sequence[str]) -> list[int]:
    tokens: list[int] = []
    for value in values:
        for part in value.replace(",", " ").split():
            try:
                tokens.append(int(part))
            except ValueError as error:
                raise ValueError(f"不是合法的 token ID：{part!r}") from error
    return tokens


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--tokens", nargs="*", default=None, help="KuaiSAR 查询 token ID，逗号/空格分隔")
    parser.add_argument("--session-id", default=None, help="从 Search 1 样本加载查询会话，例如 11208")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--self-check", action="store_true", help="本地合成响应检查，不调用真实 LLM")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    output_root = (
        _resolve_path(args.output_root)
        if args.output_root is not None
        else _resolve_path(config["search_agent"]["output_root"])
    )
    if args.self_check:
        result = run_self_check(config, output_root=output_root)
        print(json.dumps({key: result[key] for key in ("status", "passed", "total")}, indent=2))
        for item in result["checks"]:
            print(f"[self-check] {item['id']}: {item['status']} — {item['detail']}")
        return 0 if result["passed"] else 1

    tokens = parse_tokens(args.tokens) if args.tokens is not None else None
    if args.tokens is not None and not tokens:
        print("[error] --tokens 不能为空；省略则使用 --session-id 或配置默认会话", file=sys.stderr)
        return 2
    command = " ".join(["python -m recsys.experiments.search_agent", *sys.argv[1:]]) if argv is None else "python -m recsys.experiments.search_agent"
    summary = run(
        config,
        tokens=tokens,
        session_id=args.session_id,
        output_root=output_root,
        command=command,
    )
    metrics = summary["metrics"]
    print(
        json.dumps(
            {
                "status": metrics["status"],
                "passed": metrics["passed"],
                "request_count": metrics["budget"]["request_count"],
                "search_calls": metrics["budget"]["search_calls"],
                "stop_code": metrics["summary"].get("stop_code"),
                "stop_reason": metrics["summary"].get("stop_reason"),
                "return_counts": metrics["summary"].get("return_counts"),
                "run_dir": metrics["run"]["output_dir"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if metrics["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
