"""Phase 11：LangGraph 工作流 + 真实 LLM 原生 tool_calls 验收。

正常入口（LangGraph 实际执行）：

    python -m recsys.experiments.llm_tool_call --experiment-dir experiments/2026-11-08-kuasar_small-langgraph-tool-call

失败路径回归（不联网、不访问缓存）：

    python -m recsys.experiments.llm_tool_call --regression-partial --experiment-dir ...

图节点：``load_baseline`` → ``request_tool_call`` → ``execute_tool`` → ``send_result`` → ``finish``；
任何失败分支都直接进入 ``finish``，不会落到成功分支。

状态语义（修复上一版误报）：

* ``verified``：合法工具调用 → 工具执行成功 → 同 ``tool_call_id`` 回传 → 收到非空、且不再要求调用工具的最终回复；
* ``partial``：工具已执行，但第二次请求超时/报错/回复为空/再次请求工具——保留工具结果与调用 ID，CLI 非零；
* ``blocked``：首次请求前被阻塞（缺配置/基准不符/无合法 tool_calls/非法配额）；
* ``failed``：工具执行本身失败。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, TypedDict
from urllib.parse import urlparse

from langgraph.graph import END, StateGraph

from recsys.experiments.fusion_quota import (
    ALLOWED_QUOTAS,
    DEFAULT_CACHE,
    QuotaError,
    evaluate_fusion_quota,
    load_context,
    tool_schema,
    validate_quota,
)

EXPERIMENT_DEFAULT = Path("experiments/2026-11-08-kuasar_small-langgraph-tool-call")
BASELINE_RUN = Path("experiments/2026-11-08-kuasar_small-fusion-quota-agent/runs/50_50.json")
TOOL_NAME = "evaluate_fusion_quota"
ALLOWED_AFTER_BASELINE = tuple(q for q in ALLOWED_QUOTAS if q != (50, 50))
BUDGET_SECONDS = 180.0
LLM_ENV = ("DSH_AGENT_LLM_BASE_URL", "DSH_AGENT_LLM_API_KEY", "DSH_AGENT_LLM_MODEL")
OUTPUT_ROOT = Path("artifacts/kuasar_small/langgraph_tool_call")


class VerificationError(RuntimeError):
    """验收条件不满足。"""


# --------------------------------------------------------------------------- 配置与 HTTP


def load_dotenv(path: Path | None = None) -> bool:
    """读取仓库根目录 ``.env``（只补未设置的环境变量，不打印值）。"""
    env_path = path or Path(__file__).resolve().parents[3] / ".env"
    if not env_path.exists():
        return False
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
    return True


def llm_config() -> dict[str, str] | None:
    values = {name: os.environ.get(name, "").strip() for name in LLM_ENV}
    if not all(values.values()):
        return None
    return {
        "base_url": values["DSH_AGENT_LLM_BASE_URL"],
        "api_key": values["DSH_AGENT_LLM_API_KEY"],
        "model": values["DSH_AGENT_LLM_MODEL"],
    }


def host_only(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def post_chat(
    config: dict[str, str],
    messages: list[dict],
    *,
    tools: list[dict] | None,
    timeout: float,
) -> dict:
    payload: dict[str, Any] = {"model": config["model"], "messages": messages, "temperature": 0.0}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        payload["parallel_tool_calls"] = False
    request = urllib.request.Request(
        f"{config['base_url'].rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config['api_key']}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=max(1.0, timeout)) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")[:800]
        raise RuntimeError(f"HTTP {error.code}: {body}") from error


# --------------------------------------------------------------------------- 校验


def extract_single_tool_call(response: dict, *, baseline_quota: tuple[int, int]) -> dict:
    """取出唯一合法的工具调用；普通文本 JSON 一律不接受。"""
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message", {})
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        raise VerificationError("响应里没有 tool_calls（普通文本/JSON 回复不计入验收）")
    if len(tool_calls) != 1:
        raise VerificationError(f"tool_calls 数量为 {len(tool_calls)}，要求恰好 1 条")
    call = tool_calls[0]
    function = call.get("function") or {}
    if (call.get("type") or "function") != "function":
        raise VerificationError(f"不支持的 tool call 类型：{call.get('type')}")
    if function.get("name") != TOOL_NAME:
        raise VerificationError(f"未知工具：{function.get('name')}")
    if not call.get("id"):
        raise VerificationError("tool_call 缺少 id")
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError as error:
        raise VerificationError(f"工具参数不是合法 JSON：{error}") from error
    for key in ("two_tower_quota", "mind_quota"):
        if key not in arguments or not isinstance(arguments[key], int):
            raise VerificationError(f"工具参数缺少整数 {key}")
    quota = validate_quota(arguments["two_tower_quota"], arguments["mind_quota"])
    if quota == baseline_quota:
        raise VerificationError("模型选择了基准配额；本轮要求对照组")
    if quota not in ALLOWED_AFTER_BASELINE:
        raise VerificationError(f"配额不在允许集合：{quota}")
    return {"id": call["id"], "name": TOOL_NAME, "quota": quota, "message": message}


def load_baseline() -> tuple[dict, dict]:
    """读取已落盘的 50/50 基准，并核对缓存指纹仍与当前一致。"""
    if not BASELINE_RUN.exists():
        raise VerificationError(f"缺少基准文件：{BASELINE_RUN}（先跑 2.23 的本地 Agent）")
    baseline = json.loads(BASELINE_RUN.read_text(encoding="utf-8"))
    context = load_context(DEFAULT_CACHE)
    current = context.fingerprints
    recorded = baseline.get("evidence", {}).get("fingerprints", {})
    keys = ("users", "items", "split_strategy", "split_sha256", "id_mapping_sha256", "routing_seed", "k_recall")
    mismatched = [key for key in keys if recorded.get(key) != current.get(key)]
    if mismatched:
        raise VerificationError(f"基准指纹与当前缓存不一致：{mismatched}")
    based_on = {
        "quota": baseline["quota"],
        "hits": baseline["hits"],
        "recall_at_100": baseline["recall_at_100"],
        "coverage": baseline["coverage"],
        "candidates_total": baseline["candidates_total"],
        "users": baseline["users"],
        "targets": baseline["targets"],
        "fingerprints_verified": True,
    }
    return based_on, current


def build_messages(baseline: dict) -> list[dict]:
    """只发送配置与聚合指标：无用户 ID、无候选明细、无标签、无密钥。"""
    return [
        {
            "role": "system",
            "content": (
                "你是推荐系统研究 Agent。你必须调用提供的工具来评测一组融合配额，"
                "不要用普通文本代替工具调用。本轮只允许发起 1 次工具调用，"
                "参数只能是一组配额，禁止并行多个 tool_calls。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": "每个用户最多保留 100 个候选时，双塔和 MIND 怎么分配配额，才能保留更多验证目标？",
                    "candidate_budget_per_user": 100,
                    "quota_order": "双塔 / MIND",
                    "allowed_quotas": [list(pair) for pair in ALLOWED_AFTER_BASELINE],
                    "baseline_quota": baseline["quota"],
                    "baseline_aggregate_metrics": {
                        "users": baseline["users"],
                        "validation_targets": baseline["targets"],
                        "hits": baseline["hits"],
                        "recall_at_100": baseline["recall_at_100"],
                        "coverage": baseline["coverage"],
                    },
                    "selection_rule": "命中目标数优先，Recall@100 次之；持平时保留基准",
                    "instruction": "从 allowed_quotas 中选一组，调用 evaluate_fusion_quota 工具。",
                },
                ensure_ascii=False,
            ),
        },
    ]


# --------------------------------------------------------------------------- LangGraph 状态


class AgentState(TypedDict, total=False):
    status: str  # running / verified / partial / blocked / failed
    stage: str
    nodes: list[str]
    events: list[dict]
    started: float
    deadline: float
    config: dict
    baseline: dict
    fingerprints: dict
    request_count: int
    tool_calls: int
    response_ids: dict
    tool_call_id: str
    quota: list[int]
    assistant_message: dict
    tool_result: dict
    final_reply: str
    error: str
    conclusion: dict


ChatFn = Callable[..., dict]
EvalFn = Callable[..., dict]
BaselineFn = Callable[[], tuple[dict, dict]]


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


# --------------------------------------------------------------------------- 节点


def node_load_baseline(state: AgentState, baseline_fn: BaselineFn) -> AgentState:
    state = _enter(state, "load_baseline")
    if state.get("baseline"):
        return _leave(state, "load_baseline")
    config = state.get("config")
    if config is None:
        state["status"] = "blocked"
        state["error"] = "缺少真实 LLM 配置"
        _event(state, {"event": "blocked", "reason": state["error"]})
        return _leave(state, "load_baseline")
    try:
        baseline, fingerprints = baseline_fn()
    except (VerificationError, QuotaError) as error:
        state["status"] = "blocked"
        state["error"] = str(error)
        _event(state, {"event": "blocked", "reason": str(error)})
        return _leave(state, "load_baseline")
    state["baseline"] = baseline
    state["fingerprints"] = fingerprints
    _event(state, {"event": "baseline_loaded", "baseline": baseline})
    return _leave(state, "load_baseline")


def node_request_tool_call(state: AgentState, chat_fn: ChatFn) -> AgentState:
    state = _enter(state, "request_tool_call")
    if _remaining(state) <= 0:
        state["status"] = "blocked"
        state["error"] = "预算耗尽，未发起第 1 次请求"
        _event(state, {"event": "blocked", "reason": state["error"]})
        return _leave(state, "request_tool_call")
    timeout = min(30.0, max(1.0, _remaining(state)))
    state["request_count"] = int(state.get("request_count", 0)) + 1  # 按实际发起的请求计数
    _event(state, {"event": "llm_request", "request": 1, "timeout_s": round(timeout, 1), "tools": True})
    try:
        response = chat_fn(state["config"], build_messages(state["baseline"]), tools=[tool_schema()], timeout=timeout)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, RuntimeError) as error:
        state["status"] = "blocked"
        state["error"] = f"第 1 次模型请求失败：{error}"
        _event(state, {"event": "blocked", "reason": state["error"]})
        return _leave(state, "request_tool_call")
    response_id = response.get("id")
    state.setdefault("response_ids", {})["tool_call"] = response_id
    raw_message = (response.get("choices") or [{}])[0].get("message", {})
    raw_calls = raw_message.get("tool_calls") or []
    _event(
        state,
        {
            "event": "llm_response",
            "request": 1,
            "response_id": response_id,
            "tool_calls": [
                {
                    "id": call.get("id"),
                    "type": call.get("type"),
                    "name": (call.get("function") or {}).get("name"),
                    "arguments": (call.get("function") or {}).get("arguments"),
                }
                for call in raw_calls
            ],
            "content_preview": (raw_message.get("content") or "")[:200],
        },
    )
    baseline_quota = (int(state["baseline"]["quota"]["two_tower"]), int(state["baseline"]["quota"]["mind"]))
    try:
        parsed = extract_single_tool_call(response, baseline_quota=baseline_quota)
    except (VerificationError, QuotaError) as error:
        state["status"] = "blocked"
        state["error"] = str(error)
        _event(state, {"event": "blocked", "reason": str(error), "response_id": response_id})
        return _leave(state, "request_tool_call")
    state["tool_call_id"] = parsed["id"]
    state["quota"] = list(parsed["quota"])
    state["assistant_message"] = {
        "role": "assistant",
        "content": raw_message.get("content") or "",
        "tool_calls": raw_calls,
    }
    _event(state, {"event": "tool_call_accepted", "tool_call_id": parsed["id"], "quota": list(parsed["quota"])})
    return _leave(state, "request_tool_call")


def node_execute_tool(state: AgentState, eval_fn: EvalFn) -> AgentState:
    state = _enter(state, "execute_tool")
    quota = tuple(int(x) for x in state["quota"])
    started = time.perf_counter()
    try:
        result = eval_fn(
            quota[0],
            quota[1],
            cache_dir=DEFAULT_CACHE,
            output_dir=OUTPUT_ROOT,
            tag=f"{quota[0]}_{quota[1]}",
        )
    except Exception as error:  # noqa: BLE001 - 工具失败进入失败分支
        state["status"] = "failed"
        state["error"] = f"工具执行失败：{error}"
        _event(state, {"event": "tool_failed", "tool_call_id": state.get("tool_call_id"), "error": str(error)})
        return _leave(state, "execute_tool")
    state["tool_calls"] = int(state.get("tool_calls", 0)) + 1
    state["tool_result"] = result
    _event(
        state,
        {
            "event": "tool_executed",
            "tool_call_id": state.get("tool_call_id"),
            "quota": result["quota"],
            "result": {
                key: result[key]
                for key in (
                    "users",
                    "targets",
                    "candidates_total",
                    "candidates_mean",
                    "fills_total",
                    "hits",
                    "recall_at_100",
                    "coverage",
                    "elapsed_s",
                )
            },
            "tool_elapsed_s": round(time.perf_counter() - started, 2),
            "evidence": result["evidence"]["fusion"],
        },
    )
    return _leave(state, "execute_tool")


def node_send_result(state: AgentState, chat_fn: ChatFn) -> AgentState:
    state = _enter(state, "send_result")
    if _remaining(state) <= 0:
        state["status"] = "partial"
        state["error"] = "预算耗尽，未发起第 2 次请求"
        _event(state, {"event": "partial", "reason": state["error"]})
        return _leave(state, "send_result")

    result = state["tool_result"]
    followup = build_messages(state["baseline"]) + [
        state["assistant_message"],
        {
            "role": "tool",
            "tool_call_id": state["tool_call_id"],
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
                "这是真实工具结果。请对比基准与该配额，用一句话给出结论和停止理由；"
                "不要再调用工具，也不要输出 JSON 代码块。"
            ),
        },
    ]
    timeout = min(30.0, max(1.0, _remaining(state)))
    state["request_count"] = int(state.get("request_count", 0)) + 1  # 发出即计数，失败也算
    _event(
        state,
        {
            "event": "tool_result_sent",
            "tool_call_id": state["tool_call_id"],
            "request": 2,
            "timeout_s": round(timeout, 1),
        },
    )
    try:
        response = chat_fn(state["config"], followup, tools=None, timeout=timeout)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, RuntimeError) as error:
        state["status"] = "partial"
        state["error"] = f"第 2 次模型请求失败：{error}"
        _event(state, {"event": "partial", "reason": state["error"]})
        return _leave(state, "send_result")

    response_id = response.get("id")
    state.setdefault("response_ids", {})["final"] = response_id
    message = (response.get("choices") or [{}])[0].get("message", {}) or {}
    content = (message.get("content") or "").strip()
    _event(state, {"event": "llm_final", "response_id": response_id, "content": content})
    if message.get("tool_calls"):
        state["status"] = "partial"
        state["error"] = "第 2 次响应再次请求调用工具，未得到最终结论"
        _event(state, {"event": "partial", "reason": state["error"]})
        return _leave(state, "send_result")
    if not content:
        state["status"] = "partial"
        state["error"] = "第 2 次响应为空，未得到有效最终回复"
        _event(state, {"event": "partial", "reason": state["error"]})
        return _leave(state, "send_result")
    state["final_reply"] = content
    return _leave(state, "send_result")


def node_finish(state: AgentState) -> AgentState:
    state = _enter(state, "finish")
    baseline = state.get("baseline") or {}
    result = state.get("tool_result")
    if result and baseline:
        better = result["hits"] > baseline["hits"] or (
            result["hits"] == baseline["hits"] and result["recall_at_100"] > baseline["recall_at_100"]
        )
        state["conclusion"] = {
            "baseline_quota": baseline["quota"],
            "chosen_quota": result["quota"],
            "baseline_hits": baseline["hits"],
            "chosen_hits": result["hits"],
            "delta_hits": result["hits"] - baseline["hits"],
            "delta_recall_at_100": round(result["recall_at_100"] - baseline["recall_at_100"], 6),
            "winner": "chosen" if better else "baseline",
            "final_model_reply": state.get("final_reply", ""),
        }
    if state.get("status", "running") == "running":
        if state.get("final_reply") and state.get("tool_calls", 0) == 1:
            state["status"] = "verified"
        else:
            state["status"] = "partial"
            state.setdefault("error", "闭环未完成：缺少最终回复或工具执行")
    _event(state, {"event": "finish", "status": state["status"], "error": state.get("error")})
    return _leave(state, "finish")


# --------------------------------------------------------------------------- 图


def _route(next_node: str):
    def route(state: AgentState) -> str:
        if state.get("status") in {"blocked", "partial", "failed"} or state.get("error"):
            return "finish"
        return next_node

    return route


def build_graph(
    *,
    chat_fn: ChatFn = post_chat,
    eval_fn: EvalFn = evaluate_fusion_quota,
    baseline_fn: BaselineFn = load_baseline,
):
    graph = StateGraph(AgentState)
    graph.add_node("load_baseline", lambda state: node_load_baseline(state, baseline_fn))
    graph.add_node("request_tool_call", lambda state: node_request_tool_call(state, chat_fn))
    graph.add_node("execute_tool", lambda state: node_execute_tool(state, eval_fn))
    graph.add_node("send_result", lambda state: node_send_result(state, chat_fn))
    graph.add_node("finish", node_finish)
    graph.set_entry_point("load_baseline")
    graph.add_conditional_edges("load_baseline", _route("request_tool_call"))
    graph.add_conditional_edges("request_tool_call", _route("execute_tool"))
    graph.add_conditional_edges("execute_tool", _route("send_result"))
    graph.add_conditional_edges("send_result", _route("finish"))
    graph.add_edge("finish", END)
    return graph.compile()


# --------------------------------------------------------------------------- 写盘与入口


def _finalize(state: AgentState, experiment_dir: Path) -> dict:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    events_path = experiment_dir / "events.jsonl"
    events = state.get("events", [])
    events_path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + ("\n" if events else ""),
        encoding="utf-8",
    )
    summary = {
        "status": state.get("status", "blocked"),
        "stage": state.get("stage"),
        "graph_nodes": state.get("nodes", []),
        "request_count": int(state.get("request_count", 0)),
        "tool_calls": int(state.get("tool_calls", 0)),
        "response_ids": state.get("response_ids", {}),
        "tool_call_id": state.get("tool_call_id"),
        "baseline": state.get("baseline"),
        "chosen": state.get("tool_result"),
        "conclusion": state.get("conclusion"),
        "error": state.get("error"),
        "budget": {
            "elapsed_s": round(time.perf_counter() - float(state.get("started", time.perf_counter())), 2),
            "limit_seconds": BUDGET_SECONDS,
        },
    }
    (experiment_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def run(experiment_dir: Path, *, cache_dir: Path = DEFAULT_CACHE) -> dict:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    events_path = experiment_dir / "events.jsonl"
    events_path.write_text("", encoding="utf-8")
    started = time.perf_counter()
    load_dotenv()
    config = llm_config()

    if config is None:
        missing = [name for name in LLM_ENV if not os.environ.get(name)]
        state: AgentState = {
            "status": "blocked",
            "stage": "init",
            "nodes": [],
            "events": [{"event": "blocked", "reason": f"缺少真实 LLM 配置：{', '.join(missing)}"}],
            "started": started,
            "deadline": started + BUDGET_SECONDS,
            "request_count": 0,
            "tool_calls": 0,
            "response_ids": {},
            "error": f"缺少真实 LLM 配置：{', '.join(missing)}",
        }
        return _finalize(state, experiment_dir)

    state = {
        "status": "running",
        "stage": "init",
        "nodes": [],
        "events": [{"event": "llm_config", "model": config["model"], "endpoint_host": host_only(config["base_url"])}],
        "started": started,
        "deadline": started + BUDGET_SECONDS,
        "config": config,
        "request_count": 0,
        "tool_calls": 0,
        "response_ids": {},
    }
    graph = build_graph()
    final_state = graph.invoke(state)
    summary = _finalize(final_state, experiment_dir)
    (experiment_dir / "baseline.json").write_text(
        json.dumps(summary.get("baseline") or {}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def run_regression(experiment_dir: Path) -> dict:
    """失败路径回归：首次调用与工具执行成功，第二次请求超时；不联网、不访问缓存。"""
    experiment_dir.mkdir(parents=True, exist_ok=True)
    baseline = {
        "quota": {"two_tower": 50, "mind": 50},
        "hits": 4,
        "recall_at_100": 0.011719,
        "coverage": 0.026843,
        "candidates_total": 12800,
        "users": 128,
        "targets": 362,
        "fingerprints_verified": True,
    }
    calls = {"count": 0}

    def fake_chat(config, messages, *, tools, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "id": "resp-fake-1",
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-fake-1",
                                    "type": "function",
                                    "function": {
                                        "name": TOOL_NAME,
                                        "arguments": json.dumps({"two_tower_quota": 70, "mind_quota": 30}),
                                    },
                                }
                            ],
                        }
                    }
                ],
            }
        raise TimeoutError("simulated timeout on second request")

    def fake_eval(two_tower_quota, mind_quota, *, cache_dir, output_dir, tag):
        return {
            "tool": "evaluate_fusion_quota",
            "quota": {"two_tower": two_tower_quota, "mind": mind_quota},
            "users": 128,
            "targets": 362,
            "candidates_total": 12800,
            "candidates_mean": 100.0,
            "two_tower_used_total": two_tower_quota * 128,
            "mind_used_total": mind_quota * 128,
            "fills_total": 0,
            "hits": 7,
            "recall_at_100": 0.02,
            "coverage": 0.027,
            "elapsed_s": 0.0,
            "diagnostic_union_hits_at_100": 9,
            "evidence": {"fusion": str(experiment_dir / "fake_candidates.json")},
        }

    started = time.perf_counter()
    state = {
        "status": "running",
        "stage": "init",
        "nodes": [],
        "events": [{"event": "regression", "mode": "second_request_timeout"}],
        "started": started,
        "deadline": started + BUDGET_SECONDS,
        "config": {"model": "fake-model", "base_url": "http://127.0.0.1", "api_key": "fake"},
        "baseline": baseline,
        "request_count": 0,
        "tool_calls": 0,
        "response_ids": {},
    }
    graph = build_graph(chat_fn=fake_chat, eval_fn=fake_eval, baseline_fn=lambda: (baseline, {}))
    final_state = graph.invoke(state)
    summary = _finalize(final_state, experiment_dir)

    simulated_cli_exit = 0 if summary["status"] == "verified" else 2
    checks = {
        "status_is_partial": summary["status"] == "partial",
        "cli_nonzero": simulated_cli_exit != 0,
        "request_count_2": summary["request_count"] == 2,
        "tool_executed_once": summary["tool_calls"] == 1,
        "tool_result_kept": bool(summary.get("chosen")),
        "not_verified": summary["status"] != "verified",
        "error_recorded": bool(summary.get("error")),
    }
    regression = {
        "mode": "second_request_timeout",
        "status": summary["status"],
        "simulated_cli_exit": simulated_cli_exit,
        "request_count": summary["request_count"],
        "tool_calls": summary["tool_calls"],
        "error": summary.get("error"),
        "checks": checks,
        "passed": all(checks.values()),
        "events": summary.get("graph_nodes"),
        "metrics_file": str(experiment_dir / "metrics.json"),
    }
    (experiment_dir / "regression_partial.json").write_text(
        json.dumps(regression, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return regression


def run_self_check() -> dict:
    """合成响应验证解析守卫（不联网、不访问缓存）。"""
    checks: list[str] = []
    valid = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": TOOL_NAME,
                                "arguments": json.dumps({"two_tower_quota": 30, "mind_quota": 70}),
                            },
                        }
                    ]
                }
            }
        ]
    }
    assert extract_single_tool_call(valid, baseline_quota=(50, 50))["quota"] == (30, 70)
    checks.append("接受合法的单条 tool_call")

    for bad in (
        {"choices": [{"message": {"content": '{"two_tower_quota": 30, "mind_quota": 70}'}}]},
        {"choices": [{"message": {"tool_calls": [valid["choices"][0]["message"]["tool_calls"][0]] * 2}}]},
    ):
        try:
            extract_single_tool_call(bad, baseline_quota=(50, 50))
        except VerificationError:
            checks.append("拒绝无 tool_calls / 多条 tool_calls")
        else:
            raise AssertionError("非法响应未被拒绝")

    unknown = json.loads(json.dumps(valid))
    unknown["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "train_model"
    try:
        extract_single_tool_call(unknown, baseline_quota=(50, 50))
    except VerificationError:
        checks.append("拒绝未知工具")
    else:
        raise AssertionError("未知工具未被拒绝")

    illegal = json.loads(json.dumps(valid))
    illegal["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"two_tower_quota": 60, "mind_quota": 40}
    )
    try:
        extract_single_tool_call(illegal, baseline_quota=(50, 50))
    except QuotaError:
        checks.append("拒绝非法配额 60/40")
    else:
        raise AssertionError("非法配额未被拒绝")

    baseline_choice = json.loads(json.dumps(valid))
    baseline_choice["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"two_tower_quota": 50, "mind_quota": 50}
    )
    try:
        extract_single_tool_call(baseline_choice, baseline_quota=(50, 50))
    except VerificationError:
        checks.append("拒绝重复基准")
    else:
        raise AssertionError("重复基准未被拒绝")

    return {"passed": True, "checks": checks}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT_DEFAULT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--self-check", action="store_true", help="合成响应验证解析守卫（不联网）")
    parser.add_argument("--regression-partial", action="store_true", help="第二次请求失败的本地回归（不联网、不读缓存）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_check:
        print(json.dumps(run_self_check(), indent=2, ensure_ascii=False))
        return 0
    if args.regression_partial:
        result = run_regression(args.experiment_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["passed"] else 1
    try:
        summary = run(args.experiment_dir, cache_dir=args.cache_dir)
    except VerificationError as error:
        print(f"[未验收] {error}", file=sys.stderr)
        return 2
    if summary["status"] != "verified":
        print(f"[{summary['status']}] {summary.get('error') or '闭环未完成'}", file=sys.stderr)
        return 2
    print(json.dumps(summary["conclusion"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
