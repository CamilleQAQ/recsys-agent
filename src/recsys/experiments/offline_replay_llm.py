"""Phase 13 Real LLM Offline Replay：真实 LLM 三方受控回放。

复用：
* ``offline_replay.py`` 的来源校验、``ReplaySession`` 预算/配额/结果查询；
* ``llm_tool_call.py`` 的 OpenAI 兼容 HTTP 调用、LangGraph 拓扑与原生 tool_calls 校验方式。

只读已有 5 组汇总 JSON；真实 LLM 工作流只跑一次，最多 3 次模型请求。
没有训练、重新召回或真实评测。

    $env:PYTHONPATH = "src"
    python -m recsys.experiments.offline_replay_llm --self-check
    python -m recsys.experiments.offline_replay_llm
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypedDict

from langgraph.graph import END, StateGraph

from recsys.experiments import offline_replay as replay
from recsys.experiments.llm_tool_call import (
    LLM_ENV,
    VerificationError,
    host_only,
    llm_config,
    load_dotenv,
    post_chat,
)

PROJECT_ROOT = replay.PROJECT_ROOT
DEFAULT_CONFIG_PATH = replay.DEFAULT_CONFIG_PATH
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "2026-11-09-kuasar_small-offline-replay-llm"
PHASE = "Phase 13 Real LLM Offline Replay"
SCHEMA_VERSION = 1
TOOL_NAME = "get_fusion_result"
MAX_MODEL_REQUESTS = 3
MAX_REVEALS = 2


class ReplayPreflightError(RuntimeError):
    """来源、冻结基线或本地断言不满足，禁止启动真实工作流。"""


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _safe_text(value: Any, limit: int = 600) -> str:
    text = str(value or "")
    return text[:limit] + ("…" if len(text) > limit else "")


def _safe_error(error: Any, config: Mapping[str, Any] | None = None) -> str:
    text = f"{type(error).__name__}: {error}" if not isinstance(error, str) else error
    key = ""
    if config:
        key = str(config.get("api_key") or "")
    if key:
        text = text.replace(key, "***")
    return _safe_text(text, 600)


def _read_json(path_text: str) -> tuple[Any, str, Path]:
    path = (PROJECT_ROOT / path_text).resolve()
    if PROJECT_ROOT not in path.parents:
        raise ReplayPreflightError(f"路径越出项目根目录：{path_text}")
    if not path.is_file():
        raise ReplayPreflightError(f"文件不存在：{path_text}")
    document, sha = replay.read_json_with_hash(path)
    return document, sha, path


def _public_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    fields = list(replay.PUBLIC_METRIC_FIELDS)
    for field in fields:
        if field not in left or field not in right:
            return False
        if isinstance(left[field], float) or isinstance(right[field], float):
            if abs(float(left[field]) - float(right[field])) > 1e-9:
                return False
        elif left[field] != right[field]:
            return False
    return True


def _baseline_public(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return replay.public_metrics(records["50/50"])


def _quota_reveal_index(trace: Mapping[str, Any], quota: str) -> int:
    if quota == "50/50":
        return 0
    for step in trace.get("steps", []):
        returned = step.get("tool_return") or {}
        if returned.get("status") == "ok" and returned.get("result", {}).get("quota") == quota:
            return int(step["step"])
    return 0


# ---------------------------------------------------------------------------
# 来源与冻结基线
# ---------------------------------------------------------------------------


def _validate_trace_against_records(
    trace: Mapping[str, Any],
    *,
    side: str,
    baseline: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    candidates: Sequence[str],
    decimals: int,
) -> None:
    if trace.get("status") != "success":
        raise ReplayPreflightError(f"{side} 基线轨迹状态不是 success：{trace.get('status')}")
    if trace.get("budget_initial") != 2:
        raise ReplayPreflightError(f"{side} 基线预算不是 2")
    if list(trace.get("candidates") or []) != list(candidates):
        raise ReplayPreflightError(f"{side} 基线候选配额与当前配置不一致")
    if not _public_equal(trace.get("starting_history") or {}, baseline):
        raise ReplayPreflightError(f"{side} 基线起始历史与当前 50/50 指标不一致")
    revealed: list[dict[str, Any]] = []
    for index, step in enumerate(trace.get("steps", []), start=1):
        visible = step.get("visible_before") or {}
        if not _public_equal(visible.get("starting_history") or {}, baseline):
            raise ReplayPreflightError(f"{side} 第 {index} 步可见起始历史不一致")
        if list(visible.get("candidates") or []) != list(candidates):
            raise ReplayPreflightError(f"{side} 第 {index} 步可见候选配额不一致")
        prior = visible.get("revealed") or []
        if len(prior) != len(revealed):
            raise ReplayPreflightError(f"{side} 第 {index} 步已揭晓数量不一致")
        for old, expected in zip(prior, revealed):
            if not _public_equal(old, expected):
                raise ReplayPreflightError(f"{side} 第 {index} 步已揭晓结果被改动")
        decision = step.get("decision") or {}
        if decision.get("action") != "choose":
            raise ReplayPreflightError(f"{side} 第 {index} 步不是选择动作")
        quota = decision.get("quota")
        if quota not in records:
            raise ReplayPreflightError(f"{side} 第 {index} 步配额未知：{quota}")
        returned = step.get("tool_return") or {}
        if returned.get("status") != "ok":
            raise ReplayPreflightError(f"{side} 第 {index} 步工具返回不是 ok")
        expected_result = replay.public_metrics(records[quota])
        if not _public_equal(returned.get("result") or {}, expected_result):
            raise ReplayPreflightError(f"{side} 第 {index} 步工具返回与当前来源不一致")
        tool_call = step.get("tool_call") or {}
        if (tool_call.get("name") != TOOL_NAME) or (
            (tool_call.get("arguments") or {}) != {"two_tower": records[quota]["two_tower"], "mind": records[quota]["mind"]}
        ):
            raise ReplayPreflightError(f"{side} 第 {index} 步工具调用与配额不一致")
        revealed.append(expected_result)
        expected_remaining = 2 - index
        if step.get("remaining_budget_after") != expected_remaining:
            raise ReplayPreflightError(f"{side} 第 {index} 步剩余预算不一致")
        expected_best = replay.select_best([baseline, *revealed], decimals)
        if not _public_equal(step.get("best_after") or {}, replay.public_metrics(expected_best)):
            raise ReplayPreflightError(f"{side} 第 {index} 步最佳结果与当前规则不一致")
    stop = trace.get("stop") or {}
    summary = trace.get("summary") or {}
    expected_best = replay.select_best([baseline, *revealed], decimals)
    if not _public_equal(stop.get("best") or {}, replay.public_metrics(expected_best)):
        raise ReplayPreflightError(f"{side} 停止时最佳结果与当前规则不一致")
    if not _public_equal(summary.get("best") or {}, replay.public_metrics(expected_best)):
        raise ReplayPreflightError(f"{side} 汇总最佳结果与当前规则不一致")
    if int(summary.get("reveals", -1)) != len(revealed):
        raise ReplayPreflightError(f"{side} 汇总揭晓次数与轨迹不一致")


def load_frozen_baselines(
    config: Mapping[str, Any], sources: Mapping[str, Any]
) -> dict[str, Any]:
    frozen = config.get("frozen_baselines") or {}
    required = ("previous_run", "local_agent_trace", "fixed_trace", "metrics", "sources")
    missing = [key for key in required if not frozen.get(key)]
    if missing:
        raise ReplayPreflightError(f"冻结基线配置缺少：{missing}")

    local_trace, local_sha, local_path = _read_json(frozen["local_agent_trace"])
    fixed_trace, fixed_sha, fixed_path = _read_json(frozen["fixed_trace"])
    previous_metrics, metrics_sha, metrics_path = _read_json(frozen["metrics"])
    previous_sources, sources_sha, sources_path = _read_json(frozen["sources"])

    current_manifest = sources["manifest"]
    previous_cache = (previous_sources.get("recall_cache") or {}).get("version")
    current_cache = (current_manifest.get("recall_cache") or {}).get("version")
    if not previous_cache or previous_cache != current_cache:
        raise ReplayPreflightError(
            f"冻结基线的召回缓存身份与当前来源不一致：{previous_cache} ≠ {current_cache}"
        )
    previous_files = {
        item["path"]: item["sha256"] for item in previous_sources.get("source_files_read", [])
    }
    current_files = {
        path: item["sha256"] for path, item in sources["files_read"].items()
    }
    if previous_files != current_files:
        raise ReplayPreflightError("冻结基线记录的来源文件哈希与当前来源不一致")

    previous_rules = previous_metrics.get("rules") or {}
    checks = {
        "baseline_quota": previous_rules.get("baseline_quota") == config["replay"]["baseline_quota"],
        "allowed_quotas": list(previous_rules.get("allowed_quotas") or [])
        == list(config["replay"]["allowed_quotas"]),
        "budget": previous_rules.get("budget_reveals_per_side")
        == int(config["replay"]["budget_reveals_per_side"]),
        "best_rule": previous_rules.get("best_rule") == config["best_rule"],
        "fixed_order": list(previous_rules.get("fixed_order") or [])
        == list(config["fixed_policy"]["order"]),
        "agent_policy": previous_rules.get("agent_policy") == config["agent_policy"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ReplayPreflightError(f"冻结基线的规则与当前配置不一致：{failed}")

    baseline = _baseline_public(sources["records"])
    candidates = list(config["replay"]["allowed_quotas"])
    decimals = int(config["best_rule"]["recall_decimals"])
    _validate_trace_against_records(
        local_trace,
        side="local_agent",
        baseline=baseline,
        records=sources["records"],
        candidates=candidates,
        decimals=decimals,
    )
    _validate_trace_against_records(
        fixed_trace,
        side="fixed",
        baseline=baseline,
        records=sources["records"],
        candidates=candidates,
        decimals=decimals,
    )

    for side, trace in (("agent", local_trace), ("fixed", fixed_trace)):
        previous_summary = previous_metrics.get(side) or {}
        current_summary = trace.get("summary") or {}
        for field in ("best_quota", "hits", "recall_at_100", "coverage", "reveals", "invalid_requests"):
            if previous_summary.get(field) != current_summary.get(field):
                raise ReplayPreflightError(f"冻结基线 {side}.{field} 与冻结 metrics 不一致")

    return {
        "previous_run": frozen["previous_run"],
        "baseline": baseline,
        "candidates": candidates,
        "decimals": decimals,
        "files": {
            "local_agent_trace": {
                "path": frozen["local_agent_trace"],
                "sha256": local_sha,
                "bytes": local_path.stat().st_size,
            },
            "fixed_trace": {
                "path": frozen["fixed_trace"],
                "sha256": fixed_sha,
                "bytes": fixed_path.stat().st_size,
            },
            "metrics": {
                "path": frozen["metrics"],
                "sha256": metrics_sha,
                "bytes": metrics_path.stat().st_size,
            },
            "sources": {
                "path": frozen["sources"],
                "sha256": sources_sha,
                "bytes": sources_path.stat().st_size,
            },
        },
        "local_agent": {
            "status": local_trace.get("status"),
            "policy": local_trace.get("policy"),
            "summary": copy.deepcopy(local_trace.get("summary")),
            "first_best_reveal": _quota_reveal_index(
                local_trace, (local_trace.get("summary") or {}).get("best_quota", "50/50")
            ),
        },
        "fixed": {
            "status": fixed_trace.get("status"),
            "policy": fixed_trace.get("policy"),
            "summary": copy.deepcopy(fixed_trace.get("summary")),
            "first_best_reveal": _quota_reveal_index(
                fixed_trace, (fixed_trace.get("summary") or {}).get("best_quota", "50/50")
            ),
        },
        "rule_checks": checks,
        "previous_source_files": previous_files,
    }


# ---------------------------------------------------------------------------
# 消息、工具 schema 与解析
# ---------------------------------------------------------------------------


TOOL_FUNCTION = {
    "type": "function",
    "function": replay.FusionResultLookup.schema,
}


def _messages(visible: Mapping[str, Any], phase: str) -> list[dict[str, Any]]:
    if phase == "first":
        system = (
            "你是推荐系统离线回放 Agent。你只能通过 get_fusion_result 工具查询已有汇总指标，"
            "不能训练、召回或真实评测。每次最多发起一个 tool_call；禁止并行 tool_calls。"
            "参数名必须是 two_tower 与 mind，取值为整数且两路之和为 100。"
        )
        instruction = (
            "从 allowed_quotas 中选择一组此前未揭晓的配额，调用 get_fusion_result。"
            "只允许 1 个 tool_call，不要用普通文本代替工具调用。"
        )
    elif phase == "feedback":
        system = (
            "你是推荐系统离线回放 Agent。你只能通过 get_fusion_result 工具查询已有汇总指标。"
            "你可以再查一组此前未揭晓的配额，也可以主动停止并说明原因。"
            "每次最多发起一个 tool_call；禁止并行 tool_calls。"
        )
        instruction = (
            "如果继续，选择另一组未揭晓配额并只调用 1 次 get_fusion_result；"
            "如果停止，请直接给出简短停止理由，不要调用工具，也不要声称查询了未返回的配额。"
        )
    elif phase == "final":
        system = (
            "你是推荐系统离线回放 Agent。预算已经用完，不能再查询任何配额。"
            "只输出一句最终结论或停止理由，不要调用工具。"
        )
        instruction = "请只给出最终结论/停止理由，不要调用工具，不要输出 JSON 代码块。"
    else:
        raise ValueError(f"未知消息阶段：{phase}")

    payload = {
        "task": "在固定 128 用户 / 362 目标的离线回放中，选择双塔 / MIND 融合配额。",
        "quota_order": "双塔 / MIND",
        "candidate_budget_per_user": 100,
        "allowed_quotas": list(visible["candidates"]),
        "revealed_results": visible["revealed"],
        "remaining_budget": visible["remaining_budget"],
        "best_rule": "先比 Recall@100；并列选更接近 50/50；再并列按双塔配额升序；Coverage 仅作描述",
        "instruction": instruction,
    }
    if "starting_histories" in visible:
        payload["starting_histories"] = visible["starting_histories"]
    else:
        payload["starting_history"] = visible["starting_history"]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _message_content(response: Mapping[str, Any]) -> str:
    try:
        message = (response.get("choices") or [{}])[0].get("message") or {}
    except (AttributeError, IndexError, TypeError):
        return ""
    return str(message.get("content") or "")


def _message_tool_calls(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        message = (response.get("choices") or [{}])[0].get("message") or {}
    except (AttributeError, IndexError, TypeError):
        return []
    calls = message.get("tool_calls") or []
    return calls if isinstance(calls, list) else []


def _finish_reason(response: Mapping[str, Any]) -> str:
    try:
        choice = (response.get("choices") or [{}])[0]
    except (AttributeError, IndexError, TypeError):
        return ""
    return str(choice.get("finish_reason") or "")


def _reply_status(response: Mapping[str, Any]) -> tuple[str, str, str]:
    """返回 (状态, 正文, finish_reason)。

    状态：``ok`` / ``truncated`` / ``non_stop`` / ``empty`` / ``tool_calls``。
    只有 ``ok`` 允许作为成功的停止说明或最终回复。
    """

    if _message_tool_calls(response):
        return "tool_calls", "", _finish_reason(response)
    finish_reason = _finish_reason(response)
    content = _message_content(response).strip()
    if finish_reason != "stop":
        return ("truncated" if finish_reason == "length" else "non_stop"), content, finish_reason
    if not content:
        return "empty", "", finish_reason
    return "ok", content, finish_reason


def _response_summary(response: Mapping[str, Any] | None) -> dict[str, Any]:
    if response is None:
        return {"status": "missing"}
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
    try:
        choice = (response.get("choices") or [{}])[0]
    except (AttributeError, IndexError, TypeError):
        choice = {}
    return {
        "id": response.get("id"),
        "finish_reason": choice.get("finish_reason"),
        "content_chars": len(_message_content(response)),
        "tool_calls": calls,
    }


def parse_single_fusion_call(response: Mapping[str, Any]) -> dict[str, Any]:
    """严格接受恰好 1 条 get_fusion_result(two_tower, mind) 原生工具调用。"""

    calls = _message_tool_calls(response)
    if not calls:
        raise VerificationError("响应里没有 tool_calls（普通文本不算成功查询）")
    if len(calls) != 1:
        raise VerificationError(f"tool_calls 数量为 {len(calls)}，要求恰好 1 条")
    call = calls[0]
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
    if not isinstance(arguments, dict) or set(arguments) != {"two_tower", "mind"}:
        raise VerificationError("工具参数必须且只能包含 two_tower、mind")
    for key in ("two_tower", "mind"):
        if isinstance(arguments[key], bool) or not isinstance(arguments[key], int):
            raise VerificationError(f"工具参数 {key} 必须是整数")
    two_tower, mind = int(arguments["two_tower"]), int(arguments["mind"])
    if two_tower + mind != 100:
        raise VerificationError(f"配额两路之和必须为 100：{two_tower}/{mind}")
    quota = f"{two_tower}/{mind}"
    return {
        "id": str(call["id"]),
        "name": TOOL_NAME,
        "two_tower": two_tower,
        "mind": mind,
        "quota": quota,
        "arguments": {"two_tower": two_tower, "mind": mind},
    }


# ---------------------------------------------------------------------------
# LangGraph 状态
# ---------------------------------------------------------------------------


class LLMState(TypedDict, total=False):
    status: str  # running / verified / incomplete / partial / blocked
    stage: str
    nodes: list[str]
    events: list[dict]
    started: float
    deadline: float
    llm_config: dict
    model: str
    endpoint_host: str
    baseline: dict
    starting_histories: list[dict]
    candidates: list[str]
    scenario: str
    max_model_requests: int
    session: Any
    request_count: int
    request_counter: list[int]
    global_request_counter: list[int]
    max_total_requests: int
    tool_calls: int
    response_ids: dict[str, str]
    trace_steps: list[dict]
    dialogue: list[dict]
    route_after_feedback: str
    current_call_id: str
    current_quota: str
    current_arguments: dict
    first_result: dict
    second_result: dict
    final_reply: str
    stop_code: str
    stop_reason: str
    error: str
    summary: dict
    conclusion: dict
    chat_fn: Any


ChatFn = Callable[..., dict]


def _event(state: LLMState, payload: dict) -> None:
    state.setdefault("events", []).append(payload)


def _enter(state: LLMState, node: str) -> LLMState:
    state = dict(state)  # type: ignore[assignment]
    state.setdefault("nodes", []).append(node)
    state["stage"] = node
    _event(state, {"event": "node_start", "node": node})
    return state  # type: ignore[return-value]


def _leave(state: LLMState, node: str) -> LLMState:
    _event(state, {"event": "node_end", "node": node, "status": state.get("status")})
    return state


def _append_step(state: LLMState, step: dict) -> None:
    steps = copy.deepcopy(state.get("trace_steps", []))
    steps.append(step)
    state["trace_steps"] = steps


def _update_last_step(state: LLMState, **updates: Any) -> None:
    steps = copy.deepcopy(state.get("trace_steps", []))
    if not steps:
        return
    steps[-1].update(updates)
    state["trace_steps"] = steps


def _shutdown(
    state: LLMState,
    *,
    status: str,
    code: str,
    reason: str,
    error: str | None = None,
    error_code: str | None = None,
    decision_action: str = "error",
    stop_extra: Mapping[str, Any] | None = None,
) -> LLMState:
    state["status"] = status
    state["stop_code"] = code
    state["stop_reason"] = reason
    state["error"] = error or reason
    steps = state.get("trace_steps", [])
    existing_decision = steps[-1].get("decision") if steps else None
    stop_payload: dict[str, Any] = {
        "code": code,
        "reason": reason,
        "status": status,
        "error_code": error_code,
    }
    if stop_extra:
        stop_payload.update(stop_extra)
    updates: dict[str, Any] = {"stop": stop_payload}
    if not existing_decision:
        updates["decision"] = {
            "action": decision_action,
            "quota": state.get("current_quota"),
            "reason": reason,
        }
    _update_last_step(state, **updates)
    _event(
        state,
        {
            "event": "stop",
            "status": status,
            "code": code,
            "reason": reason,
            "error_code": error_code,
            **(dict(stop_extra) if stop_extra else {}),
        },
    )
    return state


def _assistant_message(response: Mapping[str, Any]) -> dict[str, Any]:
    try:
        message = (response.get("choices") or [{}])[0].get("message") or {}
    except (AttributeError, IndexError, TypeError):
        message = {}
    return {
        "role": "assistant",
        "content": str(message.get("content") or ""),
        "tool_calls": copy.deepcopy(message.get("tool_calls") or []),
    }


def _tool_message(call_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(replay.public_metrics(result), ensure_ascii=False),
    }


def _followup_user_message(visible: Mapping[str, Any], phase: str) -> dict[str, Any]:
    if phase == "feedback":
        instruction = (
            "你可以再查一组此前未揭晓的配额（只允许 1 个 tool_call，参数 two_tower/mind），"
            "也可以直接停止并说明原因。不要声称查询了未返回的配额。"
        )
    elif phase == "final":
        instruction = "预算已经用完，不能再查询。请只给出最终结论或停止理由，不要调用工具。"
    else:
        raise ValueError(phase)
    payload = {
        "allowed_quotas": list(visible["candidates"]),
        "revealed_results": visible["revealed"],
        "remaining_budget": visible["remaining_budget"],
        "instruction": instruction,
    }
    if "starting_histories" in visible:
        payload["starting_histories"] = visible["starting_histories"]
    else:
        payload["starting_history"] = visible["starting_history"]
    return {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}


def _call_model(
    state: LLMState,
    *,
    phase: str,
    tools: list[dict] | None,
    visible: Mapping[str, Any],
    messages: list[dict[str, Any]],
) -> tuple[LLMState, dict | None]:
    if state.get("status") != "running":
        return state, None
    max_requests = int(state.get("max_model_requests", MAX_MODEL_REQUESTS))
    if int(state.get("request_count", 0)) >= max_requests:
        reason = f"模型请求次数已达到上限 {max_requests}，拒绝继续请求"
        state["status"] = "partial" if (state["session"].revealed) else "incomplete"
        state["stop_code"] = "request_budget_exhausted"
        state["stop_reason"] = reason
        state["error"] = reason
        _event(state, {"event": "rejected", "code": "request_budget_exhausted", "reason": reason})
        return state, None
    remaining = float(state["deadline"]) - time.perf_counter()
    if remaining <= 0:
        reason = "总时间预算耗尽，拒绝发起模型请求"
        state["status"] = "partial" if (state["session"].revealed) else "incomplete"
        state["stop_code"] = "timeout"
        state["stop_reason"] = reason
        state["error"] = reason
        _event(state, {"event": "rejected", "code": "timeout", "reason": reason})
        return state, None
    global_counter = state.get("global_request_counter")
    max_total = state.get("max_total_requests")
    if (
        global_counter is not None
        and max_total is not None
        and int(global_counter[0]) >= int(max_total)
    ):
        reason = f"全轮模型请求次数已达到上限 {max_total}，拒绝发起请求"
        state["status"] = "partial" if (state["session"].revealed) else "incomplete"
        state["stop_code"] = "global_request_budget_exhausted"
        state["stop_reason"] = reason
        state["error"] = reason
        _event(
            state,
            {"event": "rejected", "code": "global_request_budget_exhausted", "reason": reason},
        )
        return state, None

    timeout = min(float(state["llm_config"].get("request_timeout_s", 30.0)), max(1.0, remaining))
    request_index = int(state.get("request_count", 0)) + 1
    step_index = len(state.get("trace_steps", [])) + 1
    _append_step(
        state,
        {
            "request": step_index,
            "phase": phase,
            "model_request_index": request_index,
            "visible": copy.deepcopy(dict(visible)),
            "model_request": {
                "tools": bool(tools),
                "timeout_s": round(timeout, 2),
            },
            "response": None,
            "decision": None,
            "tool_call": None,
            "tool_return": None,
            "remaining_budget_after": int(visible["remaining_budget"]),
            "best_after": None,
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
    # 在真正调用客户端前计数：客户端超时/报错、后续轨迹生成或写工件失败都不漏计。
    request_counter = state.get("request_counter")
    if request_counter is not None:
        request_counter[0] = request_index
    if global_counter is not None:
        global_counter[0] = int(global_counter[0]) + 1
    try:
        response = state["chat_fn"](
            state["llm_config"], copy.deepcopy(messages), tools=tools, timeout=timeout
        )
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, RuntimeError) as error:
        message = _safe_error(error, state.get("llm_config"))
        status = "partial" if (state["session"].revealed) else "incomplete"
        _update_last_step(
            state,
            response={"status": "error", "error": message},
        )
        _shutdown(
            state,
            status=status,
            code="llm_request_failed",
            reason=f"第 {request_index} 次模型请求失败：{message}",
            error=message,
            error_code="llm_request_failed",
        )
        _event(state, {"event": "llm_error", "request": request_index, "error": message})
        return state, None
    if not isinstance(response, dict):
        message = "模型返回不是 JSON 对象"
        status = "partial" if (state["session"].revealed) else "incomplete"
        _update_last_step(state, response={"status": "error", "error": message})
        _shutdown(
            state,
            status=status,
            code="invalid_response",
            reason=message,
            error=message,
            error_code="invalid_response",
        )
        return state, None
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
            "tool_call_count": len(summary.get("tool_calls") or []),
            "content_chars": summary.get("content_chars", 0),
            "finish_reason": summary.get("finish_reason"),
        },
    )
    return state, response


def _tool_step_update(
    state: LLMState,
    *,
    call_id: str,
    arguments: Mapping[str, Any],
    outcome: Mapping[str, Any],
) -> None:
    returned: dict[str, Any] = {
        "status": outcome.get("status"),
        "call_id": call_id,
    }
    if outcome.get("status") == "ok":
        returned["quota"] = outcome["quota"]
        returned["metrics"] = copy.deepcopy(outcome["result"])
    else:
        returned["code"] = outcome.get("code")
        returned["message"] = outcome.get("message")
    _update_last_step(
        state,
        tool_call={"name": TOOL_NAME, "id": call_id, "arguments": dict(arguments)},
        tool_return=returned,
        remaining_budget_after=int(outcome.get("remaining_budget", 0)),
        best_after=copy.deepcopy(outcome.get("best") or state["session"].best),
    )


def node_request_first(state: LLMState, chat_fn: ChatFn) -> LLMState:
    state = _enter(state, "request_first")
    visible = state["session"].snapshot()
    state, response = _call_model(
        state,
        phase="first",
        tools=[TOOL_FUNCTION],
        visible=visible,
        messages=state["dialogue"],
    )
    if response is None:
        return _leave(state, "request_first")
    state["dialogue"] = [*state["dialogue"], _assistant_message(response)]
    try:
        parsed = parse_single_fusion_call(response)
    except VerificationError as error:
        message = _safe_error(error, state.get("llm_config"))
        _shutdown(
            state,
            status="incomplete",
            code="invalid_first_request",
            reason=f"首轮请求未形成合法工具调用：{message}",
            error=message,
            error_code="invalid_first_request",
        )
        return _leave(state, "request_first")
    state["current_call_id"] = parsed["id"]
    state["current_quota"] = parsed["quota"]
    state["current_arguments"] = parsed["arguments"]
    _update_last_step(
        state,
        decision={
            "action": "choose",
            "quota": parsed["quota"],
            "call_id": parsed["id"],
            "reason": "模型首轮原生工具调用",
        },
        tool_call={"name": TOOL_NAME, "id": parsed["id"], "arguments": parsed["arguments"]},
    )
    _event(
        state,
        {
            "event": "tool_call_accepted",
            "request": state.get("request_count"),
            "tool_call_id": parsed["id"],
            "quota": parsed["quota"],
        },
    )
    return _leave(state, "request_first")


def node_execute_first(state: LLMState, chat_fn: ChatFn) -> LLMState:
    state = _enter(state, "execute_first")
    outcome = state["session"].request(state["current_quota"], step=1)
    _tool_step_update(
        state,
        call_id=state["current_call_id"],
        arguments=state["current_arguments"],
        outcome=outcome,
    )
    if outcome["status"] != "ok":
        _shutdown(
            state,
            status="incomplete",
            code="first_request_rejected",
            reason=f"首轮配额查询被拒绝（{outcome['code']}）：{outcome['message']}",
            error=outcome["message"],
            error_code=outcome["code"],
        )
        _event(
            state,
            {
                "event": "tool_rejected",
                "tool_call_id": state["current_call_id"],
                "code": outcome["code"],
                "message": outcome["message"],
            },
        )
        return _leave(state, "execute_first")
    state["first_result"] = copy.deepcopy(outcome["result"])
    state["tool_calls"] = 1
    state["dialogue"] = [
        *state["dialogue"],
        _tool_message(state["current_call_id"], outcome["result"]),
    ]
    _event(
        state,
        {
            "event": "tool_returned",
            "tool_call_id": state["current_call_id"],
            "quota": outcome["quota"],
            "result": copy.deepcopy(outcome["result"]),
            "remaining_budget": outcome["remaining_budget"],
        },
    )
    return _leave(state, "execute_first")


def node_request_feedback(state: LLMState, chat_fn: ChatFn) -> LLMState:
    state = _enter(state, "request_feedback")
    visible = state["session"].snapshot()
    state["dialogue"] = [
        *state["dialogue"],
        _followup_user_message(visible, "feedback"),
    ]
    state, response = _call_model(
        state,
        phase="feedback",
        tools=[TOOL_FUNCTION],
        visible=visible,
        messages=state["dialogue"],
    )
    if response is None:
        return _leave(state, "request_feedback")
    state["dialogue"] = [*state["dialogue"], _assistant_message(response)]
    calls = _message_tool_calls(response)
    if not calls:
        reply_status, content, finish_reason = _reply_status(response)
        if reply_status == "truncated":
            _shutdown(
                state,
                status="partial",
                code="reply_truncated",
                reason=(
                    f"模型停止说明以 finish_reason={finish_reason or 'missing'} 结束，"
                    f"正文虽非空但不能作为成功停止说明；已保留 "
                    f"{len(state['session'].revealed)} 次揭晓，剩余预算 {state['session'].remaining}。"
                ),
                error="reply truncated",
                error_code="reply_truncated",
                decision_action="stop",
                stop_extra={"finish_reason": finish_reason, "reply_status": "truncated"},
            )
            return _leave(state, "request_feedback")
        if reply_status == "non_stop":
            _shutdown(
                state,
                status="partial",
                code="non_stop_finish_reason",
                reason=(
                    f"模型停止说明的 finish_reason={finish_reason or 'missing'} 不是 stop；"
                    f"已保留 {len(state['session'].revealed)} 次揭晓，剩余预算 {state['session'].remaining}。"
                ),
                error="non-stop finish_reason",
                error_code="non_stop_finish_reason",
                decision_action="stop",
                stop_extra={"finish_reason": finish_reason, "reply_status": "non_stop"},
            )
            return _leave(state, "request_feedback")
        if reply_status == "empty":
            _shutdown(
                state,
                status="partial",
                code="empty_feedback",
                reason="第一次揭晓后的模型回复为空，无法形成停止理由",
                error="empty feedback",
                error_code="empty_feedback",
                decision_action="stop",
                stop_extra={"finish_reason": finish_reason, "reply_status": "empty"},
            )
            return _leave(state, "request_feedback")
        state["final_reply"] = _safe_text(content)
        state["stop_code"] = "model_stop"
        state["stop_reason"] = _safe_text(content)
        _update_last_step(
            state,
            decision={"action": "stop", "quota": None, "reason": _safe_text(content)},
            stop={
                "code": "model_stop",
                "reason": _safe_text(content),
                "status": "running",
                "finish_reason": finish_reason,
                "reply_status": "ok",
            },
        )
        state["route_after_feedback"] = "finish"
        _event(state, {"event": "model_stop", "request": state.get("request_count"), "reason": _safe_text(content)})
        return _leave(state, "request_feedback")
    try:
        parsed = parse_single_fusion_call(response)
    except VerificationError as error:
        message = _safe_error(error, state.get("llm_config"))
        _shutdown(
            state,
            status="partial",
            code="invalid_feedback_request",
            reason=f"第一次揭晓后的请求被拒绝：{message}",
            error=message,
            error_code="invalid_feedback_request",
        )
        return _leave(state, "request_feedback")
    state["current_call_id"] = parsed["id"]
    state["current_quota"] = parsed["quota"]
    state["current_arguments"] = parsed["arguments"]
    _update_last_step(
        state,
        decision={
            "action": "choose",
            "quota": parsed["quota"],
            "call_id": parsed["id"],
            "reason": "模型看到第一次结果后的第二次原生工具调用",
        },
        tool_call={"name": TOOL_NAME, "id": parsed["id"], "arguments": parsed["arguments"]},
    )
    state["route_after_feedback"] = "execute_second"
    _event(
        state,
        {
            "event": "tool_call_accepted",
            "request": state.get("request_count"),
            "tool_call_id": parsed["id"],
            "quota": parsed["quota"],
        },
    )
    return _leave(state, "request_feedback")


def node_execute_second(state: LLMState, chat_fn: ChatFn) -> LLMState:
    state = _enter(state, "execute_second")
    outcome = state["session"].request(state["current_quota"], step=2)
    _tool_step_update(
        state,
        call_id=state["current_call_id"],
        arguments=state["current_arguments"],
        outcome=outcome,
    )
    if outcome["status"] != "ok":
        _shutdown(
            state,
            status="partial",
            code="second_request_rejected",
            reason=f"第二次配额查询被拒绝（{outcome['code']}）：{outcome['message']}",
            error=outcome["message"],
            error_code=outcome["code"],
        )
        _event(
            state,
            {
                "event": "tool_rejected",
                "tool_call_id": state["current_call_id"],
                "code": outcome["code"],
                "message": outcome["message"],
            },
        )
        return _leave(state, "execute_second")
    state["second_result"] = copy.deepcopy(outcome["result"])
    state["tool_calls"] = 2
    state["dialogue"] = [
        *state["dialogue"],
        _tool_message(state["current_call_id"], outcome["result"]),
    ]
    _event(
        state,
        {
            "event": "tool_returned",
            "tool_call_id": state["current_call_id"],
            "quota": outcome["quota"],
            "result": copy.deepcopy(outcome["result"]),
            "remaining_budget": outcome["remaining_budget"],
        },
    )
    return _leave(state, "execute_second")


def node_request_final(state: LLMState, chat_fn: ChatFn) -> LLMState:
    state = _enter(state, "request_final")
    visible = state["session"].snapshot()
    state["dialogue"] = [
        *state["dialogue"],
        _followup_user_message(visible, "final"),
    ]
    state, response = _call_model(
        state,
        phase="final",
        tools=None,
        visible=visible,
        messages=state["dialogue"],
    )
    if response is None:
        return _leave(state, "request_final")
    state["dialogue"] = [*state["dialogue"], _assistant_message(response)]
    if _message_tool_calls(response):
        calls = _message_tool_calls(response)
        first_call = calls[0] if calls else {}
        function = first_call.get("function") or {}
        try:
            args = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"raw": _safe_text(function.get("arguments"), 100)}
        _update_last_step(
            state,
            tool_call={
                "name": function.get("name"),
                "id": first_call.get("id"),
                "arguments": args if isinstance(args, dict) else {"raw": args},
            },
            tool_return={
                "status": "error",
                "call_id": first_call.get("id"),
                "code": "final_tool_call_rejected",
                "message": "预算用尽后禁止工具调用，未揭晓新指标",
            },
        )
        _shutdown(
            state,
            status="partial",
            code="final_tool_call_rejected",
            reason="预算用尽后的最终回复再次请求工具，执行器拒绝且不揭晓新指标",
            error="final response contained tool_calls",
            error_code="final_tool_call_rejected",
            decision_action="stop",
            stop_extra={"finish_reason": _finish_reason(response), "reply_status": "tool_calls"},
        )
        return _leave(state, "request_final")
    reply_status, content, finish_reason = _reply_status(response)
    if reply_status == "truncated":
        _shutdown(
            state,
            status="partial",
            code="reply_truncated",
            reason=(
                f"模型最终回复以 finish_reason={finish_reason or 'missing'} 结束，"
                f"正文虽非空但不能作为成功最终回复；已保留 "
                f"{len(state['session'].revealed)} 次揭晓，剩余预算 {state['session'].remaining}。"
            ),
            error="reply truncated",
            error_code="reply_truncated",
            decision_action="stop",
            stop_extra={"finish_reason": finish_reason, "reply_status": "truncated"},
        )
        return _leave(state, "request_final")
    if reply_status == "non_stop":
        _shutdown(
            state,
            status="partial",
            code="non_stop_finish_reason",
            reason=(
                f"模型最终回复的 finish_reason={finish_reason or 'missing'} 不是 stop；"
                f"已保留 {len(state['session'].revealed)} 次揭晓，剩余预算 {state['session'].remaining}。"
            ),
            error="non-stop finish_reason",
            error_code="non_stop_finish_reason",
            decision_action="stop",
            stop_extra={"finish_reason": finish_reason, "reply_status": "non_stop"},
        )
        return _leave(state, "request_final")
    if reply_status == "empty":
        _shutdown(
            state,
            status="partial",
            code="empty_final_reply",
            reason="预算用尽后的最终回复为空",
            error="empty final reply",
            error_code="empty_final_reply",
            decision_action="stop",
            stop_extra={"finish_reason": finish_reason, "reply_status": "empty"},
        )
        return _leave(state, "request_final")
    state["final_reply"] = _safe_text(content)
    state["stop_code"] = "budget_exhausted_final"
    state["stop_reason"] = _safe_text(content)
    _update_last_step(
        state,
        decision={"action": "stop", "quota": None, "reason": _safe_text(content)},
        stop={
            "code": "budget_exhausted_final",
            "reason": _safe_text(content),
            "status": "running",
            "finish_reason": finish_reason,
            "reply_status": "ok",
        },
    )
    _event(state, {"event": "model_final", "request": state.get("request_count"), "reason": _safe_text(content)})
    return _leave(state, "request_final")


def node_finish(state: LLMState, chat_fn: ChatFn) -> LLMState:
    state = _enter(state, "finish")
    session = state["session"]
    summary = session.summary()
    if state.get("status", "running") == "running":
        if summary["reveals"] >= 1 and state.get("final_reply") and state.get(
            "stop_code"
        ) in {"model_stop", "budget_exhausted_final"}:
            state["status"] = "verified"
        elif summary["reveals"] == 0:
            state["status"] = "incomplete"
            state.setdefault("stop_code", "no_successful_tool_call")
            state.setdefault("stop_reason", "没有成功的工具调用")
            state.setdefault("error", state["stop_reason"])
        else:
            state["status"] = "partial"
            state.setdefault("stop_code", "incomplete_workflow")
            state.setdefault("stop_reason", "工作流未形成合法最终回复")
            state.setdefault("error", state["stop_reason"])

    max_requests = int(state.get("max_model_requests", MAX_MODEL_REQUESTS))
    if state["status"] == "verified" and int(state.get("request_count", 0)) > max_requests:
        state["status"] = "partial"
        state["stop_code"] = "request_budget_exceeded"
        state["stop_reason"] = f"模型请求次数超过 {max_requests}"
        state["error"] = state["stop_reason"]
    if state["status"] == "verified" and summary["reveals"] > session.budget:
        state["status"] = "partial"
        state["stop_code"] = "reveal_budget_exceeded"
        state["stop_reason"] = f"成功揭晓次数超过 {session.budget}"
        state["error"] = state["stop_reason"]

    best = summary["best"]
    first_best_reveal = 0
    if best["quota"] not in session.starting_keys:
        for index, result in enumerate(session.revealed, start=1):
            if result["quota"] == best["quota"]:
                first_best_reveal = index
                break
    baseline = state["baseline"]
    state["summary"] = {
        **summary,
        "status": state["status"],
        "request_count": int(state.get("request_count", 0)),
        "tool_calls": int(state.get("tool_calls", 0)),
        "first_best_reveal": first_best_reveal,
        "stop_code": state.get("stop_code"),
        "stop_reason": state.get("stop_reason"),
        "final_reply": state.get("final_reply", ""),
        "error": state.get("error"),
        "scenario": state.get("scenario", "single"),
        "max_model_requests": max_requests,
        "starting_quotas": [record["quota"] for record in session.starting_histories],
        "starting_histories": copy.deepcopy(session.starting_histories),
    }
    state["conclusion"] = {
        "status": state["status"],
        "best_quota": best["quota"],
        "best": copy.deepcopy(best),
        "baseline": copy.deepcopy(baseline),
        "delta_vs_baseline": {
            "hits": int(best["hits"]) - int(baseline["hits"]),
            "recall_at_100": float(best["recall_at_100"]) - float(baseline["recall_at_100"]),
            "coverage": float(best["coverage"]) - float(baseline["coverage"]),
        },
        "reveals": summary["reveals"],
        "request_count": int(state.get("request_count", 0)),
        "tool_calls": int(state.get("tool_calls", 0)),
        "first_best_reveal": first_best_reveal,
        "stop_code": state.get("stop_code"),
        "stop_reason": state.get("stop_reason"),
        "final_reply": state.get("final_reply", ""),
    }
    if state.get("trace_steps"):
        existing_stop = state["trace_steps"][-1].get("stop") or {}
        _update_last_step(
            state,
            stop={
                **existing_stop,
                "code": state.get("stop_code"),
                "reason": state.get("stop_reason"),
                "status": state["status"],
            },
            best_after=copy.deepcopy(best),
        )
    _event(
        state,
        {
            "event": "finish",
            "status": state["status"],
            "stop_code": state.get("stop_code"),
            "request_count": state.get("request_count"),
            "tool_calls": state.get("tool_calls"),
            "reveals": summary["reveals"],
        },
    )
    return _leave(state, "finish")


def _route(next_node: str):
    def route(state: LLMState) -> str:
        if state.get("status") != "running":
            return "finish"
        return next_node

    return route


def _route_feedback(state: LLMState) -> str:
    if state.get("status") != "running":
        return "finish"
    return "execute_second" if state.get("route_after_feedback") == "execute_second" else "finish"


def _route_after_first(state: LLMState) -> str:
    if state.get("status") != "running":
        return "finish"
    return "request_final" if state["session"].remaining == 0 else "request_feedback"


def build_graph(chat_fn: ChatFn):
    graph = StateGraph(LLMState)
    graph.add_node("request_first", lambda state: node_request_first(state, chat_fn))
    graph.add_node("execute_first", lambda state: node_execute_first(state, chat_fn))
    graph.add_node("request_feedback", lambda state: node_request_feedback(state, chat_fn))
    graph.add_node("execute_second", lambda state: node_execute_second(state, chat_fn))
    graph.add_node("request_final", lambda state: node_request_final(state, chat_fn))
    graph.add_node("finish", lambda state: node_finish(state, chat_fn))
    graph.set_entry_point("request_first")
    graph.add_conditional_edges("request_first", _route("execute_first"))
    graph.add_conditional_edges("execute_first", _route_after_first)
    graph.add_conditional_edges("request_feedback", _route_feedback)
    graph.add_conditional_edges("execute_second", _route("request_final"))
    graph.add_edge("request_final", "finish")
    graph.add_edge("finish", END)
    return graph.compile()


def run_llm_replay(
    *,
    records: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    llm_cfg: Mapping[str, Any],
    chat_fn: ChatFn = post_chat,
    started_at: float | None = None,
    session: Any | None = None,
    starting_quotas: Sequence[str] | None = None,
    allowed_quotas: Sequence[str] | None = None,
    budget: int | None = None,
    max_model_requests: int | None = None,
    scenario: str | None = None,
    request_counter: list[int] | None = None,
    global_request_counter: list[int] | None = None,
    max_total_requests: int | None = None,
) -> LLMState:
    if session is None:
        if starting_quotas is not None:
            session = replay.MultiStartReplaySession(
                records=records,
                starting_quotas=starting_quotas,
                allowed_quotas=allowed_quotas or config["replay"]["allowed_quotas"],
                budget=int(budget if budget is not None else config["replay"]["budget_reveals_per_side"]),
                decimals=int(config["best_rule"]["recall_decimals"]),
            )
        else:
            session = replay.ReplaySession(
                records=records,
                baseline_quota=config["replay"]["baseline_quota"],
                allowed_quotas=allowed_quotas or config["replay"]["allowed_quotas"],
                budget=int(budget if budget is not None else config["replay"]["budget_reveals_per_side"]),
                decimals=int(config["best_rule"]["recall_decimals"]),
            )
    max_requests = int(max_model_requests if max_model_requests is not None else MAX_MODEL_REQUESTS)
    started = started_at if started_at is not None else time.perf_counter()
    initial_visible = session.snapshot()
    dialogue = _messages(initial_visible, "first")
    state: LLMState = {
        "status": "running",
        "stage": "init",
        "nodes": [],
        "events": [
            {
                "event": "start",
                "model": llm_cfg.get("model"),
                "endpoint_host": host_only(str(llm_cfg.get("base_url", ""))),
                "max_model_requests": max_requests,
                "max_reveals": session.budget,
                "scenario": scenario,
            }
        ],
        "started": started,
        "deadline": started + float(config["llm"]["total_budget_s"]),
        "llm_config": {
            **dict(llm_cfg),
            "request_timeout_s": float(config["llm"]["request_timeout_s"]),
        },
        "model": str(llm_cfg.get("model", "")),
        "endpoint_host": host_only(str(llm_cfg.get("base_url", ""))),
        "baseline": copy.deepcopy(session.starting_histories[0]),
        "starting_histories": copy.deepcopy(session.starting_histories),
        "candidates": list(session.candidates),
        "scenario": scenario or "single",
        "max_model_requests": max_requests,
        "session": session,
        "request_count": 0,
        "request_counter": request_counter,
        "global_request_counter": global_request_counter,
        "max_total_requests": int(max_total_requests) if max_total_requests is not None else 0,
        "tool_calls": 0,
        "response_ids": {},
        "trace_steps": [],
        "dialogue": dialogue,
        "chat_fn": chat_fn,  # type: ignore[typeddict-unknown-key]
    }
    graph = build_graph(chat_fn)
    final_state = graph.invoke(state)
    return final_state  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 三方指标与比较
# ---------------------------------------------------------------------------


def _side_record(summary: Mapping[str, Any]) -> dict[str, Any]:
    best = summary.get("best") or {}
    return {
        "quota": best.get("quota"),
        "two_tower": best.get("two_tower"),
        "mind": best.get("mind"),
        "hits": best.get("hits"),
        "recall_at_100": best.get("recall_at_100"),
        "coverage": best.get("coverage"),
        "users": best.get("users"),
        "targets": best.get("targets"),
        "candidates_total": best.get("candidates_total"),
    }


def _winner(
    llm: Mapping[str, Any],
    local: Mapping[str, Any],
    fixed: Mapping[str, Any],
    decimals: int,
) -> tuple[str, dict[str, Any]]:
    records = {
        "llm": llm,
        "local": local,
        "fixed": fixed,
    }
    best = replay.select_best(records.values(), decimals)
    names = [
        name
        for name, record in records.items()
        if replay.compare_records(record, best, decimals) == 0
    ]
    winner = "tie" if len(names) > 1 else names[0]
    return winner, {
        "best_quota": best["quota"],
        "names_at_best": names,
    }


def build_metrics(
    *,
    config: Mapping[str, Any],
    sources: Mapping[str, Any],
    frozen: Mapping[str, Any],
    self_check: Mapping[str, Any],
    final: Mapping[str, Any],
    run_meta: Mapping[str, Any],
) -> dict[str, Any]:
    llm_summary = final.get("summary") or {}
    llm_record = _side_record(llm_summary)
    local_record = _side_record(frozen["local_agent"]["summary"])
    fixed_record = _side_record(frozen["fixed"]["summary"])
    decimals = int(config["best_rule"]["recall_decimals"])
    winner, winner_info = _winner(llm_record, local_record, fixed_record, decimals)
    llm_verified = llm_summary.get("status") == "verified"
    overall_best = replay.select_best([llm_record, local_record, fixed_record], decimals)
    llm_cmp = replay.compare_records(llm_record, overall_best, decimals)
    llm_outcome = "win" if llm_cmp < 0 else ("tie" if llm_cmp == 0 else "lose")

    def deltas(reference: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "hits": int(llm_record.get("hits") or 0) - int(reference.get("hits") or 0),
            "recall_at_100": float(llm_record.get("recall_at_100") or 0.0)
            - float(reference.get("recall_at_100") or 0.0),
            "coverage": float(llm_record.get("coverage") or 0.0)
            - float(reference.get("coverage") or 0.0),
            "reveals": int(llm_summary.get("reveals") or 0)
            - int(reference.get("_reveals") or 0),
        }

    local_ref = {**local_record, "_reveals": frozen["local_agent"]["summary"]["reveals"]}
    fixed_ref = {**fixed_record, "_reveals": frozen["fixed"]["summary"]["reveals"]}
    saved_local = None
    saved_fixed = None
    if llm_record.get("quota") == local_record.get("quota"):
        saved_local = int(frozen["local_agent"]["summary"]["reveals"]) - int(llm_summary.get("reveals") or 0)
    if llm_record.get("quota") == fixed_record.get("quota"):
        saved_fixed = int(frozen["fixed"]["summary"]["reveals"]) - int(llm_summary.get("reveals") or 0)

    comparison_status = "judged" if llm_verified else "not_judged"
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": llm_summary.get("status"),
        "model": {
            "name": run_meta.get("model"),
            "endpoint_host": run_meta.get("endpoint_host"),
        },
        "rules": {
            "baseline_quota": config["replay"]["baseline_quota"],
            "allowed_quotas": list(config["replay"]["allowed_quotas"]),
            "budget_reveals_per_side": int(config["replay"]["budget_reveals_per_side"]),
            "best_rule": config["best_rule"],
            "fixed_order": list(config["fixed_policy"]["order"]),
            "local_policy": config["agent_policy"],
            "llm_tool": TOOL_NAME,
            "max_model_requests": MAX_MODEL_REQUESTS,
            "max_reveals": MAX_REVEALS,
        },
        "llm": {
            **llm_summary,
            "best": llm_record,
        },
        "local": {
            **frozen["local_agent"]["summary"],
            "status": "frozen_baseline",
            "policy": frozen["local_agent"]["policy"],
            "first_best_reveal": frozen["local_agent"]["first_best_reveal"],
        },
        "fixed": {
            **frozen["fixed"]["summary"],
            "status": "frozen_baseline",
            "policy": frozen["fixed"]["policy"],
            "first_best_reveal": frozen["fixed"]["first_best_reveal"],
        },
        "comparison": {
            "status": comparison_status,
            "winner": winner if llm_verified else "not_judged",
            "llm_outcome": llm_outcome if llm_verified else "not_judged",
            "best_quota": winner_info["best_quota"] if llm_verified else None,
            "names_at_best": winner_info["names_at_best"] if llm_verified else [],
            "reason": (
                "真实 LLM 工作流达到 verified，按统一最佳规则比较三方。"
                if llm_verified
                else f"真实 LLM 工作流状态为 {llm_summary.get('status')}，不按完整结果判胜负。"
            ),
            "llm_vs_local": deltas(local_ref),
            "llm_vs_fixed": deltas(fixed_ref),
            "saved_reveals_vs_local_if_same_best": saved_local,
            "saved_reveals_vs_fixed_if_same_best": saved_fixed,
            "efficiency": {
                "llm": {
                    "reveals": int(llm_summary.get("reveals") or 0),
                    "request_count": int(llm_summary.get("request_count") or 0),
                    "elapsed_s": run_meta.get("elapsed_s"),
                },
                "local": {
                    "reveals": int(frozen["local_agent"]["summary"]["reveals"]),
                    "request_count": 0,
                    "elapsed_s": None,
                },
                "fixed": {
                    "reveals": int(frozen["fixed"]["summary"]["reveals"]),
                    "request_count": 0,
                    "elapsed_s": None,
                },
            },
        },
        "baselines": {
            "previous_run": frozen["previous_run"],
            "files": frozen["files"],
            "rule_checks": frozen["rule_checks"],
            "previous_source_files": frozen["previous_source_files"],
        },
        "consistency": sources["manifest"]["consistency"],
        "self_check": {
            "status": self_check.get("status"),
            "passed": self_check.get("passed"),
            "total": self_check.get("total"),
            "synthetic_only": True,
        },
        "scope": {
            **{key: value for key, value in config["scope"].items() if key not in {"llm_requests", "network_requests"}},
            "llm_requests": int(final.get("summary", {}).get("request_count") or 0),
            "network_requests": int(final.get("summary", {}).get("request_count") or 0),
            "other_network_requests": 0,
            "real_evaluation": 0,
            "training": 0,
            "recall": 0,
        },
        "run": {
            "command": run_meta.get("command"),
            "started_at": run_meta.get("started_at"),
            "elapsed_s": run_meta.get("elapsed_s"),
            "output_dir": run_meta.get("output_dir"),
        },
    }


# ---------------------------------------------------------------------------
# 本地替身检查（不联网）
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


def _tool_response(call_id: str, two_tower: int, mind: int) -> dict[str, Any]:
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
                                "arguments": json.dumps({"two_tower": two_tower, "mind": mind}),
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


def _first_user_payload(chat: ScriptedChat) -> dict[str, Any]:
    messages = chat.calls[0]["messages"]
    user = next(item for item in messages if item.get("role") == "user")
    return json.loads(user["content"])


def _run_fake_case(
    config: Mapping[str, Any],
    responses: Sequence[Any],
    *,
    records: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[LLMState, ScriptedChat, replay.ReplaySession]:
    chat = ScriptedChat(responses)
    table = records or replay.synthetic_table()
    session_records = table
    final = run_llm_replay(
        records=session_records,
        config=config,
        llm_cfg={"model": "fake-model", "base_url": "http://127.0.0.1", "api_key": "fake-key"},
        chat_fn=chat,
    )
    return final, chat, final["session"]


def run_self_check(config: Mapping[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(check_id: str, function: Callable[[], str]) -> None:
        try:
            detail = function()
        except Exception as error:  # noqa: BLE001 - 自检失败也写入报告
            checks.append(
                {
                    "id": check_id,
                    "status": "fail",
                    "detail": f"{type(error).__name__}: {_safe_error(error)}",
                }
            )
        else:
            checks.append({"id": check_id, "status": "pass", "detail": detail})

    def first_visibility() -> str:
        final, chat, _ = _run_fake_case(
            config,
            [_tool_response("call-first", 30, 70), _text_response("30/70 优于基准，停止")],
        )
        payload = _first_user_payload(chat)
        assert payload["starting_history"]["quota"] == "50/50"
        assert payload["revealed_results"] == []
        assert payload["remaining_budget"] == 2
        assert payload["allowed_quotas"] == list(config["replay"]["allowed_quotas"])
        raw = chat.calls[0]["messages"][-1]["content"]
        for hidden in ("0.013672", "0.009115", "0.005208"):
            assert hidden not in raw, f"首轮消息泄露隐藏指标 {hidden}"
        assert final["status"] == "verified"
        return "首轮消息只有 50/50 历史、候选标签和 2 次预算；不含其他配额指标"

    def call_id_roundtrip() -> str:
        final, chat, _ = _run_fake_case(
            config,
            [_tool_response("call-roundtrip", 30, 70), _text_response("停止")],
        )
        step = final["trace_steps"][0]
        assert step["tool_call"]["id"] == "call-roundtrip"
        assert step["tool_return"]["call_id"] == "call-roundtrip"
        second_messages = chat.calls[1]["messages"]
        tool_messages = [item for item in second_messages if item.get("role") == "tool"]
        assert len(tool_messages) == 1, tool_messages
        assert tool_messages[0]["tool_call_id"] == "call-roundtrip"
        tool_payload = json.loads(tool_messages[0]["content"])
        assert tool_payload["quota"] == "30/70"
        assert tool_payload["hits"] == 5
        return "首轮工具返回与第二次请求的 tool 消息使用同一调用 ID"

    def verified_one_reveal() -> str:
        final, _, session = _run_fake_case(
            config,
            [_tool_response("call-one", 30, 70), _text_response("30/70 命中 5，停止")],
        )
        assert final["status"] == "verified", final["status"]
        assert final["summary"]["reveals"] == 1
        assert final["summary"]["request_count"] == 2
        assert final["summary"]["tool_calls"] == 1
        assert final["summary"]["best_quota"] == "30/70"
        assert session.remaining == 1
        return "一次成功揭晓 + 模型主动停止 → verified，2 次请求 / 1 次揭晓"

    def verified_two_reveals() -> str:
        final, _, _ = _run_fake_case(
            config,
            [
                _tool_response("call-a", 30, 70),
                _tool_response("call-b", 20, 80),
                _text_response("20/80 与 30/70 持平但更远，保留 30/70"),
            ],
        )
        assert final["status"] == "verified", final["status"]
        assert final["summary"]["reveals"] == 2
        assert final["summary"]["request_count"] == 3
        assert final["summary"]["tool_calls"] == 2
        assert final["summary"]["best_quota"] == "30/70"
        return "两次成功揭晓 + 最终回复 → verified，3 次请求 / 2 次揭晓"

    def partial_final_failure() -> str:
        final, _, _ = _run_fake_case(
            config,
            [
                _tool_response("call-a", 30, 70),
                _tool_response("call-b", 20, 80),
                TimeoutError("simulated final timeout"),
            ],
        )
        assert final["status"] == "partial", final["status"]
        assert final["summary"]["reveals"] == 2
        assert final["summary"]["request_count"] == 3
        assert final["summary"]["stop_code"] == "llm_request_failed"
        assert final["summary"]["best_quota"] == "30/70"
        return "已有两次揭晓后最终回复超时 → partial，保留结果与轨迹"

    def partial_duplicate() -> str:
        final, _, _ = _run_fake_case(
            config,
            [_tool_response("call-a", 30, 70), _tool_response("call-b", 30, 70)],
        )
        assert final["status"] == "partial", final["status"]
        assert final["summary"]["reveals"] == 1
        rejected = final["session"].invalid_requests
        assert rejected and rejected[-1]["code"] == "duplicate_quota", rejected
        assert final["trace_steps"][-1]["tool_return"]["status"] == "error"
        assert "metrics" not in final["trace_steps"][-1]["tool_return"]
        return "第二次重复查询 30/70 被拒绝 → partial，未揭晓第二组指标"

    def incomplete_no_tool() -> str:
        final, _, _ = _run_fake_case(config, [_text_response("我决定停止，不调用工具")])
        assert final["status"] == "incomplete", final["status"]
        assert final["summary"]["reveals"] == 0
        assert final["summary"]["request_count"] == 1
        return "首轮无工具调用 → incomplete，0 次揭晓"

    def incomplete_illegal() -> str:
        final, _, _ = _run_fake_case(config, [_tool_response("call-bad", 60, 40)])
        assert final["status"] == "incomplete", final["status"]
        assert final["summary"]["reveals"] == 0
        assert final["session"].invalid_requests[-1]["code"] == "illegal_quota"
        return "首轮非法配额 60/40 → incomplete，未揭晓指标"

    def assert_not_judged(final: Mapping[str, Any]) -> None:
        def stub_summary(quota: str, hits: int, recall: float, coverage: float) -> dict[str, Any]:
            record = replay.synthetic_record(quota, hits, recall, coverage)
            return {
                "reveals": 2,
                "best": record,
                "best_quota": quota,
                "hits": hits,
                "recall_at_100": recall,
                "coverage": coverage,
                "stop_reason": "synthetic baseline",
            }

        frozen_stub = {
            "previous_run": "synthetic",
            "files": {},
            "local_agent": {
                "summary": stub_summary("30/70", 5, 0.013672, 0.026720),
                "policy": {"name": "local_feedback_v1"},
                "first_best_reveal": 1,
            },
            "fixed": {
                "summary": stub_summary("30/70", 5, 0.013672, 0.026720),
                "policy": {"name": "fixed_symmetric_probe"},
                "first_best_reveal": 2,
            },
            "rule_checks": {},
            "previous_source_files": {},
        }
        metrics = build_metrics(
            config=config,
            sources={"manifest": {"consistency": {"status": "pass"}}},
            frozen=frozen_stub,
            self_check={"status": "pass", "passed": 1, "total": 1},
            final=final,
            run_meta={
                "model": "fake",
                "endpoint_host": "http://127.0.0.1",
                "elapsed_s": 0.0,
                "command": "fake",
                "output_dir": "fake",
            },
        )
        assert metrics["comparison"]["status"] == "not_judged", metrics["comparison"]
        assert metrics["comparison"]["winner"] == "not_judged", metrics["comparison"]

    def partial_reply_truncated_feedback() -> str:
        final, _, session = _run_fake_case(
            config,
            [
                _tool_response("call-a", 30, 70),
                _text_response("30/70 命中 5，但这句话被截断", finish_reason="length"),
            ],
        )
        assert final["status"] == "partial", final["status"]
        assert final["summary"]["reveals"] == 1
        assert final["summary"]["remaining_budget"] == 1
        assert final["summary"]["request_count"] == 2
        assert final["summary"]["stop_code"] == "reply_truncated"
        assert final["trace_steps"][-1]["response"]["finish_reason"] == "length"
        assert final["trace_steps"][-1]["stop"]["finish_reason"] == "length"
        assert final["trace_steps"][-1]["stop"]["reply_status"] == "truncated"
        assert final["summary"]["final_reply"] == ""
        assert session.remaining == 1
        assert len(session.revealed) == 1
        assert_not_judged(final)
        return "揭晓一次后 length 停止说明 → partial / reply_truncated；1 次揭晓、1 次剩余预算、not_judged"

    def partial_reply_truncated_final() -> str:
        final, _, session = _run_fake_case(
            config,
            [
                _tool_response("call-a", 30, 70),
                _tool_response("call-b", 20, 80),
                _text_response("最终回复正文非空但被截断", finish_reason="length"),
            ],
        )
        assert final["status"] == "partial", final["status"]
        assert final["summary"]["reveals"] == 2
        assert final["summary"]["remaining_budget"] == 0
        assert final["summary"]["request_count"] == 3
        assert final["summary"]["stop_code"] == "reply_truncated"
        assert final["trace_steps"][-1]["response"]["finish_reason"] == "length"
        assert final["trace_steps"][-1]["stop"]["finish_reason"] == "length"
        assert final["trace_steps"][-1]["stop"]["reply_status"] == "truncated"
        assert final["summary"]["final_reply"] == ""
        assert session.remaining == 0
        assert len(session.revealed) == 2
        assert_not_judged(final)
        return "揭晓两次后 length 最终回复 → partial / reply_truncated；2 次揭晓、0 剩余预算、not_judged"

    def partial_final_tool_call() -> str:
        final, _, _ = _run_fake_case(
            config,
            [
                _tool_response("call-a", 30, 70),
                _tool_response("call-b", 20, 80),
                _tool_response("call-c", 70, 30),
            ],
        )
        assert final["status"] == "partial", final["status"]
        assert final["summary"]["reveals"] == 2
        assert final["summary"]["stop_code"] == "final_tool_call_rejected"
        assert "metrics" not in final["trace_steps"][-1]["tool_return"]
        return "预算用尽后最终回复再次请求工具 → partial，拒绝且不揭晓第三组"

    def over_budget_session() -> str:
        session = replay.ReplaySession(
            records=replay.synthetic_table(),
            baseline_quota=config["replay"]["baseline_quota"],
            allowed_quotas=config["replay"]["allowed_quotas"],
            budget=int(config["replay"]["budget_reveals_per_side"]),
            decimals=int(config["best_rule"]["recall_decimals"]),
        )
        assert session.request("30/70")["status"] == "ok"
        assert session.request("20/80")["status"] == "ok"
        rejected = session.request("70/30")
        assert rejected["status"] == "error"
        assert rejected["code"] == "budget_exhausted"
        assert "result" not in rejected
        assert session.remaining == 0
        assert len(session.revealed) == 2
        return "2 次揭晓后第 3 次查询被预算拒绝，无指标泄露"

    def max_request_guard() -> str:
        cases = [
            [_tool_response("c1", 30, 70), _text_response("停止")],
            [_tool_response("c1", 30, 70), _tool_response("c2", 20, 80), _text_response("停止")],
            [_tool_response("c1", 30, 70), _text_response("停止")],
        ]
        for responses in cases:
            final, _, _ = _run_fake_case(config, responses)
            assert final["summary"]["request_count"] <= 3
            assert final["summary"]["reveals"] <= 2
        return "自检场景的请求数 ≤ 3、揭晓数 ≤ 2"

    check("first_visibility", first_visibility)
    check("call_id_roundtrip", call_id_roundtrip)
    check("verified_one_reveal", verified_one_reveal)
    check("verified_two_reveals", verified_two_reveals)
    check("partial_final_failure", partial_final_failure)
    check("partial_duplicate", partial_duplicate)
    check("partial_reply_truncated_feedback", partial_reply_truncated_feedback)
    check("partial_reply_truncated_final", partial_reply_truncated_final)
    check("incomplete_no_tool", incomplete_no_tool)
    check("incomplete_illegal", incomplete_illegal)
    check("partial_final_tool_call", partial_final_tool_call)
    check("over_budget_session", over_budget_session)
    check("max_request_guard", max_request_guard)

    passed = sum(1 for item in checks if item["status"] == "pass")
    failed = len(checks) - passed
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if failed == 0 else "fail",
        "total": len(checks),
        "passed": passed,
        "failed": failed,
        "synthetic_only": True,
        "network_requests": 0,
        "checks": checks,
    }


def build_llm_trace(
    final: Mapping[str, Any], config: Mapping[str, Any], run_meta: Mapping[str, Any]
) -> dict[str, Any]:
    summary = final.get("summary") or {}
    steps = copy.deepcopy(final.get("trace_steps") or [])
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": summary.get("status"),
        "model": run_meta.get("model"),
        "endpoint_host": run_meta.get("endpoint_host"),
        "graph_nodes": list(final.get("nodes") or []),
        "budget": {
            "max_model_requests": MAX_MODEL_REQUESTS,
            "max_reveals": MAX_REVEALS,
            "request_count": int(summary.get("request_count") or 0),
            "reveals": int(summary.get("reveals") or 0),
            "elapsed_s": run_meta.get("elapsed_s"),
        },
        "steps": steps,
        "stop": {
            "code": summary.get("stop_code"),
            "reason": summary.get("stop_reason"),
            "status": summary.get("status"),
            "error": summary.get("error"),
        },
        "summary": copy.deepcopy(summary),
        "response_ids": copy.deepcopy(final.get("response_ids") or {}),
    }


def build_report(
    *,
    config: Mapping[str, Any],
    sources: Mapping[str, Any],
    frozen: Mapping[str, Any],
    self_check: Mapping[str, Any],
    final: Mapping[str, Any],
    metrics: Mapping[str, Any],
    run_meta: Mapping[str, Any],
) -> str:
    llm = metrics["llm"]
    local = metrics["local"]
    fixed = metrics["fixed"]
    comparison = metrics["comparison"]
    status = metrics["status"]
    if comparison["status"] == "judged":
        llm_text = {
            "win": "真实 LLM 胜出",
            "tie": "真实 LLM 与最佳持平",
            "lose": "真实 LLM 输给最佳",
        }[comparison.get("llm_outcome", "lose")]
        winner_text = {
            "llm": "真实 LLM 为三方最佳",
            "local": "本地规则 Agent 为三方最佳",
            "fixed": "固定策略为三方最佳",
            "tie": "三方最佳持平" if len(comparison.get("names_at_best") or []) >= 3 else "多方并列最佳",
        }[comparison["winner"]]
        outcome_text = f"**{llm_text}**；{winner_text}。{comparison['reason']}"
    else:
        outcome_text = f"**不判胜负**。{comparison['reason']}"

    def fmt_metrics(record: Mapping[str, Any]) -> str:
        if not record or record.get("hits") is None:
            return "—"
        return (
            f"{record['hits']} / {float(record['recall_at_100']):.6f} / "
            f"{float(record['coverage']):.6f}"
        )

    def cell(value: Any, limit: int = 180) -> str:
        return _safe_text(str(value or ""), limit).replace("\r", " ").replace("\n", " ").replace("|", "\\|")

    def short_stop(reason: Any, code: Any) -> str:
        if code == "budget_exhausted_final":
            return "预算用尽；最终回复见 §2"
        first_line = str(reason or "").splitlines()[0] if str(reason or "").strip() else "—"
        return cell(first_line, 100)

    def fmt_visible(visible: Mapping[str, Any]) -> str:
        parts = [f"50/50: {fmt_metrics(visible['starting_history'])}"]
        for result in visible.get("revealed") or []:
            parts.append(f"{result['quota']}: {fmt_metrics(result)}")
        return "<br>".join(parts)

    def fmt_decision(step: Mapping[str, Any]) -> str:
        decision = step.get("decision") or {}
        if decision.get("action") == "choose":
            return f"查 `{decision.get('quota')}`"
        if step.get("phase") == "final":
            return "停止（最终回复见下）"
        if decision.get("action") == "stop":
            return f"停止：{short_stop(decision.get('reason'), None)}"
        return short_stop(decision.get("reason"), None) or "—"

    def fmt_tool_return(step: Mapping[str, Any]) -> str:
        returned = step.get("tool_return") or {}
        if not returned:
            return "—"
        if returned.get("status") == "ok":
            return f"ok `{returned.get('quota')}`: {fmt_metrics(returned.get('metrics') or {})}"
        return f"error `{returned.get('code')}`"

    trace_rows = []
    phase_names = {
        "first": "首轮选择",
        "feedback": "看到反馈后决策",
        "final": "预算用尽后的最终回复",
    }
    for step in metrics.get("_llm_steps", []):
        trace_rows.append(
            "| {request} | {phase} | {visible} | {decision} | {call} | {returned} | {remaining} | {best} |".format(
                request=step.get("model_request_index"),
                phase=phase_names.get(step.get("phase"), step.get("phase")),
                visible=fmt_visible(step.get("visible") or {}),
                decision=fmt_decision(step).replace("|", "\\|"),
                call=(step.get("tool_call") or {}).get("id") or "—",
                returned=fmt_tool_return(step).replace("|", "\\|"),
                remaining=step.get("remaining_budget_after"),
                best=(step.get("best_after") or {}).get("quota") or "—",
            )
        )
    self_check_rows = "\n".join(
        "| {id} | {status} | {detail} |".format(
            id=item["id"],
            status="通过" if item["status"] == "pass" else "失败",
            detail=str(item["detail"]).replace("|", "\\|"),
        )
        for item in self_check["checks"]
    )
    source_files = "\n".join(
        f"- `{path}`（sha256 `{payload['sha256']}`，{payload['role']}）"
        for path, payload in sorted(run_meta["source_files_read"].items())
    )
    frozen_rows = "\n".join(
        "| {name} | `{path}` | `{sha}` |".format(
            name=name,
            path=item["path"],
            sha=item["sha256"],
        )
        for name, item in metrics["baselines"]["files"].items()
    )
    status_notes = {
        "verified": "verified：至少 1 次真实原生工具调用、同 ID 结果回传、合法停止和成功最终回复。",
        "incomplete": "incomplete：没有成功的工具调用，工作流未形成可验收的查询。",
        "partial": "partial：已有揭晓结果，但后续请求/最终回复失败或被拒绝；保留轨迹，不按完整结果判胜负。",
        "blocked": "blocked：来源、冻结基线、本地断言或 LLM 配置在真实请求前被阻塞。",
    }

    def fmt_saved(value: Any) -> str:
        if value is None:
            return "不适用"
        return str(value) if int(value) >= 0 else f"多 {-int(value)} 次"

    saved_line = (
        f"相对本地规则 **{fmt_saved(comparison['saved_reveals_vs_local_if_same_best'])}**，"
        f"相对固定策略 **{fmt_saved(comparison['saved_reveals_vs_fixed_if_same_best'])}**。"
        if comparison["status"] == "judged"
        else "真实 LLM 未完成，不计算查询节省；仅分别报告揭晓次数。"
    )
    return f"""# 2026-11-09 · Phase 13 真实 LLM Rec Agent 离线回放

> 本轮只读上一轮已验收的 5 组汇总 JSON 和两条冻结基线；真实 LLM 工作流只启动一次，最多 3 次模型请求。没有训练、重新召回或真实评测。除已配置的 LLM 服务外没有其他网络请求。

## 结论

- **真实 LLM 状态**：`{status}`。{status_notes.get(status, '')}
- **三方判定**：{outcome_text}
- **真实 LLM 最佳**：`{llm.get('best_quota')}`，命中 {llm.get('hits')}、Recall@100 {float(llm.get('recall_at_100') or 0):.6f}、Coverage {float(llm.get('coverage') or 0):.6f}；揭晓 {llm.get('reveals')} 次，模型请求 {llm.get('request_count')} 次。
- **本地规则 Agent（冻结）**：`{local.get('best_quota')}`，命中 {local.get('hits')}、Recall@100 {float(local.get('recall_at_100') or 0):.6f}、Coverage {float(local.get('coverage') or 0):.6f}；揭晓 {local.get('reveals')} 次。
- **固定策略（冻结）**：`{fixed.get('best_quota')}`，命中 {fixed.get('hits')}、Recall@100 {float(fixed.get('recall_at_100') or 0):.6f}、Coverage {float(fixed.get('coverage') or 0):.6f}；揭晓 {fixed.get('reveals')} 次。
- **边界**：只覆盖本次 `kuasar_small` 128 用户 / 362 目标的单次离线回放，不据此宣称正式测试集收益或策略泛化能力。

## 1. 三方汇总

| 方 | 状态 | 最佳配额 | 命中数 | Recall@100 | Coverage | 实际揭晓 | 首次达到最佳 | 停止原因 | LLM 请求 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 真实 LLM | `{status}` | `{llm.get('best_quota')}` | {llm.get('hits')} | {float(llm.get('recall_at_100') or 0):.6f} | {float(llm.get('coverage') or 0):.6f} | {llm.get('reveals')} / 2 | {llm.get('first_best_reveal')} | {short_stop(llm.get('stop_reason'), llm.get('stop_code'))} | {llm.get('request_count')} |
| 本地规则 Agent | `frozen_baseline` | `{local.get('best_quota')}` | {local.get('hits')} | {float(local.get('recall_at_100') or 0):.6f} | {float(local.get('coverage') or 0):.6f} | {local.get('reveals')} / 2 | {local.get('first_best_reveal')} | {cell(local.get('stop_reason'), 180)} | — |
| 固定策略 | `frozen_baseline` | `{fixed.get('best_quota')}` | {fixed.get('hits')} | {float(fixed.get('recall_at_100') or 0):.6f} | {float(fixed.get('coverage') or 0):.6f} | {fixed.get('reveals')} / 2 | {fixed.get('first_best_reveal')} | {cell(fixed.get('stop_reason'), 180)} | — |

- LLM 请求次数：**{llm.get('request_count')} / 3**；成功揭晓：**{llm.get('reveals')} / 2**；耗时：**{run_meta.get('elapsed_s')} s**。
- LLM 相对本地规则：命中 {comparison['llm_vs_local']['hits']:+d}、Recall@100 {comparison['llm_vs_local']['recall_at_100']:+.6f}、揭晓 {comparison['llm_vs_local']['reveals']:+d}。
- LLM 相对固定策略：命中 {comparison['llm_vs_fixed']['hits']:+d}、Recall@100 {comparison['llm_vs_fixed']['recall_at_100']:+.6f}、揭晓 {comparison['llm_vs_fixed']['reveals']:+d}。
- 相同最佳结果下的查询节省：{saved_line}

## 2. 真实 LLM 决策轨迹

| 请求 | 阶段 | 当时可见 | 模型决定 | 工具调用 ID | 工具返回 | 剩余预算 | 最佳 |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(trace_rows) if trace_rows else '| — | — | — | — | — | — | — | — |'}

停止路径：`{llm.get('stop_code')}`。模型最终回复（原文，最多 600 字符，未写入请求内容或用户/物品 ID）：

> {_safe_text(llm.get('final_reply') or llm.get('stop_reason'), 600).replace(chr(10), chr(10) + '> ')}


## 3. 冻结基线

两条基线在真实 LLM 工作流启动前从上一轮运行目录读取并核验；本轮不重跑本地规则 Agent 或固定策略。

| 基线 | 文件 | sha256 |
| --- | --- | --- |
{frozen_rows}

- 上一轮运行：`{metrics['baselines']['previous_run']}`
- 规则一致性：`{json.dumps(metrics['baselines']['rule_checks'], ensure_ascii=False)}`
- 来源缓存身份：`sha256:{sources['manifest']['recall_cache']['version']}`；冻结来源文件哈希与当前 8 个汇总 JSON 完全一致。

## 4. 来源与口径

- 数据集 / 划分：`{config['replay']['dataset']}` / `{config['replay']['split']}`；128 用户 / 362 目标；每用户 100 候选。
- 5 组 canonical 来源来自融合配额实验、LangGraph 工具调用实验、实验记忆实验；重复记录核对冲突数为 0。
- 所有来源共享同一缓存身份 `sha256:{sources['manifest']['recall_cache']['version']}`；指标定义 `{sources['manifest']['metric_definition']}`。
- 来源文件仅在 `run_meta.json` / `sources.json` 中记录路径与 sha256；未读取候选明细、缓存、数据集或实验记忆索引。

## 5. 本地替身检查（不联网）

| 断言 | 状态 | 说明 |
| --- | --- | --- |
{self_check_rows}

## 6. 执行范围与合规

| 项 | 实际 |
| --- | --- |
| 执行命令 | `{run_meta['command']}` |
| 模型 / 端点 | `{run_meta['model']}` / `{run_meta['endpoint_host']}` |
| 真实 LLM 工作流启动 | 1 次（本次调用） |
| 模型请求 | {run_meta['llm_request_count']} / 3（无自动重试） |
| 成功揭晓 | {run_meta['reveals']} / 2 |
| 训练 / 重新召回 / 真实评测 | 0 / 0 / 0 |
| 其他网络请求 | 0（仅已配置的 LLM 服务） |
| 读取的工件 | <br>{source_files.replace(chr(10), '<br>')} |
| 数据集 / 候选明细 / 缓存 / 实验记忆索引 | 未读取 / 未读取 / 未读取 / 未读取 |
| 来源与基线哈希 | 运行前后一致（见 `run_meta.json`） |
| 旧工件 | 未改写；只写入新的运行子目录 |

脱敏说明：未持久化模型请求 `messages`、API key、用户/物品 ID；`llm_trace.json` 与 `events.jsonl` 只保留可见聚合指标、模型请求序号、工具名/调用 ID、所选配额、剩余预算、停止原因和最终状态。

## 7. 验收对照

| 验收项 | 结论 | 证据 |
| --- | --- | --- |
| 同起点、同预算 | 通过 | 冻结基线核验通过；真实 LLM 首轮消息只有 50/50 历史、4 个候选和 2 次预算 |
| 按选择揭晓 | 通过 | 工具返回与当步配额、调用 ID 对应；`first_visibility` 与 `call_id_roundtrip` 断言通过 |
| 真实工作流可核验 | {'通过' if status == 'verified' else '未完成'} | 状态 `{status}`；LangGraph 节点 `{ ' → '.join(metrics.get('_graph_nodes', [])) }`；请求 {llm.get('request_count')} / 3、揭晓 {llm.get('reveals')} / 2 |
| 结果如实判定 | 通过 | 分开报告最佳指标与预算消耗；状态为 `{status}` 时按状态语义判定 |
| 执行范围可核验 | 通过 | 训练/召回/真实评测 0；只读汇总 JSON；来源和基线哈希前后一致 |
| 结论范围明确 | 通过 | 结论限定在本次回放，不扩展到测试集或其他场景 |

## 8. 状态语义

- `verified`：至少 1 次真实原生工具调用、同 `tool_call_id` 的合法结果回传、合法停止路径和成功最终回复。
- `incomplete`：首轮没有成功工具调用（无工具、非法/重复查询、请求失败等），不构成完整查询。
- `partial`：已经揭晓结果，但后续模型请求、工具查询或最终回复失败/被拒绝；保留轨迹，不按完整结果判胜负。
- `blocked`：来源、冻结基线、本地断言或 LLM 配置在真实请求前被阻塞。

## 9. 结论边界

本轮验收的是真实 LLM 在单次离线回放中能否按预算发起原生工具调用、读取工具结果并完成第二次选择或停止；不代表正式测试集收益，也不代表其他模型/其他数据规模下的泛化能力。完整 Phase 13 正式验证仍保留为待办。
"""


def _make_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = replay.utc_now().strftime("%Y%m%dT%H%M%SZ")
    candidate = output_root / f"run-{stamp}-{os.getpid()}"
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"run-{stamp}-{os.getpid()}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _command_string(config_path: Path, output_root: Path) -> str:
    parts = ["python", "-m", "recsys.experiments.offline_replay_llm"]
    if config_path.resolve() != DEFAULT_CONFIG_PATH.resolve():
        parts.extend(["--config", str(config_path)])
    if output_root.resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        parts.extend(["--output-root", str(output_root)])
    return " ".join(shlex.quote(part) for part in parts)


def _verify_hashes(files: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    after: dict[str, str] = {}
    changed: list[str] = []
    for path_text, payload in files.items():
        path = (PROJECT_ROOT / path_text).resolve()
        current = replay.sha256_bytes(path.read_bytes())
        after[path_text] = current
        if current != payload["sha256"]:
            changed.append(path_text)
    return {"before": {path: item["sha256"] for path, item in files.items()}, "after": after, "changed": changed}


def _load_llm_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    load_dotenv()
    cfg = llm_config()
    if cfg is None:
        missing = [name for name in LLM_ENV if not os.environ.get(name)]
        raise ReplayPreflightError(f"缺少真实 LLM 配置：{', '.join(missing)}")
    if int(config["llm"]["max_model_requests"]) != MAX_MODEL_REQUESTS:
        raise ReplayPreflightError("配置的 max_model_requests 必须为 3")
    if int(config["llm"]["max_tool_calls"]) != MAX_REVEALS:
        raise ReplayPreflightError("配置的 max_tool_calls 必须为 2")
    return cfg


def execute_real(
    config_path: Path,
    output_root: Path,
    run_dir: Path,
    started_wall: Any,
) -> dict[str, Any]:
    config = replay.load_config(config_path)
    sources = replay.load_sources(config)
    frozen = load_frozen_baselines(config, sources)
    self_check = run_self_check(config)
    if self_check["status"] != "pass":
        raise ReplayPreflightError(f"本地替身检查未通过：{self_check['failed']} 项失败")
    llm_cfg = _load_llm_settings(config)

    real_started = time.perf_counter()
    final = run_llm_replay(
        records=sources["records"],
        config=config,
        llm_cfg=llm_cfg,
        chat_fn=post_chat,
        started_at=real_started,
    )
    elapsed_s = round(time.perf_counter() - real_started, 3)
    finished_wall = replay.utc_now()

    source_hashes = _verify_hashes(sources["files_read"])
    frozen_hashes = _verify_hashes({item["path"]: item for item in frozen["files"].values()})
    if source_hashes["changed"]:
        raise ReplayPreflightError(f"来源文件在回放期间被改写：{source_hashes['changed']}")
    if frozen_hashes["changed"]:
        raise ReplayPreflightError(f"冻结基线在回放期间被改写：{frozen_hashes['changed']}")

    run_meta = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "command": _command_string(config_path, output_root),
        "argv": sys.argv[1:],
        "started_at": replay.iso_time(started_wall),
        "finished_at": replay.iso_time(finished_wall),
        "elapsed_s": elapsed_s,
        "python": sys.version.split()[0],
        "model": str(llm_cfg.get("model")),
        "endpoint_host": host_only(str(llm_cfg.get("base_url"))),
        "output_dir": str(run_dir.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_files_read": sources["files_read"],
        "source_hashes_unchanged": not source_hashes["changed"],
        "source_hashes_after": source_hashes["after"],
        "frozen_baselines": frozen["files"],
        "frozen_hashes_unchanged": not frozen_hashes["changed"],
        "frozen_hashes_after": frozen_hashes["after"],
        "llm_request_count": int(final.get("summary", {}).get("request_count") or 0),
        "reveals": int(final.get("summary", {}).get("reveals") or 0),
        "scope": {
            "training": 0,
            "recall": 0,
            "real_evaluation": 0,
            "llm_requests": int(final.get("summary", {}).get("request_count") or 0),
            "network_requests": int(final.get("summary", {}).get("request_count") or 0),
            "other_network_requests": 0,
            "dataset_files_read": 0,
            "candidate_detail_files_read": 0,
            "cache_files_read": 0,
            "memory_index_read": 0,
        },
        "write_scope": str(run_dir.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    metrics = build_metrics(
        config=config,
        sources=sources,
        frozen=frozen,
        self_check=self_check,
        final=final,
        run_meta=run_meta,
    )
    # 报告需要 LLM 轨迹；从 metrics 拆出大字段，避免 metrics.json 双重嵌套。
    metrics["_llm_steps"] = copy.deepcopy(final.get("trace_steps") or [])
    metrics["_graph_nodes"] = list(final.get("nodes") or [])
    llm_trace = build_llm_trace(final, config, run_meta)
    report = build_report(
        config=config,
        sources=sources,
        frozen=frozen,
        self_check=self_check,
        final=final,
        metrics=metrics,
        run_meta=run_meta,
    )
    metrics.pop("_llm_steps", None)
    metrics.pop("_graph_nodes", None)
    return {
        "config": config,
        "sources": sources,
        "frozen": frozen,
        "self_check": self_check,
        "final": final,
        "metrics": metrics,
        "llm_trace": llm_trace,
        "run_meta": run_meta,
        "report": report,
    }


def write_run(
    run_dir: Path,
    config_path: Path,
    payload: Mapping[str, Any],
) -> None:
    (run_dir / "config.toml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    replay.write_json(run_dir / "sources.json", payload["sources"]["manifest"])
    replay.write_json(run_dir / "baselines.json", {
        "previous_run": payload["frozen"]["previous_run"],
        "files": payload["frozen"]["files"],
        "local_agent": payload["frozen"]["local_agent"],
        "fixed": payload["frozen"]["fixed"],
        "rule_checks": payload["frozen"]["rule_checks"],
        "baseline": payload["frozen"]["baseline"],
        "candidates": payload["frozen"]["candidates"],
    })
    replay.write_json(run_dir / "llm_trace.json", payload["llm_trace"])
    replay.write_json(run_dir / "metrics.json", payload["metrics"])
    replay.write_json(run_dir / "self_check.json", payload["self_check"])
    replay.write_json(run_dir / "run_meta.json", payload["run_meta"])
    events = payload["final"].get("events") or []
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + ("\n" if events else ""),
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(payload["report"], encoding="utf-8")


def write_blocked(run_dir: Path, config_path: Path, error: Exception) -> None:
    if config_path.is_file():
        (run_dir / "config.toml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    replay.write_json(
        run_dir / "blocked.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "phase": PHASE,
            "reason": _safe_error(error),
            "created_at": replay.iso_time(replay.utc_now()),
        },
    )
    (run_dir / "report.md").write_text(
        f"""# Phase 13 真实 LLM Rec Agent 离线回放：本次已停止

- 状态：**blocked**
- 原因：{_safe_error(error)}
- 说明：来源、冻结基线、本地断言或 LLM 配置不满足时，按任务要求停止，未启动真实工作流。

修复后请重新运行：

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.offline_replay_llm
```

证据：[blocked.json](blocked.json)
""",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 13 真实 LLM 离线回放：真实 LLM / 本地规则 Agent / 固定策略三方对照"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--self-check", action="store_true", help="只跑本地替身检查，不联网")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_check:
        try:
            config = replay.load_config(args.config)
            result = run_self_check(config)
        except Exception as error:  # noqa: BLE001
            print(json.dumps({"status": "blocked", "reason": _safe_error(error)}, ensure_ascii=False, indent=2))
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "pass" else 1

    started_wall = replay.utc_now()
    run_dir = _make_run_dir(args.output_root)
    try:
        payload = execute_real(args.config, args.output_root, run_dir, started_wall)
    except Exception as error:  # noqa: BLE001 - 失败也要留下 blocked 证据
        write_blocked(run_dir, args.config, error)
        print(f"[blocked] {_safe_error(error)}")
        print(f"运行目录：{run_dir}")
        return 2
    write_run(run_dir, args.config, payload)
    status = payload["metrics"]["status"]
    print(f"回放完成：{run_dir}")
    print(
        f"LLM 状态 {status} / 最佳 {payload['metrics']['llm'].get('best_quota')} / "
        f"请求 {payload['metrics']['llm'].get('request_count')} / "
        f"揭晓 {payload['metrics']['llm'].get('reveals')}"
    )
    return 0 if status == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
