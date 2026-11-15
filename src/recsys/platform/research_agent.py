"""Phase 4 Research Agent：统一入口 + 总控调度。

职责（与 PROJECT_MASTER_v2 Phase 4 一致）：

* 接收用户的自然语言任务，用现有 LangGraph + LLM 客户端决策；
* 对“已有实验”的研究任务，先查历史：命中且校验通过就直接给结论，零新实验；
* 历史缺失/不匹配/工件失效时，把真实返回和剩余额度交给 LLM 选择下一步：
  检索知识库、调用推荐/搜索/广告业务、复用结果，或停止；
* 同一任务可以调用多个业务并汇总；每次分发、工具参数、返回、后续决策和停止原因
  都写入 ``trace.jsonl`` / ``decisions.json``；
* 总控与子 Agent 共用一个 :class:`~recsys.platform.budget.TaskBudget`；
* 无 LLM 时使用脚本替身决策（逐条标注 ``scripted_stand_in``），但工具仍执行真实业务代码。

统一 CLI::

    python -m recsys.platform --task "查一下已有 30/70 融合实验，能复用就复用"
    python -m recsys.platform --task "推荐和搜索一起做" --planner scripted
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, TypedDict

from langgraph.graph import END, StateGraph

from recsys.experiments.llm_tool_call import (
    LLM_ENV,
    llm_config,
    load_dotenv,
    post_chat,
)
from recsys.platform.budget import (
    BudgetExceeded,
    BudgetPolicy,
    TaskBudget,
)
from recsys.platform.rag import (
    KNOWLEDGE_TOOL_SCHEMA,
    knowledge_model_view,
    retrieve_knowledge,
)
from recsys.platform.shared_experiments import (
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_STORE_DIR,
    REPO_ROOT,
    SharedExperimentStore,
    build_record,
    display_path,
    history_model_view,
    lookup_history,
    model_registry_view,
    registry_model_view,
    resolve_path,
    reuse_reasons,
    utc_now,
)

BUSINESSES = ("recommendation", "search", "ads")
BUSINESS_LABELS = {"recommendation": "推荐", "search": "搜索", "ads": "广告"}
BUSINESS_TOOLS = {
    "recommendation": "run_recommendation_experiment",
    "search": "run_search_task",
    "ads": "run_ads_experiment",
}
TOOL_BUSINESS = {value: key for key, value in BUSINESS_TOOLS.items()}

DEFAULT_CONFIG_PATH = Path("configs/platform/research_agent.toml")

SYSTEM_PROMPT = """你是 Phase 4 的 Research Agent，同时承担 Master Agent 的总控调度。
你必须通过工具完成真实任务，规则如下：
1. 只能调用已注册工具，一次只调用一个；参数必须符合 schema，禁止并行 tool_calls。
2. 已有实验的研究任务：先看 history 结果。history.status=hit 且 reusable=true 时，
   复用对应业务的结果；其余业务继续处理。所有目标完成后再给结论。
3. history 未命中、不匹配或工件失效时，根据真实返回和剩余预算选择下一步：
   检索知识库、调用一个业务 Agent 做必要的小样本检查、或停止。不要编造实验结论。
4. 跨业务任务可以按顺序调用多个业务工具，并在最终回答里汇总各业务真实结果。
5. 预算由程序扣减：查历史、读结果、跑新实验分别计数。额度不足时程序会停止；
   不要请求 reset 预算，也不要重复相同工具+参数。
6. 工具返回 ok=false 时必须如实报告；不要把失败写成成功，也不要把模拟广告金额
   或 synthetic 转化标签当作真实业务效果。
7. 知识库检索只用于背景、实现和来源；历史实验能否复用由 history 工具判定。
8. 完成时停止调用工具，用简短中文给出结论（包含真实状态/指标/来源或失败原因）。
9. 研究/优化算法且提供 research_cases 时，调用 run_algorithm_research。业务 Agent 会写真实
   算法代码、执行实验并根据反馈继续修改。普通业务调用只运行已有算法，不能当作算法研究。
   研究历史用于形成假设，不能仅命中一条旧记录就宣布新研究任务完成。
"""


# ---------------------------------------------------------------------- 配置


@dataclass
class ResearchConfig:
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT
    store_dir: Path = DEFAULT_STORE_DIR
    history: dict[str, str] = field(
        default_factory=lambda: {
            "cache_dir": "artifacts/kuasar_small/mvp",
            "history_root": "experiments/2026-11-08-kuasar_small-fusion-quota-agent",
            "memory_dir": "experiments/2026-11-08-kuasar_small-experiment-memory/memory",
        }
    )
    budget: BudgetPolicy = field(default_factory=BudgetPolicy)
    subagents: dict[str, BudgetPolicy] = field(default_factory=dict)
    subagent_intent: dict[str, dict[str, Any]] = field(default_factory=dict)
    max_steps: int = 6
    rag_index_dir: str = "artifacts/rag/phase4"

    @classmethod
    def load(cls, path: Path | str | None = None) -> "ResearchConfig":
        config_path = resolve_path(path or DEFAULT_CONFIG_PATH)
        payload: dict[str, Any] = {}
        if config_path and config_path.exists():
            try:
                import tomllib

                payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
            except Exception as error:  # noqa: BLE001 - 配置错误时回退默认值并记录
                payload = {}
                payload["_error"] = str(error)
        paths = dict(payload.get("paths") or {})
        history = dict(payload.get("history") or {})
        budget = BudgetPolicy.from_mapping(payload.get("budget"))
        subagents: dict[str, BudgetPolicy] = {}
        for name, section in dict(payload.get("subagents") or {}).items():
            subagents[str(name)] = BudgetPolicy.from_mapping(section, base=budget)
        intent = {
            str(name): dict(section)
            for name, section in dict(payload.get("intent") or {}).items()
            if isinstance(section, Mapping)
        }
        return cls(
            experiment_root=Path(paths.get("experiment_root") or DEFAULT_EXPERIMENT_ROOT),
            store_dir=Path(paths.get("shared_store") or DEFAULT_STORE_DIR),
            history={
                "cache_dir": history.get("cache_dir")
                or "artifacts/kuasar_small/mvp",
                "history_root": history.get("history_root")
                or "experiments/2026-11-08-kuasar_small-fusion-quota-agent",
                "memory_dir": history.get("memory_dir")
                or "experiments/2026-11-08-kuasar_small-experiment-memory/memory",
            },
            budget=budget,
            subagents=subagents,
            subagent_intent=intent,
            max_steps=max(1, int(payload.get("max_steps", 6))),
            rag_index_dir=str(paths.get("rag_index_dir") or "artifacts/rag/phase4"),
        )

    def policy_for(self, business: str) -> BudgetPolicy:
        return self.subagents.get(business) or BudgetPolicy(
            max_llm_requests=1,
            max_tool_calls=4,
            max_experiments=1,
            max_history_lookups=0,
            max_artifact_reads=0,
            max_seconds=min(120.0, self.budget.max_seconds),
        )

    def intent_for(self, business: str) -> dict[str, Any]:
        return dict(self.subagent_intent.get(business) or {})


# ---------------------------------------------------------------------- 工具定义


@dataclass(frozen=True)
class ResearchTool:
    name: str
    description: str
    parameters: dict[str, Any]
    kind: str  # history | artifact | registry | rag | business

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _object(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _tools() -> dict[str, ResearchTool]:
    from recsys.ads import AD_MODEL_NAMES

    tools = [
        ResearchTool(
            name="run_algorithm_research",
            description="派发给 Rec/Search/Ads Agent：自主编写算法、执行验证实验、读取反馈后继续修改；使用同一预算和实验记忆，不训练。",
            parameters=_object({
                "business": {"type": "string", "enum": list(BUSINESSES)},
                "rounds": {"type": "integer", "minimum": 1},
            }, ["business"]),
            kind="research",
        ),
        ResearchTool(
            name="search_experiment_history",
            description=(
                "查询推荐/搜索/广告的历史实验。推荐历史会校验数据集、切分、缓存指纹、"
                "配额、工件和规则版本；共享库会校验状态、配置、切分、模型来源和工件。"
                "命中且 reusable=true 才允许直接复用；未命中会返回具体原因。"
                "可用 category=quota（数组 [双塔, MIND]）查询已有融合配额实验。"
            ),
            parameters=_object(
                {
                    "business": {
                        "type": "string",
                        "enum": ["any", "recommendation", "search", "ads"],
                        "default": "any",
                    },
                    "quota": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 100},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "task_id": {"type": "string"},
                    "dataset": {"type": "string"},
                    "split": {"type": "string"},
                    "model": {"type": "string"},
                    "session_id": {"type": "string"},
                    "tokens": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
                    "mode": {"type": "string", "enum": ["bm25", "vector", "hybrid", "rerank"]},
                }
            ),
            kind="history",
        ),
        ResearchTool(
            name="read_experiment_result",
            description=(
                "按 record_id 读取共享实验库中已有记录的指定字段（summary/metrics/result/"
                "config/artifacts），用于复用或解释已有结果，不会重跑实验。"
            ),
            parameters=_object(
                {
                    "record_id": {"type": "string"},
                    "field": {
                        "type": "string",
                        "enum": ["summary", "metrics", "result", "config", "artifacts"],
                        "default": "summary",
                    },
                },
                ["record_id"],
            ),
            kind="artifact",
        ),
        ResearchTool(
            name="query_model_registry",
            description=(
                "查看推荐、搜索、广告三个业务已有的模型注册和版本信息：模型 id、类别、"
                "完整度、权重/checkpoint 是否存在、搜索索引编码器与广告权重状态。"
            ),
            parameters=_object(
                {
                    "business": {
                        "type": "string",
                        "enum": ["any", "recommendation", "search", "ads"],
                        "default": "any",
                    }
                }
            ),
            kind="registry",
        ),
        ResearchTool(
            name="retrieve_knowledge",
            description=KNOWLEDGE_TOOL_SCHEMA["function"]["description"],
            parameters=KNOWLEDGE_TOOL_SCHEMA["function"]["parameters"],
            kind="rag",
        ),
        ResearchTool(
            name=BUSINESS_TOOLS["recommendation"],
            description=(
                "调用 Rec Agent 跑一次真实的小样本推荐链路（召回 → 融合 → LightGBM 排序 → "
                "重排 → 评估），复用已有 checkpoint，不训练、不覆盖默认权重。"
                "只在历史未命中或需要补做检查时调用。"
            ),
            parameters=_object(
                {
                    "users": {"type": "integer", "minimum": 1, "maximum": 64},
                    "split": {"type": "string", "enum": ["validation", "test"]},
                    "recall_models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 4,
                    },
                    "ranker": {"type": "string"},
                    "k_recall": {"type": "integer", "minimum": 5, "maximum": 200},
                    "k_final": {"type": "integer", "minimum": 5, "maximum": 50},
                }
            ),
            kind="business",
        ),
        ResearchTool(
            name=BUSINESS_TOOLS["search"],
            description=(
                "调用 Search Agent 做一次真实检索（BM25 / FAISS 向量 / RRF 混合 / 精排），"
                "读现有 KuaiSAR 小样本索引，不重建索引、不扫原始 CSV。"
                "可用 session_id 指定已有查询，或直接给 query_tokens。"
            ),
            parameters=_object(
                {
                    "mode": {
                        "type": "string",
                        "enum": ["bm25", "vector", "hybrid", "rerank"],
                        "default": "bm25",
                    },
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
                    "session_id": {"type": "string"},
                    "tokens": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "minItems": 1,
                        "maxItems": 64,
                    },
                    "include_knowledge": {"type": "boolean"},
                }
            ),
            kind="business",
        ),
        ResearchTool(
            name=BUSINESS_TOOLS["ads"],
            description=(
                "调用 Ads Agent 做广告 CTR 推理、请求内排序和竞价/预算模拟。"
                "广告权重是 interface_test（固定初始化）时必须标注；竞价金额是模拟值，"
                "不能表述为真实出价或预算。"
            ),
            parameters=_object(
                {
                    "model": {"type": "string", "enum": list(AD_MODEL_NAMES)},
                    "split": {"type": "string", "enum": ["train", "validation", "test"]},
                    "requests": {"type": "integer", "minimum": 1, "maximum": 10},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 2000},
                    "simulate_auction": {"type": "boolean"},
                }
            ),
            kind="business",
        ),
    ]
    return {tool.name: tool for tool in tools}


TOOLS = _tools()


# ---------------------------------------------------------------------- 校验


def _validate_arguments(spec: ResearchTool, arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ValueError("工具参数必须是 JSON 对象")
    schema = spec.parameters
    properties = schema.get("properties") or {}
    extra = sorted(set(arguments) - set(properties))
    if extra and not schema.get("additionalProperties", False):
        raise ValueError(f"{spec.name} 不接受参数 {extra}")
    missing = [key for key in schema.get("required") or [] if key not in arguments]
    if missing:
        raise ValueError(f"{spec.name} 缺少参数 {missing}")
    for key, value in arguments.items():
        definition = properties.get(key) or {}
        expected = definition.get("type")
        if expected == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{spec.name}.{key} 必须是整数")
        elif expected == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{spec.name}.{key} 必须是数值")
        elif expected == "string":
            if not isinstance(value, str):
                raise ValueError(f"{spec.name}.{key} 必须是字符串")
        elif expected == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"{spec.name}.{key} 必须是布尔值")
        elif expected == "array":
            if not isinstance(value, list):
                raise ValueError(f"{spec.name}.{key} 必须是数组")
            item_type = (definition.get("items") or {}).get("type")
            for item in value:
                if item_type == "integer" and (isinstance(item, bool) or not isinstance(item, int)):
                    raise ValueError(f"{spec.name}.{key} 只能包含整数")
        if "enum" in definition and value not in definition["enum"]:
            raise ValueError(f"{spec.name}.{key}={value!r} 不在允许集合")
        if "minimum" in definition and value < definition["minimum"]:
            raise ValueError(f"{spec.name}.{key} 小于最小值 {definition['minimum']}")
        if "maximum" in definition and value > definition["maximum"]:
            raise ValueError(f"{spec.name}.{key} 大于最大值 {definition['maximum']}")
        if expected == "array":
            if len(value) < int(definition.get("minItems", 0)):
                raise ValueError(f"{spec.name}.{key} 元素太少")
            if "maxItems" in definition and len(value) > int(definition["maxItems"]):
                raise ValueError(f"{spec.name}.{key} 元素太多")
    result = dict(arguments)
    if spec.name == "search_experiment_history" and result.get("quota"):
        two, mind = int(result["quota"][0]), int(result["quota"][1])
        if two + mind != 100:
            raise ValueError("quota 两项之和必须为 100")
        from recsys.experiments.fusion_quota import ALLOWED_QUOTAS

        if (two, mind) not in ALLOWED_QUOTAS:
            raise ValueError(f"quota {(two, mind)} 不在允许集合 {ALLOWED_QUOTAS}")
        result["quota"] = [two, mind]
    return result


# ---------------------------------------------------------------------- 工具函数


def _quota_from_text(text: str) -> tuple[int, int] | None:
    for match in re.finditer(r"(\d{1,3})\s*[/／:：]\s*(\d{1,3})", text):
        two, mind = int(match.group(1)), int(match.group(2))
        if two + mind == 100:
            return two, mind
    return None


def _is_research_task(task: str, intent: Mapping[str, Any]) -> bool:
    if intent.get("research_cases"):
        return False  # The business researcher reads history as evidence, not an early success shortcut.
    if intent.get("history_first") is not None:
        return bool(intent.get("history_first"))
    if _quota_from_text(task):
        return True
    keywords = ("历史", "已有", "复用", "实验记录", "命中", "配额", "quota", "上次", "之前的实验")
    return any(word in task for word in keywords)


def infer_businesses(task: str, intent: Mapping[str, Any] | None = None) -> list[str]:
    intent = dict(intent or {})
    forced = intent.get("businesses") or intent.get("business")
    if forced:
        if isinstance(forced, str):
            forced = [forced]
        return [str(item) for item in forced if str(item) in BUSINESSES] or ["recommendation"]
    lowered = task.lower()
    words = {
        "recommendation": ("推荐", "双塔", "mind"),
        "search": ("搜索", "检索", "bm25", "向量检索"),
        "ads": ("广告", "ctr", "竞价", "cvr", "投放"),
    }
    positions: list[tuple[int, str]] = []
    for business, tokens in words.items():
        hits = [lowered.find(token) for token in tokens if token in lowered]
        if hits:
            positions.append((min(hits), business))
    if not positions:
        return ["search"] if _search_query_filters(task, intent) else ["recommendation"]
    return [business for _, business in sorted(positions)]


def _history_business(task: str, intent: Mapping[str, Any]) -> str:
    if intent.get("quota") or _quota_from_text(task):
        return "recommendation"
    businesses = infer_businesses(task, intent)
    return businesses[0] if businesses else "any"


def _search_query_filters(task: str, intent: Mapping[str, Any]) -> dict[str, Any]:
    filters = {key: intent[key] for key in ("session_id", "tokens", "mode") if intent.get(key)}
    session = re.search(r"(?:会话|\bsession(?:[_ ]id)?)\s*[:：=#]?\s*(\d+)\b", task, re.IGNORECASE)
    if session and "session_id" not in filters:
        filters["session_id"] = session[1]
    if "mode" not in filters:
        for mode, pattern in (
            ("rerank", r"精排|\brerank\b"),
            ("hybrid", r"混合检索|\bhybrid\b"),
            ("vector", r"向量检索|\bfaiss\b"),
            ("bm25", r"\bbm25\b"),
        ):
            if re.search(pattern, task, re.IGNORECASE):
                filters["mode"] = mode
                break
    return filters


def _summarize_result(result: Mapping[str, Any], limit: int = 1200) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "ok",
        "status",
        "business",
        "agent_status",
        "record_id",
        "hit",
        "reusable",
        "source",
        "reason",
        "error",
        "warnings",
    ):
        if key in result:
            summary[key] = result.get(key)
    if result.get("metrics") is not None:
        text = json.dumps(result.get("metrics"), ensure_ascii=False, default=str)
        summary["metrics"] = result.get("metrics") if len(text) <= limit else text[:limit] + "…"
    if isinstance(result.get("hits"), list):
        summary["hits"] = [
            {"citation": hit.get("citation"), "score": hit.get("score")}
            for hit in result["hits"][:5]
        ]
    return summary


def _tool_model_view(name: str, result: Mapping[str, Any]) -> dict[str, Any]:
    if name == "search_experiment_history":
        return history_model_view(result)
    if name == "retrieve_knowledge":
        return knowledge_model_view(result)
    if name == "query_model_registry":
        return registry_model_view(result)
    if name == "read_experiment_result":
        return result
    return _summarize_result(result)


# ---------------------------------------------------------------------- 状态


class AgentState(TypedDict, total=False):
    status: str
    stage: str
    nodes: list[str]
    task: str
    task_id: str
    config: Any
    budget: Any
    store: Any
    task_dir: Any
    intent: dict
    targets: list[str]
    history: dict | None
    tool_results: list[dict]
    decisions: list[dict]
    messages: list[dict]
    pending: dict | None
    pending_call_id: str | None
    stop_reason: str
    stop_detail: str
    final_answer: str
    llm_available: bool
    llm_blocker: str | None
    llm_used: bool
    llm_requests: int
    planner_requested: str
    planner_effective: str
    scripted_stand_in: bool
    llm_config: dict | None
    started: float
    step: int
    budget_refusals: list[dict]
    summary: dict


def _event(state: AgentState, payload: dict[str, Any]) -> None:
    task_dir = state.get("task_dir")
    if task_dir is not None:
        path = Path(task_dir) / "trace.jsonl"
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"at": utc_now(), **payload}, ensure_ascii=False) + "\n")


def _enter(state: AgentState, node: str) -> AgentState:
    state = dict(state)  # type: ignore[assignment]
    state.setdefault("nodes", []).append(node)
    state["stage"] = node
    _event(state, {"event": "node_start", "node": node})
    return state  # type: ignore[return-value]


def _leave(state: AgentState, node: str) -> AgentState:
    _event(state, {"event": "node_end", "node": node, "status": state.get("status")})
    return state


def _budget(state: AgentState) -> TaskBudget:
    return state["budget"]  # type: ignore[return-value]


def _stop(state: AgentState, reason: str, detail: str, *, status: str | None = None) -> None:
    state["stop_reason"] = reason
    state["stop_detail"] = detail
    if status:
        state["status"] = status
    _event(state, {"event": "stop", "reason": reason, "detail": detail, "status": state.get("status")})


# ---------------------------------------------------------------------- 决策


def _apply_decision(
    state: AgentState,
    decision: Mapping[str, Any],
    *,
    source: str,
    visible: Mapping[str, Any],
) -> None:
    step = int(state.get("step", 0)) + 1
    state["step"] = step
    action = str(decision.get("action") or "stop")
    reason = str(decision.get("reason") or "")
    budget = _budget(state)
    record = {
        "step": step,
        "source": source,
        "action": action,
        "reason": reason,
        "visible": dict(visible),
        "budget_before": budget.snapshot(),
    }
    if action == "call":
        record["tool"] = decision.get("tool")
        record["arguments"] = decision.get("arguments") or {}
        state["pending"] = {
            "tool": decision.get("tool"),
            "arguments": decision.get("arguments") or {},
            "reason": reason,
            "source": source,
        }
        _event(state, {"event": "decision", **record})
    else:
        record["stop_reason"] = decision.get("stop_reason") or "llm_stop"
        record["detail"] = decision.get("detail") or reason
        state["decisions"].append(record)
        _event(state, {"event": "decision", **record})
        _stop(
            state,
            str(record["stop_reason"]),
            str(record["detail"]),
        )
    if action == "call":
        state["decisions"].append(record)


def _visible_state(state: AgentState) -> dict[str, Any]:
    budget = _budget(state)
    return {
        "history": history_model_view(state["history"]) if state.get("history") else None,
        "tool_results": [
            {
                "business": item.get("business"),
                "tool": item.get("tool"),
                "ok": item.get("ok"),
                "status": item.get("status"),
                "record_id": item.get("record_id"),
                "error": item.get("error"),
            }
            for item in state.get("tool_results") or []
        ],
        "budget": budget.snapshot(),
        "targets": list(state.get("targets") or []),
    }


def _completed_businesses(state: AgentState) -> set[str]:
    return {
        str(item["business"])
        for item in state.get("tool_results") or []
        if item.get("business") and item.get("ok") and (
            item["tool"] in TOOL_BUSINESS or item["tool"] == "run_algorithm_research"
            or (item["tool"] == "search_experiment_history" and item["status"] == "hit"
                and item["result_summary"].get("reusable"))
        )
    }


def scripted_decision(state: AgentState) -> dict[str, Any]:
    """离线脚本替身：读取真实 history / tool 返回值与剩余额度后决定下一步。

    只用于离线验收；每个决策都会标注 ``source=scripted_stand_in``。
    """

    history = state.get("history")
    results = list(state.get("tool_results") or [])
    budget = _budget(state)
    read_record = (state.get("intent") or {}).get("read_record")
    if read_record:
        already = any(item.get("tool") == "read_experiment_result" for item in results)
        if already:
            return {
                "action": "stop",
                "stop_reason": "scripted_stop",
                "reason": f"已读取记录 {read_record}，停止",
            }
        if budget.remaining("tool") < 1 or budget.remaining("artifact") < 1:
            return {
                "action": "stop",
                "stop_reason": "budget_exhausted",
                "detail": "读取已有结果需要 tool+artifact 各 1 次额度，当前不足",
                "reason": "预算不足，拒绝读取",
            }
        return {
            "action": "call",
            "tool": "read_experiment_result",
            "arguments": {"record_id": str(read_record), "field": "summary"},
            "reason": f"任务要求读取已有记录 {read_record}；不重跑实验",
        }
    done = _completed_businesses(state)
    pending = [business for business in state.get("targets") or [] if business not in done]
    if not pending:
        if not results:
            return {
                "action": "stop",
                "stop_reason": "no_action",
                "reason": "没有可执行的业务工具",
            }
        return {
            "action": "stop",
            "stop_reason": "scripted_stop",
            "reason": f"已完成任务要求的业务：{sorted(done)}",
        }
    business = pending[0]
    remaining_experiments = budget.remaining("experiment")
    remaining_tools = budget.remaining("tool")
    history_note = (
        f"历史 {history.get('status')}（{history.get('reason')}）" if history else "非历史任务"
    )
    if remaining_experiments < 1 or remaining_tools < 2:
        state.setdefault("budget_refusals", []).append(
            {
                "would_call": BUSINESS_TOOLS[business],
                "business": business,
                "remaining_experiments": int(remaining_experiments),
                "remaining_tool_calls": int(remaining_tools),
                "reason": "全局预算不足，子 Agent 不能重置额度",
            }
        )
        _event(
            state,
            {
                "event": "budget_refusal",
                "would_call": BUSINESS_TOOLS[business],
                "remaining_experiments": int(remaining_experiments),
                "remaining_tool_calls": int(remaining_tools),
            },
        )
        return {
            "action": "stop",
            "stop_reason": "budget_exhausted",
            "detail": (
                f"想调用 {business}，但剩余 experiments={int(remaining_experiments)}、"
                f"tool_calls={int(remaining_tools)}，不足以启动子任务"
            ),
            "reason": "预算不足，停止，不绕过额度",
        }
    return {
        "action": "call",
        "tool": BUSINESS_TOOLS[business],
        "arguments": {},
        "reason": (
            f"{history_note}；剩余 experiments={int(remaining_experiments)}、"
            f"tool_calls={int(remaining_tools)}；任务要求 {BUSINESS_LABELS[business]}，"
            f"调用一次 {BUSINESS_LABELS[business]}小样本检查"
        ),
    }


def _extract_tool_calls(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    calls = message.get("tool_calls") or []
    extracted: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        function = call.get("function") or {}
        if (call.get("type") or "function") != "function":
            raise ValueError(f"不支持的 tool_call 类型 {call.get('type')}")
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError as error:
            raise ValueError(f"tool_call 参数不是合法 JSON：{error}") from error
        extracted.append(
            {
                "id": call.get("id") or f"call_{int(time.time() * 1000)}_{index}",
                "name": function.get("name") or "",
                "arguments": arguments,
            }
        )
    return extracted


def _assistant_tool_message(call: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                },
            }
        ],
    }


def _tool_message(call_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(payload, ensure_ascii=False)[:8000],
    }


def _base_messages(state: AgentState) -> list[dict[str, Any]]:
    config: ResearchConfig = state["config"]
    budget = _budget(state)
    registry = registry_model_view(model_registry_view())
    history = history_model_view(state["history"]) if state.get("history") else None
    context = {
        "task": state.get("task"),
        "dataset": (state.get("intent") or {}).get("dataset", "kuasar_small"),
        "targets": state.get("targets"),
        "history_first": history,
        "budget": budget.snapshot(),
        "subagent_policy": {
            name: config.policy_for(name).as_dict() for name in BUSINESSES
        },
        "model_registry": registry,
        "available_tools": list(TOOLS),
        "research_cases": (state.get("intent") or {}).get("research_cases"),
        "instruction": (
            "复用 history_first 中已校验的对应业务结果；根据真实返回值与剩余预算"
            "完成其他目标。"
        ),
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(state.get("task") or "")},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]


# ---------------------------------------------------------------------- 工具执行


def _execute_tool(
    state: AgentState,
    name: str,
    arguments: Mapping[str, Any],
    *,
    call_id: str | None,
    phase: str,
    source: str,
) -> dict[str, Any] | None:
    """执行工具并扣减共享预算；预算不足时设置停止原因并返回 None。"""

    spec = TOOLS[name]
    budget = _budget(state)
    config: ResearchConfig = state["config"]
    store: SharedExperimentStore = state["store"]
    started = time.perf_counter()
    _event(
        state,
        {
            "event": "tool_start",
            "tool": name,
            "arguments": dict(arguments),
            "budget": budget.snapshot(),
        },
    )
    try:
        budget.check("tool")
        if spec.kind == "history":
            budget.check("history")
        elif spec.kind == "artifact":
            budget.check("artifact")
        elif spec.kind in {"business", "research"}:
            budget.check("experiment")
            if budget.remaining("tool") < 2:
                raise BudgetExceeded(
                    "tool",
                    f"业务子任务至少需要 1 次分发 + 1 次内部工具，当前只剩 "
                    f"{int(budget.remaining('tool'))} 次",
                    scope=str(TOOL_BUSINESS.get(name, "business")),
                    snapshot=budget.snapshot(),
                )
        elif spec.kind not in {"registry", "rag"}:
            raise ValueError(f"未知工具类型：{spec.kind}")
    except BudgetExceeded as error:
        state.setdefault("budget_refusals", []).append(error.as_dict())
        _event(state, {"event": "budget_refusal", **error.as_dict()})
        _stop(state, "budget_exhausted", error.reason)
        return None

    try:
        if spec.kind == "research":
            import importlib
            from recsys.platform.autoresearch import BUSINESS_MODULES
            business = arguments["business"]
            case_path = (state.get("intent") or {}).get("research_cases", {}).get(business)
            if not case_path:
                raise ValueError(f"没有为 {business} 提供真实研究数据，请设置 research_cases")
            budget.spend("tool")
            scope = budget.child(business, config.policy_for(business))
            result = importlib.import_module(BUSINESS_MODULES[business]).run_research(
                str(state.get("task") or ""), case_path=case_path,
                work_dir=Path(state["task_dir"]) / business, scope=scope, store=store,
                rounds=int(arguments.get("rounds", 2)),
            )
        elif spec.kind == "business":
            business = TOOL_BUSINESS[name]
            scope = budget.child(
                business,
                config.policy_for(business),
            )
            intent = {
                **config.intent_for(business),
                **dict(arguments),
                "step": int(state.get("step", 0)),
                "rag_index_dir": config.rag_index_dir,
            }
            if business == "search":
                intent.update(_search_query_filters(str(state.get("task") or ""), state.get("intent") or {}))
            from recsys.platform import adapters

            result = adapters.run_business(
                business,
                str(state.get("task") or ""),
                task_id=str(state["task_id"]),
                scope=scope,
                experiment_dir=Path(state["task_dir"]) / f"business_{business}",
                intent=intent,
                store=store,
            )
            result.setdefault("business", business)
        elif spec.kind == "history":
            budget.spend("tool")
            budget.spend("history")
            if arguments.get("business", "any") in {"search", "any"} and not arguments.get("quota"):
                arguments = {
                    **arguments,
                    **_search_query_filters(str(state.get("task") or ""), state.get("intent") or {}),
                }
            result = lookup_history(
                business=str(arguments.get("business") or "any"),
                quota=arguments.get("quota"),
                task_id=arguments.get("task_id"),
                dataset=arguments.get("dataset"),
                split=arguments.get("split"),
                model=arguments.get("model"),
                store=store,
                legacy=config.history,
                request={
                    "dataset": arguments.get("dataset") or "kuasar_small",
                    "split": arguments.get("split"),
                    "model": arguments.get("model"),
                    **{key: arguments[key] for key in ("session_id", "tokens", "mode") if key in arguments},
                },
            )
            state["history"] = result
        elif spec.kind == "artifact":
            budget.spend("tool")
            budget.spend("artifact")
            record = store.get(str(arguments.get("record_id")))
            if record is None:
                result = {
                    "ok": False,
                    "error": f"找不到记录 {arguments.get('record_id')}",
                }
            else:
                field = str(arguments.get("field") or "summary")
                value = (
                    {
                        "record_id": record.get("record_id"),
                        "task_id": record.get("task_id"),
                        "business": record.get("business"),
                        "status": record.get("status"),
                        "metrics": (record.get("result") or {}).get("metrics"),
                        "summary": (record.get("result") or {}).get("summary"),
                    }
                    if field == "summary"
                    else record.get(field)
                    if field != "metrics"
                    else (record.get("result") or {}).get("metrics")
                )
                result = {
                    "ok": True,
                    "record_id": record.get("record_id"),
                    "field": field,
                    "value": value,
                    "reuse_reasons": reuse_reasons(record, {}),
                }
        elif spec.kind == "registry":
            budget.spend("tool")
            result = model_registry_view(str(arguments.get("business") or "any"))
        elif spec.kind == "rag":
            budget.spend("tool")
            result = retrieve_knowledge(
                str(arguments.get("query") or ""),
                top_k=int(arguments.get("top_k", 3)),
                index_dir=config.rag_index_dir,
            )
        else:  # pragma: no cover
            raise ValueError(f"未知工具类型：{spec.kind}")
    except BudgetExceeded as error:
        state.setdefault("budget_refusals", []).append(error.as_dict())
        _event(state, {"event": "budget_refusal", **error.as_dict()})
        _stop(state, "budget_exhausted", error.reason)
        return None
    except Exception as error:  # noqa: BLE001 - 工具失败必须被总控看到
        result = {
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
            "business": TOOL_BUSINESS.get(name),
        }

    elapsed = round(time.perf_counter() - started, 2)
    if name == "search_experiment_history":
        # 历史未命中是合法结果，不是工具失败。
        ok = bool(result.get("ok", True))
    else:
        ok = bool(result.get("ok"))
    business = TOOL_BUSINESS.get(name) or result.get("business")
    result_record = {
        "step": int(state.get("step", 0)),
        "phase": phase,
        "source": source,
        "tool": name,
        "business": business,
        "arguments": dict(arguments),
        "ok": ok,
        "status": result.get("status") or ("hit" if result.get("hit") else None),
        "record_id": result.get("record_id"),
        "error": result.get("error"),
        "elapsed_s": elapsed,
        "result_summary": _summarize_result(result),
    }
    state.setdefault("tool_results", []).append(result_record)
    if state.get("llm_used") and call_id:
        state["messages"].append(
            _tool_message(call_id, {"result": _tool_model_view(name, result), "budget": budget.snapshot()})
        )
    _event(
        state,
        {
            "event": "tool_end",
            "tool": name,
            "ok": ok,
            "business": business,
            "record_id": result.get("record_id"),
            "elapsed_s": elapsed,
            "result_summary": _summarize_result(result),
            "budget": budget.snapshot(),
        },
    )
    return result


# ---------------------------------------------------------------------- LangGraph 节点


def node_history(state: AgentState) -> AgentState:
    state = _enter(state, "history")
    if state.get("status") != "running":
        return _leave(state, "history")
    intent = state.get("intent") or {}
    if not _is_research_task(str(state.get("task") or ""), intent):
        _event(state, {"event": "history_skipped", "reason": "not_a_research_task"})
        return _leave(state, "history")
    task = str(state.get("task") or "")
    quota = intent.get("quota") or _quota_from_text(task)
    business = _history_business(task, {**intent, "quota": quota})
    arguments: dict[str, Any] = {
        "business": business,
        **{key: intent[key] for key in ("dataset", "split", "model") if intent.get(key)},
    }
    if business == "search":
        arguments.update(_search_query_filters(task, intent))
    if "split" not in arguments and re.search(r"\btest\b|测试集", task, re.IGNORECASE):
        arguments["split"] = "test"
    if quota:
        arguments["quota"] = [int(quota[0]), int(quota[1])]
    if intent.get("task_id_filter"):
        arguments["task_id"] = str(intent["task_id_filter"])
    _apply_decision(
        state,
        {
            "action": "call",
            "tool": "search_experiment_history",
            "arguments": arguments,
            "reason": "研究任务先查历史；命中且可复用则不再跑新实验",
        },
        source="policy:history_first",
        visible=_visible_state(state),
    )
    return _leave(state, "history")


def node_plan(state: AgentState) -> AgentState:
    state = _enter(state, "plan")
    if state.get("status") != "running":
        return _leave(state, "plan")
    if state.get("pending"):
        return _leave(state, "plan")
    if state.get("stop_reason"):
        return _leave(state, "plan")
    if int(state.get("step", 0)) >= int(state["config"].max_steps):
        _stop(state, "max_steps", f"决策步数达到上限 {state['config'].max_steps}", status="partial")
        return _leave(state, "plan")
    budget = _budget(state)
    try:
        budget.check("seconds")
    except BudgetExceeded as error:
        _stop(state, "deadline", error.reason, status="partial")
        return _leave(state, "plan")

    visible = _visible_state(state)
    if state.get("planner_effective") == "llm":
        messages = list(state.get("messages") or [])
        if not messages:
            messages = _base_messages(state)
            state["messages"] = messages
        if not state.get("llm_used"):
            state["llm_used"] = True
        try:
            budget.check("llm")
            budget.spend("llm")
            state["llm_requests"] = int(state.get("llm_requests", 0)) + 1
        except BudgetExceeded as error:
            state.setdefault("budget_refusals", []).append(error.as_dict())
            _event(state, {"event": "budget_refusal", **error.as_dict()})
            _stop(state, "budget_exhausted", error.reason)
            return _leave(state, "plan")
        try:
            response = post_chat(
                state.get("llm_config") or {},
                messages,
                tools=[tool.schema() for tool in TOOLS.values()],
                timeout=float(max(1.0, min(60.0, budget.remaining("seconds")))),
            )
        except Exception as error:  # noqa: BLE001
            _stop(
                state,
                "llm_request_failed",
                f"{type(error).__name__}: {error}",
                status="partial" if state.get("tool_results") else "failed",
            )
            return _leave(state, "plan")
        try:
            calls = _extract_tool_calls(response)
        except ValueError as error:
            _stop(state, "invalid_tool_call", str(error), status="failed")
            return _leave(state, "plan")
        content = str(((response.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        if not calls:
            state["final_answer"] = content or "LLM 选择停止（无工具调用）。"
            no_tools_yet = not state.get("tool_results")
            _apply_decision(
                state,
                {
                    "action": "stop",
                    "stop_reason": "no_tool_call" if no_tools_yet else "llm_stop",
                    "detail": (
                        "LLM 在应执行工具的任务中零调用；按实际状态标记失败"
                        if no_tools_yet
                        else (content or "LLM 选择停止")
                    ),
                    "reason": content or "LLM 选择停止",
                },
                source="llm",
                visible=visible,
            )
            state["messages"].append({"role": "assistant", "content": content})
            return _leave(state, "plan")
        call = calls[0]
        try:
            arguments = _validate_arguments(TOOLS[call["name"]], call["arguments"])
        except (KeyError, ValueError) as error:
            _stop(state, "invalid_tool_call", f"{call.get('name')}: {error}", status="failed")
            return _leave(state, "plan")
        state["pending_call_id"] = call["id"]
        state["messages"].append(_assistant_tool_message(call))
        _apply_decision(
            state,
            {
                "action": "call",
                "tool": call["name"],
                "arguments": arguments,
                "reason": f"LLM 选择工具 {call['name']}",
            },
            source="llm",
            visible=visible,
        )
        if len(calls) > 1:
            _event(state, {"event": "extra_tool_calls_ignored", "count": len(calls) - 1})
        return _leave(state, "plan")

    decision = scripted_decision(state)
    try:
        if decision.get("action") == "call":
            decision = dict(decision)
            decision["arguments"] = _validate_arguments(
                TOOLS[str(decision["tool"])], decision.get("arguments") or {}
            )
    except (KeyError, ValueError) as error:
        _stop(state, "invalid_tool_call", str(error), status="failed")
        return _leave(state, "plan")
    _apply_decision(state, decision, source="scripted_stand_in", visible=visible)
    return _leave(state, "plan")


def node_execute(state: AgentState) -> AgentState:
    state = _enter(state, "execute")
    pending = state.get("pending")
    if not pending:
        return _leave(state, "execute")
    name = str(pending["tool"])
    signature = json.dumps(
        {"tool": name, "arguments": pending.get("arguments") or {}},
        sort_keys=True,
        ensure_ascii=False,
    )
    seen = {
        json.dumps({"tool": item["tool"], "arguments": item["arguments"]}, sort_keys=True, ensure_ascii=False)
        for item in state.get("tool_results") or []
    }
    if signature in seen:
        state["pending"] = None
        _stop(
            state,
            "duplicate_tool_call",
            f"拒绝重复调用 {name}（相同参数已执行）",
            status="partial",
        )
        return _leave(state, "execute")
    state["pending"] = None
    _execute_tool(
        state,
        name,
        pending.get("arguments") or {},
        call_id=state.get("pending_call_id"),
        phase=str(state.get("stage") or ""),
        source=str(pending.get("source") or "unknown"),
    )
    state["pending_call_id"] = None
    # 历史只完成对应业务。跨业务任务还要继续处理其他目标。
    if name == "search_experiment_history" and not state.get("stop_reason"):
        history = state.get("history") or {}
        if (
            history.get("hit") and history.get("reusable")
            and set(state.get("targets") or []) <= _completed_businesses(state)
            and not _budget(state).counters["experiment"]
            and not (state.get("intent") or {}).get("read_record")
        ):
            _stop(
                state,
                "history_hit",
                "历史命中且通过校验，直接复用；本轮新实验次数为 0",
            )
    return _leave(state, "execute")


def node_finish(state: AgentState) -> AgentState:
    state = _enter(state, "finish")
    budget = _budget(state)
    results = list(state.get("tool_results") or [])
    any_ok = any(item.get("ok") for item in results)
    all_ok = bool(results) and all(item.get("ok") for item in results)
    stop_reason = str(state.get("stop_reason") or "")
    if stop_reason == "history_hit":
        status = "verified"
    elif stop_reason in {"budget_exhausted", "deadline", "max_steps"}:
        status = "partial" if any_ok else "failed"
    elif stop_reason in {"llm_request_failed", "invalid_tool_call", "no_tool_call"}:
        status = "partial" if any_ok else "failed"
    elif stop_reason in {"duplicate_tool_call"}:
        status = "partial"
    elif stop_reason in {"llm_stop", "scripted_stop", "no_action"}:
        status = "verified" if all_ok and any_ok else ("partial" if any_ok else "failed")
    elif stop_reason in {"llm_unavailable"}:
        status = "blocked"
    else:
        status = "verified" if all_ok and any_ok else ("partial" if any_ok else "failed")
    if stop_reason == "history_hit" and not results:
        status = "failed"
    if status == "verified":
        read_record = (state.get("intent") or {}).get("read_record")
        missing = set(state.get("targets") or []) - _completed_businesses(state)
        if read_record:
            missing = set() if any(
                item["tool"] == "read_experiment_result" and item.get("ok")
                and item.get("record_id") == str(read_record)
                for item in results
            ) else {"read_experiment_result"}
        if missing:
            status = "partial"
            stop_reason = "incomplete_targets"
            state["final_answer"] = ""
            _stop(state, stop_reason, f"尚未完成：{', '.join(sorted(missing))}", status=status)
    state["status"] = status
    state["summary"] = _build_summary(state, status)
    _event(
        state,
        {
            "event": "task_end",
            "status": status,
            "stop_reason": stop_reason,
            "new_experiments": budget.counters.get("experiment", 0),
            "budget": budget.snapshot(),
        },
    )
    return _leave(state, "finish")


def _route_after_history(state: AgentState) -> str:
    return "finish" if state.get("stop_reason") else "plan"


def _route_after_plan(state: AgentState) -> str:
    return "execute" if state.get("pending") else "finish"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("history", node_history)
    graph.add_node("plan", node_plan)
    graph.add_node("execute", node_execute)
    graph.add_node("finish", node_finish)
    graph.set_entry_point("history")
    graph.add_conditional_edges("history", _route_after_history, {"plan": "plan", "finish": "finish"})
    graph.add_conditional_edges("plan", _route_after_plan, {"execute": "execute", "finish": "finish"})
    graph.add_edge("execute", "plan")
    graph.add_edge("finish", END)
    return graph.compile()


# ---------------------------------------------------------------------- 汇总


def _compose_answer(state: AgentState, status: str) -> str:
    explicit = str(state.get("final_answer") or "").strip()
    if explicit:
        return explicit
    budget = _budget(state)
    history = state.get("history") or {}
    if state.get("stop_reason") == "history_hit":
        source = history.get("source")
        metrics = history.get("metrics") or {}
        keys = ", ".join(f"{key}={value}" for key, value in list(metrics.items())[:4])
        return (
            f"历史命中（{source}）并已通过校验，直接复用，新实验 0 次。"
            f"关键指标：{keys or '见记录'}。"
        )
    pieces: list[str] = []
    for item in state.get("tool_results") or []:
        tool = str(item.get("tool"))
        label = (
            item.get("business")
            if tool in TOOL_BUSINESS
            else tool
        )
        if item.get("ok"):
            pieces.append(f"{label} 成功（status={item.get('status')}）")
        else:
            pieces.append(f"{label} 失败（{item.get('error') or '未知原因'}）")
    detail = str(state.get("stop_detail") or "")
    if state.get("stop_reason") == "budget_exhausted":
        head = "预算用尽，已停止并保留已完成结果"
    elif state.get("stop_reason") == "deadline":
        head = "总时限用尽，已停止并保留已完成结果"
    else:
        head = f"任务结束（{state.get('stop_reason') or status}）"
    return f"{head}；" + ("；".join(pieces) if pieces else "没有执行任何业务工具") + (
        f"；{detail}" if detail else ""
    )


def _build_summary(state: AgentState, status: str) -> dict[str, Any]:
    budget = _budget(state)
    history = state.get("history")
    results = list(state.get("tool_results") or [])
    businesses = list(
        dict.fromkeys(
            str(item.get("business"))
            for item in results
            if item.get("business") and (str(item.get("tool")) in TOOL_BUSINESS
                                         or item.get("tool") == "run_algorithm_research")
        )
    )
    record_ids = [item.get("record_id") for item in results if item.get("record_id")]
    summary = {
        "task_id": state.get("task_id"),
        "task": state.get("task"),
        "status": status,
        "planner_requested": state.get("planner_requested"),
        "planner": state.get("planner_effective"),
        "decision_source": "llm" if state.get("llm_used") else "scripted_stand_in",
        "scripted_stand_in": bool(state.get("scripted_stand_in")),
        "llm_used": bool(state.get("llm_used")),
        "llm_available": bool(state.get("llm_available")),
        "llm_blocker": state.get("llm_blocker"),
        "llm_requests": int(state.get("llm_requests", 0)),
        "stop_reason": state.get("stop_reason"),
        "stop_detail": state.get("stop_detail"),
        "history": history_model_view(history) if history else None,
        "reuse_mode": "history" if state.get("stop_reason") == "history_hit" else None,
        "new_experiments": int(budget.counters.get("experiment", 0)),
        "businesses": businesses,
        "record_ids": record_ids,
        "tool_results": results,
        "decisions": state.get("decisions") or [],
        "budget": budget.snapshot(),
        "budget_refusals": state.get("budget_refusals") or [],
        "final_answer": _compose_answer(state, status),
        "artifacts": {
            "summary": display_path(Path(state["task_dir"]) / "summary.json"),
            "trace": display_path(Path(state["task_dir"]) / "trace.jsonl"),
            "tool_results": display_path(Path(state["task_dir"]) / "tool_results.json"),
            "decisions": display_path(Path(state["task_dir"]) / "decisions.json"),
            "final_answer": display_path(Path(state["task_dir"]) / "final_answer.md"),
        },
        "elapsed_s": round(time.perf_counter() - float(state.get("started", time.perf_counter())), 2),
    }
    return summary


def _write_outputs(state: AgentState) -> None:
    task_dir = Path(state["task_dir"])
    task_dir.mkdir(parents=True, exist_ok=True)
    summary = state["summary"]
    (task_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (task_dir / "tool_results.json").write_text(
        json.dumps(summary.get("tool_results") or [], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (task_dir / "decisions.json").write_text(
        json.dumps(summary.get("decisions") or [], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (task_dir / "final_answer.md").write_text(
        f"# Research Agent 任务结果\n\n- 任务：{summary.get('task')}\n"
        f"- 状态：**{summary.get('status')}**\n- 停止原因：`{summary.get('stop_reason')}`\n"
        f"- 决策来源：`{summary.get('decision_source')}`\n"
        f"- 新实验次数：{summary.get('new_experiments')}\n\n"
        f"## 结论\n\n{summary.get('final_answer')}\n",
        encoding="utf-8",
    )
    store = state["store"]
    record = build_record(
        task_id=str(state["task_id"]),
        business="task",
        status=summary["status"],
        config={
            "task": state.get("task"),
            "planner_requested": state.get("planner_requested"),
            "planner": summary.get("planner"),
            "decision_source": summary.get("decision_source"),
            "targets": state.get("targets"),
            "max_steps": state["config"].max_steps,
        },
        data={"dataset": (state.get("intent") or {}).get("dataset", "kuasar_small"), "split": None, "split_role": None},
        model={
            "models": list(summary.get("businesses") or []),
            "weights_source": "existing_business_artifacts",
        },
        result={
            "status": summary["status"],
            "summary": summary.get("final_answer"),
            "businesses": summary.get("businesses"),
            "record_ids": summary.get("record_ids"),
            "history": summary.get("history"),
            "stop_reason": summary.get("stop_reason"),
            "new_experiments": summary.get("new_experiments"),
        },
        budget=summary.get("budget"),
        artifacts={
            "summary": display_path(task_dir / "summary.json"),
            "trace": display_path(task_dir / "trace.jsonl"),
            "decisions": display_path(task_dir / "decisions.json"),
            "tool_results": display_path(task_dir / "tool_results.json"),
        },
        evidence={
            "validated": summary["status"] == "verified",
            "reusable": False,
            "execution_mode": "real_tools",
            "decision_driver": summary.get("planner"),
            "warnings": ["task_record"] + (["scripted_stand_in"] if summary.get("scripted_stand_in") else []),
            "checks": {
                "has_tool_results": bool(summary.get("tool_results")),
                "new_experiments": summary.get("new_experiments"),
            },
            "notes": "总控任务记录；不作为业务成功证据复用。",
        },
        producer="recsys.platform.research_agent",
    )
    stored = store.append(record)
    summary["task_record_id"] = stored.get("record_id")
    (task_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------- 入口


class ResearchAgent:
    def __init__(
        self,
        config: ResearchConfig | None = None,
        *,
        planner: str = "auto",
        store: SharedExperimentStore | Path | str | None = None,
    ) -> None:
        self.config = config or ResearchConfig.load()
        self.planner = planner
        if isinstance(store, SharedExperimentStore):
            self.store = store
        else:
            self.store = SharedExperimentStore(
                resolve_path(store) if store else resolve_path(self.config.store_dir) or REPO_ROOT / DEFAULT_STORE_DIR
            )

    def run(
        self,
        task: str,
        *,
        intent: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        load_dotenv()
        llm = llm_config()
        missing = [name for name in LLM_ENV if not os.environ.get(name, "").strip()]
        llm_blocker = None if llm else f"缺少/为空的环境变量：{', '.join(missing)}"
        planner_requested = str(self.planner or "auto")
        if planner_requested == "llm" and not llm:
            return self._blocked_summary(task, intent, llm_blocker or "LLM 未配置")
        planner_effective = "llm" if (planner_requested in {"auto", "llm"} and llm) else "scripted"

        task_id = f"task-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        task_dir = (resolve_path(self.config.experiment_root) or Path(self.config.experiment_root)) / "tasks" / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "trace.jsonl").write_text("", encoding="utf-8")
        budget = TaskBudget(task_id, self.config.budget)
        intent_payload = dict(intent or {})
        if intent_payload.get("quota") and not isinstance(intent_payload["quota"], (list, tuple)):
            intent_payload["quota"] = tuple(intent_payload["quota"])
        targets = infer_businesses(task, intent_payload)
        if _is_research_task(task, intent_payload) and not (
            intent_payload.get("businesses") or intent_payload.get("business")
        ):
            quota = intent_payload.get("quota") or _quota_from_text(task)
            if quota and "recommendation" not in targets:
                targets.insert(0, "recommendation")

        state: AgentState = {
            "status": "running",
            "stage": "start",
            "nodes": [],
            "task": task,
            "task_id": task_id,
            "config": self.config,
            "budget": budget,
            "store": self.store,
            "task_dir": task_dir,
            "intent": intent_payload,
            "targets": targets,
            "history": None,
            "tool_results": [],
            "decisions": [],
            "messages": [],
            "pending": None,
            "pending_call_id": None,
            "stop_reason": "",
            "stop_detail": "",
            "final_answer": "",
            "llm_available": bool(llm),
            "llm_blocker": llm_blocker,
            "llm_used": False,
            "llm_requests": 0,
            "planner_requested": planner_requested,
            "planner_effective": planner_effective,
            "scripted_stand_in": planner_effective == "scripted",
            "llm_config": llm,
            "started": time.perf_counter(),
            "step": 0,
            "budget_refusals": [],
            "summary": {},
        }
        _event(
            state,
            {
                "event": "task_start",
                "task": task,
                "targets": targets,
                "planner_requested": planner_requested,
                "planner_effective": planner_effective,
                "llm_available": bool(llm),
                "llm_blocker": llm_blocker,
                "budget": budget.snapshot(),
            },
        )
        graph = build_graph()
        final_state = graph.invoke(state)
        summary = dict(final_state.get("summary") or {})
        if not summary:
            final_state["summary"] = _build_summary(final_state, str(final_state.get("status")))
            summary = final_state["summary"]
        final_state["status"] = summary.get("status")
        _write_outputs(final_state)
        summary = dict(final_state.get("summary") or {})
        summary["artifacts"] = {
            **(summary.get("artifacts") or {}),
            "task_record": summary.get("task_record_id"),
        }
        return summary

    # ------------------------------------------------------------------ 阻止

    def _blocked_summary(
        self,
        task: str,
        intent: Mapping[str, Any] | None,
        blocker: str,
    ) -> dict[str, Any]:
        task_id = f"task-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        budget = TaskBudget(task_id, self.config.budget)
        task_dir = (resolve_path(self.config.experiment_root) or Path(self.config.experiment_root)) / "tasks" / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "trace.jsonl").write_text("", encoding="utf-8")
        summary = {
            "task_id": task_id,
            "task": task,
            "status": "blocked",
            "planner_requested": "llm",
            "planner": "none",
            "decision_source": "none",
            "scripted_stand_in": False,
            "llm_used": False,
            "llm_available": False,
            "llm_blocker": blocker,
            "llm_requests": 0,
            "stop_reason": "llm_unavailable",
            "stop_detail": blocker,
            "history": None,
            "reuse_mode": None,
            "new_experiments": 0,
            "businesses": [],
            "record_ids": [],
            "tool_results": [],
            "decisions": [],
            "budget": budget.snapshot(),
            "budget_refusals": [],
            "final_answer": f"真实 LLM 不可用，未执行任何工具：{blocker}",
            "artifacts": {
                "summary": display_path(task_dir / "summary.json"),
                "trace": display_path(task_dir / "trace.jsonl"),
            },
            "elapsed_s": 0.0,
            "intent": dict(intent or {}),
        }
        _event(
            {"task_dir": task_dir},
            {"event": "blocked", "reason": "llm_unavailable", "detail": blocker, "task": task},
        )
        (task_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (task_dir / "final_answer.md").write_text(
            f"# Research Agent 任务结果\n\n- 状态：**blocked**\n- 原因：{blocker}\n",
            encoding="utf-8",
        )
        record = build_record(
            task_id=task_id,
            business="task",
            status="blocked",
            config={"task": task, "planner_requested": "llm", "planner": "none"},
            data={"dataset": "kuasar_small", "split": None, "split_role": None},
            model={"models": [], "weights_source": "none"},
            result={"status": "blocked", "summary": summary["final_answer"], "stop_reason": "llm_unavailable"},
            budget=budget.snapshot(),
            artifacts={
                "summary": display_path(task_dir / "summary.json"),
                "final_answer": display_path(task_dir / "final_answer.md"),
                "trace": display_path(task_dir / "trace.jsonl"),
            },
            evidence={
                "validated": False,
                "reusable": False,
                "execution_mode": "real_tools",
                "decision_driver": "none",
                "warnings": ["llm_unavailable"],
                "notes": blocker,
            },
            producer="recsys.platform.research_agent",
        )
        stored = self.store.append(record)
        summary["task_record_id"] = stored.get("record_id")
        (task_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary


def run_task(
    task: str,
    *,
    config: ResearchConfig | None = None,
    planner: str = "auto",
    intent: Mapping[str, Any] | None = None,
    store: SharedExperimentStore | Path | str | None = None,
) -> dict[str, Any]:
    return ResearchAgent(config, planner=planner, store=store).run(task, intent=intent)


__all__ = [
    "ResearchAgent",
    "ResearchConfig",
    "ResearchTool",
    "TOOLS",
    "build_graph",
    "infer_businesses",
    "run_task",
    "scripted_decision",
]
