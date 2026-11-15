"""Search Agent：BM25 / FAISS 向量 / 混合检索 / 精排四个工具 + 预算约束。

复用 Search 1 的 KuaiSAR 小样本索引，通过原生 tool_calls 让模型先选择检索工具：

1. 第一次必须调用 ``search_bm25`` / ``search_vector`` / ``search_hybrid`` 之一，
   且 ``query_tokens`` 等于规范化后的原始查询（``original_query_tokens``）；
2. 收到结果后，模型可以停止、调用 ``search_rerank`` 精排，或从原查询里保留
   一部分 token 再查一次（第二次检索，可换工具）；
3. 最多 2 次检索 + 1 次精排、3 次模型请求；用完预算后只允许最终回复；
4. 程序在执行工具前检查参数、重复、子集、精排查询来源和预算；不自动重试。

真实工作流（需要 LLM 配置）：

    $env:PYTHONPATH = "src"
    python -m recsys.experiments.search_agent --self-check
    python -m recsys.experiments.search_agent `
        --tokens "3538762,3181091,4061567,3525929" `
        --output-root experiments/2026-11-09-kuasar-search-agent

本地替身驱动（不联网，但检索/精排/落盘执行真实代码）：

    python -m recsys.experiments.search_agent --scripted-plan hybrid_rerank `
        --output-root experiments/2026-11-09-kuasar-search-phase2/agent_scripted
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
from recsys.search.encoders import build_encoder
from recsys.search.hybrid import SOURCE_BM25, SOURCE_VECTOR
from recsys.search.query import candidate_rewrites, normalize_query
from recsys.search.rerank import RERANKER_NAME
from recsys.search.service import SearchService, SearchServiceError
from recsys.search.vector_index import FaissFlatIndex, VectorIndexError
from recsys.search.tool import (
    SEARCH_TOOL_NAMES,
    TOOL_HYBRID,
    TOOL_NAME,
    TOOL_RERANK,
    TOOL_SCHEMAS,
    TOOL_NAMES,
    TOOL_VECTOR,
    SearchBM25Tool,
    SearchToolbox,
    ToolRejected,
    normalize_top_k,
    normalize_tokens,
    validate_first_query,
    validate_second_query,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "search_agent.toml"
DEFAULT_REGRESSION_ROOT = PROJECT_ROOT / "artifacts" / "search_agent_regression"
SCHEMA_VERSION = 2
TRACE_FORMAT = "recsys.search_agent.trace.v2"
DRIVER_LLM = "llm"
DRIVER_SCRIPTED = "scripted_driver"


def run_research(task: str, **kwargs) -> dict:
    """Search Agent's code-editing research on real retrieval candidates."""
    from recsys.platform.autoresearch import run_research as research
    return research(task, business="search", **kwargs)

#: Phase 4：可选的项目知识库检索工具。默认不配置 ``[knowledge]`` 时不注册，
#: 原有四个搜索工具、预算和检查项完全不变。
TOOL_KNOWLEDGE = "retrieve_knowledge"
MAX_KNOWLEDGE_CALLS = 1


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


def _find_session_by_tokens(
    sessions: Sequence[Mapping[str, Any]], tokens: Sequence[int]
) -> Mapping[str, Any] | None:
    """能在样本中遇到同 query 时会话就返回；样本外查询返回 None。"""
    for session in sessions:
        if [int(token) for token in session.get("keyword", [])] == list(tokens):
            return session
    return None


class SearchContext:
    """一次 Agent 运行用到的索引、搜索服务、查询形态与追溯信息。"""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        index: BM25Index,
        service: SearchService,
        toolbox: SearchToolbox,
        tool: SearchBM25Tool,
        index_path: Path,
        sample_path: Path,
        index_sha256: str,
        sample_sha256: str,
        query_tokens: list[int],
        original_tokens: list[int],
        query_normalization: str,
        session_id: str | None,
        query_source: str,
        clicked_items_count: int,
        clicked_joined_count: int,
        vector_manifest: dict[str, Any] | None,
        vector_payload: dict[str, str],
        knowledge: Any = None,
    ) -> None:
        self.config = config
        self.index = index
        self.service = service
        self.toolbox = toolbox
        self.tool = tool
        self.index_path = index_path
        self.sample_path = sample_path
        self.index_sha256 = index_sha256
        self.sample_sha256 = sample_sha256
        self.query_tokens = query_tokens
        self.original_tokens = original_tokens
        self.query_normalization = query_normalization
        self.session_id = session_id
        self.query_source = query_source
        self.clicked_items_count = clicked_items_count
        self.clicked_joined_count = clicked_joined_count
        self.vector_manifest = vector_manifest or {}
        self.vector_payload = vector_payload
        self.knowledge = knowledge
        self.rewrites = candidate_rewrites(query_tokens)

    def identity(self) -> dict[str, Any]:
        return {
            "path": _display_path(self.index_path),
            "sha256": self.index_sha256,
            "num_docs": self.index.num_docs,
            "vocab_size": self.index.vocab_size,
            "avgdl": round(self.index.avgdl, 6),
        }

    def query_payload(self) -> dict[str, Any]:
        return {
            "original_tokens": list(self.original_tokens),
            "normalized_tokens": list(self.query_tokens),
            "normalization": self.query_normalization,
            "rewrites": [form.as_dict() for form in self.rewrites],
            "session_id": self.session_id,
            "source": self.query_source,
            "clicked_items_count": self.clicked_items_count,
            "clicked_joined_count": self.clicked_joined_count,
        }

    def backend_identity(self) -> dict[str, Any]:
        described = self.service.describe()
        return {
            "bm25": described["bm25"],
            "vector": {
                "backend_actual": described["vector"]["backend_actual"],
                "faiss_active": described["vector"]["faiss_active"],
                "faiss_version": described["vector"]["faiss_version"],
                "num_items": described["vector"]["num_items"],
                "dim": described["vector"]["dim"],
                "encoder": described["vector"]["encoder"]["encoder"],
                "encoder_config_hash": described["vector"]["encoder_config_hash"],
                "cache_dir": described["vector"]["cache_dir"],
                "faiss_search_calls": described["vector"]["faiss_search_calls"],
                "payload": self.vector_payload,
            },
            "hybrid": described["hybrid"],
            "rerank": described["rerank"],
            "tools": list(self.toolbox.names()),
            "knowledge": {
                "enabled": bool(self.knowledge),
                "tool": TOOL_KNOWLEDGE if self.knowledge else None,
            },
            "vector_manifest": self.vector_manifest or None,
        }


def _search_backend_config(config: Mapping[str, Any]) -> dict[str, Any]:
    backend = dict(config.get("search_backend") or {})
    if not backend:
        raise AgentBlocked("配置缺少 [search_backend]（向量索引、编码器与融合/精排参数）")
    for key in ("vector_index_dir", "encoder"):
        if not backend.get(key):
            raise AgentBlocked(f"[search_backend] 缺少 {key}")
    return backend


def _encoder_config(backend: Mapping[str, Any]) -> dict[str, Any]:
    encoder_name = str(backend.get("encoder") or "token_hash")
    return {
        "name": encoder_name,
        "dim": int(backend.get("encoder_dim", 256)),
        "seed": int(backend.get("encoder_seed", 20261109)),
        "include_category": bool(backend.get("include_category", True)),
        "category_weight": float(backend.get("category_weight", 0.25)),
        "model_path": backend.get("encoder_model_path", ""),
    }


def load_context(
    config: Mapping[str, Any],
    *,
    tokens: Sequence[int] | None,
    session_id: str | None,
) -> SearchContext:
    search_cfg = config["search_agent"]
    backend_cfg = _search_backend_config(config)
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

    # 搜索向量索引独立于推荐索引；必须走 FAISS 后端。
    vector_dir = _resolve_path(backend_cfg["vector_index_dir"])
    manifest_path = _resolve_path(
        backend_cfg.get("vector_manifest")
        or (Path(backend_cfg["vector_index_dir"]).parent / "manifest.json")
    )
    vector_manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        vector_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_manifest_sha = str(backend_cfg.get("expected_vector_manifest_sha256") or "")
        if expected_manifest_sha and _sha256_file(manifest_path) != expected_manifest_sha:
            raise AgentBlocked("搜索向量索引 manifest sha256 与配置记录不一致")
        manifest_items = int((vector_manifest.get("index") or {}).get("num_items", -1))
        if manifest_items != index.num_docs:
            raise AgentBlocked(
                f"向量索引物品数 {manifest_items} 与 BM25 索引 {index.num_docs} 不一致"
            )
    elif backend_cfg.get("require_vector_manifest", True):
        raise AgentBlocked(
            f"缺少搜索向量索引 manifest：{_display_path(manifest_path)}；"
            "先运行 python -m recsys.search.build_vector_index"
        )

    encoder = build_encoder(_encoder_config(backend_cfg))
    payload_path = vector_dir / FaissFlatIndex.PAYLOAD
    try:
        service = SearchService.load(
            index_path,
            vector_dir,
            encoder,
            rrf_k=int(backend_cfg.get("rrf_k", 60)),
            hybrid_pool=int(backend_cfg.get("hybrid_pool", 50)),
            rerank_method=backend_cfg.get("rerank_method", RERANKER_NAME),
            rerank_pool=int(backend_cfg.get("rerank_pool", 50)),
            prior_weight=float(backend_cfg.get("prior_weight", 0.35)),
            coverage_weight=float(backend_cfg.get("coverage_weight", 0.65)),
        )
        if not service.vector.faiss_active:
            raise AgentBlocked(
                f"搜索向量索引没有走 FAISS（backend={service.vector.backend_actual}）"
            )
        vector_payload = {"path": _display_path(payload_path), "sha256": _sha256_file(payload_path)}
    except (SearchServiceError, VectorIndexError, OSError, ValueError) as error:
        raise AgentBlocked(f"搜索服务初始化失败：{error}") from error

    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    sessions = sample.get("sessions") or []
    if not sessions:
        raise AgentBlocked("sample.json 里没有 sessions")

    explicit_tokens = list(tokens) if tokens is not None else None
    session: Mapping[str, Any] | None = None
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
        # 只传 --tokens：先用输入查索引；能遇到同样 query 的样本会话就补充元信息，
        # 样本外查询也继续执行，不再 blocked。
        session = _find_session_by_tokens(sessions, explicit_tokens)
        if session is not None:
            session_id = str(session.get("session_id"))
            query_source = "cli_tokens_matched_session"
        else:
            session_id = None
            query_source = "cli_tokens_out_of_sample"
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
    form = normalize_query(query_tokens, max_tokens=int(search_cfg["max_query_tokens"]))
    if not form.normalized:
        raise AgentBlocked("规范化后的查询为空")
    normalized_tokens = list(form.normalized)

    toolbox = SearchToolbox(service, max_results=int(search_cfg["max_results"]))
    tool = SearchBM25Tool(index, max_results=int(search_cfg["max_results"]))
    knowledge = None
    knowledge_cfg = dict(config.get("knowledge") or {})
    if knowledge_cfg.get("enabled"):
        try:
            from recsys.platform.rag import KnowledgeTool

            knowledge = KnowledgeTool(
                index_dir=knowledge_cfg.get("index_dir"),
                default_top_k=int(knowledge_cfg.get("top_k", 2)),
            )
        except Exception as error:  # noqa: BLE001 - 知识库不可用时明确 blocked
            raise AgentBlocked(f"知识库工具初始化失败：{error}") from error
    return SearchContext(
        config=dict(config),
        index=index,
        service=service,
        toolbox=toolbox,
        tool=tool,
        index_path=index_path,
        sample_path=sample_path,
        index_sha256=index_sha,
        sample_sha256=sample_sha,
        query_tokens=normalized_tokens,
        original_tokens=query_tokens,
        query_normalization=form.normalization,
        session_id=session_id,
        query_source=query_source,
        clicked_items_count=len((session or {}).get("clicked_items") or []),
        clicked_joined_count=len((session or {}).get("clicked_joined_items") or []),
        vector_manifest=vector_manifest,
        vector_payload=vector_payload,
        knowledge=knowledge,
    )


# ---------------------------------------------------------------------------
# 消息与工具调用解析
# ---------------------------------------------------------------------------


def _messages_first(context: SearchContext) -> list[dict[str, Any]]:
    system = (
        "你是短视频搜索 Agent。你必须在第一轮调用一个检索工具："
        "search_bm25（关键词）、search_vector（FAISS 向量）或 search_hybrid（混合）。"
        "第一次 query_tokens 必须与 original_query_tokens 完全一致（顺序也不能变），"
        "不能增删改。search_rerank 只能在已有检索结果之后调用。"
        "禁止用普通文本代替工具调用，禁止并行多个 tool_calls。每次最多返回 5 条结果。"
    )
    payload = {
        "task": "在 KuaiSAR 小样本索引中检索短视频。",
        "original_query_tokens": list(context.query_tokens),
        "normalized_from": list(context.original_tokens),
        "query_normalization": context.query_normalization,
        "session_id": context.session_id,
        "available_tools": list(context.toolbox.names()),
        "max_results_per_query": context.toolbox.max_results,
        "remaining_search_budget": int(context.config["search_agent"]["max_search_calls"]),
        "remaining_rerank_budget": int(context.config["search_agent"].get("max_rerank_calls", 1)),
        "instruction": (
            "第一次必须原样使用 original_query_tokens 调用一个检索工具"
            "（search_bm25 / search_vector / search_hybrid）；只允许 1 个 tool_call，禁止并行。"
            + (
                " 如需项目背景，可在首轮先调用一次 retrieve_knowledge，再按同样规则检索。"
                if context.knowledge
                else ""
            )
        ),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _available_tool_schemas(state: "AgentState") -> list[dict[str, Any]] | None:
    """当前还能用的工具；没有剩余工具时返回 None（只允许最终回复）。"""
    context: SearchContext = state["context"]
    allowed = set(context.toolbox.names())
    available = set(allowed)
    if _pending_search_budget(state) <= 0:
        available -= set(SEARCH_TOOL_NAMES)
    if _pending_rerank_budget(state) <= 0:
        available.discard(TOOL_RERANK)
    if not available:
        return None
    return [TOOL_SCHEMAS[name] for name in context.toolbox.names() if name in available]


def _followup_message(state: "AgentState", phase: str) -> dict[str, Any]:
    context: SearchContext = state["context"]
    remaining = _pending_search_budget(state)
    rerank_remaining = _pending_rerank_budget(state)
    available = [name for name in context.toolbox.names() if name in _available_tool_names(state)]
    if phase == "feedback":
        instruction = (
            "第一次检索结果已经通过 tool 消息返回。你现在可以选择："
            "(1) 停止并说明原因；"
            "(2) 调用 search_rerank 对融合候选池重新打分（query_tokens 必须等于本次任务里"
            "已经检索过的查询）；"
            "(3) 从 original_query_tokens 中保留一部分 token 再查一次（第二次检索，可换工具）。"
            "如果继续检索：query_tokens 必须非空、与第一次不同、只能使用原查询 token，"
            "并且至少去掉一个 token；只允许 1 个 tool_call，禁止并行。"
        )
    elif phase == "final":
        instruction = (
            "检索与精排预算已经用完或不再继续，不能再调用任何工具。"
            "请只输出一句最终结论或停止理由，不要调用工具，不要输出 JSON 代码块。"
        )
    else:
        raise ValueError(f"未知阶段：{phase}")
    payload = {
        "phase": phase,
        "original_query_tokens": list(context.query_tokens),
        "previous_queries": [list(query) for query in state.get("queries_used", [])],
        "remaining_search_budget": remaining,
        "remaining_rerank_budget": rerank_remaining,
        "available_tools": available,
        "max_results_per_query": context.toolbox.max_results,
        "instruction": instruction,
    }
    return {
        "role": "user",
        "content": json.dumps(payload, ensure_ascii=False),
    }


def _available_tool_names(state: "AgentState") -> set[str]:
    context: SearchContext = state["context"]
    names = set(context.toolbox.names())
    if _pending_search_budget(state) <= 0:
        names -= set(SEARCH_TOOL_NAMES)
    if _pending_rerank_budget(state) <= 0:
        names.discard(TOOL_RERANK)
    return names


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


def _record_final_reply_status(
    state: "AgentState", *, reply_status: str, content: str
) -> None:
    """只记录完成状态和长度，不保存模型自由文本。"""
    state["final_reply_status"] = reply_status
    state["final_reply_chars"] = len(content)


def _stop_summary(state: "AgentState", *, stop_type: str) -> str:
    """程序生成的停止摘要；不引用模型自由文本。"""
    search_calls = int(state.get("search_calls", 0))
    rerank_calls = int(state.get("rerank_calls", 0))
    counts: list[str] = []
    if search_calls >= 1:
        counts.append(str(int(state.get("first_result_count", 0))))
    if search_calls >= 2:
        counts.append(str(int(state.get("second_result_count", 0))))
    tools = [str(name) for name in state.get("tool_sequence", [])]
    return (
        f"模型在 {search_calls} 次检索 / {rerank_calls} 次精排后停止（{stop_type}）；"
        f"模型请求 {int(state.get('request_count', 0))}/"
        f"{int(state.get('max_model_requests', 3))}，"
        f"检索 {search_calls}/{int(state.get('max_search_calls', 2))}，"
        f"精排 {rerank_calls}/{int(state.get('max_rerank_calls', 1))}，"
        f"工具序列 {'/'.join(tools) if tools else '0'}，"
        f"各次检索返回数量 {'/'.join(counts) if counts else '0'}；"
        f"最终回复状态 {state.get('final_reply_status', 'missing')}"
        f"（{int(state.get('final_reply_chars', 0))} 字符，正文不落盘）。"
    )


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
    return {
        "id": response.get("id"),
        "finish_reason": _finish_reason(response),
        "content_chars": len(content),
        "content_present": bool(content.strip()),
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


def parse_single_tool_call(
    response: Mapping[str, Any],
    allowed_tools: Sequence[str] | None = None,
) -> dict[str, Any]:
    """解析恰好一个工具调用，并校验工具名与参数。"""
    calls = _message_tool_calls(response)
    if not calls:
        raise ToolRejected("no_tool_call", "响应里没有 tool_calls；普通文本不算检索")
    if len(calls) != 1:
        raise ToolRejected("multiple_tool_calls", f"tool_calls 数量为 {len(calls)}，要求恰好 1 条")
    call = calls[0]
    function = call.get("function") or {}
    if (call.get("type") or "function") != "function":
        raise ToolRejected("unsupported_tool_type", f"不支持的 tool call 类型：{call.get('type')}")
    name = str(function.get("name") or "")
    if name not in TOOL_NAMES:
        raise ToolRejected("unknown_tool", f"未知工具：{name}")
    if allowed_tools is not None and name not in allowed_tools:
        raise ToolRejected(
            "tool_not_allowed",
            f"当前阶段不允许调用 {name}；可用：{list(allowed_tools)}",
        )
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
        "name": name,
        "query_tokens": list(query_tokens),
        "top_k": top_k,
        "arguments": {"query_tokens": list(query_tokens), "top_k": top_k},
    }


def parse_single_search_call(response: Mapping[str, Any]) -> dict[str, Any]:
    """兼容旧入口：只允许 ``search_bm25``。"""
    return parse_single_tool_call(response, allowed_tools=(TOOL_NAME,))


def knowledge_tool_schema() -> dict[str, Any]:
    """Phase 4 知识库工具 schema；仅当配置启用 ``[knowledge]`` 时暴露。"""

    try:
        from recsys.platform.rag import KNOWLEDGE_TOOL_SCHEMA

        return copy.deepcopy(KNOWLEDGE_TOOL_SCHEMA)
    except Exception:  # pragma: no cover - 没有平台包时保持搜索 Agent 可独立运行
        return {
            "type": "function",
            "function": {
                "name": TOOL_KNOWLEDGE,
                "description": "检索本地项目知识库并返回带来源的片段。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }


def parse_single_knowledge_call(response: Mapping[str, Any]) -> dict[str, Any]:
    """解析唯一的知识库检索调用；参数只有 query / top_k。"""

    calls = _message_tool_calls(response)
    if not calls:
        raise ToolRejected("no_tool_call", "响应里没有 tool_calls")
    if len(calls) != 1:
        raise ToolRejected("multiple_tool_calls", f"tool_calls 数量为 {len(calls)}，要求恰好 1 条")
    call = calls[0]
    function = call.get("function") or {}
    if (call.get("type") or "function") != "function":
        raise ToolRejected("unsupported_tool_type", f"不支持的 tool call 类型：{call.get('type')}")
    if str(function.get("name") or "") != TOOL_KNOWLEDGE:
        raise ToolRejected("unknown_tool", f"不是知识库工具：{function.get('name')}")
    call_id = call.get("id")
    if not call_id:
        raise ToolRejected("missing_tool_call_id", "tool_call 缺少 id")
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError as error:
        raise ToolRejected("invalid_arguments_json", f"工具参数不是合法 JSON：{error}") from error
    if not isinstance(arguments, dict):
        raise ToolRejected("invalid_arguments", "工具参数必须是 JSON 对象")
    extra = set(arguments) - {"query", "top_k"}
    if extra:
        raise ToolRejected("unexpected_arguments", f"工具参数包含未知字段：{sorted(extra)}")
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ToolRejected("empty_query", "query 必须是非空字符串")
    top_k = arguments.get("top_k", 2)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 5:
        raise ToolRejected("top_k_out_of_range", "top_k 必须是 1..5 的整数")
    return {
        "id": str(call_id),
        "name": TOOL_KNOWLEDGE,
        "query": query.strip(),
        "top_k": top_k,
        "arguments": {"query": query.strip(), "top_k": top_k},
    }


def _is_knowledge_call(response: Mapping[str, Any]) -> bool:
    calls = _message_tool_calls(response)
    if len(calls) != 1:
        return False
    function = calls[0].get("function") or {}
    return str(function.get("name") or "") == TOOL_KNOWLEDGE


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
    driver: str
    query_tokens: list[int]
    session_id: str | None
    request_count: int
    search_calls: int
    max_search_calls: int
    max_model_requests: int
    rerank_calls: int
    max_rerank_calls: int
    knowledge_calls: int
    max_knowledge_calls: int
    knowledge_exchange: list[dict]
    knowledge_results: list[dict]
    response_ids: dict[str, str]
    trace_steps: list[dict]
    dialogue: list[dict]
    queries_used: list[list[int]]
    tool_call_ids: list[str]
    tool_result_ids: list[str]
    tool_sequence: list[str]
    tool_result_sent: bool
    route_after_feedback: str
    current_call_id: str
    current_tool_name: str
    current_query_tokens: list[int]
    current_top_k: int
    current_arguments: dict
    first_tool_name: str
    second_tool_name: str
    first_result_count: int
    second_result_count: int
    rerank_result_count: int
    final_reply_status: str
    final_reply_chars: int
    stop_code: str
    stop_type: str
    stop_summary: str
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


def _pending_rerank_budget(state: AgentState) -> int:
    return max(0, int(state.get("max_rerank_calls", 1)) - int(state.get("rerank_calls", 0)))


def _visible(state: AgentState) -> dict[str, Any]:
    context: SearchContext = state["context"]
    return {
        "original_query_tokens": list(context.query_tokens),
        "previous_queries": [list(query) for query in state.get("queries_used", [])],
        "remaining_search_budget": _pending_search_budget(state),
        "remaining_rerank_budget": _pending_rerank_budget(state),
        "available_tools": [
            name
            for name in context.toolbox.names()
            if name in _available_tool_names(state)
        ],
        "max_results_per_query": context.toolbox.max_results,
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
    state["stop_type"] = code
    state["stop_summary"] = reason
    state["stop_reason"] = reason
    state["error"] = error or reason
    steps = state.get("trace_steps", [])
    existing_decision = steps[-1].get("decision") if steps else None
    updates: dict[str, Any] = {
        "stop": {
            "code": code,
            "stop_type": code,
            "summary": reason,
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
    if remaining < 1.0:
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

    timeout = min(float(state["config"]["llm"]["request_timeout_s"]), remaining)
    request_index = int(state.get("request_count", 0)) + 1
    step_index = len(state.get("trace_steps", [])) + 1
    tool_return = state["trace_steps"][-1]["tool_return"] if state.get("tool_result_sent") else None
    _append_step(
        state,
        {
            "request": step_index,
            "phase": phase,
            "visible": copy.deepcopy(_visible(state)),
            "tool_result_in_context": {
                "sent": bool(state.get("tool_result_sent")),
                "call_id": state.get("current_call_id") if state.get("tool_result_sent") else None,
                "count": tool_return["count"] if tool_return else None,
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
    tool_name: str,
    query_tokens: Sequence[int],
    top_k: int,
    result: Mapping[str, Any],
) -> None:
    _update_last_step(
        state,
        tool_return={
            "status": "ok",
            "call_id": call_id,
            "tool": tool_name,
            "query_tokens": list(query_tokens),
            "top_k": top_k,
            "count": int(result["count"]),
            # 只记录精排后的最终分数（不落候选 ID），用于核对「确实重新打分」。
            "rerank_scores": (
                [round(float(hit["score"]), 6) for hit in result.get("results") or []]
                if tool_name == TOOL_RERANK
                else None
            ),
            "remaining_search_budget": _pending_search_budget(state),
            "remaining_rerank_budget": _pending_rerank_budget(state),
        },
    )


def node_request_first(state: AgentState, chat_fn: ChatFn) -> AgentState:
    state = _enter(state, "request_first")
    messages = _messages_first(state["context"])
    if state.get("knowledge_exchange"):
        messages = [*messages, *state["knowledge_exchange"]]
    state["dialogue"] = messages
    first_tools = [TOOL_SCHEMAS[name] for name in SEARCH_TOOL_NAMES]
    if state["context"].knowledge is not None:
        first_tools = [*first_tools, knowledge_tool_schema()]
    state, response = _call_model(state, phase="first", tools=first_tools, messages=messages)
    if response is None:
        return _leave(state, "request_first")
    state["dialogue"] = [*state["dialogue"], _assistant_message(response)]
    if _is_knowledge_call(response):
        try:
            parsed = parse_single_knowledge_call(response)
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
                    reason=f"首轮知识库请求不合法：{error.message}",
                    error=error.message,
                    decision_action="reject",
                ),
                "request_first",
            )
        state["current_call_id"] = parsed["id"]
        state["current_tool_name"] = TOOL_KNOWLEDGE
        state["current_arguments"] = parsed["arguments"]
        state.setdefault("tool_call_ids", []).append(parsed["id"])
        state["knowledge_exchange"] = [
            *state.get("knowledge_exchange", []),
            _assistant_message(response),
        ]
        _update_last_step(
            state,
            decision={
                "action": "knowledge",
                "reason": "optional_project_context",
                "tool": TOOL_KNOWLEDGE,
                "query": parsed["query"],
            },
            tool_call={
                "id": parsed["id"],
                "name": TOOL_KNOWLEDGE,
                "arguments": parsed["arguments"],
            },
        )
        _event(
            state,
            {
                "event": "knowledge_call_accepted",
                "request": state.get("request_count"),
                "tool_call_id": parsed["id"],
                "query_chars": len(parsed["query"]),
            },
        )
        return _leave(state, "request_first")
    try:
        parsed = parse_single_tool_call(response, allowed_tools=SEARCH_TOOL_NAMES)
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
    state["current_tool_name"] = parsed["name"]
    state["current_query_tokens"] = parsed["query_tokens"]
    state["current_top_k"] = parsed["top_k"]
    state["current_arguments"] = parsed["arguments"]
    state.setdefault("tool_call_ids", []).append(parsed["id"])
    _update_last_step(
        state,
        decision={
            "action": "search",
            "reason": "first_query_must_equal_original",
            "tool": parsed["name"],
            "query_tokens": parsed["query_tokens"],
        },
        tool_call={
            "id": parsed["id"],
            "name": parsed["name"],
            "arguments": parsed["arguments"],
        },
    )
    _event(
        state,
        {
            "event": "tool_call_accepted",
            "request": state.get("request_count"),
            "tool_call_id": parsed["id"],
            "tool": parsed["name"],
            "query_tokens": parsed["query_tokens"],
        },
    )
    return _leave(state, "request_first")


def node_execute_knowledge(state: AgentState) -> AgentState:
    state = _enter(state, "execute_knowledge")
    if int(state.get("knowledge_calls", 0)) >= int(
        state.get("max_knowledge_calls", MAX_KNOWLEDGE_CALLS)
    ):
        _event(state, {"event": "rejected", "code": "knowledge_budget_exhausted"})
        return _leave(
            _shutdown(
                state,
                status="partial",
                code="knowledge_budget_exhausted",
                reason="知识库检索次数已用完",
                decision_action="reject",
            ),
            "execute_knowledge",
        )
    context: SearchContext = state["context"]
    if context.knowledge is None:
        return _leave(
            _shutdown(
                state,
                status="failed",
                code="knowledge_tool_unavailable",
                reason="配置未启用知识库工具",
            ),
            "execute_knowledge",
        )
    arguments = state.get("current_arguments") or {}
    call_id = str(state.get("current_call_id") or "")
    try:
        result = context.knowledge.retrieve(
            str(arguments.get("query") or ""),
            int(arguments.get("top_k") or 2),
        )
    except Exception as error:  # noqa: BLE001 - 知识库异常进入失败分支
        message = _safe_error(error, state.get("llm_config"))
        _update_last_step(
            state,
            tool_return={"status": "error", "call_id": call_id, "error": message},
        )
        return _leave(
            _shutdown(
                state,
                status="failed",
                code="knowledge_tool_failed",
                reason=f"知识库检索失败：{message}",
                error=message,
            ),
            "execute_knowledge",
        )
    state["knowledge_calls"] = int(state.get("knowledge_calls", 0)) + 1
    hits = list(result.get("hits") or [])
    citations = [str(hit.get("citation")) for hit in hits if hit.get("citation")]
    summary = {
        "ok": bool(result.get("ok")),
        "hit_count": len(hits),
        "citations": citations[:5],
        "error": result.get("error"),
    }
    state.setdefault("knowledge_results", []).append(summary)
    state["knowledge_exchange"] = [
        *state.get("knowledge_exchange", []),
        _tool_message(call_id, result),
    ]
    if call_id:
        state.setdefault("tool_result_ids", []).append(call_id)
    _update_last_step(
        state,
        tool_return={
            "status": "ok" if result.get("ok") else "error",
            "call_id": call_id,
            "tool": TOOL_KNOWLEDGE,
            "hit_count": len(hits),
            "citations": citations[:5],
        },
    )
    _event(
        state,
        {
            "event": "knowledge_returned",
            "tool_call_id": call_id,
            "ok": bool(result.get("ok")),
            "hit_count": len(hits),
            "citations": citations[:5],
        },
    )
    return _leave(state, "execute_knowledge")


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
        result = state["context"].toolbox.run(
            state["current_tool_name"],
            state["current_arguments"],
            prior_queries=state.get("queries_used", []),
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
    state["first_tool_name"] = state["current_tool_name"]
    state.setdefault("tool_sequence", []).append(state["current_tool_name"])
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
        tool_name=state["current_tool_name"],
        query_tokens=state["current_query_tokens"],
        top_k=state["current_top_k"],
        result=result,
    )
    _event(
        state,
        {
            "event": "tool_returned",
            "tool_call_id": state["current_call_id"],
            "tool": state["current_tool_name"],
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
        tools=_available_tool_schemas(state),
        messages=state["dialogue"],
    )
    if response is None:
        return _leave(state, "request_feedback")
    state["dialogue"] = [*state["dialogue"], _assistant_message(response)]
    if not _message_tool_calls(response):
        reply_status, content, finish_reason = _reply_status(response)
        if reply_status == "ok":
            _record_final_reply_status(state, reply_status="ok", content=content)
            search_calls = int(state.get("search_calls", 0))
            stop_type = (
                "model_stop_after_first_search"
                if search_calls <= 1
                else "model_stop_after_second_search"
            )
            stop_summary = _stop_summary(state, stop_type=stop_type)
            state["stop_code"] = "model_stop"
            state["stop_type"] = stop_type
            state["stop_summary"] = stop_summary
            state["stop_reason"] = stop_summary
            state["status"] = "verified"
            _update_last_step(
                state,
                decision={"action": "stop", "reason": stop_summary, "stop_type": stop_type},
                stop={
                    "code": "model_stop",
                    "stop_type": stop_type,
                    "summary": stop_summary,
                    "status": "verified",
                    "finish_reason": finish_reason,
                    "reply_status": "ok",
                },
            )
            _event(
                state,
                {
                    "event": "model_stop",
                    "request": state.get("request_count"),
                    "stop_type": stop_type,
                    "summary": stop_summary,
                },
            )
            state["route_after_feedback"] = "finish"
            return _leave(state, "request_feedback")
        _record_final_reply_status(state, reply_status=reply_status, content=content)
        code = (
            "reply_truncated"
            if reply_status == "truncated"
            else ("non_stop_finish_reason" if reply_status == "non_stop" else "empty_feedback")
        )
        state["stop_type"] = reply_status
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
        allowed = tuple(sorted(_available_tool_names(state)))
        parsed = parse_single_tool_call(response, allowed_tools=allowed)
        if parsed["name"] == TOOL_RERANK:
            if list(parsed["query_tokens"]) not in [
                list(query) for query in state.get("queries_used", [])
            ]:
                raise ToolRejected(
                    "rerank_query_mismatch",
                    "精排的 query_tokens 必须等于本次任务里已经检索过的某个查询",
                )
        else:
            validate_second_query(
                parsed["query_tokens"],
                state["query_tokens"],
                state.get("queries_used", []),
            )
    except ToolRejected as error:
        attempted = ""
        raw_calls = _message_tool_calls(response)
        if raw_calls:
            attempted = str(((raw_calls[0] or {}).get("function") or {}).get("name") or "")
        _update_last_step(
            state,
            tool_call={
                "name": attempted or TOOL_NAME,
                "arguments": {"raw": _safe_text(error.message, 200)},
            },
            decision={"action": "reject", "reason": error.message},
        )
        _event(state, {"event": "tool_rejected", "code": error.code, "message": error.message})
        code_prefix = (
            "invalid_rerank_request" if error.code == "rerank_query_mismatch"
            else "invalid_second_request"
        )
        return _leave(
            _shutdown(
                state,
                status="partial",
                code=f"{code_prefix}:{error.code}",
                reason=f"反馈阶段的工具调用被拒绝：{error.message}",
                error=error.message,
                decision_action="reject",
            ),
            "request_feedback",
        )

    state["current_call_id"] = parsed["id"]
    state["current_tool_name"] = parsed["name"]
    state["current_query_tokens"] = parsed["query_tokens"]
    state["current_top_k"] = parsed["top_k"]
    state["current_arguments"] = parsed["arguments"]
    state.setdefault("tool_call_ids", []).append(parsed["id"])
    is_rerank = parsed["name"] == TOOL_RERANK
    _update_last_step(
        state,
        decision={
            "action": "rerank" if is_rerank else "search",
            "reason": (
                "model_reranked_fused_candidates" if is_rerank
                else "model_issued_second_query_after_first_result"
            ),
            "tool": parsed["name"],
            "query_tokens": parsed["query_tokens"],
        },
        tool_call={
            "id": parsed["id"],
            "name": parsed["name"],
            "arguments": parsed["arguments"],
        },
    )
    _event(
        state,
        {
            "event": "tool_call_accepted",
            "request": state.get("request_count"),
            "tool_call_id": parsed["id"],
            "tool": parsed["name"],
            "query_tokens": parsed["query_tokens"],
        },
    )
    state["route_after_feedback"] = "execute_rerank" if is_rerank else "execute_second"
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
        result = state["context"].toolbox.run(
            state["current_tool_name"],
            state["current_arguments"],
            prior_queries=state.get("queries_used", []),
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
    state["second_tool_name"] = state["current_tool_name"]
    state.setdefault("tool_sequence", []).append(state["current_tool_name"])
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
        tool_name=state["current_tool_name"],
        query_tokens=state["current_query_tokens"],
        top_k=state["current_top_k"],
        result=result,
    )
    _event(
        state,
        {
            "event": "tool_returned",
            "tool_call_id": state["current_call_id"],
            "tool": state["current_tool_name"],
            "query_tokens": state["current_query_tokens"],
            "count": int(result["count"]),
            "remaining_search_budget": _pending_search_budget(state),
        },
    )
    return _leave(state, "execute_second")


def node_execute_rerank(state: AgentState) -> AgentState:
    """对融合候选池执行精排；精排后只剩最终回复。"""
    state = _enter(state, "execute_rerank")
    if int(state.get("rerank_calls", 0)) >= int(state.get("max_rerank_calls", 1)):
        _event(state, {"event": "rejected", "code": "rerank_budget_exhausted"})
        return _leave(
            _shutdown(
                state,
                status="partial",
                code="rerank_budget_exhausted",
                reason="精排前预算已耗尽",
                decision_action="reject",
            ),
            "execute_rerank",
        )
    try:
        result = state["context"].toolbox.run(
            TOOL_RERANK,
            state["current_arguments"],
            prior_queries=state.get("queries_used", []),
        )
    except ToolRejected as error:
        _update_last_step(
            state,
            tool_return={
                "status": "rejected",
                "call_id": state["current_call_id"],
                "code": error.code,
            },
        )
        return _leave(
            _shutdown(
                state,
                status="partial",
                code=f"rerank_rejected:{error.code}",
                reason=f"精排被拒绝：{error.message}",
                error=error.message,
                decision_action="reject",
            ),
            "execute_rerank",
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
                code="rerank_failed",
                reason=f"精排执行失败：{message}",
                error=message,
            ),
            "execute_rerank",
        )

    state["rerank_calls"] = int(state.get("rerank_calls", 0)) + 1
    state.setdefault("tool_sequence", []).append(TOOL_RERANK)
    state["rerank_result_count"] = int(result["count"])
    state["dialogue"] = [
        *state["dialogue"],
        _tool_message(state["current_call_id"], result),
    ]
    state.setdefault("tool_result_ids", []).append(state["current_call_id"])
    _tool_return_update(
        state,
        call_id=state["current_call_id"],
        tool_name=TOOL_RERANK,
        query_tokens=state["current_query_tokens"],
        top_k=state["current_top_k"],
        result=result,
    )
    _event(
        state,
        {
            "event": "tool_returned",
            "tool_call_id": state["current_call_id"],
            "tool": TOOL_RERANK,
            "query_tokens": state["current_query_tokens"],
            "count": int(result["count"]),
            "remaining_rerank_budget": _pending_rerank_budget(state),
        },
    )
    return _leave(state, "execute_rerank")


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
        _record_final_reply_status(
            state, reply_status="tool_calls", content=_message_content(response)
        )
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
        _record_final_reply_status(state, reply_status="ok", content=content)
        stop_type = "budget_exhausted_final"
        stop_summary = _stop_summary(state, stop_type=stop_type)
        state["stop_code"] = "budget_exhausted_final"
        state["stop_type"] = stop_type
        state["stop_summary"] = stop_summary
        state["stop_reason"] = stop_summary
        state["status"] = "verified"
        _update_last_step(
            state,
            decision={"action": "stop", "reason": stop_summary, "stop_type": stop_type},
            stop={
                "code": "budget_exhausted_final",
                "stop_type": stop_type,
                "summary": stop_summary,
                "status": "verified",
                "finish_reason": finish_reason,
                "reply_status": "ok",
            },
        )
        _event(
            state,
            {
                "event": "model_final",
                "request": state.get("request_count"),
                "stop_type": stop_type,
                "summary": stop_summary,
            },
        )
        return _leave(state, "request_final")

    _record_final_reply_status(state, reply_status=reply_status, content=content)
    state["stop_type"] = reply_status
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
            and state.get("final_reply_status") == "ok"
            and state.get("stop_code") in {"model_stop", "budget_exhausted_final"}
        ):
            state["status"] = "verified"
        elif int(state.get("search_calls", 0)) == 0:
            state["status"] = "incomplete"
            state.setdefault("stop_code", "no_successful_search")
            state.setdefault("stop_reason", "没有成功的检索")
            state.setdefault("error", state["stop_reason"])
        else:
            state["status"] = "partial"
            state.setdefault("stop_code", "incomplete_workflow")
            state.setdefault("stop_reason", "工作流未形成合法最终回复")
            state.setdefault("error", state["stop_reason"])

    max_requests = int(state.get("max_model_requests", 3))
    max_searches = int(state.get("max_search_calls", 2))
    max_reranks = int(state.get("max_rerank_calls", 1))
    if int(state.get("request_count", 0)) > max_requests:
        state["status"] = "partial"
        state["stop_code"] = "request_budget_exceeded"
        state["stop_reason"] = f"模型请求次数超过 {max_requests}"
        state["error"] = state["stop_reason"]
    if int(state.get("search_calls", 0)) > max_searches:
        state["status"] = "partial"
        state["stop_code"] = "search_budget_exceeded"
        state["stop_reason"] = f"检索次数超过 {max_searches}"
        state["error"] = state["stop_reason"]
    if int(state.get("rerank_calls", 0)) > max_reranks:
        state["status"] = "partial"
        state["stop_code"] = "rerank_budget_exceeded"
        state["stop_reason"] = f"精排次数超过 {max_reranks}"
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
    rerank_calls = int(state.get("rerank_calls", 0))
    request_count = int(state.get("request_count", 0))
    queries_used = [list(query) for query in state.get("queries_used", [])]
    tool_sequence = [str(name) for name in state.get("tool_sequence", [])]
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
    trace_has_model_text = any(
        isinstance(step.get("response"), dict)
        and any(key in step["response"] for key in ("content_preview", "content", "final_reply"))
        for step in state.get("trace_steps", [])
    )
    first_tool = str(state.get("first_tool_name") or "")
    tools_registered = set(context.toolbox.names()) == set(TOOL_NAMES) and all(
        name in TOOL_SCHEMAS for name in TOOL_NAMES
    )
    rerank_hits = [
        hit
        for step in state.get("trace_steps", [])
        for hit in ((step.get("tool_return") or {}).get("rerank_scores") or [])
    ]
    rerank_scores_ok = True
    if rerank_calls:
        rerank_scores_ok = (
            bool(rerank_hits)
            and all(isinstance(hit, (int, float)) for hit in rerank_hits)
            and list(rerank_hits) == sorted(rerank_hits, reverse=True)
        )
    checks = {
        "index_loaded": context.index.num_docs
        == int(state["config"]["search_agent"]["expected_num_docs"]),
        "index_unchanged": state.get("index_sha_after") == context.index_sha256,
        "vector_faiss_backend": context.service.vector.faiss_active
        and context.service.vector.backend_actual == "faiss.IndexFlatIP",
        "tools_registered": tools_registered,
        "tool_choice_ok": bool(first_tool) and first_tool in SEARCH_TOOL_NAMES,
        "query_forms_preserved": bool(
            (state.get("query_normalized_tokens") or context.query_tokens)
            and (state.get("query_original_tokens") or context.original_tokens)
            and (state.get("query_normalization") or context.query_normalization)
        ),
        "native_tool_call": bool(state.get("tool_call_ids")),
        "tool_result_sent": bool(state.get("tool_result_sent")),
        "tool_call_id_roundtrip": tool_roundtrip,
        "return_counts_present": bool(return_counts) and all(count >= 0 for count in return_counts),
        "request_budget_ok": request_count <= max_requests,
        "search_budget_ok": search_calls <= max_searches,
        "rerank_budget_ok": rerank_calls <= max_reranks,
        "second_query_rule_ok": second_query_valid,
        "rerank_scores_ok": rerank_scores_ok,
        "final_reply_ok": state.get("status") == "verified"
        and state.get("final_reply_status") == "ok",
        "no_candidate_dump": not trace_has_results and not trace_has_model_text,
    }
    state["checks"] = checks
    state["summary"] = {
        "status": state.get("status"),
        "driver": state.get("driver", DRIVER_LLM),
        "request_count": request_count,
        "search_calls": search_calls,
        "max_model_requests": max_requests,
        "max_search_calls": max_searches,
        "rerank_calls": rerank_calls,
        "max_rerank_calls": max_reranks,
        "stop_code": state.get("stop_code"),
        "stop_type": state.get("stop_type"),
        "stop_summary": state.get("stop_summary") or state.get("stop_reason", ""),
        "final_reply_status": state.get("final_reply_status", "missing"),
        "final_reply_chars": int(state.get("final_reply_chars", 0)),
        "queries_used": queries_used,
        "tool_sequence": tool_sequence,
        "first_tool": first_tool,
        "second_tool": str(state.get("second_tool_name") or ""),
        "return_counts": return_counts,
        "rerank_result_count": int(state.get("rerank_result_count", 0)),
        "knowledge_calls": int(state.get("knowledge_calls", 0)),
        "knowledge_citations": [
            str(citation)
            for item in state.get("knowledge_results") or []
            for citation in (item.get("citations") or [])
        ][:5],
        "index_sha256_before": context.index_sha256,
        "index_sha256_after": state.get("index_sha_after"),
        "elapsed_s": round(time.perf_counter() - float(state.get("started", time.perf_counter())), 3),
    }
    _update_last_step(
        state,
        stop={
            "code": state.get("stop_code"),
            "stop_type": state.get("stop_type"),
            "summary": state.get("stop_summary") or state.get("stop_reason", ""),
            "status": state.get("status"),
        },
    )
    _event(
        state,
        {
            "event": "finish",
            "status": state.get("status"),
            "driver": state.get("driver", DRIVER_LLM),
            "request_count": request_count,
            "search_calls": search_calls,
            "rerank_calls": rerank_calls,
            "tool_sequence": tool_sequence,
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
    route = state.get("route_after_feedback")
    if route == "execute_second":
        return "execute_second"
    if route == "execute_rerank":
        return "execute_rerank"
    return "finish"


def _route_after_second(state: AgentState) -> str:
    if state.get("status") != "running":
        return "finish"
    return "request_final"


def _route_after_rerank(state: AgentState) -> str:
    if state.get("status") != "running":
        return "finish"
    return "request_final"


def _route_after_first(state: AgentState) -> str:
    if state.get("status") != "running":
        return "finish"
    if state.get("current_tool_name") == TOOL_KNOWLEDGE:
        return "execute_knowledge"
    return "execute_first"


def build_graph(chat_fn: ChatFn):
    graph = StateGraph(AgentState)
    graph.add_node("request_first", lambda state: node_request_first(state, chat_fn))
    graph.add_node("execute_knowledge", node_execute_knowledge)
    graph.add_node("execute_first", node_execute_first)
    graph.add_node("request_feedback", lambda state: node_request_feedback(state, chat_fn))
    graph.add_node("execute_second", node_execute_second)
    graph.add_node("execute_rerank", node_execute_rerank)
    graph.add_node("request_final", lambda state: node_request_final(state, chat_fn))
    graph.add_node("finish", node_finish)
    graph.set_entry_point("request_first")
    graph.add_conditional_edges(
        "request_first",
        _route_after_first,
        {"execute_first": "execute_first", "execute_knowledge": "execute_knowledge", "finish": "finish"},
    )
    graph.add_edge("execute_knowledge", "request_first")
    graph.add_conditional_edges("execute_first", _route("request_feedback"))
    graph.add_conditional_edges("request_feedback", _route_after_feedback)
    graph.add_conditional_edges("execute_second", _route_after_second)
    graph.add_conditional_edges("execute_rerank", _route_after_rerank)
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
    driver: str = DRIVER_LLM,
) -> AgentState:
    start = started_at if started_at is not None else time.perf_counter()
    state: AgentState = {
        "status": "running",
        "stage": "init",
        "nodes": [],
        "events": [
            {
                "event": "start",
                "driver": driver,
                "model": llm_cfg.get("model"),
                "endpoint_host": host_only(str(llm_cfg.get("base_url", ""))),
                "max_model_requests": int(config["search_agent"]["max_model_requests"]),
                "max_search_calls": int(config["search_agent"]["max_search_calls"]),
                "max_rerank_calls": int(config["search_agent"].get("max_rerank_calls", 1)),
                "query_tokens": list(context.query_tokens),
                "original_query_tokens": list(context.original_tokens),
                "session_id": context.session_id,
            }
        ],
        "started": start,
        "deadline": start + float(config["llm"]["total_budget_s"]),
        "config": dict(config),
        "context": context,
        "llm_config": dict(llm_cfg),
        "driver": driver,
        "query_tokens": list(context.query_tokens),
        "query_original_tokens": list(context.original_tokens),
        "query_normalized_tokens": list(context.query_tokens),
        "query_normalization": context.query_normalization,
        "session_id": context.session_id,
        "request_count": 0,
        "search_calls": 0,
        "max_search_calls": int(config["search_agent"]["max_search_calls"]),
        "max_model_requests": int(config["search_agent"]["max_model_requests"]),
        "rerank_calls": 0,
        "max_rerank_calls": int(config["search_agent"].get("max_rerank_calls", 1)),
        "knowledge_calls": 0,
        "max_knowledge_calls": MAX_KNOWLEDGE_CALLS,
        "knowledge_exchange": [],
        "knowledge_results": [],
        "response_ids": {},
        "trace_steps": [],
        "dialogue": [],
        "queries_used": [],
        "tool_call_ids": [],
        "tool_result_ids": [],
        "tool_sequence": [],
        "tool_result_sent": False,
        "route_after_feedback": "finish",
        "first_tool_name": "",
        "second_tool_name": "",
        "first_result_count": 0,
        "second_result_count": 0,
        "rerank_result_count": 0,
        "final_reply_status": "",
        "final_reply_chars": 0,
        "stop_code": "",
        "stop_type": "",
        "stop_summary": "",
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
        "vector_faiss_backend": False,
        "tools_registered": False,
        "tool_choice_ok": False,
        "query_forms_preserved": False,
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
        "rerank_budget_ok": int(state.get("rerank_calls", 0)) <= int(
            state.get("max_rerank_calls", 1)
        ),
        "second_query_rule_ok": True,
        "rerank_scores_ok": True,
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
        checks["vector_faiss_backend"] = context.service.vector.faiss_active
        checks["tools_registered"] = set(context.toolbox.names()) == set(TOOL_NAMES)
        checks["query_forms_preserved"] = bool(
            context.query_tokens and context.original_tokens
        )
    summary = state.get("summary") or {
        "status": state.get("status"),
        "driver": state.get("driver", DRIVER_LLM),
        "request_count": int(state.get("request_count", 0)),
        "search_calls": int(state.get("search_calls", 0)),
        "max_model_requests": int(state.get("max_model_requests", 3)),
        "max_search_calls": int(state.get("max_search_calls", 2)),
        "rerank_calls": int(state.get("rerank_calls", 0)),
        "max_rerank_calls": int(state.get("max_rerank_calls", 1)),
        "stop_code": state.get("stop_code"),
        "stop_type": state.get("stop_type"),
        "stop_summary": state.get("stop_summary") or state.get("stop_reason", ""),
        "final_reply_status": state.get("final_reply_status", "missing"),
        "final_reply_chars": int(state.get("final_reply_chars", 0)),
        "queries_used": [list(query) for query in state.get("queries_used", [])],
        "tool_sequence": [str(name) for name in state.get("tool_sequence", [])],
        "first_tool": str(state.get("first_tool_name") or ""),
        "second_tool": str(state.get("second_tool_name") or ""),
        "return_counts": [],
        "rerank_result_count": int(state.get("rerank_result_count", 0)),
        "knowledge_calls": int(state.get("knowledge_calls", 0)),
        "knowledge_citations": [
            str(citation)
            for item in state.get("knowledge_results") or []
            for citation in (item.get("citations") or [])
        ][:5],
        "index_sha256_before": context.index_sha256 if context else None,
        "index_sha256_after": state.get("index_sha_after"),
        "elapsed_s": round(elapsed_s, 3),
    }
    state["summary"] = summary
    state["checks"] = checks
    passed = state.get("status") == "verified" and all(bool(value) for value in checks.values())
    model = state.get("llm_config") or {}
    index_identity = context.identity() if context else None
    backend_identity = context.backend_identity() if context else None
    query_payload = context.query_payload() if context else None
    driver = str(state.get("driver") or DRIVER_LLM)
    trace = {
        "format": TRACE_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "driver": driver,
        "status": state.get("status"),
        "query": query_payload,
        "index": index_identity,
        "search_backend": backend_identity,
        "rules": {
            "tools": list(TOOL_NAMES),
            "search_tools": list(SEARCH_TOOL_NAMES),
            "max_results_per_query": int(config["search_agent"]["max_results"]),
            "max_search_calls": int(config["search_agent"]["max_search_calls"]),
            "max_rerank_calls": int(config["search_agent"].get("max_rerank_calls", 1)),
            "max_model_requests": int(config["search_agent"]["max_model_requests"]),
            "first_query": "must_equal_normalized_original",
            "second_query": "non_empty_true_subset_of_original_and_different",
            "rerank_query": "must_equal_a_previously_searched_query",
            "candidates_persisted": False,
            "model_free_text_persisted": False,
        },
        "steps": state.get("trace_steps", []),
        "checks": checks,
        "summary": summary,
        "candidates_persisted": False,
        "model_free_text_persisted": False,
        "persistence_note": (
            "候选只通过工具消息回传模型；落盘只保留程序生成的调用 ID、查询 token、"
            "返回数量、精排分数、停止类型、预算和状态。"
        ),
        "error": state.get("error"),
    }
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "phase": config["search_agent"]["phase"],
        "status": state.get("status"),
        "passed": passed,
        "driver": driver,
        "model": {
            "name": model.get("model"),
            "endpoint_host": host_only(str(model.get("base_url", ""))) if model else None,
        },
        "index": index_identity,
        "search_backend": backend_identity,
        "query": query_payload,
        "budget": {
            "request_count": int(state.get("request_count", 0)),
            "max_model_requests": int(config["search_agent"]["max_model_requests"]),
            "search_calls": int(state.get("search_calls", 0)),
            "max_search_calls": int(config["search_agent"]["max_search_calls"]),
            "rerank_calls": int(state.get("rerank_calls", 0)),
            "max_rerank_calls": int(config["search_agent"].get("max_rerank_calls", 1)),
            "elapsed_s": round(elapsed_s, 3),
            "total_budget_s": float(config["llm"]["total_budget_s"]),
        },
        "steps": state.get("trace_steps", []),
        "checks": checks,
        "summary": summary,
        "candidates_persisted": False,
        "model_free_text_persisted": False,
        "persistence_note": (
            "候选只通过工具消息回传模型；落盘只保留程序生成的调用 ID、查询 token、"
            "返回数量、精排分数、停止类型、预算和状态。"
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
        "format": "recsys.search_agent.run_meta.v2",
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "started_at": started_wall,
        "elapsed_s": round(elapsed_s, 3),
        "driver": driver,
        "status": state.get("status"),
        "passed": passed,
        "model": metrics["model"],
        "index": index_identity,
        "query": query_payload,
        "output_dir": _display_path(run_dir),
        "trace_file": _display_path(run_dir / "trace.json"),
        "metrics_file": _display_path(run_dir / "metrics.json"),
        "candidates_persisted": False,
        "model_free_text_persisted": False,
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
        "final_reply_status": "missing",
        "final_reply_chars": 0,
        "stop_code": "blocked",
        "stop_type": "blocked",
        "stop_summary": reason,
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
            "stop_type": "blocked",
            "stop_summary": reason,
            "final_reply_status": "missing",
            "final_reply_chars": 0,
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
# 本地替身驱动（不联网；检索 / 精排 / 落盘执行真实代码）
# ---------------------------------------------------------------------------


SEARCH_MODE_PLANS = {
    "bm25": "bm25_stop",
    "vector": "vector_stop",
    "hybrid": "hybrid_stop",
    "rerank": "hybrid_rerank",
}

SCRIPTED_PLANS: dict[str, str] = {
    "hybrid_rerank": "混合检索 → 精排 → 最终回复",
    "bm25_stop": "BM25 检索 → 停止",
    "vector_stop": "FAISS 向量检索 → 停止",
    "two_search": "BM25 检索 → 向量子集二次检索 → 最终回复",
    "hybrid_stop": "混合检索 → 停止（传无匹配 token 可验空结果分支）",
    "rerank_first": "首轮直接请求精排（失败分支，应被拒绝）",
    **{f"knowledge_{mode}": f"知识库检索 → {mode}（需启用 [knowledge]）" for mode in SEARCH_MODE_PLANS},
}


def _scripted_responses(context: SearchContext, plan: str) -> list[Any]:
    tokens = list(context.query_tokens)
    subset = list(tokens[:-1]) if len(tokens) > 1 else []
    if plan == "hybrid_rerank":
        return [
            _tool_response("call-hybrid", tokens, tool=TOOL_HYBRID),
            _tool_response("call-rerank", tokens, tool=TOOL_RERANK),
            _text_response("混合检索与精排结果已返回，停止。"),
        ]
    if plan == "bm25_stop":
        return [
            _tool_response("call-bm25", tokens, tool=TOOL_NAME),
            _text_response("BM25 结果已返回，停止。"),
        ]
    if plan == "vector_stop":
        return [
            _tool_response("call-vector", tokens, tool=TOOL_VECTOR),
            _text_response("向量检索结果已返回，停止。"),
        ]
    if plan == "two_search":
        if not subset:
            raise AgentBlocked("two_search 计划需要至少 2 个查询 token")
        return [
            _tool_response("call-bm25", tokens, tool=TOOL_NAME),
            _tool_response("call-vector-subset", subset, tool=TOOL_VECTOR),
            _text_response("二次检索结果已返回，预算用尽，停止。"),
        ]
    if plan == "hybrid_stop":
        return [
            _tool_response("call-hybrid", tokens, tool=TOOL_HYBRID),
            _text_response("混合检索结果已返回，停止。"),
        ]
    if plan == "rerank_first":
        return [_tool_response("call-rerank-first", tokens, tool=TOOL_RERANK)]
    if plan.startswith("knowledge_"):
        mode = plan.removeprefix("knowledge_")
        return [
            _knowledge_response("call-knowledge", "实验记忆 指纹 复用 预算", 2),
            *_scripted_responses(context, SEARCH_MODE_PLANS[mode]),
        ]
    raise AgentBlocked(f"未知的替身计划 {plan!r}；可选：{sorted(SCRIPTED_PLANS)}")


def run_scripted(
    config: Mapping[str, Any],
    *,
    plan: str = "hybrid_rerank",
    tokens: Sequence[int] | None = None,
    session_id: str | None = None,
    output_root: Path,
) -> dict[str, Any]:
    """用本地脚本替身驱动 Agent 决策；工具执行、索引读取和落盘都是真实代码。"""
    if plan not in SCRIPTED_PLANS:
        raise AgentBlocked(f"未知的替身计划 {plan!r}；可选：{sorted(SCRIPTED_PLANS)}")
    started_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    started = time.perf_counter()
    run_dir = Path(output_root) / f"scripted-{plan}"
    context = load_context(config, tokens=tokens, session_id=session_id)
    responses = _scripted_responses(context, plan)
    state = run_workflow(
        context,
        config,
        llm_cfg={
            "model": "scripted-driver",
            "base_url": "http://127.0.0.1",
            "api_key": "not-used",
        },
        chat_fn=ScriptedChat(responses),
        started_at=started,
        driver=DRIVER_SCRIPTED,
    )
    state["elapsed_s"] = round(time.perf_counter() - started, 3)
    return _finalize(
        state,
        run_dir=run_dir,
        config=config,
        context=context,
        command=f"python -m recsys.experiments.search_agent --scripted-plan {plan}",
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


def _tool_response(
    call_id: str,
    tokens: Sequence[int],
    top_k: int = 5,
    tool: str = TOOL_NAME,
) -> dict[str, Any]:
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
                                "name": tool,
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


def _knowledge_response(call_id: str, query: str, top_k: int = 2) -> dict[str, Any]:
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
                                "name": TOOL_KNOWLEDGE,
                                "arguments": json.dumps(
                                    {"query": query, "top_k": top_k}, ensure_ascii=False
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
    *,
    driver: str = DRIVER_SCRIPTED,
) -> tuple[AgentState, ScriptedChat]:
    chat = ScriptedChat(responses)
    final = run_workflow(
        context,
        config,
        llm_cfg={"model": "fake-model", "base_url": "http://127.0.0.1", "api_key": "fake-key"},
        chat_fn=chat,
        driver=driver,
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
        assert final["final_reply_status"] == "ok"
        assert final["stop_summary"]
        return "一次检索 + 模型主动停止 → verified；2 次请求 / 1 次检索"

    def two_query_then_final() -> str:
        second = list(context.query_tokens[:-1])
        assert second and second != list(context.query_tokens)
        final, _ = _fake_run(
            context,
            config,
            [
                _tool_response("call-first", context.query_tokens),
                _tool_response("call-second", second, tool=TOOL_VECTOR),
                _text_response("第二次检索已返回结果，预算用尽，停止。"),
            ],
        )
        assert final["status"] == "verified", final["status"]
        assert final["summary"]["search_calls"] == 2, final["summary"]
        assert final["summary"]["request_count"] == 3, final["summary"]
        assert final["summary"]["queries_used"][1] == second
        assert final["summary"]["tool_sequence"] == [TOOL_NAME, TOOL_VECTOR]
        return "两次检索（BM25 → 向量）+ 最终回复 → verified；3 次请求 / 2 次检索"

    def hybrid_then_rerank() -> str:
        final, _ = _fake_run(
            context,
            config,
            [
                _tool_response("call-hybrid", context.query_tokens, tool=TOOL_HYBRID),
                _tool_response("call-rerank", context.query_tokens, tool=TOOL_RERANK),
                _text_response("混合检索与精排结果已返回，停止。"),
            ],
        )
        assert final["status"] == "verified", final["status"]
        assert final["summary"]["search_calls"] == 1, final["summary"]
        assert final["summary"]["rerank_calls"] == 1, final["summary"]
        assert final["summary"]["request_count"] == 3, final["summary"]
        assert final["summary"]["tool_sequence"] == [TOOL_HYBRID, TOOL_RERANK]
        assert final["checks"]["rerank_scores_ok"] is True
        assert final["checks"]["no_candidate_dump"] is True
        return "混合检索 → 精排 → 最终回复 → verified；3 请求 / 1 检索 / 1 精排"

    def rerank_before_search() -> str:
        final, _ = _fake_run(
            context,
            config,
            [_tool_response("call-rerank-first", context.query_tokens, tool=TOOL_RERANK)],
        )
        assert final["status"] == "incomplete", final["status"]
        assert final["summary"]["search_calls"] == 0, final["summary"]
        assert "tool_not_allowed" in str(final.get("stop_code")), final.get("stop_code")
        return "首轮直接请求精排 → 拒绝，incomplete"

    def rerank_unknown_query() -> str:
        bad = [context.query_tokens[0] + 1]
        final, _ = _fake_run(
            context,
            config,
            [
                _tool_response("call-first", context.query_tokens),
                _tool_response("call-bad-rerank", bad, tool=TOOL_RERANK),
            ],
        )
        assert final["status"] == "partial", final["status"]
        assert final["summary"]["rerank_calls"] == 0, final["summary"]
        assert "rerank_query_mismatch" in str(final.get("stop_code")), final.get("stop_code")
        return "精排使用未检索过的查询 → 拒绝，保留检索结果"

    def hybrid_provenance() -> str:
        result = context.toolbox.run(
            TOOL_HYBRID, {"query_tokens": context.query_tokens, "top_k": 5}
        )
        assert result["count"] > 0, result
        ids = [hit["item_id"] for hit in result["results"]]
        assert len(ids) == len(set(ids)), ids
        for hit in result["results"]:
            assert hit["sources"], hit
            assert {source["source"] for source in hit["sources"]} <= {SOURCE_BM25, SOURCE_VECTOR}
        return "混合检索按 item_id 去重，每条保留 bm25/vector 来源、名次与分数"

    def empty_result_is_empty() -> str:
        no_match = [10**12]
        bm25 = context.toolbox.run(TOOL_NAME, {"query_tokens": no_match, "top_k": 5})
        hybrid = context.toolbox.run(TOOL_HYBRID, {"query_tokens": no_match, "top_k": 5})
        assert bm25["count"] == 0, bm25
        assert hybrid["count"] == 0, hybrid
        return "无匹配 token：BM25 与混合检索都返回 0 条（空结果分支）"

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

    def model_text_not_persisted() -> str:
        # 普通句子列出两个候选 ID，正文没有 item_id / 结果列表等标记。
        final, _ = _fake_run(
            context,
            config,
            [
                _tool_response("call-first", context.query_tokens),
                _text_response("相关性足够，停止。返回的物品编号是 987654321、987654322。"),
            ],
        )
        assert final["status"] == "verified", final["status"]
        assert final["final_reply_status"] == "ok", final["final_reply_status"]
        assert final["final_reply_chars"] > 0
        assert final["stop_summary"]
        assert "final_reply" not in final
        persisted = json.dumps(
            {"steps": final["trace_steps"], "summary": final["summary"]},
            ensure_ascii=False,
        )
        assert "987654321" not in persisted
        assert "987654322" not in persisted
        assert "相关性足够" not in persisted
        assert "content_preview" not in final["trace_steps"][1]["response"]
        assert final["checks"]["no_candidate_dump"] is True
        return "模型最终回复只在内存中判断完成状态；轨迹/状态不含正文或候选 ID"

    _check(checks, "one_query_stop", one_query_stop)
    _check(checks, "two_query_then_final", two_query_then_final)
    _check(checks, "hybrid_then_rerank", hybrid_then_rerank)
    _check(checks, "rerank_before_search", rerank_before_search)
    _check(checks, "rerank_unknown_query", rerank_unknown_query)
    _check(checks, "hybrid_provenance", hybrid_provenance)
    _check(checks, "empty_result_is_empty", empty_result_is_empty)
    _check(checks, "first_query_mismatch", first_query_mismatch)
    _check(checks, "second_not_subset", second_not_subset)
    _check(checks, "second_duplicate", second_duplicate)
    _check(checks, "second_empty", second_empty)
    _check(checks, "final_tool_call_rejected", final_tool_call_rejected)
    _check(checks, "truncated_stop_rejected", truncated_stop_rejected)
    _check(checks, "tool_call_id_roundtrip", tool_call_id_roundtrip)
    _check(checks, "budget_limits", budget_limits)
    _check(checks, "no_candidate_dump", no_candidate_dump)
    _check(checks, "model_text_not_persisted", model_text_not_persisted)

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
# 本地替身回归（不联网，不写旧证据）
# ---------------------------------------------------------------------------


class CandidateRepeatingChat:
    """替身模型：首轮原查询，次轮用普通句子复述工具返回的候选 ID。"""

    def __init__(self, call_id: str) -> None:
        self.call_id = call_id
        self.calls: list[dict[str, Any]] = []
        self.final_text = ""

    def __call__(self, config, messages, *, tools, timeout):
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
                "timeout": timeout,
            }
        )
        if len(self.calls) == 1:
            user = next(
                (item for item in reversed(messages) if item.get("role") == "user"),
                None,
            )
            payload = json.loads(user["content"]) if user else {}
            original = payload.get("original_query_tokens") or []
            return _tool_response(self.call_id, original)
        tool_messages = [item for item in messages if item.get("role") == "tool"]
        candidate_ids: list[str] = []
        if tool_messages:
            try:
                payload = json.loads(tool_messages[-1]["content"])
            except (KeyError, TypeError, json.JSONDecodeError):
                payload = {}
            candidate_ids = [
                str(item.get("item_id"))
                for item in payload.get("results", [])
                if item.get("item_id") is not None
            ]
        if candidate_ids:
            self.final_text = (
                "相关性足够，停止。返回的物品编号是 "
                + "、".join(candidate_ids)
                + "。"
            )
        else:
            self.final_text = "没有返回物品，停止。"
        return _text_response(self.final_text)


def _sample_query_tokens(context: SearchContext) -> set[int]:
    sample = json.loads(context.sample_path.read_text(encoding="utf-8"))
    return {
        int(token)
        for session in sample.get("sessions", [])
        for token in (session.get("keyword") or [])
    }


def _find_out_of_sample_hit_token(context: SearchContext) -> int:
    sample_tokens = _sample_query_tokens(context)
    index_tokens = {
        token for item in context.index.items for token in item.caption_tokens
    }
    for token in sorted(index_tokens - sample_tokens):
        hits = context.index.search([token], top_k=5)
        if hits and all(hit.item_id != token for hit in hits):
            return token
    raise AgentBlocked("找不到「不在样本查询中且能命中索引」的合法 token")


def _read_case_artifacts(case_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    trace = json.loads((case_dir / "trace.json").read_text(encoding="utf-8"))
    metrics = json.loads((case_dir / "metrics.json").read_text(encoding="utf-8"))
    run_meta = json.loads((case_dir / "run_meta.json").read_text(encoding="utf-8"))
    serialized = "\n".join(
        (case_dir / name).read_text(encoding="utf-8")
        for name in ("trace.json", "metrics.json", "run_meta.json")
    )
    return trace, metrics, run_meta, serialized


def _run_regression_case(
    name: str,
    context: SearchContext,
    config: Mapping[str, Any],
    root: Path,
    *,
    expected_session_id: str | None,
    expected_source: str,
    expected_nonzero: bool,
) -> list[dict[str, Any]]:
    case_dir = root / name
    chat = CandidateRepeatingChat(f"call-regression-{name}")
    started = time.perf_counter()
    state = run_workflow(
        context,
        config,
        llm_cfg={"model": "fake-model", "base_url": "http://127.0.0.1", "api_key": "fake-key"},
        chat_fn=chat,
        started_at=started,
        driver=DRIVER_SCRIPTED,
    )
    _finalize(
        state,
        run_dir=case_dir,
        config=config,
        context=context,
        command=f"regression:{name}",
        started_wall=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        elapsed_s=time.perf_counter() - started,
    )
    trace, metrics, _run_meta, serialized = _read_case_artifacts(case_dir)

    checks: list[dict[str, Any]] = []

    def add(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"id": f"{name}.{check_id}", "status": "pass" if passed else "fail", "detail": detail})

    add("verified", state.get("status") == "verified", f"status={state.get('status')}")
    add(
        "query_session",
        trace.get("query", {}).get("session_id") == expected_session_id,
        f"session_id={trace.get('query', {}).get('session_id')!r}",
    )
    add(
        "query_source",
        trace.get("query", {}).get("source") == expected_source,
        f"source={trace.get('query', {}).get('source')!r}",
    )
    first_step = (trace.get("steps") or [{}])[0]
    tool_call = first_step.get("tool_call") or {}
    tool_return = first_step.get("tool_return") or {}
    add(
        "first_query_original",
        tool_call.get("arguments", {}).get("query_tokens") == list(context.query_tokens),
        "首查参数的 query_tokens 与输入一致",
    )
    add(
        "tool_call_id",
        tool_call.get("id") == f"call-regression-{name}",
        f"tool_call_id={tool_call.get('id')!r}",
    )
    count = int(tool_return.get("count", -1))
    add("return_count", count == int(state.get("first_result_count", -1)), f"count={count}")
    add(
        "return_nonzero",
        (count > 0) if expected_nonzero else (count == 0),
        f"expected_nonzero={expected_nonzero}, count={count}",
    )
    add(
        "stop_status",
        (trace.get("summary") or {}).get("stop_code") == "model_stop",
        f"stop_code={(trace.get('summary') or {}).get('stop_code')!r}",
    )
    add(
        "stop_summary",
        bool((trace.get("summary") or {}).get("stop_summary")),
        "停止摘要是程序生成的",
    )
    add(
        "no_model_text_rule",
        trace.get("rules", {}).get("model_free_text_persisted") is False,
        "trace rules 标记 model_free_text_persisted=false",
    )
    second_step = (trace.get("steps") or [{}, {}])[1]
    response = second_step.get("response") or {}
    add(
        "no_content_preview",
        "content_preview" not in response and "content" not in response,
        "轨迹 response 不含模型正文",
    )
    raw_text_present = bool(chat.final_text) and chat.final_text in serialized
    add("raw_reply_absent", not raw_text_present, "模型原始回复不在序列化工件中")
    # 候选 ID 只用于断言，不写入 summary/报告。
    tool_messages = [
        item for item in (chat.calls[1]["messages"] if len(chat.calls) > 1 else [])
        if item.get("role") == "tool"
    ]
    candidate_ids: list[str] = []
    if tool_messages:
        payload = json.loads(tool_messages[-1]["content"])
        candidate_ids = [str(item.get("item_id")) for item in payload.get("results", [])]
    leaked = [item_id for item_id in candidate_ids if item_id in serialized]
    add(
        "candidate_ids_absent",
        not leaked,
        f"候选 {len(candidate_ids)} 个，落盘泄漏 {len(leaked)} 个",
    )
    second_context = second_step.get("tool_result_in_context") or {}
    add(
        "tool_result_sent_to_model",
        second_context.get("sent") is True
        and second_context.get("call_id") == tool_call.get("id")
        and int(second_context.get("count", -1)) == count,
        "第二次请求通过同一 tool_call_id 收到工具返回",
    )
    return checks


def run_local_regression(
    config: Mapping[str, Any],
    *,
    output_root: Path,
    tokens: Sequence[int] | None = None,
) -> dict[str, Any]:
    """本地替身回归，不调用 LLM；写入 artifacts/ 下的独立目录。"""
    root = output_root / f"regression-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    root.mkdir(parents=True, exist_ok=True)

    old_evidence_paths = {
        "index": PROJECT_ROOT / "artifacts/search/kuasar_bm25/index.json",
        "sample": PROJECT_ROOT / "artifacts/search/kuasar_bm25/sample.json",
        "real_trace": PROJECT_ROOT / "experiments/2026-11-09-kuasar-search-agent/run-20261109T080657Z-19556/trace.json",
        "real_metrics": PROJECT_ROOT / "experiments/2026-11-09-kuasar-search-agent/run-20261109T080657Z-19556/metrics.json",
        "real_run_meta": PROJECT_ROOT / "experiments/2026-11-09-kuasar-search-agent/run-20261109T080657Z-19556/run_meta.json",
        "old_self_check": PROJECT_ROOT / "experiments/2026-11-09-kuasar-search-agent/self_check.json",
    }
    old_before = {name: _sha256_file(path) for name, path in old_evidence_paths.items()}

    self_check = run_self_check(config, output_root=root / "self_check")
    checks: list[dict[str, Any]] = [
        {
            "id": "self_check",
            "status": "pass" if self_check.get("passed") else "fail",
            "detail": f"{self_check.get('total')} 项本地自检",
        }
    ]

    matched_context = load_context(config, tokens=None, session_id=None)
    checks.extend(
        _run_regression_case(
            "matched_session",
            matched_context,
            config,
            root,
            expected_session_id=str(config["search_agent"]["default_session_id"]),
            expected_source="config_default_session",
            expected_nonzero=True,
        )
    )

    if tokens is None:
        out_tokens = [_find_out_of_sample_hit_token(matched_context)]
    else:
        out_tokens = list(tokens)
    out_context = load_context(config, tokens=out_tokens, session_id=None)
    if out_context.session_id is not None:
        raise AgentBlocked("回归要求样本外查询，但输入 token 命中了样本会话")
    checks.extend(
        _run_regression_case(
            "out_of_sample_hit",
            out_context,
            config,
            root,
            expected_session_id=None,
            expected_source="cli_tokens_out_of_sample",
            expected_nonzero=True,
        )
    )

    no_match_context = load_context(config, tokens=[10**12], session_id=None)
    checks.extend(
        _run_regression_case(
            "out_of_sample_no_match",
            no_match_context,
            config,
            root,
            expected_session_id=None,
            expected_source="cli_tokens_out_of_sample",
            expected_nonzero=False,
        )
    )

    try:
        load_context(config, tokens=out_tokens, session_id=str(config["search_agent"]["default_session_id"]))
    except AgentBlocked:
        checks.append(
            {
                "id": "session_token_mismatch.blocked",
                "status": "pass",
                "detail": "--session-id 与 --tokens 不一致时被拒绝",
            }
        )
    else:
        checks.append(
            {
                "id": "session_token_mismatch.blocked",
                "status": "fail",
                "detail": "--session-id 与 --tokens 不一致时未拒绝",
            }
        )

    old_after = {name: _sha256_file(path) for name, path in old_evidence_paths.items()}
    old_unchanged = old_before == old_after
    checks.append(
        {
            "id": "old_evidence.unchanged",
            "status": "pass" if old_unchanged else "fail",
            "detail": "索引/样本/旧真实工件/旧 self_check 哈希前后一致",
        }
    )

    passed = all(item["status"] == "pass" for item in checks)
    summary = {
        "format": "recsys.search_agent.regression.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_dir": _display_path(root),
        "passed": passed,
        "checks": checks,
        "cases": {
            "matched_session": {
                "status": "pass" if all(c["status"] == "pass" for c in checks if c["id"].startswith("matched_session.")) else "fail",
                "session_id": str(config["search_agent"]["default_session_id"]),
            },
            "out_of_sample_hit": {
                "status": "pass" if all(c["status"] == "pass" for c in checks if c["id"].startswith("out_of_sample_hit.")) else "fail",
                "session_id": None,
                "query_token_count": len(out_tokens),
            },
            "out_of_sample_no_match": {
                "status": "pass" if all(c["status"] == "pass" for c in checks if c["id"].startswith("out_of_sample_no_match.")) else "fail",
                "session_id": None,
            },
        },
        "self_check": {
            "passed": bool(self_check.get("passed")),
            "total": self_check.get("total"),
        },
        "old_evidence": {
            "unchanged": old_unchanged,
            "hashes": old_after,
        },
        "scope": {
            "real_llm_requests": 0,
            "index_rebuild": 0,
            "real_smoke_rerun": 0,
            "candidate_details_persisted": False,
        },
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


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
    parser.add_argument(
        "--regression",
        action="store_true",
        help="本地替身回归：候选不落盘、样本外查询、旧证据不变；不调用真实 LLM",
    )
    parser.add_argument(
        "--scripted-plan",
        choices=sorted(SCRIPTED_PLANS),
        default=None,
        help=(
            "本地替身驱动一次完整工作流（不联网）：决策由脚本给出，"
            "检索/精排/索引读取/落盘执行真实代码"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    output_root = (
        _resolve_path(args.output_root)
        if args.output_root is not None
        else (
            DEFAULT_REGRESSION_ROOT
            if args.regression
            else (
                PROJECT_ROOT / "artifacts" / "search_agent_scripted"
                if args.scripted_plan
                else (
                    PROJECT_ROOT / "artifacts" / "search_agent_self_check"
                    if args.self_check
                    else _resolve_path(config["search_agent"]["output_root"])
                )
            )
        )
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
    if args.scripted_plan:
        try:
            result = run_scripted(
                config,
                plan=args.scripted_plan,
                tokens=tokens,
                session_id=args.session_id,
                output_root=output_root,
            )
        except (AgentBlocked, OSError, ValueError, json.JSONDecodeError) as error:
            print(f"[scripted blocked] {_safe_error(error)}", file=sys.stderr)
            return 2
        metrics = result["metrics"]
        print(
            json.dumps(
                {
                    "driver": metrics["driver"],
                    "status": metrics["status"],
                    "passed": metrics["passed"],
                    "tool_sequence": metrics["summary"].get("tool_sequence"),
                    "request_count": metrics["budget"]["request_count"],
                    "search_calls": metrics["budget"]["search_calls"],
                    "rerank_calls": metrics["budget"]["rerank_calls"],
                    "stop_code": metrics["summary"].get("stop_code"),
                    "stop_summary": metrics["summary"].get("stop_summary"),
                    "run_dir": metrics["run"]["output_dir"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if metrics["passed"] else 2
    if args.regression:
        try:
            result = run_local_regression(config, output_root=output_root, tokens=tokens)
        except (AgentBlocked, OSError, ValueError, json.JSONDecodeError) as error:
            print(f"[regression blocked] {_safe_error(error)}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "passed": result["passed"],
                    "cases": result["cases"],
                    "self_check": result["self_check"],
                    "old_evidence_unchanged": result["old_evidence"]["unchanged"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        for item in result["checks"]:
            print(f"[regression] {item['id']}: {item['status']} — {item['detail']}")
        return 0 if result["passed"] else 1

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
                "stop_type": metrics["summary"].get("stop_type"),
                "stop_summary": metrics["summary"].get("stop_summary"),
                "final_reply_status": metrics["summary"].get("final_reply_status"),
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
