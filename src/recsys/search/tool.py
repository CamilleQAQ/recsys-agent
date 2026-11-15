"""Search Agent 的搜索工具：BM25、FAISS 向量、混合检索与精排。

``SearchBM25Tool`` 是最早的窄封装，行为保持不变（兼容旧入口）；
``SearchToolbox`` 注册四个工具并统一校验参数：

- ``search_bm25``：关键词检索，直接调用 ``BM25Index.search``；
- ``search_vector``：FAISS 向量召回（查询与文档同一套 token 编码）；
- ``search_hybrid``：BM25 + 向量按 item_id 去重后 RRF 融合，保留来源/名次/分数；
- ``search_rerank``：对融合候选池重新打分，返回最终 Top-K。

工具只读已加载的索引，不加载原始 CSV、不重建索引、不写盘；候选明细只通过
tool 消息回传模型，落盘策略由 Search Agent 统一控制。
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from recsys.search.bm25 import BM25Index, SearchHit
from recsys.search.service import (
    MODE_BM25,
    MODE_HYBRID,
    MODE_RERANK,
    MODE_VECTOR,
    SearchService,
)

TOOL_NAME = "search_bm25"
TOOL_VECTOR = "search_vector"
TOOL_HYBRID = "search_hybrid"
TOOL_RERANK = "search_rerank"
SEARCH_TOOL_NAMES: tuple[str, ...] = (TOOL_NAME, TOOL_VECTOR, TOOL_HYBRID)
TOOL_NAMES: tuple[str, ...] = (*SEARCH_TOOL_NAMES, TOOL_RERANK)
DEFAULT_TOP_K = 5
MAX_TOP_K = 5
MAX_QUERY_TOKENS = 64

TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "在 KuaiSAR 小样本 BM25 索引中检索短视频。返回最多 5 条结果，"
            "每条包含 item_id、BM25 score 和可读类目。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_tokens": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 1,
                    "maxItems": MAX_QUERY_TOKENS,
                    "description": (
                        "KuaiSAR token ID。第一次必须原样等于 original_query_tokens；"
                        "第二次必须是原查询的非空真子集（至少去掉一个 token）。"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_TOP_K,
                    "default": DEFAULT_TOP_K,
                    "description": "返回条数，最多 5。",
                },
            },
            "required": ["query_tokens"],
            "additionalProperties": False,
        },
    },
}


class ToolRejected(RuntimeError):
    """工具调用未通过参数/预算/重复检查。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def normalize_tokens(values: Any, *, max_tokens: int = MAX_QUERY_TOKENS) -> tuple[int, ...]:
    """把模型参数里的 query_tokens 规范成非空正整数元组。"""
    if not isinstance(values, (list, tuple)):
        raise ToolRejected("query_tokens_not_list", "query_tokens 必须是整数数组")
    tokens: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolRejected("query_token_not_integer", "query_tokens 只能包含整数")
        if value <= 0:
            raise ToolRejected("query_token_out_of_range", "query_tokens 必须是正整数")
        tokens.append(value)
    if not tokens:
        raise ToolRejected("empty_query", "query_tokens 不能为空")
    if len(tokens) > max_tokens:
        raise ToolRejected("query_too_long", f"query_tokens 最多 {max_tokens} 个")
    return tuple(tokens)


def normalize_top_k(value: Any) -> int:
    if value is None:
        return DEFAULT_TOP_K
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolRejected("top_k_not_integer", "top_k 必须是整数")
    if value < 1:
        raise ToolRejected("top_k_out_of_range", "top_k 必须 >= 1")
    return min(value, MAX_TOP_K)


def validate_first_query(tokens: Sequence[int], original: Sequence[int]) -> None:
    if list(tokens) != list(original):
        raise ToolRejected(
            "first_query_mismatch",
            "第一次查询必须与 original_query_tokens 完全一致（顺序也不能变）",
        )


def validate_second_query(
    tokens: Sequence[int],
    original: Sequence[int],
    previous_queries: Iterable[Sequence[int]],
) -> None:
    query = list(tokens)
    original_list = list(original)
    if not query:
        raise ToolRejected("empty_second_query", "第二次查询不能为空")
    if query in [list(previous) for previous in previous_queries]:
        raise ToolRejected("duplicate_query", "第二次查询不能与之前查询相同")
    original_counts = Counter(original_list)
    query_counts = Counter(query)
    if query_counts - original_counts:
        raise ToolRejected(
            "second_query_not_subset",
            "第二次查询只能使用 original_query_tokens 中已有的 token",
        )
    if query_counts == original_counts:
        raise ToolRejected(
            "second_query_not_reduced",
            "第二次查询必须从原查询中去掉至少一个 token（不能只是重排）",
        )


class SearchBM25Tool:
    """对现有 ``BM25Index`` 的窄封装。"""

    def __init__(self, index: BM25Index, *, max_results: int = MAX_TOP_K) -> None:
        self._index = index
        self._max_results = min(max(1, int(max_results)), MAX_TOP_K)

    @property
    def index(self) -> BM25Index:
        return self._index

    @property
    def max_results(self) -> int:
        return self._max_results

    def search(self, query_tokens: Sequence[int], top_k: Any = None) -> dict[str, Any]:
        """直接调用 BM25 接口；返回 JSON 友好的结果（最多 5 条）。"""
        tokens = normalize_tokens(query_tokens)
        requested_top_k = normalize_top_k(top_k)
        effective_top_k = min(requested_top_k, self._max_results)
        hits: list[SearchHit] = self._index.search(tokens, top_k=effective_top_k)
        return {
            "tool": TOOL_NAME,
            "query_tokens": list(tokens),
            "requested_top_k": requested_top_k,
            "top_k": effective_top_k,
            "count": len(hits),
            "results": [
                {
                    "rank": hit.rank,
                    "item_id": hit.item_id,
                    "score": round(float(hit.score), 6),
                    "category": hit.category,
                }
                for hit in hits
            ],
        }


# ---------------------------------------------------------------------------
# 四个搜索工具：schema 与统一调度
# ---------------------------------------------------------------------------


def _search_tool_schema(name: str, description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query_tokens": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                        "maxItems": MAX_QUERY_TOKENS,
                        "description": (
                            "KuaiSAR 查询 token ID。第一次必须原样等于 original_query_tokens；"
                            "第二次必须是原查询的非空真子集（至少去掉一个 token）。"
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_TOP_K,
                        "default": DEFAULT_TOP_K,
                        "description": "返回条数，最多 5。",
                    },
                },
                "required": ["query_tokens"],
                "additionalProperties": False,
            },
        },
    }


TOOL_VECTOR_SCHEMA: dict[str, Any] = _search_tool_schema(
    TOOL_VECTOR,
    "在 KuaiSAR 小样本的 FAISS 向量索引里做语义相似度召回（查询与视频描述使用同一套 "
    "整数 token 编码）。返回最多 5 条 item_id、余弦分数与可读类目。",
)
TOOL_HYBRID_SCHEMA: dict[str, Any] = _search_tool_schema(
    TOOL_HYBRID,
    "混合检索：BM25 与向量召回各取候选，按物品 ID 去重后用 RRF 融合。"
    "返回最多 5 条 item_id、融合分、可读类目，以及每条来自哪一路、该路名次和分数。",
)
TOOL_RERANK_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_RERANK,
        "description": (
            "对混合检索候选池重新打分并返回最终 Top-K（按 token 覆盖率与融合先验加权）。"
            "query_tokens 必须等于本次任务里已经检索过的某个查询。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_tokens": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 1,
                    "maxItems": MAX_QUERY_TOKENS,
                    "description": "要精排的查询 token；必须与之前某次检索的 query_tokens 相同。",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_TOP_K,
                    "default": DEFAULT_TOP_K,
                    "description": "最终返回条数，最多 5。",
                },
            },
            "required": ["query_tokens"],
            "additionalProperties": False,
        },
    },
}

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    TOOL_NAME: TOOL_SCHEMA,
    TOOL_VECTOR: TOOL_VECTOR_SCHEMA,
    TOOL_HYBRID: TOOL_HYBRID_SCHEMA,
    TOOL_RERANK: TOOL_RERANK_SCHEMA,
}

_MODE_BY_TOOL = {
    TOOL_NAME: MODE_BM25,
    TOOL_VECTOR: MODE_VECTOR,
    TOOL_HYBRID: MODE_HYBRID,
    TOOL_RERANK: MODE_RERANK,
}

_ALLOWED_ARGUMENTS = {"query_tokens", "top_k"}


class SearchToolbox:
    """把 ``SearchService`` 的四条入口包装成 Agent 可调用的工具集合。"""

    def __init__(self, service: SearchService, *, max_results: int = MAX_TOP_K) -> None:
        if not isinstance(service, SearchService):
            raise TypeError("SearchToolbox 需要 SearchService")
        self._service = service
        self._max_results = min(max(1, int(max_results)), MAX_TOP_K)

    @property
    def service(self) -> SearchService:
        return self._service

    @property
    def max_results(self) -> int:
        return self._max_results

    def names(self) -> tuple[str, ...]:
        return TOOL_NAMES

    def schemas(self) -> list[dict[str, Any]]:
        return [TOOL_SCHEMAS[name] for name in TOOL_NAMES]

    def schema(self, name: str) -> dict[str, Any]:
        try:
            return TOOL_SCHEMAS[name]
        except KeyError:
            raise ToolRejected("unknown_tool", f"未知工具：{name}") from None

    @staticmethod
    def is_search_tool(name: str) -> bool:
        return name in SEARCH_TOOL_NAMES

    @staticmethod
    def is_rerank_tool(name: str) -> bool:
        return name == TOOL_RERANK

    def run(
        self,
        name: str,
        arguments: Mapping[str, Any] | None,
        *,
        prior_queries: Sequence[Sequence[int]] = (),
    ) -> dict[str, Any]:
        """校验参数并执行一个工具；返回 JSON 友好结果。"""
        if name not in TOOL_SCHEMAS:
            raise ToolRejected("unknown_tool", f"未知工具：{name}")
        arguments = dict(arguments or {})
        extra = set(arguments) - _ALLOWED_ARGUMENTS
        if extra:
            raise ToolRejected("unexpected_arguments", f"工具参数包含未知字段：{sorted(extra)}")
        tokens = normalize_tokens(arguments.get("query_tokens"))
        requested_top_k = normalize_top_k(arguments.get("top_k"))
        effective_top_k = min(requested_top_k, self._max_results)

        if name == TOOL_RERANK:
            prior = [list(query) for query in prior_queries]
            if list(tokens) not in prior:
                raise ToolRejected(
                    "rerank_query_mismatch",
                    "精排的 query_tokens 必须等于本次任务里已经检索过的某个查询",
                )

        hits = self._service.search(_MODE_BY_TOOL[name], tokens, effective_top_k)
        payload: dict[str, Any] = {
            "tool": name,
            "query_tokens": list(tokens),
            "requested_top_k": requested_top_k,
            "top_k": effective_top_k,
            "count": len(hits),
            "results": [
                _result_payload(hit.as_dict(with_sources=True)) for hit in hits
            ],
        }
        if name == TOOL_HYBRID:
            payload["retrievers"] = list(self._service.hybrid.describe()["retrievers"])
            payload["fusion"] = {
                "name": "rrf",
                "rrf_k": self._service.hybrid.rrf_k,
                "pool": self._service.hybrid.pool,
            }
        if name == TOOL_RERANK:
            payload["rerank"] = {
                "method": self._service.reranker.describe()["reranker"],
                "pool": self._service.reranker.pool,
            }
        return payload


def _result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """裁剪单条候选：只保留模型需要的字段，分数保留 6 位。"""
    out = {
        "rank": int(payload["rank"]),
        "item_id": int(payload["item_id"]),
        "score": round(float(payload["score"]), 6),
        "category": str(payload.get("category") or ""),
        "sources": [
            {
                "source": str(source.get("source")),
                "rank": int(source.get("rank", 0)),
                "score": round(float(source.get("score", 0.0)), 6),
            }
            for source in payload.get("sources") or []
        ],
    }
    details = payload.get("details") or {}
    if details:
        out["details"] = {
            key: (round(float(value), 6) if isinstance(value, (int, float)) else value)
            for key, value in details.items()
        }
    return out
