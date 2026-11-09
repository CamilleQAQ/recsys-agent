"""Search Agent 的 ``search_bm25`` 工具。

工具只做三件事：校验查询 token、调用 Search 1 的 ``BM25Index.search``、
把结果裁剪成最多 5 条（item_id / score / 可读类目）。不加载原始 CSV，
不重建索引，不保存候选明细。
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

from recsys.search.bm25 import BM25Index, SearchHit

TOOL_NAME = "search_bm25"
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
