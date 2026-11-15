"""查询规范化与受约束改写（KuaiSAR 整数 token）。

数据限制：KuaiSAR 的 ``keyword`` 是分词后随机哈希的整数 token，没有词表、
没有原始中文查询文本，因此这里**不做**自然语言理解或语义改写，只做两类
可解释、可追踪的处理：

- **规范化**：校验正整数、去重（保序）、限制长度；
- **受约束改写**：从原查询里删掉一个 token（或指定 token），得到原查询的
  非空真子集——与 Search Agent 的「第二次查询只能是原查询子集」规则一致。

原始查询与规范化/改写后的查询都随结果一起返回，便于追踪。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

MAX_QUERY_TOKENS = 64


@dataclass(frozen=True)
class QueryForm:
    """一次查询的原始形态与处理后形态。"""

    original: tuple[int, ...]
    normalized: tuple[int, ...]
    normalization: str
    rewrite_rule: str = "none"
    notes: str = ""

    @property
    def tokens(self) -> tuple[int, ...]:
        return self.normalized

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_tokens": list(self.original),
            "normalized_tokens": list(self.normalized),
            "normalization": self.normalization,
            "rewrite_rule": self.rewrite_rule,
            "notes": self.notes,
        }


def _positive_ints(tokens: Iterable[int]) -> list[int]:
    out: list[int] = []
    for token in tokens:
        if isinstance(token, bool):
            raise ValueError("查询 token 不能是布尔值")
        value = int(token)
        if value <= 0:
            raise ValueError(f"查询 token 必须是正整数，实际 {value}")
        out.append(value)
    return out


def normalize_query(
    tokens: Sequence[int], *, max_tokens: int = MAX_QUERY_TOKENS
) -> QueryForm:
    """校验并规范化查询：去重保序、限制长度。空查询返回空形态。"""
    original = tuple(int(token) for token in tokens)
    positive = _positive_ints(original)
    if not positive:
        return QueryForm(
            original=original,
            normalized=(),
            normalization="empty",
            notes="空查询：检索返回空结果",
        )
    seen: set[int] = set()
    deduped: list[int] = []
    for token in positive:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    truncated = len(deduped) > max_tokens
    normalized = tuple(deduped[:max_tokens])
    if truncated:
        normalization = "dedupe+truncate"
        notes = f"去重后超过 {max_tokens} 个，截断尾部"
    elif len(normalized) != len(original):
        normalization = "dedupe"
        notes = "去掉了重复 token"
    else:
        normalization = "identity"
        notes = "原样使用"
    return QueryForm(original=original, normalized=normalized, normalization=normalization, notes=notes)


def rewrite_query(
    tokens: Sequence[int], rule: str = "drop_last"
) -> QueryForm:
    """从原查询派生一个非空真子集（受约束改写）。"""
    original = tuple(int(token) for token in tokens)
    normalized = tuple(dict.fromkeys(_positive_ints(original)))
    if not normalized:
        raise ValueError("空查询不能改写")
    if rule == "drop_last":
        rewritten = normalized[:-1]
    elif rule == "drop_first":
        rewritten = normalized[1:]
    elif rule.startswith("drop_token:"):
        victim = int(rule.split(":", 1)[1])
        rewritten = tuple(token for token in normalized if token != victim)
        if len(rewritten) == len(normalized):
            raise ValueError(f"原查询里没有 token {victim}")
    else:
        raise ValueError(f"未知改写规则 {rule!r}")
    if not rewritten:
        raise ValueError(f"改写规则 {rule!r} 会产生空查询")
    return QueryForm(
        original=original,
        normalized=tuple(rewritten),
        normalization="subset",
        rewrite_rule=rule,
        notes="受约束改写：只从原查询删除 token",
    )


def candidate_rewrites(tokens: Sequence[int], limit: int = 3) -> list[QueryForm]:
    """列出可用的受约束改写，供 Agent 提示或本地替身选择。"""
    forms: list[QueryForm] = []
    seen: set[tuple[int, ...]] = set()
    for rule in ("drop_last", "drop_first"):
        try:
            form = rewrite_query(tokens, rule)
        except ValueError:
            continue
        if form.normalized in seen:
            continue
        seen.add(form.normalized)
        forms.append(form)
        if len(forms) >= limit:
            return forms
    for token in dict.fromkeys(int(value) for value in tokens):
        if len(forms) >= limit:
            break
        try:
            form = rewrite_query(tokens, f"drop_token:{token}")
        except ValueError:
            continue
        if form.normalized in seen:
            continue
        seen.add(form.normalized)
        forms.append(form)
    return forms


def is_subset_query(tokens: Sequence[int], original: Sequence[int]) -> bool:
    """``tokens`` 是否是 ``original`` 的（多重集）子集。"""
    from collections import Counter

    return not (Counter(int(token) for token in tokens) - Counter(int(token) for token in original))


__all__ = [
    "MAX_QUERY_TOKENS",
    "QueryForm",
    "candidate_rewrites",
    "is_subset_query",
    "normalize_query",
    "rewrite_query",
]
