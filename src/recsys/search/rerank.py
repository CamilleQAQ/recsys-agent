"""搜索精排：接在融合候选后面的轻量 token 精排。

复用 SkimSearchAgent（固定提交 4794575）：

- ``Reranker`` 是上游的精排契约：``rerank(query, [(doc_id, text)], k)``；
- ``register_reranker`` 是上游的注册点：精排方法按名字可选；
- ``RerankedEngine`` 是上游的组合：先让 base retriever 取 ``pool`` 个候选，
  交给 reranker 重新打分，再返回最终 Top-K。

当前没有可用的真实文本也没有本地预训练精排器（``sentence-transformers`` 未安装），
所以用基于现有 KuaiSAR token 的轻量规则精排 ``kuasar_token_overlap``：

    coverage(d) = Σ_{t ∈ Q_unique} idf(t) · 1[t ∈ d] / Σ_{t ∈ Q_unique} idf(t)
    prior(d)    = 融合候选分数在候选池内 min-max 归一化
    score(d)    = coverage_weight · coverage(d) + prior_weight · prior(d)

排序键 ``(-score, item_id)``，同分按 item_id 升序。精排只读 token 与融合分数，
**不使用点击标签**；点击只在验收时用于核对，不参与打分。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from recsys.search.hybrid import HybridHit, HybridSearcher
from recsys.search.upstream import RerankedEngine, Reranker, register_reranker

RERANKER_NAME = "kuasar_token_overlap"
DEFAULT_POOL = 50
DEFAULT_PRIOR_WEIGHT = 0.35
DEFAULT_COVERAGE_WEIGHT = 0.65


def token_signature(tokens: Sequence[int]) -> str:
    """把整数 token 序列变成上游精排接口要求的查询字符串。"""
    return " ".join(str(int(token)) for token in tokens)


def parse_token_signature(text: str) -> list[int]:
    """反向解析 token 字符串；只保留整数 token，忽略非数字内容。"""
    tokens: list[int] = []
    for part in str(text or "").split():
        try:
            tokens.append(int(part))
        except ValueError:
            continue
    return tokens


@register_reranker
class TokenOverlapReranker(Reranker):
    """基于现有 token 与融合先验的轻量精排规则。

    这是一个真实的 ``Reranker`` 实现：由上游 ``RerankedEngine`` 调用，
    对候选**重新打分**，不读取点击标签。
    """

    name = RERANKER_NAME
    default_model = None  # 不读任何预训练权重
    default_max_length = 256

    def __init__(
        self,
        idf: Mapping[int, float] | None = None,
        *,
        prior_weight: float = DEFAULT_PRIOR_WEIGHT,
        coverage_weight: float = DEFAULT_COVERAGE_WEIGHT,
    ) -> None:
        if prior_weight < 0 or coverage_weight < 0:
            raise ValueError("精排权重必须 >= 0")
        if prior_weight + coverage_weight <= 0:
            raise ValueError("精排权重之和必须 > 0")
        self._idf = {int(token): float(value) for token, value in (idf or {}).items()}
        self.prior_weight = float(prior_weight)
        self.coverage_weight = float(coverage_weight)
        self._query_tokens: tuple[int, ...] = ()
        self._prior: dict[int, float] = {}
        self._prior_norm: dict[int, float] = {}
        #: 最近一次精排的逐候选明细，供上层追溯（不落盘候选 ID）。
        self.last_components: dict[int, dict[str, float]] = {}

    def prepare(self, query_tokens: Sequence[int], prior_scores: Mapping[int, float]) -> None:
        """注入本次查询的 token 与融合先验分；每次精排前调用。"""
        self._query_tokens = tuple(int(token) for token in query_tokens)
        self._prior = {int(item_id): float(score) for item_id, score in prior_scores.items()}
        if self._prior:
            low = min(self._prior.values())
            high = max(self._prior.values())
            span = high - low
            self._prior_norm = {
                item_id: (0.5 if span <= 0 else (score - low) / span)
                for item_id, score in self._prior.items()
            }
        else:
            self._prior_norm = {}
        self.last_components = {}

    def _coverage(self, query_tokens: Sequence[int], doc_tokens: Sequence[int]) -> float:
        if not query_tokens:
            return 0.0
        doc_set = set(int(token) for token in doc_tokens)
        weights = [self._idf.get(int(token), 1.0) for token in query_tokens]
        total = sum(weights)
        if total <= 0:
            return 0.0
        hit = sum(
            weight for token, weight in zip(query_tokens, weights) if int(token) in doc_set
        )
        return hit / total

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[str, str]],
        k: int | None = None,
    ) -> list[tuple[str, float]]:
        query_tokens = list(self._query_tokens) or parse_token_signature(query)
        scored: list[tuple[str, float]] = []
        for doc_id, text in candidates:
            item_id = int(doc_id)
            doc_tokens = parse_token_signature(text)
            coverage = self._coverage(query_tokens, doc_tokens)
            prior_norm = float(self._prior_norm.get(item_id, 0.0))
            score = (
                self.coverage_weight * coverage + self.prior_weight * prior_norm
            )
            score = round(score, 9)
            self.last_components[item_id] = {
                "coverage": round(coverage, 9),
                "prior": round(float(self._prior.get(item_id, 0.0)), 9),
                "prior_norm": round(prior_norm, 9),
                "score": score,
            }
            scored.append((doc_id, score))
        scored.sort(key=lambda row: (-row[1], int(row[0])))
        return scored[:k] if k is not None else scored

    def describe(self) -> dict[str, Any]:
        return {
            "reranker": self.name,
            "prior_weight": self.prior_weight,
            "coverage_weight": self.coverage_weight,
            "training_free": True,
            "uses_click_label": False,
            "query_side": "idf-weighted token coverage over KuaiSAR integer tokens",
        }


class _HybridPoolBase:
    """``RerankedEngine`` 的 base：按查询签名调用真实的混合检索并缓存候选池。"""

    def __init__(self, searcher: HybridSearcher, pool: int) -> None:
        self._searcher = searcher
        self.pool = int(pool)
        self._cache: dict[tuple[tuple[int, ...], int], list[HybridHit]] = {}

    def fetch(self, tokens: Sequence[int], k: int | None = None) -> list[HybridHit]:
        key = (tuple(int(token) for token in tokens), int(k or self.pool))
        if key not in self._cache:
            self._cache[key] = self._searcher.search(
                list(tokens), top_k=key[1], pool=key[1]
            )
        return list(self._cache[key])

    def top_k_doc_ids(self, query: str, k: int | None = None) -> list[str]:
        hits = self.fetch(parse_token_signature(query), k or self.pool)
        return [str(hit.item_id) for hit in hits]

    def search_scored(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        hits = self.fetch(parse_token_signature(query), k or self.pool)
        return [(str(hit.item_id), hit.score) for hit in hits]


@dataclass(frozen=True)
class RerankHit:
    """精排后的一条结果：新分数 + 融合先验 + 来源明细。"""

    rank: int
    item_id: int
    score: float
    coverage: float
    prior_score: float
    prior_rank: int
    sources: tuple

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "item_id": self.item_id,
            "score": self.score,
            "coverage": self.coverage,
            "prior_score": self.prior_score,
            "prior_rank": self.prior_rank,
            "sources": [
                source.as_dict() if hasattr(source, "as_dict") else dict(source)
                for source in self.sources
            ],
        }


class SearchReranker:
    """把上游 ``RerankedEngine`` 接到融合候选上，并保留项目需要的追溯字段。"""

    def __init__(
        self,
        searcher: HybridSearcher,
        *,
        text_of: Callable[[int], str],
        idf: Mapping[int, float] | None = None,
        pool: int = DEFAULT_POOL,
        prior_weight: float = DEFAULT_PRIOR_WEIGHT,
        coverage_weight: float = DEFAULT_COVERAGE_WEIGHT,
    ) -> None:
        if pool < 1:
            raise ValueError(f"精排候选池必须 >= 1，实际 {pool}")
        self._searcher = searcher
        self._text_of = text_of
        self._base = _HybridPoolBase(searcher, pool)
        self.reranker = TokenOverlapReranker(
            idf=idf, prior_weight=prior_weight, coverage_weight=coverage_weight
        )
        self.pool = int(pool)
        # 上游组合：base retriever -> reranker -> 最终 Top-K。
        self._engine = RerankedEngine(
            self._base,
            self.reranker,
            self._candidate_text,
            pool=self.pool,
            base_name="hybrid",
        )

    def _candidate_text(self, doc_id: str) -> str:
        return self._text_of(int(doc_id))

    def rerank(
        self,
        tokens: Sequence[int],
        top_k: int = 10,
    ) -> list[RerankHit]:
        """对融合候选池重新打分并返回最终 Top-K。"""
        pool_hits = self._base.fetch(tokens, self.pool)
        if not pool_hits:
            self.reranker.prepare(tokens, {})
            return []
        prior = {hit.item_id: hit.score for hit in pool_hits}
        by_id = {hit.item_id: hit for hit in pool_hits}
        self.reranker.prepare(tokens, prior)
        ranked = self._engine.search_scored(token_signature(tokens), int(top_k))
        hits: list[RerankHit] = []
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            item_id = int(doc_id)
            source_hit = by_id.get(item_id)
            components = self.reranker.last_components.get(item_id, {})
            hits.append(
                RerankHit(
                    rank=rank,
                    item_id=item_id,
                    score=float(score),
                    coverage=float(components.get("coverage", 0.0)),
                    prior_score=float(source_hit.score if source_hit else 0.0),
                    prior_rank=int(source_hit.rank if source_hit else 0),
                    sources=tuple(source_hit.sources if source_hit else ()),
                )
            )
        return hits

    def describe(self) -> dict[str, Any]:
        return {
            "base": "hybrid",
            "pool": self.pool,
            **self.reranker.describe(),
            "engine": "agent_search.retrievers.reranked.RerankedEngine",
        }


def idf_from_bm25(bm25) -> dict[int, float]:
    """从现有 BM25 索引取 token IDF，供精排覆盖率加权（不改 BM25 本身）。"""
    idf = getattr(bm25, "idf", None)
    if idf:
        return {int(token): float(value) for token, value in idf.items()}
    # 兜底：没有 IDF 时给一个中性权重。
    return {}


__all__ = [
    "DEFAULT_COVERAGE_WEIGHT",
    "DEFAULT_POOL",
    "DEFAULT_PRIOR_WEIGHT",
    "RERANKER_NAME",
    "RerankHit",
    "SearchReranker",
    "TokenOverlapReranker",
    "idf_from_bm25",
    "parse_token_signature",
    "token_signature",
]
