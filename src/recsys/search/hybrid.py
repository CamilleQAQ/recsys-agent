"""混合检索：BM25 + 向量召回，按物品 ID 去重后用上游 RRF 融合。

复用 SkimSearchAgent（固定提交 4794575）：

- ``agent_search.retrievers.fusion.rrf.RRF``：倒数排名融合
  ``score(d) = Σ 1 / (k + rank(d))``，只读名次、忽略原始分数、同分按 doc id 升序；
- 上游 ``HybridEngine`` 只返回 doc id，不保留各路来源与分数，所以这里写薄适配：
  融合顺序**直接调用上游 RRF**，项目侧只负责收集两路候选、按 item_id 去重并
  记录每路的来源名次和实际分数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from recsys.search.bm25 import BM25Index
from recsys.search.upstream import RRF, RRF_DEFAULT_K
from recsys.search.vector_index import SearchVectorIndex

SOURCE_BM25 = "bm25"
SOURCE_VECTOR = "vector"
DEFAULT_RRF_K = RRF_DEFAULT_K
DEFAULT_POOL = 50


@dataclass(frozen=True)
class SourceHit:
    """某个来源里的一条候选：来源名、名次（从 1 开始）、该来源的实际分数。"""

    source: str
    rank: int
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {"source": self.source, "rank": self.rank, "score": self.score}


@dataclass(frozen=True)
class HybridHit:
    """融合后的一条结果：去重后的 item_id、RRF 融合分、各路来源明细。"""

    rank: int
    item_id: int
    score: float
    sources: tuple[SourceHit, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "item_id": self.item_id,
            "score": self.score,
            "sources": [source.as_dict() for source in self.sources],
        }


class HybridSearcher:
    """两路召回 + RRF 融合，保留来源、名次和实际分数。"""

    def __init__(
        self,
        bm25: BM25Index,
        vector: SearchVectorIndex,
        *,
        rrf_k: int = DEFAULT_RRF_K,
        pool: int = DEFAULT_POOL,
    ) -> None:
        if rrf_k < 1:
            raise ValueError(f"rrf_k 必须 >= 1，实际 {rrf_k}")
        if pool < 1:
            raise ValueError(f"pool 必须 >= 1，实际 {pool}")
        self.bm25 = bm25
        self.vector = vector
        self.rrf_k = int(rrf_k)
        self.pool = int(pool)
        self._rrf = RRF(k=self.rrf_k)

    def rankings(
        self, tokens: Sequence[int], pool: int | None = None
    ) -> dict[str, list[tuple[int, float]]]:
        """两路召回候选；每路按该路分数降序，最多 ``pool`` 条。"""
        pool = int(pool or self.pool)
        bm25_ranking = [
            (hit.item_id, float(hit.score))
            for hit in self.bm25.search(tokens, top_k=pool)
        ]
        vector_ranking = [
            (hit.item_id, float(hit.score))
            for hit in self.vector.search_scored(tokens, top_k=pool)
        ]
        return {
            SOURCE_BM25: _dedupe(bm25_ranking),
            SOURCE_VECTOR: _dedupe(vector_ranking),
        }

    def search(
        self,
        tokens: Sequence[int],
        top_k: int | None = None,
        pool: int | None = None,
    ) -> list[HybridHit]:
        """融合 Top-K；同一 item_id 只出现一次，来源合并。"""
        pool = int(pool or self.pool)
        rankings = self.rankings(tokens, pool=pool)
        if not any(rankings.values()):
            return []

        # 融合顺序直接来自上游 RRF，项目侧只补来源与分数。
        fused_ids = self._rrf.fuse(list(rankings.values()), None)
        provenance: dict[int, list[SourceHit]] = {}
        fused_scores: dict[int, float] = {}
        for source, ranking in rankings.items():
            for rank, (item_id, score) in enumerate(ranking, start=1):
                provenance.setdefault(item_id, []).append(
                    SourceHit(source=source, rank=rank, score=score)
                )
                fused_scores[item_id] = fused_scores.get(item_id, 0.0) + 1.0 / (
                    self.rrf_k + rank
                )

        hits: list[HybridHit] = []
        for rank, doc_id in enumerate(fused_ids, start=1):
            item_id = int(doc_id)
            hits.append(
                HybridHit(
                    rank=rank,
                    item_id=item_id,
                    score=float(fused_scores.get(item_id, 0.0)),
                    sources=tuple(provenance.get(item_id, ())),
                )
            )
            if top_k is not None and len(hits) >= int(top_k):
                break
        return hits

    def describe(self) -> dict[str, Any]:
        return {
            "retrievers": [SOURCE_BM25, SOURCE_VECTOR],
            "fusion": "rrf",
            "rrf_k": self.rrf_k,
            "pool": self.pool,
            "upstream": "agent_search.retrievers.fusion.rrf.RRF",
        }


def _dedupe(ranking: Iterable[tuple[int, float]]) -> list[tuple[int, float]]:
    """同一来源内按 item_id 去重，保留第一次（最好的）名次。"""
    seen: set[int] = set()
    out: list[tuple[int, float]] = []
    for item_id, score in ranking:
        if item_id in seen:
            continue
        seen.add(item_id)
        out.append((item_id, float(score)))
    return out


__all__ = [
    "DEFAULT_POOL",
    "DEFAULT_RRF_K",
    "HybridHit",
    "HybridSearcher",
    "SOURCE_BM25",
    "SOURCE_VECTOR",
    "SourceHit",
]
