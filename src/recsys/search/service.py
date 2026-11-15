"""搜索服务门面：BM25 / FAISS 向量 / 混合 / 精排 四条入口。

一个 ``SearchService`` 实例持有一份 BM25 索引、一份搜索向量索引和物品目录，
供 CLI、Search Agent 工具和验收脚本共用。四条入口都返回同一形状的
``ResultHit``：名次、item_id、分数、可读类目和来源明细。

索引独立性：BM25 索引在 ``artifacts/search/kuasar_bm25/``，搜索向量索引在
``artifacts/search/kuasar_vector/``，与推荐侧 FAISS 索引（``artifacts/faiss`` 等）
完全分开，不复用配置或数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from recsys.search.bm25 import BM25Index, IndexedItem
from recsys.search.encoders import SearchEncoder
from recsys.search.hybrid import (
    DEFAULT_POOL,
    DEFAULT_RRF_K,
    SOURCE_BM25,
    SOURCE_VECTOR,
    HybridSearcher,
    SourceHit,
)
from recsys.search.rerank import (
    DEFAULT_COVERAGE_WEIGHT,
    DEFAULT_PRIOR_WEIGHT,
    RERANKER_NAME,
    SearchReranker,
    idf_from_bm25,
)
from recsys.search.vector_index import SearchVectorIndex

MODE_BM25 = "bm25"
MODE_VECTOR = "vector"
MODE_HYBRID = "hybrid"
MODE_RERANK = "rerank"
MODES = (MODE_BM25, MODE_VECTOR, MODE_HYBRID, MODE_RERANK)


class SearchServiceError(RuntimeError):
    """搜索服务配置/索引不满足。"""


@dataclass(frozen=True)
class ResultHit:
    """统一结果条目；``sources`` 是各路来源、名次和实际分数。"""

    rank: int
    item_id: int
    score: float
    category: str = ""
    category_zh: str = ""
    item_type: str = ""
    sources: tuple[SourceHit, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self, *, with_sources: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "rank": self.rank,
            "item_id": self.item_id,
            "score": self.score,
            "category": self.category,
        }
        if with_sources:
            payload["sources"] = [source.as_dict() for source in self.sources]
        if self.details:
            payload["details"] = dict(self.details)
        return payload


class SearchService:
    """BM25、向量、混合、精排四条检索入口。"""

    def __init__(
        self,
        bm25: BM25Index,
        vector: SearchVectorIndex,
        *,
        rrf_k: int = DEFAULT_RRF_K,
        hybrid_pool: int = DEFAULT_POOL,
        rerank_method: str = RERANKER_NAME,
        rerank_pool: int = DEFAULT_POOL,
        prior_weight: float = DEFAULT_PRIOR_WEIGHT,
        coverage_weight: float = DEFAULT_COVERAGE_WEIGHT,
        bm25_index_path: Path | None = None,
        vector_index_dir: Path | None = None,
    ) -> None:
        if rerank_method != RERANKER_NAME:
            raise SearchServiceError(
                f"KuaiSAR token 搜索不支持精排方法 {rerank_method!r}，可用方法为 {RERANKER_NAME!r}"
            )
        if vector.num_items != bm25.num_docs:
            raise SearchServiceError(
                "搜索向量索引与 BM25 索引的物品数量不一致："
                f"{vector.num_items} vs {bm25.num_docs}；请重建向量索引"
            )
        bm25_ids = {item.item_id for item in bm25.items}
        vector_ids = set(vector.item_ids)
        if bm25_ids != vector_ids:
            raise SearchServiceError(
                "搜索向量索引与 BM25 索引的物品 ID 集合不一致；请重建向量索引"
            )
        self.bm25 = bm25
        self.vector = vector
        self.bm25_index_path = Path(bm25_index_path) if bm25_index_path else None
        self.vector_index_dir = Path(vector_index_dir) if vector_index_dir else None
        self._catalog: dict[int, IndexedItem] = {item.item_id: item for item in bm25.items}
        self.hybrid = HybridSearcher(bm25, vector, rrf_k=rrf_k, pool=hybrid_pool)
        self.reranker = SearchReranker(
            self.hybrid,
            text_of=self._candidate_text,
            idf=idf_from_bm25(bm25),
            pool=rerank_pool,
            prior_weight=prior_weight,
            coverage_weight=coverage_weight,
        )

    # -- 构造 ---------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        bm25_index_path: Path | str,
        vector_index_dir: Path | str,
        encoder: SearchEncoder,
        **kwargs: Any,
    ) -> "SearchService":
        bm25_path = Path(bm25_index_path)
        if not bm25_path.exists():
            raise SearchServiceError(
                f"找不到 BM25 索引：{bm25_path}；先运行 python -m recsys.search.prepare"
            )
        bm25 = BM25Index.load(bm25_path)
        vector = SearchVectorIndex.load(vector_index_dir, encoder)
        return cls(
            bm25,
            vector,
            bm25_index_path=bm25_path,
            vector_index_dir=Path(vector_index_dir),
            **kwargs,
        )

    # -- 四条入口 -----------------------------------------------------------------

    def search_bm25(self, tokens: Sequence[int], top_k: int = 10) -> list[ResultHit]:
        hits = self.bm25.search(tokens, top_k=top_k)
        return [
            self._enrich(
                hit.item_id,
                hit.score,
                sources=(SourceHit(source=SOURCE_BM25, rank=hit.rank, score=hit.score),),
                rank=hit.rank,
            )
            for hit in hits
        ]

    def search_vector(self, tokens: Sequence[int], top_k: int = 10) -> list[ResultHit]:
        hits = self.vector.search_scored(tokens, top_k=top_k)
        return [
            self._enrich(
                hit.item_id,
                hit.score,
                sources=(SourceHit(source=SOURCE_VECTOR, rank=hit.rank, score=hit.score),),
                rank=hit.rank,
            )
            for hit in hits
        ]

    def search_hybrid(
        self, tokens: Sequence[int], top_k: int = 10, pool: int | None = None
    ) -> list[ResultHit]:
        hits = self.hybrid.search(tokens, top_k=top_k, pool=pool)
        return [
            self._enrich(hit.item_id, hit.score, sources=hit.sources, rank=hit.rank)
            for hit in hits
        ]

    def rerank(self, tokens: Sequence[int], top_k: int = 10) -> list[ResultHit]:
        hits = self.reranker.rerank(tokens, top_k=top_k)
        return [
            self._enrich(
                hit.item_id,
                hit.score,
                sources=hit.sources,
                rank=hit.rank,
                details={
                    "coverage": hit.coverage,
                    "prior_score": hit.prior_score,
                    "prior_rank": hit.prior_rank,
                },
            )
            for hit in hits
        ]

    def search(self, mode: str, tokens: Sequence[int], top_k: int = 10) -> list[ResultHit]:
        mode = str(mode).strip().lower()
        if mode not in MODES:
            raise SearchServiceError(f"未知检索模式 {mode!r}；可选：{list(MODES)}")
        if mode == MODE_BM25:
            return self.search_bm25(tokens, top_k)
        if mode == MODE_VECTOR:
            return self.search_vector(tokens, top_k)
        if mode == MODE_HYBRID:
            return self.search_hybrid(tokens, top_k)
        return self.rerank(tokens, top_k)

    # -- 内部 ---------------------------------------------------------------------

    def _candidate_text(self, item_id: int) -> str:
        """精排读取的候选文本：KuaiSAR caption token 的整数串（不是原文）。"""
        item = self._catalog.get(int(item_id))
        if item is None:
            return ""
        return " ".join(str(token) for token in item.caption_tokens)

    def _enrich(
        self,
        item_id: int,
        score: float,
        *,
        sources: Iterable[SourceHit],
        rank: int,
        details: Mapping[str, Any] | None = None,
    ) -> ResultHit:
        item = self._catalog.get(int(item_id))
        return ResultHit(
            rank=int(rank),
            item_id=int(item_id),
            score=float(score),
            category=item.category if item else "",
            category_zh=item.category_zh if item else "",
            item_type=item.item_type if item else "",
            sources=tuple(sources),
            details=dict(details or {}),
        )

    # -- 描述 ---------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "bm25": {
                "path": str(self.bm25_index_path) if self.bm25_index_path else None,
                "num_docs": self.bm25.num_docs,
                "vocab_size": self.bm25.vocab_size,
                "avgdl": round(self.bm25.avgdl, 6),
                "k1": self.bm25.k1,
                "b": self.bm25.b,
            },
            "vector": self.vector.describe(),
            "hybrid": self.hybrid.describe(),
            "rerank": self.reranker.describe(),
        }


__all__ = [
    "MODE_BM25",
    "MODE_HYBRID",
    "MODE_RERANK",
    "MODE_VECTOR",
    "MODES",
    "ResultHit",
    "SearchService",
    "SearchServiceError",
]
