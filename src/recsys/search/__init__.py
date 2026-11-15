"""搜索模块：KuaiSAR 小样本上的 BM25、FAISS 向量、混合检索与精排。

- ``bm25``         关键词索引与检索（保留原实现）；
- ``encoders``     可插拔编码器：无训练的 token 哈希编码 / 真实文本语义编码占位；
- ``vector_index`` 搜索侧 FAISS 索引（独立于推荐索引），支持保存/重载/更新；
- ``hybrid``       BM25 + 向量按 item_id 去重后 RRF 融合，保留来源与分数；
- ``rerank``       接在融合候选后的精排（上游 Reranker 契约）；
- ``service``      四条检索入口的统一门面；
- ``tool``         Search Agent 工具（``search_bm25`` / ``search_vector`` /
                   ``search_hybrid`` / ``search_rerank``）；
- ``query``        查询规范化与受约束改写。

Search Agent 工作流在 ``recsys.experiments.search_agent``。
"""

from recsys.search.bm25 import (
    INDEX_FORMAT,
    BM25Index,
    IndexedItem,
    SearchHit,
    load_index,
)
from recsys.search.encoders import (
    EncoderUnavailable,
    SearchEncoder,
    TextSemanticEncoder,
    TokenHashEncoder,
    build_encoder,
)
from recsys.search.hybrid import HybridHit, HybridSearcher, SourceHit
from recsys.search.query import QueryForm, normalize_query, rewrite_query
from recsys.search.rerank import RerankHit, SearchReranker
from recsys.search.service import MODES, ResultHit, SearchService
from recsys.search.tool import (
    SEARCH_TOOL_NAMES,
    TOOL_HYBRID,
    TOOL_NAME,
    TOOL_NAMES,
    TOOL_RERANK,
    TOOL_SCHEMAS,
    TOOL_VECTOR,
    SearchBM25Tool,
    SearchToolbox,
    ToolRejected,
    normalize_top_k,
    normalize_tokens,
)
from recsys.search.vector_index import (
    FaissFlatIndex,
    SearchVectorIndex,
    VectorHit,
    VectorIndexError,
)

__all__ = [
    "BM25Index",
    "EncoderUnavailable",
    "INDEX_FORMAT",
    "IndexedItem",
    "MODES",
    "QueryForm",
    "RerankHit",
    "ResultHit",
    "SEARCH_TOOL_NAMES",
    "SearchBM25Tool",
    "SearchEncoder",
    "SearchHit",
    "SearchReranker",
    "SearchService",
    "SearchToolbox",
    "SearchVectorIndex",
    "SourceHit",
    "TOOL_HYBRID",
    "TOOL_NAME",
    "TOOL_NAMES",
    "TOOL_RERANK",
    "TOOL_SCHEMAS",
    "TOOL_VECTOR",
    "TextSemanticEncoder",
    "TokenHashEncoder",
    "ToolRejected",
    "VectorHit",
    "VectorIndexError",
    "build_encoder",
    "load_index",
    "normalize_query",
    "normalize_top_k",
    "normalize_tokens",
    "rewrite_query",
]
