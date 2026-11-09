"""Search MVP：KuaiSAR 搜索小样本 + BM25 检索。

本轮只包含数据准备、BM25 索引、查询接口和独立 CLI；Search Agent、
向量检索、排序/重排留到后续阶段。
"""

from recsys.search.bm25 import (
    INDEX_FORMAT,
    BM25Index,
    IndexedItem,
    SearchHit,
    load_index,
)

__all__ = [
    "INDEX_FORMAT",
    "BM25Index",
    "IndexedItem",
    "SearchHit",
    "load_index",
]
