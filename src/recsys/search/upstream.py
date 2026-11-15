"""SkimSearchAgent 上游组件的导入与再导出（固定提交 4794575）。

上游仓库在 ``third_party/skim-search-agent``，本模块只做两件事：

1. 把上游仓库根目录加入 ``sys.path``（项目没有把上游打包安装）；
2. 再导出搜索侧真正复用的组件，**不改上游源码**：

- ``agent_search.retrievers.dense.vector_index``：向量索引的存储、持久化与加载；
- ``agent_search.retrievers.fusion.rrf``：倒数排名融合（RRF）；
- ``agent_search.retrievers.rerankers``：精排接口 ``Reranker`` 与注册表；
- ``agent_search.retrievers.reranked``：``retriever + reranker`` 组合 ``RerankedEngine``。

上游的 ``HybridEngine`` / ``RerankedEngine`` 只返回 doc id，不带各路来源与分数；
项目的搜索接口要求保留来源、名次和实际分数，所以在 ``recsys.search.hybrid`` /
``recsys.search.rerank`` 里写的是**薄适配**：融合算法仍调用上游 ``RRF``，
精排仍实现上游 ``Reranker`` 契约并由 ``RerankedEngine`` 组合。
"""

from __future__ import annotations

import sys
from pathlib import Path

UPSTREAM_COMMIT = "4794575e5de52251e97b3dd6bb1482f3b142a960"
UPSTREAM_SHORT = "4794575"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "skim-search-agent"


class UpstreamUnavailable(RuntimeError):
    """找不到上游源码快照。"""


def ensure_upstream_on_path() -> Path:
    """把 ``third_party/skim-search-agent`` 放到 ``sys.path`` 最前面。"""
    if not (UPSTREAM_ROOT / "agent_search").is_dir():
        raise UpstreamUnavailable(
            f"找不到 SkimSearchAgent 源码：{UPSTREAM_ROOT}；"
            "请确认 third_party/skim-search-agent 处于固定提交 "
            f"{UPSTREAM_SHORT}"
        )
    root = str(UPSTREAM_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return UPSTREAM_ROOT


ensure_upstream_on_path()

# 上游导入放在 sys.path 设置之后；这里都是轻量模块，不加载 torch/sentence-transformers。
from agent_search.retrievers.dense import vector_index as upstream_vector_index  # noqa: E402
from agent_search.retrievers.fusion.base import Fusion  # noqa: E402
from agent_search.retrievers.fusion.rrf import DEFAULT_K as RRF_DEFAULT_K  # noqa: E402
from agent_search.retrievers.fusion.rrf import RRF  # noqa: E402
from agent_search.retrievers.reranked import RerankedEngine  # noqa: E402
from agent_search.retrievers.rerankers import (  # noqa: E402
    Reranker,
    build_reranker,
    register_reranker,
)

VectorIndex = upstream_vector_index.VectorIndex
FlatIndex = upstream_vector_index.FlatIndex
build_vector_index = upstream_vector_index.build_index
save_vector_index = upstream_vector_index.save_index
load_vector_index = upstream_vector_index.load_index
vector_index_exists = upstream_vector_index.index_exists


def faiss_module():
    """上游的 faiss 探测函数：可用时返回模块，否则 ``None``。"""
    return upstream_vector_index._faiss()


def faiss_version() -> str | None:
    faiss = faiss_module()
    return str(getattr(faiss, "__version__", "")) or None if faiss is not None else None


def register_vector_backend(cls: type) -> None:
    """把项目的 ``VectorIndex`` 子类登记进上游 backend 表。

    上游的 ``save_index`` / ``load_index`` 通过 ``meta.json`` 里的 ``backend``
    名字回到 ``_BACKENDS``，所以自定义 backend 必须登记一次，持久化才能往返。
    这是适配层唯一触碰上游私有表的地方，固定提交内有效。
    """
    upstream_vector_index._BACKENDS[cls.backend] = cls


__all__ = [
    "Fusion",
    "FlatIndex",
    "RRF",
    "RRF_DEFAULT_K",
    "RerankedEngine",
    "Reranker",
    "UPSTREAM_COMMIT",
    "UPSTREAM_ROOT",
    "UPSTREAM_SHORT",
    "UpstreamUnavailable",
    "VectorIndex",
    "build_reranker",
    "build_vector_index",
    "ensure_upstream_on_path",
    "faiss_module",
    "faiss_version",
    "load_vector_index",
    "register_reranker",
    "register_vector_backend",
    "save_vector_index",
    "upstream_vector_index",
    "vector_index_exists",
]
