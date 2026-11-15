"""搜索侧向量索引：复用 SkimSearchAgent 的向量索引组件，接上现有 faiss-cpu。

与推荐侧 ``recsys.retrieval.faiss_index`` 完全独立：索引目录、
编码器、物品映射和配置都由搜索模块自己管理，不共享推荐索引。

复用的上游组件（固定提交 4794575）：

- ``agent_search.retrievers.dense.vector_index.VectorIndex``：索引基类契约；
- ``FlatIndex``：无 faiss 时的 NumPy 精确检索后端（可选）；
- ``build_index`` / ``save_index`` / ``load_index`` / ``index_exists``：
  索引构建、原子落盘（``doc_ids.json`` + ``meta.json`` + payload）与重载。

必要的适配：

- 上游只提供 NumPy ``FlatIndex``、FAISS ``HNSW`` 与 ``IVFPQ``；搜索小样本要
  **精确** 且要能核对「确实调用了 FAISS」，所以新增 ``FaissFlatIndex``
  （``faiss.IndexFlatIP``，归一化向量上的内积即余弦），并登记进上游 backend 表，
  这样上游的 ``save_index`` / ``load_index`` 可以原样往返；
- 上游 ``doc_ids`` 是字符串；项目要求物品 ID 映射，所以索引同时保留
  ``item_ids: list[int]``，搜索直接返回整数 ``item_id``；
- 增加 ``add_items`` 更新入口（FAISS 直接 ``index.add``，NumPy 后端重建），
  更新后仍需显式 ``save``；
- 索引 meta 里记录编码器描述与配置哈希，重载时校验，避免用不同编码器查索引。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from recsys.search.encoders import SearchEncoder
from recsys.search.upstream import (
    FlatIndex,
    VectorIndex,
    faiss_module,
    faiss_version,
    load_vector_index,
    register_vector_backend,
    save_vector_index,
    vector_index_exists,
)

FAISS_BACKEND = "faiss"
NUMPY_BACKEND = "numpy"
INDEX_META_FORMAT = "recsys.search.vector_index.v1"


class VectorIndexError(RuntimeError):
    """向量索引缺失、损坏或与当前编码器/物品集合不一致。"""


@dataclass(frozen=True)
class VectorHit:
    """一条向量检索结果；score 是余弦相似度。"""

    rank: int
    item_id: int
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {"rank": self.rank, "item_id": self.item_id, "score": self.score}


class FaissFlatIndex(VectorIndex):
    """精确 FAISS 索引：``IndexFlatIP`` + 整数物品 ID 映射。

    归一化向量上的内积等于余弦相似度。上游 ``VectorIndex`` 要求
    ``doc_ids``（字符串）与 ``_save_payload`` / ``_load_payload``，这里都实现，
    因此可以直接用上游 ``save_index`` / ``load_index`` 保存和重载。
    """

    backend = "faiss_flat"
    PAYLOAD = "ann.faiss"

    def __init__(self, index, doc_ids: Sequence[str], item_ids: Sequence[int] | None = None) -> None:
        super().__init__([str(doc_id) for doc_id in doc_ids])
        self.index = index
        self.item_ids = [int(value) for value in (item_ids or doc_ids)]
        if len(self.item_ids) != len(self.doc_ids):
            raise VectorIndexError(
                f"物品映射长度不一致：{len(self.item_ids)} vs {len(self.doc_ids)}"
            )
        self.faiss_search_calls = 0

    @classmethod
    def build(cls, matrix: np.ndarray, item_ids: Sequence[int]) -> "FaissFlatIndex":
        faiss = faiss_module()
        if faiss is None:
            raise VectorIndexError(
                "搜索向量索引要求 faiss-cpu：当前 import faiss 失败。"
                "请安装 requirements/faiss.txt（faiss-cpu>=1.7.4）后重建索引。"
            )
        matrix = _check_matrix(matrix)
        item_ids = [int(value) for value in item_ids]
        if len(item_ids) != matrix.shape[0]:
            raise VectorIndexError("向量行数与物品数量不一致")
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(np.ascontiguousarray(matrix, dtype=np.float32))
        return cls(index, [str(value) for value in item_ids], item_ids)

    def add(self, matrix: np.ndarray, item_ids: Sequence[int]) -> None:
        matrix = _check_matrix(matrix)
        new_ids = [int(value) for value in item_ids]
        if len(new_ids) != matrix.shape[0]:
            raise VectorIndexError("新增向量行数与物品数量不一致")
        if len(set(new_ids)) != len(new_ids):
            raise VectorIndexError("新增物品 ID 有重复")
        duplicates = set(new_ids) & set(self.item_ids)
        if duplicates:
            raise VectorIndexError(f"物品已在索引中：{sorted(duplicates)[:5]}")
        if matrix.shape[1] != self.index.d:
            raise VectorIndexError(
                f"向量维度不一致：新增 {matrix.shape[1]}，索引 {self.index.d}"
            )
        self.index.add(np.ascontiguousarray(matrix, dtype=np.float32))
        self.item_ids.extend(new_ids)
        self.doc_ids.extend(str(value) for value in new_ids)

    def search_positions(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self.index.ntotal == 0:
            return []
        k = min(int(k), int(self.index.ntotal))
        if k <= 0:
            return []
        query = np.ascontiguousarray(np.asarray(query_vector, dtype=np.float32)[None, :])
        scores, positions = self.index.search(query, k)
        self.faiss_search_calls += 1
        return [
            (int(position), float(score))
            for position, score in zip(positions[0], scores[0])
            if position != -1
        ]

    def search_scored(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        return [
            (self.item_ids[position], score)
            for position, score in self.search_positions(query_vector, k)
        ]

    def search(self, query_vector: np.ndarray, k: int) -> list[str]:
        """上游 ``VectorIndex`` 契约：返回 doc id（字符串）。"""
        return [str(item_id) for item_id, _score in self.search_scored(query_vector, k)]

    def _save_payload(self, cache_dir: str) -> None:
        faiss = faiss_module()
        if faiss is None:  # pragma: no cover - 构建时已经检查
            raise VectorIndexError("保存 FAISS 索引需要 faiss-cpu")
        target = os.path.join(cache_dir, self.PAYLOAD)
        tmp = f"{target}.tmp.{os.getpid()}"
        faiss.write_index(self.index, tmp)
        os.replace(tmp, target)

    @classmethod
    def _load_payload(cls, cache_dir: str, doc_ids: list[str]) -> "FaissFlatIndex":
        faiss = faiss_module()
        if faiss is None:
            raise VectorIndexError("重载 FAISS 索引需要 faiss-cpu")
        index = faiss.read_index(os.path.join(cache_dir, cls.PAYLOAD))
        return cls(index, doc_ids)


register_vector_backend(FaissFlatIndex)


def _check_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
    if matrix.ndim != 2:
        raise VectorIndexError("向量矩阵必须是二维 [n, dim]")
    if matrix.shape[0] == 0:
        raise VectorIndexError("向量矩阵为空")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise VectorIndexError("向量必须 L2 归一化后再建内积索引")
    return matrix


class SearchVectorIndex:
    """搜索向量索引门面：编码、Top-K、物品 ID 映射、保存/重载/更新。"""

    def __init__(
        self,
        index: VectorIndex,
        encoder: SearchEncoder,
        *,
        backend: str,
        cache_dir: Path | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> None:
        self._index = index
        self.encoder = encoder
        self.backend = backend
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.extra_meta = dict(extra_meta or {})
        # 索引里出现过的 token 集合：查询里的 OOV token 不参与编码，避免哈希碰撞
        # 让「索引里根本没有的词」也召回文档（与 BM25 的 posting 查不到一致）。
        vocab = self.extra_meta.get("vocab") or ()
        self._vocab = {int(token) for token in vocab}
        self._item_ids = _item_ids_of(index)
        self._positions = {item_id: position for position, item_id in enumerate(self._item_ids)}
        if len(self._positions) != len(self._item_ids):
            raise VectorIndexError("索引里的物品 ID 有重复")

    # -- 基本属性 -----------------------------------------------------------------

    @property
    def num_items(self) -> int:
        return len(self._item_ids)

    @property
    def item_ids(self) -> list[int]:
        return list(self._item_ids)

    @property
    def dim(self) -> int:
        return int(self.encoder.dim)

    @property
    def backend_actual(self) -> str:
        if isinstance(self._index, FaissFlatIndex):
            return "faiss.IndexFlatIP"
        return f"numpy.{type(self._index).__name__}"

    @property
    def faiss_active(self) -> bool:
        return isinstance(self._index, FaissFlatIndex)

    @property
    def faiss_search_calls(self) -> int:
        if isinstance(self._index, FaissFlatIndex):
            return int(self._index.faiss_search_calls)
        return 0

    # -- 构建 / 保存 / 重载 -------------------------------------------------------

    @classmethod
    def build(
        cls,
        items: Iterable[Any],
        encoder: SearchEncoder,
        *,
        cache_dir: Path | str | None = None,
        backend: str = FAISS_BACKEND,
        extra_meta: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> "SearchVectorIndex":
        """用物品的 caption token（可选类目）建索引。

        ``items`` 需要 ``item_id`` 与 ``caption_tokens`` 属性，类目可选
        （``recsys.search.bm25.IndexedItem`` 已满足）。
        """
        items = list(items)
        if not items:
            raise VectorIndexError("建向量索引至少需要一个物品")
        item_ids = [int(item.item_id) for item in items]
        if len(set(item_ids)) != len(item_ids):
            raise VectorIndexError("物品 ID 不能重复")
        matrix = np.vstack(
            [
                encoder.encode_document(
                    getattr(item, "caption_tokens", ()),
                    getattr(item, "category", ""),
                )
                for item in items
            ]
        )
        if backend == FAISS_BACKEND:
            index = FaissFlatIndex.build(matrix, item_ids)
        elif backend == NUMPY_BACKEND:
            index = FlatIndex.build(matrix, [str(value) for value in item_ids])
        else:
            raise VectorIndexError(f"未知向量后端 {backend!r}；可选 faiss / numpy")

        meta = dict(extra_meta or {})
        meta["vocab"] = sorted(
            {
                int(token)
                for item in items
                for token in getattr(item, "caption_tokens", ())
            }
        )
        facade = cls(
            index,
            encoder,
            backend=backend,
            cache_dir=Path(cache_dir) if cache_dir else None,
            extra_meta=meta,
        )
        if persist and cache_dir is not None:
            facade.save(cache_dir)
        return facade

    @classmethod
    def load(
        cls,
        cache_dir: Path | str,
        encoder: SearchEncoder | None = None,
        *,
        strict_encoder: bool = True,
    ) -> "SearchVectorIndex":
        """从磁盘重载索引；默认校验编码器配置。"""
        cache_dir = Path(cache_dir)
        if not vector_index_exists(str(cache_dir)):
            raise VectorIndexError(
                f"找不到完整的搜索向量索引：{cache_dir}；"
                "先运行 python -m recsys.search.build_vector_index"
            )
        index = load_vector_index(str(cache_dir))
        if index is None:
            raise VectorIndexError(f"搜索向量索引重载失败：{cache_dir}")
        backend = FAISS_BACKEND if isinstance(index, FaissFlatIndex) else NUMPY_BACKEND
        meta = dict(getattr(index, "meta", {}) or {})
        built_encoder = meta.get("encoder") or {}
        built_hash = str(meta.get("encoder_config_hash") or "")
        if encoder is not None and strict_encoder and built_encoder:
            current = encoder.describe()
            if built_encoder.get("encoder") != current.get("encoder"):
                raise VectorIndexError(
                    f"索引由 {built_encoder.get('encoder')!r} 构建，当前编码器 "
                    f"{current.get('encoder')!r}；请重建索引"
                )
            if int(built_encoder.get("dim", 0)) != int(encoder.dim):
                raise VectorIndexError(
                    f"索引维度 {built_encoder.get('dim')} 与当前编码器 {encoder.dim} 不一致"
                )
            if built_hash and built_hash != encoder.config_hash():
                raise VectorIndexError(
                    "索引的编码器配置哈希与当前编码器不一致；"
                    "查询和文档必须使用同一套编码配置，请重建索引"
                )
        if encoder is None:
            if built_encoder.get("encoder") != "token_hash":
                raise VectorIndexError(
                    "重载索引需要显式传入与建索引一致的编码器"
                    f"（索引记录：{built_encoder.get('encoder')!r}）"
                )
            from recsys.search.encoders import build_encoder

            encoder = build_encoder(
                {
                    "name": "token_hash",
                    "dim": int(built_encoder.get("dim", 0)),
                    "seed": int(built_encoder.get("seed", 0)),
                    "include_category": bool(built_encoder.get("include_category", True)),
                    "category_weight": float(built_encoder.get("category_weight", 0.0)),
                }
            )
        try:
            return cls(
                index,
                encoder,
                backend=backend,
                cache_dir=cache_dir,
                extra_meta={key: value for key, value in meta.items()
                            if key not in {"backend", "n", "format"}},
            )
        except VectorIndexError:
            raise

    def save(self, cache_dir: Path | str | None = None) -> Path:
        target = Path(cache_dir) if cache_dir is not None else self.cache_dir
        if target is None:
            raise VectorIndexError("没有指定索引目录")
        # 注意：上游 save_index 自己写 ``backend``（索引类的真实 backend 名），
        # 这里的请求后端必须用别的键，否则会覆盖掉上游的 backend 导致无法重载。
        meta = {
            key: value
            for key, value in self.extra_meta.items()
            if key not in {"backend", "n", "format"}
        }
        meta.update(
            {
                "format": INDEX_META_FORMAT,
                "encoder": self.encoder.describe(),
                "encoder_config_hash": self.encoder.config_hash(),
                "dim": self.dim,
                "requested_backend": self.backend,
            }
        )
        save_vector_index(self._index, str(target), extra_meta=meta)
        self.cache_dir = target
        return target

    # -- 查询与更新 ---------------------------------------------------------------

    def encode_query(self, tokens: Sequence[int]) -> np.ndarray:
        """查询编码：先按索引词表过滤 OOV token，再走与文档相同的编码器。"""
        values = [int(token) for token in tokens]
        if self._vocab:
            values = [token for token in values if token in self._vocab]
        return self.encoder.encode_tokens(values)

    def search_scored(self, tokens: Sequence[int], top_k: int = 10) -> list[VectorHit]:
        """向量 Top-K；返回整数 item_id 与余弦分数，分数降序。"""
        if top_k <= 0 or self.num_items == 0:
            return []
        query_vector = self.encode_query(tokens)
        if not np.any(query_vector):
            return []
        pairs = self._scored_pairs(query_vector, min(int(top_k), self.num_items))
        return [
            VectorHit(rank=rank, item_id=item_id, score=score)
            for rank, (item_id, score) in enumerate(pairs, start=1)
        ]

    def _scored_pairs(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        if isinstance(self._index, FaissFlatIndex):
            pairs = self._index.search_scored(query_vector, k)
        else:
            # NumPy 后端：上游 FlatIndex 只返回 doc id，这里补算余弦分数。
            ids = self._index.search(query_vector, k)
            matrix = np.asarray(self._index.emb, dtype=np.float32)
            query = np.asarray(query_vector, dtype=np.float32)
            pairs = [
                (int(doc_id), float(np.dot(matrix[self._positions[int(doc_id)]], query)))
                for doc_id in ids
            ]
        # 余弦为 0 说明查询与文档在编码空间里没有任何共同特征（哈希 token 不重合）；
        # 保留它们会让「无匹配查询」看起来有结果，因此只返回正相似度。
        pairs = [(item_id, score) for item_id, score in pairs if score > 0.0]
        pairs.sort(key=lambda row: (-row[1], row[0]))
        return pairs

    def search(self, tokens: Sequence[int], top_k: int = 10) -> list[int]:
        return [hit.item_id for hit in self.search_scored(tokens, top_k)]

    def add_items(self, items: Iterable[Any]) -> int:
        """增量更新：追加物品并刷新映射；返回新增数量。"""
        items = list(items)
        if not items:
            return 0
        item_ids = [int(item.item_id) for item in items]
        duplicates = set(item_ids) & set(self._item_ids)
        if duplicates:
            raise VectorIndexError(f"物品已在索引中：{sorted(duplicates)[:5]}")
        matrix = np.vstack(
            [
                self.encoder.encode_document(
                    getattr(item, "caption_tokens", ()),
                    getattr(item, "category", ""),
                )
                for item in items
            ]
        )
        if isinstance(self._index, FaissFlatIndex):
            self._index.add(matrix, item_ids)
        else:
            combined = np.vstack(
                [np.asarray(self._index.emb, dtype=np.float32), matrix]
            )
            combined_ids = self._item_ids + item_ids
            self._index = FlatIndex.build(combined, [str(value) for value in combined_ids])
        self._item_ids.extend(item_ids)
        self._positions = {
            item_id: position for position, item_id in enumerate(self._item_ids)
        }
        self._vocab.update(
            int(token)
            for item in items
            for token in getattr(item, "caption_tokens", ())
        )
        self.extra_meta["vocab"] = sorted(self._vocab)
        return len(item_ids)

    # -- 描述 ---------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "backend_actual": self.backend_actual,
            "faiss_active": self.faiss_active,
            "faiss_version": faiss_version(),
            "faiss_search_calls": self.faiss_search_calls,
            "num_items": self.num_items,
            "dim": self.dim,
            "vocab_size": len(self._vocab),
            "oov_filter": bool(self._vocab),
            "encoder": self.encoder.describe(),
            "encoder_config_hash": self.encoder.config_hash(),
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "index_meta": {
                key: value
                for key, value in self.extra_meta.items()
                if key != "vocab"
            },
        }


def _item_ids_of(index: VectorIndex) -> list[int]:
    item_ids = getattr(index, "item_ids", None)
    if item_ids is None:
        item_ids = [int(doc_id) for doc_id in index.doc_ids]
    return [int(value) for value in item_ids]


__all__ = [
    "FAISS_BACKEND",
    "FaissFlatIndex",
    "NUMPY_BACKEND",
    "SearchVectorIndex",
    "VectorHit",
    "VectorIndexError",
]
