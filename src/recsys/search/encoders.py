"""可插拔的搜索向量编码器。

当前 KuaiSAR 的查询和视频描述都是**匿名整数 token**（论文说明：分词后随机哈希成
整数），没有自然语言文本，也没有官方词表。因此本轮先用**无需训练的本地 token
向量编码**跑通链路：

``TokenHashEncoder``
    把整数 token 经稳定哈希（blake2b，Python 进程间一致，不用内置 ``hash``）映射到
    固定维度、带符号的稀疏→稠密向量，按次线性词频加权后 L2 归一化。查询和文档
    **共用同一个编码器实例与同一套配置**（同一维度、同一 seed、同一 token 通道），
    所以余弦相似度的本质是「哈希后的加权 token 重合」，不是中文语义。整数 token
    转成字符串只是通道键（``t:<id>``），**不当作原始中文语义**。

``TextSemanticEncoder``
    真实文本 + 本地预训练权重时启用的语义编码适配器。当前仓库没有可用的
    KuaiSAR 文本字段，``sentence-transformers`` 也未安装，所以它会在构造时
    明确报错并列出缺口；接口和数据契约先保留，等有文本时直接换配置即可。

两个编码器都实现 ``SearchEncoder``：``encode_tokens`` 走查询侧，
``encode_document`` 走文档侧。``describe()`` 随索引一起落盘，重载时校验编码配置，
避免用不同编码器查同一个索引。
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

DEFAULT_DIM = 256
DEFAULT_SEED = 20261109
DEFAULT_CATEGORY_WEIGHT = 0.25


class EncoderUnavailable(RuntimeError):
    """编码器所需的数据/依赖/权重缺失；附带具体缺口。"""


def _stable_hash(key: str, seed: int) -> int:
    payload = f"{seed}:{key}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


class SearchEncoder(ABC):
    """查询与文档共用的编码契约。"""

    name: str = "base"
    #: 是否需要训练/微调；本地 token 编码为 False。
    requires_training: bool = False
    #: 是否是真实文本语义编码。
    text_semantic: bool = False

    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @abstractmethod
    def encode_tokens(self, tokens: Iterable[int]) -> np.ndarray:
        """把查询 token 编码成 L2 归一化的 float32 向量。"""

    @abstractmethod
    def encode_document(self, tokens: Iterable[int], category: str = "") -> np.ndarray:
        """把物品 token（可选类目文本）编码成同一空间的 float32 向量。"""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        ...

    def config_hash(self) -> str:
        payload = json.dumps(self.describe(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return vector
    return (vector / norm).astype(np.float32)


class TokenHashEncoder(SearchEncoder):
    """无需训练的整数 token 哈希编码；查询与文档共用同一配置。"""

    name = "token_hash"
    requires_training = False
    text_semantic = False

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        seed: int = DEFAULT_SEED,
        include_category: bool = True,
        category_weight: float = DEFAULT_CATEGORY_WEIGHT,
    ) -> None:
        dim = int(dim)
        if dim <= 0:
            raise ValueError(f"encoder dim 必须 > 0，实际 {dim}")
        self._dim = dim
        self.seed = int(seed)
        self.include_category = bool(include_category)
        self.category_weight = float(category_weight)
        if self.category_weight < 0:
            raise ValueError("category_weight 必须 >= 0")

    @property
    def dim(self) -> int:
        return self._dim

    def _add_key(self, vector: np.ndarray, key: str, weight: float) -> None:
        hashed = _stable_hash(key, self.seed)
        position = hashed % self._dim
        sign = 1.0 if (hashed >> 63) & 1 else -1.0
        vector[position] += sign * weight

    def _token_channel(self, vector: np.ndarray, tokens: Iterable[int]) -> None:
        counts = Counter(int(token) for token in tokens)
        for token, tf in counts.items():
            # 次线性词频，避免长 caption 单纯靠重复刷分。
            self._add_key(vector, f"t:{token}", 1.0 + math.log(tf))

    def encode_tokens(self, tokens: Iterable[int]) -> np.ndarray:
        vector = np.zeros(self._dim, dtype=np.float32)
        self._token_channel(vector, tokens)
        return _normalize(vector)

    def encode_document(self, tokens: Iterable[int], category: str = "") -> np.ndarray:
        vector = np.zeros(self._dim, dtype=np.float32)
        self._token_channel(vector, tokens)
        if self.include_category and category:
            words = [word for word in str(category).lower().replace("/", " ").split() if word]
            counts = Counter(words)
            for word, tf in counts.items():
                self._add_key(vector, f"w:{word}", self.category_weight * (1.0 + math.log(tf)))
        return _normalize(vector)

    def describe(self) -> dict[str, Any]:
        return {
            "encoder": self.name,
            "dim": self._dim,
            "seed": self.seed,
            "include_category": self.include_category,
            "category_weight": self.category_weight,
            "requires_training": self.requires_training,
            "text_semantic": self.text_semantic,
            "note": (
                "KuaiSAR 匿名整数 token 的稳定哈希编码；同一编码配置用于查询与文档，"
                "余弦相似度≈加权 token 重合，不代表中文语义。"
            ),
        }


class TextSemanticEncoder(SearchEncoder):
    """真实文本 + 本地预训练权重时的语义编码适配器（当前数据不具备条件）。"""

    name = "sentence_transformer"
    requires_training = False
    text_semantic = True

    def __init__(
        self,
        model_path: str | Path,
        *,
        query_prefix: str = "query: ",
        document_prefix: str = "passage: ",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        missing: list[str] = []
        path = Path(model_path) if model_path else None
        if path is None or not path.exists():
            missing.append(f"本地权重目录不存在：{model_path!r}")
        try:
            import sentence_transformers  # noqa: F401
        except Exception:  # noqa: BLE001
            missing.append("未安装 sentence-transformers")
        if missing:
            raise EncoderUnavailable(
                "文本语义编码不可用：" + "；".join(missing)
                + "。KuaiSAR 当前只有匿名整数 token，没有 caption/query 原文，"
                "也没有可用的本地预训练权重；请继续使用 token_hash 编码。"
            )
        from sentence_transformers import SentenceTransformer

        self.model_path = str(path)
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = int(batch_size)
        self.device = device
        self._model = SentenceTransformer(self.model_path, device=device)

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def _encode(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [np.asarray(vector, dtype=np.float32) for vector in vectors]

    def encode_tokens(self, tokens: Iterable[int]) -> np.ndarray:
        raise EncoderUnavailable(
            "该编码器只接受真实文本；KuaiSAR 的查询是匿名整数 token，"
            "不能把整数 token 当成中文语义编码。"
        )

    def encode_document(self, tokens: Iterable[int], category: str = "") -> np.ndarray:
        raise EncoderUnavailable(
            "该编码器只接受真实文本；KuaiSAR 的 caption 是匿名整数 token。"
        )

    def encode_texts(self, texts: list[str], *, query: bool = False) -> list[np.ndarray]:
        """真实文本入口（有文本语料时使用）。"""
        prefix = self.query_prefix if query else self.document_prefix
        return self._encode([prefix + text for text in texts])

    def describe(self) -> dict[str, Any]:
        return {
            "encoder": self.name,
            "model_path": self.model_path,
            "dim": self.dim,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "requires_training": self.requires_training,
            "text_semantic": self.text_semantic,
        }


_ENCODERS: dict[str, type[SearchEncoder]] = {
    TokenHashEncoder.name: TokenHashEncoder,
    TextSemanticEncoder.name: TextSemanticEncoder,
}


def register_encoder(cls: type[SearchEncoder]) -> type[SearchEncoder]:
    """注册新的编码器实现（可插拔接口）。"""
    _ENCODERS[cls.name] = cls
    return cls


def encoder_names() -> list[str]:
    return sorted(_ENCODERS)


def build_encoder(config: Mapping[str, Any] | None = None) -> SearchEncoder:
    """按配置构造编码器；未知编码器直接报错，不回退默认值。"""
    config = dict(config or {})
    name = str(config.get("name") or config.get("encoder") or TokenHashEncoder.name).strip()
    if name not in _ENCODERS:
        raise ValueError(f"未知编码器 {name!r}；可选：{encoder_names()}")
    cls = _ENCODERS[name]
    if cls is TokenHashEncoder:
        return TokenHashEncoder(
            dim=int(config.get("dim", DEFAULT_DIM)),
            seed=int(config.get("seed", DEFAULT_SEED)),
            include_category=bool(config.get("include_category", True)),
            category_weight=float(config.get("category_weight", DEFAULT_CATEGORY_WEIGHT)),
        )
    if cls is TextSemanticEncoder:
        return TextSemanticEncoder(
            model_path=config.get("model_path", ""),
            query_prefix=str(config.get("query_prefix", "query: ")),
            document_prefix=str(config.get("document_prefix", "passage: ")),
            batch_size=int(config.get("batch_size", 32)),
            device=config.get("device"),
        )
    return cls(**{key: value for key, value in config.items() if key != "name"})  # type: ignore[call-arg]


__all__ = [
    "DEFAULT_CATEGORY_WEIGHT",
    "DEFAULT_DIM",
    "DEFAULT_SEED",
    "EncoderUnavailable",
    "SearchEncoder",
    "TextSemanticEncoder",
    "TokenHashEncoder",
    "build_encoder",
    "encoder_names",
    "register_encoder",
]
