"""RQ-VAE（残差向量量化）与 Semantic ID 构建组件。

本项目没有现成的物品视觉/文本 embedding，因此用物品元数据（一级/二级类目、
item_type、upload_type、作者频次、caption 长度、上传日期、is_ad）构造固定的
内容向量，再用 RQ-VAE 做残差量化，得到层级 Semantic ID。这是生成式召回的
``RQ-VAE / Semantic ID`` 组件；TIGER 的生成器直接消费这里产出的码本与 SID。

上游参考：``third_party/TorchEasyRec/tzrec/models/sid_rqvae.py`` 与
``third_party/genrec/genrec/models/rqvae.py``。两者都依赖各自平台的训练栈，
这里按论文重写最小可运行版本：编码器 MLP -> 3 级残差码本 -> 解码器 MLP。

数据限制（明确记录）：KuaiSAR 本地快照没有视频封面/文本 embedding，
所以本组件的“多模态内容向量”实际上是**元数据向量**，不是真正的视觉/文本特征。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CONTENT_DIM = 32
LATENT_DIM = 16
NUM_LEVELS = 3
CODEBOOK_SIZE = 64
CATEGORICAL_COLUMNS = (
    "category_first_code",
    "category_second_code",
    "item_type_code",
    "upload_type_code",
)
CATEGORICAL_CARDINALITIES = (38, 294, 4, 30)  # 含 0 号“未知/缺失”
NUMERIC_COLUMNS = (
    "author_freq",
    "caption_tokens",
    "upload_days",
    "is_ad",
    "upload_valid",
)


def build_content_vectors(
    item_features,
    *,
    dim: int = CONTENT_DIM,
    seed: int = 0,
) -> np.ndarray:
    """把物品元数据压成 ``[num_items, dim]`` 的 L2 归一化内容向量。

    类目字段用固定的随机投影（等价于固定随机 embedding），数值字段先做
    log1p/缩放再投影。投影矩阵由 seed 固定，因此同一份特征表每次得到同一
    组向量；它**不是**训练得到的，避免在没有监督信号时过拟合。
    """
    if item_features is None:
        raise ValueError("数据集没有 item_features，无法构建 Semantic ID 内容向量")
    rng = np.random.default_rng(seed)
    num_items = int(item_features.num_items)
    vectors = np.zeros((num_items, dim), dtype=np.float32)

    scale = 1.0 / np.sqrt(dim)
    for name, cardinality in zip(CATEGORICAL_COLUMNS, CATEGORICAL_CARDINALITIES):
        projection = rng.normal(0.0, scale, size=(cardinality, dim)).astype(np.float32)
        codes = np.asarray(item_features.column(name)).astype(np.int64)
        codes = np.where((codes >= 0) & (codes < cardinality), codes, 0)
        vectors += projection[codes]

    numeric = np.stack(
        [
            np.log1p(np.clip(np.asarray(item_features.column("author_freq"), dtype=np.float32), 0, None)),
            np.log1p(np.clip(np.asarray(item_features.column("caption_tokens"), dtype=np.float32), 0, None)),
            np.clip(np.asarray(item_features.column("upload_days"), dtype=np.float32), 0, None) / 365.25,
            np.asarray(item_features.column("is_ad"), dtype=np.float32),
            np.asarray(item_features.column("upload_valid"), dtype=np.float32),
        ],
        axis=1,
    )
    numeric = (numeric - numeric.mean(axis=0, keepdims=True)) / (numeric.std(axis=0, keepdims=True) + 1e-6)
    numeric_projection = rng.normal(0.0, scale, size=(len(NUMERIC_COLUMNS), dim)).astype(np.float32)
    vectors += numeric @ numeric_projection

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class RQVAEConfig:
    input_dim: int = CONTENT_DIM
    latent_dim: int = LATENT_DIM
    num_levels: int = NUM_LEVELS
    codebook_size: int = CODEBOOK_SIZE
    hidden_dim: int = 64
    commitment: float = 0.25

    def as_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "num_levels": self.num_levels,
            "codebook_size": self.codebook_size,
            "hidden_dim": self.hidden_dim,
            "commitment": self.commitment,
        }


class RQVAE(nn.Module):
    """残差量化自编码器；``codes`` 即 Semantic ID 的各级 token。"""

    def __init__(
        self,
        input_dim: int = CONTENT_DIM,
        latent_dim: int = LATENT_DIM,
        num_levels: int = NUM_LEVELS,
        codebook_size: int = CODEBOOK_SIZE,
        hidden_dim: int = 64,
        commitment: float = 0.25,
    ) -> None:
        super().__init__()
        self.config = RQVAEConfig(
            input_dim, latent_dim, num_levels, codebook_size, hidden_dim, commitment
        )
        self.encoder = MLP(input_dim, hidden_dim, latent_dim)
        self.decoder = MLP(latent_dim, hidden_dim, input_dim)
        self.codebooks = nn.Parameter(
            torch.randn(num_levels, codebook_size, latent_dim) * 0.1
        )

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(F.normalize(x, dim=1))

    def quantize(self, latent: torch.Tensor):
        residual = latent
        quantized = torch.zeros_like(latent)
        codes: list[torch.Tensor] = []
        for level in range(self.config.num_levels):
            book = self.codebooks[level]  # [C, D]
            distances = (
                residual.pow(2).sum(dim=1, keepdim=True)
                - 2 * residual @ book.T
                + book.pow(2).sum(dim=1).unsqueeze(0)
            )
            index = distances.argmin(dim=1)
            entry = book[index]
            quantized = quantized + entry
            residual = residual - entry
            codes.append(index)
        return quantized, codes, residual

    def forward(self, x: torch.Tensor):
        latent = self.encode_latent(x)
        quantized, codes, residual = self.quantize(latent)
        # straight-through：解码器看到量化值，编码器拿到重建梯度。
        straight_through = latent + (quantized - latent).detach()
        reconstruction = self.decoder(straight_through)
        return {
            "reconstruction": reconstruction,
            "latent": latent,
            "quantized": quantized,
            "residual": residual,
            "codes": codes,
        }

    def loss(self, batch: torch.Tensor, outputs: dict) -> tuple[torch.Tensor, dict[str, float]]:
        recon = F.mse_loss(outputs["reconstruction"], F.normalize(batch, dim=1))
        commit = F.mse_loss(outputs["residual"].detach(), outputs["latent"]) + F.mse_loss(
            outputs["latent"].detach(), outputs["residual"]
        )
        total = recon + self.config.commitment * commit
        return total, {"reconstruction": float(recon.item()), "commitment": float(commit.item())}

    @torch.no_grad()
    def encode_codes(self, x: torch.Tensor, batch_size: int = 8192) -> np.ndarray:
        """返回 ``[num_items, num_levels]`` 的 int32 Semantic ID 矩阵。"""
        self.eval()
        chunks: list[np.ndarray] = []
        for start in range(0, x.shape[0], batch_size):
            latent = self.encode_latent(x[start : start + batch_size])
            _, codes, _ = self.quantize(latent)
            chunks.append(torch.stack(codes, dim=1).cpu().numpy().astype(np.int32))
        return np.concatenate(chunks, axis=0)


def count_unique_sids(codes: np.ndarray) -> int:
    return len({tuple(row) for row in np.asarray(codes).tolist()})


def build_semantic_ids_kmeans(
    vectors: np.ndarray,
    *,
    num_levels: int = NUM_LEVELS,
    codebook_size: int = CODEBOOK_SIZE,
    seed: int = 42,
) -> np.ndarray:
    """RQ-KMeans 兜底：对残差逐级做 MiniBatchKMeans，得到层级码。

    RQ-VAE 在小样本/短训练下可能码本塌缩（大量物品共享同一 SID）。RQ-KMeans
    与 RQ-VAE 同属残差量化，只是用聚类中心代替可学习码本；当 RQ-VAE 的
    唯一 SID 数明显偏低时，导出码表会退化到本方法，并在产物里记录选择原因。
    """
    from sklearn.cluster import MiniBatchKMeans

    residual = np.asarray(vectors, dtype=np.float32).copy()
    codes = np.zeros((residual.shape[0], num_levels), dtype=np.int32)
    for level in range(num_levels):
        clusters = min(int(codebook_size), residual.shape[0])
        if clusters <= 1:
            break
        model = MiniBatchKMeans(
            n_clusters=clusters,
            random_state=seed + level,
            n_init=3,
            batch_size=8192,
            max_iter=100,
        )
        labels = model.fit_predict(residual)
        codes[:, level] = labels.astype(np.int32)
        residual -= model.cluster_centers_[labels]
    return codes


def save_rqvae(path: Path, model: RQVAE, *, codes: np.ndarray, provenance: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config.as_dict(),
            "codes": torch.from_numpy(codes.astype(np.int64)),
            "provenance": provenance or {},
        },
        path,
    )


def load_rqvae(path: Path, device: torch.device) -> tuple[RQVAE, np.ndarray, dict]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 RQ-VAE checkpoint：{path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = RQVAE(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    codes = checkpoint.get("codes")
    codes_array = codes.cpu().numpy().astype(np.int32) if codes is not None else np.zeros((0, 0), dtype=np.int32)
    return model, codes_array, checkpoint


class SemanticIDIndex:
    """SID 元组 → 物品列表 + 前缀集合（供约束解码使用）。"""

    def __init__(self, codes: np.ndarray) -> None:
        if codes.ndim != 2 or codes.shape[0] == 0:
            raise ValueError("codes 必须是 [num_items, num_levels] 的非空矩阵")
        self.codes = np.asarray(codes, dtype=np.int64)
        self.num_levels = int(codes.shape[1])
        self.sid_to_items: dict[tuple[int, ...], list[int]] = {}
        self.prefixes: list[set[tuple[int, ...]]] = [set() for _ in range(self.num_levels + 1)]
        self.children: list[dict[tuple[int, ...], set[int]]] = [
            {} for _ in range(self.num_levels)
        ]
        self.prefixes[0].add(())
        for item, row in enumerate(self.codes):
            sid = tuple(int(value) for value in row)
            self.sid_to_items.setdefault(sid, []).append(item)
            for level in range(1, self.num_levels + 1):
                self.prefixes[level].add(sid[:level])
            for level in range(self.num_levels):
                self.children[level].setdefault(sid[:level], set()).add(sid[level])

    def valid_next(self, prefix: tuple[int, ...], level: int) -> list[int]:
        """在 ``level`` 层，给定前缀后合法的下一个 token（预计算，O(1) 查表）。"""
        return sorted(self.children[level].get(tuple(prefix), ()))

    def items_for(self, sid: tuple[int, ...]) -> list[int]:
        return self.sid_to_items.get(tuple(sid), [])

    @property
    def num_sids(self) -> int:
        return len(self.sid_to_items)


__all__ = [
    "CATEGORICAL_CARDINALITIES",
    "CATEGORICAL_COLUMNS",
    "CODEBOOK_SIZE",
    "CONTENT_DIM",
    "LATENT_DIM",
    "NUMERIC_COLUMNS",
    "NUM_LEVELS",
    "RQVAE",
    "RQVAEConfig",
    "SemanticIDIndex",
    "build_content_vectors",
    "build_semantic_ids_kmeans",
    "count_unique_sids",
    "load_rqvae",
    "save_rqvae",
]
