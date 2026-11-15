"""训练 RQ-VAE 并导出物品 Semantic ID（生成式召回的表示层）。"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from recsys.config import DEFAULT_DATASET
from recsys.data.loader import SUPPORTED_DATASETS, prepare_dataset, resolve_positive_threshold
from recsys.models.rqvae import (
    CODEBOOK_SIZE,
    CONTENT_DIM,
    LATENT_DIM,
    NUM_LEVELS,
    RQVAE,
    build_content_vectors,
    build_semantic_ids_kmeans,
    count_unique_sids,
    save_rqvae,
)
from recsys.training.trainer import resolve_device
from recsys.utils.checkpoint import json_ready_config
from recsys.utils.provenance import build_provenance
from recsys.utils.seed import set_random_seed


@dataclass(frozen=True)
class RQVAETrainConfig:
    dataset: str = DEFAULT_DATASET
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    epochs: int = 8
    batch_size: int = 4096
    learning_rate: float = 1e-3
    content_dim: int = CONTENT_DIM
    latent_dim: int = LATENT_DIM
    num_levels: int = NUM_LEVELS
    codebook_size: int = CODEBOOK_SIZE
    hidden_dim: int = 64
    seed: int = 42
    device: str = "auto"
    checkpoint_name: str = "rqvae"


def train_rqvae(config: RQVAETrainConfig) -> dict:
    set_random_seed(config.seed)
    device = resolve_device(config.device)
    data = prepare_dataset(
        dataset=config.dataset,
        data_dir=config.data_dir,
        positive_threshold=resolve_positive_threshold(config.dataset, None),
    )
    print(f"Device: {device}; 构建物品内容向量 …")
    started = time.perf_counter()
    vectors = build_content_vectors(data.item_features, dim=config.content_dim, seed=config.seed)
    tensor = torch.from_numpy(vectors).to(device)
    print(f"内容向量 {tuple(tensor.shape)}，耗时 {time.perf_counter() - started:.1f}s")

    model = RQVAE(
        input_dim=config.content_dim,
        latent_dim=config.latent_dim,
        num_levels=config.num_levels,
        codebook_size=config.codebook_size,
        hidden_dim=config.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    history: list[dict] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = torch.randperm(tensor.shape[0], generator=generator)
        totals = {"loss": 0.0, "reconstruction": 0.0, "commitment": 0.0}
        batches = 0
        for start in range(0, tensor.shape[0], config.batch_size):
            index = permutation[start : start + config.batch_size]
            batch = tensor[index]
            outputs = model(batch)
            loss, parts = model.loss(batch, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.item())
            totals["reconstruction"] += parts["reconstruction"]
            totals["commitment"] += parts["commitment"]
            batches += 1
        row = {key: value / max(1, batches) for key, value in totals.items()}
        row["epoch"] = epoch
        history.append(row)
        print(
            f"epoch={epoch:02d} loss={row['loss']:.6f} "
            f"recon={row['reconstruction']:.6f} commit={row['commitment']:.6f}"
        )

    codes = model.encode_codes(tensor)
    rqvae_unique = count_unique_sids(codes)
    kmeans_codes = build_semantic_ids_kmeans(
        vectors,
        num_levels=config.num_levels,
        codebook_size=config.codebook_size,
        seed=config.seed,
    )
    kmeans_unique = count_unique_sids(kmeans_codes)
    if kmeans_unique > rqvae_unique:
        exported_codes = kmeans_codes
        code_method = "rq_kmeans_fallback"
        code_selection_reason = (
            f"RQ-VAE 唯一 SID={rqvae_unique} 明显少于 RQ-KMeans={kmeans_unique}，"
            "说明码本塌缩；导出码表使用 RQ-KMeans"
        )
    else:
        exported_codes = codes
        code_method = "rqvae"
        code_selection_reason = f"RQ-VAE 唯一 SID={rqvae_unique} ≥ RQ-KMeans={kmeans_unique}"
    used_codes = exported_codes
    unique_sids = count_unique_sids(used_codes)
    code_usage = [
        len(set(used_codes[:, level].tolist())) for level in range(used_codes.shape[1])
    ]
    checkpoint_path = config.artifacts_dir / config.dataset / f"{config.checkpoint_name}.pt"
    save_rqvae(
        checkpoint_path,
        model,
        codes=used_codes,
        provenance={
            "data": build_provenance(data, dataset=config.dataset, positive_threshold=1.0),
            "train_config": json_ready_config(config),
            "history": history,
            "unique_sids": unique_sids,
            "rqvae_unique_sids": rqvae_unique,
            "rq_kmeans_unique_sids": kmeans_unique,
            "code_method": code_method,
            "code_selection_reason": code_selection_reason,
            "code_usage_per_level": code_usage,
            "note": "内容向量来自物品元数据，本地数据没有视觉/文本 embedding",
        },
    )
    print(
        f"Checkpoint saved to {checkpoint_path}; method={code_method}; "
        f"unique SIDs={unique_sids:,} (rqvae={rqvae_unique}, kmeans={kmeans_unique}); "
        f"per-level usage={code_usage}"
    )
    return {
        "checkpoint": str(checkpoint_path),
        "code_method": code_method,
        "unique_sids": unique_sids,
        "rqvae_unique_sids": rqvae_unique,
        "rq_kmeans_unique_sids": kmeans_unique,
        "code_usage_per_level": code_usage,
        "history": history,
        "codes": used_codes,
    }


def parse_args() -> RQVAETrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--content-dim", type=int, default=CONTENT_DIM)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--num-levels", type=int, default=NUM_LEVELS)
    parser.add_argument("--codebook-size", type=int, default=CODEBOOK_SIZE)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-name", default="rqvae")
    return RQVAETrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train_rqvae(parse_args())
