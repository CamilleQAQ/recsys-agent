"""训练 LightGCN（BPR 目标），沿用项目的统一数据/评测/工件契约。"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from recsys.config import DEFAULT_DATASET
from recsys.data.loader import (
    SUPPORTED_DATASETS,
    prepare_dataset,
    resolve_positive_threshold,
)
from recsys.data.schema import InteractionDataset, PreparedData, seen_items_for_split
from recsys.evaluation.evaluator import (
    brute_force_top_k,
    evaluate_recommendations,
    get_evaluation_interactions,
    group_items_by_user,
)
from recsys.models.lightgcn import (
    LightGCN,
    bpr_loss,
    build_normalized_adjacency,
    edges_from_interactions,
)
from recsys.training.samplers import sample_negative_items
from recsys.training.trainer import create_dataloader, resolve_device
from recsys.utils.checkpoint import json_ready_config
from recsys.utils.provenance import build_provenance
from recsys.utils.seed import set_random_seed


@dataclass(frozen=True)
class LightGCNTrainConfig:
    dataset: str = DEFAULT_DATASET
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    epochs: int = 10
    batch_size: int = 2048
    learning_rate: float = 1e-3
    embedding_dim: int = 64
    num_layers: int = 2
    reg_weight: float = 1e-4
    validation_users: int = 0
    positive_threshold: float | None = None
    seed: int = 42
    device: str = "auto"
    max_batches_per_epoch: int | None = None
    checkpoint_name: str = "lightgcn"


def evaluate_lightgcn(
    model: LightGCN,
    adjacency: torch.Tensor,
    data: PreparedData,
    split: str,
    *,
    ks: tuple[int, ...] = (10, 20),
    device: torch.device,
) -> dict[str, float]:
    """用传播后的用户/物品表示做全量精确排序并计算统一指标。"""
    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model.split_embeddings(adjacency)
    interactions = get_evaluation_interactions(data, split)
    users = sorted(group_items_by_user(interactions))
    users_tensor = torch.tensor(users, dtype=torch.long, device=device)
    recommendations = brute_force_top_k(
        user_indices=users,
        user_embeddings=user_embeddings[users_tensor],
        item_embeddings=item_embeddings,
        train_items_by_user=seen_items_for_split(data, split),
        k=max(ks),
    )
    metrics = evaluate_recommendations(recommendations, interactions, ks)
    from recsys.evaluation.metrics import catalog_coverage

    metrics["Coverage"] = catalog_coverage(recommendations, data.num_items)
    return metrics


def train_lightgcn(config: LightGCNTrainConfig) -> tuple[LightGCN, list[float]]:
    set_random_seed(config.seed)
    device = resolve_device(config.device)
    positive_threshold = resolve_positive_threshold(
        config.dataset, config.positive_threshold
    )
    data = prepare_dataset(
        dataset=config.dataset,
        data_dir=config.data_dir,
        positive_threshold=positive_threshold,
    )
    from recsys.training.subsets import limit_validation_users

    data = limit_validation_users(data, config.validation_users, config.seed)
    users, items = edges_from_interactions(data.train_interactions)
    adjacency = build_normalized_adjacency(
        users, items, data.num_users, data.num_items, device
    )
    model = LightGCN(
        num_users=data.num_users,
        num_items=data.num_items,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loader = create_dataloader(
        InteractionDataset(data.train_interactions), config.batch_size, config.seed
    )
    train_items_by_user = group_items_by_user(data.train_interactions)
    negative_generator = torch.Generator().manual_seed(config.seed)

    print(f"Device: {device}")
    print(
        f"Dataset: {config.dataset} (positive threshold {positive_threshold:g}); "
        f"graph edges={int(adjacency._nnz()):,}, layers={config.num_layers}"
    )
    losses: list[float] = []
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    best_state = None
    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for batch_number, batch in enumerate(loader):
            if (
                config.max_batches_per_epoch is not None
                and batch_number >= config.max_batches_per_epoch
            ):
                break
            batch_users = batch["user_idx"]
            positive_items = batch["item_idx"]
            negative_items = sample_negative_items(
                batch_users, train_items_by_user, data.num_items, negative_generator
            )
            user_embeddings, item_embeddings = model.split_embeddings(adjacency)
            loss = bpr_loss(
                user_embeddings[batch_users.to(device)],
                item_embeddings[positive_items.to(device)],
                item_embeddings[negative_items.to(device)],
                reg_weight=config.reg_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1
        if batches == 0:
            raise RuntimeError("训练 DataLoader 没有产生任何 batch")
        average = total / batches
        losses.append(average)
        print(f"Epoch {epoch:02d}/{config.epochs:02d} - loss: {average:.6f}")
        if data.validation_interactions:
            metrics = evaluate_lightgcn(
                model, adjacency, data, "validation", device=device
            )
            print(f"  validation Recall@20: {metrics['Recall@20']:.6f}")
            if not best_metrics or metrics["Recall@20"] > best_metrics["Recall@20"]:
                best_epoch = epoch
                best_metrics = metrics
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = (
        config.artifacts_dir / config.dataset / f"{config.checkpoint_name}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "split_strategy": data.split_strategy,
            "dataset": config.dataset,
            "positive_threshold": positive_threshold,
            "model_config": {
                "num_users": data.num_users,
                "num_items": data.num_items,
                "embedding_dim": config.embedding_dim,
                "num_layers": config.num_layers,
            },
            "graph": {
                "users": torch.from_numpy(users),
                "items": torch.from_numpy(items),
            },
            "train_config": json_ready_config(config),
            "provenance": build_provenance(
                data, dataset=config.dataset, positive_threshold=positive_threshold
            ),
            "epoch_losses": losses,
            "best_epoch": best_epoch,
            "best_validation_metrics": best_metrics,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    return model, losses


def parse_args() -> LightGCNTrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--reg-weight", type=float, default=1e-4)
    parser.add_argument("--validation-users", type=int, default=0)
    parser.add_argument("--positive-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--checkpoint-name", default="lightgcn")
    return LightGCNTrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train_lightgcn(parse_args())
