"""训练 MMRec-VBPR（内容增强的 BPR 协同过滤，本项目最小适配）。"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from recsys.config import DEFAULT_DATASET
from recsys.data.loader import SUPPORTED_DATASETS, prepare_dataset, resolve_positive_threshold
from recsys.data.schema import InteractionDataset
from recsys.evaluation.evaluator import evaluate_embedding_model, group_items_by_user
from recsys.models.mmrec_vbpr import MMRecVBPR, build_content, vbpr_loss
from recsys.training.samplers import sample_negative_items
from recsys.training.trainer import create_dataloader, resolve_device
from recsys.utils.checkpoint import json_ready_config
from recsys.utils.provenance import build_provenance
from recsys.utils.seed import set_random_seed


@dataclass(frozen=True)
class MMRecTrainConfig:
    dataset: str = DEFAULT_DATASET
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    epochs: int = 10
    batch_size: int = 2048
    learning_rate: float = 1e-3
    embedding_dim: int = 64
    content_dim: int = 32
    content_embed_dim: int = 32
    reg_weight: float = 1e-5
    validation_users: int = 0
    positive_threshold: float | None = None
    seed: int = 42
    device: str = "auto"
    max_batches_per_epoch: int | None = None
    checkpoint_name: str = "mmrec_vbpr"


def train_mmrec(config: MMRecTrainConfig) -> tuple[MMRecVBPR, dict]:
    set_random_seed(config.seed)
    device = resolve_device(config.device)
    data = prepare_dataset(
        dataset=config.dataset,
        data_dir=config.data_dir,
        positive_threshold=resolve_positive_threshold(config.dataset, config.positive_threshold),
    )
    from recsys.training.subsets import limit_validation_users

    data = limit_validation_users(data, config.validation_users, config.seed)
    content = build_content(data.item_features, seed=config.seed, dim=config.content_dim)
    model = MMRecVBPR(
        num_users=data.num_users,
        num_items=data.num_items,
        content_dim=config.content_dim,
        embedding_dim=config.embedding_dim,
        content_embed_dim=config.content_embed_dim,
    ).to(device)
    model.set_content(content)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loader = create_dataloader(
        InteractionDataset(data.train_interactions), config.batch_size, config.seed
    )
    train_items_by_user = group_items_by_user(data.train_interactions)
    generator = torch.Generator().manual_seed(config.seed)
    print(
        f"Device: {device}; content_dim={config.content_dim}; "
        f"train={len(data.train_interactions):,}"
    )
    best_metrics: dict[str, float] = {}
    best_state = None
    best_epoch = 0
    losses: list[float] = []
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
            users = batch["user_idx"]
            positives = batch["item_idx"]
            negatives = sample_negative_items(
                users, train_items_by_user, data.num_items, generator
            )
            users, positives, negatives = users.to(device), positives.to(device), negatives.to(device)
            loss = vbpr_loss(
                model.score(users, positives),
                model.score(users, negatives),
                users=users,
                positive_items=positives,
                negative_items=negatives,
                model=model,
                reg_weight=config.reg_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1
        average = total / max(1, batches)
        losses.append(average)
        print(f"Epoch {epoch:02d}/{config.epochs:02d} - loss: {average:.6f}")
        if data.validation_interactions:
            metrics = evaluate_embedding_model(
                model, data, "validation", (10, 20), 512, device
            )
            print(f"  validation Recall@20: {metrics['Recall@20']:.6f}")
            if not best_metrics or metrics["Recall@20"] > best_metrics["Recall@20"]:
                best_epoch = epoch
                best_metrics = metrics
                best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = config.artifacts_dir / config.dataset / f"{config.checkpoint_name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "split_strategy": data.split_strategy,
            "dataset": config.dataset,
            "positive_threshold": resolve_positive_threshold(config.dataset, config.positive_threshold),
            "model_config": {
                "num_users": data.num_users,
                "num_items": data.num_items,
                "content_dim": config.content_dim,
                "embedding_dim": config.embedding_dim,
                "content_embed_dim": config.content_embed_dim,
            },
            "train_config": json_ready_config(config),
            "provenance": build_provenance(
                data,
                dataset=config.dataset,
                positive_threshold=resolve_positive_threshold(config.dataset, config.positive_threshold),
            ),
            "epoch_losses": losses,
            "best_epoch": best_epoch,
            "best_validation_metrics": best_metrics,
            "known_limitation": "内容向量为物品元数据，不是视觉/文本 embedding",
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    return model, {"best_epoch": best_epoch, "metrics": best_metrics, "losses": losses}


def parse_args() -> MMRecTrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--content-dim", type=int, default=32)
    parser.add_argument("--content-embed-dim", type=int, default=32)
    parser.add_argument("--reg-weight", type=float, default=1e-5)
    parser.add_argument("--validation-users", type=int, default=0)
    parser.add_argument("--positive-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--checkpoint-name", default="mmrec_vbpr")
    return MMRecTrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train_mmrec(parse_args())
