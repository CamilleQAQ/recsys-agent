"""训练 MIND 召回模型（torch-rechub 实现 + 本项目数据/评测契约）。

    python -m recsys.training.train_mind --dataset kuasar_small --epochs 10

与 Two-Tower 训练的区别只在「用户侧输入」：MIND 读的是一段历史序列，
用户塔用胶囊网络 + 动态路由输出 K 个兴趣向量；损失仍是 in-batch softmax。
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from recsys.config import DEFAULT_DATASET
from recsys.data.dataset import MindTrainDataset, build_observed_histories
from recsys.data.loader import prepare_dataset, resolve_positive_threshold, SUPPORTED_DATASETS
from recsys.evaluation.evaluator import evaluate_recommendations, group_items_by_user
from recsys.data.schema import seen_items_for_split
from recsys.models.mind import (
    DEFAULT_EMBED_DIM,
    DEFAULT_INTEREST_NUM,
    DEFAULT_MAX_LENGTH,
    DEFAULT_TEMPERATURE,
    build_batch,
    build_mind_model,
    default_model_config,
    score_users,
)
from recsys.training.trainer import resolve_device
from recsys.utils.checkpoint import json_ready_config
from recsys.utils.provenance import build_provenance
from recsys.utils.seed import set_random_seed


@dataclass(frozen=True)
class MindTrainConfig:
    dataset: str = DEFAULT_DATASET
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    epochs: int = 10
    batch_size: int = 1024
    learning_rate: float = 1e-2
    embed_dim: int = DEFAULT_EMBED_DIM
    max_length: int = DEFAULT_MAX_LENGTH
    interest_num: int = DEFAULT_INTEREST_NUM
    temperature: float = DEFAULT_TEMPERATURE
    checkpoint_name: str = "mind"
    positive_threshold: float | None = None
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "auto"
    max_batches_per_epoch: int | None = None


def _validation_recommendations(
    model: nn.Module,
    prepared_data,
    max_length: int,
    device: torch.device,
    k: int,
) -> dict[int, list[int]]:
    """验证集 Top-K：历史取训练集，分数取 K 个兴趣上的最大值。"""
    targets = prepared_data.validation_interactions
    users = sorted(group_items_by_user(targets))
    histories = build_observed_histories(prepared_data, "validation")
    scores, _ = score_users(
        model, users, histories, prepared_data.num_items, max_length=max_length, device=device
    )
    blocked = seen_items_for_split(prepared_data, "validation")
    recommendations: dict[int, list[int]] = {}
    for row, user in enumerate(users):
        user_scores = scores[row]
        seen = blocked.get(user, set())
        if seen:
            user_scores[list(seen)] = -np.inf
        top = np.argpartition(-user_scores, k)[:k]
        top = top[np.argsort(-user_scores[top])]
        recommendations[user] = [int(item) for item in top]
    return recommendations


def train(config: MindTrainConfig) -> tuple[nn.Module, list[float]]:
    set_random_seed(config.seed)
    device = resolve_device(config.device)
    threshold = resolve_positive_threshold(config.dataset, config.positive_threshold)
    prepared = prepare_dataset(config.dataset, config.data_dir, positive_threshold=threshold)

    dataset = MindTrainDataset(
        prepared.train_interactions,
        num_items=prepared.num_items,
        max_seq_len=config.max_length,
        seed=config.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    print(f"Dataset: {config.dataset} | users={prepared.num_users:,} items={prepared.num_items:,}")
    print(f"MIND samples/epoch: {len(dataset):,} | max_length={config.max_length} "
          f"interest_num={config.interest_num} embed_dim={config.embed_dim}")

    model = build_mind_model(
        prepared.num_users,
        prepared.num_items,
        embed_dim=config.embed_dim,
        max_length=config.max_length,
        interest_num=config.interest_num,
        temperature=config.temperature,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    epoch_losses: list[float] = []
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    best_state = None

    for epoch in range(config.epochs):
        model.train()
        model.mode = "in_batch"
        total_loss = 0.0
        batches = 0
        for batch_index, batch in enumerate(loader):
            if config.max_batches_per_epoch is not None and batch_index >= config.max_batches_per_epoch:
                break
            inputs = build_batch(
                batch["user_idx"].numpy(),
                batch["history"].numpy(),
                batch["item_token"].numpy(),
                device,
            )
            best_interest, positive = model(inputs)
            logits = (best_interest.squeeze(1) @ positive.squeeze(1).T) / config.temperature
            labels = torch.arange(logits.shape[0], device=device)
            loss = nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        average_loss = total_loss / max(batches, 1)
        epoch_losses.append(average_loss)
        print(f"Epoch {epoch + 1:02d}/{config.epochs:02d} - loss: {average_loss:.6f}")

        if prepared.validation_interactions:
            recommendations = _validation_recommendations(
                model, prepared, config.max_length, device, k=20
            )
            metrics = evaluate_recommendations(
                recommendations, prepared.validation_interactions, ks=(10, 20)
            )
            print(f"  validation Recall@20: {metrics['Recall@20']:.6f}")
            if not best_metrics or metrics["Recall@20"] > best_metrics.get("Recall@20", -1.0):
                best_epoch = epoch + 1
                best_metrics = metrics
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = config.artifacts_dir / config.dataset / f"{config.checkpoint_name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": default_model_config(
                prepared.num_users,
                prepared.num_items,
                embed_dim=config.embed_dim,
                max_length=config.max_length,
                interest_num=config.interest_num,
                temperature=config.temperature,
            ),
            "split_strategy": prepared.split_strategy,
            "dataset": config.dataset,
            "positive_threshold": threshold,
            "train_config": json_ready_config(config),
            "provenance": build_provenance(
                prepared, dataset=config.dataset, positive_threshold=threshold
            ),
            "epoch_losses": epoch_losses,
            "best_epoch": best_epoch,
            "best_validation_metrics": best_metrics,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    return model, epoch_losses


def parse_args(argv: list[str] | None = None) -> MindTrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--interest-num", type=int, default=DEFAULT_INTEREST_NUM)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--checkpoint-name", default="mind")
    parser.add_argument("--positive-threshold", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    args = parser.parse_args(argv)
    return MindTrainConfig(**vars(args))


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
