"""序列模型统一训练入口：BERT4Rec / HSTU（SASRec 保留原入口）。

    python -m recsys.training.train_sequence --model bert4rec --dataset kuasar_small
    python -m recsys.training.train_sequence --model hstu --dataset kuasar_small

两种模型共享同一套数据准备、验证集选轮、checkpoint 格式和全量排序评测：

* BERT4Rec：``Bert4RecTrainDataset`` + 掩码语言建模（sampled softmax）；
* HSTU：``SASRecTrainDataset`` + 下一物品二分类（与 SASRec 同口径）。
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from recsys.config import DEFAULT_DATASET
from recsys.data.dataset import SASRecTrainDataset
from recsys.data.loader import (
    SUPPORTED_DATASETS,
    prepare_dataset,
    resolve_positive_threshold,
)
from recsys.evaluation.sasrec_evaluator import generate_sasrec_recommendations
from recsys.evaluation.evaluator import evaluate_recommendations, get_evaluation_interactions
from recsys.models.bert4rec import BERT4Rec, Bert4RecTrainDataset
from recsys.models.hstu import HSTU
from recsys.training.trainer import resolve_device
from recsys.utils.checkpoint import json_ready_config
from recsys.utils.provenance import build_provenance
from recsys.utils.seed import set_random_seed

MODEL_CLASSES = {"bert4rec": BERT4Rec, "hstu": HSTU}
MODEL_IDS = {"bert4rec": "bert4rec", "hstu": "hstu"}


@dataclass(frozen=True)
class SequenceTrainConfig:
    model: str = "hstu"
    dataset: str = DEFAULT_DATASET
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    max_epochs: int = 20
    patience: int = 4
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    embedding_dim: int = 64
    num_blocks: int = 2
    num_heads: int = 2
    max_seq_len: int = 50
    dropout: float = 0.2
    num_negatives: int = 256
    validation_users: int = 0
    positive_threshold: float | None = None
    seed: int = 42
    device: str = "auto"
    max_batches_per_epoch: int | None = None
    checkpoint_name: str = ""


def evaluate_sequence_model(
    model,
    data,
    split: str,
    *,
    ks: tuple[int, ...],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """复用 SASRec 的全量排序评测实现（只依赖 max_seq_len / score_all_items）。"""
    recommendations = generate_sasrec_recommendations(
        model, data, split, max(ks), batch_size, device
    )
    metrics = evaluate_recommendations(
        recommendations, get_evaluation_interactions(data, split), ks
    )
    from recsys.evaluation.metrics import catalog_coverage

    metrics["Coverage"] = catalog_coverage(recommendations, data.num_items)
    return metrics


def _train_bert4rec_epoch(
    model: BERT4Rec,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_negatives: int,
    generator: torch.Generator,
    max_batches: int | None,
) -> float:
    model.train()
    total = 0.0
    batches = 0
    num_items = model.num_items
    output_weight = model.item_embedding.weight[1 : num_items + 1]
    for batch_number, batch in enumerate(loader):
        if max_batches is not None and batch_number >= max_batches:
            break
        tokens = batch["input_sequence"].to(device)
        labels = batch["labels"].to(device)
        hidden = model(tokens)
        masked = labels.ne(0)
        if not bool(masked.any()):
            continue
        representations = hidden[masked]
        targets = labels[masked]
        negatives = torch.randint(
            1,
            num_items + 1,
            (targets.numel(), num_negatives),
            generator=generator,
            device=device,
        )
        positive_scores = (representations * output_weight[targets - 1]).sum(dim=1)
        negative_embeddings = output_weight[negatives - 1]  # [N, K, D]
        negative_scores = torch.bmm(
            representations.unsqueeze(1), negative_embeddings.transpose(1, 2)
        ).squeeze(1)
        logits = torch.cat([positive_scores.unsqueeze(1), negative_scores], dim=1)
        loss = F.cross_entropy(logits, torch.zeros(logits.shape[0], dtype=torch.long, device=device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.item())
        batches += 1
    if batches == 0:
        raise RuntimeError("训练 DataLoader 没有产生任何 batch")
    return total / batches


def _train_hstu_epoch(
    model: HSTU,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None,
) -> float:
    model.train()
    total = 0.0
    batches = 0
    for batch_number, batch in enumerate(loader):
        if max_batches is not None and batch_number >= max_batches:
            break
        inputs = batch["input_sequence"].to(device)
        positives = batch["positive_sequence"].to(device)
        negatives = batch["negative_sequence"].to(device)
        hidden = model(inputs)
        positive_embeddings = model.item_embedding(positives)
        negative_embeddings = model.item_embedding(negatives)
        positive_logits = (hidden * positive_embeddings).sum(dim=-1)
        negative_logits = (hidden * negative_embeddings).sum(dim=-1)
        valid = positives.ne(0)
        loss = F.binary_cross_entropy_with_logits(
            positive_logits[valid], torch.ones_like(positive_logits[valid])
        ) + F.binary_cross_entropy_with_logits(
            negative_logits[valid], torch.zeros_like(negative_logits[valid])
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.item())
        batches += 1
    if batches == 0:
        raise RuntimeError("训练 DataLoader 没有产生任何 batch")
    return total / batches


def train_sequence(config: SequenceTrainConfig):
    model_name = config.model.lower()
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"未知序列模型 {config.model!r}，可选 {sorted(MODEL_CLASSES)}")
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
    model_class = MODEL_CLASSES[model_name]
    model = model_class(
        num_items=data.num_items,
        max_seq_len=config.max_seq_len,
        hidden_dim=config.embedding_dim,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout=config.dropout,
    ).to(device)
    if model_name == "bert4rec":
        dataset = Bert4RecTrainDataset(
            data.train_interactions,
            num_items=data.num_items,
            max_seq_len=config.max_seq_len,
            seed=config.seed,
        )
        collate = None
    else:
        dataset = SASRecTrainDataset(
            data.train_interactions,
            num_items=data.num_items,
            max_seq_len=config.max_seq_len,
            seed=config.seed,
        )
        collate = None
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    print(f"Device: {device}; model={model_name}; samples={len(dataset):,}")
    history: list[dict] = []
    best_metrics: dict[str, float] = {}
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, config.max_epochs + 1):
        if model_name == "bert4rec":
            loss = _train_bert4rec_epoch(
                model,
                loader,
                optimizer,
                device,
                config.num_negatives,
                generator,
                config.max_batches_per_epoch,
            )
        else:
            loss = _train_hstu_epoch(
                model, loader, optimizer, device, config.max_batches_per_epoch
            )
        metrics = evaluate_sequence_model(
            model,
            data,
            "validation",
            ks=(10, 20),
            batch_size=config.batch_size,
            device=device,
        )
        history.append({"epoch": epoch, "train_loss": loss, **metrics})
        print(
            f"epoch={epoch:02d} loss={loss:.6f} val_R@20={metrics['Recall@20']:.6f}"
        )
        if not best_metrics or metrics["Recall@20"] > best_metrics["Recall@20"]:
            best_epoch = epoch
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                print(f"Early stopping after epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("训练没有产生 checkpoint")
    model.load_state_dict(best_state)
    checkpoint_name = config.checkpoint_name or MODEL_IDS[model_name]
    checkpoint_path = config.artifacts_dir / config.dataset / f"{checkpoint_name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "split_strategy": data.split_strategy,
            "dataset": config.dataset,
            "positive_threshold": positive_threshold,
            "model_config": {
                "num_items": data.num_items,
                "max_seq_len": config.max_seq_len,
                "hidden_dim": config.embedding_dim,
                "num_blocks": config.num_blocks,
                "num_heads": config.num_heads,
                "dropout": config.dropout,
            },
            "train_config": {**json_ready_config(config), "model_name": model_name},
            "provenance": build_provenance(
                data, dataset=config.dataset, positive_threshold=positive_threshold
            ),
            "best_epoch": best_epoch,
            "best_validation_metrics": best_metrics,
            "history": history,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    return model, best_metrics


def parse_args() -> SequenceTrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_CLASSES), default="hstu")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-negatives", type=int, default=256)
    parser.add_argument("--validation-users", type=int, default=0, help="仅冒烟时限制验证用户数，0=全部")
    parser.add_argument("--positive-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--checkpoint-name", default="")
    return SequenceTrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train_sequence(parse_args())
