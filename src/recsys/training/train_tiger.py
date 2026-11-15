"""训练 TIGER 生成式检索器（依赖已训练好的 RQ-VAE Semantic ID）。"""

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
from recsys.data.loader import SUPPORTED_DATASETS, prepare_dataset, resolve_positive_threshold
from recsys.evaluation.tiger_evaluator import evaluate_tiger_model
from recsys.models.rqvae import SemanticIDIndex, load_rqvae
from recsys.models.tiger import DEFAULT_MAX_HISTORY_ITEMS, TigerTrainDataset, TigerTransformer
from recsys.training.trainer import resolve_device
from recsys.utils.checkpoint import json_ready_config
from recsys.utils.provenance import build_provenance
from recsys.utils.seed import set_random_seed


@dataclass(frozen=True)
class TigerTrainConfig:
    dataset: str = DEFAULT_DATASET
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    rqvae_checkpoint: Path | None = None
    epochs: int = 10
    batch_size: int = 512
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    num_blocks: int = 2
    num_heads: int = 2
    dropout: float = 0.1
    max_history_items: int = DEFAULT_MAX_HISTORY_ITEMS
    max_history_tokens: int = 60
    beam_size: int = 8
    val_users: int = 256
    seed: int = 42
    device: str = "auto"
    max_batches_per_epoch: int | None = None
    checkpoint_name: str = "tiger"


def train_tiger(config: TigerTrainConfig) -> dict:
    set_random_seed(config.seed)
    device = resolve_device(config.device)
    data = prepare_dataset(
        dataset=config.dataset,
        data_dir=config.data_dir,
        positive_threshold=resolve_positive_threshold(config.dataset, None),
    )
    rqvae_path = config.rqvae_checkpoint or (
        config.artifacts_dir / config.dataset / "rqvae.pt"
    )
    _, codes, rqvae_checkpoint = load_rqvae(Path(rqvae_path), device)
    if codes.shape[0] != data.num_items:
        raise ValueError(
            f"RQ-VAE 码表 {codes.shape[0]} 与数据物品数 {data.num_items} 不一致"
        )
    num_levels = int(codes.shape[1])
    codebook_size = int(rqvae_checkpoint["model_config"]["codebook_size"])
    sid_index = SemanticIDIndex(codes)

    dataset = TigerTrainDataset(
        data.train_interactions,
        codes,
        num_levels,
        max_history_items=config.max_history_items,
        max_history_tokens=config.max_history_tokens,
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    model = TigerTransformer(
        codebook_size=codebook_size,
        num_levels=num_levels,
        hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout=config.dropout,
        max_history_tokens=config.max_history_tokens,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    print(
        f"Device: {device}; SIDs={sid_index.num_sids:,}; samples={len(dataset):,}; "
        f"levels={num_levels}; codebook={codebook_size}"
    )

    history: list[dict] = []
    best_metrics: dict[str, float] = {}
    best_state = None
    best_epoch = 0
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
            history_tokens = batch["history_tokens"].to(device)
            target_codes = batch["target_codes"].to(device)
            logits = model(history_tokens, target_codes)
            loss = F.cross_entropy(
                logits.reshape(-1, codebook_size), target_codes.reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item())
            batches += 1
        loss = total / max(1, batches)
        metrics = evaluate_tiger_model(
            model,
            data,
            "validation",
            codes,
            sid_index,
            ks=(10, 20),
            beam_size=config.beam_size,
            max_users=config.val_users,
        )
        history.append({"epoch": epoch, "train_loss": loss, **metrics})
        print(
            f"epoch={epoch:02d} loss={loss:.6f} val_R@20={metrics['Recall@20']:.6f} "
            f"(users<={config.val_users})"
        )
        if not best_metrics or metrics["Recall@20"] > best_metrics["Recall@20"]:
            best_epoch = epoch
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("训练没有产生 checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path = config.artifacts_dir / config.dataset / f"{config.checkpoint_name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "split_strategy": data.split_strategy,
            "dataset": config.dataset,
            "positive_threshold": 1.0,
            "model_config": {
                "codebook_size": codebook_size,
                "num_levels": num_levels,
                "hidden_dim": config.hidden_dim,
                "num_blocks": config.num_blocks,
                "num_heads": config.num_heads,
                "dropout": config.dropout,
                "max_history_tokens": config.max_history_tokens,
            },
            "codes": torch.from_numpy(codes.astype(np.int64)),
            "rqvae_checkpoint": str(rqvae_path),
            "train_config": json_ready_config(config),
            "provenance": build_provenance(
                data, dataset=config.dataset, positive_threshold=1.0
            ),
            "best_epoch": best_epoch,
            "best_validation_metrics": best_metrics,
            "history": history,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    return {"checkpoint": str(checkpoint_path), "metrics": best_metrics, "history": history}


def parse_args() -> TigerTrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--rqvae-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-history-items", type=int, default=DEFAULT_MAX_HISTORY_ITEMS)
    parser.add_argument("--max-history-tokens", type=int, default=60)
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--val-users", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--checkpoint-name", default="tiger")
    return TigerTrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train_tiger(parse_args())
