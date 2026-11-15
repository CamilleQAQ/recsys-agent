"""训练神经排序模型：DIN / DIEN / DCNv2 / Ultra-HSTU。

数据来自父数据集 ``processed/ranking`` 的请求级候选（click 为标签），按原始
ID 对齐到当前数据集词表；物品特征与用户活跃度由 ``ranking.features`` 统一构造。
四个模型共享同一份 batch 构造与评估口径（BCE + AUC/LogLoss + request NDCG@5）。
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from recsys.config import DEFAULT_DATASET
from recsys.data.loader import SUPPORTED_DATASETS, prepare_dataset, resolve_positive_threshold
from recsys.data.schema import PreparedData
from recsys.evaluation.metrics import log_loss, roc_auc
from recsys.models.dcnv2 import DCNv2
from recsys.models.dien import DIEN
from recsys.models.din import DIN
from recsys.models.ultra_hstu import UltraHSTU
from recsys.ranking.features import (
    CATEGORICAL_CARDINALITIES,
    FeatureContext,
    NUMERIC_ITEM_COLUMNS,
    NUMERIC_USER_COLUMNS,
)
from recsys.ranking.samples import (
    RankingTimestampError,
    RankingView,
    build_history_source,
    histories_for_requests,
    load_ranking_view,
)
from recsys.training.trainer import resolve_device
from recsys.utils.checkpoint import json_ready_config
from recsys.utils.provenance import build_provenance
from recsys.utils.seed import set_random_seed

MODEL_NAMES = ("din", "dien", "dcnv2", "ultra_hstu")
#: 这些模型读取用户行为序列，训练/评估必须拿到请求时间戳。
HISTORY_MODELS = frozenset({"din", "dien", "ultra_hstu"})


@dataclass(frozen=True)
class NeuralRankerTrainConfig:
    model: str = "din"
    dataset: str = DEFAULT_DATASET
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    parent_ranking: Path | None = None
    epochs: int = 3
    batch_size: int = 2048
    learning_rate: float = 1e-3
    max_history: int = 20
    embedding_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.1
    max_train_rows: int = 20_000
    max_valid_rows: int = 50_000
    positive_threshold: float | None = None
    seed: int = 42
    device: str = "auto"
    max_batches_per_epoch: int | None = None
    checkpoint_name: str = ""


def build_model(name: str, data: PreparedData, config: NeuralRankerTrainConfig):
    common = {
        "cat_cardinalities": CATEGORICAL_CARDINALITIES,
        "numeric_count": len(NUMERIC_ITEM_COLUMNS),
        "user_numeric_count": len(NUMERIC_USER_COLUMNS),
        "dropout": config.dropout,
    }
    if name in {"din", "dien"}:
        model_class = DIN if name == "din" else DIEN
        return model_class(
            num_items=data.num_items,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            **common,
        )
    if name == "dcnv2":
        return DCNv2(cross_layers=2, hidden_dim=config.hidden_dim, **common)
    if name == "ultra_hstu":
        return UltraHSTU(
            num_items=data.num_items,
            max_seq_len=config.max_history,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_blocks=2,
            num_heads=2,
            **common,
        )
    raise ValueError(f"未知排序模型 {name!r}，可选 {MODEL_NAMES}")


def model_config_for(name: str, model) -> dict:
    if name in {"din", "dien"}:
        return {
            "num_items": model.num_items,
            "embedding_dim": model.embedding_dim,
            "hidden_dim": model.output[0].out_features,
            "attention_dim": model.attention[0].out_features,
            "cat_cardinalities": [table.num_embeddings for table in model.cat_embeddings],
            "numeric_count": 5,
            "user_numeric_count": 2,
            "dropout": model.output[2].p,
        }
    if name == "dcnv2":
        return {
            "cat_cardinalities": [table.num_embeddings for table in model.cat_embeddings],
            "numeric_count": 5,
            "user_numeric_count": 2,
            "embedding_dim": model.cat_embeddings[0].embedding_dim,
            "cross_layers": len(model.cross.layers),
            "hidden_dim": model.deep[0].out_features,
            "dropout": model.deep[2].p,
        }
    return {
        "num_items": model.num_items,
        "max_seq_len": model.max_seq_len,
        "embedding_dim": model.encoder.hidden_dim,
        "hidden_dim": model.output[0].out_features,
        "num_blocks": len(model.encoder.blocks),
        "num_heads": 2,
        "dropout": model.output[2].p,
        "cat_cardinalities": [table.num_embeddings for table in model.cat_embeddings],
        "numeric_count": 5,
        "user_numeric_count": 2,
    }


def build_batch(
    users: np.ndarray,
    items: np.ndarray,
    timestamps: np.ndarray | None,
    histories: dict[tuple[int, int], np.ndarray],
    context: FeatureContext,
    max_history: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """同一请求的候选共享一份历史；历史只包含请求时间之前的交互。"""
    batch_size = users.size
    history = np.zeros((batch_size, max_history), dtype=np.int64)
    if timestamps is not None:
        for row, (user, timestamp) in enumerate(zip(users.tolist(), timestamps.tolist())):
            tokens = histories.get((int(user), int(timestamp)))
            if tokens is not None:
                history[row] = tokens
    item_cat, item_numeric = context.item_arrays(items)
    user_numeric = context.user_array(users)
    return {
        "history": torch.from_numpy(history).to(device),
        "candidates": torch.from_numpy(np.asarray(items, dtype=np.int64)).to(device),
        "item_cat": torch.from_numpy(item_cat).to(device),
        "item_numeric": torch.from_numpy(item_numeric).to(device),
        "user_numeric": torch.from_numpy(user_numeric).to(device),
    }


def ndcg_at_k_per_request(
    scores: np.ndarray, labels: np.ndarray, groups: np.ndarray, k: int = 5
) -> float:
    """请求级 NDCG@K：一个请求的候选为一个 group。"""
    starts = np.concatenate(([0], np.cumsum(groups)))
    values: list[float] = []
    for start, end in zip(starts[:-1], starts[1:]):
        group_scores = scores[start:end]
        group_labels = labels[start:end]
        order = np.argsort(-group_scores)[:k]
        gains = group_labels[order]
        dcg = np.sum(gains / np.log2(np.arange(2, gains.size + 2)))
        ideal = np.sort(group_labels)[::-1][:k]
        idcg = np.sum(ideal / np.log2(np.arange(2, ideal.size + 2)))
        values.append(float(dcg / idcg) if idcg > 0 else 0.0)
    return float(np.mean(values)) if values else 0.0


@torch.no_grad()
def evaluate_ranking_view(
    model,
    view: RankingView,
    histories: dict[tuple[int, int], np.ndarray],
    context: FeatureContext,
    *,
    max_history: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    scores: list[np.ndarray] = []
    for start in range(0, len(view), batch_size):
        batch = build_batch(
            view.users[start : start + batch_size],
            view.items[start : start + batch_size],
            view.timestamps[start : start + batch_size] if view.timestamps is not None else None,
            histories,
            context,
            max_history,
            device,
        )
        logits = model(**batch)
        scores.append(torch.sigmoid(logits).cpu().numpy())
    score_array = np.concatenate(scores) if scores else np.zeros(0)
    probabilities = np.clip(score_array, 1e-6, 1 - 1e-6)
    return {
        "AUC": float(roc_auc(score_array.tolist(), view.labels.tolist())),
        "LogLoss": float(log_loss(probabilities.tolist(), view.labels.tolist())),
        "NDCG@5": ndcg_at_k_per_request(score_array, view.labels, view.groups, k=5),
        "rows": int(view.items.size),
        "requests": int(view.groups.size),
    }


def train_neural_ranker(config: NeuralRankerTrainConfig) -> dict:
    if config.model not in MODEL_NAMES:
        raise ValueError(f"未知排序模型 {config.model!r}")
    set_random_seed(config.seed)
    device = resolve_device(config.device)
    data = prepare_dataset(
        dataset=config.dataset,
        data_dir=config.data_dir,
        positive_threshold=resolve_positive_threshold(config.dataset, config.positive_threshold),
    )
    parent_ranking = config.parent_ranking or (
        config.data_dir / "kuasar" / "processed" / "ranking"
    )
    require_history = config.model in HISTORY_MODELS
    try:
        train_view = load_ranking_view(
            data,
            Path(parent_ranking),
            split="train",
            max_rows=config.max_train_rows,
            require_timestamps=require_history,
        )
        valid_view = load_ranking_view(
            data,
            Path(parent_ranking),
            split="validation",
            max_rows=config.max_valid_rows,
            require_both_labels=False,
            require_timestamps=require_history,
        )
    except RankingTimestampError as error:
        raise RankingTimestampError(f"{config.model}: {error}") from error
    context = FeatureContext(data)
    train_histories: dict[tuple[int, int], np.ndarray] = {}
    valid_histories: dict[tuple[int, int], np.ndarray] = {}
    if require_history:
        train_source = build_history_source(data, "train")
        train_histories = histories_for_requests(
            train_source,
            zip(train_view.users.tolist(), train_view.timestamps.tolist()),
            max_history=config.max_history,
        )
        # 验证请求发生在 train 之后；仍显式用 train 交互并按请求时间过滤。
        valid_source = build_history_source(data, "train")
        valid_histories = histories_for_requests(
            valid_source,
            zip(valid_view.users.tolist(), valid_view.timestamps.tolist()),
            max_history=config.max_history,
        )
    model = build_model(config.model, data, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    print(
        f"Device: {device}; model={config.model}; train rows={len(train_view):,} "
        f"requests={train_view.requests:,}; valid rows={len(valid_view):,}"
    )

    history: list[dict] = []
    best_auc = -1.0
    best_state = None
    best_metrics: dict[str, float] = {}
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = np.random.default_rng(config.seed + epoch).permutation(len(train_view))
        total = 0.0
        batches = 0
        for batch_number, start in enumerate(range(0, len(train_view), config.batch_size)):
            if (
                config.max_batches_per_epoch is not None
                and batch_number >= config.max_batches_per_epoch
            ):
                break
            index = permutation[start : start + config.batch_size]
            batch = build_batch(
                train_view.users[index],
                train_view.items[index],
                train_view.timestamps[index] if train_view.timestamps is not None else None,
                train_histories,
                context,
                config.max_history,
                device,
            )
            labels = torch.from_numpy(train_view.labels[index]).float().to(device)
            logits = model(**batch)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item())
            batches += 1
        train_loss = total / max(1, batches)
        metrics = evaluate_ranking_view(
            model,
            valid_view,
            valid_histories,
            context,
            max_history=config.max_history,
            batch_size=config.batch_size,
            device=device,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        print(
            f"epoch={epoch:02d} loss={train_loss:.6f} AUC={metrics['AUC']:.4f} "
            f"LogLoss={metrics['LogLoss']:.4f} NDCG@5={metrics['NDCG@5']:.4f}"
        )
        if metrics["AUC"] > best_auc:
            best_auc = metrics["AUC"]
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    if best_state is None:
        raise RuntimeError("训练没有产生 checkpoint")
    model.load_state_dict(best_state)
    checkpoint_name = config.checkpoint_name or config.model
    checkpoint_path = config.artifacts_dir / config.dataset / f"{checkpoint_name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "split_strategy": data.split_strategy,
            "dataset": config.dataset,
            "positive_threshold": 1.0,
            "model_config": model_config_for(config.model, model),
            "train_config": {
                **json_ready_config(config),
                "model_name": config.model,
                "history_protocol": (
                    "strictly_before_request" if require_history else "not_used"
                ),
            },
            "provenance": build_provenance(
                data, dataset=config.dataset, positive_threshold=1.0
            ),
            "best_epoch": best_epoch,
            "best_validation_metrics": best_metrics,
            "history": history,
            "known_limitation": (
                "历史按请求时间戳严格早于请求构造；同请求候选共享同一份历史。"
                if require_history
                else "DCNv2 不使用序列历史，因此不需要请求时间戳。"
            ),
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    return {"checkpoint": str(checkpoint_path), "metrics": best_metrics, "history": history}


def parse_args() -> NeuralRankerTrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--parent-ranking", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-train-rows", type=int, default=20_000)
    parser.add_argument("--max-valid-rows", type=int, default=50_000)
    parser.add_argument("--positive-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--checkpoint-name", default="")
    return NeuralRankerTrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train_neural_ranker(parse_args())
