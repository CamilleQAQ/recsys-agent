"""DSSM 训练入口：复用 Two-Tower 训练器，固定「MLP 塔 + 物品侧特征」配置。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from recsys.config import DEFAULT_DATASET
from recsys.training.trainer import TrainConfig, train


def dssm_config(overrides: dict | None = None) -> TrainConfig:
    """DSSM = 带侧特征 MLP 塔的双塔；其余训练超参与 Two-Tower 一致。"""
    base = TrainConfig(
        tower_type="mlp",
        use_item_features=True,
        checkpoint_name="dssm",
        embedding_dim=64,
        hidden_dim=128,
        output_dim=64,
        temperature=0.1,
        epochs=10,
        # KuaiSAR 词表 2,000 × 454,717 > 2e7，稠密假负例表不可行；
        # 批内假负例概率约 1e-5，关闭后与 Two-Tower 冒烟口径一致。
        mask_false_negatives=False,
    )
    return replace(base, **(overrides or {}))


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--positive-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--validation-users", type=int, default=0)
    return dssm_config(vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
