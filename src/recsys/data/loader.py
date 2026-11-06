"""各命令行入口共用的数据集选择逻辑。"""

from __future__ import annotations

from pathlib import Path

from recsys.config import (
    DEFAULT_MOVIELENS_POSITIVE_THRESHOLD,
    KUAIREC_WATCH_RATIO_THRESHOLD,
)
from recsys.data.kuairec import prepare_kuairec
from recsys.data.movielens import prepare_movielens
from recsys.data.schema import PreparedData


SUPPORTED_DATASETS = ("kuairec", "movielens")


def resolve_positive_threshold(dataset: str, positive_threshold: float | None) -> float:
    """返回该数据集实际使用的正反馈阈值；默认值写在 ``recsys.config`` 里。"""
    if positive_threshold is not None:
        return float(positive_threshold)
    normalized_name = dataset.lower()
    if normalized_name == "movielens":
        return float(DEFAULT_MOVIELENS_POSITIVE_THRESHOLD)
    if normalized_name == "kuairec":
        return float(KUAIREC_WATCH_RATIO_THRESHOLD)
    raise ValueError(
        f"Unsupported dataset {dataset!r}; choose one of {SUPPORTED_DATASETS}"
    )


def resolve_checkpoint_dataset(checkpoint: dict) -> str:
    """读回 checkpoint 训练时用的数据集。

    在加 ``--dataset`` 之前保存的 checkpoint 里没有这个字段——那时候只有
    MovieLens 一条数据线，所以按 MovieLens 处理。这些旧 checkpoint 仍然要能
    被评出来，否则已经公布的数字就不可复现了。
    """
    dataset = checkpoint.get("dataset")
    if dataset is None:
        dataset = checkpoint.get("train_config", {}).get("dataset")
    if dataset is None:
        return "movielens"
    normalized_name = str(dataset).lower()
    if normalized_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Checkpoint names an unsupported dataset: {dataset!r}")
    return normalized_name


def resolve_dataset_data_dir(dataset: str, data_dir: Path) -> Path:
    """返回该数据集自己的目录。

    MovieLens 是仓库里最早的一条数据线，处理结果一直在 ``data/`` 根下
    （``data/processed/``、``data/raw/``），所以这里保持不动；后来的数据集
    各自住进 ``data/<name>/``。这样 ``prepare_dataset`` 和
    ``prepare_movielens`` 指向同一份文件，不会出现两份互不相干的处理结果。
    """
    normalized_name = dataset.lower()
    if normalized_name == "movielens":
        return data_dir
    if normalized_name == "kuairec":
        return data_dir / "kuairec"
    raise ValueError(
        f"Unsupported dataset {dataset!r}; choose one of {SUPPORTED_DATASETS}"
    )


def prepare_dataset(
    dataset: str,
    data_dir: Path,
    positive_threshold: float | None = None,
) -> PreparedData:
    """Prepare one supported dataset without a factory or registry abstraction."""
    normalized_name = dataset.lower()
    threshold = resolve_positive_threshold(normalized_name, positive_threshold)
    dataset_dir = resolve_dataset_data_dir(normalized_name, data_dir)
    if normalized_name == "movielens":
        return prepare_movielens(dataset_dir, positive_threshold=threshold)
    return prepare_kuairec(dataset_dir, watch_ratio_threshold=float(threshold))
