"""各命令行入口共用的数据集选择逻辑。

当前只维护 KuaiRec（流程模板/辅助线）；KuaiSAR 的 `prepare_kuasar` 在 Phase 1.3 接入，
届时在 `SUPPORTED_DATASETS` 与 `prepare_dataset` 里登记即可。
"""

from __future__ import annotations

from pathlib import Path

from recsys.config import KUAIREC_WATCH_RATIO_THRESHOLD
from recsys.data.kuairec import prepare_kuairec
from recsys.data.schema import PreparedData

SUPPORTED_DATASETS = ("kuairec",)


def resolve_positive_threshold(dataset: str, positive_threshold: float | None) -> float:
    """返回该数据集实际使用的正反馈阈值；默认值写在 ``recsys.config`` 里。"""
    if positive_threshold is not None:
        return float(positive_threshold)
    normalized_name = dataset.lower()
    if normalized_name == "kuairec":
        return float(KUAIREC_WATCH_RATIO_THRESHOLD)
    raise ValueError(
        f"Unsupported dataset {dataset!r}; choose one of {SUPPORTED_DATASETS}"
    )


def resolve_checkpoint_dataset(checkpoint: dict) -> str:
    """读回 checkpoint 训练时用的数据集名。"""
    dataset = checkpoint.get("dataset")
    if dataset is None:
        dataset = checkpoint.get("train_config", {}).get("dataset")
    if dataset is None:
        raise ValueError(
            "Checkpoint 没有记录 dataset 字段，无法确定该用哪份数据评测；请重新训练。"
        )
    normalized_name = str(dataset).lower()
    if normalized_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Checkpoint names an unsupported dataset: {dataset!r}")
    return normalized_name


def resolve_dataset_data_dir(dataset: str, data_dir: Path) -> Path:
    """返回该数据集自己的目录：每个数据集住进 ``data/<name>/``。"""
    normalized_name = dataset.lower()
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
    return prepare_kuairec(dataset_dir, watch_ratio_threshold=float(threshold))
