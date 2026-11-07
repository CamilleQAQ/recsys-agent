"""各命令行入口共用的数据集选择逻辑。

- `kuasar`：主数据线，读 `data/kuasar/processed/` 的 npy 存储（先跑 `kuasar_store`）；
- `kuairec`：流程模板/辅助线，直接解析原始 CSV。

两个数据集的切分协议不同（KuaiSAR 按请求、KuaiRec 按点击），
`split_strategy_for()` 给出「该数据集应该用哪个协议」，checkpoint / 评测都用它校验。
"""

from __future__ import annotations

from pathlib import Path

from recsys.config import KUAIREC_WATCH_RATIO_THRESHOLD, SPLIT_STRATEGY
from recsys.data.kuairec import prepare_kuairec
from recsys.data.kuasar_store import prepare_kuasar
from recsys.data.schema import PreparedData

SUPPORTED_DATASETS = ("kuairec", "kuasar", "kuasar_small")

#: KuaiSAR 的正样本就是 click == 1，没有阈值概念；这里给一个统一接口需要的数值。
KUASAR_POSITIVE_CLICK = 1.0
KUASAR_SPLIT_STRATEGY = "per_user_last_two_requests"

#: 数据名 → `data/` 下的子目录。`kuasar_small` 是从全量切出来的验证子集，协议与全量一致。
_DATASET_DIRS = {
    "kuairec": "kuairec",
    "kuasar": "kuasar",
    "kuasar_small": "kuasar_small",
}


def _is_kuasar_family(dataset: str) -> bool:
    return dataset in {"kuasar", "kuasar_small"}


def resolve_positive_threshold(dataset: str, positive_threshold: float | None) -> float:
    """返回该数据集实际使用的正反馈阈值；默认值写在 ``recsys.config`` 里。"""
    if positive_threshold is not None:
        return float(positive_threshold)
    normalized_name = dataset.lower()
    if normalized_name == "kuairec":
        return float(KUAIREC_WATCH_RATIO_THRESHOLD)
    if _is_kuasar_family(normalized_name):
        return KUASAR_POSITIVE_CLICK
    raise ValueError(
        f"Unsupported dataset {dataset!r}; choose one of {SUPPORTED_DATASETS}"
    )


def split_strategy_for(dataset: str) -> str:
    """该数据集应使用的切分协议（写进 checkpoint，评测时据此校验）。"""
    normalized_name = dataset.lower()
    if normalized_name == "kuairec":
        return SPLIT_STRATEGY
    if _is_kuasar_family(normalized_name):
        return KUASAR_SPLIT_STRATEGY
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
    if normalized_name in _DATASET_DIRS:
        return data_dir / _DATASET_DIRS[normalized_name]
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
    if _is_kuasar_family(normalized_name):
        # 全量与子集共用同一份 processed 布局，只是目录不同
        return prepare_kuasar(dataset_dir)
    return prepare_kuairec(dataset_dir, watch_ratio_threshold=float(threshold))
