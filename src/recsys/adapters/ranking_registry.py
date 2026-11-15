"""排序模型注册表：LightGBM LambdaRank + DIN/DIEN/DCNv2/Ultra-HSTU。"""

from __future__ import annotations

from pathlib import Path

from recsys.adapters.base import AdapterError, RankingAdapter
from recsys.adapters.neural_ranking import (
    DCNv2Adapter,
    DIENAdapter,
    DINAdapter,
    UltraHSTUAdapter,
)
from recsys.adapters.ranking import LightGBMRankerAdapter

#: model_id → 适配器类。顺序即推荐链路默认尝试顺序。
RANKERS: dict[str, type[RankingAdapter]] = {
    "lightgbm_ranker": LightGBMRankerAdapter,
    "din": DINAdapter,
    "dien": DIENAdapter,
    "dcnv2": DCNv2Adapter,
    "ultra_hstu": UltraHSTUAdapter,
}

#: 需要 checkpoint 的排序模型（LightGBM 也走文件，但加载器签名不同）。
CHECKPOINT_RANKERS = frozenset(RANKERS)


def ranker_ids() -> list[str]:
    return list(RANKERS)


def describe_rankers() -> list[dict[str, str]]:
    return [
        {
            "model_id": model_id,
            "adapter": adapter.__name__,
            "model_type": getattr(adapter, "model_type", "gbdt"),
            "needs_checkpoint": "yes",
        }
        for model_id, adapter in RANKERS.items()
    ]


def load_ranker(
    model_id: str,
    checkpoint_path: Path | None,
    device_name: str = "auto",
) -> RankingAdapter:
    if model_id not in RANKERS:
        raise AdapterError(
            f"未知排序模型 {model_id!r}；可选：{', '.join(RANKERS)}"
        )
    if checkpoint_path is None:
        raise AdapterError(f"{model_id} 需要 checkpoint 才能加载")
    path = Path(checkpoint_path)
    if model_id == "lightgbm_ranker":
        return LightGBMRankerAdapter.load(path)
    from recsys.training.trainer import resolve_device

    return RANKERS[model_id].load(path, resolve_device(device_name))


__all__ = [
    "CHECKPOINT_RANKERS",
    "RANKERS",
    "describe_rankers",
    "load_ranker",
    "ranker_ids",
]
