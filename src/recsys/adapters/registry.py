"""模型 id → 适配器 的注册表。

只需要适配器类；checkpoint 路径、设备等由调用方决定，因此这里不持有任何状态。
新增模型一律经本表进入推荐链路和 Agent 工具，避免两条注册路径漂移。
"""

from __future__ import annotations

from pathlib import Path

from recsys.adapters.base import AdapterError, RecommenderAdapter
from recsys.adapters.builtin import (
    BUILTIN_ADAPTERS,
    DSSMAdapter,
    MatrixFactorizationAdapter,
    PopularityAdapter,
    SASRecAdapter,
    TwoTowerAdapter,
)
from recsys.adapters.graph import LightGCNAdapter
from recsys.adapters.mmrec import MMRecAdapter
from recsys.adapters.rechub_mind import MindAdapter
from recsys.adapters.sequence import Bert4RecAdapter, HSTUAdapter
from recsys.adapters.tiger import TigerAdapter
from recsys.training.trainer import resolve_device

#: 统一接口里可用的模型（顺序与 Model Zoo 推荐顺序一致）。
ADAPTERS: dict[str, type[RecommenderAdapter]] = {
    adapter.model_id: adapter
    for adapter in (
        *BUILTIN_ADAPTERS,
        MindAdapter,
        LightGCNAdapter,
        Bert4RecAdapter,
        HSTUAdapter,
        TigerAdapter,
        MMRecAdapter,
    )
}

#: 需要 checkpoint 才能加载的模型。
CHECKPOINT_REQUIRED = frozenset(
    model_id for model_id in ADAPTERS if model_id != "popularity"
)


def adapter_ids() -> list[str]:
    """返回已接入统一接口的模型 id。"""
    return list(ADAPTERS)


def describe() -> list[dict[str, str]]:
    """给 CLI / 文档用的列表：模型 id、实现类、是否需要 checkpoint。"""
    return [
        {
            "model_id": model_id,
            "adapter": adapter.__name__,
            "model_type": adapter.model_type,
            "needs_checkpoint": "yes" if model_id in CHECKPOINT_REQUIRED else "no",
        }
        for model_id, adapter in ADAPTERS.items()
    ]


def load_adapter(
    model_id: str,
    checkpoint_path: Path | None = None,
    device_name: str = "auto",
) -> RecommenderAdapter:
    """按模型 id 构造适配器；需要 checkpoint 的模型必须给路径。"""
    if model_id not in ADAPTERS:
        known = ", ".join(ADAPTERS)
        raise AdapterError(f"未知模型 {model_id!r}；可选：{known}")
    adapter_class = ADAPTERS[model_id]
    device = resolve_device(device_name)

    if model_id in CHECKPOINT_REQUIRED:
        if checkpoint_path is None:
            raise AdapterError(f"{model_id} 需要 --checkpoint 才能加载")
        return adapter_class.load(Path(checkpoint_path), device)

    if checkpoint_path is not None:
        raise AdapterError(f"{model_id} 没有 checkpoint，请不要传 --checkpoint")
    return adapter_class()


__all__ = [
    "ADAPTERS",
    "CHECKPOINT_REQUIRED",
    "DSSMAdapter",
    "LightGCNAdapter",
    "MatrixFactorizationAdapter",
    "MindAdapter",
    "PopularityAdapter",
    "SASRecAdapter",
    "TwoTowerAdapter",
    "adapter_ids",
    "describe",
    "load_adapter",
]
