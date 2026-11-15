"""广告模型注册表：模型枚举、权重解析与状态标注。

权重分两类，目录严格分离：

* **业务默认权重**：``artifacts/<dataset>/ads/weights/<model>.pt``。
  只有真实训练产出的权重才允许写到这里（本轮没有训练，因此该目录不存在）。
* **接口测试权重**：``artifacts/<dataset>/ads/interface_weights/<model>.pt``。
  固定初始化权重，只用于验证「能加载、能推理、能保存/加载往返」，
  状态必须标注为 ``interface_test``，不能写成已训练或有效果。

解析顺序：业务默认权重 → 接口测试权重 → 不可用（调用方必须报错，不能伪造）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from recsys.ads.models import AD_MODELS, AD_MODEL_SPECS, model_spec
from recsys.config import DEFAULT_ARTIFACTS_DIR

WEIGHT_KIND_DEFAULT = "default"
WEIGHT_KIND_INTERFACE = "interface_test"
STATUS_INTEGRATED = "integrated"
STATUS_INTERFACE_TEST = "interface_test"
STATUS_UNAVAILABLE = "unavailable"

FRAMEWORK_LABELS = {
    "project": "项目内实现（Phase 1 复用）",
    "deepctr_torch": "DeepCTR-Torch（本地源码）",
}


def weights_root(artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR, dataset: str = "kuasar_small") -> Path:
    return Path(artifacts_dir) / dataset / "ads" / "weights"


def interface_weights_root(
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR, dataset: str = "kuasar_small"
) -> Path:
    return Path(artifacts_dir) / dataset / "ads" / "interface_weights"


def resolve_weight(
    name: str,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    dataset: str = "kuasar_small",
) -> dict[str, Any]:
    model_spec(name)  # 未知模型直接报错
    default_path = weights_root(artifacts_dir, dataset) / f"{name}.pt"
    interface_path = interface_weights_root(artifacts_dir, dataset) / f"{name}.pt"
    if default_path.exists():
        return {
            "path": default_path,
            "weight_kind": WEIGHT_KIND_DEFAULT,
            "status": STATUS_INTEGRATED,
            "is_trained": None,  # 是否真实训练过由 checkpoint 内部字段决定
        }
    if interface_path.exists():
        return {
            "path": interface_path,
            "weight_kind": WEIGHT_KIND_INTERFACE,
            "status": STATUS_INTERFACE_TEST,
            "is_trained": False,
        }
    return {
        "path": None,
        "weight_kind": None,
        "status": STATUS_UNAVAILABLE,
        "is_trained": None,
    }


def list_ad_models(
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR, dataset: str = "kuasar_small"
) -> list[dict[str, Any]]:
    """六模型的注册信息 + 权重状态（Agent 的 ``list_ad_models`` 工具数据源）。"""
    models: list[dict[str, Any]] = []
    for name in AD_MODELS:
        spec = AD_MODEL_SPECS[name]
        weight = resolve_weight(name, artifacts_dir, dataset)
        models.append(
            {
                "name": spec.name,
                "display": spec.display,
                "framework": FRAMEWORK_LABELS.get(spec.framework, spec.framework),
                "task": spec.task,
                "needs_history": spec.needs_history,
                "upstream": spec.upstream,
                "note": spec.note,
                "weight_kind": weight["weight_kind"],
                "status": weight["status"],
                "checkpoint": (
                    str(_relative(Path(weight["path"]))) if weight["path"] else None
                ),
                "is_trained": weight["is_trained"],
                "usable": weight["path"] is not None,
            }
        )
    return models


def model_status(
    name: str,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    dataset: str = "kuasar_small",
) -> dict[str, Any]:
    for item in list_ad_models(artifacts_dir, dataset):
        if item["name"] == name:
            return item
    raise KeyError(name)


def _relative(path: Path) -> Path:
    try:
        return path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return path


__all__ = [
    "FRAMEWORK_LABELS",
    "STATUS_INTEGRATED",
    "STATUS_INTERFACE_TEST",
    "STATUS_UNAVAILABLE",
    "WEIGHT_KIND_DEFAULT",
    "WEIGHT_KIND_INTERFACE",
    "interface_weights_root",
    "list_ad_models",
    "model_status",
    "resolve_weight",
    "weights_root",
]
