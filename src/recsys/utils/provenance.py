"""模型工件的溯源信息（PROJECT_MASTER §5.2）。

训练产物不能只有权重：评审一次实验时，必须能回答「这份权重是在哪个数据版本、
哪份 ID 映射、哪份代码和哪套依赖上训练出来的」。本模块把这些信息打包成一个
可直接 ``torch.save`` 的普通字典，挂在 checkpoint 的 ``provenance`` 字段下。

设计取舍：

* **不把完整 ID 映射塞进 checkpoint**（KuaiSAR 量级会到几十 MB）。存映射的 sha256 +
  规模 + 少量样例，完整映射仍在 ``data/<dataset>/processed/id_mappings.json``；
* **代码指纹**用 ``src/recsys/**/*.py`` 的内容哈希，没有 git 也能发现代码漂移；
* 所有值都是 str/int/list/dict/None，保证 ``torch.load(weights_only=True)`` 能读。
"""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from recsys.config import SPLIT_STRATEGY
from recsys.data.schema import Interaction, PreparedData

SRC_ROOT = Path(__file__).resolve().parents[1]  # src/recsys
PACKAGE_NAME = "recsys-agent"
PROJECT_VERSION_FALLBACK = "0.1.0-dev"

#: 记录版本的关键依赖；没装就跳过，不报错。
TRACKED_PACKAGES = ("torch", "numpy", "faiss-cpu", "lightgbm", "tensorflow")


def _package_version() -> str:
    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return PROJECT_VERSION_FALLBACK


def _hash_interactions(split: str, rows: list[Interaction]) -> str:
    digest = hashlib.sha256()
    digest.update(f"#{split}:{len(rows)}\n".encode())
    for row in sorted(rows, key=lambda item: (item.user_idx, item.timestamp, item.item_idx)):
        # repr 保证 float 往返稳定；同一份数据在任何机器上都得到同一个哈希。
        digest.update(f"{row.user_idx},{row.item_idx},{row.timestamp!r}\n".encode())
    return digest.hexdigest()


def data_fingerprint(data: PreparedData) -> dict[str, Any]:
    """三个切分的行数与内容哈希 + 曝光 mask 的规模/哈希。"""
    splits = {
        "train": data.train_interactions,
        "validation": data.validation_interactions,
        "test": data.test_interactions,
    }
    masks: dict[str, dict[str, Any] | None] = {}
    for name, mask in (
        ("validation", data.validation_seen_mask),
        ("test", data.test_seen_mask),
    ):
        if mask is None:
            masks[name] = None
        else:
            array = np.ascontiguousarray(mask)
            masks[name] = {
                "shape": [int(size) for size in array.shape],
                "nnz": int(array.sum()),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
    return {
        "num_users": data.num_users,
        "num_items": data.num_items,
        "counts": {name: len(rows) for name, rows in splits.items()},
        "split_sha256": {name: _hash_interactions(name, rows) for name, rows in splits.items()},
        "seen_masks": masks,
    }


def id_mapping_summary(data: PreparedData, sample_size: int = 5) -> dict[str, Any]:
    """映射的可核对摘要；完整映射请引用 processed/id_mappings.json。"""
    payload = json.dumps(
        {
            "user": data.mappings.index_to_user,
            "item": data.mappings.index_to_item,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "num_users": data.num_users,
        "num_items": data.num_items,
        "sample": {
            "user_index_to_original": data.mappings.index_to_user[:sample_size],
            "item_index_to_original": data.mappings.index_to_item[:sample_size],
        },
    }


def code_fingerprint(root: Path = SRC_ROOT) -> dict[str, Any]:
    """对 src/recsys 下全部 .py 的内容做哈希；文件顺序固定，结果可复现。"""
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*.py"))
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "sha256": digest.hexdigest(),
        "files": len(files),
        "package_version": _package_version(),
    }


def environment_info() -> dict[str, str]:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in TRACKED_PACKAGES:
        try:
            info[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return info


def build_provenance(
    data: PreparedData,
    *,
    dataset: str,
    positive_threshold: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """组装挂到 checkpoint['provenance'] 的字典。"""
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset,
        "positive_threshold": positive_threshold,
        "split_strategy": SPLIT_STRATEGY,
        "data": data_fingerprint(data),
        "id_mapping": id_mapping_summary(data),
        "code": code_fingerprint(),
        "environment": environment_info(),
        "extra": extra or {},
    }


def describe_provenance(provenance: dict[str, Any]) -> str:
    """一行摘要，评测脚本/CLI 打印用。"""
    data = provenance.get("data", {})
    code = provenance.get("code", {})
    return (
        f"data={data.get('split_sha256', {}).get('test', '?')[:12]} "
        f"code={str(code.get('sha256', '?'))[:12]} "
        f"users={data.get('num_users', '?')} items={data.get('num_items', '?')} "
        f"created_at={provenance.get('created_at', '?')}"
    )
