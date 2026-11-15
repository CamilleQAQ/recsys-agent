"""工件守卫：缺失/不匹配时明确失败，绝不静默训练、换模型或重建索引。

Phase 5 的演示与恢复路径只允许加载已有工件。推荐链路的 LightGBM 排序权重历史上
在 ``run_pipeline`` 找不到时会自动训练；本模块提供统一的「只加载」检查，让调用方
在训练之前就把具体文件名报出来。

只做三件事：

* ``require_file``：文件不存在时抛 :class:`ArtifactError`，错误信息包含绝对路径；
* ``file_identity`` / ``sha256_file``：工件身份（路径、大小、sha256）；
* ``snapshot_paths`` / ``diff_snapshots``：在只读检查前后对比文件系统，证明没有
  新增、覆盖或重建。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_CHUNK = 1 << 20


class ArtifactError(RuntimeError):
    """必需的已有工件缺失或校验不匹配。"""

    def __init__(self, message: str, *, path: Path | str | None = None, role: str = "artifact") -> None:
        super().__init__(message)
        self.path = str(path) if path is not None else None
        self.role = role

    def as_dict(self) -> dict[str, Any]:
        return {"error": str(self), "path": self.path, "role": self.role}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(DEFAULT_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path | str, *, with_sha256: bool = True) -> dict[str, Any]:
    target = Path(path)
    payload: dict[str, Any] = {
        "path": str(target),
        "exists": target.exists(),
        "is_file": target.is_file(),
    }
    if target.is_file():
        stat = target.stat()
        payload["size_bytes"] = int(stat.st_size)
        payload["mtime_ns"] = int(stat.st_mtime_ns)
        if with_sha256:
            payload["sha256"] = sha256_file(target)
    return payload


def require_file(
    path: Path | str | None,
    *,
    role: str,
    hint: str = "",
) -> Path:
    """确认工件存在；缺失时报出具体文件，而不是回退到训练/重建。"""

    if path is None or str(path).strip() == "":
        raise ArtifactError(f"{role} 路径为空；只允许加载已有工件", path=path, role=role)
    target = Path(path)
    if not target.exists():
        suffix = f"；{hint}" if hint else ""
        raise ArtifactError(
            f"{role} 缺失：{target}（不存在）。只允许加载已有工件，"
            f"不会自动训练、换模型或重建索引{suffix}",
            path=target,
            role=role,
        )
    if not target.is_file():
        raise ArtifactError(
            f"{role} 不是文件：{target}。只允许加载已有工件",
            path=target,
            role=role,
        )
    return target


def require_sha256(path: Path | str, expected: str | None, *, role: str) -> str:
    target = require_file(path, role=role)
    actual = sha256_file(target)
    if expected and actual.lower() != str(expected).lower():
        raise ArtifactError(
            f"{role} 校验不匹配：{target} 的实际 sha256={actual}，"
            f"清单记录 {expected}。不会自动训练或重建",
            path=target,
            role=role,
        )
    return actual


def snapshot_paths(paths: Iterable[Path | str]) -> dict[str, dict[str, Any]]:
    """记录一组文件/目录的存在性、大小与 sha256（只读，不做任何写操作）。"""

    snapshot: dict[str, dict[str, Any]] = {}
    for value in paths:
        path = Path(value)
        key = str(path)
        item: dict[str, Any] = {"exists": path.exists()}
        if path.is_file():
            stat = path.stat()
            item.update(
                {
                    "is_file": True,
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "sha256": sha256_file(path),
                }
            )
        elif path.is_dir():
            item["is_file"] = False
            item["entries"] = sorted(entry.name for entry in path.iterdir())
        snapshot[key] = item
    return snapshot


def diff_snapshots(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """返回前后差异；空列表即「没有被顺带创建或改写」。"""

    changes: list[dict[str, Any]] = []
    for key, previous in before.items():
        current = after.get(key) or {}
        if previous.get("exists") != current.get("exists"):
            changes.append(
                {
                    "path": key,
                    "change": "created" if current.get("exists") else "removed",
                    "before": dict(previous),
                    "after": dict(current),
                }
            )
            continue
        if not previous.get("exists"):
            continue
        for field in ("size_bytes", "mtime_ns", "sha256", "entries"):
            if previous.get(field) != current.get(field):
                changes.append(
                    {
                        "path": key,
                        "change": f"{field}_changed",
                        "before": previous.get(field),
                        "after": current.get(field),
                    }
                )
                break
    return changes


def find_new_files(root: Path | str, *, since_ns: int, suffixes: tuple[str, ...] = ()) -> list[str]:
    """列出 ``since_ns`` 之后新出现的文件（用于证明没有新训练产物落盘）。"""

    base = Path(root)
    if not base.exists():
        return []
    fresh: list[str] = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if suffixes and path.suffix not in suffixes:
            continue
        try:
            if int(path.stat().st_mtime_ns) >= int(since_ns):
                fresh.append(str(path))
        except OSError:  # pragma: no cover - 并发删除
            continue
    return sorted(fresh)


__all__ = [
    "ArtifactError",
    "diff_snapshots",
    "file_identity",
    "find_new_files",
    "require_file",
    "require_sha256",
    "sha256_file",
    "snapshot_paths",
]
