"""跨进程文件锁：共享实验 JSONL 追加与 RAG 索引更新共用。

Windows 使用 ``msvcrt.locking``（非阻塞 + 重试），POSIX 使用 ``fcntl.flock``。
只锁一个小字节区间，锁文件与数据文件分开，避免锁住正在读取的 JSONL。
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class LockTimeout(RuntimeError):
    """在超时时间内没有拿到锁。"""


@contextmanager
def file_lock(path: Path, *, timeout_s: float = 30.0) -> Iterator[None]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    locked = False
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    try:
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    locked = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise LockTimeout(f"等待文件锁超时：{path}") from None
                    time.sleep(0.05)
        else:  # pragma: no cover - 本项目主运行环境是 Windows
            import fcntl

            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise LockTimeout(f"等待文件锁超时：{path}") from None
                    time.sleep(0.05)
        yield
    finally:
        if locked:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


__all__ = ["LockTimeout", "file_lock"]
