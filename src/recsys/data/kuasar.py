"""KuaiSAR 2.0 数据获取与结构勘查（Phase 1.1）。

这个模块**只负责把数据安全地取到本地**：下载 → md5 校验 → 解压 → 勘查目录结构。
它不解析成 `PreparedData`，也不定义正负样本——那是 Phase 1.2/1.3 的事，
在没核对真实字段之前写解析逻辑只会返工。

用法：

    python -m recsys.data.kuasar                # 完整版 v2（845 MB）
    python -m recsys.data.kuasar --small        # KuaiSAR-small（488 MB）
    python -m recsys.data.kuasar --survey-only  # 只勘查已解压目录，不联网

产物：

    data/kuasar/raw/KuaiSAR_v2.zip       原始压缩包（保留，便于复验 md5）
    data/kuasar/raw/<解压目录>/...
    artifacts/kuasar_archive_survey.json 目录结构、文件大小、CSV 表头

许可：CC BY-NC-SA 4.0（非商业 + 相同方式共享）。数据不入库，不随代码分发。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from recsys.config import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_DIR

KUASAR_ARCHIVE_URL = "https://zenodo.org/records/8181109/files/KuaiSAR_v2.zip"
KUASAR_ARCHIVE_MD5 = "ed630f47fb013c5b31bb020c483ef262"
KUASAR_SMALL_ARCHIVE_URL = "https://zenodo.org/records/8181109/files/KuaiSAR.zip"
KUASAR_SMALL_ARCHIVE_MD5 = "daea8cbf605db6bd5841740f0e4a12d9"
KUASAR_LICENSE = "CC BY-NC-SA 4.0"
KUASAR_DATASET_DIRNAME = "kuasar"

#: 下载时带的 UA：Zenodo 对无 UA 的脚本请求会拒绝或限速。
USER_AGENT = "recsys-agent/0.1 (research; local use)"

_HEADER_MAX_CHARS = 400
_CHUNK_SIZE = 4 * 1024 * 1024
#: 每下载多少字节打印一次进度（约 64 MB）。
_PROGRESS_STEP = 64 * 1024 * 1024


class KuaiSARError(RuntimeError):
    """下载、校验或解压失败时抛出，附带可操作的提示。"""


def dataset_dir(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """`data/kuasar`；与 KuaiRec 的 `data/kuairec` 保持同构。"""
    return data_dir / KUASAR_DATASET_DIRNAME


def raw_dir(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    return dataset_dir(data_dir) / "raw"


def archive_spec(small: bool) -> tuple[str, str, str]:
    """返回 (url, md5, 文件名)。"""
    if small:
        return KUASAR_SMALL_ARCHIVE_URL, KUASAR_SMALL_ARCHIVE_MD5, "KuaiSAR.zip"
    return KUASAR_ARCHIVE_URL, KUASAR_ARCHIVE_MD5, "KuaiSAR_v2.zip"


def md5_of(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - 校验上游公布的 md5，不是安全用途
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, expected_md5: str) -> None:
    if not path.is_file():
        raise KuaiSARError(f"压缩包不存在：{path}")
    actual = md5_of(path)
    if actual != expected_md5:
        raise KuaiSARError(
            f"md5 不匹配：{path}\n  实际 {actual}\n  期望 {expected_md5}\n"
            "  处理：删除该文件后重跑；半截压缩包会解析出静默错误的数据。"
        )


def download_archive(
    data_dir: Path = DEFAULT_DATA_DIR, *, small: bool = False, force: bool = False
) -> Path:
    """下载压缩包；已存在且 md5 匹配时直接复用。"""
    url, expected_md5, filename = archive_spec(small)
    target_dir = raw_dir(data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename

    if target.exists() and not force:
        print(f"[skip] 已存在 {target}，校验 md5 …")
        verify_archive(target, expected_md5)
        print("[ok] md5 匹配，复用本地文件")
        return target

    temporary = target.with_suffix(target.suffix + ".part")
    print(f"[download] {url}")
    print(f"[download] -> {target}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - 固定 https 地址
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        next_mark = _PROGRESS_STEP
        with temporary.open("wb") as file:
            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                file.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_mark:
                    if total:
                        print(
                            f"  {downloaded / 1048576:7.1f} MB / {total / 1048576:.1f} MB "
                            f"({downloaded / total * 100:5.1f}%)"
                        )
                    else:
                        print(f"  {downloaded / 1048576:7.1f} MB")
                    next_mark += _PROGRESS_STEP
    temporary.replace(target)
    print(f"[download] 完成 {downloaded / 1048576:.1f} MB，校验 md5 …")
    verify_archive(target, expected_md5)
    print("[ok] md5 匹配")
    return target


def extract_archive(archive: Path, destination: Path | None = None) -> Path:
    """安全解压（拒绝越出目标目录的路径），返回解压根目录。"""
    destination = destination or raw_dir()
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if member.filename.startswith("__MACOSX/"):
                continue
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise KuaiSARError(f"压缩包内路径越界：{member.filename}")
        print(f"[extract] {archive.name} -> {destination}")
        bundle.extractall(destination)
    return destination


def _first_line(path: Path, limit: int = _HEADER_MAX_CHARS) -> str:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
            return file.readline().rstrip("\r\n")[:limit]
    except OSError as error:  # pragma: no cover - 文件被占用等
        return f"<读取失败: {error}>"


def survey(root: Path) -> dict[str, Any]:
    """目录结构 + 文件大小 + 文本类文件的表头，供 Phase 1.2 核对字段。"""
    if not root.exists():
        raise KuaiSARError(f"解压目录不存在：{root}")
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        total_bytes += size
        entry: dict[str, Any] = {
            "path": path.relative_to(root).as_posix(),
            "size_mb": round(size / 1048576, 3),
        }
        if path.suffix.lower() in {".csv", ".txt", ".tsv"}:
            entry["header"] = _first_line(path)
        files.append(entry)
    return {
        "root": root.as_posix(),
        "license": KUASAR_LICENSE,
        "file_count": len(files),
        "total_mb": round(total_bytes / 1048576, 1),
        "files": files,
    }


def format_survey(payload: dict[str, Any]) -> str:
    lines = [
        f"root      : {payload['root']}",
        f"license   : {payload['license']}",
        f"files     : {payload['file_count']} 个，共 {payload['total_mb']} MB",
        "",
        f"{'size(MB)':>9}  path",
    ]
    for entry in payload["files"]:
        lines.append(f"{entry['size_mb']:9.2f}  {entry['path']}")
    headers = [(entry["path"], entry["header"]) for entry in payload["files"] if "header" in entry]
    if headers:
        lines += ["", "文本文件表头（用于 Phase 1.2 字段核对）："]
        for path, header in headers:
            lines.append(f"  {path}\n    {header}")
    return "\n".join(lines)


def write_survey(payload: dict[str, Any], output: Path = DEFAULT_ARTIFACTS_DIR / "kuasar_archive_survey.json") -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--small", action="store_true", help="下载 KuaiSAR-small（488 MB）")
    parser.add_argument("--survey-only", action="store_true", help="只勘查已解压目录，不下载")
    parser.add_argument("--force", action="store_true", help="忽略本地压缩包，强制重新下载")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target_dir = raw_dir(args.data_dir)

    if not args.survey_only:
        archive = download_archive(args.data_dir, small=args.small, force=args.force)
        extract_archive(archive, target_dir)
    elif not target_dir.exists():
        print(f"[error] 解压目录不存在：{target_dir}", file=sys.stderr)
        return 2

    payload = survey(target_dir)
    print()
    print(format_survey(payload))
    output = write_survey(payload)
    print(f"\n[ok] 勘查结果写入 {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
