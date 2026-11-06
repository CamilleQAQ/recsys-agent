"""项目级一致性校验：配置 ↔ 代码常量、文档链接、生成物是否过期、残留缓存。

    python scripts/check_project.py

把它当作「提交前的最小检查」：任何一条失败都说明文档或配置已经和代码对不上了。
不依赖真实数据集，也不需要安装除 torch/numpy 之外的包。
"""

from __future__ import annotations

import re
import sys
import tomllib
from dataclasses import fields
from pathlib import Path
from urllib.parse import unquote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
SKIP_LINK_PREFIXES = ("http://", "https://", "mailto:", "#", "tel:")
SKIP_LINK_CHARS = set("<>{}$")

#: 这些 Markdown 里的相对链接需要校验（third_party 下上游 README 的链接指向已删数据/外部资源，不校验）。
EXTRA_CHECKED_MARKDOWN = (
    "third_party/README.md",
    "third_party/LICENSES.md",
    "third_party/BERT4Rec/data/README.md",
    "third_party/SASRec.pytorch/python/data/README.md",
)


def load_toml(path: Path) -> dict:
    with path.open("rb") as file:
        return tomllib.load(file)


def discover_toml() -> list[Path]:
    paths = [PROJECT_ROOT / "pyproject.toml"]
    for directory in ("configs", "models"):
        paths.extend(sorted((PROJECT_ROOT / directory).rglob("*.toml")))
    return [path for path in paths if path.is_file()]


def check_toml_parsing() -> list[str]:
    errors = []
    for path in discover_toml():
        try:
            load_toml(path)
        except tomllib.TOMLDecodeError as error:
            errors.append(f"{path.relative_to(PROJECT_ROOT)}: TOML 解析失败（{error}）")
    return errors


def check_dataset_configs() -> list[str]:
    """数据集配置里的阈值/切分/去重必须与代码常量一致。"""
    from recsys.config import (
        DEFAULT_MOVIELENS_POSITIVE_THRESHOLD,
        KUAIREC_DEDUPLICATE_USER_ITEM_PAIRS,
        KUAIREC_WATCH_RATIO_THRESHOLD,
        SPLIT_STRATEGY,
    )
    from recsys.data.kuairec import KUAIREC_ARCHIVE_MD5, KUAIREC_ARCHIVE_URL
    from recsys.data.kuasar import (
        KUASAR_ARCHIVE_MD5,
        KUASAR_ARCHIVE_URL,
        KUASAR_SMALL_ARCHIVE_MD5,
        KUASAR_SMALL_ARCHIVE_URL,
    )

    errors: list[str] = []
    kuairec = load_toml(PROJECT_ROOT / "configs" / "datasets" / "kuairec.toml")
    movielens = load_toml(PROJECT_ROOT / "configs" / "datasets" / "movielens.toml")
    kuasar = load_toml(PROJECT_ROOT / "configs" / "datasets" / "kuasar.toml")

    pairs = [
        ("kuairec 正反馈阈值", kuairec["labels"]["positive_threshold"], KUAIREC_WATCH_RATIO_THRESHOLD),
        ("kuairec 去重开关", kuairec["labels"]["deduplicate_user_item_pairs"], KUAIREC_DEDUPLICATE_USER_ITEM_PAIRS),
        ("kuairec 下载地址", kuairec["dataset"]["source_url"], KUAIREC_ARCHIVE_URL),
        ("kuairec 压缩包 md5", kuairec["dataset"]["archive_md5"], KUAIREC_ARCHIVE_MD5),
        ("kuairec 切分协议", kuairec["split"]["strategy"], SPLIT_STRATEGY),
        ("kuasar 下载地址", kuasar["dataset"]["source_url"], KUASAR_ARCHIVE_URL),
        ("kuasar 压缩包 md5", kuasar["dataset"]["archive_md5"], KUASAR_ARCHIVE_MD5),
        ("kuasar small 下载地址", kuasar["dataset"]["small_source_url"], KUASAR_SMALL_ARCHIVE_URL),
        ("kuasar small 压缩包 md5", kuasar["dataset"]["small_archive_md5"], KUASAR_SMALL_ARCHIVE_MD5),
        ("movielens 正反馈阈值", movielens["labels"]["positive_threshold"], DEFAULT_MOVIELENS_POSITIVE_THRESHOLD),
        ("movielens 切分协议", movielens["split"]["strategy"], SPLIT_STRATEGY),
    ]
    for label, config_value, code_value in pairs:
        if config_value != code_value:
            errors.append(f"{label}：配置 {config_value!r} ≠ 代码 {code_value!r}")
    return errors


def check_model_configs() -> list[str]:
    """configs/models/*.toml 的默认值必须是训练 dataclass 默认值的子集。"""
    from recsys.training.train_mf import MFTrainConfig
    from recsys.training.train_sasrec import SASRecTrainConfig
    from recsys.training.trainer import TrainConfig

    dataclasses = {
        "two_tower": TrainConfig,
        "sasrec": SASRecTrainConfig,
        "matrix_factorization": MFTrainConfig,
    }
    errors: list[str] = []
    for stem, config_class in dataclasses.items():
        path = PROJECT_ROOT / "configs" / "models" / f"{stem}.toml"
        if not path.exists():
            errors.append(f"configs/models/{stem}.toml 缺失")
            continue
        config = load_toml(path)
        defaults = config.get("defaults", {})
        available = {field.name: field.default for field in fields(config_class)}
        for key, value in defaults.items():
            if key not in available:
                errors.append(f"{stem}.toml: 未知字段 {key}（{config_class.__name__} 里没有）")
            elif available[key] != value:
                errors.append(
                    f"{stem}.toml: {key}={value!r} 与 {config_class.__name__} 默认值 {available[key]!r} 不一致"
                )
    return errors


def discover_markdown() -> list[Path]:
    paths = [path for path in PROJECT_ROOT.rglob("*.md") if "third_party" not in path.relative_to(PROJECT_ROOT).parts]
    for relative in EXTRA_CHECKED_MARKDOWN:
        path = PROJECT_ROOT / relative
        if path.exists():
            paths.append(path)
    return sorted(set(paths))


def check_markdown_links() -> list[str]:
    errors: list[str] = []
    for path in discover_markdown():
        text = path.read_text(encoding="utf-8")
        for match in LINK_PATTERN.finditer(text):
            target = match.group(1)
            if target.startswith(SKIP_LINK_PREFIXES) or SKIP_LINK_CHARS & set(target):
                continue
            target_path = unquote(target.split("#", maxsplit=1)[0])
            if not target_path:
                continue
            resolved = (path.parent / target_path).resolve()
            if not resolved.exists():
                errors.append(
                    f"{path.relative_to(PROJECT_ROOT)}: 链接受损 → {target}"
                )
    return errors


def check_generated_docs() -> list[str]:
    """zoo.toml 与生成文档是否已同步。"""
    import zoo

    errors = []
    for path, content in zoo.render_all(zoo.load_zoo()).items():
        relative = path.relative_to(PROJECT_ROOT)
        if not path.exists():
            errors.append(f"{relative} 不存在，请运行 python scripts/zoo.py render")
        elif path.read_text(encoding="utf-8") != content:
            errors.append(f"{relative} 与 zoo.toml 不同步，请运行 python scripts/zoo.py render")
    return errors


def check_leftovers() -> tuple[list[str], list[str]]:
    """版本库与缓存残留。

    - 项目根的 `.git` 是正常仓库，不算残留；只有子目录里的 `.git`（嵌套仓库）才是错误；
    - `__pycache__` 等缓存已被 `.gitignore` 排除，本地开发出现属正常，只做提醒。
    """
    errors: list[str] = []
    warnings: list[str] = []
    cache_names = {
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".ipynb_checkpoints",
    }
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_dir():
            continue
        relative = path.relative_to(PROJECT_ROOT)
        if relative.parts and relative.parts[0] in {"data", "artifacts"}:
            continue
        if path.name in cache_names:
            warnings.append(f"本地缓存目录（已 gitignore）：{relative}")
        elif path.name == ".git" and path.parent != PROJECT_ROOT:
            errors.append(f"嵌套 .git：{relative}")
    return errors, warnings


def check_repo_hygiene() -> tuple[list[str], list[str]]:
    """第三方仓库是否仍带 .git，以及是否有异常大文件。"""
    errors: list[str] = []
    warnings: list[str] = []
    third_party = PROJECT_ROOT / "third_party"
    for repo in sorted(path for path in third_party.iterdir() if path.is_dir()):
        if (repo / ".git").exists():
            errors.append(f"{repo.relative_to(PROJECT_ROOT)} 仍有 .git")
    largest = sorted(
        ((path.stat().st_size, path) for path in third_party.rglob("*") if path.is_file()),
        reverse=True,
    )[:3]
    for size, path in largest:
        if size > 5 * 1024 * 1024:
            warnings.append(
                f"third_party 大文件 {path.relative_to(PROJECT_ROOT)}（{size / 1024 / 1024:.1f} MB）"
            )
    return errors, warnings


def check_vendor_sources() -> tuple[list[str], list[str]]:
    """third_party/SOURCES.toml 必须覆盖全部仓库，且 commit/许可证可核对。"""
    errors: list[str] = []
    warnings: list[str] = []
    sources_path = PROJECT_ROOT / "third_party" / "SOURCES.toml"
    if not sources_path.exists():
        return ["third_party/SOURCES.toml 缺失"], warnings
    repos = load_toml(sources_path).get("repos", [])
    registered = {repo.get("name", "") for repo in repos}

    for repo in repos:
        name = repo.get("name", "")
        repo_dir = PROJECT_ROOT / "third_party" / name
        if not repo_dir.is_dir():
            errors.append(f"SOURCES.toml 登记的仓库不存在：third_party/{name}")
            continue
        if not re.fullmatch(r"[0-9a-f]{40}", str(repo.get("commit", ""))):
            errors.append(f"{name}: commit 必须是完整 40 位 SHA")
        if not str(repo.get("url", "")).startswith("https://github.com/"):
            errors.append(f"{name}: url 必须是 GitHub 地址")
        license_file = str(repo.get("license_file", ""))
        if license_file:
            if not (repo_dir / license_file).is_file():
                errors.append(f"{name}: 许可证文件缺失 third_party/{name}/{license_file}")
        else:
            warnings.append(f"{name}: 上游根目录无 LICENSE（{repo.get('license', '')}）")

    actual = {path.name for path in (PROJECT_ROOT / "third_party").iterdir() if path.is_dir()}
    for missing in sorted(actual - registered):
        errors.append(f"third_party/{missing} 未在 SOURCES.toml 登记")
    return errors, warnings


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    errors += check_toml_parsing()
    errors += check_dataset_configs()
    errors += check_model_configs()
    errors += check_markdown_links()
    errors += check_generated_docs()
    leftover_errors, leftover_warnings = check_leftovers()
    errors += leftover_errors
    warnings += leftover_warnings
    hygiene_errors, hygiene_warnings = check_repo_hygiene()
    errors += hygiene_errors
    warnings += hygiene_warnings
    source_errors, source_warnings = check_vendor_sources()
    errors += source_errors
    warnings += source_warnings

    for warning in warnings:
        print(f"[warn] {warning}")
    for error in errors:
        print(f"[error] {error}")

    toml_count = len(discover_toml())
    markdown_count = len(discover_markdown())
    repo_count = sum(1 for path in (PROJECT_ROOT / "third_party").iterdir() if path.is_dir())
    print(
        f"检查完成：{toml_count} 个 TOML、{markdown_count} 个 Markdown、{repo_count} 个第三方仓库；"
        f"{len(errors)} 个错误、{len(warnings)} 个警告"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
