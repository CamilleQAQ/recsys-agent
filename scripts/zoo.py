"""Model Zoo 目录的校验与文档渲染。

`models/zoo.toml` 是唯一的机读事实来源；本脚本负责两件事：

    python scripts/zoo.py check    # 校验字段、源码路径、许可证文件、适配器注册
    python scripts/zoo.py render   # 从 zoo.toml 生成 docs/model-zoo.md 与分类 README

只用标准库（tomllib），避免为了读一个索引再引入 YAML 依赖。
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections import Counter, OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZOO_PATH = PROJECT_ROOT / "models" / "zoo.toml"

CATEGORIES = OrderedDict(
    [
        ("retrieval", "召回模型"),
        ("ranking", "排序模型"),
        ("infrastructure", "基础设施"),
    ]
)
FAMILIES = {"first_party", "vendor"}
COMPLETENESS = {"ready", "upstream_ready", "blocked"}
REQUIRED_FIELDS = (
    "id",
    "name",
    "category",
    "family",
    "repo",
    "code",
    "license",
    "upstream_url",
    "upstream_commit",
    "principle",
    "use_cases",
    "deps",
    "python",
    "train_entry",
    "infer_entry",
    "data_format",
    "checkpoint",
    "completeness",
    "completeness_note",
    "adapter",
)
COMPLETENESS_LABELS = {
    "ready": "可直接运行",
    "upstream_ready": "上游可用，未接入",
    "blocked": "受依赖/权重限制",
}
COMPLETENESS_ORDER = ("ready", "upstream_ready", "blocked")

GENERATED_HEADER = (
    "<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；"
    "改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->"
)


def load_zoo(path: Path = ZOO_PATH) -> dict:
    with path.open("rb") as file:
        return tomllib.load(file)


def repo_license_files(repo_dir: Path) -> list[str]:
    """返回仓库根下看起来是许可证的文件名（大小写不敏感）。"""
    if not repo_dir.is_dir():
        return []
    candidates = []
    for entry in sorted(repo_dir.iterdir()):
        if not entry.is_file():
            continue
        lowered = entry.name.lower()
        if lowered.startswith("license") or lowered.startswith("licence"):
            candidates.append(entry.name)
    return candidates


def validate(zoo: dict) -> tuple[list[str], list[str]]:
    """返回 (errors, warnings)；errors 会让 check 失败。"""
    errors: list[str] = []
    warnings: list[str] = []
    models = zoo.get("models") or []
    if not models:
        return ["zoo.toml 里没有任何 [[models]] 条目"], warnings

    for key in (zoo.get("categories") or {}):
        if key not in CATEGORIES:
            errors.append(f"[categories.{key}] 不是已知分类（可选：{', '.join(CATEGORIES)}）")

    seen_ids: set[str] = set()
    for index, model in enumerate(models, start=1):
        model_id = model.get("id") or f"<第 {index} 条缺少 id>"
        missing = [field for field in REQUIRED_FIELDS if field not in model]
        if missing:
            errors.append(f"{model_id}: 缺字段 {', '.join(missing)}")
            continue
        if model_id in seen_ids:
            errors.append(f"{model_id}: id 重复")
        seen_ids.add(model_id)

        if model["category"] not in CATEGORIES:
            errors.append(f"{model_id}: category 非法（{model['category']}）")
        if model["family"] not in FAMILIES:
            errors.append(f"{model_id}: family 非法（{model['family']}）")
        if model["completeness"] not in COMPLETENESS:
            errors.append(f"{model_id}: completeness 非法（{model['completeness']}）")

        for relative in model["code"]:
            if not (PROJECT_ROOT / relative).exists():
                errors.append(f"{model_id}: code 路径不存在 {relative}")

        if model["family"] == "vendor":
            repo = model["repo"]
            if not repo:
                errors.append(f"{model_id}: vendor 模型必须填 repo")
                continue
            repo_dir = PROJECT_ROOT / "third_party" / repo
            if not repo_dir.is_dir():
                errors.append(f"{model_id}: 找不到仓库目录 third_party/{repo}")
                continue
            if not repo_license_files(repo_dir):
                warnings.append(
                    f"{model_id}: third_party/{repo} 根目录没有 LICENSE 文件（见 third_party/LICENSES.md）"
                )
            if not str(model["upstream_commit"]).strip():
                warnings.append(f"{model_id}: vendor 模型缺少 upstream_commit")
        else:
            if model["repo"]:
                errors.append(f"{model_id}: first_party 模型不应填 repo")

        if model["adapter"]:
            if not (PROJECT_ROOT / "src" / "recsys" / "adapters" / "builtin.py").exists():
                warnings.append(f"{model_id}: 引用适配器但 adapters/builtin.py 不存在")
    return errors, warnings


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def render_summary_table(models: list[dict]) -> str:
    counts = Counter(model["completeness"] for model in models)
    lines = [
        "| 分类 | 模型数 | " + " | ".join(COMPLETENESS_LABELS[key] for key in COMPLETENESS_ORDER) + " |",
        "| --- | --- | " + " | ".join("---" for _ in COMPLETENESS_ORDER) + " |",
    ]
    for category, _title in CATEGORIES.items():
        in_category = [model for model in models if model["category"] == category]
        counts_cell = [
            str(sum(1 for model in in_category if model["completeness"] == key))
            for key in COMPLETENESS_ORDER
        ]
        lines.append(
            f"| {category} | {len(in_category)} | " + " | ".join(counts_cell) + " |"
        )
    totals = [str(counts.get(key, 0)) for key in COMPLETENESS_ORDER]
    lines.append(f"| **合计** | **{len(models)}** | " + " | ".join(f"**{v}**" for v in totals) + " |")
    return "\n".join(lines)


def render_model_card(model: dict) -> str:
    if model["family"] == "vendor":
        source = f"third_party/{model['repo']}（上游 {model['upstream_url']}）"
    else:
        source = "本项目自研（src/）"
    code = "、".join(f"`{path}`" for path in model["code"])
    deps = "、".join(model["deps"]) if model["deps"] else "无"
    lines = [
        f"### {model['name']}",
        "",
        f"- **ID**：`{model['id']}`",
        f"- **分类**：{CATEGORIES[model['category']]} / {model['family']}",
        f"- **来源**：{source}",
    ]
    if model["upstream_commit"]:
        lines.append(f"- **Commit**：`{model['upstream_commit']}`")
    lines += [
        f"- **许可证**：{model['license']}",
        f"- **源码**：{code}",
        f"- **原理**：{model['principle']}",
        f"- **适用场景**：{model['use_cases']}",
        f"- **依赖 / Python**：{deps}；Python {model['python']}",
        f"- **训练入口**：{model['train_entry']}",
        f"- **推理入口**：{model['infer_entry']}",
        f"- **数据格式**：{model['data_format']}",
        f"- **保存/加载**：{model['checkpoint']}",
        f"- **代码完整度**：{COMPLETENESS_LABELS[model['completeness']]}（{model['completeness']}）—— {model['completeness_note']}",
        f"- **适配层**：{model['adapter'] or '尚未接入'}",
    ]
    if model.get("notes"):
        lines.append(f"- **备注**：{model['notes']}")
    lines.append("")
    return "\n".join(lines)


def render_overview(zoo: dict) -> str:
    models = zoo["models"]
    lines = [
        GENERATED_HEADER,
        "",
        "# Model Zoo 总索引",
        "",
        "每个模型的名称、来源、原理、依赖、训练/推理入口、数据格式、保存方式和代码完整度。",
        "召回/排序/基础设施三个分类的细分视图见 `models/<category>/README.md`。",
        "",
        "## 统计",
        "",
        render_summary_table(models),
        "",
        "## 一览",
        "",
        "| id | 模型 | 分类 | 来源 | 完整度 | 适配层 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model in models:
        family = "自研" if model["family"] == "first_party" else f"vendor:{model['repo']}"
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                model["id"],
                _cell(model["name"]),
                model["category"],
                family,
                COMPLETENESS_LABELS[model["completeness"]],
                model["adapter"] or "—",
            )
        )
    for category in CATEGORIES:
        lines += ["", f"## {category_meta(zoo, category)['title']}（{category}）", ""]
        summary = category_meta(zoo, category)["summary"]
        if summary:
            lines += [summary, ""]
        for model in models:
            if model["category"] == category:
                lines.append(render_model_card(model))
    return "\n".join(lines)


def category_meta(zoo: dict, category: str) -> dict[str, str]:
    """分类页的标题/说明；缺省时用内置标题，保证渲染永不失败。"""
    configured = (zoo.get("categories") or {}).get(category) or {}
    return {
        "title": configured.get("title", CATEGORIES[category]),
        "summary": configured.get("summary", ""),
        "hint": configured.get("hint", ""),
    }


def render_category_page(
    category: str, title: str, models: list[dict], summary: str = "", hint: str = ""
) -> str:
    in_category = [model for model in models if model["category"] == category]
    lines = [
        GENERATED_HEADER,
        "",
        f"# {title}（{category}）",
        "",
        "本页由 `models/zoo.toml` 生成；源码仍在 `src/`（自研）或 `third_party/`（vendor），"
        "这里只做分类索引，不搬运代码，避免破坏上游相对导入。",
        "",
    ]
    if summary:
        lines += [summary, ""]
    if hint:
        lines += [f"> {hint}", ""]
    lines += [
        "| id | 模型 | 来源 | 关键源码 | 完整度 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for model in in_category:
        family = "自研" if model["family"] == "first_party" else f"vendor:{model['repo']}"
        first_code = model["code"][0] if model["code"] else "—"
        lines.append(
            "| `{}` | {} | {} | `{}` | {} |".format(
                model["id"],
                _cell(model["name"]),
                family,
                first_code,
                COMPLETENESS_LABELS[model["completeness"]],
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_dossier(model: dict) -> str:
    """单模型技术档案（PROJECT_MASTER §9.2），只对已接入（ready）的模型生成。"""
    family = "本项目自研" if model["family"] == "first_party" else f"vendor：third_party/{model['repo']}"
    lines = [
        GENERATED_HEADER,
        "",
        f"# {model['name']} 技术档案",
        "",
        "## 基本信息",
        "",
        f"- **ID**：`{model['id']}`",
        f"- **分类**：{CATEGORIES[model['category']]}（{model['category']}）",
        f"- **来源**：{family}",
        f"- **许可证**：{model['license']}",
        f"- **上游**：{model['upstream_url']}",
    ]
    if model["upstream_commit"]:
        lines.append(f"- **Commit**：`{model['upstream_commit']}`")
    lines += [
        f"- **源码**：{'、'.join(f'`{path}`' for path in model['code'])}",
        "",
        "## 原理与适用场景",
        "",
        f"- **原理**：{model['principle']}",
        f"- **适用场景**：{model['use_cases']}",
        "",
        "## 运行方式",
        "",
        f"- **依赖 / Python**：{'、'.join(model['deps']) or '无'}；Python {model['python']}",
        f"- **训练入口**：{model['train_entry']}",
        f"- **推理入口**：{model['infer_entry']}",
        f"- **数据格式**：{model['data_format']}",
        f"- **保存/加载**：{model['checkpoint']}",
        f"- **适配层**：{model['adapter'] or '尚未接入'}",
        "",
        "## 状态与局限",
        "",
        f"- **代码完整度**：{COMPLETENESS_LABELS[model['completeness']]}（{model['completeness']}）",
        f"- **依据**：{model['completeness_note']}",
        f"- **已知局限**：{model.get('limitations', '暂无记录（接入实验后补充）')}",
        "",
        "## 实验记录",
        "",
        "每次正式实验在本节追加一行：日期、数据版本、关键参数、指标、结论、"
        "对应 `experiments/<run>/` 目录。没有实验记录时不要写结论。",
        "",
        "| 日期 | 数据集 | 参数摘要 | 指标 | 结论 | 记录目录 |",
        "| --- | --- | --- | --- | --- | --- |",
        "| — | — | — | — | 尚未开始系统性实验 | — |",
        "",
    ]
    return "\n".join(lines)


def render_dossier_index(ready_models: list[dict]) -> str:
    lines = [
        GENERATED_HEADER,
        "",
        "# 模型技术档案",
        "",
        "已接入统一接口（`completeness = ready`）的模型各有一份独立档案，"
        "记录原理、入口、工件、已知局限与实验记录；尚未接入的模型信息在 "
        "[../model-zoo.md](../model-zoo.md) 的卡片里。",
        "",
        "规则（由 `scripts/zoo.py` 自动执行）：`models/zoo.toml` 里某条记录把 "
        "`completeness` 改成 `ready` 并重新 render，这里就会多一份档案；"
        "`limitations` 字段直接写进「已知局限」，没填时显示「暂无记录」——宁可留空，也不编。",
        "",
        "| 模型 | 档案 |",
        "| --- | --- |",
    ]
    for model in ready_models:
        lines.append(f"| {_cell(model['name'])} | [`{model['id']}.md`]({model['id']}.md) |")
    lines.append("")
    return "\n".join(lines)


def render_all(zoo: dict) -> dict[Path, str]:
    models = zoo["models"]
    outputs = {PROJECT_ROOT / "docs" / "model-zoo.md": render_overview(zoo)}
    for category in CATEGORIES:
        meta = category_meta(zoo, category)
        outputs[PROJECT_ROOT / "models" / category / "README.md"] = render_category_page(
            category, meta["title"], models, meta["summary"], meta["hint"]
        )
    ready_models = [model for model in models if model["completeness"] == "ready"]
    dossier_dir = PROJECT_ROOT / "docs" / "models"
    outputs[dossier_dir / "README.md"] = render_dossier_index(ready_models)
    for model in ready_models:
        outputs[dossier_dir / f"{model['id']}.md"] = render_dossier(model)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("check", "render"), nargs="?", default="check"
    )
    args = parser.parse_args(argv)

    zoo = load_zoo()
    models = zoo.get("models", [])
    errors, warnings = validate(zoo)
    for warning in warnings:
        print(f"[warn] {warning}")
    for error in errors:
        print(f"[error] {error}")

    if args.command == "render":
        if errors:
            print("\n有错误，拒绝渲染。")
            return 1
        for path, content in render_all(zoo).items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(f"[write] {path.relative_to(PROJECT_ROOT).as_posix()}")
        return 0

    counts = Counter(model.get("completeness", "?") for model in models)
    by_category = Counter(model.get("category", "?") for model in models)
    print(
        "模型 {total} 个：{retrieval} 召回 / {ranking} 排序 / {infrastructure} 基础设施；"
        "ready={ready}, upstream_ready={upstream_ready}, blocked={blocked}".format(
            total=len(models),
            retrieval=by_category.get("retrieval", 0),
            ranking=by_category.get("ranking", 0),
            infrastructure=by_category.get("infrastructure", 0),
            ready=counts.get("ready", 0),
            upstream_ready=counts.get("upstream_ready", 0),
            blocked=counts.get("blocked", 0),
        )
    )
    if errors:
        print(f"\n校验失败：{len(errors)} 个错误")
        return 1
    print("校验通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
