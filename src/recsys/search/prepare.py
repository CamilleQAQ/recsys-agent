"""Search MVP 数据准备：KuaiSAR 搜索小样本 + BM25 索引。

流程（只读原始 CSV 的前若干个行，不扫全量）：

1. 流式读 ``src_inter.csv`` 前 ``src_scan_rows`` 行，按搜索会话聚合
   query / 曝光 / 点击；
2. 收集这些会话里出现过的 item_id，再流式读 ``item_features.csv`` 前
   ``item_scan_rows`` 行，只保留目标物品的 caption token 与类目；
3. 选最多 30 个会话，按 item_id 关联后建 BM25 索引，保存到 ``artifacts/``。

词表结论与证据见 ``docs/search.md``；如果证据不足，脚本会拒绝用 KuaiSAR
token ID 建索引，避免把两套词表混用。

用法：

    python -m recsys.search.prepare
    python -m recsys.search.prepare --check
    python -m recsys.search.prepare --force
    python -m recsys.search.prepare \
        --summary-out experiments/2026-11-09-kuasar-search-bm25/build_summary.json

产物（默认 ``artifacts/search/kuasar_bm25/``）：

    index.json      可复用 BM25 索引（跨进程加载）
    sample.json     30 个会话的 query / 曝光 / 点击 / 关联结果（审计用）
    manifest.json   来源、扫描上限、计数、覆盖率、词表证据与 sha256
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import time
import tomllib
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from recsys.search.bm25 import BM25Index, IndexedItem

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "search" / "kuasar_search.toml"

# caption 里有超过 128 KB 的极端值（最长 24,180 个 token），默认上限会抛 _csv.Error。
csv.field_size_limit(2**31 - 1)

_EMPTY_CATEGORY = {"", "empty", "null", "none", "nan", "unknown", "空", "未知"}
_PAPER_URL = "https://arxiv.org/html/2306.07705v4"
_PAPER_QUOTE = (
    "Textual information, such as queries and video titles, underwent word segmentation "
    "and sensitive word removal. Furthermore, terms in texts were randomly hashed into integers."
)


@dataclass(frozen=True)
class ItemRecord:
    item_id: int
    caption_tokens: tuple[int, ...]
    category: str
    category_zh: str
    item_type: str
    author_id: int


@dataclass(frozen=True)
class ExposureRow:
    item_id: int
    click_cnt: int
    click: bool


@dataclass
class SessionRecord:
    session_id: str
    user_id: str
    timestamp: str
    source: str
    keyword: tuple[int, ...]
    order: int
    exposures: list[ExposureRow] = field(default_factory=list)
    keyword_consistent: bool = True


def _display_path(path: Path) -> str:
    """尽量写相对项目根目录的路径，便于报告引用。"""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_token_ids(text: str) -> tuple[int, ...]:
    """解析 KuaiSAR 的 ``[1, 2, 3]`` / ``1,2,3`` token 列表。"""
    text = str(text or "").strip()
    if not text or text.lower() in {"[]", "null", "none"}:
        return ()
    if text.startswith("["):
        try:
            values = json.loads(text)
        except json.JSONDecodeError:
            values = None
        if isinstance(values, list):
            try:
                return tuple(int(value) for value in values)
            except (TypeError, ValueError):
                pass
    inner = text.strip().strip("[]")
    if not inner:
        return ()
    try:
        return tuple(int(part.strip()) for part in inner.split(",") if part.strip())
    except ValueError as error:
        raise ValueError(f"无法解析 token 列表：{text[:80]!r}") from error


def _category_path(values: Iterable[str]) -> str:
    parts: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text.lower() in _EMPTY_CATEGORY:
            continue
        parts.append(text)
    return " / ".join(parts)


def _get(row: list[str], index: dict[str, int], name: str, default: str = "") -> str:
    position = index.get(name)
    return row[position] if position is not None and position < len(row) else default


def scan_sessions(source: Path, max_rows: int) -> tuple[list[SessionRecord], dict[str, Any]]:
    """读 src_inter 前缀并按会话聚合；扫描上限落在会话中间时丢弃该不完整会话。"""
    sessions: OrderedDict[tuple[str, str], SessionRecord] = OrderedDict()
    rows_read = 0
    keyword_mismatch = 0
    partial_sessions_dropped = 0
    has_more = False
    last_key: tuple[str, str] | None = None

    with source.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        index = {name: position for position, name in enumerate(header)}
        required = (
            "user_id",
            "search_session_id",
            "search_session_timestamp",
            "search_session_source",
            "keyword",
            "item_id",
            "click_cnt",
        )
        missing = [name for name in required if name not in index]
        if missing:
            raise ValueError(f"{source} 缺少列 {missing}")

        for row in reader:
            if rows_read >= max_rows:
                has_more = True
                key = (row[index["user_id"]].strip(), row[index["search_session_id"]].strip())
                if last_key is not None and key == last_key:
                    sessions.pop(last_key, None)
                    partial_sessions_dropped += 1
                break

            rows_read += 1
            key = (row[index["user_id"]].strip(), row[index["search_session_id"]].strip())
            last_key = key
            keyword = parse_token_ids(row[index["keyword"]])
            record = sessions.get(key)
            if record is None:
                record = SessionRecord(
                    session_id=key[1],
                    user_id=key[0],
                    timestamp=row[index["search_session_timestamp"]].strip(),
                    source=row[index["search_session_source"]].strip(),
                    keyword=keyword,
                    order=len(sessions),
                )
                sessions[key] = record
            elif keyword != record.keyword:
                record.keyword_consistent = False
                keyword_mismatch += 1

            click_cnt = int(float(row[index["click_cnt"]]))
            record.exposures.append(
                ExposureRow(
                    item_id=int(row[index["item_id"]]),
                    click_cnt=click_cnt,
                    click=click_cnt > 0,
                )
            )

    stats = {
        "src_rows_scanned": rows_read,
        "src_scan_cap_rows": max_rows,
        "src_truncated": has_more,
        "sessions_scanned": len(sessions),
        "partial_sessions_dropped": partial_sessions_dropped,
        "keyword_mismatch_sessions": sum(1 for s in sessions.values() if not s.keyword_consistent),
        "keyword_mismatch_rows": keyword_mismatch,
    }
    return list(sessions.values()), stats


def _wanted_item_ids(sessions: Iterable[SessionRecord]) -> set[int]:
    return {row.item_id for session in sessions for row in session.exposures}


def scan_item_catalog(
    source: Path,
    max_rows: int,
    wanted_ids: set[int],
) -> tuple[dict[int, ItemRecord], set[int], dict[str, Any]]:
    """读 item_features 前缀。

    只把 ``wanted_ids`` 里的物品存进 catalog（控制内存）；同时统计全部扫描行的
    caption token 集合，用于词表证据。没有 caption 的物品不进入 catalog。
    """
    catalog: dict[int, ItemRecord] = {}
    caption_universe: set[int] = set()
    rows_read = 0
    empty_caption_rows = 0
    matched_wanted = 0
    has_more = False

    with source.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        index = {name: position for position, name in enumerate(header)}
        required = ("item_id", "caption", "item_type", "author_id")
        missing = [name for name in required if name not in index]
        if missing:
            raise ValueError(f"{source} 缺少列 {missing}")

        for row in reader:
            if rows_read >= max_rows:
                has_more = True
                break
            rows_read += 1
            tokens = parse_token_ids(row[index["caption"]])
            if not tokens:
                empty_caption_rows += 1
                continue
            caption_universe.update(tokens)
            item_id = int(row[index["item_id"]])
            if item_id not in wanted_ids:
                continue
            matched_wanted += 1
            category = _category_path(
                [
                    _get(row, index, "first_level_category_name_en"),
                    _get(row, index, "second_level_category_name_en"),
                    _get(row, index, "third_level_category_name_en"),
                    _get(row, index, "fourth_level_category_name_en"),
                ]
            )
            category_zh = _category_path(
                [
                    _get(row, index, "first_level_category_name"),
                    _get(row, index, "second_level_category_name"),
                    _get(row, index, "third_level_category_name"),
                    _get(row, index, "fourth_level_category_name"),
                ]
            )
            if not category:
                category = category_zh or "UNKNOWN"
            try:
                author_id = int(float(_get(row, index, "author_id", "-1")))
            except ValueError:
                author_id = -1
            catalog[item_id] = ItemRecord(
                item_id=item_id,
                caption_tokens=tokens,
                category=category,
                category_zh=category_zh,
                item_type=_get(row, index, "item_type").strip() or "UNKNOWN",
                author_id=author_id,
            )

    stats = {
        "item_rows_scanned": rows_read,
        "item_scan_cap_rows": max_rows,
        "item_truncated": has_more,
        "catalog_items_with_caption": len(catalog),
        "empty_caption_rows": empty_caption_rows,
        "wanted_items": len(wanted_ids),
        "wanted_items_found": matched_wanted,
        "caption_token_universe_size": len(caption_universe),
    }
    return catalog, caption_universe, stats


def _unique_items(session: SessionRecord) -> list[int]:
    return list(dict.fromkeys(row.item_id for row in session.exposures))


def _clicked_items(session: SessionRecord, unique_items: Iterable[int]) -> list[int]:
    clicked = {row.item_id for row in session.exposures if row.click}
    return [item_id for item_id in unique_items if item_id in clicked]


def _session_view(session: SessionRecord, catalog: dict[int, ItemRecord]) -> dict[str, Any]:
    unique = _unique_items(session)
    joined = [item_id for item_id in unique if item_id in catalog]
    clicked = _clicked_items(session, unique)
    clicked_joined = [item_id for item_id in clicked if item_id in catalog]
    return {
        "session": session,
        "unique_items": unique,
        "joined_items": joined,
        "clicked_items": clicked,
        "clicked_joined_items": clicked_joined,
        "coverage": len(joined) / len(unique) if unique else 0.0,
    }


def select_sessions(
    sessions: list[SessionRecord],
    catalog: dict[int, ItemRecord],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    """按固定规则选会话与物品池；返回 (选中会话视图, 物品池, 统计)。"""
    max_sessions = int(config["max_sessions"])
    max_items = int(config["max_items"])
    min_unique = int(config["min_unique_items"])
    min_coverage = float(config["min_item_coverage"])
    require_clicked = bool(config.get("require_clicked_join", False))

    candidates: list[dict[str, Any]] = []
    for session in sessions:
        if not session.keyword or not session.keyword_consistent:
            continue
        view = _session_view(session, catalog)
        if len(view["unique_items"]) < min_unique:
            continue
        if view["coverage"] < min_coverage:
            continue
        if require_clicked and not view["clicked_joined_items"]:
            continue
        view["has_clicked_join"] = bool(view["clicked_joined_items"])
        candidates.append(view)

    candidates.sort(
        key=lambda view: (
            0 if view["has_clicked_join"] else 1,
            -len(view["unique_items"]),
            -view["coverage"],
            view["session"].order,
        )
    )
    selected = candidates[:max_sessions]

    pool = sorted({item_id for view in selected for item_id in view["joined_items"]})
    items_truncated = len(pool) > max_items
    pool = pool[:max_items]
    pool_set = set(pool)
    if items_truncated:
        # 物品池按 item_id 截断后，重新计算每个会话的实际关联覆盖率。
        for view in selected:
            view["joined_items"] = [i for i in view["unique_items"] if i in pool_set]
            view["clicked_joined_items"] = [
                i for i in view["clicked_items"] if i in pool_set
            ]
            view["coverage"] = (
                len(view["joined_items"]) / len(view["unique_items"])
                if view["unique_items"]
                else 0.0
            )

    stats = {
        "sessions_candidates": len(candidates),
        "sessions_selected": len(selected),
        "session_limit_hit": len(selected) == max_sessions,
        "sessions_with_clicked_join": sum(1 for view in selected if view["clicked_joined_items"]),
        "items_selected": len(pool),
        "items_truncated": items_truncated,
    }
    return selected, pool, stats


def build_vocab_evidence(
    sessions: list[SessionRecord],
    selected: list[dict[str, Any]],
    catalog: dict[int, ItemRecord],
    caption_universe: set[int],
    config: dict[str, Any],
) -> dict[str, Any]:
    """词表核验证据：文档陈述 + token 范围/重合 + 点击物品命中率。"""
    keyword_universe = {token for session in sessions for token in session.keyword}
    intersection = keyword_universe & caption_universe
    keyword_max = max(keyword_universe) if keyword_universe else 0
    caption_max = max(caption_universe) if caption_universe else 0
    expected = (
        len(keyword_universe) * len(caption_universe) / max(keyword_max, 1)
        if keyword_universe and caption_universe
        else 0.0
    )

    baseline_ids = sorted(catalog)
    baseline_count = min(int(config.get("baseline_items", 5000)), len(baseline_ids))
    baseline_sample = (
        random.Random(int(config.get("baseline_seed", 42))).sample(
            baseline_ids, baseline_count
        )
        if baseline_count
        else []
    )

    clicked_pairs = 0
    clicked_pairs_with_hit = 0
    clicked_hits: list[int] = []
    exposed_pairs = 0
    exposed_pairs_with_hit = 0
    exposed_hits: list[int] = []
    random_means: list[float] = []
    wins = ties = losses = 0
    examples: list[dict[str, Any]] = []

    for view in selected:
        session = view["session"]
        query = set(session.keyword)
        joined = view["joined_items"]
        if not query or not joined:
            continue
        joined_hits = [len(query & set(catalog[item_id].caption_tokens)) for item_id in joined]
        exposed_pairs += len(joined_hits)
        exposed_pairs_with_hit += sum(1 for hit in joined_hits if hit > 0)
        exposed_hits.extend(joined_hits)

        clicked = view["clicked_joined_items"]
        clicked_hits_for_session = [
            len(query & set(catalog[item_id].caption_tokens)) for item_id in clicked
        ]
        clicked_pairs += len(clicked_hits_for_session)
        clicked_pairs_with_hit += sum(1 for hit in clicked_hits_for_session if hit > 0)
        clicked_hits.extend(clicked_hits_for_session)

        random_hits = [
            len(query & set(catalog[item_id].caption_tokens)) for item_id in baseline_sample
        ]
        random_mean = statistics.mean(random_hits) if random_hits else 0.0
        random_means.append(random_mean)
        if clicked_hits_for_session:
            clicked_mean = statistics.mean(clicked_hits_for_session)
            if clicked_mean > random_mean:
                wins += 1
            elif clicked_mean == random_mean:
                ties += 1
            else:
                losses += 1
            if len(examples) < 3:
                best_item = max(
                    clicked, key=lambda item_id: len(query & set(catalog[item_id].caption_tokens))
                )
                tokens = catalog[best_item].caption_tokens
                examples.append(
                    {
                        "session_id": session.session_id,
                        "item_id": best_item,
                        "query_tokens": list(session.keyword),
                        "caption_prefix": list(tokens[:20]),
                        "hit_tokens": sorted(query & set(tokens)),
                        "category": catalog[best_item].category,
                    }
                )

    clicked_mean_hits = statistics.mean(clicked_hits) if clicked_hits else 0.0
    exposed_mean_hits = statistics.mean(exposed_hits) if exposed_hits else 0.0
    random_mean_hits = statistics.mean(random_means) if random_means else 0.0
    observed_vs_expected = len(intersection) / expected if expected else 0.0

    shared_token_space = bool(
        clicked_pairs >= 10
        and clicked_pairs_with_hit / clicked_pairs >= 0.30
        and len(intersection) > 2.0 * expected
        and keyword_max > 0
        and caption_max > 0
        and min(keyword_max, caption_max) / max(keyword_max, caption_max) >= 0.90
    )

    evidence = {
        "conclusion": "shared_token_space" if shared_token_space else "not_confirmed",
        "decision": "use_kuasar_token_ids" if shared_token_space else "local_fixture_required",
        "confidence": "high" if shared_token_space else "insufficient",
        "sources": [
            {
                "kind": "paper",
                "url": _PAPER_URL,
                "section": "3.2 Data Construction",
                "quote": _PAPER_QUOTE,
                "interpretation": (
                    "query 与 video title 使用同一套分词/哈希流程，token 是整数哈希值；"
                    "archive 没有单独提供词表文件。"
                ),
            },
            {
                "kind": "archive",
                "detail": (
                    "KuaiSAR_v2.zip 只有 item_features / rec_inter / src_inter / "
                    "social_network / user_features 五个 CSV，没有 vocab/dictionary 元信息。"
                ),
            },
        ],
        "token_space": {
            "keyword_unique": len(keyword_universe),
            "caption_unique": len(caption_universe),
            "intersection": len(intersection),
            "jaccard": round(len(intersection) / len(keyword_universe | caption_universe), 6)
            if keyword_universe | caption_universe
            else 0.0,
            "keyword_min": min(keyword_universe) if keyword_universe else None,
            "keyword_max": keyword_max or None,
            "caption_min": min(caption_universe) if caption_universe else None,
            "caption_max": caption_max or None,
            "keyword_tokens_absent_from_caption_sample": len(keyword_universe - caption_universe),
            "expected_intersection_uniform": round(expected, 2),
            "observed_vs_expected": round(observed_vs_expected, 2),
        },
        "session_overlap": {
            "baseline_items": baseline_count,
            "baseline_seed": int(config.get("baseline_seed", 42)),
            "clicked_pairs": clicked_pairs,
            "clicked_pairs_with_hit": clicked_pairs_with_hit,
            "clicked_hit_rate": round(clicked_pairs_with_hit / clicked_pairs, 6)
            if clicked_pairs
            else 0.0,
            "clicked_mean_hits": round(clicked_mean_hits, 4),
            "exposed_pairs": exposed_pairs,
            "exposed_pairs_with_hit": exposed_pairs_with_hit,
            "exposed_hit_rate": round(exposed_pairs_with_hit / exposed_pairs, 6)
            if exposed_pairs
            else 0.0,
            "exposed_mean_hits": round(exposed_mean_hits, 4),
            "random_mean_hits": round(random_mean_hits, 6),
            "clicked_vs_random_ratio": round(clicked_mean_hits / random_mean_hits, 2)
            if random_mean_hits
            else None,
            "per_session_clicked_vs_random": {
                "wins": wins,
                "ties": ties,
                "losses": losses,
            },
        },
        "examples": examples,
        "caveat": (
            "没有官方词表文件；token 是随机哈希后的整数，无法反解为文本。"
            "结论基于论文陈述与抽样观测的重合信号；若上游 tokenizer 变化需重新核验。"
        ),
    }
    return evidence


def _session_payload(view: dict[str, Any], catalog: dict[int, ItemRecord]) -> dict[str, Any]:
    session: SessionRecord = view["session"]
    joined = set(view["joined_items"])
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "timestamp": session.timestamp,
        "source": session.source,
        "keyword": list(session.keyword),
        "keyword_consistent": session.keyword_consistent,
        "exposures": [
            {
                "item_id": row.item_id,
                "click_cnt": row.click_cnt,
                "click": int(row.click),
                "joined": row.item_id in joined,
            }
            for row in session.exposures
        ],
        "unique_items": view["unique_items"],
        "joined_items": view["joined_items"],
        "clicked_items": view["clicked_items"],
        "clicked_joined_items": view["clicked_joined_items"],
        "coverage": round(view["coverage"], 6),
    }


def _coverage_stats(selected: list[dict[str, Any]]) -> dict[str, Any]:
    coverages = [view["coverage"] for view in selected]
    return {
        "session_coverage_mean": round(statistics.mean(coverages), 6) if coverages else 0.0,
        "session_coverage_min": round(min(coverages), 6) if coverages else 0.0,
        "session_coverage_max": round(max(coverages), 6) if coverages else 0.0,
        "sessions_fully_covered": sum(1 for value in coverages if value >= 1.0),
        "sessions_ge_50pct": sum(1 for value in coverages if value >= 0.5),
    }


def _counts(
    selected: list[dict[str, Any]],
    catalog: dict[int, ItemRecord],
    pool: list[int],
    session_scan: dict[str, Any],
    selection_stats: dict[str, Any],
    item_stats: dict[str, Any],
) -> dict[str, Any]:
    exposure_rows = sum(len(view["session"].exposures) for view in selected)
    joined_rows = sum(
        1
        for view in selected
        for row in view["session"].exposures
        if row.item_id in set(view["joined_items"])
    )
    clicked_rows = sum(
        1 for view in selected for row in view["session"].exposures if row.click
    )
    clicked_joined_rows = sum(
        1
        for view in selected
        for row in view["session"].exposures
        if row.click and row.item_id in set(view["joined_items"])
    )
    unique_items = {item_id for view in selected for item_id in view["unique_items"]}
    unique_joined = {item_id for view in selected for item_id in view["joined_items"]}
    clicked_unique = {item_id for view in selected for item_id in view["clicked_items"]}
    clicked_joined_unique = {
        item_id for view in selected for item_id in view["clicked_joined_items"]
    }
    return {
        **session_scan,
        **selection_stats,
        **item_stats,
        "sessions_scanned": session_scan.get("sessions_scanned", 0),
        "item_rows_scanned": item_stats.get("item_rows_scanned", 0),
        "exposure_rows": exposure_rows,
        "exposure_rows_joined": joined_rows,
        "exposure_row_coverage": round(joined_rows / exposure_rows, 6) if exposure_rows else 0.0,
        "unique_exposure_items": len(unique_items),
        "unique_items_joined": len(unique_joined),
        "unique_item_coverage": round(len(unique_joined) / len(unique_items), 6)
        if unique_items
        else 0.0,
        "clicked_rows": clicked_rows,
        "clicked_rows_joined": clicked_joined_rows,
        "clicked_unique_items": len(clicked_unique),
        "clicked_unique_items_joined": len(clicked_joined_unique),
        "indexed_items": len(pool),
        "catalog_items": len(catalog),
    }


def build_search_assets(
    config: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    """构建/复用索引与样本；返回 manifest。"""
    search_config = config["search"]
    sampling = config["sampling"]
    bm25_config = config["bm25"]

    source_root = _resolve_path(search_config["source_root"])
    artifact_root = _resolve_path(search_config["artifact_root"])
    index_path = artifact_root / search_config["index_file"]
    sample_path = artifact_root / search_config["sample_file"]
    manifest_path = artifact_root / search_config["manifest_file"]

    config_fingerprint = _fingerprint(config)
    if not force and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_fingerprint") != config_fingerprint:
            print("[stale] 配置已变化，重建索引", flush=True)
        else:
            problems = []
            for key, path in (("index", index_path), ("sample", sample_path)):
                expected = manifest.get(key, {}).get("sha256")
                if not path.exists():
                    problems.append(f"{path.name} 不存在")
                elif expected and _sha256_file(path) != expected:
                    problems.append(f"{path.name} sha256 不一致")
            if not problems:
                print(f"[reuse] {artifact_root} 校验通过（{manifest['index']['num_docs']} 个物品）")
                return manifest
            print(f"[stale] 产物校验失败，重建：{problems[:3]}", flush=True)

    src_path = source_root / sampling["src_inter"]
    item_path = source_root / sampling["item_features"]
    for path in (src_path, item_path):
        if not path.exists():
            raise FileNotFoundError(
                f"找不到 {path}；先运行 python -m recsys.data.kuasar 下载并解压 KuaiSAR_v2。"
            )

    started = time.time()
    print(f"[1/3] 扫描搜索日志 {_display_path(src_path)}（上限 {sampling['src_scan_rows']:,} 行）…", flush=True)
    sessions, session_scan = scan_sessions(src_path, int(sampling["src_scan_rows"]))
    print(
        f"      {session_scan['src_rows_scanned']:,} 行 / {session_scan['sessions_scanned']:,} 个完整会话"
        f"；丢弃截断会话 {session_scan['partial_sessions_dropped']} 个",
        flush=True,
    )

    wanted_ids = _wanted_item_ids(sessions)
    print(
        f"[2/3] 扫描物品特征 {_display_path(item_path)}（上限 {sampling['item_scan_rows']:,} 行；"
        f"目标物品 {len(wanted_ids):,} 个）…",
        flush=True,
    )
    catalog, caption_universe, item_scan = scan_item_catalog(
        item_path, int(sampling["item_scan_rows"]), wanted_ids
    )
    print(
        f"      命中目标物品 {item_scan['wanted_items_found']:,} 个；"
        f"caption token 词表 {len(caption_universe):,}",
        flush=True,
    )

    selected, pool, selection_stats = select_sessions(sessions, catalog, sampling)
    if not selected or not pool:
        raise ValueError(
            "没有选中可用会话/物品；请检查扫描上限与 min_item_coverage。"
            f"（会话 {session_scan['sessions_scanned']}，候选 {selection_stats['sessions_candidates']}）"
        )

    vocab_evidence = build_vocab_evidence(
        sessions, selected, catalog, caption_universe, sampling
    )
    if vocab_evidence["decision"] != "use_kuasar_token_ids":
        raise RuntimeError(
            "词表证据不足，拒绝用 KuaiSAR token ID 建索引；"
            "请改用明确标注的本地夹具，并在 docs/search.md 记录原因。"
        )

    print(f"[3/3] 建 BM25 索引：{len(pool)} 个物品（k1={bm25_config['k1']}, b={bm25_config['b']}）…", flush=True)
    items = [
        IndexedItem(
            item_id=item_id,
            caption_tokens=catalog[item_id].caption_tokens,
            category=catalog[item_id].category,
            category_zh=catalog[item_id].category_zh,
            item_type=catalog[item_id].item_type,
            author_id=catalog[item_id].author_id,
        )
        for item_id in pool
    ]
    index = BM25Index(items, k1=float(bm25_config["k1"]), b=float(bm25_config["b"]))
    index.save(index_path)

    sample_payload = {
        "format": "recsys.search.sample.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {
            "dataset": search_config["dataset"],
            "raw_root": _display_path(source_root),
            "src_inter": _display_path(src_path),
            "item_features": _display_path(item_path),
        },
        "sampling": sampling,
        "counts": _counts(
            selected, catalog, pool, session_scan, selection_stats, item_scan
        ),
        "coverage": _coverage_stats(selected),
        "vocab_evidence": vocab_evidence,
        "sessions": [_session_payload(view, catalog) for view in selected],
    }
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(
        json.dumps(sample_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    index_sha = _sha256_file(index_path)
    sample_sha = _sha256_file(sample_path)
    elapsed = round(time.time() - started, 3)
    manifest = {
        "format": "recsys.search.build_manifest.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": elapsed,
        "config_fingerprint": config_fingerprint,
        "source": {
            "dataset": search_config["dataset"],
            "raw_root": _display_path(source_root),
            "src_inter": {
                "path": _display_path(src_path),
                "size_bytes": src_path.stat().st_size,
                **session_scan,
            },
            "item_features": {
                "path": _display_path(item_path),
                "size_bytes": item_path.stat().st_size,
                **item_scan,
            },
        },
        "sampling": sampling,
        "counts": sample_payload["counts"],
        "coverage": sample_payload["coverage"],
        "vocab_evidence": vocab_evidence,
        "index": {
            "path": _display_path(index_path),
            "sha256": index_sha,
            "size_bytes": index_path.stat().st_size,
            **index.summary(),
        },
        "sample": {
            "path": _display_path(sample_path),
            "sha256": sample_sha,
            "size_bytes": sample_path.stat().st_size,
        },
        "notes": [
            "只扫描 src_inter / item_features 的前缀；两个 truncated 标记见 source 字段。",
            "样本最多 30 个会话、1,000 个物品；本轮实际计数见 counts。",
            "点击只用于覆盖率与证据检查，BM25 排序只使用 caption token。",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[ok] 写入 {_display_path(index_path)}（{index_path.stat().st_size:,} B）", flush=True)
    print(f"[ok] 写入 {_display_path(sample_path)}（{sample_path.stat().st_size:,} B）", flush=True)
    return manifest


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    overrides = {
        "src_scan_rows": args.src_scan_rows,
        "item_scan_rows": args.item_scan_rows,
        "max_sessions": args.max_sessions,
        "max_items": args.max_items,
    }
    for key, value in overrides.items():
        if value is not None:
            config["sampling"][key] = value
    return config


def check_assets(config: dict[str, Any]) -> int:
    search_config = config["search"]
    artifact_root = _resolve_path(search_config["artifact_root"])
    manifest_path = artifact_root / search_config["manifest_file"]
    if not manifest_path.exists():
        print(f"[error] 找不到 {_display_path(manifest_path)}；先运行 python -m recsys.search.prepare")
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    for key in ("index", "sample"):
        path = artifact_root / search_config[f"{key}_file"]
        expected = manifest.get(key, {}).get("sha256")
        if not path.exists():
            problems.append(f"{path.name} 不存在")
        elif expected and _sha256_file(path) != expected:
            problems.append(f"{path.name} sha256 不一致")
    if problems:
        print(f"[fail] {len(problems)} 项不一致：{problems[:5]}")
        return 1
    index = BM25Index.load(artifact_root / search_config["index_file"])
    print(
        f"[ok] 索引有效：{index.num_docs} 个物品 / {index.vocab_size} 个 token / "
        f"avgdl {index.avgdl:.2f}；sha256 {manifest['index']['sha256']}"
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--force", action="store_true", help="忽略已有产物，强制重建")
    parser.add_argument("--check", action="store_true", help="只校验产物哈希与索引可加载性")
    parser.add_argument("--summary-out", type=Path, default=None, help="额外写一份汇总 JSON（不含逐条样本）")
    parser.add_argument("--src-scan-rows", type=int, default=None)
    parser.add_argument("--item-scan-rows", type=int, default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = apply_overrides(load_config(args.config), args)
    if args.check:
        return check_assets(config)

    manifest = build_search_assets(config, force=args.force)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[ok] 汇总写入 {args.summary_out}")
    print(
        json.dumps(
            {
                "sessions": manifest["counts"]["sessions_selected"],
                "exposure_rows": manifest["counts"]["exposure_rows"],
                "items": manifest["counts"]["indexed_items"],
                "coverage": manifest["coverage"],
                "index": {
                    "num_docs": manifest["index"]["num_docs"],
                    "vocab_size": manifest["index"]["vocab_size"],
                    "sha256": manifest["index"]["sha256"],
                },
                "elapsed_s": manifest["elapsed_s"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
