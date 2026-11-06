"""KuaiSAR 全量数据画像（Phase 1.2）。

流式扫描 5 个 CSV，产出可用于决策的统计：规模、行为分布、用户/物品活跃度、
时间覆盖、物品特征覆盖率。内存策略：

* 逐行解析，不把整表读进内存；
* 只保留 `user_id` / `item_id` 两列（int32）用于去重与分布统计，其余指标边读边算；
* 物品特征文件只取需要的那几列，caption 只统计 token 个数，不保留内容。

用法：

    python -m recsys.experiments.analyze_kuasar                  # 全量
    python -m recsys.experiments.analyze_kuasar --max-rows 200000  # 抽样试跑

产出：

    artifacts/kuasar_profile.json   全部统计（供文档与后续决策引用）
    stdout                          人类可读摘要
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from recsys.config import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_DIR
from recsys.data.kuasar import KUASAR_DATASET_DIRNAME

#: item_features 里有超长 caption 字段（>128 KB），默认上限会直接抛 _csv.Error。
csv.field_size_limit(2**31 - 1)

REC_INTER = "KuaiSAR_v2/rec_inter.csv"
SRC_INTER = "KuaiSAR_v2/src_inter.csv"
ITEM_FEATURES = "KuaiSAR_v2/item_features.csv"
USER_FEATURES = "KuaiSAR_v2/user_features.csv"
SOCIAL_NETWORK = "KuaiSAR_v2/social_network.csv"

BEHAVIOR_COLUMNS = ("forward", "like", "follow", "search_photo_related", "search", "click")
_CHUNK_ROWS = 500_000


class ProfileError(RuntimeError):
    """数据文件缺失或格式不符时抛出。"""


def _paths(data_dir: Path) -> dict[str, Path]:
    root = data_dir / KUASAR_DATASET_DIRNAME / "raw"
    paths = {
        "rec_inter": root / REC_INTER,
        "src_inter": root / SRC_INTER,
        "item_features": root / ITEM_FEATURES,
        "user_features": root / USER_FEATURES,
        "social_network": root / SOCIAL_NETWORK,
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise ProfileError(
            f"缺少数据文件：{missing}。先运行 python -m recsys.data.kuasar 下载并解压。"
        )
    return paths


def _open_csv(path: Path):
    file = path.open("r", encoding="utf-8-sig", newline="")
    reader = csv.reader(file)
    header = next(reader)
    return file, reader, {name: index for index, name in enumerate(header)}


def _ratio_histogram(value: float, bins: tuple[float, ...]) -> int:
    for index in range(len(bins) - 1):
        if value < bins[index + 1]:
            return index
    return len(bins) - 1


def _summarise_counts(counts: np.ndarray) -> dict[str, float | int]:
    """把「每个实体一条记录」的计数数组压成分布摘要。"""
    if counts.size == 0:
        return {"entities": 0}
    return {
        "entities": int(counts.size),
        "mean": round(float(counts.mean()), 2),
        "median": round(float(np.median(counts)), 2),
        "p90": round(float(np.percentile(counts, 90)), 2),
        "p99": round(float(np.percentile(counts, 99)), 2),
        "max": int(counts.max()),
        "single_interaction_entities": int((counts == 1).sum()),
    }


def profile_rec_inter(path: Path, max_rows: int | None = None) -> dict[str, Any]:
    """推荐场景：规模、行为占比、曝光结构、时间范围、重复对。"""
    started = time.time()
    file, reader, index = _open_csv(path)
    columns = {name: index[name] for name in ("user_id", "item_id", "timestamp", *BEHAVIOR_COLUMNS)}
    behavior = {name: 0 for name in BEHAVIOR_COLUMNS}
    rows = 0
    all_zero = 0
    duration_zero = 0
    playing_exceeds_duration = 0
    ratio_bins = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, float("inf"))
    ratio_hist = [0] * (len(ratio_bins) - 1)
    ts_min, ts_max = float("inf"), float("-inf")
    user_chunks: list[np.ndarray] = []
    item_chunks: list[np.ndarray] = []
    buffer_users: list[int] = []
    buffer_items: list[int] = []

    try:
        for row in reader:
            rows += 1
            if max_rows is not None and rows > max_rows:
                rows -= 1
                break
            user_id = int(row[columns["user_id"]])
            item_id = int(row[columns["item_id"]])
            buffer_users.append(user_id)
            buffer_items.append(item_id)
            if len(buffer_users) >= _CHUNK_ROWS:
                user_chunks.append(np.asarray(buffer_users, dtype=np.int64))
                item_chunks.append(np.asarray(buffer_items, dtype=np.int64))
                buffer_users.clear()
                buffer_items.clear()

            timestamp = float(row[columns["timestamp"]])
            if timestamp < ts_min:
                ts_min = timestamp
            if timestamp > ts_max:
                ts_max = timestamp

            hit = False
            for name in BEHAVIOR_COLUMNS:
                value = int(row[columns[name]])
                if value:
                    behavior[name] += 1
                    hit = True
            if not hit:
                all_zero += 1

            duration = float(row[index["duration_ms"]])
            playing = float(row[index["playing_time"]])
            if duration <= 0:
                duration_zero += 1
            else:
                if playing > duration:
                    playing_exceeds_duration += 1
                ratio_hist[_ratio_histogram(playing / duration, ratio_bins)] += 1
    finally:
        file.close()

    if buffer_users:
        user_chunks.append(np.asarray(buffer_users, dtype=np.int64))
        item_chunks.append(np.asarray(buffer_items, dtype=np.int64))
    users = np.concatenate(user_chunks) if user_chunks else np.empty(0, dtype=np.int64)
    items = np.concatenate(item_chunks) if item_chunks else np.empty(0, dtype=np.int64)
    unique_users, user_counts = np.unique(users, return_counts=True)
    unique_items, item_counts = np.unique(items, return_counts=True)
    # 用「最大 item + 1」做基数，避免固定 10^10 在超大 ID 上产生碰撞
    item_base = int(unique_items.max()) + 1 if unique_items.size else 1
    pair_keys = users * item_base + items
    pair_count = int(np.unique(pair_keys).size)

    return {
        "file": path.name,
        "size_mb": round(path.stat().st_size / 1048576, 1),
        "rows": rows,
        "sampled": max_rows is not None,
        "elapsed_s": round(time.time() - started, 1),
        "users": int(unique_users.size),
        "items": int(unique_items.size),
        "unique_user_item_pairs": pair_count,
        "duplicate_pair_rate": round((rows - pair_count) / rows, 6) if rows else 0.0,
        "behavior_counts": behavior,
        "behavior_rates": {name: round(value / rows, 6) for name, value in behavior.items()},
        "all_behavior_zero_rows": all_zero,
        "all_behavior_zero_rate": round(all_zero / rows, 6) if rows else 0.0,
        "timestamp_range": {
            "min_ms": int(ts_min) if rows else None,
            "max_ms": int(ts_max) if rows else None,
            "span_days": round((ts_max - ts_min) / 86400000, 2) if rows else None,
        },
        "duration_zero_rows": duration_zero,
        "playing_exceeds_duration_rows": playing_exceeds_duration,
        "watch_ratio_histogram": {
            f"[{ratio_bins[i]},{ratio_bins[i + 1] if ratio_bins[i + 1] != float('inf') else 'inf'})": ratio_hist[i]
            for i in range(len(ratio_hist))
        },
        "user_activity": _summarise_counts(user_counts),
        "item_popularity": _summarise_counts(item_counts),
    }


def profile_src_inter(path: Path, max_rows: int | None = None) -> dict[str, Any]:
    """搜索场景：会话规模、来源、点击次数分布、时间范围。"""
    started = time.time()
    file, reader, index = _open_csv(path)
    rows = 0
    sources: Counter[str] = Counter()
    click_hist = {"0": 0, "1": 0, "2+": 0}
    ts_min, ts_max = float("inf"), float("-inf")
    user_buffer: list[int] = []
    session_buffer: list[int] = []
    user_chunks: list[np.ndarray] = []
    session_chunks: list[np.ndarray] = []
    try:
        for row in reader:
            rows += 1
            if max_rows is not None and rows > max_rows:
                rows -= 1
                break
            user_buffer.append(int(row[index["user_id"]]))
            session_buffer.append(int(float(row[index["search_session_id"]])))
            if len(user_buffer) >= _CHUNK_ROWS:
                user_chunks.append(np.asarray(user_buffer, dtype=np.int64))
                session_chunks.append(np.asarray(session_buffer, dtype=np.int64))
                user_buffer.clear()
                session_buffer.clear()
            sources[row[index["search_session_source"]]] += 1
            clicks = int(float(row[index["click_cnt"]]))
            click_hist["0" if clicks == 0 else ("1" if clicks == 1 else "2+")] += 1
            timestamp = float(row[index["search_session_timestamp"]])
            ts_min = min(ts_min, timestamp)
            ts_max = max(ts_max, timestamp)
    finally:
        file.close()

    if user_buffer:
        user_chunks.append(np.asarray(user_buffer, dtype=np.int64))
        session_chunks.append(np.asarray(session_buffer, dtype=np.int64))
    users = np.concatenate(user_chunks) if user_chunks else np.empty(0, dtype=np.int64)
    sessions = np.concatenate(session_chunks) if session_chunks else np.empty(0, dtype=np.int64)
    unique_users, user_counts = np.unique(users, return_counts=True)
    unique_sessions, session_counts = np.unique(sessions, return_counts=True)
    return {
        "file": path.name,
        "size_mb": round(path.stat().st_size / 1048576, 1),
        "rows": rows,
        "sampled": max_rows is not None,
        "elapsed_s": round(time.time() - started, 1),
        "users": int(unique_users.size),
        "search_sessions": int(unique_sessions.size),
        "session_source": dict(sources.most_common()),
        "click_cnt_distribution": click_hist,
        "timestamp_range": {
            "min_ms": int(ts_min) if rows else None,
            "max_ms": int(ts_max) if rows else None,
            "span_days": round((ts_max - ts_min) / 86400000, 2) if rows else None,
        },
        "user_search_rows": _summarise_counts(user_counts),
        "session_size": _summarise_counts(session_counts),
    }


def profile_item_features(
    path: Path, max_rows: int | None = None, rec_items: set[int] | None = None
) -> dict[str, Any]:
    """物品特征：类目分布、类型、作者、caption token 数、与交互物品的覆盖关系。"""
    started = time.time()
    file, reader, index = _open_csv(path)
    rows = 0
    level1: Counter[str] = Counter()
    item_types: Counter[str] = Counter()
    upload_types: Counter[str] = Counter()
    authors: set[int] = set()
    item_ids: set[int] = set()
    caption_tokens: list[int] = []
    upload_min, upload_max = None, None
    try:
        for row in reader:
            rows += 1
            if max_rows is not None and rows > max_rows:
                rows -= 1
                break
            item_ids.add(int(row[index["item_id"]]))
            level1[row[index["first_level_category_name"]]] += 1
            item_types[row[index["item_type"]]] += 1
            upload_types[row[index["upload_type"]]] += 1
            authors.add(int(float(row[index["author_id"]])))
            # upload_time 是 YYYY-MM-DD 字符串（不是 epoch 毫秒），用字典序取最值即可
            upload_time = row[index["upload_time"]].strip()
            if upload_time:
                upload_min = upload_time if upload_min is None else min(upload_min, upload_time)
                upload_max = upload_time if upload_max is None else max(upload_max, upload_time)
            if len(caption_tokens) < 200_000:
                caption = row[index["caption"]].strip()
                if not caption or caption in {"[]", "null"}:
                    caption_tokens.append(0)
                else:
                    inner = caption.strip("[]")
                    caption_tokens.append(inner.count(",") + 1 if inner else 0)
    finally:
        file.close()

    coverage: dict[str, Any] = {}
    if rec_items is not None:
        covered = len(rec_items & item_ids)
        coverage = {
            "rec_items": len(rec_items),
            "items_with_features": len(item_ids),
            "rec_items_covered": covered,
            "rec_items_coverage_rate": round(covered / len(rec_items), 6) if rec_items else 0.0,
        }
    token_stats = {}
    if caption_tokens:
        array = np.asarray(caption_tokens)
        token_stats = {
            "sampled_rows": int(array.size),
            "mean_tokens": round(float(array.mean()), 2),
            "median_tokens": float(np.median(array)),
            "max_tokens": int(array.max()),
        }
    return {
        "file": path.name,
        "size_mb": round(path.stat().st_size / 1048576, 1),
        "rows": rows,
        "sampled": max_rows is not None,
        "elapsed_s": round(time.time() - started, 1),
        "items": len(item_ids),
        "authors": len(authors),
        "top_first_level_categories": level1.most_common(15),
        "item_type": dict(item_types.most_common()),
        "upload_type": dict(upload_types.most_common()),
        "upload_time_range": {"min": upload_min, "max": upload_max},
        "caption_tokens": token_stats,
        "coverage": coverage,
    }


def profile_user_features(path: Path) -> dict[str, Any]:
    started = time.time()
    file, reader, index = _open_csv(path)
    rows = 0
    levels = {"search_active_level": Counter(), "rec_active_level": Counter()}
    users: set[int] = set()
    try:
        for row in reader:
            rows += 1
            users.add(int(row[index["user_id"]]))
            for column in levels:
                levels[column][row[index[column]]] += 1
    finally:
        file.close()
    return {
        "file": path.name,
        "rows": rows,
        "users": len(users),
        "elapsed_s": round(time.time() - started, 1),
        "search_active_level": dict(sorted(levels["search_active_level"].items())),
        "rec_active_level": dict(sorted(levels["rec_active_level"].items())),
    }


def profile_social_network(path: Path) -> dict[str, Any]:
    started = time.time()
    file, reader, index = _open_csv(path)
    rows = 0
    users: set[int] = set()
    try:
        for row in reader:
            rows += 1
            users.add(int(row[index["user_id"]]))
    finally:
        file.close()
    return {
        "file": path.name,
        "rows": rows,
        "users": len(users),
        "elapsed_s": round(time.time() - started, 1),
    }


def run(data_dir: Path, output: Path, max_rows: int | None = None) -> dict[str, Any]:
    paths = _paths(data_dir)
    print(f"[1/5] rec_inter（{max_rows or '全量'}）…", flush=True)
    rec = profile_rec_inter(paths["rec_inter"], max_rows=max_rows)
    print(f"      rows={rec['rows']:,} users={rec['users']:,} items={rec['items']:,}", flush=True)

    print("[2/5] src_inter …", flush=True)
    src = profile_src_inter(paths["src_inter"], max_rows=max_rows)
    print(f"      rows={src['rows']:,} sessions={src['search_sessions']:,}", flush=True)

    print("[3/5] item_features …", flush=True)
    # 覆盖率需要 rec_inter 的物品集合；抽样模式下它只是子集，标注 sampled 即可
    rec_items = set(_collect_items(paths["rec_inter"], max_rows))
    items = profile_item_features(paths["item_features"], max_rows=max_rows, rec_items=rec_items)
    print(f"      rows={items['rows']:,} items={items['items']:,}", flush=True)

    print("[4/5] user_features …", flush=True)
    users = profile_user_features(paths["user_features"])
    print("[5/5] social_network …", flush=True)
    social = profile_social_network(paths["social_network"])

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_dir": data_dir.as_posix(),
        "sampled": max_rows is not None,
        "max_rows": max_rows,
        "rec_inter": rec,
        "src_inter": src,
        "item_features": items,
        "user_features": users,
        "social_network": social,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ok] 画像写入 {output}")
    return payload


def _collect_items(path: Path, max_rows: int | None) -> list[int]:
    """只再扫一遍取 item_id（覆盖率用），比在内存里常驻全表便宜。"""
    file, reader, index = _open_csv(path)
    column = index["item_id"]
    items: list[int] = []
    try:
        for count, row in enumerate(reader, start=1):
            items.append(int(row[column]))
            if max_rows is not None and count >= max_rows:
                break
    finally:
        file.close()
    return items


def format_summary(payload: dict[str, Any]) -> str:
    rec = payload["rec_inter"]
    src = payload["src_inter"]
    items = payload["item_features"]
    lines = [
        f"采样            : {'是（前 %s 行）' % payload['max_rows'] if payload['sampled'] else '否（全量）'}",
        f"rec_inter       : {rec['rows']:,} 行 / {rec['users']:,} 用户 / {rec['items']:,} 物品 / "
        f"重复对 {rec['duplicate_pair_rate']:.4%}",
        f"  行为占比      : " + " / ".join(f"{k} {v:.2%}" for k, v in rec["behavior_rates"].items()),
        f"  纯曝光行      : {rec['all_behavior_zero_rows']:,}（{rec['all_behavior_zero_rate']:.2%}）",
        f"  时间跨度      : {rec['timestamp_range']['span_days']} 天",
        f"  每用户交互    : 中位 {rec['user_activity']['median']} / p99 {rec['user_activity']['p99']} / 最大 {rec['user_activity']['max']}",
        f"  物品流行度    : 中位 {rec['item_popularity']['median']} / p99 {rec['item_popularity']['p99']} / 最大 {rec['item_popularity']['max']}",
        f"src_inter       : {src['rows']:,} 行 / {src['users']:,} 用户 / {src['search_sessions']:,} 会话",
        f"  来源          : {src['session_source']}",
        f"  点击数分布    : {src['click_cnt_distribution']}",
        f"item_features   : {items['rows']:,} 行 / {items['items']:,} 物品 / {items['authors']:,} 作者",
        f"  交互物品覆盖  : {items['coverage']}",
        f"  caption token : {items['caption_tokens']}",
        f"user_features   : {payload['user_features']['rows']:,} 用户",
        f"social_network  : {payload['social_network']['rows']:,} 条关注边",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_ARTIFACTS_DIR / "kuasar_profile.json"
    )
    parser.add_argument("--max-rows", type=int, default=None, help="抽样行数（调试用）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run(args.data_dir, args.output, max_rows=args.max_rows)
    except ProfileError as error:
        print(f"[error] {error}")
        return 2
    print("\n" + format_summary(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
