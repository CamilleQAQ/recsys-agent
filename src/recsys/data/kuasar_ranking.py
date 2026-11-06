"""KuaiSAR 排序候选与特征样本（Phase 1.4 收尾项）。

召回用「正样本对」，排序用「曝光候选 + 点击标签」。KuaiSAR 的 `rec_inter` 天然就是
``一次请求 = 一组候选``：同一 `(user_id, timestamp)` 的所有行是这次请求曝光的物品，
`click` 是标签（0/1）。

产出（`data/kuasar/processed/ranking/`）：

    {train,validation,test}_items.npy      候选 item_idx（int32，按请求连续排列）
    {train,validation,test}_labels.npy     点击标签（int8）
    {train,validation,test}_users.npy      候选所属用户（int32）
    {train,validation,test}_groups.npy     每个请求的候选数（int32，LightGBM 的 group 参数）
    user_levels.npy                        用户活跃度（num_users × 2，int8）
    schema.json                            统计口径与计数

切分与 `kuasar_store` 完全一致（请求级、train < validation < test），不重新定义口径。
特征不落盘：候选矩阵在训练时按批 join 物品/用户特征（见 `iter_candidate_batches`），
避免 14M × 9 列 float32 ≈ 500 MB 常驻。

用法：

    python -m recsys.data.kuasar_ranking          # 构建
    python -m recsys.data.kuasar_ranking --check  # 校验指纹
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from recsys.config import DEFAULT_DATA_DIR
from recsys.data.kuasar import KUASAR_DATASET_DIRNAME
from recsys.data.kuasar_features import load_item_features
from recsys.data.kuasar_store import dataset_dir as kuasar_dataset_dir
from recsys.data.kuasar_store import load_processed, processed_dir

REC_INTER = "KuaiSAR_v2/rec_inter.csv"
USER_FEATURES = "KuaiSAR_v2/user_features.csv"
POSITIVE_FIELD = "click"

#: 默认候选特征列（物品侧）；顺序固定，写进 schema。
ITEM_FEATURE_COLUMNS = (
    "category_first_code",
    "category_second_code",
    "author_freq",
    "item_type_code",
    "upload_type_code",
    "upload_days",
    "upload_valid",
    "caption_tokens",
    "is_ad",
)
USER_FEATURE_COLUMNS = ("search_active_level", "rec_active_level")


def ranking_dir(dataset_root: Path) -> Path:
    return dataset_root / "processed" / "ranking"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_exposures(path: Path, max_rows: int | None = None) -> dict[str, np.ndarray]:
    """读 rec_inter 的四列（原始 id / 时间戳 / 点击），数组化后再映射，避免逐行 searchsorted。"""
    import csv

    csv.field_size_limit(2**31 - 1)
    file = path.open("r", encoding="utf-8-sig", newline="")
    reader = csv.reader(file)
    header = next(reader)
    index = {name: position for position, name in enumerate(header)}
    for required in ("user_id", "item_id", "timestamp", POSITIVE_FIELD):
        if required not in index:
            raise ValueError(f"{path} 缺少列 {required}")

    users: list[np.ndarray] = []
    items: list[np.ndarray] = []
    stamps: list[np.ndarray] = []
    clicks: list[np.ndarray] = []
    buf_u: list[int] = []
    buf_i: list[int] = []
    buf_t: list[int] = []
    buf_c: list[int] = []
    rows = 0

    def flush() -> None:
        if not buf_u:
            return
        users.append(np.asarray(buf_u, dtype=np.int32))
        items.append(np.asarray(buf_i, dtype=np.int32))
        stamps.append(np.asarray(buf_t, dtype=np.int64))
        clicks.append(np.asarray(buf_c, dtype=np.int8))
        buf_u.clear()
        buf_i.clear()
        buf_t.clear()
        buf_c.clear()

    try:
        for row in reader:
            rows += 1
            if max_rows is not None and rows > max_rows:
                rows -= 1
                break
            buf_u.append(int(row[index["user_id"]]))
            buf_i.append(int(row[index["item_id"]]))
            buf_t.append(int(float(row[index["timestamp"]])))
            buf_c.append(1 if int(float(row[index[POSITIVE_FIELD]])) else 0)
            if len(buf_u) >= 1_000_000:
                flush()
    finally:
        file.close()
    flush()
    return {
        "users": np.concatenate(users),
        "items": np.concatenate(items),
        "timestamps": np.concatenate(stamps),
        "clicks": np.concatenate(clicks),
        "rows": rows,
    }


def _user_boundaries(store: dict[str, Any], num_users: int) -> tuple[np.ndarray, np.ndarray]:
    """从 processed 切分里恢复每个用户的验证/测试请求时间戳。"""
    sentinel = np.iinfo(np.int64).min
    val = np.full(num_users, sentinel, dtype=np.int64)
    test = np.full(num_users, sentinel, dtype=np.int64)
    split = store["split"]
    val_users = np.asarray(split["validation"]["users"])
    test_users = np.asarray(split["test"]["users"])
    if val_users.size:
        val[val_users] = np.asarray(split["validation"]["timestamps"])
    if test_users.size:
        test[test_users] = np.asarray(split["test"]["timestamps"])
    return val, test


def _load_user_levels(dataset_root: Path, user_ids: np.ndarray) -> np.ndarray | None:
    """user_features.csv → (num_users, 2) int8；按 user_idx 对齐。"""
    source = dataset_root / "raw" / USER_FEATURES
    if not source.exists():
        return None
    import csv

    levels = np.zeros((user_ids.size, len(USER_FEATURE_COLUMNS)), dtype=np.int8)
    with source.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            raw = int(row["user_id"])
            position = int(np.searchsorted(user_ids, raw))
            if position >= user_ids.size or int(user_ids[position]) != raw:
                continue
            for column, name in enumerate(USER_FEATURE_COLUMNS):
                levels[position, column] = int(row[name])
    return levels


def build_ranking_samples(
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    max_rows: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    dataset_root = kuasar_dataset_dir(data_dir)
    target = ranking_dir(dataset_root)
    if target.exists() and not force and (target / "schema.json").exists():
        schema = json.loads((target / "schema.json").read_text(encoding="utf-8"))
        problems = verify_ranking_store(dataset_root, schema)
        if not problems:
            print(f"[reuse] {target} 排序样本校验通过")
            return schema
        print(f"[stale] 排序样本校验失败，重建：{problems[:3]}", flush=True)

    started = time.time()
    print("[1/4] 读取曝光记录 …", flush=True)
    raw = _read_exposures(dataset_root / "raw" / REC_INTER, max_rows=max_rows)
    print(f"      {raw['rows']:,} 行", flush=True)

    print("[2/4] 映射 ID 并对齐切分 …", flush=True)
    store = load_processed(dataset_root)
    user_ids = np.asarray(store["user_ids"])
    item_ids = np.asarray(store["item_ids"])
    user_index = np.searchsorted(user_ids, raw["users"])
    item_index = np.searchsorted(item_ids, raw["items"])
    # 理论上所有 ID 都在词表内（词表就是从同一份文件建的），越界说明缓存过期
    if (user_index >= user_ids.size).any() or (item_index >= item_ids.size).any():
        raise ValueError("曝光记录里出现词表外的 ID；processed 缓存可能过期，请重建")
    if max_rows is not None:
        print("      注意：抽样构建只用于调试，统计会与全量不一致", flush=True)

    val_boundary, test_boundary = _user_boundaries(store, user_ids.size)
    sentinel = np.iinfo(np.int64).min
    user_val = val_boundary[user_index]
    user_test = test_boundary[user_index]
    ts = raw["timestamps"]
    is_validation = (user_val != sentinel) & (ts == user_val)
    is_test = (user_test != sentinel) & (ts == user_test)
    before_validation = (user_val != sentinel) & (ts < user_val)
    no_holdout = user_val == sentinel
    split_codes = np.full(raw["rows"], -1, dtype=np.int8)  # -1 = 丢弃
    split_codes[before_validation | no_holdout] = 0
    split_codes[is_validation] = 1
    split_codes[is_test] = 2

    print("[3/4] 按请求聚合候选 …", flush=True)
    # 按 (user, 时间戳) 排序，保证同一请求的候选连续
    order = np.lexsort((ts, user_index))
    u = user_index[order]
    i = item_index[order]
    t = ts[order]
    c = raw["clicks"][order]
    codes = split_codes[order]
    new_request = np.ones(u.size, dtype=bool)
    if u.size > 1:
        new_request[1:] = (u[1:] != u[:-1]) | (t[1:] != t[:-1])
    request_start = np.flatnonzero(new_request)
    request_code = codes[request_start]
    request_sizes = np.diff(np.append(request_start, u.size)).astype(np.int32)

    arrays: dict[str, dict[str, np.ndarray]] = {}
    counts: dict[str, int] = {}
    for name, code in (("train", 0), ("validation", 1), ("test", 2)):
        keep_request = request_code == code
        row_mask = np.repeat(keep_request, request_sizes)
        group_sizes = request_sizes[keep_request]
        arrays[name] = {
            "items": i[row_mask],
            "labels": c[row_mask],
            "users": u[row_mask],
            "groups": group_sizes,
        }
        counts[name] = int(row_mask.sum())
        if counts[name] and int(group_sizes.sum()) != counts[name]:
            raise AssertionError("候选分组与行数不一致")
    dropped = int((split_codes == -1).sum())
    assigned = int(sum(counts.values()))
    if assigned + dropped != raw["rows"]:
        raise AssertionError(
            f"候选行数对不上：train+val+test={assigned:,} + dropped={dropped:,} "
            f"≠ rows={raw['rows']:,}"
        )

    print("[4/4] 保存 …", flush=True)
    target.mkdir(parents=True, exist_ok=True)
    fingerprints: dict[str, str] = {}
    for name, payload in arrays.items():
        for column, values in payload.items():
            path = target / f"{name}_{column}.npy"
            np.save(path, values)
            fingerprints[f"{name}_{column}"] = _file_sha256(path)

    user_levels = _load_user_levels(dataset_root, user_ids)
    if user_levels is not None:
        path = target / "user_levels.npy"
        np.save(path, user_levels)
        fingerprints["user_levels"] = _file_sha256(path)

    stats = {
        "rows_read": raw["rows"],
        "candidates": counts,
        "click_rates": {
            name: round(float(arrays[name]["labels"].mean()), 6) if counts[name] else 0.0
            for name in arrays
        },
        "requests": {name: int(arrays[name]["groups"].size) for name in arrays},
        "candidates_per_request": {
            name: (
                {
                    "mean": round(float(arrays[name]["groups"].mean()), 2),
                    "median": float(np.median(arrays[name]["groups"])),
                    "max": int(arrays[name]["groups"].max()),
                }
                if counts[name]
                else {}
            )
            for name in arrays
        },
        "dropped_between_splits": dropped,
        "item_feature_columns": list(ITEM_FEATURE_COLUMNS),
        "user_feature_columns": list(USER_FEATURE_COLUMNS) if user_levels is not None else [],
        "sampled": max_rows is not None,
        "elapsed_s": round(time.time() - started, 1),
    }
    schema = {
        "split_unit": "request=(user_id, timestamp)",
        "positive_field": POSITIVE_FIELD,
        "stats": stats,
        "fingerprints": fingerprints,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (target / "schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] 写入 {target}", flush=True)
    return schema


def verify_ranking_store(dataset_root: Path, schema: dict[str, Any]) -> list[str]:
    base = ranking_dir(dataset_root)
    problems: list[str] = []
    for name, expected in (schema.get("fingerprints") or {}).items():
        path = base / f"{name}.npy"
        if not path.exists():
            problems.append(f"缺少 {name}.npy")
        elif _file_sha256(path) != expected:
            problems.append(f"{name}.npy 哈希不一致")
    return problems


def load_ranking_samples(
    dataset_root: Path | None = None, split: str = "train", *, verify: bool = True
) -> dict[str, np.ndarray]:
    """读取某个切分的候选样本（mmap）。"""
    dataset_root = dataset_root or kuasar_dataset_dir(DEFAULT_DATA_DIR)
    base = ranking_dir(dataset_root)
    schema_path = base / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"{base} 不存在；先运行 python -m recsys.data.kuasar_ranking。"
        )
    if split not in {"train", "validation", "test"}:
        raise ValueError("split 必须是 train / validation / test")
    if verify:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        problems = verify_ranking_store(dataset_root, schema)
        if problems:
            raise ValueError(
                f"排序样本校验失败（{len(problems)} 项，前 3 项：{problems[:3]}）；"
                "重新运行 python -m recsys.data.kuasar_ranking --force。"
            )
    return {
        "items": np.load(base / f"{split}_items.npy", mmap_mode="r"),
        "labels": np.load(base / f"{split}_labels.npy", mmap_mode="r"),
        "users": np.load(base / f"{split}_users.npy", mmap_mode="r"),
        "groups": np.load(base / f"{split}_groups.npy", mmap_mode="r"),
    }


def iter_candidate_batches(
    samples: dict[str, np.ndarray],
    item_features: Any,
    *,
    item_columns: tuple[str, ...] = ITEM_FEATURE_COLUMNS,
    user_levels: np.ndarray | None = None,
    max_requests: int = 20_000,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """按请求分批产出 ``(X, y, groups)``，供 LightGBM 等逐批消费。

    全量矩阵是 ``候选数 × 特征数 × 4B``（14M × 9 ≈ 500 MB），不做整表物化。
    ``groups`` 是这批请求各自的候选数，直接作为 LightGBM 的 group 参数。
    """
    groups = np.asarray(samples["groups"])
    items = samples["items"]  # 通常是 mmap，按行切片不会整表加载
    labels = samples["labels"]
    users = samples["users"]
    # 行偏移只算一次：row_start_of_request[i] 是第 i 个请求的起始行
    starts = np.concatenate(([0], np.cumsum(groups)))
    request_start = 0
    while request_start < groups.size:
        request_end = min(request_start + max_requests, groups.size)
        row_start = int(starts[request_start])
        row_end = int(starts[request_end])
        batch_items = np.asarray(items[row_start:row_end])
        columns = [
            item_features.column(name)[batch_items].astype(np.float32)
            for name in item_columns
        ]
        X = np.stack(columns, axis=1)
        if user_levels is not None:
            extra = user_levels[np.asarray(users[row_start:row_end])].astype(np.float32)
            X = np.concatenate([X, extra], axis=1)
        group_slice = groups[request_start:request_end].astype(np.int32)
        yield X, np.asarray(labels[row_start:row_end]).astype(np.int32), group_slice
        request_start = request_end


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-rows", type=int, default=None, help="抽样行数（调试用）")
    parser.add_argument("--force", action="store_true", help="忽略已有缓存，强制重建")
    parser.add_argument("--check", action="store_true", help="只校验指纹，不重建")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = kuasar_dataset_dir(args.data_dir)
    if args.check:
        schema_path = ranking_dir(dataset_root) / "schema.json"
        if not schema_path.exists():
            print("[error] 排序样本不存在")
            return 2
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        problems = verify_ranking_store(dataset_root, schema)
        if problems:
            print(f"[fail] {len(problems)} 项不一致：{problems[:5]}")
            return 1
        print(f"[ok] 排序样本有效：{schema['stats']['candidates']}")
        return 0

    schema = build_ranking_samples(
        args.data_dir, max_rows=args.max_rows, force=args.force
    )
    print(json.dumps(schema["stats"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
