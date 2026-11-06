"""KuaiSAR → processed npy 存储（Phase 1.3）。

为什么不是直接产出 `PreparedData`：

* 稠密 `seen_mask` 在 KuaiSAR 上需要 25,877 × 4,046,367 bool ≈ 97.5 GiB，不可行；
* 724 万条正样本若用 `list[Interaction]` 表示约 1.2 GB，训练时无法与模型共存。

因此这里落一份**数组化**的中间层：每个切分一组 npy（user/item/timestamp），
曝光历史用按用户的 CSR（indptr + items）表示，ID 映射也存 npy。
下游（1.4 样本构建 / 1.6 缓存）用 `mmap` 读，内存开销接近 0。

产出（`data/kuasar/processed/`）：

    ids/user_ids.npy, ids/item_ids.npy        原始 ID（index -> original）
    split/{train,validation,test}_{users,items,timestamps}.npy
    exposure/{validation,test}_{indptr,items}.npy
    stats.json                                规模与分布
    protocol.json                             口径、指纹、可复现参数

用法：

    python -m recsys.data.kuasar_store            # 全量构建
    python -m recsys.data.kuasar_store --max-rows 500000   # 抽样试跑
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from recsys.config import DEFAULT_DATA_DIR
from recsys.data.kuasar import KUASAR_ARCHIVE_MD5, KUASAR_DATASET_DIRNAME
from recsys.data.schema import (
    ArrayInteractions,
    IdMappings,
    ItemFeatures,
    PreparedData,
    SparseExposure,
    SortedIndexLookup,
)

REC_INTER = "KuaiSAR_v2/rec_inter.csv"
POSITIVE_FIELD = "click"
DEDUPE_POLICY = "by_user_item_keep_first_click"
EXPOSURE_POLICY = "all_rows_before_target_timestamp"

#: 按这个大小成批解析，避免一次性吃掉整表。
_READ_CHUNK = 1_000_000


def dataset_dir(data_dir: Path) -> Path:
    """`data/kuasar`（与 KuaiRec 的 `data/kuairec` 同构）。"""
    return data_dir / KUASAR_DATASET_DIRNAME


def processed_dir(dataset_root: Path) -> Path:
    """`dataset_root` 是 `data/kuasar` 这一层，不是 `data`。"""
    return dataset_root / "processed"


def _rec_inter_path(dataset_root: Path) -> Path:
    return dataset_root / "raw" / REC_INTER


def _read_columns(path: Path, max_rows: int | None = None) -> dict[str, np.ndarray]:
    """流式读取需要的列，分批转成 numpy 再拼接。"""
    file = path.open("r", encoding="utf-8-sig", newline="")
    reader = csv.reader(file)
    header = next(reader)
    index = {name: position for position, name in enumerate(header)}
    for required in ("user_id", "item_id", "timestamp", POSITIVE_FIELD):
        if required not in index:
            raise ValueError(f"{path} 缺少必需列 {required}")

    user_pos, item_pos, ts_pos, click_pos = (
        index["user_id"],
        index["item_id"],
        index["timestamp"],
        index[POSITIVE_FIELD],
    )
    user_parts: list[np.ndarray] = []
    item_parts: list[np.ndarray] = []
    ts_parts: list[np.ndarray] = []
    click_parts: list[np.ndarray] = []
    buffer_users: list[int] = []
    buffer_items: list[int] = []
    buffer_ts: list[float] = []
    buffer_click: list[int] = []
    rows = 0
    next_report = 2_000_000

    def flush() -> None:
        if not buffer_users:
            return
        user_parts.append(np.asarray(buffer_users, dtype=np.int32))
        item_parts.append(np.asarray(buffer_items, dtype=np.int32))
        ts_parts.append(np.asarray(buffer_ts, dtype=np.int64))
        click_parts.append(np.asarray(buffer_click, dtype=np.int8))
        buffer_users.clear()
        buffer_items.clear()
        buffer_ts.clear()
        buffer_click.clear()

    try:
        for row in reader:
            rows += 1
            if max_rows is not None and rows > max_rows:
                rows -= 1
                break
            buffer_users.append(int(row[user_pos]))
            buffer_items.append(int(row[item_pos]))
            buffer_ts.append(int(float(row[ts_pos])))
            buffer_click.append(1 if int(float(row[click_pos])) else 0)
            if len(buffer_users) >= _READ_CHUNK:
                flush()
            if rows >= next_report:
                print(f"  parsed {rows:,} rows", flush=True)
                next_report += 2_000_000
    finally:
        file.close()
    flush()
    return {
        "users": np.concatenate(user_parts),
        "items": np.concatenate(item_parts),
        "timestamps": np.concatenate(ts_parts),
        "clicks": np.concatenate(click_parts),
        "rows": rows,
    }


def _map_ids(users: np.ndarray, items: np.ndarray) -> dict[str, Any]:
    """原始 ID → 连续索引；返回索引数组与可逆映射。"""
    unique_users, user_index = np.unique(users, return_inverse=True)
    unique_items, item_index = np.unique(items, return_inverse=True)
    return {
        "user_ids": unique_users.astype(np.int64),
        "item_ids": unique_items.astype(np.int64),
        "user_index": user_index.astype(np.int32),
        "item_index": item_index.astype(np.int32),
    }


def _split_positives(
    user_index: np.ndarray,
    item_index: np.ndarray,
    timestamps: np.ndarray,
    clicks: np.ndarray,
    num_users: int,
) -> dict[str, Any]:
    """按「请求」为单位切分：最后一个请求做测试，倒数第二个做验证。

    为什么不是按单条点击切：KuaiSAR 的 `timestamp` 是按请求记录的，同一次推荐请求里
    点开的多个视频共享同一个毫秒时间戳。实测 25,877 个用户里只有 3,752 个的
    「最后三次点击」时间戳严格递增——按点击切会让 85% 的用户失去评测资格，
    而且把同一请求内的物品拆到 train/test 会泄漏（它们同时发生）。

    口径：

    * 正样本先按 `(user, item)` 去重、保留首次点击；
    * 请求 = `(user, timestamp)`；用户有 >= 3 个请求才参与评测；
    * `train` = 早于验证请求的全部正样本；`validation` = 倒数第二个请求；
      `test` = 最后一个请求；介于验证与测试之间的请求丢弃（数量很小），
      保证 train < validation < test 的时间边界干净，不把验证之后的信息带进验证。
    """
    positive_mask = clicks == 1
    p_users = user_index[positive_mask]
    p_items = item_index[positive_mask]
    p_ts = timestamps[positive_mask]

    # 1) 去重：同一 (user, item) 只保留最早一次点击
    order = np.lexsort((p_ts, p_items, p_users))
    p_users, p_items, p_ts = p_users[order], p_items[order], p_ts[order]
    pair_key = p_users.astype(np.int64) * np.int64(int(item_index.max()) + 1) + p_items
    _, first_positions = np.unique(pair_key, return_index=True)
    p_users, p_items, p_ts = p_users[first_positions], p_items[first_positions], p_ts[first_positions]

    # 2) 排序后取出「每个请求」的首行，即 (user, ts) 去重
    order = np.lexsort((p_ts, p_users))
    p_users, p_items, p_ts = p_users[order], p_items[order], p_ts[order]
    request_start = np.ones(p_users.size, dtype=bool)
    request_start[1:] = (p_users[1:] != p_users[:-1]) | (p_ts[1:] != p_ts[:-1])
    request_rows = np.flatnonzero(request_start)
    request_users = p_users[request_rows]
    request_ts = p_ts[request_rows]

    request_count = np.bincount(request_users, minlength=num_users)
    last_request = np.concatenate(([0], np.cumsum(request_count)[:-1])) + request_count - 1
    has_holdout = request_count >= 3

    # 每个用户的验证/测试请求时间戳；没有留出的用户用最小值哨兵，比较天然为 False
    sentinel = np.iinfo(np.int64).min
    val_boundary = np.full(num_users, sentinel, dtype=np.int64)
    test_boundary = np.full(num_users, sentinel, dtype=np.int64)
    selected_users = np.flatnonzero(has_holdout)
    val_boundary[selected_users] = request_ts[last_request[selected_users] - 1]
    test_boundary[selected_users] = request_ts[last_request[selected_users]]

    split = np.zeros(p_users.size, dtype=np.int8)  # 0 train / 1 validation / 2 test / 3 dropped
    user_val = val_boundary[p_users]
    user_test = test_boundary[p_users]
    is_validation = (user_val != sentinel) & (p_ts == user_val)
    is_test = (user_test != sentinel) & (p_ts == user_test)
    before_validation = (user_val != sentinel) & (p_ts < user_val)
    no_holdout = ~has_holdout[p_users]
    split[is_validation] = 1
    split[is_test] = 2
    split[(before_validation | no_holdout) & ~is_validation & ~is_test] = 0
    split[(~is_validation) & (~is_test) & (~before_validation) & (~no_holdout)] = 3

    return {
        "users": p_users,
        "items": p_items,
        "timestamps": p_ts,
        "split": split,
        "num_users_with_holdout": int(has_holdout.sum()),
        "num_users_insufficient": int((~has_holdout).sum()),
        "num_users_selected": int(request_count[has_holdout].sum()),
        "requests_per_user_median": float(np.median(request_count[request_count > 0])),
        "rows_dropped_between_splits": int((split == 3).sum()),
        "val_boundary": val_boundary,
        "test_boundary": test_boundary,
    }


def _exposure_csr(
    sorted_users: np.ndarray,
    sorted_items: np.ndarray,
    sorted_ts: np.ndarray,
    boundary: np.ndarray,
    num_users: int,
) -> dict[str, np.ndarray]:
    """按用户聚合「目标时刻之前真实曝光过的物品」（含 click=0）。

    入参是**已经按用户排好序**的三列，两个切分共享同一次排序。
    """
    keep = sorted_ts < boundary[sorted_users]
    kept_users = sorted_users[keep]
    kept_items = sorted_items[keep]
    counts = np.bincount(kept_users, minlength=num_users)
    indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return {"indptr": indptr, "items": kept_items.astype(np.int32)}


def _file_sha256(path: Path) -> str:
    """文件级 sha256（分块读，不把 55 MB 数组整体载入内存）。

    用文件哈希而不是数组字节哈希：校验时不必反序列化，1 秒内能检查完 250 MB。
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(
    processed: Path,
    mapped: dict[str, Any],
    split_data: dict[str, Any],
    exposure: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    ids_dir = processed / "ids"
    split_dir = processed / "split"
    exposure_dir = processed / "exposure"
    for directory in (ids_dir, split_dir, exposure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    def save(array: np.ndarray, path: Path, key: str, fingerprints: dict[str, str]) -> None:
        np.save(path, array)
        fingerprints[key] = _file_sha256(path)

    fingerprints: dict[str, str] = {}
    save(mapped["user_ids"], ids_dir / "user_ids.npy", "ids/user_ids", fingerprints)
    save(mapped["item_ids"], ids_dir / "item_ids.npy", "ids/item_ids", fingerprints)

    split = split_data["split"]
    for name, code in (("train", 0), ("validation", 1), ("test", 2)):
        mask = split == code
        save(split_data["users"][mask], split_dir / f"{name}_users.npy", f"split/{name}_users", fingerprints)
        save(split_data["items"][mask], split_dir / f"{name}_items.npy", f"split/{name}_items", fingerprints)
        save(
            split_data["timestamps"][mask],
            split_dir / f"{name}_timestamps.npy",
            f"split/{name}_timestamps",
            fingerprints,
        )

    for name, csr in exposure.items():
        save(csr["indptr"], exposure_dir / f"{name}_indptr.npy", f"exposure/{name}_indptr", fingerprints)
        save(csr["items"], exposure_dir / f"{name}_items.npy", f"exposure/{name}_items", fingerprints)

    return {"arrays": fingerprints, "data_version": _data_version(fingerprints)}


def _data_version(fingerprints: dict[str, str]) -> str:
    """把「所有文件哈希 + 排序后的键名」压成一个短版本号，便于人工对比。"""
    digest = hashlib.sha256()
    for key in sorted(fingerprints):
        digest.update(f"{key}={fingerprints[key]}\n".encode("utf-8"))
    return digest.hexdigest()[:16]


def verify_store(processed: Path, fingerprints: dict[str, str]) -> list[str]:
    """逐个文件核对 sha256；返回不一致清单（空表示缓存有效）。"""
    problems: list[str] = []
    for key in sorted(fingerprints):
        path = processed / f"{key}.npy"
        if not path.exists():
            problems.append(f"缺少文件 {key}.npy")
            continue
        actual = _file_sha256(path)
        if actual != fingerprints[key]:
            problems.append(f"{key}.npy 哈希不一致")
    return problems


def build_processed_store(
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    max_rows: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """`data_dir` 是仓库根下的 `data/`（里面是 `kuasar/`）。

    已存在且指纹校验通过时直接复用（除非 `force=True`）；不一致则重建并给出提示。
    """
    dataset_root = dataset_dir(data_dir)
    target = processed_dir(dataset_root)
    if target.exists() and not force:
        existing = _read_protocol(target)
        if existing is not None:
            problems = verify_store(target, _fingerprints_of(existing))
            if not problems:
                print(f"[reuse] {target} 指纹校验通过（data_version={existing.get('data_version')}）")
                return {
                    "stats": existing.get("stats", {}),
                    "protocol": existing,
                    "output_dir": target,
                    "reused": True,
                }
            print(f"[stale] {target} 校验失败，重建：{problems[:3]}", flush=True)

    source = _rec_inter_path(dataset_root)
    if not source.exists():
        raise FileNotFoundError(
            f"找不到 {source}；先运行 python -m recsys.data.kuasar 下载并解压。"
        )
    started = time.time()
    print(f"[1/5] 解析 {source.name} …", flush=True)
    columns = _read_columns(source, max_rows=max_rows)
    print(f"      {columns['rows']:,} 行", flush=True)

    print("[2/5] 生成 ID 映射 …", flush=True)
    mapped = _map_ids(columns["users"], columns["items"])
    num_users = int(mapped["user_ids"].size)
    num_items = int(mapped["item_ids"].size)
    print(f"      users={num_users:,} items={num_items:,}", flush=True)

    print("[3/5] 去重正样本并按请求切分 …", flush=True)
    positives = _split_positives(
        mapped["user_index"], mapped["item_index"], columns["timestamps"], columns["clicks"], num_users
    )
    val_boundary = positives["val_boundary"]
    test_boundary = positives["test_boundary"]

    print("[4/5] 构造稀疏曝光历史 …", flush=True)
    order = np.argsort(mapped["user_index"], kind="stable")
    sorted_users = mapped["user_index"][order]
    sorted_items = mapped["item_index"][order]
    sorted_ts = columns["timestamps"][order]
    exposure = {
        "validation": _exposure_csr(sorted_users, sorted_items, sorted_ts, val_boundary, num_users),
        "test": _exposure_csr(sorted_users, sorted_items, sorted_ts, test_boundary, num_users),
    }

    print("[5/5] 保存 npy 与元信息 …", flush=True)
    fingerprints = _save(target, mapped, positives, exposure)

    split_counts = {
        name: int((positives["split"] == code).sum())
        for name, code in (("train", 0), ("validation", 1), ("test", 2))
    }
    stats = {
        "rows_read": columns["rows"],
        "num_users": num_users,
        "num_items": num_items,
        "num_positive_pairs": int(positives["users"].size),
        "positive_rows": int(columns["clicks"].sum()),
        "split_unit": "request=(user_id, timestamp)",
        "split_counts": split_counts,
        "users_with_holdout": positives["num_users_with_holdout"],
        "users_insufficient_requests": positives["num_users_insufficient"],
        "requests_per_user_median": positives["requests_per_user_median"],
        "rows_dropped_between_splits": positives["rows_dropped_between_splits"],
        "exposure_pairs": {
            name: int(csr["items"].size) for name, csr in exposure.items()
        },
        "sampled": max_rows is not None,
        "elapsed_s": round(time.time() - started, 1),
    }
    protocol = {
        "dataset": "kuasar",
        "dataset_version": "KuaiSAR_v2",
        "archive_md5": KUASAR_ARCHIVE_MD5,
        "split_strategy": "per_user_last_two_requests",
        "split_unit": "request=(user_id, timestamp)",
        "positive_field": POSITIVE_FIELD,
        "dedupe_policy": DEDUPE_POLICY,
        "exposure_policy": EXPOSURE_POLICY,
        "search_side": "excluded",
        "ad_items": "kept",
        "stats": stats,
        "fingerprints": fingerprints,
        "data_version": fingerprints["data_version"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (target / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (target / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] 写入 {target}（data_version={protocol['data_version']}）", flush=True)
    return {"stats": stats, "protocol": protocol, "output_dir": target, "reused": False}


def _read_protocol(base: Path) -> dict[str, Any] | None:
    path = base / "protocol.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _fingerprints_of(protocol: dict[str, Any]) -> dict[str, str]:
    """兼容两种格式：新格式 `fingerprints.arrays`，旧格式直接是 dict。"""
    payload = protocol.get("fingerprints") or {}
    return payload.get("arrays", payload) if isinstance(payload, dict) else {}


def load_processed(
    dataset_root: Path | None = None, mmap: bool = True, *, verify: bool = True
) -> dict[str, Any]:
    """读取 processed 存储；默认 mmap，常驻内存接近 0。

    默认会按 `protocol.json` 里的 sha256 校验每个 npy（约 1–2 秒），
    防止「代码改了、缓存还是旧的」这类静默错误；`verify=False` 可跳过。
    """
    dataset_root = dataset_root or dataset_dir(DEFAULT_DATA_DIR)
    base = processed_dir(dataset_root)
    protocol = _read_protocol(base)
    if protocol is None:
        raise FileNotFoundError(
            f"{base} 还没有构建；先运行 python -m recsys.data.kuasar_store。"
        )
    if verify:
        problems = verify_store(base, _fingerprints_of(protocol))
        if problems:
            raise ValueError(
                f"processed 缓存校验失败（{len(problems)} 项，前 3 项：{problems[:3]}）；"
                "重新运行 python -m recsys.data.kuasar_store --force。"
            )
    mode = "r" if mmap else None
    return {
        "protocol": protocol,
        "stats": json.loads((base / "stats.json").read_text(encoding="utf-8")),
        "user_ids": np.load(base / "ids" / "user_ids.npy", mmap_mode=mode),
        "item_ids": np.load(base / "ids" / "item_ids.npy", mmap_mode=mode),
        "split": {
            name: {
                "users": np.load(base / "split" / f"{name}_users.npy", mmap_mode=mode),
                "items": np.load(base / "split" / f"{name}_items.npy", mmap_mode=mode),
                "timestamps": np.load(base / "split" / f"{name}_timestamps.npy", mmap_mode=mode),
            }
            for name in ("train", "validation", "test")
        },
        "exposure": {
            name: {
                "indptr": np.load(base / "exposure" / f"{name}_indptr.npy", mmap_mode=mode),
                "items": np.load(base / "exposure" / f"{name}_items.npy", mmap_mode=mode),
            }
            for name in ("validation", "test")
        },
    }


def _load_item_features(dataset_root: Path, num_items: int) -> ItemFeatures | None:
    """特征表是可选的：没构建就返回 None，构建过但长度对不上则直接报错。"""
    from recsys.data.kuasar_features import load_item_features

    payload = load_item_features(dataset_root)
    if payload is None:
        return None
    features = ItemFeatures(arrays=payload["arrays"], schema=payload["schema"])
    if features.num_items != num_items:
        raise ValueError(
            f"特征表长度 {features.num_items:,} 与物品词表 {num_items:,} 不一致；"
            "请重新运行 python -m recsys.data.kuasar_features。"
        )
    return features


def prepare_kuasar(dataset_root: Path | None = None) -> PreparedData:
    """把 processed 存储包装成统一的 `PreparedData`（数组后端 + 稀疏曝光）。

    `dataset_root` 是 `data/kuasar`（loader 传进来的就是这一层）。

    与 KuaiRec 的差别：

    * 交互是 `ArrayInteractions`（mmap 视图），不是 `list[Interaction]`；
    * 曝光是 `SparseExposure`（CSR），不是稠密 mask；
    * `split_strategy` 取协议里的 `per_user_last_two_requests`。
    """
    store = load_processed(dataset_root)
    stats = store["stats"]
    split_data = store["split"]

    mappings = IdMappings(
        user_to_index=SortedIndexLookup(store["user_ids"]),
        index_to_user=store["user_ids"],
        item_to_index=SortedIndexLookup(store["item_ids"]),
        index_to_item=store["item_ids"],
    )

    def interactions(name: str) -> ArrayInteractions:
        payload = split_data[name]
        return ArrayInteractions(payload["users"], payload["items"], payload["timestamps"])

    item_features = _load_item_features(dataset_root, num_items=mappings.index_to_item.size)

    prepared = PreparedData(
        train_interactions=interactions("train"),
        validation_interactions=interactions("validation"),
        test_interactions=interactions("test"),
        mappings=mappings,
        validation_seen=SparseExposure(
            store["exposure"]["validation"]["indptr"], store["exposure"]["validation"]["items"]
        ),
        test_seen=SparseExposure(
            store["exposure"]["test"]["indptr"], store["exposure"]["test"]["items"]
        ),
        item_features=item_features,
        split_strategy=str(store["protocol"].get("split_strategy", "per_user_last_two_requests")),
    )

    # 与统计对账：切分文件被手工改动或缓存过期时，早失败好过静默评错。
    expected = stats["split_counts"]
    actual = {
        "train": len(prepared.train_interactions),
        "validation": len(prepared.validation_interactions),
        "test": len(prepared.test_interactions),
    }
    mismatched = {
        name: (expected.get(name), actual[name])
        for name in actual
        if expected.get(name) != actual[name]
    }
    if mismatched:
        raise ValueError(
            f"processed 存储与 stats.json 不一致（expected, actual）：{mismatched}"
        )
    return prepared


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-rows", type=int, default=None, help="抽样行数（调试用）")
    parser.add_argument("--force", action="store_true", help="忽略已有缓存，强制重建")
    parser.add_argument("--check", action="store_true", help="只校验缓存指纹，不重建")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        base = processed_dir(dataset_dir(args.data_dir))
        protocol = _read_protocol(base)
        if protocol is None:
            print(f"[error] {base} 不存在或缺少 protocol.json")
            return 2
        problems = verify_store(base, _fingerprints_of(protocol))
        if problems:
            print(f"[fail] {len(problems)} 项不一致：{problems[:5]}")
            return 1
        print(f"[ok] 缓存有效，data_version={protocol.get('data_version')}")
        return 0

    result = build_processed_store(args.data_dir, max_rows=args.max_rows, force=args.force)
    print(json.dumps(result["stats"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
