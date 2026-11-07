"""从 KuaiSAR 全量 processed 里切一个可控子集（Phase 2 小规模验证用）。

思路：**按用户抽子集，不破坏请求与曝光结构**。

* 选一批「有留出请求、正样本数 ≥ N」的用户；
* 物品词表 = 这些用户交互过的物品（自动落到几十万级，CPU 可训练）；
* 曝光 CSR 按用户切片，只保留词表内物品并重新编码；
* 切分标签沿用全量的请求级结果，不重新定义口径。

产物与主 store 同构（`ids/`、`split/`、`exposure/` + `protocol.json`），
因此 `prepare_dataset("kuasar_small")` 直接可用。

用法：

    python -m recsys.data.kuasar_subset --users 2000 --min-positives 30
    python -m recsys.data.kuasar_subset --users 500 --output data/kuasar_tiny
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from recsys.config import DEFAULT_DATA_DIR
from recsys.data.kuasar import KUASAR_ARCHIVE_MD5, KUASAR_DATASET_DIRNAME
from recsys.data.kuasar_features import COLUMNS, features_dir, _file_sha256
from recsys.data.kuasar_store import _save, dataset_dir as kuasar_dataset_dir, load_processed

DEFAULT_OUTPUT = Path("data") / "kuasar_small"


def _slice_exposure(
    csr: dict[str, np.ndarray],
    chosen_users: np.ndarray,
    item_map: np.ndarray,
    num_chosen: int,
) -> dict[str, np.ndarray]:
    """按选中的用户切片 CSR，丢掉词表外物品并重新编码。"""
    indptr = np.asarray(csr["indptr"])
    items = np.asarray(csr["items"])
    counts = np.zeros(num_chosen, dtype=np.int64)
    kept_items: list[np.ndarray] = []
    for new_user, old_user in enumerate(chosen_users):
        start, end = int(indptr[old_user]), int(indptr[old_user + 1])
        block = items[start:end]
        mapped = item_map[block]
        valid = mapped >= 0
        block = mapped[valid]
        counts[new_user] = block.size
        kept_items.append(block)
    new_indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    flat = np.concatenate(kept_items) if kept_items else np.empty(0, dtype=np.int32)
    return {"indptr": new_indptr, "items": flat.astype(np.int32)}


def slice_item_features(
    parent_root: Path, target_root: Path, item_positions: np.ndarray
) -> dict[str, Any] | None:
    """把全量特征表按子集物品切片；父数据集没有特征表时返回 None。"""
    source = features_dir(parent_root) / "schema.json"
    if not source.exists():
        return None
    schema = json.loads(source.read_text(encoding="utf-8"))
    target = features_dir(target_root)
    target.mkdir(parents=True, exist_ok=True)
    fingerprints: dict[str, str] = {}
    for name in COLUMNS:
        values = np.asarray(np.load(features_dir(parent_root) / f"{name}.npy", mmap_mode="r"))
        sliced = values[item_positions]
        path = target / f"{name}.npy"
        np.save(path, sliced)
        fingerprints[name] = _file_sha256(path)
    sliced_schema = {
        **schema,
        "num_items": int(item_positions.size),
        "matched_items": int(item_positions.size),
        "is_ad_items": int(np.asarray(np.load(target / "is_ad.npy")).sum()),
        "fingerprints": fingerprints,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (target / "schema.json").write_text(
        json.dumps(sliced_schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"      特征表已切片：{len(COLUMNS)} 列 → {target}", flush=True)
    return sliced_schema


def build_subset(
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    output: Path = DEFAULT_OUTPUT,
    users: int = 2000,
    min_positives: int = 30,
    seed: int = 42,
    force: bool = False,
) -> dict[str, Any]:
    output = Path(output)
    if (output / "processed" / "protocol.json").exists() and not force:
        print(f"[reuse] {output} 已存在；加 --force 重建")
        return json.loads((output / "processed" / "protocol.json").read_text(encoding="utf-8"))

    started = time.time()
    store = load_processed(kuasar_dataset_dir(data_dir))
    user_ids = np.asarray(store["user_ids"])
    item_ids = np.asarray(store["item_ids"])
    split = store["split"]

    print("[1/4] 选择用户 …", flush=True)
    concat_users = np.concatenate([np.asarray(split[name]["users"]) for name in ("train", "validation", "test")])
    positive_counts = np.bincount(concat_users, minlength=user_ids.size)
    holdout = np.unique(
        np.concatenate([np.asarray(split["validation"]["users"]), np.asarray(split["test"]["users"])])
    )
    eligible = holdout[positive_counts[holdout] >= min_positives]
    if eligible.size == 0:
        raise ValueError(f"没有用户满足 min_positives={min_positives}")
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(eligible, size=min(users, eligible.size), replace=False))
    print(f"      符合条件 {eligible.size:,}，抽样 {chosen.size:,}", flush=True)

    print("[2/4] 抽取正样本并重建词表 …", flush=True)
    user_map = np.full(user_ids.size, -1, dtype=np.int32)
    user_map[chosen] = np.arange(chosen.size, dtype=np.int32)

    kept_users: list[np.ndarray] = []
    kept_items: list[np.ndarray] = []
    kept_ts: list[np.ndarray] = []
    kept_labels: list[np.ndarray] = []
    for code, name in ((0, "train"), (1, "validation"), (2, "test")):
        rows = np.asarray(split[name]["users"])
        mask = user_map[rows] >= 0
        kept_users.append(user_map[rows[mask]])
        kept_items.append(np.asarray(split[name]["items"])[mask])
        kept_ts.append(np.asarray(split[name]["timestamps"])[mask])
        kept_labels.append(np.full(int(mask.sum()), code, dtype=np.int8))
        print(f"      {name}: {int(mask.sum()):,} 行", flush=True)

    all_users = np.concatenate(kept_users)
    all_items = np.concatenate(kept_items)
    all_ts = np.concatenate(kept_ts)
    all_labels = np.concatenate(kept_labels)

    # 物品词表 = 子集用户交互过的物品；按原始 ID 排序保证可复现
    unique_items = np.unique(all_items)
    item_map = np.full(item_ids.size, -1, dtype=np.int32)
    item_map[unique_items] = np.arange(unique_items.size, dtype=np.int32)
    remapped_items = item_map[all_items]
    if (remapped_items < 0).any():
        raise AssertionError("物品重编码失败")

    print("[3/4] 切片曝光历史 …", flush=True)
    exposure = {
        name: _slice_exposure(store["exposure"][name], chosen, item_map, chosen.size)
        for name in ("validation", "test")
    }

    print("[4/4] 保存 …", flush=True)
    processed = output / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    mapped = {"user_ids": user_ids[chosen], "item_ids": item_ids[unique_items]}
    split_data = {
        "users": all_users,
        "items": remapped_items,
        "timestamps": all_ts,
        "split": all_labels,
    }
    fingerprints = _save(processed, mapped, split_data, exposure)
    # 物品特征按子集词表切片，否则子集上的排序/特征实验没有特征可用
    feature_schema = slice_item_features(
        kuasar_dataset_dir(data_dir), output, unique_items
    )

    split_counts = {
        name: int((all_labels == code).sum()) for code, name in ((0, "train"), (1, "validation"), (2, "test"))
    }
    stats = {
        "parent_dataset": "kuasar",
        "users": int(chosen.size),
        "items": int(unique_items.size),
        "positive_pairs": int(all_users.size),
        "split_counts": split_counts,
        "requests": {name: int(np.unique(all_ts[all_labels == code]).size) for code, name in ((0, "train"), (1, "validation"), (2, "test"))},
        "exposure_pairs": {name: int(csr["items"].size) for name, csr in exposure.items()},
        "item_features": (
            {"columns": len(feature_schema["columns"]), "ad_items": feature_schema.get("is_ad_items")}
            if feature_schema
            else None
        ),
        "min_positives": min_positives,
        "seed": seed,
        "elapsed_s": round(time.time() - started, 1),
    }
    protocol = {
        "dataset": "kuasar_small",
        "parent_dataset": "kuasar",
        "parent_archive_md5": KUASAR_ARCHIVE_MD5,
        "split_strategy": "per_user_last_two_requests",
        "split_unit": "request=(user_id, timestamp)",
        "subset": {"users": int(chosen.size), "min_positives": min_positives, "seed": seed},
        "stats": stats,
        "fingerprints": fingerprints,
        "data_version": fingerprints["data_version"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (processed / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (processed / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] 写入 {processed}（{stats['users']:,} 用户 / {stats['items']:,} 物品 / "
          f"{stats['positive_pairs']:,} 正样本）", flush=True)
    return protocol


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--users", type=int, default=2000)
    parser.add_argument("--min-positives", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = build_subset(
        args.data_dir,
        output=args.output,
        users=args.users,
        min_positives=args.min_positives,
        seed=args.seed,
        force=args.force,
    )
    print(json.dumps(protocol["stats"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
