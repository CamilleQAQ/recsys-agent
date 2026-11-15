"""KuaiSAR 广告样本管线：按曝光请求构造广告候选与点击标签。

数据来源与口径（全部复用已有 Phase 1 工件，不重新扫原始 CSV）：

* 曝光与请求分组：``data/<dataset>/processed/ranking/{split}_*.npy``。
  KuaiSAR 的 ``(user_id, timestamp)`` 就是一次曝光请求，同一请求的候选连续排列，
  ``groups`` 是每个请求的候选数；标签是 ``click``（0/1）。
* 广告候选：候选物品在物品特征表里 ``is_ad == 1``（KuaiSAR ``item_type == AD``）。
  因此「一次请求的广告候选」= 该请求曝光物品里的广告物品。
* 用户/物品词表：``data/<dataset>/processed/ids/*.npy``；父数据集 ID 通过
  ``ranking.samples.map_parent_to_target`` 映射到当前（子集）词表。
* 切分：完全沿用父数据集 ``processed/ranking`` 的请求级切分，不重新定义口径。

真实转化标签核对：``audit_conversion_labels()`` 读取 ``rec_inter.csv`` 表头，
KuaiSAR 只有 click / like / follow / forward / play_time 等行为，**没有转化字段**。
``schema.json`` 里明确记录 ``conversion_label.available = false``；CVR 只在
``recsys.ads.simulated`` 的独立模拟样本上验证，不能当作真实业务效果。

用法：

    python -m recsys.ads.data                 # 构建（已存在且校验通过则复用）
    python -m recsys.ads.data --check         # 只校验
    python -m recsys.ads.data --force         # 强制重建
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from recsys.config import DEFAULT_DATA_DIR
from recsys.data.loader import resolve_dataset_data_dir
from recsys.ranking.samples import map_parent_to_target

AD_SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "test")
#: 可能表示转化的字段名（不区分大小写）。KuaiSAR 里一个都不存在，用于如实记录缺口。
CONVERSION_FIELD_CANDIDATES = (
    "conversion",
    "convert",
    "conversion_label",
    "purchase",
    "order",
    "pay",
    "payment",
    "buy",
    "cvr",
)


@dataclass(frozen=True)
class AdView:
    """某个切分的广告候选视图；每个请求的广告候选连续排列。"""

    dataset: str
    split: str
    users: np.ndarray
    items: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    timestamps: np.ndarray

    def __len__(self) -> int:
        return int(self.items.size)

    @property
    def requests(self) -> int:
        return int(self.groups.size)

    def starts(self) -> np.ndarray:
        return np.concatenate(([0], np.cumsum(self.groups)))

    def request_sizes(self) -> np.ndarray:
        return np.asarray(self.groups)


def ads_dir(dataset_root: Path) -> Path:
    return Path(dataset_root) / "processed" / "ads"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_parent_ranking(parent: Path, split: str) -> dict[str, np.ndarray]:
    timestamps = parent / f"{split}_timestamps.npy"
    if not timestamps.exists():
        raise FileNotFoundError(
            f"{timestamps} 不存在：旧排序工件没有请求时间戳，无法构造「请求时间之前」的序列历史。\n"
            "重建命令：python -m recsys.data.kuasar_ranking --force"
        )
    return {
        "users": np.load(parent / f"{split}_users.npy", mmap_mode="r"),
        "items": np.load(parent / f"{split}_items.npy", mmap_mode="r"),
        "labels": np.load(parent / f"{split}_labels.npy", mmap_mode="r"),
        "groups": np.asarray(np.load(parent / f"{split}_groups.npy", mmap_mode="r")),
        "timestamps": np.asarray(np.load(timestamps, mmap_mode="r")),
    }


def _build_split(
    split: str,
    parent: dict[str, np.ndarray],
    user_map: np.ndarray,
    item_map: np.ndarray,
    is_ad: np.ndarray,
    *,
    max_requests: int | None,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    groups = parent["groups"].astype(np.int64)
    starts = np.concatenate(([0], np.cumsum(groups)))
    row_request = np.repeat(np.arange(groups.size, dtype=np.int64), groups)

    raw_users = np.asarray(parent["users"])
    raw_items = np.asarray(parent["items"])
    mapped_users = user_map[raw_users]
    mapped_items = item_map[raw_items]
    # is_ad 是当前数据集词表下的数组：先按映射后的物品索引取，再与有效性掩码组合。
    ad_mask = np.zeros(raw_items.size, dtype=bool)
    valid_items = mapped_items >= 0
    ad_mask[valid_items] = is_ad[mapped_items[valid_items]] > 0
    keep_row = (mapped_users >= 0) & valid_items & ad_mask

    ad_counts = np.bincount(row_request[keep_row], minlength=groups.size)
    request_keep = ad_counts > 0
    # 父表同一用户/同一请求的时间戳必须只有一个；否则说明排序工件损坏。
    request_timestamps = parent["timestamps"]
    if request_timestamps.size != groups.size:
        raise AssertionError(
            f"{split}: 请求时间戳 {request_timestamps.size} 与请求数 {groups.size} 不一致"
        )

    chosen = np.flatnonzero(request_keep)
    sampled = False
    if max_requests is not None and chosen.size > max_requests:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(chosen, size=max_requests, replace=False))
        sampled = True
    selected_requests = np.zeros(groups.size, dtype=bool)
    selected_requests[chosen] = True
    keep_row = keep_row & selected_requests[row_request]

    rows = np.flatnonzero(keep_row)
    new_groups = np.bincount(row_request[rows], minlength=groups.size)[chosen].astype(np.int32)
    if int(new_groups.sum()) != rows.size:
        raise AssertionError(f"{split}: 广告候选分组与行数不一致")

    payload = {
        "users": np.asarray(mapped_users[rows], dtype=np.int32),
        "items": np.asarray(mapped_items[rows], dtype=np.int32),
        "labels": np.asarray(parent["labels"])[rows].astype(np.int8),
        "groups": new_groups,
        "timestamps": np.asarray(request_timestamps[chosen], dtype=np.int64),
    }
    if payload["labels"].size and not np.isin(payload["labels"], (0, 1)).all():
        raise AssertionError(f"{split}: 广告标签不是 0/1")
    if payload["users"].size != payload["items"].size or payload["users"].size != payload["labels"].size:
        raise AssertionError(f"{split}: 用户/物品/标签长度不一致")
    if payload["timestamps"].size != payload["groups"].size:
        raise AssertionError(f"{split}: 请求时间戳与请求数不一致")

    stats = {
        "parent_candidates": int(groups.sum()),
        "ad_candidates": int(rows.size),
        "requests_with_ads": int(chosen.size),
        "requests_with_2plus_ads": int((new_groups >= 2).sum()),
        "ads_per_request": {
            "mean": round(float(new_groups.mean()), 3) if new_groups.size else 0.0,
            "max": int(new_groups.max()) if new_groups.size else 0,
        },
        "clicks": int(payload["labels"].sum()),
        "click_rate": (
            round(float(payload["labels"].mean()), 6) if payload["labels"].size else 0.0
        ),
        "sampled": sampled,
        "seed": seed if sampled else None,
    }
    return payload, stats


def build_ad_store(
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    dataset: str = "kuasar_small",
    parent_dataset: str = "kuasar",
    max_train_requests: int | None = None,
    seed: int = 42,
    force: bool = False,
) -> dict[str, Any]:
    dataset_root = resolve_dataset_data_dir(dataset, Path(data_dir))
    parent_root = resolve_dataset_data_dir(parent_dataset, Path(data_dir))
    target = ads_dir(dataset_root)
    if (target / "schema.json").exists() and not force:
        schema = json.loads((target / "schema.json").read_text(encoding="utf-8"))
        problems = verify_ad_store(target, schema)
        if not problems:
            print(f"[reuse] {target} 广告样本校验通过")
            return schema
        print(f"[stale] 广告样本校验失败，重建：{problems[:3]}", flush=True)

    started = time.time()
    target_item_ids = np.load(dataset_root / "processed" / "ids" / "item_ids.npy")
    target_user_ids = np.load(dataset_root / "processed" / "ids" / "user_ids.npy")
    parent_item_ids = np.load(parent_root / "processed" / "ids" / "item_ids.npy")
    parent_user_ids = np.load(parent_root / "processed" / "ids" / "user_ids.npy")
    is_ad = np.asarray(np.load(dataset_root / "processed" / "features" / "is_ad.npy"))
    if is_ad.size != target_item_ids.size:
        raise ValueError("is_ad.npy 与 item_ids.npy 长度不一致；先重建物品特征表")
    item_map = map_parent_to_target(parent_item_ids, target_item_ids)
    user_map = map_parent_to_target(parent_user_ids, target_user_ids)
    print(
        f"[1/3] 词表映射：物品 {int((item_map >= 0).sum()):,}/{parent_item_ids.size:,}，"
        f"用户 {int((user_map >= 0).sum()):,}/{parent_user_ids.size:,}；"
        f"广告物品 {int(is_ad.sum()):,}",
        flush=True,
    )

    parent_ranking = parent_root / "processed" / "ranking"
    if not (parent_ranking / "train_groups.npy").exists():
        raise FileNotFoundError(
            f"{parent_ranking} 不存在；先运行 python -m recsys.data.kuasar_ranking"
        )

    arrays: dict[str, dict[str, np.ndarray]] = {}
    split_stats: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        print(f"[2/3] 构造 {split} 广告候选 …", flush=True)
        parent = _load_parent_ranking(parent_ranking, split)
        limit = max_train_requests if split == "train" else None
        payload, stats = _build_split(
            split,
            parent,
            user_map,
            item_map,
            is_ad,
            max_requests=limit,
            seed=seed,
        )
        arrays[split] = payload
        split_stats[split] = stats
        print(
            f"      {stats['ad_candidates']:,} 广告候选 / {stats['requests_with_ads']:,} 请求 / "
            f"CTR {stats['click_rate']:.4f} / 多广告请求 {stats['requests_with_2plus_ads']:,}",
            flush=True,
        )

    print("[3/3] 保存 …", flush=True)
    target.mkdir(parents=True, exist_ok=True)
    fingerprints: dict[str, str] = {}
    for split, payload in arrays.items():
        for column, values in payload.items():
            path = target / f"{split}_{column}.npy"
            np.save(path, values)
            fingerprints[f"{split}_{column}"] = _sha256(path)

    schema = {
        "schema_version": AD_SCHEMA_VERSION,
        "dataset": dataset,
        "parent_dataset": parent_dataset,
        "parent_ranking": str(parent_ranking),
        "candidate_definition": (
            "同一曝光请求 (user_id, timestamp) 内 item_features.is_ad == 1 的物品"
        ),
        "label": {
            "field": "click",
            "source": "KuaiSAR_v2/rec_inter.csv",
            "values": [0, 1],
            "meaning": "该广告曝光是否被点击；不是转化",
        },
        "conversion_label": audit_conversion_labels(dataset_root, parent_dataset),
        "split_unit": "request=(user_id, timestamp)",
        "split_source": f"{parent_ranking}（与父数据集请求级切分一致）",
        "history_protocol": "序列模型只读取 interaction_timestamp < request_timestamp 的交互",
        "splits": split_stats,
        "fingerprints": fingerprints,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    problems = verify_ad_store(target, schema)
    if problems:
        raise AssertionError(f"广告样本自检失败：{problems[:5]}")
    (target / "schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[ok] 写入 {target}（{round(time.time() - started, 1)} s）", flush=True)
    return schema


def verify_ad_store(target: Path, schema: dict[str, Any] | None = None) -> list[str]:
    """校验 npy 哈希；返回问题列表（空表示通过）。"""
    if schema is None:
        return ["缺少 schema"]
    problems: list[str] = []
    base = Path(target)
    for name, expected in (schema.get("fingerprints") or {}).items():
        path = base / f"{name}.npy"
        if not path.exists():
            problems.append(f"缺少 {name}.npy")
        elif _sha256(path) != expected:
            problems.append(f"{name}.npy 哈希不一致")
    return problems


def load_ad_store(
    dataset: str = "kuasar_small",
    data_dir: Path = DEFAULT_DATA_DIR,
    split: str = "test",
    *,
    verify: bool = True,
) -> AdView:
    if split not in SPLITS:
        raise ValueError(f"split 必须是 {SPLITS}")
    dataset_root = resolve_dataset_data_dir(dataset, Path(data_dir))
    base = ads_dir(dataset_root)
    schema_path = base / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"{base} 不存在；先运行 python -m recsys.ads.data --dataset {dataset}"
        )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if verify:
        problems = verify_ad_store(base, schema)
        if problems:
            raise ValueError(
                f"广告样本校验失败（{len(problems)} 项，前 3 项：{problems[:3]}）；"
                f"重建命令：python -m recsys.ads.data --force"
            )
    view = AdView(
        dataset=dataset,
        split=split,
        users=np.load(base / f"{split}_users.npy", mmap_mode="r"),
        items=np.load(base / f"{split}_items.npy", mmap_mode="r"),
        labels=np.load(base / f"{split}_labels.npy", mmap_mode="r"),
        groups=np.asarray(np.load(base / f"{split}_groups.npy", mmap_mode="r")),
        timestamps=np.asarray(np.load(base / f"{split}_timestamps.npy", mmap_mode="r")),
    )
    if int(view.groups.sum()) != len(view):
        raise ValueError("广告样本分组与行数不一致；重建广告样本")
    return view


def ad_store_schema(
    dataset: str = "kuasar_small", data_dir: Path = DEFAULT_DATA_DIR
) -> dict[str, Any]:
    dataset_root = resolve_dataset_data_dir(dataset, Path(data_dir))
    path = ads_dir(dataset_root) / "schema.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在；先运行 python -m recsys.ads.data")
    return json.loads(path.read_text(encoding="utf-8"))


def audit_conversion_labels(
    dataset_root: Path, parent_dataset: str = "kuasar"
) -> dict[str, Any]:
    """如实核对真实转化标签：只读原始 CSV 表头，不假设一定存在。"""
    header_path = (
        Path(dataset_root).parents[0]
        / parent_dataset
        / "raw"
        / "KuaiSAR_v2"
        / "rec_inter.csv"
    )
    fields: list[str] = []
    if header_path.exists():
        with header_path.open("r", encoding="utf-8-sig", newline="") as file:
            header = file.readline().strip()
        fields = [part.strip() for part in header.split(",") if part.strip()]
    lowered = {field.lower() for field in fields}
    matched = [name for name in CONVERSION_FIELD_CANDIDATES if name in lowered]
    return {
        "available": bool(matched),
        "checked_fields": fields or ["header_not_found"],
        "matched_conversion_fields": matched,
        "interaction_fields": fields,
        "note": (
            "KuaiSAR rec_inter 只有 click / like / follow / forward / play_time 等行为，"
            "没有转化（conversion/purchase）字段。CVR 只能在 recsys.ads.simulated 的"
            "独立模拟样本上验证，不能表述为真实转化效果。"
            if not matched
            else "发现疑似转化字段；需人工核对口径后才能接真实 CVR。"
        ),
        "cvr_mode": "simulated" if not matched else "real_label_pending_review",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dataset", default="kuasar_small")
    parser.add_argument("--parent-dataset", default="kuasar")
    parser.add_argument("--max-train-requests", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = resolve_dataset_data_dir(args.dataset, args.data_dir)
    target = ads_dir(dataset_root)
    if args.check:
        schema_path = target / "schema.json"
        if not schema_path.exists():
            print(f"[error] 广告样本不存在：{target}")
            return 2
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        problems = verify_ad_store(target, schema)
        if problems:
            print(f"[fail] {len(problems)} 项不一致：{problems[:5]}")
            return 1
        print(f"[ok] 广告样本有效：{json.dumps(schema['splits'], ensure_ascii=False)}")
        return 0

    schema = build_ad_store(
        args.data_dir,
        dataset=args.dataset,
        parent_dataset=args.parent_dataset,
        max_train_requests=args.max_train_requests,
        seed=args.seed,
        force=args.force,
    )
    print(json.dumps(schema["splits"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
