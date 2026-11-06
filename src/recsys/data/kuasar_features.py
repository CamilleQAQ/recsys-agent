"""KuaiSAR 物品特征通道（Phase 1.5）。

`item_features.csv` 有 695 万行 / 1.9 GB，但真正要建模的只有交互词表里的
4,046,367 个物品。这里做一件事：把宽表压成**按 item_idx 对齐的定长数组**
（npy，mmap 读取），供排序模型与 LightGBM 使用。

产出（`data/kuasar/processed/features/`）：

    category_first_id.npy / category_first_code.npy    一级类目（原始 id / 连续编码）
    category_second_id.npy / category_second_code.npy  二级类目
    author_id.npy / author_freq.npy                    作者 id 与其出现频次
    item_type_code.npy                                 NORMAL / AD / UNKNOWN / LIVEGAME_PLAYBACK
    upload_type_code.npy                               上传来源（约 45 种）
    upload_days.npy / upload_valid.npy                 上传日期（1970 起天数 / 是否有效）
    caption_tokens.npy                                 caption 的 token 数（不是内容）
    is_ad.npy                                          是否广告物品
    schema.json                                        每列含义、dtype、基数、填充率

约定：数组长度都等于 `ids/item_ids.npy` 的长度，下标就是模型里的 item_idx；
取不到特征的物品填 -1（连续编码列）或 0（布尔/计数列），由 `fill_rate` 记录缺多少。

用法：

    python -m recsys.data.kuasar_features            # 全量构建
    python -m recsys.data.kuasar_features --max-rows 200000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from recsys.config import DEFAULT_DATA_DIR
from recsys.data.kuasar import KUASAR_DATASET_DIRNAME

#: caption 里有超过 128 KB 的极端值（最长 24,180 个 token），默认上限会抛 _csv.Error。
csv.field_size_limit(2**31 - 1)

ITEM_FEATURES_CSV = "KuaiSAR_v2/item_features.csv"
_EPOCH = date(1970, 1, 1)
_AD_TYPE = "AD"

COLUMNS = (
    "category_first_id",
    "category_first_code",
    "category_second_id",
    "category_second_code",
    "author_id",
    "author_freq",
    "item_type_code",
    "upload_type_code",
    "upload_days",
    "upload_valid",
    "caption_tokens",
    "is_ad",
)


def dataset_dir(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    return data_dir / KUASAR_DATASET_DIRNAME


def features_dir(dataset_root: Path) -> Path:
    return dataset_root / "processed" / "features"


def _feature_spec() -> dict[str, dict[str, Any]]:
    """列语义；基数在构建后回填。"""
    return {
        "category_first_id": {"dtype": "int32", "role": "categorical_raw", "missing": -1},
        "category_first_code": {"dtype": "int32", "role": "categorical_code", "missing": -1},
        "category_second_id": {"dtype": "int32", "role": "categorical_raw", "missing": -1},
        "category_second_code": {"dtype": "int32", "role": "categorical_code", "missing": -1},
        "author_id": {"dtype": "int32", "role": "categorical_raw", "missing": -1},
        "author_freq": {"dtype": "int32", "role": "count", "missing": 0},
        "item_type_code": {"dtype": "int8", "role": "categorical_code", "missing": -1},
        "upload_type_code": {"dtype": "int8", "role": "categorical_code", "missing": -1},
        "upload_days": {"dtype": "int32", "role": "numeric", "missing": -1},
        "upload_valid": {"dtype": "int8", "role": "flag", "missing": 0},
        "caption_tokens": {"dtype": "int32", "role": "numeric", "missing": 0},
        "is_ad": {"dtype": "int8", "role": "flag", "missing": 0},
    }


def _parse_days(text: str) -> tuple[int, int]:
    """'YYYY-MM-DD' → (1970 起天数, 是否有效)。1970 哨兵视为缺失。"""
    text = text.strip()
    if len(text) < 10:
        return -1, 0
    try:
        year, month, day = int(text[0:4]), int(text[5:7]), int(text[8:10])
        if year <= 1970:
            return -1, 0
        return date(year, month, day).toordinal() - _EPOCH.toordinal(), 1
    except ValueError:
        return -1, 0


def _caption_tokens(text: str) -> int:
    text = text.strip()
    if not text or text in {"[]", "null"}:
        return 0
    inner = text.strip("[]")
    return inner.count(",") + 1 if inner else 0


def _code(values: np.ndarray) -> tuple[np.ndarray, int]:
    """把 -1 之外的取值压成 0..k 的连续编码，返回 (codes, k)。"""
    codes = np.full(values.shape, -1, dtype=np.int32)
    valid = values >= 0
    if not valid.any():
        return codes, 0
    unique, inverse = np.unique(values[valid], return_inverse=True)
    codes[valid] = inverse.astype(np.int32)
    return codes, int(unique.size)


def build_feature_store(
    data_dir: Path = DEFAULT_DATA_DIR, *, max_rows: int | None = None
) -> dict[str, Any]:
    dataset_root = dataset_dir(data_dir)
    processed = dataset_root / "processed"
    item_ids_path = processed / "ids" / "item_ids.npy"
    source = dataset_root / "raw" / ITEM_FEATURES_CSV
    if not item_ids_path.exists():
        raise FileNotFoundError(
            f"找不到 {item_ids_path}；先运行 python -m recsys.data.kuasar_store。"
        )
    if not source.exists():
        raise FileNotFoundError(f"找不到 {source}；先运行 python -m recsys.data.kuasar。")

    started = time.time()
    item_ids = np.load(item_ids_path)  # 已排序，供 searchsorted 使用
    num_items = int(item_ids.size)
    print(f"[1/3] 物品词表 {num_items:,}，流式读取 {source.name} …", flush=True)

    arrays: dict[str, np.ndarray] = {
        "category_first_id": np.full(num_items, -1, dtype=np.int32),
        "category_second_id": np.full(num_items, -1, dtype=np.int32),
        "author_id": np.full(num_items, -1, dtype=np.int32),
        "item_type_code": np.full(num_items, -1, dtype=np.int8),
        "upload_type_code": np.full(num_items, -1, dtype=np.int8),
        "upload_days": np.full(num_items, -1, dtype=np.int32),
        "upload_valid": np.zeros(num_items, dtype=np.int8),
        "caption_tokens": np.zeros(num_items, dtype=np.int32),
        "is_ad": np.zeros(num_items, dtype=np.int8),
    }
    item_type_codes: dict[str, int] = {}
    upload_type_codes: dict[str, int] = {}

    file = source.open("r", encoding="utf-8-sig", newline="")
    reader = csv.reader(file)
    header = next(reader)
    index = {name: position for position, name in enumerate(header)}
    required = (
        "item_id",
        "first_level_category_id",
        "second_level_category_id",
        "author_id",
        "item_type",
        "upload_type",
        "upload_time",
        "caption",
    )
    missing = [name for name in required if name not in index]
    if missing:
        file.close()
        raise ValueError(f"{source} 缺少列 {missing}")

    rows = 0
    matched = 0
    try:
        for row in reader:
            rows += 1
            if max_rows is not None and rows > max_rows:
                rows -= 1
                break
            raw_id = int(row[index["item_id"]])
            position = int(np.searchsorted(item_ids, raw_id))
            if position >= num_items or int(item_ids[position]) != raw_id:
                continue  # 特征表里的冷启动物品，不在交互词表内
            matched += 1
            arrays["category_first_id"][position] = int(row[index["first_level_category_id"]])
            arrays["category_second_id"][position] = int(row[index["second_level_category_id"]])
            arrays["author_id"][position] = int(float(row[index["author_id"]]))
            item_type = row[index["item_type"]].strip()
            arrays["item_type_code"][position] = item_type_codes.setdefault(
                item_type, len(item_type_codes)
            )
            upload_type = row[index["upload_type"]].strip()
            arrays["upload_type_code"][position] = upload_type_codes.setdefault(
                upload_type, len(upload_type_codes)
            )
            days, valid = _parse_days(row[index["upload_time"]])
            arrays["upload_days"][position] = days
            arrays["upload_valid"][position] = valid
            arrays["caption_tokens"][position] = _caption_tokens(row[index["caption"]])
            arrays["is_ad"][position] = 1 if item_type == _AD_TYPE else 0
    finally:
        file.close()

    print(f"      读到 {rows:,} 行，命中词表 {matched:,} 个物品", flush=True)

    print("[2/3] 生成类目编码与作者频次 …", flush=True)
    arrays["category_first_code"], first_cardinality = _code(arrays["category_first_id"])
    arrays["category_second_code"], second_cardinality = _code(arrays["category_second_id"])
    author = arrays["author_id"]
    asr_freq = np.zeros(num_items, dtype=np.int32)
    valid_author = author >= 0
    if valid_author.any():
        unique_authors, inverse, counts = np.unique(
            author[valid_author], return_inverse=True, return_counts=True
        )
        asr_freq[valid_author] = counts[inverse].astype(np.int32)
    arrays["author_freq"] = asr_freq

    print("[3/3] 保存 npy 与 schema …", flush=True)
    target = features_dir(dataset_root)
    target.mkdir(parents=True, exist_ok=True)
    fingerprints: dict[str, str] = {}
    fill_rates: dict[str, float] = {}
    for name in COLUMNS:
        array = arrays[name]
        np.save(target / f"{name}.npy", array)
        fingerprints[name] = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        if name.endswith("_id") or name.endswith("_code"):
            fill_rates[name] = round(float((array >= 0).mean()), 6)
        elif name == "upload_valid":
            fill_rates[name] = round(float(array.mean()), 6)
        else:
            fill_rates[name] = round(float((array > 0).mean()), 6)

    schema = {
        "num_items": num_items,
        "source": source.name,
        "rows_read": rows,
        "matched_items": matched,
        "columns": {
            name: {**_feature_spec()[name], "fill_rate": fill_rates[name]}
            for name in COLUMNS
        },
        "cardinality": {
            "category_first": first_cardinality,
            "category_second": second_cardinality,
            "item_type": len(item_type_codes),
            "upload_type": len(upload_type_codes),
        },
        "item_type_values": {value: code for code, value in item_type_codes.items()},
        "upload_type_values": {value: code for code, value in upload_type_codes.items()},
        "is_ad_items": int(arrays["is_ad"].sum()),
        "fingerprints": fingerprints,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sampled": max_rows is not None,
        "elapsed_s": round(time.time() - started, 1),
    }
    (target / "schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] 写入 {target}（{len(COLUMNS)} 列）", flush=True)
    return schema


def load_item_features(dataset_root: Path | None = None) -> dict[str, Any] | None:
    """mmap 读取特征表；没有构建过时返回 None（特征是可选的）。"""
    dataset_root = dataset_root or dataset_dir(DEFAULT_DATA_DIR)
    base = features_dir(dataset_root)
    schema_path = base / "schema.json"
    if not schema_path.exists():
        return None
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    arrays = {name: np.load(base / f"{name}.npy", mmap_mode="r") for name in COLUMNS}
    return {"schema": schema, "arrays": arrays}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-rows", type=int, default=None, help="抽样行数（调试用）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    schema = build_feature_store(args.data_dir, max_rows=args.max_rows)
    print(json.dumps({k: schema[k] for k in ("num_items", "matched_items", "is_ad_items", "cardinality", "elapsed_s")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
