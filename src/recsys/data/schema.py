"""Dataset-independent interaction schema and temporal split helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from recsys.config import SPLIT_STRATEGY


@dataclass(frozen=True)
class Interaction:
    """One encoded positive user-item interaction."""

    user_idx: int
    item_idx: int
    # KuaiRec timestamps contain milliseconds, so the common representation is float.
    timestamp: float


class ArrayInteractions(Sequence[Interaction]):
    """数组后端的交互序列（KuaiSAR 用）。

    7M+ 条交互如果物化成 `list[Interaction]` 要 1 GB 以上；这里只持有三列 numpy
    数组（通常是 mmap），按下标惰性构造 `Interaction`。`len()` / 索引 / 迭代都可用，
    但**不要**对它调用 `list()`——那会把内存开销一次性付清。
    """

    def __init__(
        self,
        users: np.ndarray,
        items: np.ndarray,
        timestamps: np.ndarray,
    ) -> None:
        if not (len(users) == len(items) == len(timestamps)):
            raise ValueError("users / items / timestamps 长度必须一致")
        self._users = users
        self._items = items
        self._timestamps = timestamps

    def __len__(self) -> int:
        return len(self._users)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                Interaction(int(self._users[i]), int(self._items[i]), float(self._timestamps[i]))
                for i in range(*index.indices(len(self)))
            ]
        return Interaction(
            int(self._users[index]),
            int(self._items[index]),
            float(self._timestamps[index]),
        )

    def __iter__(self) -> Iterator[Interaction]:
        for user, item, timestamp in zip(self._users, self._items, self._timestamps):
            yield Interaction(int(user), int(item), float(timestamp))


class SparseExposure:
    """按用户的曝光历史（CSR）：`indptr[u]:indptr[u+1]` 是用户 u 曝光过的物品。

    取代稠密 `[num_users, num_items]` mask —— KuaiSAR 上后者需要约 97.5 GiB。
    """

    def __init__(self, indptr: np.ndarray, items: np.ndarray) -> None:
        if indptr.size == 0:
            raise ValueError("indptr 不能为空")
        if int(indptr[-1]) != len(items):
            raise ValueError("indptr 末尾必须等于 items 长度")
        self.indptr = indptr
        self.items = items

    @property
    def num_users(self) -> int:
        return int(self.indptr.size) - 1

    def __len__(self) -> int:
        return int(self.items.size)

    def items_for(self, user_idx: int) -> np.ndarray:
        start = int(self.indptr[user_idx])
        end = int(self.indptr[user_idx + 1])
        return self.items[start:end]

    def as_set(self, user_idx: int) -> set[int]:
        return set(self.items_for(user_idx).tolist())


class SortedIndexLookup(Mapping[int, int]):
    """原始 ID → 连续索引，基于已排序数组的二分查找。

    不物化 `dict[int, int]`：KuaiSAR 有 405 万物品，字典约占 400–600 MB。
    """

    def __init__(self, original_ids: np.ndarray) -> None:
        self._ids = original_ids

    def __getitem__(self, key: int) -> int:
        position = int(np.searchsorted(self._ids, key))
        if position >= self._ids.size or int(self._ids[position]) != int(key):
            raise KeyError(key)
        return position

    def __iter__(self) -> Iterator[int]:
        return iter(self._ids)

    def __len__(self) -> int:
        return int(self._ids.size)


class SeenItemsView(Mapping[int, set[int]]):
    """把稀疏曝光包装成 `{user: set(items)}` 的只读视图，取哪个用户才构造哪个集合。"""

    def __init__(self, exposure: SparseExposure, users: Sequence[int]) -> None:
        self._exposure = exposure
        self._users = tuple(int(user) for user in users)

    def __getitem__(self, user_idx: int) -> set[int]:
        return self._exposure.as_set(int(user_idx))

    def __iter__(self) -> Iterator[int]:
        return iter(self._users)

    def __len__(self) -> int:
        return len(self._users)


@dataclass(frozen=True)
class IdMappings:
    """Bidirectional mappings between source IDs and contiguous model indices.

    `index_to_*` 可以是 numpy 数组（KuaiSAR），`*_to_index` 可以是
    `SortedIndexLookup`；只有需要写文件时才要求可枚举的 dict。
    """

    user_to_index: Mapping[int, int]
    index_to_user: Sequence[int]
    item_to_index: Mapping[int, int]
    index_to_item: Sequence[int]


@dataclass(frozen=True)
class ItemFeatures:
    """按 item_idx 对齐的物品特征表（列名 → 数组）。

    列可以是原始 id、连续编码、计数或布尔标记，具体语义写在 `schema["columns"]`
    （由 `kuasar_features.py` 生成）。数组长度必须等于物品词表大小。
    """

    arrays: Mapping[str, np.ndarray]
    schema: Mapping[str, object]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.arrays)

    @property
    def num_items(self) -> int:
        return int(next(iter(self.arrays.values())).size) if self.arrays else 0

    def column(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError as error:
            raise KeyError(f"未知物品特征列 {name!r}；可选：{', '.join(self.arrays)}") from error

    def fingerprint(self) -> str:
        """特征表指纹（取 schema 里记录的每列 sha256，按列名排序拼接）。"""
        fingerprints = (self.schema or {}).get("fingerprints") or {}
        digest = hashlib.sha256()
        for name in sorted(fingerprints):
            digest.update(f"{name}={fingerprints[name]}\n".encode("utf-8"))
        return digest.hexdigest()


@dataclass(frozen=True)
class PreparedData:
    """The common data object consumed by every recommendation model."""

    train_interactions: Sequence[Interaction]
    validation_interactions: Sequence[Interaction]
    test_interactions: Sequence[Interaction]
    mappings: IdMappings
    # Optional exact pre-target exposure masks, including non-positive watches.
    # Shape: [num_users, num_items]. Synthetic tests may omit these.
    validation_seen_mask: np.ndarray | None = None
    test_seen_mask: np.ndarray | None = None
    # 稀疏曝光历史（KuaiSAR）：与 mask 二选一，优先使用。
    validation_seen: SparseExposure | None = None
    test_seen: SparseExposure | None = None
    # 物品特征通道（Phase 1.5）；没有特征表时为 None。
    item_features: ItemFeatures | None = None
    # 数据集自己的切分协议；写进 checkpoint，评测时据此校验。
    split_strategy: str = SPLIT_STRATEGY

    @property
    def num_users(self) -> int:
        return len(self.mappings.index_to_user)

    @property
    def num_items(self) -> int:
        return len(self.mappings.index_to_item)


class InteractionDataset(Dataset):
    """PyTorch dataset exposing the same tensors for every source dataset."""

    def __init__(self, interactions: Sequence[Interaction]) -> None:
        # 不再复制成 list：KuaiSAR 的序列是数组视图，复制会白付一次内存。
        self.interactions = interactions

    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        interaction = self.interactions[index]
        return {
            "user_idx": torch.tensor(interaction.user_idx, dtype=torch.long),
            "item_idx": torch.tensor(interaction.item_idx, dtype=torch.long),
            "timestamp": torch.tensor(interaction.timestamp, dtype=torch.float64),
        }


def seen_items_for_split(
    data: PreparedData, split: str, include_validation_in_test: bool = True
) -> Mapping[int, set[int]]:
    """Items actually observed before the target, never at or after its time."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    sparse = (
        data.validation_seen
        if split == "validation" or not include_validation_in_test
        else data.test_seen
    )
    targets = data.validation_interactions if split == "validation" else data.test_interactions
    users = {row.user_idx for row in targets}
    if sparse is not None:
        # 只按需构造每个用户的集合，避免 2400 万条曝光一次性进 Python 对象
        return SeenItemsView(sparse, sorted(users))
    mask = (
        data.validation_seen_mask
        if split == "validation" or not include_validation_in_test
        else data.test_seen_mask
    )
    if mask is not None:
        return {user: set(np.flatnonzero(mask[user]).tolist()) for user in users}
    # Backwards-compatible fallback for hand-built small PreparedData objects.
    visible = list(data.train_interactions)
    if split == "test" and include_validation_in_test:
        visible.extend(data.validation_interactions)
    result = {user: set() for user in users}
    for row in visible:
        if row.user_idx in result:
            result[row.user_idx].add(row.item_idx)
    return result


def temporal_train_validation_test_split(
    interactions: Sequence[Interaction],
) -> tuple[list[Interaction], list[Interaction], list[Interaction]]:
    """Hold out the last two positives only when their timestamps are unambiguous.

    Users with fewer than three positives stay entirely in train.  This keeps
    their IDs available without evaluating a user who has no training history.
    A user's train/validation/test boundary must satisfy train_time < val_time
    < test_time. If any boundary shares a timestamp, all of that user's rows
    stay in train; item IDs never stand in for an unknown event order.
    """
    interactions_by_user: dict[int, list[Interaction]] = defaultdict(list)
    for interaction in interactions:
        interactions_by_user[interaction.user_idx].append(interaction)

    train: list[Interaction] = []
    validation: list[Interaction] = []
    test: list[Interaction] = []
    for user_interactions in interactions_by_user.values():
        chronological = sorted(user_interactions, key=lambda row: row.timestamp)
        if (
            len(chronological) < 3
            or not chronological[-3].timestamp < chronological[-2].timestamp
            or not chronological[-2].timestamp < chronological[-1].timestamp
        ):
            train.extend(chronological)
        else:
            train.extend(chronological[:-2])
            validation.append(chronological[-2])
            test.append(chronological[-1])

    sort_key = lambda row: (row.user_idx, row.timestamp, row.item_idx)
    train.sort(key=sort_key)
    validation.sort(key=sort_key)
    test.sort(key=sort_key)
    return train, validation, test


def save_processed_data(
    processed_dir: Path,
    train_interactions: Sequence[Interaction],
    validation_interactions: Sequence[Interaction],
    test_interactions: Sequence[Interaction],
    mappings: IdMappings,
    protocol_metadata: dict[str, str | float] | None = None,
) -> None:
    """Save common interactions and reversible source-ID mappings."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "user": {
            "original_to_index": {
                str(original_id): index
                for original_id, index in mappings.user_to_index.items()
            },
            "index_to_original": mappings.index_to_user,
        },
        "item": {
            "original_to_index": {
                str(original_id): index
                for original_id, index in mappings.item_to_index.items()
            },
            "index_to_original": mappings.index_to_item,
        },
    }
    with (processed_dir / "id_mappings.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    with (processed_dir / "protocol.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"split_strategy": SPLIT_STRATEGY, **(protocol_metadata or {})},
            file,
            ensure_ascii=False,
            indent=2,
        )

    with (processed_dir / "interactions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["user_idx", "item_idx", "timestamp", "split"])
        for split_name, rows in (
            ("train", train_interactions),
            ("validation", validation_interactions),
            ("test", test_interactions),
        ):
            for row in rows:
                writer.writerow([row.user_idx, row.item_idx, row.timestamp, split_name])
