"""排序样本视图与「严格早于请求」的历史构造。

`kuasar_small` 是从全量 KuaiSAR 抽出的用户子集，processed 里只有正样本与曝光
历史，没有请求级候选表。排序模型仍复用父数据集 ``processed/ranking`` 的
train 候选（每个请求的曝光物品 + click 标签），按原始 ID 映射到子集词表。

修复历史泄漏后，排序样本必须携带请求时间戳；训练/评估时按
``(user, request_timestamp)`` 构造历史，只允许 ``interaction_timestamp <
request_timestamp`` 的交互进入模型输入。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import numpy as np

from recsys.data.schema import Interaction, PreparedData


class RankingTimestampError(RuntimeError):
    """排序工件缺少请求时间戳，序列排序模型无法安全构造历史。"""


def _map_parent_to_target(
    parent_ids: np.ndarray, target_ids: np.ndarray
) -> np.ndarray:
    """parent 索引 → target 索引；不可映射为 -1。两边 id 都必须已排序。"""
    positions = np.searchsorted(target_ids, parent_ids)
    clipped = np.minimum(positions, max(target_ids.size - 1, 0))
    valid = (positions < target_ids.size) & (target_ids[clipped] == parent_ids)
    return np.where(valid, positions, -1).astype(np.int64)


#: 广告样本构建等下游模块复用的公开别名（实现见 ``_map_parent_to_target``）。
map_parent_to_target = _map_parent_to_target


@dataclass(frozen=True)
class RankingView:
    users: np.ndarray
    items: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    requests: int
    parent_rows: int
    #: 每个候选行的请求时间戳；旧工件没有时间戳时为 None。
    timestamps: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.items.size)


def load_ranking_view(
    data: PreparedData,
    parent_ranking: Path,
    *,
    split: str = "train",
    max_rows: int = 20_000,
    max_requests: int | None = None,
    require_both_labels: bool = True,
    require_timestamps: bool = False,
) -> RankingView:
    """读取父表 split，保留「用户与物品都能映射」的完整请求。

    ``groups`` 是每个请求的候选数（连续分组），``timestamps`` 是该请求的
    时间戳（每个候选行重复一份）。旧工件没有时间戳时：
    - ``require_timestamps=False``：``timestamps=None``（LightGBM/DCNv2 兼容）；
    - ``require_timestamps=True``：抛 ``RankingTimestampError`` 并给出重建命令。
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("split 必须是 train/validation/test")
    ids_dir = parent_ranking.parent / "ids"
    parent_users = np.load(ids_dir / "user_ids.npy")
    parent_items = np.load(ids_dir / "item_ids.npy")
    target_users = np.asarray(data.mappings.index_to_user, dtype=np.int64)
    target_items = np.asarray(data.mappings.index_to_item, dtype=np.int64)
    user_map = _map_parent_to_target(parent_users, target_users)
    item_map = _map_parent_to_target(parent_items, target_items)

    users_array = np.load(parent_ranking / f"{split}_users.npy", mmap_mode="r")
    items_array = np.load(parent_ranking / f"{split}_items.npy", mmap_mode="r")
    labels_array = np.load(parent_ranking / f"{split}_labels.npy", mmap_mode="r")
    groups = np.asarray(np.load(parent_ranking / f"{split}_groups.npy", mmap_mode="r"))
    starts = np.concatenate(([0], np.cumsum(groups)))
    request_start_users = users_array[starts[:-1]]
    mapped_users = user_map[request_start_users]
    request_keep = mapped_users >= 0

    row_user_map = np.repeat(mapped_users, groups)
    row_item_map = item_map[items_array]
    row_keep = (row_user_map >= 0) & (row_item_map >= 0)

    labels = labels_array.astype(np.int64)
    positive = np.add.reduceat(labels * row_keep, starts[:-1])
    negative = np.add.reduceat((1 - labels) * row_keep, starts[:-1])
    counts = np.add.reduceat(row_keep.astype(np.int64), starts[:-1])
    if require_both_labels:
        request_keep &= (positive > 0) & (negative > 0)
    request_keep &= counts > 0

    timestamp_path = parent_ranking / f"{split}_timestamps.npy"
    request_timestamps = (
        np.asarray(np.load(timestamp_path, mmap_mode="r"))
        if timestamp_path.exists()
        else None
    )
    if require_timestamps and request_timestamps is None:
        raise RankingTimestampError(
            f"{timestamp_path} 不存在：旧排序工件没有请求时间戳，"
            "无法为 DIN/DIEN/Ultra-HSTU 构造严格早于请求的历史。\n"
            "重建命令：python -m recsys.data.kuasar_ranking --force"
        )
    if request_timestamps is not None and request_timestamps.size != groups.size:
        raise RankingTimestampError(
            f"{timestamp_path} 长度 {request_timestamps.size} 与请求数 {groups.size} 不一致；"
            "排序工件损坏，重建命令：python -m recsys.data.kuasar_ranking --force"
        )

    cumulative = np.cumsum(np.where(request_keep, counts, 0))
    chosen = np.flatnonzero(request_keep & (cumulative <= max_rows))
    if max_requests is not None:
        chosen = chosen[:max_requests]
    if chosen.size == 0:
        raise RuntimeError("没有满足条件的排序请求（检查词表映射与标签）")

    selected_rows = np.zeros(len(users_array), dtype=bool)
    for request in chosen:
        selected_rows[starts[request] : starts[request + 1]] = True
    rows = np.flatnonzero(selected_rows & row_keep)
    group_sizes = counts[chosen].astype(np.int32)
    if int(group_sizes.sum()) != rows.size:
        raise AssertionError("请求分组长度与样本行数不一致")
    row_timestamps = (
        np.repeat(request_timestamps[chosen], group_sizes)
        if request_timestamps is not None
        else None
    )
    if row_timestamps is not None and row_timestamps.size != rows.size:
        raise AssertionError("请求时间戳与候选行数不一致")
    return RankingView(
        users=np.asarray(row_user_map[rows], dtype=np.int64),
        items=np.asarray(row_item_map[rows], dtype=np.int64),
        labels=np.asarray(labels[rows], dtype=np.int32),
        groups=group_sizes,
        requests=int(chosen.size),
        parent_rows=int(len(users_array)),
        timestamps=row_timestamps,
    )


def history_interactions(
    data: PreparedData,
    split: str,
    *,
    include_validation_in_test_history: bool = True,
) -> Iterable[Interaction]:
    """某个切分在预测时可用的历史交互（尚未按请求时间过滤）。"""
    if split in {"train", "validation"}:
        return data.train_interactions
    if split == "test":
        if include_validation_in_test_history:
            return chain(data.train_interactions, data.validation_interactions)
        return data.train_interactions
    raise ValueError("split 必须是 train/validation/test")


def build_history_source(
    data: PreparedData,
    split: str,
    *,
    include_validation_in_test_history: bool = True,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """按用户整理 (timestamps, item_idx)，供按请求时间做二分查找。"""
    rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in history_interactions(
        data, split, include_validation_in_test_history=include_validation_in_test_history
    ):
        rows[row.user_idx].append((int(row.timestamp), int(row.item_idx)))
    source: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for user, values in rows.items():
        values.sort(key=lambda pair: (pair[0], pair[1]))
        timestamps = np.asarray([pair[0] for pair in values], dtype=np.int64)
        items = np.asarray([pair[1] for pair in values], dtype=np.int64)
        source[user] = (timestamps, items)
    return source


def histories_for_requests(
    source: Mapping[int, tuple[np.ndarray, np.ndarray]],
    requests: Iterable[tuple[int, int]],
    *,
    max_history: int,
) -> dict[tuple[int, int], np.ndarray]:
    """为每个 ``(user, request_timestamp)`` 构造严格早于请求的历史 token。"""
    if max_history <= 0:
        raise ValueError("max_history 必须为正")
    output: dict[tuple[int, int], np.ndarray] = {}
    for user, timestamp in requests:
        key = (int(user), int(timestamp))
        if key in output:
            continue
        padded = np.zeros(max_history, dtype=np.int64)
        entry = source.get(key[0])
        if entry is not None:
            timestamps, items = entry
            # side="left"：同一时刻的点击一律排除，满足严格小于。
            count = int(np.searchsorted(timestamps, key[1], side="left"))
            if count > 0:
                selected = items[max(0, count - max_history) : count] + 1
                padded[-selected.size :] = selected
        output[key] = padded
    return output


def target_request_timestamps(data: PreparedData, split: str) -> dict[int, int]:
    """每个评测用户最早留出目标的时间戳（保守的上界，避免未来历史泄漏）。"""
    targets = data.validation_interactions if split == "validation" else data.test_interactions
    output: dict[int, int] = {}
    for row in targets:
        timestamp = int(row.timestamp)
        current = output.get(row.user_idx)
        if current is None or timestamp < current:
            output[row.user_idx] = timestamp
    return output


def histories_for_split_users(
    data: PreparedData,
    split: str,
    request_timestamps: Mapping[int, int],
    *,
    max_history: int,
    include_validation_in_test_history: bool = True,
) -> dict[int, np.ndarray]:
    """按用户构造历史（每个用户使用其请求时间戳过滤）。"""
    source = build_history_source(
        data,
        split,
        include_validation_in_test_history=include_validation_in_test_history,
    )
    per_request = histories_for_requests(
        source,
        ((user, timestamp) for user, timestamp in request_timestamps.items()),
        max_history=max_history,
    )
    return {
        int(user): per_request[(int(user), int(timestamp))]
        for user, timestamp in request_timestamps.items()
    }


__all__ = [
    "RankingTimestampError",
    "RankingView",
    "build_history_source",
    "histories_for_requests",
    "histories_for_split_users",
    "history_interactions",
    "load_ranking_view",
    "target_request_timestamps",
]
