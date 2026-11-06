"""Dataset-independent interaction schema and temporal split helpers."""

from __future__ import annotations

import csv
import json
import numpy as np
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class IdMappings:
    """Bidirectional mappings between source IDs and contiguous model indices."""

    user_to_index: dict[int, int]
    index_to_user: list[int]
    item_to_index: dict[int, int]
    index_to_item: list[int]


@dataclass(frozen=True)
class PreparedData:
    """The common data object consumed by every recommendation model."""

    train_interactions: list[Interaction]
    validation_interactions: list[Interaction]
    test_interactions: list[Interaction]
    mappings: IdMappings
    # Optional exact pre-target exposure masks, including non-positive watches.
    # Shape: [num_users, num_items]. Synthetic tests may omit these.
    validation_seen_mask: np.ndarray | None = None
    test_seen_mask: np.ndarray | None = None

    @property
    def num_users(self) -> int:
        return len(self.mappings.index_to_user)

    @property
    def num_items(self) -> int:
        return len(self.mappings.index_to_item)


class InteractionDataset(Dataset):
    """PyTorch dataset exposing the same tensors for every source dataset."""

    def __init__(self, interactions: Sequence[Interaction]) -> None:
        self.interactions = list(interactions)

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
) -> dict[int, set[int]]:
    """Items actually observed before the target, never at or after its time."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    mask = (
        data.validation_seen_mask
        if split == "validation" or not include_validation_in_test
        else data.test_seen_mask
    )
    targets = data.validation_interactions if split == "validation" else data.test_interactions
    users = {row.user_idx for row in targets}
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
