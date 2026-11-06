"""为 SASRec 构造序列，并保证不会把验证/测试信息泄漏进输入。"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from recsys.data.schema import Interaction, PreparedData, seen_items_for_split
from recsys.training.samplers import sample_negative_item


def build_user_sequences(
    interactions: Sequence[Interaction],
) -> dict[int, list[int]]:
    """把每个用户交互过的内部 item_idx 按时间顺序整理成一个列表。"""
    rows_by_user: dict[int, list[Interaction]] = defaultdict(list)
    for interaction in interactions:
        rows_by_user[interaction.user_idx].append(interaction)
    return {
        user_idx: [
            row.item_idx
            for row in sorted(rows, key=lambda row: (row.timestamp, row.item_idx))
        ]
        for user_idx, rows in rows_by_user.items()
    }


def pad_item_sequence(item_indices: Sequence[int], max_seq_len: int) -> list[int]:
    """把 item_idx 整体加 1、只保留最新的若干项，再在右侧用 0 补齐。

    加 1 是因为 token 0 要留给补齐位置；只取末尾是因为 SASRec 只关心最近的历史。
    """
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    shifted_items = [item_idx + 1 for item_idx in item_indices[-max_seq_len:]]
    return shifted_items + [0] * (max_seq_len - len(shifted_items))


class SASRecTrainDataset(Dataset):
    """每个样本是一条用户序列，并且已经对齐好「下一个物品」的目标。"""

    def __init__(
        self,
        train_interactions: Sequence[Interaction],
        num_items: int,
        max_seq_len: int,
        seed: int = 42,
    ) -> None:
        if num_items <= 0:
            raise ValueError("num_items must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.user_sequences = build_user_sequences(train_interactions)
        rows_by_user: dict[int, list[Interaction]] = defaultdict(list)
        for interaction in train_interactions:
            rows_by_user[interaction.user_idx].append(interaction)
        self.user_rows = {
            user_idx: sorted(rows, key=lambda row: (row.timestamp, row.item_idx))
            for user_idx, rows in rows_by_user.items()
        }
        self.user_indices = sorted(
            user_idx
            for user_idx, rows in self.user_rows.items()
            if len({row.timestamp for row in rows[-(max_seq_len + 1) :]}) >= 2
        )
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.random = random.Random(seed)

    def __len__(self) -> int:
        return len(self.user_indices)

    def _sample_negative(self, seen_items: set[int]) -> int:
        """随机取一个该用户从未交互过的物品（复用 samplers 里的实现）。"""
        return sample_negative_item(self.num_items, seen_items, self.random)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user_idx = self.user_indices[index]
        full_rows = self.user_rows[user_idx]
        full_sequence = [row.item_idx for row in full_rows]
        # 最多只需要 max_seq_len 组对齐好的 (输入, 下一个物品) 对。
        recent_items = full_sequence[-(self.max_seq_len + 1) :]
        input_items = recent_items[:-1]
        positive_items = recent_items[1:]
        recent_rows = full_rows[-(self.max_seq_len + 1) :]
        # 同一时间戳无法确定谁先谁后；这些位置可以留在上下文里，
        # 但不参与下一个物品的监督。
        ordered_targets = [
            next_row.timestamp > input_row.timestamp
            for input_row, next_row in zip(recent_rows[:-1], recent_rows[1:])
        ]

        # 每个正样本位置配一个负样本。采样结果已经加过 1，
        # 因此可以像正样本一样直接送进 embedding。
        seen_items = set(full_sequence)
        negative_items = [self._sample_negative(seen_items) for _ in positive_items]

        input_sequence = pad_item_sequence(input_items, self.max_seq_len)
        positive_sequence = [
            item_idx + 1 if ordered else 0
            for item_idx, ordered in zip(positive_items, ordered_targets)
        ]
        positive_sequence += [0] * (self.max_seq_len - len(positive_sequence))
        negative_sequence = [
            item_idx if ordered else 0
            for item_idx, ordered in zip(negative_items, ordered_targets)
        ] + [0] * (
            self.max_seq_len - len(negative_items)
        )
        return {
            "user_idx": torch.tensor(user_idx, dtype=torch.long),
            "input_sequence": torch.tensor(input_sequence, dtype=torch.long),
            "positive_sequence": torch.tensor(positive_sequence, dtype=torch.long),
            "negative_sequence": torch.tensor(negative_sequence, dtype=torch.long),
        }


@dataclass(frozen=True)
class SequenceEvaluationExample:
    """一个评测样本：输入序列 + 目标物品 + 需要过滤掉的已看物品。"""

    user_idx: int
    input_sequence: list[int]
    target_item_idx: int
    seen_item_indices: set[int]


def build_observed_histories(
    prepared_data: PreparedData,
    split: str,
    include_validation_in_test_history: bool = True,
) -> dict[int, list[int]]:
    """返回指定切分下模型「允许看到」的物品历史。"""
    observed_interactions = list(prepared_data.train_interactions)
    if split == "test" and include_validation_in_test_history:
        # 测试目标发生的时刻，验证集交互早已被观测到，
        # 因此它属于测试阶段的历史。
        observed_interactions += list(prepared_data.validation_interactions)
    return build_user_sequences(observed_interactions)


def build_evaluation_examples(
    prepared_data: PreparedData,
    split: str,
    max_seq_len: int,
    include_validation_in_test_history: bool = True,
) -> list[SequenceEvaluationExample]:
    """按时间顺序构造验证集或测试集的输入。

    验证集只用训练历史做预测；测试集用「训练 + 已观测到的验证交互」做预测。
    测试目标自身永远不会被放进输入序列。
    """
    if split == "validation":
        target_interactions = prepared_data.validation_interactions
    elif split == "test":
        target_interactions = prepared_data.test_interactions
    else:
        raise ValueError("split must be 'validation' or 'test'")

    histories = build_observed_histories(
        prepared_data,
        split,
        include_validation_in_test_history=include_validation_in_test_history,
    )
    # The controlled train-history-only variant changes the model input, not
    # which items had actually been watched by the test timestamp.
    filtered_items = seen_items_for_split(prepared_data, split)

    examples = []
    visible_rows = list(prepared_data.train_interactions)
    if split == "test" and include_validation_in_test_history:
        visible_rows.extend(prepared_data.validation_interactions)
    latest_visible_time: dict[int, float] = {}
    for row in visible_rows:
        latest_visible_time[row.user_idx] = max(
            latest_visible_time.get(row.user_idx, float("-inf")), row.timestamp
        )
    for target in target_interactions:
        if latest_visible_time.get(target.user_idx, float("-inf")) >= target.timestamp:
            raise ValueError("Evaluation history must strictly precede its target")
        history = histories.get(target.user_idx, [])
        if not history:
            continue
        examples.append(
            SequenceEvaluationExample(
                user_idx=target.user_idx,
                input_sequence=pad_item_sequence(history, max_seq_len),
                target_item_idx=target.item_idx,
                seen_item_indices=filtered_items[target.user_idx],
            )
        )
    return examples
