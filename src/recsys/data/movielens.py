"""MovieLens-1M 数据集的加载与预处理，供 Two-Tower 基线使用。"""

from __future__ import annotations

import shutil
import math
import ssl
import urllib.request
import zipfile
import numpy as np
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import certifi

from recsys.data.schema import (
    IdMappings,
    Interaction,
    InteractionDataset,
    PreparedData,
    save_processed_data,
    temporal_train_validation_test_split,
)


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
RATINGS_MEMBER = "ml-1m/ratings.dat"


@dataclass(frozen=True)
class RawRating:
    """ratings.dat 文件中的一行原始评分。"""

    original_user_id: int
    original_item_id: int
    rating: int
    timestamp: int


MovieLensDataset = InteractionDataset


def download_and_extract_movielens(data_dir: Path) -> Path:
    """按需下载并解压 MovieLens-1M，返回 ratings.dat 的路径。"""
    raw_dir = data_dir / "raw"
    ratings_path = raw_dir / RATINGS_MEMBER
    if ratings_path.exists():
        return ratings_path

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ml-1m.zip"
    if not zip_path.exists():
        # 先下载到 .part 临时文件，成功后再改名，避免中断留下半个压缩包。
        temporary_path = zip_path.with_suffix(".zip.part")
        print(f"Downloading MovieLens-1M from {MOVIELENS_URL}")
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(
            MOVIELENS_URL, context=ssl_context, timeout=120
        ) as response:
            with temporary_path.open("wb") as file:
                shutil.copyfileobj(response, file)
        temporary_path.replace(zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extract(RATINGS_MEMBER, raw_dir)

    return ratings_path


def read_ratings(ratings_path: Path) -> list[RawRating]:
    """读取 ``UserID::MovieID::Rating::Timestamp`` 格式的 MovieLens 评分文件。"""
    ratings: list[RawRating] = []

    # MovieLens-1M 使用 latin-1 编码，字段之间以 "::" 分隔（不是单个冒号）。
    with ratings_path.open("r", encoding="latin-1", newline="") as file:
        for line_number, line in enumerate(file, start=1):
            fields = line.rstrip("\n").split("::")
            if len(fields) != 4:
                raise ValueError(
                    f"Invalid ratings row at line {line_number}: expected 4 fields"
                )
            user_id, item_id, rating, timestamp = map(int, fields)
            ratings.append(RawRating(user_id, item_id, rating, timestamp))
    if not ratings:
        raise ValueError(f"No ratings found in {ratings_path}")
    return ratings


def build_id_mappings(positive_ratings: Sequence[RawRating]) -> IdMappings:
    """把 MovieLens 原始 ID 映射为从 0 开始的连续索引。

    模型 embedding 是按稠密的内部索引去查表的，而原始 ID 稀疏且不连续
    （例如 movie_id 可以到几千），所以这里先做一次编码，并同时保留两个方向：
    编码用于建表，解码用于把推荐结果还原成原始 ID。
    """
    original_user_ids = sorted({rating.original_user_id for rating in positive_ratings})
    original_item_ids = sorted({rating.original_item_id for rating in positive_ratings})
    return IdMappings(
        user_to_index={
            original_user_id: user_idx
            for user_idx, original_user_id in enumerate(original_user_ids)
        },
        index_to_user=original_user_ids,
        item_to_index={
            original_item_id: item_idx
            for item_idx, original_item_id in enumerate(original_item_ids)
        },
        index_to_item=original_item_ids,
    )


def encode_positive_interactions(
    ratings: Sequence[RawRating], positive_threshold: float = 4
) -> tuple[list[Interaction], IdMappings]:
    """过滤出正向反馈，并把原始 ID 编码成从 0 开始的连续索引。

    这是一个隐式反馈任务：评分 >= ``positive_threshold`` 记为「喜欢」，
    低于阈值的评分不参与训练，但在评估时仍属于已看过的物品。
    """
    if not math.isfinite(positive_threshold):
        raise ValueError("positive_threshold must be finite")
    positive_ratings = [
        rating for rating in ratings if rating.rating >= positive_threshold
    ]
    if not positive_ratings:
        raise ValueError("No positive interactions remain after rating filtering")

    mappings = build_id_mappings(positive_ratings)
    interactions = [
        Interaction(
            user_idx=mappings.user_to_index[rating.original_user_id],
            item_idx=mappings.item_to_index[rating.original_item_id],
            timestamp=rating.timestamp,
        )
        for rating in positive_ratings
    ]

    # 先按「用户 -> 时间」稳定排序，后面按用户切分时才能保证时间顺序确定。
    interactions.sort(key=lambda row: (row.user_idx, row.timestamp, row.item_idx))
    return interactions, mappings


split_interactions_per_user = temporal_train_validation_test_split


def prepare_movielens_from_file(
    ratings_path: Path,
    processed_dir: Path,
    positive_threshold: float = 4,
) -> PreparedData:
    """从本地 ratings.dat 生成预处理结果并保存到磁盘。"""
    ratings = read_ratings(ratings_path)
    interactions, mappings = encode_positive_interactions(
        ratings, positive_threshold=positive_threshold
    )
    train_interactions, validation_interactions, test_interactions = (
        temporal_train_validation_test_split(interactions)
    )
    validation_times = {row.user_idx: row.timestamp for row in validation_interactions}
    test_times = {row.user_idx: row.timestamp for row in test_interactions}
    validation_seen = np.zeros((len(mappings.index_to_user), len(mappings.index_to_item)), dtype=np.bool_)
    test_seen = np.zeros_like(validation_seen)
    for rating in ratings:
        user_idx = mappings.user_to_index.get(rating.original_user_id)
        item_idx = mappings.item_to_index.get(rating.original_item_id)
        if user_idx is None or item_idx is None:
            continue
        if rating.timestamp < validation_times.get(user_idx, float("-inf")):
            validation_seen[user_idx, item_idx] = True
        if rating.timestamp < test_times.get(user_idx, float("-inf")):
            test_seen[user_idx, item_idx] = True
    save_processed_data(
        processed_dir,
        train_interactions,
        validation_interactions,
        test_interactions,
        mappings,
        protocol_metadata={
            "positive_rule": "rating >= threshold",
            "positive_threshold": positive_threshold,
            "seen_filter": "all ratings strictly before target time",
        },
    )
    return PreparedData(
        train_interactions, validation_interactions, test_interactions, mappings,
        validation_seen, test_seen,
    )


def prepare_movielens(
    data_dir: Path,
    positive_threshold: float = 4,
) -> PreparedData:
    """下载、编码、按时间切分并保存 MovieLens-1M，是整个数据流程的入口。"""
    ratings_path = download_and_extract_movielens(data_dir)
    return prepare_movielens_from_file(
        ratings_path=ratings_path,
        processed_dir=data_dir / "processed",
        positive_threshold=positive_threshold,
    )
