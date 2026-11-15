"""效果专项实验的公共件：数据、候选、指标、成对 bootstrap、交互先验特征。

设计约束
--------
* 指标公式与 ``recsys.evaluation.metrics`` 逐字一致（同样的 hits / ideal DCG），
  并提供一个 ``assert_matches_official`` 校验，防止自算指标和项目口径漂移。
* 这里不写任何评测协议以外的逻辑：候选池由已落盘的 ``fusion.json`` 提供，
  只改变排序顺序。
* 交互先验特征一律「严格早于请求时间」（``ts < t``），训练侧与推理侧同一实现。
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recsys.data.loader import prepare_dataset
from recsys.data.schema import PreparedData

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
BASELINE_DIR = REPO / "experiments" / "2026-11-10-formal-e2e" / "recommendation"
DATA_DIR = REPO / "data"
RANKING_DIR = DATA_DIR / "kuasar" / "processed" / "ranking"


# --------------------------------------------------------------------------- 数据


def load_data() -> PreparedData:
    return prepare_dataset("kuasar_small", DATA_DIR)


def targets_by_user(data: PreparedData, split: str = "test") -> dict[int, set[int]]:
    interactions = data.test_interactions if split == "test" else data.validation_interactions
    out: dict[int, set[int]] = defaultdict(set)
    for row in interactions:
        out[int(row.user_idx)].add(int(row.item_idx))
    return dict(out)


def load_fusion(path: Path = BASELINE_DIR / "fusion.json") -> dict[int, list[dict]]:
    """读取已落盘的融合候选池：user -> [ {item_idx, fusion_rank, sources}, ... ]。"""
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    users: dict[int, list[dict]] = {}
    for raw_user, rows in blob.items():
        ordered = sorted(rows, key=lambda row: row["fusion_rank"])
        users[int(raw_user)] = [
            {
                "item": int(row["item_idx"]),
                "fusion_rank": int(row["fusion_rank"]),
                "sources": row.get("sources") or {},
            }
            for row in ordered
        ]
    return users


# --------------------------------------------------------------------------- 指标


def per_user_metrics(
    recommendations: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    ks: tuple[int, ...] = (20, 100),
) -> dict[str, np.ndarray]:
    """逐用户指标数组；公式与 ``recsys.evaluation.metrics`` 一致。"""
    users = sorted(ground_truth)
    out: dict[str, np.ndarray] = {}
    for k in ks:
        recall, ndcg, hit, precision = [], [], [], []
        for user in users:
            relevant = ground_truth[user]
            top = recommendations.get(user, [])[:k]
            hits = len(set(top) & relevant)
            recall.append(hits / len(relevant))
            precision.append(hits / k)
            hit.append(1.0 if hits else 0.0)
            dcg = sum(
                1.0 / math.log2(rank + 2)
                for rank, item in enumerate(top)
                if item in relevant
            )
            ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
            ndcg.append(dcg / ideal)
        out[f"Recall@{k}"] = np.asarray(recall, dtype=np.float64)
        out[f"NDCG@{k}"] = np.asarray(ndcg, dtype=np.float64)
        out[f"HitRate@{k}"] = np.asarray(hit, dtype=np.float64)
        out[f"Precision@{k}"] = np.asarray(precision, dtype=np.float64)
    return out


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "se": float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0,
    }


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    resamples: int = 10_000,
    seed: int = 20261110,
) -> dict[str, float]:
    """成对 bootstrap：返回 ``a - b`` 的均值差与 95% 区间（按用户重采样）。"""
    if a.shape != b.shape:
        raise ValueError("成对比较要求两个数组长度一致")
    rng = np.random.default_rng(seed)
    n = a.size
    diff = a - b
    observed = float(diff.mean())
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = rng.integers(0, n, size=n)
        draws[index] = diff[sample].mean()
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "delta": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "crosses_zero": bool(low <= 0.0 <= high),
        "resamples": resamples,
    }


def assert_matches_official(
    metrics: dict[str, np.ndarray],
    expected: dict[str, float],
    *,
    tol: float = 1e-6,
    keys: tuple[str, ...] = ("Recall@20", "NDCG@20", "HitRate@20", "Precision@20"),
) -> None:
    """自算的逐用户均值必须和项目官方口径一致。"""
    for key in keys:
        actual = float(metrics[key].mean())
        if abs(actual - float(expected[key])) > tol:
            raise AssertionError(f"{key} 自算 {actual} 与官方 {expected[key]} 不一致")


# --------------------------------------------------------------------------- 特征


@dataclass
class InteractionPriors:
    """严格早于请求时间的交互先验特征。

    每个查询 ``(user, item, timestamp)`` 只使用 ``train`` 切分里
    ``interaction_timestamp < timestamp`` 的点击，训练行与推理行同一实现，
    因此不存在「训练时看到未来、推理时看不到」的口径差。
    """

    data: PreparedData

    def __post_init__(self) -> None:
        features = self.data.item_features
        if features is None:
            raise ValueError("数据集缺少 item_features")
        self.item_category = np.asarray(features.column("category_first_code"), dtype=np.int64)
        self.item_author = np.asarray(features.column("author_id"), dtype=np.int64)

        per_item: dict[int, list[int]] = defaultdict(list)
        per_author: dict[int, list[int]] = defaultdict(list)
        per_user: dict[int, list[int]] = defaultdict(list)
        per_user_cat: dict[tuple[int, int], list[int]] = defaultdict(list)
        per_user_author: dict[tuple[int, int], list[int]] = defaultdict(list)
        for row in self.data.train_interactions:
            user = int(row.user_idx)
            item = int(row.item_idx)
            timestamp = int(row.timestamp)
            category = int(self.item_category[item])
            author = int(self.item_author[item])
            per_item[item].append(timestamp)
            per_user[user].append(timestamp)
            if author >= 0:
                per_author[author].append(timestamp)
                per_user_author[(user, author)].append(timestamp)
            if category >= 0:
                per_user_cat[(user, category)].append(timestamp)

        def sorted_arrays(source: dict) -> dict:
            return {key: np.sort(np.asarray(values, dtype=np.int64)) for key, values in source.items()}

        self.item_ts = sorted_arrays(per_item)
        self.author_ts = sorted_arrays(per_author)
        self.user_ts = sorted_arrays(per_user)
        self.user_cat_ts = sorted_arrays(per_user_cat)
        self.user_author_ts = sorted_arrays(per_user_author)

    @staticmethod
    def _count(store: dict, key, timestamp: int) -> int:
        values = store.get(key)
        if values is None:
            return 0
        return int(np.searchsorted(values, timestamp, side="left"))

    def build(
        self,
        users: np.ndarray,
        items: np.ndarray,
        timestamps: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """返回 5 列交互先验（未做 log 的原始计数，训练侧统一 log1p）。"""
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        timestamps = np.asarray(timestamps, dtype=np.int64)
        if not (users.shape == items.shape == timestamps.shape):
            raise ValueError("users / items / timestamps 长度必须一致")

        columns = {name: np.zeros(items.size, dtype=np.float32) for name in (
            "item_pop_before",
            "author_pop_before",
            "user_activity_before",
            "user_category_before",
            "user_author_before",
        )}
        for index in range(items.size):
            user = int(users[index])
            item = int(items[index])
            timestamp = int(timestamps[index])
            category = int(self.item_category[item])
            author = int(self.item_author[item])
            columns["item_pop_before"][index] = self._count(self.item_ts, item, timestamp)
            columns["author_pop_before"][index] = (
                self._count(self.author_ts, author, timestamp) if author >= 0 else 0.0
            )
            columns["user_activity_before"][index] = self._count(self.user_ts, user, timestamp)
            columns["user_category_before"][index] = (
                self._count(self.user_cat_ts, (user, category), timestamp) if category >= 0 else 0.0
            )
            columns["user_author_before"][index] = (
                self._count(self.user_author_ts, (user, author), timestamp) if author >= 0 else 0.0
            )
        for name, column in columns.items():
            columns[name] = np.log1p(column).astype(np.float32)
        return columns
