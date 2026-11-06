"""推荐评测常用的 Top-K 排序指标。"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence


# 推荐结果：用户索引 -> 按分数从高到低排好的物品索引序列。
Recommendations = Mapping[int, Sequence[int]]
# 标准答案：用户索引 -> 该用户真正相关的物品集合。
GroundTruth = Mapping[int, Collection[int]]


def _require_relevant_items_for_every_user(ground_truth: GroundTruth) -> None:
    """每个参与评测的用户都必须至少有一个相关物品。"""
    if any(len(relevant_items) == 0 for relevant_items in ground_truth.values()):
        raise ValueError("each evaluated user must have at least one relevant item")


def _validate_inputs(ground_truth: GroundTruth, k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")
    if not ground_truth:
        raise ValueError("ground_truth must contain at least one user")
    _require_relevant_items_for_every_user(ground_truth)


def recall_at_k(
    recommendations: Recommendations, ground_truth: GroundTruth, k: int
) -> float:
    """每个用户在前 K 个推荐里命中相关物品的平均比例。"""
    _validate_inputs(ground_truth, k)
    recalls = []
    for user_idx, relevant_items in ground_truth.items():
        recommended_items = recommendations.get(user_idx, ())[:k]
        hits = len(set(recommended_items).intersection(relevant_items))
        recalls.append(hits / len(relevant_items))
    return sum(recalls) / len(recalls)


def hit_rate_at_k(
    recommendations: Recommendations, ground_truth: GroundTruth, k: int
) -> float:
    """前 K 个推荐里至少命中一个相关物品的用户比例。

    在 leave-one-out（每个用户只留一个测试物品）的协议下，每个用户的相关
    物品只有一个，因此 HitRate@K 与 Recall@K 的数值完全相同。
    """
    _validate_inputs(ground_truth, k)
    hits = 0
    for user_idx, relevant_items in ground_truth.items():
        recommended_items = recommendations.get(user_idx, ())[:k]
        if set(recommended_items).intersection(relevant_items):
            hits += 1
    return hits / len(ground_truth)


def ndcg_at_k(
    recommendations: Recommendations, ground_truth: GroundTruth, k: int
) -> float:
    """二值相关性下，归一化折损累计增益（NDCG）的平均值。"""
    _validate_inputs(ground_truth, k)
    user_scores = []

    for user_idx, relevant_items in ground_truth.items():
        recommended_items = recommendations.get(user_idx, ())[:k]
        # DCG 只累加「前 K 名里真正命中的位置」的折损增益。
        dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank, item_idx in enumerate(recommended_items)
            if item_idx in relevant_items
        )
        # 理想排序会把所有相关物品尽量排在前面。
        # min(len(relevant_items), k) 表示其中最多还能放进前 K 名的个数。
        ideal_hits = min(len(relevant_items), k)
        ideal_dcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
        user_scores.append(dcg / ideal_dcg)

    return sum(user_scores) / len(user_scores)


def precision_at_k(
    recommendations: Recommendations, ground_truth: GroundTruth, k: int
) -> float:
    """前 K 个推荐中相关物品的平均比例（分母固定为 K）。

    与 Recall 的区别：Recall 的分母是该用户的相关物品数，Precision 的分母是 K。
    两者都保留，是因为在候选集大小不同的实验里它们会给出不同结论。
    """
    _validate_inputs(ground_truth, k)
    precisions = []
    for user_idx, relevant_items in ground_truth.items():
        recommended_items = recommendations.get(user_idx, ())[:k]
        hits = len(set(recommended_items).intersection(relevant_items))
        precisions.append(hits / k)
    return sum(precisions) / len(precisions)


def catalog_coverage(recommendations: Recommendations, num_items: int) -> float:
    """推荐结果覆盖到的物品占全部物品的比例。"""
    if num_items <= 0:
        raise ValueError("num_items must be positive")
    if not recommendations:
        raise ValueError("recommendations must contain at least one user")
    recommended = {item for items in recommendations.values() for item in items}
    return len(recommended) / num_items


def compute_ranking_metrics(
    recommendations: Recommendations,
    ground_truth: GroundTruth,
    ks: Sequence[int] = (10, 20),
    num_items: int | None = None,
) -> dict[str, float]:
    """对每个截断位置 K 计算全部排序指标。

    ``num_items`` 给了才会附带 ``Coverage``：覆盖率是全局指标，不是按 K 平均。
    """
    if not ks:
        raise ValueError("ks must contain at least one cutoff")

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"Recall@{k}"] = recall_at_k(recommendations, ground_truth, k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(recommendations, ground_truth, k)
        metrics[f"HitRate@{k}"] = hit_rate_at_k(
            recommendations, ground_truth, k
        )
        metrics[f"Precision@{k}"] = precision_at_k(
            recommendations, ground_truth, k
        )
    if num_items is not None:
        metrics["Coverage"] = catalog_coverage(recommendations, num_items)
    return metrics


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """ROC-AUC（Mann-Whitney U 形式），不依赖 sklearn。"""
    _validate_binary_inputs(scores, labels)
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0])
    rank_sum = 0.0
    positive_count = 0
    index = 0
    # 同一分数共享平均秩，保证并列时 AUC 不被顺序影响。
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end + 1) / 2
        for position in range(index, end + 1):
            if pairs[position][1] == 1:
                rank_sum += average_rank
                positive_count += 1
        index = end + 1
    negative_count = len(pairs) - positive_count
    if positive_count == 0:
        raise ValueError("auc needs at least one positive label")
    if negative_count == 0:
        raise ValueError("auc needs at least one negative label")
    return (rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def log_loss(scores: Sequence[float], labels: Sequence[int], eps: float = 1e-15) -> float:
    """二分类 LogLoss；分数按概率处理，先裁剪到 [eps, 1-eps] 防止 log(0)。"""
    _validate_binary_inputs(scores, labels)
    total = 0.0
    for score, label in zip(scores, labels, strict=True):
        probability = min(max(float(score), eps), 1.0 - eps)
        total -= label * math.log(probability) + (1 - label) * math.log(1.0 - probability)
    return total / len(labels)


def _validate_binary_inputs(scores: Sequence[float], labels: Sequence[int]) -> None:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if not scores:
        raise ValueError("scores must not be empty")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("labels must be 0 or 1")
    if any(not 0.0 <= float(score) <= 1.0 for score in scores):
        raise ValueError("scores must be probabilities in [0, 1]")


def compute_ranking_model_metrics(
    scores: Sequence[float], labels: Sequence[int]
) -> dict[str, float]:
    """排序/CTR 模型的分类侧指标（AUC、LogLoss）。"""
    return {"AUC": roc_auc(scores, labels), "LogLoss": log_loss(scores, labels)}
