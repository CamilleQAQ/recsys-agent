"""候选打分：统一接入 LightGBM LambdaRank 与神经排序模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from recsys.adapters.base import RankingAdapter
from recsys.adapters.ranking_registry import load_ranker
from recsys.data.schema import PreparedData
from recsys.pipeline.config import PipelineConfig
from recsys.pipeline.fusion import FusedCandidate
from recsys.ranking.features import FeatureContext
from recsys.utils.artifact_guard import ArtifactError, require_file


@dataclass
class RankedCandidate:
    item: int
    score: float
    rank: int
    sources: dict[str, dict] = field(default_factory=dict)
    fusion_rank: int = 0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "item": self.item,
            "score": self.score,
            "rank": self.rank,
            "fusion_rank": self.fusion_rank,
            "sources": self.sources,
            **self.extra,
        }


def rank_fused(
    ranker: RankingAdapter,
    data: PreparedData,
    fused: dict[int, list[FusedCandidate]],
    *,
    split: str,
    request_timestamps: dict[int, int] | None = None,
    include_validation_in_test_history: bool = True,
) -> dict[int, list[RankedCandidate]]:
    users = sorted(fused)
    items_by_user = {
        user: [candidate.item for candidate in fused[user]] for user in users
    }
    scores_by_user: dict[int, dict[int, float]] = {}
    if hasattr(ranker, "score_candidates"):
        scores_by_user = ranker.score_candidates(  # type: ignore[attr-defined]
            data,
            users,
            items_by_user,
            split=split,
            request_timestamps=request_timestamps,
            include_validation_in_test_history=include_validation_in_test_history,
        )
    else:
        context = FeatureContext(data)
        for user in users:
            items = np.asarray(items_by_user[user], dtype=np.int64)
            if items.size == 0:
                scores_by_user[user] = {}
                continue
            features = context.lightgbm_features(items)
            predictions = ranker.predict(features)
            scores_by_user[user] = {
                int(item): float(score)
                for item, score in zip(items.tolist(), predictions.tolist())
            }

    ranked: dict[int, list[RankedCandidate]] = {}
    for user in users:
        pool = fused[user]
        scores = scores_by_user.get(user, {})
        ordered = sorted(
            pool,
            key=lambda candidate: (
                -float(scores.get(candidate.item, float("-inf"))),
                candidate.fusion_rank,
            ),
        )
        ranked[user] = [
            RankedCandidate(
                item=candidate.item,
                score=float(scores.get(candidate.item, 0.0)),
                rank=position + 1,
                sources=candidate.sources,
                fusion_rank=candidate.fusion_rank,
            )
            for position, candidate in enumerate(ordered)
        ]
    return ranked


def train_lightgbm(
    data: PreparedData,
    users: list[int],
    *,
    checkpoint: Path,
    max_rows: int,
    num_boost_round: int,
    parent_ranking: Path | None,
    params: dict | None = None,
) -> Path:
    """复用 MVP 的排序样本构造与特征口径训练 LightGBM LambdaRank。"""
    from recsys.adapters.ranking import LightGBMRankerAdapter
    from recsys.experiments.recsys_mvp import build_features, build_ranking_train

    parent = parent_ranking or Path("data/kuasar/processed/ranking")
    view = build_ranking_train(data, np.asarray(users, dtype=np.int64), max_rows, parent)
    features = build_features(data, view["items"])
    ranker = LightGBMRankerAdapter(params=params or {})
    ranker.fit(
        features,
        view["labels"],
        groups=view["groups"],
        num_boost_round=num_boost_round,
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    ranker.save(checkpoint)
    return checkpoint


def load_or_train_ranker(
    config: PipelineConfig,
    data: PreparedData,
    users: list[int],
) -> RankingAdapter | None:
    """加载排序模型；``train_if_missing=False`` 时缺失即失败，不会顺手训练。"""

    ranking = config.ranking
    if ranking.model in ("", "none"):
        return None
    checkpoint = ranking.checkpoint
    if checkpoint is None:
        checkpoint = (
            config.run.artifacts_dir
            / config.run.dataset
            / f"{ranking.model}_pipeline.txt"
        )
    checkpoint = Path(checkpoint)
    exists = checkpoint.exists()
    need_train = ranking.train_if_missing and (
        ranking.model == "lightgbm_ranker" and not exists
    )
    if need_train:
        train_lightgbm(
            data,
            users,
            checkpoint=checkpoint,
            max_rows=ranking.max_rows,
            num_boost_round=ranking.num_boost_round,
            parent_ranking=ranking.parent_ranking,
            params=ranking.params,
        )
    elif not exists:
        # Phase 5：只加载已有权重。这里必须显式失败，不能训练、换模型或用随机权重。
        raise ArtifactError(
            f"排序权重缺失：{checkpoint}（model={ranking.model}）。"
            "当前配置 train_if_missing=False，只允许加载已有权重；"
            "不会自动训练、不会换模型、不会改用随机权重。显式训练请用 "
            "python -m recsys.experiments.rec_agent --train lightgbm_ranker",
            path=checkpoint,
            role="ranking_checkpoint",
        )
    require_file(checkpoint, role=f"排序权重({ranking.model})")
    return load_ranker(ranking.model, checkpoint, config.run.device)


__all__ = ["RankedCandidate", "load_or_train_ranker", "rank_fused", "train_lightgbm"]
