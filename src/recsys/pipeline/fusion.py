"""候选去重与融合：轮流、RRF、配额三种策略，全部保留来源与通道内名次。"""

from __future__ import annotations

from dataclasses import dataclass, field

from recsys.pipeline.config import FusionConfig
from recsys.pipeline.recall import Candidate


@dataclass
class FusedCandidate:
    item: int
    fusion_rank: int
    sources: dict[str, dict] = field(default_factory=dict)
    fused_score: float | None = None

    def as_dict(self) -> dict:
        return {
            "item": self.item,
            "fusion_rank": self.fusion_rank,
            "sources": self.sources,
            "fused_score": self.fused_score,
        }


def _source(name: str, candidate: Candidate, weight: float = 1.0) -> dict:
    return {
        "rank": candidate.rank,
        "score": candidate.score,
        "weight": float(weight),
    }


def round_robin_fusion(
    channels: dict[str, dict[int, list[Candidate]]],
    *,
    limit: int,
    weights: dict[str, float] | None = None,
) -> dict[int, list[FusedCandidate]]:
    weights = weights or {}
    users = sorted({user for mapping in channels.values() for user in mapping})
    fused: dict[int, list[FusedCandidate]] = {}
    for user in users:
        rows = {name: mapping.get(user, []) for name, mapping in channels.items()}
        seen: set[int] = set()
        pool: list[FusedCandidate] = []
        rank = 0
        while len(pool) < limit and any(rank < len(values) for values in rows.values()):
            for name, values in rows.items():
                if len(pool) >= limit:
                    break
                if rank >= len(values):
                    continue
                candidate = values[rank]
                if candidate.item in seen:
                    continue
                seen.add(candidate.item)
                pool.append(
                    FusedCandidate(
                        item=candidate.item,
                        fusion_rank=len(pool) + 1,
                        sources={name: _source(name, candidate, weights.get(name, 1.0))},
                    )
                )
            rank += 1
        fused[user] = pool
    return fused


def rrf_fusion(
    channels: dict[str, dict[int, list[Candidate]]],
    *,
    limit: int,
    rrf_k: int = 60,
    weights: dict[str, float] | None = None,
) -> dict[int, list[FusedCandidate]]:
    weights = weights or {}
    users = sorted({user for mapping in channels.values() for user in mapping})
    fused: dict[int, list[FusedCandidate]] = {}
    for user in users:
        scores: dict[int, float] = {}
        sources: dict[int, dict[str, dict]] = {}
        first_rank: dict[int, int] = {}
        for name, mapping in channels.items():
            for candidate in mapping.get(user, []):
                weight = float(weights.get(name, 1.0))
                scores[candidate.item] = scores.get(candidate.item, 0.0) + weight / (
                    rrf_k + candidate.rank
                )
                sources.setdefault(candidate.item, {})[name] = _source(name, candidate, weight)
                first_rank.setdefault(candidate.item, candidate.rank)
        ordered = sorted(scores, key=lambda item: (-scores[item], first_rank.get(item, 0), item))[
            :limit
        ]
        fused[user] = [
            FusedCandidate(
                item=item,
                fusion_rank=rank + 1,
                sources=sources[item],
                fused_score=round(scores[item], 8),
            )
            for rank, item in enumerate(ordered)
        ]
    return fused


def quota_fusion(
    channels: dict[str, dict[int, list[Candidate]]],
    quotas: dict[str, int],
    *,
    limit: int,
) -> dict[int, list[FusedCandidate]]:
    """先按配额取每路前 N，再轮流补位；配额必须覆盖所有通道名。"""
    missing = sorted(set(channels) - set(quotas))
    if missing:
        raise ValueError(f"配额缺少通道 {missing}")
    users = sorted({user for mapping in channels.values() for user in mapping})
    fill_pool = round_robin_fusion(channels, limit=limit * 2)
    fused: dict[int, list[FusedCandidate]] = {}
    for user in users:
        seen: set[int] = set()
        pool: list[FusedCandidate] = []
        for name, mapping in channels.items():
            values = mapping.get(user, [])[: max(0, int(quotas[name]))]
            for candidate in values:
                if candidate.item in seen or len(pool) >= limit:
                    continue
                seen.add(candidate.item)
                pool.append(
                    FusedCandidate(
                        item=candidate.item,
                        fusion_rank=len(pool) + 1,
                        sources={name: _source(name, candidate)},
                    )
                )
        if len(pool) < limit:
            for candidate in fill_pool.get(user, []):
                if len(pool) >= limit:
                    break
                if candidate.item in seen:
                    continue
                seen.add(candidate.item)
                pool.append(
                    FusedCandidate(
                        item=candidate.item,
                        fusion_rank=len(pool) + 1,
                        sources=candidate.sources,
                    )
                )
        fused[user] = pool
    return fused


def fuse_channels(
    channels: dict[str, dict[int, list[Candidate]]],
    config: FusionConfig,
    *,
    weights: dict[str, float] | None = None,
) -> dict[int, list[FusedCandidate]]:
    mode = config.mode
    if mode == "round_robin":
        return round_robin_fusion(channels, limit=config.limit, weights=weights)
    if mode == "rrf":
        return rrf_fusion(
            channels, limit=config.limit, rrf_k=config.rrf_k, weights=weights
        )
    if mode == "quota":
        return quota_fusion(channels, config.quota, limit=config.limit)
    raise ValueError(f"未知融合模式 {mode!r}（可选 round_robin / rrf / quota）")


__all__ = [
    "FusedCandidate",
    "fuse_channels",
    "quota_fusion",
    "rrf_fusion",
    "round_robin_fusion",
]
