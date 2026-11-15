"""统一推荐链路：多路召回 → 去重融合 → 候选打分 → 类目/多样性重排 → 离线评估。"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from recsys.adapters.base import AdapterError
from recsys.adapters.registry import load_adapter
from recsys.data.loader import prepare_dataset, resolve_positive_threshold
from recsys.data.schema import PreparedData, seen_items_for_split
from recsys.evaluation.evaluator import evaluate_recommendations, group_items_by_user
from recsys.evaluation.metrics import catalog_coverage
from recsys.pipeline.config import PipelineConfig
from recsys.pipeline.fusion import FusedCandidate, fuse_channels
from recsys.pipeline.ranking import RankedCandidate, load_or_train_ranker, rank_fused
from recsys.pipeline.recall import Candidate, run_channel
from recsys.pipeline.rerank import diversify
from recsys.ranking.samples import target_request_timestamps
from recsys.utils.artifact_guard import require_file
from recsys.utils.provenance import build_provenance


@dataclass
class PipelineResult:
    config: PipelineConfig
    split: str
    users: list[int]
    channels: dict[str, dict[int, list[Candidate]]]
    fused: dict[int, list[FusedCandidate]]
    ranked: dict[int, list[RankedCandidate]]
    final: dict[int, list[RankedCandidate]]
    metrics: dict[str, dict[str, float]]
    diagnostics: dict
    timings: dict[str, float] = field(default_factory=dict)
    relaxations: int = 0
    output_dir: Path | None = None

    @property
    def final_items(self) -> dict[int, list[int]]:
        return {user: [candidate.item for candidate in rows] for user, rows in self.final.items()}

    def summary(self) -> dict:
        return {
            "dataset": self.config.run.dataset,
            "split": self.split,
            "users": len(self.users),
            "channels": list(self.channels),
            "fusion": self.config.fusion.mode,
            "ranker": self.config.ranking.model,
            "k_final": self.config.run.k_final,
            "k_recall": self.config.run.k_recall,
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
            "timings": self.timings,
            "relaxations": self.relaxations,
            "output_dir": str(self.output_dir) if self.output_dir else None,
        }


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_users(
    data: PreparedData, count: int, seed: int, split: str
) -> list[int]:
    interactions = data.validation_interactions if split == "validation" else data.test_interactions
    candidates = sorted({row.user_idx for row in interactions})
    if not candidates:
        raise ValueError(f"{split} 切分没有可评测用户")
    if count >= len(candidates):
        return candidates
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(candidates), size=count, replace=False)
    return sorted(int(user) for user in chosen)


def metrics_for(
    recommendations: dict[int, list[int]], interactions, ks: tuple[int, ...]
) -> dict[str, float]:
    metrics = evaluate_recommendations(recommendations, interactions, ks=ks)
    return {key: round(float(value), 6) for key, value in metrics.items()}


class RecommendationPipeline:
    """配置驱动的一次完整推荐运行。"""

    def __init__(self, config: PipelineConfig, data: PreparedData | None = None) -> None:
        self.config = config
        self._data = data
        self._adapters: dict[str, object] = {}
        self.result: PipelineResult | None = None

    @property
    def data(self) -> PreparedData:
        if self._data is None:
            self._data = prepare_dataset(
                dataset=self.config.run.dataset,
                data_dir=self.config.run.data_dir,
                positive_threshold=resolve_positive_threshold(self.config.run.dataset, None),
            )
        return self._data

    def _users(self, split: str) -> list[int]:
        run = self.config.run
        data = self.data
        if run.user_ids:
            missing = [raw for raw in run.user_ids if raw not in data.mappings.user_to_index]
            if missing:
                raise ValueError(f"配置里的 user_ids 不在词表中：{missing}")
            return [int(data.mappings.user_to_index[raw]) for raw in run.user_ids]
        return select_users(data, run.users, run.seed, split)

    def _load_channel_adapter(self, channel) -> object:
        if channel.name in self._adapters:
            return self._adapters[channel.name]
        # 召回权重同样只加载已有文件：缺失时直接报出路径，不静默换模型。
        if channel.checkpoint is not None:
            require_file(channel.checkpoint, role=f"召回权重({channel.model})")
        adapter = load_adapter(
            channel.model,
            channel.checkpoint,
            device_name=self.config.run.device,
        )
        self._adapters[channel.name] = adapter
        return adapter

    def run(
        self,
        *,
        split: str | None = None,
        users: list[int] | None = None,
        write: bool = True,
        output_dir: Path | None = None,
    ) -> PipelineResult:
        config = self.config
        split = split or config.run.split
        if split not in {"validation", "test"}:
            raise ValueError("split 必须是 validation 或 test")
        started = time.perf_counter()
        timings: dict[str, float] = {}
        data = self.data
        users = users or self._users(split)
        users = [int(user) for user in users]
        history_setting = config.run.include_validation_in_test_history
        request_timestamps = target_request_timestamps(data, split)

        channels: dict[str, dict[int, list[Candidate]]] = {}
        channel_meta: dict[str, dict] = {}
        for channel in config.recall:
            stage = time.perf_counter()
            adapter = self._load_channel_adapter(channel)
            k = channel.k or config.run.k_recall
            channels[channel.name] = run_channel(
                adapter,
                data,
                users,
                split=split,
                k=k,
                include_validation_in_test_history=history_setting,
            )
            timings[f"recall_{channel.name}_s"] = round(time.perf_counter() - stage, 2)
            channel_meta[channel.name] = {
                "model": channel.model,
                "checkpoint": str(channel.checkpoint) if channel.checkpoint else None,
                "k": k,
                "weight": channel.weight,
            }

        stage = time.perf_counter()
        fused = fuse_channels(
            channels,
            config.fusion,
            weights={channel.name: channel.weight for channel in config.recall},
        )
        timings["fusion_s"] = round(time.perf_counter() - stage, 2)

        stage = time.perf_counter()
        ranker = load_or_train_ranker(config, data, users)
        ranked = (
            rank_fused(
                ranker,
                data,
                fused,
                split=split,
                request_timestamps=request_timestamps,
                include_validation_in_test_history=history_setting,
            )
            if ranker is not None
            else {
                user: [
                    RankedCandidate(
                        item=candidate.item,
                        score=float(candidate.fused_score or 0.0),
                        rank=candidate.fusion_rank,
                        sources=candidate.sources,
                        fusion_rank=candidate.fusion_rank,
                    )
                    for candidate in fused[user]
                ]
                for user in fused
            }
        )
        timings["ranking_s"] = round(time.perf_counter() - stage, 2)

        stage = time.perf_counter()
        categories = (
            np.asarray(data.item_features.column("category_first_code"))
            if data.item_features is not None
            else np.asarray([], dtype=np.int32)
        )
        authors = (
            np.asarray(data.item_features.column("author_id"))
            if data.item_features is not None and config.rerank.author_cap > 0
            else None
        )
        final: dict[int, list[RankedCandidate]] = {}
        relaxations = 0
        for user in sorted(ranked):
            picked, relaxed = diversify(
                ranked[user],
                categories=categories,
                authors=authors,
                k=config.run.k_final,
                category_cap=config.rerank.category_cap,
                author_cap=config.rerank.author_cap if config.rerank.diversity else 0,
            )
            final[user] = picked
            relaxations += relaxed
        timings["rerank_s"] = round(time.perf_counter() - stage, 2)
        timings["total_s"] = round(time.perf_counter() - started, 2)

        interactions = (
            data.validation_interactions if split == "validation" else data.test_interactions
        )
        user_set = set(users)
        targets = [row for row in interactions if row.user_idx in user_set]
        target_by_user = group_items_by_user(targets)
        stages: dict[str, dict[int, list[int]]] = {}
        for name, mapping in channels.items():
            stages[f"recall_{name}"] = {
                user: [candidate.item for candidate in mapping.get(user, [])[: config.run.k_final]]
                for user in users
            }
        stages["fusion"] = {
            user: [candidate.item for candidate in fused.get(user, [])[: config.run.k_final]]
            for user in users
        }
        stages["ranked"] = {
            user: [candidate.item for candidate in ranked.get(user, [])[: config.run.k_final]]
            for user in users
        }
        stages["rerank"] = {
            user: [candidate.item for candidate in final.get(user, [])]
            for user in users
        }
        ks = (config.run.k_final,)
        metrics = {
            name: {
                **metrics_for(recommendations, targets, ks),
                "Coverage": round(
                    float(catalog_coverage(recommendations, data.num_items)), 6
                ),
            }
            for name, recommendations in stages.items()
        }

        # ---- 诊断：目标物品在哪个环节丢失（Top-20 零命中的直接证据）
        union_at_k = {
            user: {candidate.item for mapping in channels.values() for candidate in mapping.get(user, [])}
            for user in users
        }
        loss_counts = {
            "not_in_any_channel_top_k": 0,
            "in_union_not_in_fusion_pool": 0,
            "in_fusion_pool_not_top20": 0,
            "in_fusion_top20_not_ranked_top20": 0,
            "in_ranked_top20_not_final": 0,
            "survived": 0,
        }
        target_details: list[dict] = []
        for user in users:
            fused_pool = [candidate.item for candidate in fused.get(user, [])[: config.run.k_recall]]
            fused_items = [candidate.item for candidate in fused.get(user, [])[: config.run.k_final]]
            ranked_items = [candidate.item for candidate in ranked.get(user, [])[: config.run.k_final]]
            final_items = [candidate.item for candidate in final.get(user, [])]
            for item in sorted(target_by_user.get(user, set())):
                channel_ranks = {}
                for name, mapping in channels.items():
                    for candidate in mapping.get(user, []):
                        if candidate.item == item:
                            channel_ranks[name] = candidate.rank
                            break
                if item not in union_at_k.get(user, set()):
                    stage = "not_in_any_channel_top_k"
                elif item not in fused_pool:
                    stage = "in_union_not_in_fusion_pool"
                elif item not in fused_items:
                    stage = "in_fusion_pool_not_top20"
                elif item not in ranked_items:
                    stage = "in_fusion_top20_not_ranked_top20"
                elif item not in final_items:
                    stage = "in_ranked_top20_not_final"
                else:
                    stage = "survived"
                loss_counts[stage] += 1
                target_details.append(
                    {
                        "user": int(data.mappings.index_to_user[user]),
                        "item": int(data.mappings.index_to_item[item]),
                        "stage": stage,
                        "channel_ranks": channel_ranks,
                        "fusion_rank": next(
                            (
                                candidate.fusion_rank
                                for candidate in fused.get(user, [])
                                if candidate.item == item
                            ),
                            None,
                        ),
                        "ranked_rank": next(
                            (
                                candidate.rank
                                for candidate in ranked.get(user, [])
                                if candidate.item == item
                            ),
                            None,
                        ),
                        "final_rank": (
                            final_items.index(item) + 1 if item in final_items else None
                        ),
                    }
                )
        diagnostics = {
            "users": len(users),
            "targets": len(targets),
            "targets_by_stage": loss_counts,
            "targets_per_user": round(len(targets) / max(1, len(users)), 3),
            "channel_hits_at_k": {
                name: sum(
                    len(
                        {candidate.item for candidate in mapping.get(user, [])[: config.run.k_recall]}
                        & target_by_user.get(user, set())
                    )
                    for user in users
                )
                for name, mapping in channels.items()
            },
            "union_hits_at_k_recall": sum(
                len(union_at_k.get(user, set()) & target_by_user.get(user, set()))
                for user in users
            ),
            "target_details": target_details,
            "note": (
                "top-k 指标基于当前 split 的留出目标；targets_by_stage 逐目标记录最先丢失的环节。"
            ),
        }

        output_dir = Path(output_dir) if output_dir else (
            config.run.artifacts_dir / config.run.dataset / config.run.output_name
        )
        result = PipelineResult(
            config=config,
            split=split,
            users=users,
            channels=channels,
            fused=fused,
            ranked=ranked,
            final=final,
            metrics=metrics,
            diagnostics=diagnostics,
            timings=timings,
            relaxations=relaxations,
            output_dir=output_dir,
        )
        self.result = result
        if write:
            self.write_artifacts(result, channel_meta, targets)
        return result

    def write_artifacts(
        self,
        result: PipelineResult,
        channel_meta: dict[str, dict],
        targets: list,
    ) -> None:
        data = self.data
        output_dir = result.output_dir or Path("artifacts")
        output_dir.mkdir(parents=True, exist_ok=True)

        def item_view(item: int) -> dict:
            return {
                "item": int(data.mappings.index_to_item[item]),
                "item_idx": int(item),
            }

        _json_dump(
            output_dir / "final.json",
            {
                str(user): {
                    "user": int(data.mappings.index_to_user[user]),
                    "items": [
                        {
                            **item_view(candidate.item),
                            "rank": rank + 1,
                            "score": round(float(candidate.score), 6),
                            "fusion_rank": candidate.fusion_rank,
                            "sources": candidate.sources,
                        }
                        for rank, candidate in enumerate(result.final.get(user, []))
                    ],
                }
                for user in result.users
            },
        )
        for name, mapping in result.channels.items():
            _json_dump(
                output_dir / f"recall_{name}.json",
                {
                    str(user): [
                        {
                            **item_view(candidate.item),
                            "rank": candidate.rank,
                            "score": None if candidate.score is None else round(candidate.score, 6),
                        }
                        for candidate in mapping.get(user, [])
                    ]
                    for user in result.users
                },
            )
        _json_dump(
            output_dir / "fusion.json",
            {
                str(user): [
                    {
                        **item_view(candidate.item),
                        "fusion_rank": candidate.fusion_rank,
                        "fused_score": candidate.fused_score,
                        "sources": candidate.sources,
                    }
                    for candidate in result.fused.get(user, [])
                ]
                for user in result.users
            },
        )
        _json_dump(
            output_dir / "ranked.json",
            {
                str(user): [
                    {
                        **item_view(candidate.item),
                        "rank": candidate.rank,
                        "score": round(float(candidate.score), 6),
                        "fusion_rank": candidate.fusion_rank,
                        "sources": candidate.sources,
                    }
                    for candidate in result.ranked.get(user, [])
                ]
                for user in result.users
            },
        )
        _json_dump(
            output_dir / "metrics.json",
            {"split": result.split, "k_final": result.config.run.k_final, "stages": result.metrics},
        )
        _json_dump(output_dir / "diagnostics.json", result.diagnostics)

        provenance = build_provenance(
            data,
            dataset=result.config.run.dataset,
            positive_threshold=resolve_positive_threshold(result.config.run.dataset, None),
        )
        checkpoints = {}
        for channel in result.config.recall:
            if channel.checkpoint and Path(channel.checkpoint).exists():
                checkpoints[channel.name] = {
                    "path": str(channel.checkpoint),
                    "sha256": _sha256_file(Path(channel.checkpoint)),
                }
        ranking_checkpoint = result.config.ranking.checkpoint
        if ranking_checkpoint and Path(ranking_checkpoint).exists():
            checkpoints["ranker"] = {
                "path": str(ranking_checkpoint),
                "sha256": _sha256_file(Path(ranking_checkpoint)),
            }
        run_meta = {
            "config_path": str(result.config.source_path) if result.config.source_path else None,
            "config": {
                "run": {
                    "dataset": result.config.run.dataset,
                    "split": result.split,
                    "seed": result.config.run.seed,
                    "k_recall": result.config.run.k_recall,
                    "k_final": result.config.run.k_final,
                    "include_validation_in_test_history": (
                        result.config.run.include_validation_in_test_history
                    ),
                },
                "recall": channel_meta,
                "fusion": {
                    "mode": result.config.fusion.mode,
                    "limit": result.config.fusion.limit,
                    "rrf_k": result.config.fusion.rrf_k,
                    "quota": result.config.fusion.quota,
                },
                "ranking": {
                    "model": result.config.ranking.model,
                    "num_boost_round": result.config.ranking.num_boost_round,
                    "max_rows": result.config.ranking.max_rows,
                },
                "rerank": {
                    "category_cap": result.config.rerank.category_cap,
                    "author_cap": result.config.rerank.author_cap,
                    "diversity": result.config.rerank.diversity,
                },
            },
            "users": [int(data.mappings.index_to_user[user]) for user in result.users],
            "timings": result.timings,
            "relaxations": result.relaxations,
            "provenance": provenance,
            "checkpoints": checkpoints,
            "metrics": result.metrics,
            "diagnostics": {
                key: value
                for key, value in result.diagnostics.items()
                if key != "target_details"
            },
            "assertions": {
                "no_duplicate_candidates": all(
                    len([candidate.item for candidate in result.fused[user]])
                    == len({candidate.item for candidate in result.fused[user]})
                    for user in result.users
                ),
                "no_seen_leak": _no_seen_leak(result, data),
                "ranking_within_fusion": all(
                    {candidate.item for candidate in result.ranked[user]}
                    <= {candidate.item for candidate in result.fused[user]}
                    for user in result.users
                ),
                "rerank_within_ranked": all(
                    {candidate.item for candidate in result.final[user]}
                    <= {candidate.item for candidate in result.ranked[user]}
                    for user in result.users
                ),
            },
        }
        _json_dump(output_dir / "run_meta.json", run_meta)


def _no_seen_leak(result: PipelineResult, data: PreparedData) -> bool:
    seen = seen_items_for_split(data, result.split)
    return all(
        all(
            candidate.item not in seen.get(user, set())
            for candidate in result.final.get(user, [])
        )
        for user in result.users
    )


def load_result(path: Path) -> dict:
    """读取一次运行的最终结果（``final.json`` + ``metrics.json`` + ``run_meta.json``）。"""
    path = Path(path)
    payload = {}
    for name in ("final", "metrics", "run_meta", "diagnostics"):
        file = path / f"{name}.json"
        if file.exists():
            payload[name] = json.loads(file.read_text(encoding="utf-8"))
    if not payload:
        raise FileNotFoundError(f"{path} 下没有可读取的运行工件")
    return payload


__all__ = [
    "PipelineResult",
    "RecommendationPipeline",
    "load_result",
    "metrics_for",
    "select_users",
]
