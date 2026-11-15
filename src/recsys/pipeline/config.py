"""推荐链路的统一配置对象（TOML → dataclass）。"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    dataset: str = "kuasar_small"
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    output_name: str = "pipeline"
    split: str = "validation"
    users: int = 128
    user_ids: tuple[int, ...] = ()
    seed: int = 42
    k_recall: int = 100
    k_final: int = 20
    include_validation_in_test_history: bool = True
    device: str = "cpu"


@dataclass(frozen=True)
class RecallChannelConfig:
    name: str
    model: str
    checkpoint: Path | None = None
    k: int | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class FusionConfig:
    mode: str = "round_robin"  # round_robin | rrf | quota
    limit: int = 100
    rrf_k: int = 60
    quota: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RankingConfig:
    model: str = "lightgbm_ranker"
    checkpoint: Path | None = None
    train_if_missing: bool = False
    num_boost_round: int = 20
    max_rows: int = 10_000
    parent_ranking: Path | None = None
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RerankConfig:
    category_cap: int = 5
    author_cap: int = 0
    diversity: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    run: RunConfig
    recall: tuple[RecallChannelConfig, ...]
    fusion: FusionConfig
    ranking: RankingConfig
    rerank: RerankConfig
    source_path: Path | None = None


def _path(value) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def load_pipeline_config(path: Path) -> PipelineConfig:
    with Path(path).open("rb") as file:
        raw = tomllib.load(file)
    if "run" not in raw or "recall" not in raw:
        raise ValueError(f"{path} 缺少 [run] 或 [recall] 配置")

    run_raw = raw["run"]
    run = RunConfig(
        dataset=str(run_raw.get("dataset", "kuasar_small")),
        data_dir=Path(run_raw.get("data_dir", "data")),
        artifacts_dir=Path(run_raw.get("artifacts_dir", "artifacts")),
        output_name=str(run_raw.get("output_name", "pipeline")),
        split=str(run_raw.get("split", "validation")),
        users=int(run_raw.get("users", 128)),
        user_ids=tuple(int(value) for value in run_raw.get("user_ids", ())),
        seed=int(run_raw.get("seed", 42)),
        k_recall=int(run_raw.get("k_recall", 100)),
        k_final=int(run_raw.get("k_final", 20)),
        include_validation_in_test_history=bool(
            run_raw.get("include_validation_in_test_history", True)
        ),
        device=str(run_raw.get("device", "cpu")),
    )

    recall_section = raw["recall"]
    channels: list[RecallChannelConfig] = []
    for name, values in recall_section.items():
        if not isinstance(values, dict) or "model" not in values:
            raise ValueError(f"[recall.{name}] 必须包含 model 字段")
        channels.append(
            RecallChannelConfig(
                name=str(name),
                model=str(values["model"]),
                checkpoint=_path(values.get("checkpoint")),
                k=int(values["k"]) if values.get("k") is not None else None,
                weight=float(values.get("weight", 1.0)),
            )
        )
    if not channels:
        raise ValueError("至少配置一路召回")

    fusion_raw = raw.get("fusion", {})
    fusion = FusionConfig(
        mode=str(fusion_raw.get("mode", "round_robin")),
        limit=int(fusion_raw.get("limit", run.k_recall)),
        rrf_k=int(fusion_raw.get("rrf_k", 60)),
        quota={str(key): int(value) for key, value in (fusion_raw.get("quota") or {}).items()},
    )

    ranking_raw = raw.get("ranking", {})
    ranking = RankingConfig(
        model=str(ranking_raw.get("model", "lightgbm_ranker")),
        checkpoint=_path(ranking_raw.get("checkpoint")),
        train_if_missing=bool(ranking_raw.get("train_if_missing", False)),
        num_boost_round=int(ranking_raw.get("num_boost_round", 20)),
        max_rows=int(ranking_raw.get("max_rows", 10_000)),
        parent_ranking=_path(ranking_raw.get("parent_ranking")),
        params=dict(ranking_raw.get("params") or {}),
    )

    rerank_raw = raw.get("rerank", {})
    rerank = RerankConfig(
        category_cap=int(rerank_raw.get("category_cap", 5)),
        author_cap=int(rerank_raw.get("author_cap", 0)),
        diversity=bool(rerank_raw.get("diversity", True)),
    )

    return PipelineConfig(
        run=run,
        recall=tuple(channels),
        fusion=fusion,
        ranking=ranking,
        rerank=rerank,
        source_path=Path(path),
    )


__all__ = [
    "FusionConfig",
    "PipelineConfig",
    "RankingConfig",
    "RecallChannelConfig",
    "RerankConfig",
    "RunConfig",
    "load_pipeline_config",
]
