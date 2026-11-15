"""Ads Agent 的工具注册表：工具只做参数校验与调度，业务全部在 ``service``。

结构沿用 YieldAgent ``agents/console/tools.py`` 的写法：

* 每个工具是一个薄函数，docstring 就是给模型的工具说明；
* 工具按统一 schema 注册，模型只能调用注册过的工具；
* 工具不 import 业务实现细节，只调用业务层（``recsys.ads.service``），
  与 YieldAgent 通过 ``Connector`` 隔离平台逻辑是同一结构；
* 所有工具返回结构化结果（``ok`` + 指标/摘要），失败由注册表统一转成
  ``ok=false`` + 具体原因，绝不把失败写成成功。

预算/时间/重复调用由 ``recsys.experiments.rec_tools.ToolRegistry`` /
``ToolBudget`` 统一执行（本项目已有实现，直接复用）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recsys.ads import AD_MODEL_NAMES
from recsys.ads.config import AdsSimConfig
from recsys.ads.runtime import AdsRuntime
from recsys.ads.service import (
    label_availability,
    list_models,
    predict_ad_ctr,
    predict_multitask,
    rank_ad_candidates,
    simulate_ad_auction,
)
from recsys.config import DEFAULT_ARTIFACTS_DIR
from recsys.experiments.rec_tools import (
    ToolBudget,
    ToolCallError,
    ToolRegistry,
    ToolSpec,
)

SPLITS = ("train", "validation", "test")
MULTITASK_MODELS = ("mmoe", "ple")


@dataclass
class AdsToolContext:
    """一次 Ads Agent 会话的共享状态（数据、模型、预算、轨迹）。"""

    runtime: AdsRuntime
    config: AdsSimConfig
    experiment_dir: Path | None = None
    budget: ToolBudget = field(default_factory=ToolBudget)
    state: dict[str, Any] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    executions: list[dict] = field(default_factory=list)

    def log(self, payload: dict) -> None:
        self.events.append(payload)
        if self.experiment_dir is not None:
            path = Path(self.experiment_dir) / "events.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _object(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _model_property(*, multitask_only: bool = False) -> dict:
    values = list(MULTITASK_MODELS) if multitask_only else list(AD_MODEL_NAMES)
    return {"type": "string", "enum": values}


# --------------------------------------------------------------------------- 工具


def _tool_list_ad_models(context: AdsToolContext, arguments: dict) -> dict:
    """List the six ad models with framework, task and weight status.

    Call this first. `status=interface_test` means only fixed-initialization
    weights exist (no training): usable for prediction, but NOT a trained or
    validated model. `usable=false` means no checkpoint at all.
    """
    return list_models(context.runtime)


def _tool_predict_ad_ctr(context: AdsToolContext, arguments: dict) -> dict:
    """Run real CTR inference with one ad model on KuaiSAR ad candidates.

    Args:
        model: one of deepfm/din/dien/dcnv2/mmoe/ple (mmoe/ple return CTR head).
        split: train/validation/test ad candidates (test is the fixed small sample).
        limit: max candidate rows scored from the split (fixed order).

    Returns AUC/LogLoss/CTR computed against the real `click` label, plus weight
    metadata. Metrics from interface_test weights are interface checks only.
    """
    model = arguments["model"]
    split = arguments.get("split", "test")
    limit = int(arguments.get("limit", 500))
    result = predict_ad_ctr(context.runtime, model, split=split, limit=limit)
    context.state["last_prediction"] = {
        "model": model,
        "split": split,
        "rows": result["rows"],
        "metrics": result["metrics"],
    }
    return result


def _tool_rank_ads(context: AdsToolContext, arguments: dict) -> dict:
    """Rank ad candidates inside each exposure request by predicted CTR.

    Args:
        model: ad model used to score candidates.
        split: which ad split to read.
        requests: how many requests to rank.
        min_candidates: only requests with at least this many ad candidates
            (use 2+ to see real competition; most KuaiSAR requests expose 1 ad).
        top_k: how many ranked items to return per request.

    Returns per-request rankings (item, ctr, click label for diagnostics) and a
    summary. This calls real model inference, not cached results.
    """
    model = arguments["model"]
    result = rank_ad_candidates(
        context.runtime,
        model,
        split=arguments.get("split", "test"),
        requests=int(arguments.get("requests", 3)),
        min_candidates=int(arguments.get("min_candidates", 1)),
        top_k=int(arguments.get("top_k", 3)),
    )
    context.state["last_ranking"] = result["summary"]
    return result


def _tool_simulate_auction(context: AdsToolContext, arguments: dict) -> dict:
    """Run the ad auction + CPC budget deduction for selected exposure requests.

    Rules: eCPM = bid_cpc * pCTR * 1000, second-price GSP clearing, CPC billing,
    charge at click time (sample click label in this offline simulation). Bids and
    budgets are SIMULATED values from config (KuaiSAR has no real bids/budgets).

    Args:
        model: ad model used for pCTR.
        split: ad split to read.
        requests: number of requests to simulate.
        min_candidates: prefer requests with multiple ad candidates (2+).
        top_k: ranked candidates returned per request.
        total_budget / per_advertiser_budget: optional budget overrides; use 0 to
            verify the insufficient-budget behavior (no winner, no charge).

    Returns winner, clearing CPC, charge, remaining budget per request, plus a
    summary. `no_winner_reason=budget_exhausted_*` means budget blocked the bid.
    """
    model = arguments["model"]
    result = simulate_ad_auction(
        context.runtime,
        model,
        context.config,
        split=arguments.get("split", "train"),
        requests=int(arguments.get("requests", 3)),
        min_candidates=int(arguments.get("min_candidates", 2)),
        top_k=int(arguments.get("top_k", 3)),
        total_budget=arguments.get("total_budget"),
        per_advertiser_budget=arguments.get("per_advertiser_budget"),
    )
    context.state["last_auction"] = {
        "run_id": result["run_id"],
        "model": model,
        "split": result["split"],
        "summary": result["summary"],
        "budget": result["summary"]["budget"],
        "overrides": result["budget_overrides"],
    }
    return result


def _tool_get_budget_state(context: AdsToolContext, arguments: dict) -> dict:
    """Return the budget state of the most recent auction simulation.

    Use it after simulate_auction to report spend and remaining budget, and after
    a zero-budget run to show that no winner was served and nothing was charged.
    """
    state = context.state.get("last_auction")
    if not state:
        return {
            "ok": True,
            "available": False,
            "note": "还没有跑过 simulate_auction；先调用竞价工具。",
        }
    return {
        "ok": True,
        "available": True,
        **state,
        "pricing_rule": "second_price_gsp",
        "billing_unit": "CPC",
        "charge_timing": "click",
    }


def _tool_check_label_availability(context: AdsToolContext, arguments: dict) -> dict:
    """Check which ad labels really exist in KuaiSAR (click vs conversion).

    Returns the real click label definition and the conversion-label gap. CVR has
    no real label: MMoE/PLE multi-task output can only be checked on the synthetic
    fixture (`synthetic=true`), never reported as real conversion or business lift.
    """
    return label_availability(context.runtime)


def _tool_predict_multitask(context: AdsToolContext, arguments: dict) -> dict:
    """Run MMoE/PLE multi-task inference (CTR + CVR) on the synthetic fixture.

    Only mmoe/ple support this. The conversion labels are synthetic, generated by
    a documented rule; this verifies that both task heads produce outputs, it does
    NOT measure real CVR.
    """
    result = predict_multitask(
        context.runtime,
        arguments["model"],
        limit=int(arguments.get("limit", 512)),
    )
    context.state["last_multitask"] = {
        "model": arguments["model"],
        "rows": result["rows"],
        "synthetic": True,
    }
    return result


def _tool_retrieve_knowledge(context: AdsToolContext, arguments: dict) -> dict:
    """Search the local project knowledge base for ad model/experiment background.

    Returns cited snippets with path:start-end. Retrieval is read-only and does
    NOT replace experiment validation; use it for background and provenance.
    """
    from recsys.platform.rag import retrieve_knowledge

    return retrieve_knowledge(
        str(arguments["query"]),
        top_k=int(arguments.get("top_k", 3)),
        index_dir=context.state.get("rag_index_dir"),
    )


# --------------------------------------------------------------------------- 注册表


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="list_ad_models",
            description=_tool_list_ad_models.__doc__.strip(),
            parameters=_object({}),
            execute=_tool_list_ad_models,
        )
    )
    registry.register(
        ToolSpec(
            name="predict_ad_ctr",
            description=_tool_predict_ad_ctr.__doc__.strip(),
            parameters=_object(
                {
                    "model": _model_property(),
                    "split": {"type": "string", "enum": list(SPLITS)},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10000},
                },
                ["model"],
            ),
            execute=_tool_predict_ad_ctr,
        )
    )
    registry.register(
        ToolSpec(
            name="rank_ads",
            description=_tool_rank_ads.__doc__.strip(),
            parameters=_object(
                {
                    "model": _model_property(),
                    "split": {"type": "string", "enum": list(SPLITS)},
                    "requests": {"type": "integer", "minimum": 1, "maximum": 50},
                    "min_candidates": {"type": "integer", "minimum": 1, "maximum": 10},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                ["model"],
            ),
            execute=_tool_rank_ads,
            mutates=False,
        )
    )
    registry.register(
        ToolSpec(
            name="simulate_auction",
            description=_tool_simulate_auction.__doc__.strip(),
            parameters=_object(
                {
                    "model": _model_property(),
                    "split": {"type": "string", "enum": list(SPLITS)},
                    "requests": {"type": "integer", "minimum": 1, "maximum": 50},
                    "min_candidates": {"type": "integer", "minimum": 1, "maximum": 10},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                    "total_budget": {"type": "number", "minimum": 0},
                    "per_advertiser_budget": {"type": "number", "minimum": 0},
                },
                ["model"],
            ),
            execute=_tool_simulate_auction,
            mutates=True,
        )
    )
    registry.register(
        ToolSpec(
            name="get_budget_state",
            description=_tool_get_budget_state.__doc__.strip(),
            parameters=_object({}),
            execute=_tool_get_budget_state,
        )
    )
    registry.register(
        ToolSpec(
            name="check_label_availability",
            description=_tool_check_label_availability.__doc__.strip(),
            parameters=_object({}),
            execute=_tool_check_label_availability,
        )
    )
    registry.register(
        ToolSpec(
            name="retrieve_knowledge",
            description=_tool_retrieve_knowledge.__doc__.strip(),
            parameters=_object(
                {
                    "query": {"type": "string", "minLength": 1},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                ["query"],
            ),
            execute=_tool_retrieve_knowledge,
        )
    )
    registry.register(
        ToolSpec(
            name="predict_multitask",
            description=_tool_predict_multitask.__doc__.strip(),
            parameters=_object(
                {
                    "model": _model_property(multitask_only=True),
                    "limit": {"type": "integer", "minimum": 1, "maximum": 5000},
                },
                ["model"],
            ),
            execute=_tool_predict_multitask,
        )
    )
    return registry


def build_context(
    runtime: AdsRuntime | None = None,
    *,
    dataset: str = "kuasar_small",
    data_dir: Path | None = None,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    config: AdsSimConfig | None = None,
    experiment_dir: Path | None = None,
    max_tool_calls: int = 9,
    max_seconds: float = 180.0,
) -> AdsToolContext:
    from recsys.ads.config import load_ads_config
    from recsys.ads.models import AdModelConfig
    from recsys.config import DEFAULT_DATA_DIR

    config = config or load_ads_config()
    if runtime is None:
        models = config.models
        runtime = AdsRuntime(
            dataset=dataset,
            data_dir=data_dir or DEFAULT_DATA_DIR,
            artifacts_dir=artifacts_dir,
            model_config=AdModelConfig(
                embedding_dim=models.embedding_dim,
                hidden_dim=models.hidden_dim,
                attention_dim=models.attention_dim,
                dropout=models.dropout,
                seed=models.seed,
                max_history=models.max_history,
                batch_size=models.batch_size,
                device=models.device,
            ),
        )
    return AdsToolContext(
        runtime=runtime,
        config=config,
        experiment_dir=experiment_dir,
        budget=ToolBudget(max_tool_calls=max_tool_calls, max_seconds=max_seconds),
    )


__all__ = [
    "AdsToolContext",
    "MULTITASK_MODELS",
    "SPLITS",
    "build_context",
    "build_registry",
]
