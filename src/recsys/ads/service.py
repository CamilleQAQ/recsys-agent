"""广告业务服务层：预测、排序、竞价与预算模拟的真实执行函数。

工具层（``recsys.ads.tools``）只做参数校验与调度，业务全部在这里完成，
与 YieldAgent 的「工具 / 业务分离、Connector 承载平台逻辑」结构一致。
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import numpy as np

from recsys.ads.auction import new_budget_state, run_request_auction, summarize_auction
from recsys.ads.config import AdsSimConfig
from recsys.ads.features import build_click_metrics
from recsys.ads.models import AdModelUnavailable, model_spec
from recsys.ads.ranking import rank_candidates, summarize_rankings
from recsys.ads.registry import list_ad_models
from recsys.ads.runtime import AdsRuntime


def _weight_info(runtime: AdsRuntime, model: str, adapter) -> dict[str, Any]:
    info = runtime.weight_info(model)
    payload = runtime.model_payload(model)
    return {
        "weight_kind": payload.get("weight_kind", info.get("weight_kind")),
        "is_trained": bool(payload.get("is_trained", False)),
        "checkpoint": info.get("path"),
        "created_at": info.get("created_at"),
        "framework": model_spec(model).framework,
        "note": payload.get("note") or payload.get("purpose"),
    }


def list_models(runtime: AdsRuntime) -> dict[str, Any]:
    models = list_ad_models(runtime.artifacts_dir, runtime.dataset)
    return {
        "ok": True,
        "dataset": runtime.dataset,
        "models": models,
        "usable_models": [item["name"] for item in models if item["usable"]],
        "note": (
            "status=interface_test 表示只有固定初始化权重，"
            "只能证明接口可加载/可推理，不代表已训练或有效果。"
        ),
    }


def label_availability(runtime: AdsRuntime) -> dict[str, Any]:
    schema = runtime.ad_schema()
    conversion = dict(schema.get("conversion_label") or {})
    simulated = runtime.simulated_schema()
    return {
        "ok": True,
        "click_label": dict(schema.get("label") or {}),
        "conversion_label": conversion,
        "simulated_fixture": {
            "path": f"data/{runtime.dataset}/processed/ads_simulated/schema.json",
            "rows": simulated.get("counts", {}).get("rows"),
            "conversions": simulated.get("counts", {}).get("conversions"),
            "cvr_percent": simulated.get("label_provenance", {}).get("cvr_percent"),
            "synthetic": True,
            "not_real_conversion": True,
            "not_for_effect_claims": True,
        },
        "guidance": (
            "CTR 用真实 click 标签；CVR 没有真实标签，只能在 synthetic=true 的独立"
            "模拟样本上验证 MMoE/PLE 的多任务输入输出，不得表述为真实 CVR 效果。"
        ),
    }


def predict_ad_ctr(
    runtime: AdsRuntime,
    model: str,
    *,
    split: str = "test",
    limit: int = 500,
    preview: int = 3,
) -> dict[str, Any]:
    view = runtime.view(split)
    row_count = min(len(view), max(1, int(limit)))
    rows = np.arange(row_count, dtype=np.int64)
    selected = runtime.batch(split, rows)
    adapter = runtime.model(model)
    scores = adapter.score(selected.as_dict())
    ctr = np.asarray(scores["ctr"], dtype=np.float64)
    metrics = build_click_metrics(selected.labels, ctr)
    order = np.argsort(-ctr)[: max(1, int(preview))]
    return {
        "ok": True,
        "tool": "predict_ad_ctr",
        "model": model,
        "split": split,
        "scope": f"split 的前 {row_count} 个广告候选（固定顺序）",
        "rows": int(row_count),
        "metrics": metrics,
        "preview": [
            {
                "row": int(rows[position]),
                "user": int(selected.users[position]),
                "item": int(selected.items[position]),
                "pctr": round(float(ctr[position]), 6),
                "click_label": int(selected.labels[position]),
            }
            for position in order
        ],
        "weight": _weight_info(runtime, model, adapter),
    }


def rank_ad_candidates(
    runtime: AdsRuntime,
    model: str,
    *,
    split: str = "test",
    requests: int = 3,
    min_candidates: int = 1,
    top_k: int = 3,
    prefer_large: bool = True,
) -> dict[str, Any]:
    view = runtime.view(split)
    chosen, rows = runtime.request_rows(
        split,
        count=requests,
        min_candidates=min_candidates,
        prefer_large=prefer_large,
    )
    batch = runtime.batch(split, rows)
    adapter = runtime.model(model)
    ctr = np.asarray(adapter.score(batch.as_dict())["ctr"], dtype=np.float64)
    items = np.asarray(view.items)[rows]
    labels = np.asarray(view.labels)[rows]
    advertisers = runtime.advertiser_ids(items)
    request_ids = np.repeat(np.arange(chosen.size), np.asarray(view.groups)[chosen])
    rankings: list[dict[str, Any]] = []
    for position, request_index in enumerate(chosen):
        mask = request_ids == position
        rankings.append(
            {
                "request_index": int(request_index),
                "user": int(np.asarray(view.users)[rows][mask][0]),
                "timestamp": int(view.timestamps[request_index]),
                "candidate_count": int(mask.sum()),
                "ranking": rank_candidates(
                    items[mask],
                    ctr[mask],
                    labels=labels[mask],
                    advertiser_ids=advertisers[mask],
                    top_k=top_k,
                ),
            }
        )
    return {
        "ok": True,
        "tool": "rank_ads",
        "model": model,
        "split": split,
        "top_k": int(top_k),
        "summary": summarize_rankings(rankings),
        "rankings": rankings,
        "weight": _weight_info(runtime, model, adapter),
    }


def simulate_ad_auction(
    runtime: AdsRuntime,
    model: str,
    config: AdsSimConfig,
    *,
    split: str = "train",
    requests: int = 3,
    min_candidates: int = 2,
    top_k: int = 3,
    total_budget: float | None = None,
    per_advertiser_budget: float | None = None,
    prefer_large: bool = True,
) -> dict[str, Any]:
    view = runtime.view(split)
    chosen, rows = runtime.request_rows(
        split,
        count=requests,
        min_candidates=min_candidates,
        prefer_large=prefer_large,
    )
    batch = runtime.batch(split, rows)
    adapter = runtime.model(model)
    ctr = np.asarray(adapter.score(batch.as_dict())["ctr"], dtype=np.float64)
    items = np.asarray(view.items)[rows]
    labels = np.asarray(view.labels)[rows]
    advertisers = runtime.advertiser_ids(items)
    request_ids = np.repeat(np.arange(chosen.size), np.asarray(view.groups)[chosen])
    budget = new_budget_state(
        config.budget,
        config.auction,
        total=total_budget,
        per_advertiser=per_advertiser_budget,
    )
    user_rows = np.asarray(view.users)[rows]
    records: list[dict[str, Any]] = []
    for position in np.lexsort((chosen, view.timestamps[chosen])):
        request_index = chosen[position]
        mask = request_ids == position
        records.append(
            run_request_auction(
                request_index=int(request_index),
                user=int(user_rows[mask][0]),
                timestamp=int(view.timestamps[request_index]),
                items=items[mask],
                pctr=ctr[mask],
                labels=labels[mask],
                advertiser_ids=advertisers[mask],
                auction=config.auction,
                budget=budget,
                top_k=top_k,
            )
        )
    run_id = uuid4().hex[:12]
    return {
        "ok": True,
        "tool": "simulate_auction",
        "run_id": run_id,
        "model": model,
        "split": split,
        "requests": int(chosen.size),
        "min_candidates": int(min_candidates),
        "summary": summarize_auction(records, budget),
        "auction": config.auction.as_dict(),
        "budget_config": config.budget.as_dict(),
        "budget_overrides": {
            "total": total_budget,
            "per_advertiser": per_advertiser_budget,
        },
        "sources": {
            "bids": config.auction.bid_source,
            "budget": config.budget.budget_source,
            "clicks": (
                "KuaiSAR click 标签（离线模拟点击扣费）"
                if config.auction.billing_basis == "realized_click"
                else "期望扣费：clearing_cpc × pCTR"
            ),
        },
        "assumptions": [
            "KuaiSAR 没有真实广告出价，base_cpc 与广告主预算来自配置文件模拟值。",
            "一次曝光请求内的广告候选为该请求 is_ad=1 的物品；单候选请求按保留价成交。",
            "realized_click 用样本 click 标签模拟是否产生点击，不是线上计费日志。",
        ],
        "requests_detail": records,
        "weight": _weight_info(runtime, model, adapter),
    }


def predict_multitask(
    runtime: AdsRuntime,
    model: str,
    *,
    limit: int = 512,
    preview: int = 3,
) -> dict[str, Any]:
    spec = model_spec(model)
    if spec.task != "multitask":
        raise AdModelUnavailable(
            f"{model} 是单任务 CTR 模型；多任务预测只支持 "
            + ", ".join(name for name in ("mmoe", "ple"))
        )
    view = runtime.simulated_view()
    row_count = min(len(view), max(1, int(limit)))
    rows = np.arange(row_count, dtype=np.int64)
    batch = runtime.simulated_batch(rows)
    adapter = runtime.model(model)
    outputs = adapter.score(batch.as_dict())
    if "cvr" not in outputs:
        raise AdModelUnavailable(f"{model} 没有返回 CVR 输出")
    click = np.asarray(batch.labels, dtype=np.float64)
    conversion = np.asarray(view.conversion[rows], dtype=np.float64)
    ctr_metrics = build_click_metrics(click, outputs["ctr"])
    cvr_metrics = build_click_metrics(conversion, outputs["cvr"])
    return {
        "ok": True,
        "tool": "predict_multitask",
        "model": model,
        "dataset": runtime.dataset,
        "scope": "独立模拟转化样本（synthetic，前 N 行）",
        "rows": int(row_count),
        "synthetic": True,
        "not_real_conversion": True,
        "not_for_effect_claims": True,
        "metrics": {
            "ctr_task": ctr_metrics,
            "cvr_task": cvr_metrics,
        },
        "preview": [
            {
                "row": int(position),
                "item": int(batch.items[position]),
                "ctr": round(float(outputs["ctr"][position]), 6),
                "cvr": round(float(outputs["cvr"][position]), 6),
                "click_label": int(click[position]),
                "conversion_label_synthetic": int(conversion[position]),
            }
            for position in range(min(int(preview), row_count))
        ],
        "label_warning": (
            "CVR 标签是 synthetic（blake2b 生成），只用于验证多任务输入输出，"
            "不能当作真实转化或业务效果。"
        ),
        "weight": _weight_info(runtime, model, adapter),
    }


__all__ = [
    "label_availability",
    "list_models",
    "predict_ad_ctr",
    "predict_multitask",
    "rank_ad_candidates",
    "simulate_ad_auction",
]
