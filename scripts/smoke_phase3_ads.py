"""Phase 3 广告系统验收冒烟（小样本、只做接口检查）。

    $env:PYTHONPATH = "src"
    python scripts/smoke_phase3_ads.py

覆盖：

1. 广告样本：train/validation/test 三个切分可读、分组守恒、点击率；
2. 真实标签核对：click 可用、conversion 缺失；模拟转化样本可读且标注 synthetic；
3. 六个模型：权重可解析、可加载、可推理（概率范围/形状）；
4. 广告排序：同一请求内按 pCTR 降序；
5. 竞价与预算：赢家/结算/扣费/剩余预算一致；0 预算时不出价、不扣费；
6. Ads Agent 自检（14 项）与固定工具链。

不训练、不调参、不跑全量数据。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recsys.ads import AD_MODEL_NAMES  # noqa: E402
from recsys.ads.agent import run_self_check  # noqa: E402
from recsys.ads.config import load_ads_config  # noqa: E402
from recsys.ads.models import AdModelConfig, load_ad_model  # noqa: E402
from recsys.ads.ranking import rank_candidates  # noqa: E402
from recsys.ads.registry import resolve_weight  # noqa: E402
from recsys.ads.runtime import AdsRuntime  # noqa: E402
from recsys.ads.service import simulate_ad_auction  # noqa: E402

DEFAULT_REPORT = PROJECT_ROOT / "experiments" / "2026-11-09-kuasar_small-ads-phase3" / "smoke.json"


def _model_config(config) -> AdModelConfig:
    models = config.models
    return AdModelConfig(
        embedding_dim=models.embedding_dim,
        hidden_dim=models.hidden_dim,
        attention_dim=models.attention_dim,
        dropout=models.dropout,
        seed=models.seed,
        max_history=models.max_history,
        batch_size=models.batch_size,
        device=models.device,
    )


def check_data(runtime: AdsRuntime) -> dict:
    detail = {}
    for split in ("train", "validation", "test"):
        view = runtime.view(split)
        detail[split] = {
            "candidates": len(view),
            "requests": view.requests,
            "click_rate": round(float(np.asarray(view.labels).mean()), 6),
            "groups_sum_ok": int(np.asarray(view.groups).sum()) == len(view),
            "requests_with_2plus_ads": int((np.asarray(view.groups) >= 2).sum()),
        }
    schema = runtime.ad_schema()
    simulated = runtime.simulated_schema()
    return {
        "splits": detail,
        "label_audit": {
            "click": schema["label"]["field"],
            "conversion_available": schema["conversion_label"]["available"],
            "cvr_mode": schema["conversion_label"]["cvr_mode"],
        },
        "simulated": {
            "rows": simulated["counts"]["rows"],
            "conversions": simulated["counts"]["conversions"],
            "synthetic": simulated["synthetic"],
        },
        "passed": (
            all(item["groups_sum_ok"] for item in detail.values())
            and schema["conversion_label"]["available"] is False
            and simulated["synthetic"] is True
            and simulated["not_real_conversion"] is True
        ),
    }


def check_models(runtime: AdsRuntime) -> dict:
    runtime.view("test")
    rows = np.arange(min(8, len(runtime.view("test"))), dtype=np.int64)
    batch = runtime.batch("test", rows)
    detail = {}
    passed = True
    for name in AD_MODEL_NAMES:
        weight = resolve_weight(name, runtime.artifacts_dir, runtime.dataset)
        entry = {
            "checkpoint": str(weight["path"]) if weight["path"] else None,
            "weight_kind": weight["weight_kind"],
            "status": weight["status"],
        }
        try:
            adapter, payload = load_ad_model(
                Path(weight["path"]),
                runtime.data(),
                runtime.model_config,
                expected_items=int(runtime.data().num_items),
            )
            outputs = adapter.score(batch.as_dict())
            ctr = np.asarray(outputs["ctr"], dtype=np.float64)
            entry["inference_rows"] = int(ctr.size)
            entry["ctr_range_ok"] = bool(ctr.min() >= 0 and ctr.max() <= 1)
            entry["ctr_mean"] = round(float(ctr.mean()), 6)
            entry["tasks"] = sorted(outputs)
            entry["is_trained"] = bool(payload.get("is_trained", False))
            entry["ok"] = entry["ctr_range_ok"] and entry["is_trained"] is False
        except Exception as error:  # noqa: BLE001 - 逐个记录
            entry["ok"] = False
            entry["error"] = f"{type(error).__name__}: {error}"
        passed = passed and bool(entry.get("ok"))
        detail[name] = entry
    return {"models": detail, "passed": passed}


def check_ranking(runtime: AdsRuntime, config) -> dict:
    split = "train"
    chosen, rows = runtime.request_rows(split, count=3, min_candidates=2)
    batch = runtime.batch(split, rows)
    adapter = runtime.model("din")
    ctr = np.asarray(adapter.score(batch.as_dict())["ctr"], dtype=np.float64)
    items = np.asarray(runtime.view(split).items)[rows]
    request_ids = np.repeat(np.arange(chosen.size), np.asarray(runtime.view(split).groups)[chosen])
    per_request = []
    passed = True
    for position, request_index in enumerate(chosen):
        mask = request_ids == position
        ranking = rank_candidates(items[mask], ctr[mask], top_k=3)
        ordered = all(
            ranking[index]["ctr"] >= ranking[index + 1]["ctr"]
            for index in range(len(ranking) - 1)
        )
        per_request.append(
            {
                "request_index": int(request_index),
                "candidate_count": int(mask.sum()),
                "ranking": ranking,
                "order_ok": ordered,
            }
        )
        passed = passed and ordered
    return {"split": split, "requests": per_request, "passed": passed}


def check_auction(runtime: AdsRuntime, config) -> dict:
    normal = simulate_ad_auction(
        runtime,
        "din",
        config,
        split="train",
        requests=3,
        min_candidates=2,
        total_budget=20.0,
    )
    zero = simulate_ad_auction(
        runtime,
        "din",
        config,
        split="train",
        requests=3,
        min_candidates=2,
        total_budget=0.0,
        per_advertiser_budget=0.0,
    )
    normal_summary = normal["summary"]
    zero_summary = zero["summary"]
    checks = {
        "normal_has_winner": normal_summary["winners"] >= 1,
        "normal_spend_positive": normal_summary["spend"] > 0,
        "remaining_consistent": abs(
            normal_summary["budget"]["remaining"]
            - (normal_summary["budget"]["total"] - normal_summary["budget"]["spent"])
        )
        < 1e-6,
        "zero_budget_no_winner": zero_summary["winners"] == 0,
        "zero_budget_no_spend": zero_summary["spend"] == 0,
        "zero_budget_reason": all(
            record.get("no_winner_reason") for record in zero["requests_detail"]
        ),
        "billing_unit_cpc": normal_summary["billing_unit"] == "CPC",
        "charge_timing_click": normal_summary["charge_timing"] == "click",
    }
    return {
        "normal": normal_summary,
        "zero_budget": zero_summary,
        "checks": checks,
        "passed": all(checks.values()),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ads/ads_sim.toml"))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-agent", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_ads_config(args.config)
    runtime = AdsRuntime(model_config=_model_config(config))

    report = {
        "data": check_data(runtime),
        "models": check_models(runtime),
        "ranking": check_ranking(runtime, config),
        "auction": check_auction(runtime, config),
    }
    checks = {
        "data": report["data"]["passed"],
        "models": report["models"]["passed"],
        "ranking": report["ranking"]["passed"],
        "auction": report["auction"]["passed"],
    }
    if not args.skip_agent:
        self_check = run_self_check(args.report.parent / "self_check.json")
        report["agent_self_check"] = {
            "passed": self_check["passed"],
            "total": self_check["total"],
            "ok": self_check["ok"],
        }
        checks["agent_self_check"] = self_check["ok"]
    report["checks"] = checks
    report["passed"] = all(checks.values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(f"[{'ok' if report['passed'] else 'fail'}] 报告写入 {args.report}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
