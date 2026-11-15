"""广告业务链路 CLI：广告候选打分 → 排序 → 二价竞价 → 预算扣减。

    python -m recsys.ads.pipeline --model din --split train --requests 3 --min-candidates 2
    python -m recsys.ads.pipeline --model din --budget-check

输出逐请求的候选排名、赢家、结算价（clearing CPC）、模拟扣费、剩余预算；
``--budget-check`` 会再用 0 预算跑一次，验证「预算不足时不出价、不扣费」。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recsys.ads import AD_MODEL_NAMES
from recsys.ads.config import DEFAULT_ADS_CONFIG, load_ads_config
from recsys.ads.models import AdModelConfig, AdModelUnavailable
from recsys.ads.runtime import AdsRuntime
from recsys.ads.service import simulate_ad_auction
from recsys.config import DEFAULT_DATA_DIR


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


def _compact_run(payload: dict) -> dict:
    summary = payload["summary"]
    return {
        "run_id": payload["run_id"],
        "model": payload["model"],
        "split": payload["split"],
        "requests": payload["requests"],
        "summary": summary,
        "price_rule": {
            "mode": payload["auction"]["mode"],
            "slots": payload["auction"]["slots"],
            "billing_unit": "CPC",
            "charge_timing": "click",
            "reserve_cpc": payload["auction"]["reserve_cpc"],
            "billing_basis": payload["auction"]["billing_basis"],
        },
        "sources": payload["sources"],
        "assumptions": payload["assumptions"],
        "weight": payload["weight"],
        "requests_detail": [
            {
                "request_index": record["request_index"],
                "timestamp": record["timestamp"],
                "user": record["user"],
                "candidate_count": record["candidate_count"],
                "ranking": [
                    {
                        "rank": entry["rank"],
                        "item": entry["item"],
                        "advertiser": entry["advertiser"],
                        "pctr": entry["pctr"],
                        "bid_cpc": entry["bid_cpc"],
                        "ecpm": entry["ecpm"],
                        "eligible": entry["eligible"],
                        "click_label": entry["click_label"],
                    }
                    for entry in record["ranking"]
                ],
                "winner": record["winner"],
                "winners": record["winners"],
                "clearing_cpc": record["clearing_cpc"],
                "charge": record["charge"],
                "billed_click": record["billed_click"],
                "budget_remaining": record.get("budget_remaining"),
                "no_winner_reason": record.get("no_winner_reason"),
                "partial_charge": record.get("partial_charge", False),
            }
            for record in payload["requests_detail"]
        ],
    }


def budget_insufficiency_check(payload: dict, zero_payload: dict) -> dict:
    zero_summary = zero_payload["summary"]
    checks = {
        "zero_budget_winners": zero_summary["winners"] == 0,
        "zero_budget_spend": abs(zero_summary["spend"]) < 1e-9,
        "zero_budget_reasons_present": all(
            record.get("no_winner_reason") for record in zero_payload["requests_detail"]
        ),
        "zero_budget_remaining": abs(zero_summary["budget"]["remaining"]) < 1e-9,
        "normal_spend_equals_sum_of_charges": abs(
            payload["summary"]["spend"]
            - sum(record["charge"] for record in payload["requests_detail"])
        )
        < 1e-6,
        "normal_remaining_consistent": abs(
            payload["summary"]["budget"]["remaining"]
            - (
                payload["summary"]["budget"]["total"]
                - payload["summary"]["budget"]["spent"]
            )
        )
        < 1e-6,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "zero_budget_summary": zero_summary,
        "zero_budget_reasons": [
            record.get("no_winner_reason") for record in zero_payload["requests_detail"]
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=AD_MODEL_NAMES, default="din")
    parser.add_argument("--dataset", default="kuasar_small")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--config", type=Path, default=DEFAULT_ADS_CONFIG)
    parser.add_argument("--split", default="train")
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--min-candidates", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=float, default=None, help="覆盖配置里的模拟总预算")
    parser.add_argument(
        "--per-advertiser-budget", type=float, default=None, help="覆盖配置里的单广告主预算"
    )
    parser.add_argument("--budget-check", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_ads_config(args.config)
    runtime = AdsRuntime(
        dataset=args.dataset,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        model_config=_model_config(config),
    )
    try:
        payload = simulate_ad_auction(
            runtime,
            args.model,
            config,
            split=args.split,
            requests=args.requests,
            min_candidates=args.min_candidates,
            top_k=args.top_k,
            total_budget=args.budget,
            per_advertiser_budget=args.per_advertiser_budget,
        )
    except (AdModelUnavailable, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False, indent=2))
        return 2

    report = {"ok": True, "run": _compact_run(payload)}
    exit_code = 0
    if args.budget_check:
        zero = simulate_ad_auction(
            runtime,
            args.model,
            config,
            split=args.split,
            requests=args.requests,
            min_candidates=args.min_candidates,
            top_k=args.top_k,
            total_budget=0.0,
            per_advertiser_budget=0.0,
        )
        check = budget_insufficiency_check(payload, zero)
        report["budget_insufficiency_check"] = check
        exit_code = 0 if check["passed"] else 1

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[ok] 结果写入 {args.output}")
    print(json.dumps(report["run"]["summary"], ensure_ascii=False, indent=2))
    for record in report["run"]["requests_detail"]:
        print(
            f"  request {record['request_index']} 候选 {record['candidate_count']} "
            f"赢家 {record['winner']['item'] if record['winner'] else None} "
            f"结算 CPC {record['clearing_cpc']} 扣费 {record['charge']} "
            f"剩余 {record['budget_remaining']} "
            f"{record['no_winner_reason'] or ''}"
        )
    if args.budget_check:
        print(json.dumps(report["budget_insufficiency_check"], ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
