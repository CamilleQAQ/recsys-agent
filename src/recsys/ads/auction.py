"""广告竞价与预算模拟（二价 GSP + CPC 计费）。

规则（全部写入输出，便于核对）：

1. **出价**：每个广告候选的 CPC 出价
   ``bid = base_cpc × (1 + step × (advertiser_id % mod))``。
   KuaiSAR 没有真实出价，``base_cpc`` / ``step`` / ``mod`` 都来自
   ``configs/ads/ads_sim.toml``，输出里标注 ``bid_source = simulated_config``。
2. **排序**：按 ``eCPM = bid × pCTR × 1000`` 降序，平局按 (item_id, row) 升序。
3. **计费单位**：CPC（按点击计费）。
4. **扣费时机**：
   * ``billing_basis = realized_click``（默认）：用样本的真实 ``click`` 标签模拟
     「是否真的产生点击」，点击才扣费；这是离线模拟，不是线上计费；
   * ``billing_basis = expected_click``：按 ``clearing_cpc × pCTR`` 记期望扣费。
5. **二价结算**：按排名最多填充 ``slots`` 个广告位。每位赢家按请求开始时
   有资格竞价的下一名 eCPM 计算 CPC，下限为保留价，上限为自己的出价。
6. **预算**：总预算与广告主预算同时扣减；广告主剩余预算低于保留价时不再参与竞价；
   实际扣费超过剩余预算时只扣到 0 并标记 ``budget_exhausted``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from recsys.ads.config import AuctionConfig, BudgetConfig


@dataclass
class BudgetState:
    total: float
    per_advertiser: float
    currency: str = "CNY"
    spent: float = 0.0
    served: int = 0
    clicks: int = 0
    exhausted_requests: int = 0
    advertiser_spent: dict[int, float] = field(default_factory=dict)

    @property
    def remaining(self) -> float:
        return max(0.0, self.total - self.spent)

    def advertiser_remaining(self, advertiser: int) -> float:
        return max(0.0, self.per_advertiser - self.advertiser_spent.get(advertiser, 0.0))

    def can_bid(self, advertiser: int, minimum: float) -> bool:
        return self.remaining >= minimum and self.advertiser_remaining(advertiser) >= minimum

    def charge(self, advertiser: int, amount: float) -> float:
        amount = max(0.0, float(amount))
        if amount <= 0:
            return 0.0
        paid = min(amount, self.remaining, self.advertiser_remaining(advertiser))
        self.spent += paid
        self.advertiser_spent[advertiser] = self.advertiser_spent.get(advertiser, 0.0) + paid
        return paid

    def as_dict(self, *, limit_advertisers: int | None = 10) -> dict[str, Any]:
        advertisers = sorted(self.advertiser_spent.items(), key=lambda item: -item[1])
        if limit_advertisers is not None:
            advertisers = advertisers[:limit_advertisers]
        return {
            "currency": self.currency,
            "total": round(self.total, 4),
            "spent": round(self.spent, 4),
            "remaining": round(self.remaining, 4),
            "per_advertiser_budget": round(self.per_advertiser, 4),
            "served": self.served,
            "clicks": self.clicks,
            "advertiser_spent_top": {
                str(advertiser): round(spent, 4) for advertiser, spent in advertisers
            },
            "budget_exhausted_requests": self.exhausted_requests,
        }


def advertiser_bids(
    advertiser_ids: np.ndarray, auction: AuctionConfig
) -> np.ndarray:
    step = float(auction.author_multiplier_step)
    mod = max(1, int(auction.author_multiplier_mod))
    multipliers = 1.0 + step * (np.asarray(advertiser_ids, dtype=np.int64) % mod)
    return float(auction.base_cpc) * multipliers


def run_request_auction(
    *,
    request_index: int,
    user: int,
    timestamp: int,
    items: np.ndarray,
    pctr: np.ndarray,
    labels: np.ndarray,
    advertiser_ids: np.ndarray,
    auction: AuctionConfig,
    budget: BudgetState,
    top_k: int = 5,
) -> dict[str, Any]:
    """一次曝光请求内的排序 + 竞价 + 预算扣减。"""
    items = np.asarray(items, dtype=np.int64)
    pctr = np.clip(np.asarray(pctr, dtype=np.float64), 1e-6, 1 - 1e-6)
    labels = np.asarray(labels, dtype=np.int64)
    advertiser_ids = np.asarray(advertiser_ids, dtype=np.int64)

    bids = advertiser_bids(advertiser_ids, auction)
    ecpm = bids * pctr * 1000.0
    eligible = np.asarray(
        [
            budget.can_bid(int(advertiser), float(auction.reserve_cpc))
            for advertiser in advertiser_ids
        ],
        dtype=bool,
    )
    score_order = np.lexsort((np.arange(items.size), items, -ecpm))

    candidates = []
    for rank, position in enumerate(score_order, start=1):
        candidates.append(
            {
                "rank": rank,
                "row": int(position),
                "item": int(items[position]),
                "advertiser": int(advertiser_ids[position]),
                "pctr": round(float(pctr[position]), 6),
                "bid_cpc": round(float(bids[position]), 4),
                "ecpm": round(float(ecpm[position]), 4),
                "eligible": bool(eligible[position]),
                "click_label": int(labels[position]),
            }
        )

    record: dict[str, Any] = {
        "request_index": int(request_index),
        "user": int(user),
        "timestamp": int(timestamp),
        "candidate_count": int(items.size),
        "pricing_rule": "second_price_gsp",
        "billing_unit": "CPC",
        "bid_source": auction.bid_source,
        "candidates": candidates[: max(1, int(top_k))],
        "ranking": candidates,
        "winner": None,
        "winners": [],
        "clearing_cpc": 0.0,
        "charge": 0.0,
        "billed_click": 0,
        "bid_ranking": [int(items[position]) for position in score_order],
    }

    if budget.remaining <= 0:
        record["no_winner_reason"] = "budget_exhausted_total"
        budget.exhausted_requests += 1
        return record
    eligible_positions = score_order[eligible[score_order]]
    if eligible_positions.size == 0:
        record["no_winner_reason"] = "budget_exhausted_advertisers"
        budget.exhausted_requests += 1
        return record

    winners = record["winners"]
    reserve = float(auction.reserve_cpc)
    charge = charge_due = 0.0
    billed_clicks = 0
    for rank, position in enumerate(eligible_positions):
        if len(winners) == auction.slots or budget.remaining <= 0:
            break
        advertiser = int(advertiser_ids[position])
        if not budget.can_bid(advertiser, reserve):
            continue
        winner_pctr = float(pctr[position])
        winner_bid = float(bids[position])
        next_ecpm = (
            float(ecpm[eligible_positions[rank + 1]]) if rank + 1 < eligible_positions.size else 0.0
        )
        clearing_cpc = min(winner_bid, max(reserve, next_ecpm / (winner_pctr * 1000.0)))
        billed = int(labels[position])
        factor = billed if auction.billing_basis == "realized_click" else winner_pctr
        due = clearing_cpc * factor
        paid = budget.charge(advertiser, due)
        winner = {
            "item": int(items[position]),
            "advertiser": advertiser,
            "pctr": round(winner_pctr, 6),
            "bid_cpc": round(winner_bid, 4),
            "ecpm": round(float(ecpm[position]), 4),
            "click_label": billed,
            "billed_click": billed,
        }
        if not winners:
            record["winner"] = winner
            record["clearing_cpc"] = round(clearing_cpc, 4)
        winners.append(
            {
                **winner,
                "slot": len(winners) + 1,
                "clearing_cpc": round(clearing_cpc, 4),
                "charge": round(paid, 4),
            }
        )
        charge += paid
        charge_due += due
        billed_clicks += billed

    if charge + 1e-9 < charge_due:
        record["partial_charge"] = True
        record["budget_exhausted"] = True
        budget.exhausted_requests += 1

    budget.served += len(winners)
    budget.clicks += billed_clicks
    record["charge"] = round(charge, 4)
    record["charge_due"] = round(charge_due, 4)
    record["billed_click"] = billed_clicks
    record["budget_remaining"] = round(budget.remaining, 4)
    return record


def new_budget_state(
    budget: BudgetConfig,
    auction: AuctionConfig,
    *,
    total: float | None = None,
    per_advertiser: float | None = None,
) -> BudgetState:
    return BudgetState(
        total=float(budget.total if total is None else total),
        per_advertiser=float(
            budget.per_advertiser if per_advertiser is None else per_advertiser
        ),
        currency=auction.currency,
    )


def summarize_auction(records: list[dict[str, Any]], budget: BudgetState) -> dict[str, Any]:
    winners = [winner for record in records for winner in record["winners"]]
    spend = sum(record["charge"] for record in records)
    clicks = sum(winner["billed_click"] for winner in winners)
    impressions = len(winners)
    exhausted = [record for record in records if record.get("no_winner_reason")]
    return {
        "requests": len(records),
        "winners": impressions,
        "clicks": int(clicks),
        "spend": round(spend, 4),
        "mean_charge": round(spend / impressions, 4) if impressions else 0.0,
        "ctr": round(float(clicks / impressions), 6) if impressions else 0.0,
        "mean_clearing_cpc": (
            round(float(np.mean([winner["clearing_cpc"] for winner in winners])), 4)
            if winners
            else 0.0
        ),
        "no_winner_requests": len(exhausted),
        "no_winner_reasons": [record.get("no_winner_reason") for record in exhausted],
        "partial_charges": sum(1 for record in records if record.get("partial_charge")),
        "budget": budget.as_dict(),
        "pricing_rule": "second_price_gsp",
        "billing_unit": "CPC",
        "charge_timing": "click",
    }


__all__ = [
    "BudgetState",
    "advertiser_bids",
    "new_budget_state",
    "run_request_auction",
    "summarize_auction",
]
