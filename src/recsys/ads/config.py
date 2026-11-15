"""广告竞价/预算与 Ads Agent 配置。

真实出价、预算和转化数据在 KuaiSAR 中都不存在（见 ``recsys.ads.data`` 的标签核对），
所以竞价参数全部来自配置文件里的**模拟值**，并在输出里通过
``bid_source`` / ``budget_source`` / ``assumptions`` 明确标注来源。修改配置只影响
模拟结果，不代表真实广告主出价。
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_ADS_CONFIG = Path("configs") / "ads" / "ads_sim.toml"
DEFAULT_AGENT_CONFIG = Path("configs") / "ads" / "ads_agent.toml"


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = 8
    hidden_dim: int = 32
    attention_dim: int = 16
    dropout: float = 0.1
    seed: int = 42
    max_history: int = 20
    batch_size: int = 512
    device: str = "cpu"


@dataclass(frozen=True)
class AuctionConfig:
    mode: str = "second_price"
    slots: int = 1
    reserve_cpc: float = 0.30
    base_cpc: float = 1.00
    author_multiplier_step: float = 0.10
    author_multiplier_mod: int = 3
    currency: str = "CNY"
    billing_basis: str = "realized_click"
    bid_source: str = "simulated_config"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BudgetConfig:
    total: float = 20.0
    per_advertiser: float = 8.0
    advertiser_key: str = "author_id"
    budget_source: str = "simulated_config"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdsSimConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    auction: AuctionConfig = field(default_factory=AuctionConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    sources: dict[str, str] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "models": asdict(self.models),
            "auction": self.auction.as_dict(),
            "budget": self.budget.as_dict(),
            "sources": dict(self.sources),
        }


@dataclass(frozen=True)
class AgentConfig:
    max_tool_calls: int = 8
    max_llm_requests: int = 6
    max_seconds: float = 180.0
    default_model: str = "din"
    default_split: str = "test"
    default_requests: int = 3
    top_k: int = 3
    min_candidates: int = 1
    auction_min_candidates: int = 2
    auction_split: str = "train"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    section = payload.get(name) or {}
    if not isinstance(section, dict):
        raise ValueError(f"配置节 [{name}] 必须是表")
    return section


def load_ads_config(path: Path = DEFAULT_ADS_CONFIG) -> AdsSimConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"广告配置不存在：{path}")
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    models = ModelConfig(**_section(payload, "models"))
    auction = AuctionConfig(**_section(payload, "auction"))
    budget = BudgetConfig(**_section(payload, "budget"))
    sources = payload.get("sources") or {}
    if not isinstance(sources, dict):
        sources = {}
    if auction.mode != "second_price":
        raise ValueError("当前只实现 second_price（GSP 二价）竞价")
    if type(auction.slots) is not int or auction.slots < 1:
        raise ValueError("auction.slots 必须是正整数")
    if auction.billing_basis not in {"realized_click", "expected_click"}:
        raise ValueError("auction.billing_basis 只能是 realized_click / expected_click")
    if budget.total <= 0 or budget.per_advertiser < 0:
        raise ValueError("预算必须为正")
    return AdsSimConfig(
        models=models,
        auction=auction,
        budget=budget,
        sources={str(key): str(value) for key, value in sources.items()},
    )


def load_agent_config(path: Path = DEFAULT_AGENT_CONFIG) -> AgentConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ads Agent 配置不存在：{path}")
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    agent = _section(payload, "agent")
    # 允许把竞价/预算节写在同一个文件里；未知键忽略。
    allowed = set(AgentConfig.__dataclass_fields__)
    return AgentConfig(**{key: value for key, value in agent.items() if key in allowed})


__all__ = [
    "AdsSimConfig",
    "AgentConfig",
    "AuctionConfig",
    "BudgetConfig",
    "DEFAULT_ADS_CONFIG",
    "DEFAULT_AGENT_CONFIG",
    "ModelConfig",
    "load_ads_config",
    "load_agent_config",
]
