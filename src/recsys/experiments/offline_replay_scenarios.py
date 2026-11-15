"""Phase 13 小规模多场景离线对照（场景 A / B）。

只读 ``kuasar_small`` 已有 5 组融合汇总 JSON。先冻结两个场景的本地规则
Agent 与固定策略轨迹，再各启动一次真实 LLM 工作流；不重训、不重新召回、
不重跑评测。

    $env:PYTHONPATH = "src"
    python -m recsys.experiments.offline_replay_scenarios --self-check
    python -m recsys.experiments.offline_replay_scenarios
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from recsys.experiments import offline_replay as replay
from recsys.experiments import offline_replay_llm as llm
from recsys.experiments.llm_tool_call import host_only, llm_config, load_dotenv

PROJECT_ROOT = replay.PROJECT_ROOT
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "offline_replay_scenarios.toml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "2026-11-09-kuasar_small-offline-replay-scenarios"
DEFAULT_PAIRED_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "2026-11-09-kuasar_small-70-30-history-ablation-paired"
PHASE = "Phase 13 Multi-Scenario Offline Replay"
SCHEMA_VERSION = 1
TOOL_NAME = "get_fusion_result"


class ScenarioError(RuntimeError):
    """场景配置、来源或冻结基线不满足要求。"""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha(path: Path) -> str:
    return replay.sha256_bytes(path.read_bytes())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe(value: Any, limit: int = 700) -> str:
    return llm._safe_text(value, limit)


# ---------------------------------------------------------------------------
# 配置与场景
# ---------------------------------------------------------------------------


def load_scenario_config(path: Path, base_override: Path | None = None) -> dict[str, Any]:
    try:
        with path.open("rb") as file:
            config = tomllib.load(file)
    except FileNotFoundError as error:
        raise ScenarioError(f"场景配置不存在：{path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ScenarioError(f"场景配置 TOML 解析失败：{error}") from error
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ScenarioError(f"schema_version 必须为 {SCHEMA_VERSION}")
    run = config.get("run") or {}
    base_path = base_override or (PROJECT_ROOT / run.get("base_config", ""))
    if not base_path.is_file():
        raise ScenarioError(f"基础回放配置不存在：{base_path}")
    base = replay.load_config(base_path)
    scenarios = config.get("scenarios") or {}
    if set(scenarios) != {"A", "B"}:
        raise ScenarioError("场景必须且只能是 A、B")
    mode = str(run.get("mode") or "multi_scenario")
    if mode not in {"multi_scenario", "history_ablation"}:
        raise ScenarioError(f"未知运行模式：{mode}")
    normalized: dict[str, dict[str, Any]] = {}
    for scenario_id in ("A", "B"):
        raw = scenarios[scenario_id]
        starts = [replay.parse_quota_text(value)[0] for value in raw["starting_quotas"]]
        allowed = [replay.parse_quota_text(value)[0] for value in raw["allowed_quotas"]]
        budget = int(raw["budget"])
        max_requests = int(raw["max_model_requests"])
        if set(starts) & set(allowed):
            raise ScenarioError(f"场景 {scenario_id} 的起始历史与可揭晓配额重叠")
        if not set(starts) <= {"50/50", "70/30", "30/70", "80/20", "20/80"}:
            raise ScenarioError(f"场景 {scenario_id} 起始配额不在已有 5 组来源内")
        if not set(allowed) <= {"70/30", "30/70", "80/20", "20/80"}:
            raise ScenarioError(f"场景 {scenario_id} 可揭晓配额不在已有 4 组未知配额内")
        normalized[scenario_id] = {
            "id": scenario_id,
            "label": raw.get("label", scenario_id),
            "observation": raw.get("observation", ""),
            "starting_quotas": starts,
            "allowed_quotas": allowed,
            "budget": budget,
            "max_model_requests": max_requests,
        }
    if mode == "multi_scenario":
        if normalized["A"]["starting_quotas"] != ["50/50"] or normalized["A"]["budget"] != 1 or normalized["A"]["max_model_requests"] != 2:
            raise ScenarioError("场景 A 必须是 50/50 起始、预算 1、最多 2 次请求")
        if normalized["B"]["starting_quotas"] != ["50/50", "70/30"] or normalized["B"]["budget"] != 2 or normalized["B"]["max_model_requests"] != 3:
            raise ScenarioError("场景 B 必须是 50/50+70/30 起始、预算 2、最多 3 次请求")
        if set(normalized["A"]["allowed_quotas"]) != {"70/30", "30/70", "80/20", "20/80"}:
            raise ScenarioError("场景 A 的可揭晓配额必须覆盖四个未知配额")
        if set(normalized["B"]["allowed_quotas"]) != {"30/70", "80/20", "20/80"}:
            raise ScenarioError("场景 B 的可揭晓配额必须排除 70/30")
        expected_fixed = ["70/30", "30/70", "80/20", "20/80"]
        max_total_requests = 5
    else:
        if normalized["A"]["starting_quotas"] != ["50/50"] or normalized["A"]["budget"] != 2 or normalized["A"]["max_model_requests"] != 3:
            raise ScenarioError("消融对照组必须是 50/50 起始、预算 2、最多 3 次请求")
        if normalized["B"]["starting_quotas"] != ["50/50", "70/30"] or normalized["B"]["budget"] != 2 or normalized["B"]["max_model_requests"] != 3:
            raise ScenarioError("消融历史组必须是 50/50+70/30 起始、预算 2、最多 3 次请求")
        expected_allowed = {"30/70", "80/20", "20/80"}
        if set(normalized["A"]["allowed_quotas"]) != expected_allowed or set(normalized["B"]["allowed_quotas"]) != expected_allowed:
            raise ScenarioError("消融两组的可揭晓配额必须相同且排除 70/30")
        expected_fixed = ["30/70", "80/20"]
        max_total_requests = 6
    fixed = config.get("fixed_policy") or {}
    fixed_order = [replay.parse_quota_text(value)[0] for value in fixed.get("order", [])]
    if fixed_order != expected_fixed:
        raise ScenarioError(f"固定策略顺序必须为 {' → '.join(expected_fixed)}")
    if int(normalized["A"]["max_model_requests"]) + int(
        normalized["B"]["max_model_requests"]
    ) > max_total_requests:
        raise ScenarioError(f"两个场景的模型请求上限合计不能超过 {max_total_requests}")
    paired_raw = config.get("paired") or {}
    paired = None
    if paired_raw:
        if mode != "history_ablation":
            raise ScenarioError("只有 history_ablation 模式支持成对复验")
        pairs = int(paired_raw.get("pairs", 0))
        order = [str(value).upper() for value in paired_raw.get("order", [])]
        paired_max = int(paired_raw.get("max_total_requests", 0))
        expected_order = ["A", "B", "B", "A", "A", "B"]
        if pairs != 3 or order != expected_order or paired_max != 18:
            raise ScenarioError("成对复验必须是 3 对，顺序 A→B、B→A、A→B，合计请求上限 18")
        reference_run = str(paired_raw.get("reference_run") or "").strip()
        if not reference_run:
            raise ScenarioError("成对复验必须配置已冻结的 reference_run")
        if not (PROJECT_ROOT / reference_run / "frozen" / "manifest.json").is_file():
            raise ScenarioError(f"冻结参照目录不存在：{reference_run}")
        paired = {
            "pairs": pairs,
            "order": order,
            "max_total_requests": paired_max,
            "reference_run": reference_run,
        }
    return {
        "path": path,
        "base_path": base_path,
        "base": base,
        "config": config,
        "mode": mode,
        "phase": run.get("phase", PHASE),
        "scenarios": normalized,
        "fixed_order": fixed_order,
        "local_policy": config.get("local_policy") or {},
        "paired": paired,
    }


def _validate_scenarios_against_sources(
    scenarios: Mapping[str, Any], sources: Mapping[str, Any], mode: str = "multi_scenario"
) -> None:
    records = sources["records"]
    for scenario in scenarios.values():
        missing = [quota for quota in scenario["starting_quotas"] + scenario["allowed_quotas"] if quota not in records]
        if missing:
            raise ScenarioError(f"场景 {scenario['id']} 缺少来源：{missing}")
    if mode == "history_ablation":
        baseline = records["50/50"]
        history = records["70/30"]
        if float(history["recall_at_100"]) >= float(baseline["recall_at_100"]):
            raise ScenarioError("70/30 的 Recall@100 必须低于 50/50，才符合本次消融前提")
        if replay.compare_records(history, baseline, 6) <= 0:
            raise ScenarioError("70/30 必须按完整最佳规则低于 50/50")
        for scenario in scenarios.values():
            starts = [records[quota] for quota in scenario["starting_quotas"]]
            best = replay.select_best(starts, 6)
            if best["quota"] != "50/50":
                raise ScenarioError(f"场景 {scenario['id']} 的起始最佳必须是 50/50")
            if "70/30" in scenario["allowed_quotas"]:
                raise ScenarioError(f"场景 {scenario['id']} 不能查询 70/30")


def _make_session(scenarios: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]], scenario_id: str):
    scenario = scenarios[scenario_id]
    return replay.MultiStartReplaySession(
        records=records,
        starting_quotas=scenario["starting_quotas"],
        allowed_quotas=scenario["allowed_quotas"],
        budget=scenario["budget"],
        decimals=6,
    )


# ---------------------------------------------------------------------------
# 本地规则 Agent / 固定策略
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioObservation:
    step: int
    candidates: tuple[str, ...]
    starting_histories: tuple[dict[str, Any], ...]
    revealed: tuple[dict[str, Any], ...]
    remaining_budget: int


def _all_start_best(observation: ScenarioObservation) -> Mapping[str, Any]:
    return replay.select_best(observation.starting_histories, 6)


def _unrevealed(observation: ScenarioObservation) -> list[str]:
    seen = {record["quota"] for record in observation.starting_histories}
    seen.update(record["quota"] for record in observation.revealed)
    return [quota for quota in observation.candidates if quota not in seen]


def _same_direction(record: Mapping[str, Any], observation: ScenarioObservation) -> str | None:
    candidates = [replay.parse_quota_text(quota) for quota in _unrevealed(observation)]
    if int(record["two_tower"]) > 50:
        higher = [item for item in candidates if item[1] > int(record["two_tower"])]
        return min(higher, key=lambda item: item[1])[0] if higher else None
    if int(record["two_tower"]) < 50:
        lower = [item for item in candidates if item[1] < int(record["two_tower"])]
        return max(lower, key=lambda item: item[1])[0] if lower else None
    return None


def _mirror(record: Mapping[str, Any], observation: ScenarioObservation) -> str | None:
    quota = f"{int(record['mind'])}/{int(record['two_tower'])}"
    return quota if quota in _unrevealed(observation) else None


class LocalScenarioPolicy:
    def __init__(self, rules: Mapping[str, Any]) -> None:
        self.info = {
            "name": "local_feedback_v1",
            "first_step": rules.get("first_step", ""),
            "better_step": rules.get("better_step", ""),
            "worse_step": rules.get("worse_step", ""),
            "tie_step": rules.get("tie_step", ""),
            "uses_history": True,
            "uses_feedback": True,
        }

    def decide(self, observation: ScenarioObservation) -> replay.PolicyAction:
        if observation.remaining_budget <= 0:
            return replay.PolicyAction("stop", reason="本场景揭晓预算已用完")
        if observation.step == 1:
            extras = [record for record in observation.starting_histories if record["quota"] != "50/50"]
            if extras:
                feedback = extras[-1]
                base = next(
                    (record for record in observation.starting_histories if record["quota"] == "50/50"),
                    observation.starting_histories[0],
                )
                feedback_recall = float(feedback["recall_at_100"])
                base_recall = float(base["recall_at_100"])
                if feedback_recall == base_recall:
                    return replay.PolicyAction(
                        "stop",
                        reason=(
                            f"起始历史 {feedback['quota']} 的 Recall@100 与 50/50 持平，"
                            "没有方向性收益，停止。"
                        ),
                    )
                if feedback_recall < base_recall:
                    mirror = _mirror(feedback, observation)
                    if mirror is not None:
                        return replay.PolicyAction(
                            "choose",
                            quota=mirror,
                            reason=(
                                f"起始历史 {feedback['quota']} 的 Recall@100 {feedback_recall:.6f} "
                                f"低于 50/50 的 {base_recall:.6f}；按原反馈规则转向镜像 {mirror}。"
                            ),
                        )
                else:
                    candidate = _same_direction(feedback, observation)
                    if candidate is not None:
                        return replay.PolicyAction(
                            "choose",
                            quota=candidate,
                            reason=(
                                f"起始历史 {feedback['quota']} 的 Recall@100 {feedback_recall:.6f} "
                                f"高于 50/50 的 {base_recall:.6f}；按同方向再探一档 {candidate}。"
                            ),
                        )
            unrevealed = _unrevealed(observation)
            if not unrevealed:
                return replay.PolicyAction("stop", reason="没有未揭晓候选")
            first = min(
                unrevealed,
                key=lambda quota: (
                    abs(replay.parse_quota_text(quota)[1] - 50)
                    + abs(replay.parse_quota_text(quota)[2] - 50),
                    replay.parse_quota_text(quota)[1],
                ),
            )
            return replay.PolicyAction(
                "choose",
                quota=first,
                reason=(
                    f"首步规则：{first} 与 50/50 的 L1 距离最小"
                    "（并列时双塔配额升序）；起始历史不计预算。"
                ),
            )

        latest = observation.revealed[-1]
        best_before = replay.select_best(
            [*observation.starting_histories, *observation.revealed[:-1]], 6
        )
        latest_recall = float(latest["recall_at_100"])
        best_recall = float(best_before["recall_at_100"])
        if latest_recall == best_recall:
            return replay.PolicyAction(
                "stop",
                reason=(
                    f"新结果 {latest['quota']} 的 Recall@100 与当前最佳 {best_before['quota']} 持平，停止。"
                ),
            )
        if latest_recall > best_recall:
            candidate = _same_direction(latest, observation)
            if candidate is None:
                return replay.PolicyAction("stop", reason="同方向没有未揭晓候选，停止。")
            return replay.PolicyAction(
                "choose",
                quota=candidate,
                reason=(
                    f"新结果 {latest['quota']} 的 Recall@100 {latest_recall:.6f} 高于当前最佳 "
                    f"{best_before['quota']} 的 {best_recall:.6f}；沿同方向再探 {candidate}。"
                ),
            )
        mirror = _mirror(latest, observation)
        if mirror is None:
            return replay.PolicyAction("stop", reason="镜像配额不可用或已揭晓，停止。")
        return replay.PolicyAction(
            "choose",
            quota=mirror,
            reason=(
                f"新结果 {latest['quota']} 的 Recall@100 {latest_recall:.6f} 低于当前最佳 "
                f"{best_before['quota']} 的 {best_recall:.6f}；转向镜像 {mirror}。"
            ),
        )


class FixedScenarioPolicy:
    def __init__(self, order: Sequence[str]) -> None:
        self._order = list(order)
        self._index = 0
        self.info = {"name": "fixed_symmetric_probe", "order": list(order), "uses_feedback": False}

    def decide(self, observation: ScenarioObservation) -> replay.PolicyAction:
        if observation.remaining_budget <= 0:
            return replay.PolicyAction("stop", reason="本场景揭晓预算已用完")
        starts = {record["quota"] for record in observation.starting_histories}
        while self._index < len(self._order) and self._order[self._index] in starts:
            self._index += 1
        if self._index >= len(self._order):
            return replay.PolicyAction("stop", reason="固定策略预设顺序执行完毕")
        quota = self._order[self._index]
        self._index += 1
        return replay.PolicyAction(
            "choose",
            quota=quota,
            reason=f"固定策略按预设顺序执行下一项：{quota}（自动跳过起始历史）",
        )


# ---------------------------------------------------------------------------
# 本地轨迹
# ---------------------------------------------------------------------------


def run_policy(
    session: replay.MultiStartReplaySession,
    policy: Any,
    scenario_id: str,
    side: str,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    stop: dict[str, Any] | None = None
    status = "success"
    for step in range(1, session.budget + 4):
        if session.remaining <= 0:
            stop = {
                "code": "budget_exhausted",
                "reason": f"场景 {scenario_id} 的 {session.budget} 次揭晓预算已用完",
                "remaining_budget": 0,
                "best": copy.deepcopy(session.best),
            }
            break
        visible = session.snapshot()
        action = policy.decide(
            ScenarioObservation(
                step=step,
                candidates=tuple(session.candidates),
                starting_histories=tuple(copy.deepcopy(session.starting_histories)),
                revealed=tuple(copy.deepcopy(session.revealed)),
                remaining_budget=session.remaining,
            )
        )
        if action.kind == "stop":
            stop = {
                "code": "policy_stop",
                "reason": action.reason or "策略主动停止",
                "remaining_budget": session.remaining,
                "best": copy.deepcopy(session.best),
            }
            break
        quota = action.quota or ""
        outcome = session.request(quota, step=step)
        if outcome["status"] != "ok":
            status = "failed"
            steps.append(
                {
                    "step": step,
                    "visible_before": visible,
                    "decision": {"action": "choose", "quota": quota, "reason": action.reason},
                    "tool_call": {"name": TOOL_NAME, "arguments": visible_probe(quota)},
                    "tool_return": {"status": "error", "code": outcome["code"], "message": outcome["message"]},
                    "remaining_budget_after": session.remaining,
                    "best_after": copy.deepcopy(session.best),
                    "best_changed": False,
                }
            )
            stop = {
                "code": outcome["code"],
                "reason": f"本地策略请求被拒绝：{outcome['message']}",
                "remaining_budget": session.remaining,
                "best": copy.deepcopy(session.best),
            }
            break
        args = visible_probe(outcome["quota"])
        steps.append(
            {
                "step": step,
                "visible_before": visible,
                "decision": {"action": "choose", "quota": outcome["quota"], "reason": action.reason},
                "tool_call": {"name": TOOL_NAME, "arguments": args},
                "tool_return": {"status": "ok", "result": copy.deepcopy(outcome["result"])},
                "remaining_budget_after": outcome["remaining_budget"],
                "best_after": copy.deepcopy(outcome["best"]),
                "best_changed": outcome["best_changed"],
            }
        )
    if stop is None:
        status = "failed"
        stop = {
            "code": "no_stop",
            "reason": "本地策略循环未记录停止原因",
            "remaining_budget": session.remaining,
            "best": copy.deepcopy(session.best),
        }
    summary = session.summary()
    summary.update(
        {
            "status": status,
            "stop_code": stop["code"],
            "stop_reason": stop["reason"],
            "invalid_requests": len(session.invalid_requests),
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario": scenario_id,
        "side": side,
        "policy": copy.deepcopy(policy.info),
        "budget_initial": session.budget,
        "candidates": list(session.candidates),
        "starting_histories": copy.deepcopy(session.starting_histories),
        "steps": steps,
        "invalid_requests": copy.deepcopy(session.invalid_requests),
        "stop": stop,
        "summary": summary,
        "status": status,
        "error": None if status == "success" else stop["reason"],
    }


def visible_probe(quota: str) -> dict[str, int]:
    _, two_tower, mind = replay.parse_quota_text(quota)
    return {"two_tower": two_tower, "mind": mind}


# ---------------------------------------------------------------------------
# 自检（合成、不联网）
# ---------------------------------------------------------------------------


def _first_payload(chat: llm.ScriptedChat) -> dict[str, Any]:
    user = next(item for item in chat.calls[0]["messages"] if item.get("role") == "user")
    return json.loads(user["content"])


def _fake_llm_case(
    base_config: Mapping[str, Any],
    scenario: Mapping[str, Any],
    responses: Sequence[Any],
) -> tuple[dict[str, Any], llm.ScriptedChat, replay.MultiStartReplaySession]:
    records = replay.synthetic_table()
    session = replay.MultiStartReplaySession(
        records=records,
        starting_quotas=scenario["starting_quotas"],
        allowed_quotas=scenario["allowed_quotas"],
        budget=scenario["budget"],
        decimals=6,
    )
    chat = llm.ScriptedChat(responses)
    final = llm.run_llm_replay(
        records=records,
        config=base_config,
        llm_cfg={"model": "fake-model", "base_url": "http://127.0.0.1", "api_key": "fake-key"},
        chat_fn=chat,
        session=session,
        max_model_requests=scenario["max_model_requests"],
        scenario=scenario["id"],
    )
    return final, chat, session


def run_self_check(config_bundle: Mapping[str, Any]) -> dict[str, Any]:
    base = config_bundle["base"]
    scenarios = config_bundle["scenarios"]
    fixed_order = config_bundle["fixed_order"]
    mode = config_bundle.get("mode", "multi_scenario")
    checks: list[dict[str, Any]] = []

    def add(check_id: str, status: str, detail: str) -> None:
        checks.append({"id": check_id, "status": status, "detail": detail})

    def check(check_id: str, function: Callable[[], str]) -> None:
        try:
            add(check_id, "pass", function())
        except Exception as error:  # noqa: BLE001
            add(check_id, "fail", f"{type(error).__name__}: {llm._safe_error(error)}")

    def initial_visibility() -> str:
        for scenario_id in ("A", "B"):
            scenario = scenarios[scenario_id]
            final, chat, _ = _fake_llm_case(
                base,
                scenario,
                [llm._tool_response("call-initial", 30, 70), llm._text_response("停止")],
            )
            payload = _first_payload(chat)
            assert [item["quota"] for item in payload["starting_histories"]] == scenario["starting_quotas"]
            assert payload["allowed_quotas"] == scenario["allowed_quotas"]
            assert payload["revealed_results"] == []
            assert payload["remaining_budget"] == scenario["budget"]
            raw = chat.calls[0]["messages"][-1]["content"]
            assert "0.013672" not in raw
            if scenario_id == "B" and "70/30" in scenario["starting_quotas"]:
                assert "0.009115" in raw
            if mode == "history_ablation" and scenario_id == "A":
                assert "0.009115" not in raw
            assert final["trace_steps"][0]["visible"]["revealed"] == []
        return "A 首轮只有 50/50；B 首轮有 50/50+70/30；均无其他隐藏指标"

    def local_and_fixed_policies() -> str:
        records = replay.synthetic_table()
        results = {}
        for scenario_id in ("A", "B"):
            scenario = scenarios[scenario_id]
            for side, policy in (
                ("local", LocalScenarioPolicy(config_bundle["local_policy"])),
                ("fixed", FixedScenarioPolicy(fixed_order)),
            ):
                session = replay.MultiStartReplaySession(
                    records=records,
                    starting_quotas=scenario["starting_quotas"],
                    allowed_quotas=scenario["allowed_quotas"],
                    budget=scenario["budget"],
                    decimals=6,
                )
                results[(scenario_id, side)] = run_policy(session, policy, scenario_id, side)
        a_local = results[("A", "local")]
        a_fixed = results[("A", "fixed")]
        b_local = results[("B", "local")]
        b_fixed = results[("B", "fixed")]
        a_local_choices = [s["decision"]["quota"] for s in a_local["steps"]]
        a_fixed_choices = [s["decision"]["quota"] for s in a_fixed["steps"]]
        b_local_choices = [s["decision"]["quota"] for s in b_local["steps"]]
        b_fixed_choices = [s["decision"]["quota"] for s in b_fixed["steps"]]
        if mode == "history_ablation":
            assert a_local_choices == ["30/70", "20/80"], a_local_choices
            assert a_fixed_choices == ["30/70", "80/20"], a_fixed_choices
            assert b_local_choices == ["30/70", "20/80"], b_local_choices
            assert b_fixed_choices == ["30/70", "80/20"], b_fixed_choices
            assert a_local["summary"]["best_quota"] == "30/70"
            assert a_fixed["summary"]["best_quota"] == "30/70"
            assert b_local["summary"]["best_quota"] == "30/70"
            assert b_fixed["summary"]["best_quota"] == "30/70"
            assert a_local["summary"]["reveals"] == 2
            assert a_fixed["summary"]["reveals"] == 2
            assert b_local["summary"]["reveals"] == 2
            assert b_fixed["summary"]["reveals"] == 2
            assert b_local["summary"]["starting_quotas"] == ["50/50", "70/30"]
            return "两组本地 30/70→20/80、固定 30/70→80/20；历史 70/30 未改变选择且未计预算"
        assert a_local_choices == ["30/70"]
        assert a_fixed_choices == ["70/30"]
        assert a_local["summary"]["best_quota"] == "30/70"
        assert a_fixed["summary"]["best_quota"] == "50/50"
        assert b_local_choices == ["30/70", "20/80"]
        assert b_fixed_choices == ["30/70", "80/20"]
        assert b_local["summary"]["best_quota"] == "30/70"
        assert b_fixed["summary"]["best_quota"] == "30/70"
        assert b_local["summary"]["reveals"] == 2
        assert b_fixed["summary"]["reveals"] == 2
        assert b_local["summary"]["starting_quotas"] == ["50/50", "70/30"]
        return "A 本地 30/70、固定 70/30；B 本地 30/70→20/80、固定 30/70→80/20；70/30 历史未计预算"

    def history_not_charged() -> str:
        records = replay.synthetic_table()
        scenario = scenarios["B"]
        session = replay.MultiStartReplaySession(
            records=records,
            starting_quotas=scenario["starting_quotas"],
            allowed_quotas=scenario["allowed_quotas"],
            budget=scenario["budget"],
            decimals=6,
        )
        assert session.remaining == 2
        assert session.summary()["best_quota"] == "50/50"
        rejected = session.request("70/30")
        assert rejected["status"] == "error"
        assert rejected["code"] == "duplicate_history"
        assert "result" not in rejected
        assert session.remaining == 2
        assert session.revealed == []
        accepted = session.request("30/70")
        assert accepted["status"] == "ok"
        assert session.remaining == 1
        assert len(session.revealed) == 1
        if mode == "history_ablation":
            control = replay.MultiStartReplaySession(
                records=records,
                starting_quotas=scenarios["A"]["starting_quotas"],
                allowed_quotas=scenarios["A"]["allowed_quotas"],
                budget=scenarios["A"]["budget"],
                decimals=6,
            )
            impossible = control.request("70/30")
            assert impossible["status"] == "error"
            assert impossible["code"] == "illegal_quota"
            assert "result" not in impossible
            assert control.remaining == 2
            return "历史组 70/30 不扣预算且重复请求被拒；对照组请求 70/30 被拒绝（不可查询）"
        return "B 起始 70/30 不扣预算；重复请求 70/30 被拒绝且无指标"

    def llm_verified_paths() -> str:
        if mode == "history_ablation":
            cases = [
                (
                    scenarios["A"],
                    [
                        llm._tool_response("call-a1", 30, 70),
                        llm._tool_response("call-a2", 20, 80),
                        llm._text_response("最佳保持 30/70"),
                    ],
                ),
                (
                    scenarios["B"],
                    [
                        llm._tool_response("call-b1", 30, 70),
                        llm._tool_response("call-b2", 20, 80),
                        llm._text_response("最佳保持 30/70"),
                    ],
                ),
            ]
            for scenario, responses in cases:
                final, _, session = _fake_llm_case(base, scenario, responses)
                assert final["status"] == "verified", final["status"]
                assert final["summary"]["request_count"] == 3
                assert final["summary"]["reveals"] == 2
                assert final["summary"]["best_quota"] == "30/70"
                assert session.remaining == 0
            return "对照组/历史组均 3 请求 / 2 揭晓 / verified"

        a_final, _, a_session = _fake_llm_case(
            base,
            scenarios["A"],
            [llm._tool_response("call-a", 30, 70), llm._text_response("30/70 命中 5，停止")],
        )
        b_final, _, b_session = _fake_llm_case(
            base,
            scenarios["B"],
            [
                llm._tool_response("call-b1", 30, 70),
                llm._tool_response("call-b2", 20, 80),
                llm._text_response("最佳保持 30/70"),
            ],
        )
        assert a_final["status"] == "verified"
        assert a_final["summary"]["request_count"] == 2
        assert a_final["summary"]["reveals"] == 1
        assert a_final["summary"]["best_quota"] == "30/70"
        assert a_session.remaining == 0
        assert b_final["status"] == "verified"
        assert b_final["summary"]["request_count"] == 3
        assert b_final["summary"]["reveals"] == 2
        assert b_final["summary"]["best_quota"] == "30/70"
        assert b_session.remaining == 0
        return "A 2 请求/1 揭晓、B 3 请求/2 揭晓，均 verified"

    def llm_truncated_paths() -> str:
        if mode == "history_ablation":
            for scenario_id, prefix in (("A", "a"), ("B", "b")):
                scenario = scenarios[scenario_id]
                final, _, _ = _fake_llm_case(
                    base,
                    scenario,
                    [
                        llm._tool_response(f"call-{prefix}1", 30, 70),
                        llm._tool_response(f"call-{prefix}2", 20, 80),
                        llm._text_response("截断的最终回复", finish_reason="length"),
                    ],
                )
                assert final["status"] == "partial", final["status"]
                assert final["summary"]["stop_code"] == "reply_truncated"
                assert final["summary"]["reveals"] == 2
            return "对照组/历史组 length 最终回复均为 partial/reply_truncated，保留 2 次揭晓"

        a_final, _, _ = _fake_llm_case(
            base,
            scenarios["A"],
            [
                llm._tool_response("call-a", 30, 70),
                llm._text_response("截断的最终回复", finish_reason="length"),
            ],
        )
        b_final, _, _ = _fake_llm_case(
            base,
            scenarios["B"],
            [
                llm._tool_response("call-b1", 30, 70),
                llm._tool_response("call-b2", 20, 80),
                llm._text_response("截断的最终回复", finish_reason="length"),
            ],
        )
        assert a_final["status"] == "partial", a_final["status"]
        assert a_final["summary"]["stop_code"] == "reply_truncated"
        assert a_final["summary"]["reveals"] == 1
        assert b_final["status"] == "partial", b_final["status"]
        assert b_final["summary"]["stop_code"] == "reply_truncated"
        assert b_final["summary"]["reveals"] == 2
        return "A/B 的 length 最终回复均为 partial/reply_truncated，保留揭晓结果"

    def request_budget_caps() -> str:
        if mode == "history_ablation":
            cases = [
                (
                    scenarios["A"],
                    [
                        llm._tool_response("call-a1", 30, 70),
                        llm._tool_response("call-a2", 20, 80),
                        llm._text_response("停止"),
                    ],
                ),
                (
                    scenarios["B"],
                    [
                        llm._tool_response("call-b1", 30, 70),
                        llm._tool_response("call-b2", 20, 80),
                        llm._text_response("停止"),
                    ],
                ),
            ]
            total = 0
            for scenario, responses in cases:
                final, _, _ = _fake_llm_case(base, scenario, responses)
                assert final["summary"]["request_count"] <= scenario["max_model_requests"]
                assert final["summary"]["reveals"] <= scenario["budget"]
                total += int(final["summary"]["request_count"])
            assert total <= 6
            return f"两场景请求合计 {total} ≤ 6；各自揭晓不超过 2"

        cases = [
            (scenarios["A"], [llm._tool_response("call-a", 30, 70), llm._text_response("停止")]),
            (
                scenarios["B"],
                [
                    llm._tool_response("call-b1", 30, 70),
                    llm._tool_response("call-b2", 20, 80),
                    llm._text_response("停止"),
                ],
            ),
        ]
        total = 0
        for scenario, responses in cases:
            final, _, _ = _fake_llm_case(base, scenario, responses)
            assert final["summary"]["request_count"] <= scenario["max_model_requests"]
            assert final["summary"]["reveals"] <= scenario["budget"]
            total += int(final["summary"]["request_count"])
        assert total <= 5
        return f"两场景请求合计 {total} ≤ 5；各自揭晓不超过预算"

    def ablation_only_history_diff() -> str:
        payloads = {}
        for scenario_id in ("A", "B"):
            _, chat, _ = _fake_llm_case(
                base,
                scenarios[scenario_id],
                [llm._tool_response(f"call-{scenario_id}", 30, 70), llm._text_response("停止")],
            )
            payloads[scenario_id] = _first_payload(chat)
        control = copy.deepcopy(payloads["A"])
        history = copy.deepcopy(payloads["B"])
        assert control["starting_histories"][0]["quota"] == "50/50"
        assert [item["quota"] for item in control["starting_histories"]] == ["50/50"]
        assert [item["quota"] for item in history["starting_histories"]] == ["50/50", "70/30"]
        assert control["allowed_quotas"] == history["allowed_quotas"]
        assert control["remaining_budget"] == history["remaining_budget"] == 2
        control.pop("starting_histories")
        history.pop("starting_histories")
        assert control == history
        raw_control = json.dumps(payloads["A"], ensure_ascii=False)
        raw_history = json.dumps(payloads["B"], ensure_ascii=False)
        assert "0.009115" not in raw_control
        assert "0.009115" in raw_history
        return "去掉历史组的 70/30 起始记录后，两组首轮 payload 完全一致；候选/预算/指令相同"

    def paired_plan_check() -> str:
        paired = config_bundle.get("paired")
        assert paired is not None
        assert paired["pairs"] == 3
        assert paired["order"] == ["A", "B", "B", "A", "A", "B"]
        assert paired["max_total_requests"] == 18

        def fake_final(quota: str, hits: int, recall: float, reveals: int, status: str = "verified") -> dict[str, Any]:
            record = replay.synthetic_record(quota, hits, recall, 0.02672)
            return {
                "summary": {
                    "status": status,
                    "best": record,
                    "best_quota": quota,
                    "hits": hits,
                    "recall_at_100": recall,
                    "coverage": 0.02672,
                    "reveals": reveals,
                    "first_best_reveal": 1,
                    "stop_reason": "synthetic",
                    "stop_code": "budget_exhausted_final",
                },
                "steps": [],
            }

        reproduced = judge_pair(
            fake_final("30/70", 5, 0.013672, 2),
            fake_final("30/70", 5, 0.013672, 1),
        )
        assert reproduced["status"] == "reproduced", reproduced
        tie = judge_pair(
            fake_final("30/70", 5, 0.013672, 2),
            fake_final("30/70", 5, 0.013672, 2),
        )
        assert tie["status"] == "tie", tie
        degraded = judge_pair(
            fake_final("30/70", 5, 0.013672, 2),
            fake_final("70/30", 3, 0.009115, 1),
        )
        assert degraded["status"] == "degraded_budget_gain", degraded
        not_judgeable = judge_pair(
            fake_final("30/70", 5, 0.013672, 2),
            fake_final("30/70", 5, 0.013672, 1, status="partial"),
        )
        assert not_judgeable["status"] == "not_judgeable", not_judgeable
        session = replay.MultiStartReplaySession(
            records=replay.synthetic_table(),
            starting_quotas=scenarios["A"]["starting_quotas"],
            allowed_quotas=scenarios["A"]["allowed_quotas"],
            budget=scenarios["A"]["budget"],
            decimals=6,
        )
        chat = llm.ScriptedChat([])
        request_counter = [0]
        global_counter = [18]
        blocked = llm.run_llm_replay(
            records=replay.synthetic_table(),
            config=base,
            llm_cfg={"model": "fake", "base_url": "http://127.0.0.1", "api_key": "fake"},
            chat_fn=chat,
            session=session,
            max_model_requests=scenarios["A"]["max_model_requests"],
            request_counter=request_counter,
            global_request_counter=global_counter,
            max_total_requests=18,
        )
        assert blocked["status"] == "incomplete", blocked["status"]
        assert request_counter[0] == 0
        assert global_counter[0] == 18
        assert len(chat.calls) == 0
        return "配对顺序固定为 A→B、B→A、A→B；复现/持平/退化/不可判定规则按预期；全轮 18 次上限在客户端调用前拒绝"

    state_checks = llm.run_self_check(base)
    for item in state_checks["checks"]:
        checks.append({"id": f"state.{item['id']}", "status": item["status"], "detail": item["detail"]})

    check("scenario.initial_visibility", initial_visibility)
    check("scenario.local_and_fixed_policies", local_and_fixed_policies)
    check("scenario.history_not_charged", history_not_charged)
    check("scenario.llm_verified_paths", llm_verified_paths)
    check("scenario.llm_truncated_paths", llm_truncated_paths)
    check("scenario.request_budget_caps", request_budget_caps)
    if mode == "history_ablation":
        check("scenario.only_history_diff", ablation_only_history_diff)
    if config_bundle.get("paired"):
        check("paired.plan_and_rules", paired_plan_check)

    passed = sum(1 for item in checks if item["status"] == "pass")
    failed = len(checks) - passed
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if failed == 0 else "fail",
        "total": len(checks),
        "passed": passed,
        "failed": failed,
        "synthetic_only": True,
        "network_requests": 0,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# 冻结与三方指标
# ---------------------------------------------------------------------------


def freeze_local(
    run_dir: Path,
    config_bundle: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    frozen_dir = run_dir / "frozen"
    generated_at = _iso(_utc_now())
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "source_manifest_sha256": _sha(run_dir / "sources.json"),
        "scenarios": {},
    }
    traces: dict[str, dict[str, Any]] = {}
    for scenario_id in ("A", "B"):
        scenario = config_bundle["scenarios"][scenario_id]
        local_session = replay.MultiStartReplaySession(
            records=records,
            starting_quotas=scenario["starting_quotas"],
            allowed_quotas=scenario["allowed_quotas"],
            budget=scenario["budget"],
            decimals=6,
        )
        fixed_session = replay.MultiStartReplaySession(
            records=records,
            starting_quotas=scenario["starting_quotas"],
            allowed_quotas=scenario["allowed_quotas"],
            budget=scenario["budget"],
            decimals=6,
        )
        local_trace = run_policy(
            local_session, LocalScenarioPolicy(config_bundle["local_policy"]), scenario_id, "local"
        )
        fixed_trace = run_policy(
            fixed_session, FixedScenarioPolicy(config_bundle["fixed_order"]), scenario_id, "fixed"
        )
        local_path = frozen_dir / scenario_id / "local_trace.json"
        fixed_path = frozen_dir / scenario_id / "fixed_trace.json"
        _write_json(local_path, local_trace)
        _write_json(fixed_path, fixed_trace)
        traces[scenario_id] = {"local": local_trace, "fixed": fixed_trace}
        manifest["scenarios"][scenario_id] = {
            "starting_quotas": scenario["starting_quotas"],
            "allowed_quotas": scenario["allowed_quotas"],
            "budget": scenario["budget"],
            "local": {
                "path": str(local_path.relative_to(run_dir)).replace("\\", "/"),
                "sha256": _sha(local_path),
                "summary": local_trace["summary"],
            },
            "fixed": {
                "path": str(fixed_path.relative_to(run_dir)).replace("\\", "/"),
                "sha256": _sha(fixed_path),
                "summary": fixed_trace["summary"],
            },
        }
    _write_json(frozen_dir / "manifest.json", manifest)
    return {"traces": traces, "manifest": manifest, "dir": frozen_dir}


def _first_choice(trace: Mapping[str, Any]) -> str | None:
    steps = trace.get("steps") or trace.get("trace_steps") or []
    for step in steps:
        if step.get("decision", {}).get("action") == "choose" and (
            (step.get("tool_return") or {}).get("status") == "ok"
        ):
            return step["decision"]["quota"]
    return None


def _query_sequence(trace: Mapping[str, Any]) -> list[str]:
    steps = trace.get("steps") or trace.get("trace_steps") or []
    return [
        step["decision"]["quota"]
        for step in steps
        if step.get("decision", {}).get("action") == "choose"
    ]


def _side_metrics(summary: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    best = summary.get("best") or {}
    return {
        "first_choice": _first_choice(trace),
        "best_quota": summary.get("best_quota"),
        "hits": summary.get("hits"),
        "recall_at_100": summary.get("recall_at_100"),
        "coverage": summary.get("coverage"),
        "reveals": summary.get("reveals"),
        "remaining_budget": summary.get("remaining_budget"),
        "first_best_reveal": summary.get("first_best_reveal", 0),
        "stop_code": summary.get("stop_code"),
        "stop_reason": summary.get("stop_reason"),
        "status": summary.get("status"),
        "best": copy.deepcopy(best),
    }


def _same_best(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left.get("best_quota") == right.get("best_quota")
        and int(left.get("hits") or 0) == int(right.get("hits") or 0)
        and abs(float(left.get("recall_at_100") or 0) - float(right.get("recall_at_100") or 0))
        <= 1e-9
    )


def judge_pair(
    a_final: Mapping[str, Any] | None,
    b_final: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not a_final or not b_final:
        return {
            "status": "not_judgeable",
            "reason": "至少一组未产出结果",
            "same_best": False,
            "a_reveals": None,
            "b_reveals": None,
            "saved_reveals": None,
        }
    a = _side_metrics(a_final.get("summary") or {}, a_final)
    b = _side_metrics(b_final.get("summary") or {}, b_final)
    same_best = _same_best(a, b)
    saved = int(a["reveals"]) - int(b["reveals"])
    result = {
        "same_best": same_best,
        "a_reveals": a["reveals"],
        "b_reveals": b["reveals"],
        "saved_reveals": saved,
        "a": a,
        "b": b,
    }
    if a.get("status") != "verified" or b.get("status") != "verified":
        result.update(
            {
                "status": "not_judgeable",
                "reason": f"A={a.get('status')}, B={b.get('status')}；任一组未 verified 即不可判定",
            }
        )
        return result
    reproduced = (
        same_best
        and a.get("best_quota") == "30/70"
        and int(a.get("hits") or 0) == 5
        and abs(float(a.get("recall_at_100") or 0) - 0.013672) <= 1e-9
        and int(a["reveals"]) == 2
        and int(b["reveals"]) == 1
    )
    if reproduced:
        result.update({"status": "reproduced", "reason": "A 2 次揭晓 / B 1 次揭晓，最佳均为 30/70 (5, 0.013672)"})
    elif same_best and int(a["reveals"]) == int(b["reveals"]):
        result.update({"status": "tie", "reason": "最佳结果相同且揭晓次数相同，记为持平"})
    elif saved > 0 and not same_best:
        result.update({"status": "degraded_budget_gain", "reason": "B 少揭晓但最佳指标下降，分别报告两项差异，不计复现"})
    elif same_best:
        result.update({"status": "same_best_budget_difference", "reason": "最佳相同但揭晓次数不同，未达到本轮复现定义"})
    else:
        result.update({"status": "other", "reason": "两组最佳与预算组合不符合预定义复现/持平/退化规则"})
    return result


def _comparison(
    llm_metrics: Mapping[str, Any],
    local_metrics: Mapping[str, Any],
    fixed_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    records = {
        "llm": llm_metrics["best"],
        "local": local_metrics["best"],
        "fixed": fixed_metrics["best"],
    }
    best = replay.select_best(records.values(), 6)
    names = [
        name for name, record in records.items() if replay.compare_records(record, best, 6) == 0
    ]
    winner = "tie" if len(names) > 1 else names[0]
    llm_cmp = replay.compare_records(llm_metrics["best"], best, 6)
    llm_outcome = "win" if llm_cmp < 0 else ("tie" if llm_cmp == 0 else "lose")
    judged = llm_metrics["status"] == "verified"
    saved_local = None
    saved_fixed = None
    if judged and llm_metrics["best_quota"] == local_metrics["best_quota"]:
        saved_local = int(local_metrics["reveals"]) - int(llm_metrics["reveals"])
    if judged and llm_metrics["best_quota"] == fixed_metrics["best_quota"]:
        saved_fixed = int(fixed_metrics["reveals"]) - int(llm_metrics["reveals"])
    return {
        "status": "judged" if judged else "not_judged",
        "winner": winner if judged else "not_judged",
        "llm_outcome": llm_outcome if judged else "not_judged",
        "best_quota": best["quota"] if judged else None,
        "names_at_best": names if judged else [],
        "deltas": {
            "llm_vs_local": {
                "hits": int(llm_metrics["hits"]) - int(local_metrics["hits"]),
                "recall_at_100": float(llm_metrics["recall_at_100"])
                - float(local_metrics["recall_at_100"]),
                "reveals": int(llm_metrics["reveals"]) - int(local_metrics["reveals"]),
            },
            "llm_vs_fixed": {
                "hits": int(llm_metrics["hits"]) - int(fixed_metrics["hits"]),
                "recall_at_100": float(llm_metrics["recall_at_100"])
                - float(fixed_metrics["recall_at_100"]),
                "reveals": int(llm_metrics["reveals"]) - int(fixed_metrics["reveals"]),
            },
        },
        "saved_reveals_vs_local": saved_local,
        "saved_reveals_vs_fixed": saved_fixed,
    }


def build_scenario_metrics(
    scenario: Mapping[str, Any],
    local_trace: Mapping[str, Any],
    fixed_trace: Mapping[str, Any],
    llm_final: Mapping[str, Any],
    llm_run_meta: Mapping[str, Any],
) -> dict[str, Any]:
    local_metrics = _side_metrics(local_trace["summary"], local_trace)
    fixed_metrics = _side_metrics(fixed_trace["summary"], fixed_trace)
    llm_metrics = _side_metrics(llm_final["summary"], llm_final)
    llm_metrics["request_count"] = llm_final["summary"].get("request_count")
    llm_metrics["elapsed_s"] = llm_run_meta.get("elapsed_s")
    llm_metrics["model"] = llm_run_meta.get("model")
    llm_metrics["endpoint_host"] = llm_run_meta.get("endpoint_host")
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario": scenario["id"],
        "label": scenario["label"],
        "starting_quotas": scenario["starting_quotas"],
        "allowed_quotas": scenario["allowed_quotas"],
        "budget": scenario["budget"],
        "llm": llm_metrics,
        "local": {**local_metrics, "policy": local_trace.get("policy")},
        "fixed": {**fixed_metrics, "policy": fixed_trace.get("policy")},
        "comparison": _comparison(llm_metrics, local_metrics, fixed_metrics),
    }


# ---------------------------------------------------------------------------
# 运行与报告
# ---------------------------------------------------------------------------


def _verify_hashes(files: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    after: dict[str, str] = {}
    changed: list[str] = []
    for key, payload in files.items():
        relative = str(payload.get("path") or key)
        path = (PROJECT_ROOT / relative).resolve()
        current = _sha(path)
        after[key] = current
        if current != payload["sha256"]:
            changed.append(key)
    return {"before": {key: item["sha256"] for key, item in files.items()}, "after": after, "changed": changed}


def _hash_map(files: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {"path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"), "sha256": _sha(path)}
        for name, path in files.items()
    }


def _scenario_llm_config(base: Mapping[str, Any]) -> dict[str, Any]:
    load_dotenv()
    cfg = llm_config()
    if cfg is None:
        missing = [name for name in ("DSH_AGENT_LLM_BASE_URL", "DSH_AGENT_LLM_API_KEY", "DSH_AGENT_LLM_MODEL") if not os.environ.get(name)]
        raise ScenarioError(f"缺少真实 LLM 配置：{', '.join(missing)}")
    return cfg


def _write_blocked(run_dir: Path, message: str) -> None:
    _write_json(
        run_dir / "blocked.json",
        {"schema_version": SCHEMA_VERSION, "status": "blocked", "phase": PHASE, "reason": message},
    )
    (run_dir / "report.md").write_text(
        f"""# Phase 13 多场景离线对照：已停止

- 状态：**blocked**
- 原因：{_safe(message)}

来源、场景配置、本地自检或 LLM 配置不满足时，按任务要求停止；未启动真实 LLM 的部分不补跑。
""",
        encoding="utf-8",
    )


def execute(
    config_bundle: Mapping[str, Any],
    output_root: Path,
    run_dir: Path,
    started_wall: datetime,
) -> dict[str, Any]:
    base = config_bundle["base"]
    sources = replay.load_sources(base)
    _validate_scenarios_against_sources(config_bundle["scenarios"], sources, config_bundle["mode"])
    scenarios = config_bundle["scenarios"]
    _write_json(run_dir / "sources.json", sources["manifest"])
    (run_dir / "scenario_config.toml").write_text(
        config_bundle["path"].read_text(encoding="utf-8"), encoding="utf-8"
    )
    (run_dir / "base_config.toml").write_text(
        config_bundle["base_path"].read_text(encoding="utf-8"), encoding="utf-8"
    )
    self_check = run_self_check(config_bundle)
    _write_json(run_dir / "self_check.json", self_check)
    if self_check["status"] != "pass":
        raise ScenarioError(f"本地替身检查未通过：{self_check['failed']} 项失败")

    frozen = freeze_local(run_dir, config_bundle, sources["records"])
    _write_json(run_dir / "frozen" / "manifest.json", frozen["manifest"])
    frozen_paths = {
        f"{scenario_id}.{side}": run_dir / "frozen" / scenario_id / f"{side}_trace.json"
        for scenario_id in ("A", "B")
        for side in ("local", "fixed")
    }
    frozen_before = _hash_map(frozen_paths)
    source_before = _verify_hashes(sources["files_read"])
    if source_before["changed"] or _verify_hashes(frozen_before)["changed"]:
        raise ScenarioError("本地冻结阶段输入或基线发生改写")

    llm_cfg = _scenario_llm_config(base)
    scenario_results: dict[str, Any] = {}
    total_requests = 0
    for scenario_id in ("A", "B"):
        scenario = scenarios[scenario_id]
        scenario_dir = run_dir / "scenarios" / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            session = replay.MultiStartReplaySession(
                records=sources["records"],
                starting_quotas=scenario["starting_quotas"],
                allowed_quotas=scenario["allowed_quotas"],
                budget=scenario["budget"],
                decimals=6,
            )
            final = llm.run_llm_replay(
                records=sources["records"],
                config=base,
                llm_cfg=llm_cfg,
                chat_fn=llm.post_chat,
                session=session,
                max_model_requests=scenario["max_model_requests"],
                scenario=scenario_id,
            )
            elapsed_s = round(time.perf_counter() - started, 3)
            llm_run_meta = {
                "model": llm_cfg.get("model"),
                "endpoint_host": host_only(str(llm_cfg.get("base_url"))),
                "elapsed_s": elapsed_s,
            }
            llm_trace = llm.build_llm_trace(final, base, llm_run_meta)
            metrics = build_scenario_metrics(
                scenario,
                frozen["traces"][scenario_id]["local"],
                frozen["traces"][scenario_id]["fixed"],
                final,
                llm_run_meta,
            )
            metrics["llm_trace"] = "llm_trace.json"
            metrics["events"] = "events.jsonl"
            _write_json(scenario_dir / "llm_trace.json", llm_trace)
            _write_json(scenario_dir / "metrics.json", metrics)
            metrics["_llm_steps"] = copy.deepcopy(final.get("trace_steps") or [])
            events = final.get("events") or []
            (scenario_dir / "events.jsonl").write_text(
                "\n".join(json.dumps(event, ensure_ascii=False) for event in events)
                + ("\n" if events else ""),
                encoding="utf-8",
            )
            total_requests += int(final["summary"].get("request_count") or 0)
            scenario_results[scenario_id] = {
                "status": final["summary"].get("status"),
                "metrics": metrics,
                "llm_trace": llm_trace,
                "final": final,
                "elapsed_s": elapsed_s,
            }
        except Exception as error:  # noqa: BLE001 - 单场景失败保留证据，不补跑
            message = llm._safe_error(error, llm_cfg)
            _write_json(
                scenario_dir / "blocked.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "blocked",
                    "scenario": scenario_id,
                    "reason": message,
                },
            )
            scenario_results[scenario_id] = {"status": "blocked", "error": message}

    source_after = _verify_hashes(sources["files_read"])
    frozen_after = _verify_hashes(frozen_before)
    if source_after["changed"] or frozen_after["changed"]:
        raise ScenarioError("真实 LLM 阶段来源或冻结基线发生改写")
    max_total_requests = 6 if config_bundle.get("mode") == "history_ablation" else 5
    if total_requests > max_total_requests:
        raise ScenarioError(f"两个场景的模型请求合计 {total_requests} 超过 {max_total_requests}")

    finished_wall = _utc_now()
    phase_name = config_bundle.get("phase", PHASE)
    run_meta = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase_name,
        "command": _command(config_bundle["path"], output_root),
        "argv": sys.argv[1:],
        "started_at": _iso(started_wall),
        "finished_at": _iso(finished_wall),
        "elapsed_s": round((finished_wall - started_wall).total_seconds(), 3),
        "model": llm_cfg.get("model"),
        "endpoint_host": host_only(str(llm_cfg.get("base_url"))),
        "output_dir": str(run_dir.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_files_read": sources["files_read"],
        "source_hashes_unchanged": not source_after["changed"],
        "source_hashes_after": source_after["after"],
        "frozen_files": frozen_before,
        "frozen_hashes_unchanged": not frozen_after["changed"],
        "frozen_hashes_after": frozen_after["after"],
        "llm_request_total": total_requests,
        "scope": {
            "training": 0,
            "recall": 0,
            "real_evaluation": 0,
            "llm_requests": total_requests,
            "network_requests": total_requests,
            "other_network_requests": 0,
            "dataset_files_read": 0,
            "candidate_detail_files_read": 0,
            "cache_files_read": 0,
            "memory_index_read": 0,
        },
        "write_scope": str(run_dir.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    overall = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase_name,
        "status": "completed"
        if all(item.get("status") in {"verified", "partial", "incomplete"} for item in scenario_results.values())
        else "blocked",
        "scenarios": {sid: item.get("metrics") for sid, item in scenario_results.items() if item.get("metrics")},
        "scenario_status": {sid: item.get("status") for sid, item in scenario_results.items()},
        "frozen": frozen["manifest"],
        "self_check": {
            "path": "self_check.json",
            "status": self_check["status"],
            "passed": self_check["passed"],
            "total": self_check["total"],
        },
        "source_manifest": "sources.json",
        "run": run_meta,
        "scope": run_meta["scope"],
    }
    _write_json(run_dir / "run_meta.json", run_meta)
    public_overall = copy.deepcopy(overall)
    for scenario_metrics in (public_overall.get("scenarios") or {}).values():
        if isinstance(scenario_metrics, dict):
            scenario_metrics.pop("_llm_steps", None)
    _write_json(run_dir / "metrics.json", public_overall)
    report = build_report(config_bundle, sources, frozen, self_check, overall, run_meta)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    return {
        "overall": overall,
        "self_check": self_check,
        "frozen": frozen,
        "scenario_results": scenario_results,
        "run_meta": run_meta,
        "report": report,
    }


def load_reference_frozen(
    config_bundle: Mapping[str, Any],
    sources: Mapping[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    paired = config_bundle["paired"]
    reference_run = (PROJECT_ROOT / paired["reference_run"]).resolve()
    manifest_path = reference_run / "frozen" / "manifest.json"
    if not manifest_path.is_file():
        raise ScenarioError(f"冻结参照 manifest 不存在：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_manifest_sha256") != _sha(run_dir / "sources.json"):
        raise ScenarioError("冻结参照的 sources.json 哈希与当前来源不一致")
    traces: dict[str, dict[str, Any]] = {}
    for scenario_id in ("A", "B"):
        scenario = config_bundle["scenarios"][scenario_id]
        traces[scenario_id] = {}
        for side in ("local", "fixed"):
            entry = manifest["scenarios"][scenario_id][side]
            path = reference_run / entry["path"]
            if not path.is_file():
                raise ScenarioError(f"冻结参照轨迹不存在：{path}")
            if _sha(path) != entry["sha256"]:
                raise ScenarioError(f"冻结参照轨迹哈希不一致：{path}")
            trace = json.loads(path.read_text(encoding="utf-8"))
            if trace.get("budget_initial") != scenario["budget"]:
                raise ScenarioError(f"{scenario_id}/{side} 冻结预算与当前配置不一致")
            if [item["quota"] for item in trace.get("starting_histories", [])] != scenario["starting_quotas"]:
                raise ScenarioError(f"{scenario_id}/{side} 冻结起始历史与当前配置不一致")
            if list(trace.get("candidates") or []) != scenario["allowed_quotas"]:
                raise ScenarioError(f"{scenario_id}/{side} 冻结候选与当前配置不一致")
            if trace.get("status") != "success":
                raise ScenarioError(f"{scenario_id}/{side} 冻结轨迹不是成功状态")
            if int(trace["summary"].get("reveals", -1)) > scenario["budget"]:
                raise ScenarioError(f"{scenario_id}/{side} 冻结轨迹超出预算")
            traces[scenario_id][side] = trace
    manifest = copy.deepcopy(manifest)
    manifest["reference_run"] = paired["reference_run"]
    return {"manifest": manifest, "traces": traces, "reference_run": paired["reference_run"]}


def _run_llm_workflow(
    *,
    base: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    scenario: Mapping[str, Any],
    llm_cfg: Mapping[str, Any],
    pair_index: int,
    position: int,
    request_counter: list[int],
    global_request_counter: list[int],
    max_total_requests: int,
) -> dict[str, Any]:
    session = replay.MultiStartReplaySession(
        records=records,
        starting_quotas=scenario["starting_quotas"],
        allowed_quotas=scenario["allowed_quotas"],
        budget=scenario["budget"],
        decimals=6,
    )
    started = time.perf_counter()
    final = llm.run_llm_replay(
        records=records,
        config=base,
        llm_cfg=llm_cfg,
        chat_fn=llm.post_chat,
        session=session,
        max_model_requests=scenario["max_model_requests"],
        scenario=f"pair-{pair_index}-{scenario['id']}",
        request_counter=request_counter,
        global_request_counter=global_request_counter,
        max_total_requests=max_total_requests,
    )
    elapsed_s = round(time.perf_counter() - started, 3)
    run_meta = {
        "model": llm_cfg.get("model"),
        "endpoint_host": host_only(str(llm_cfg.get("base_url"))),
        "elapsed_s": elapsed_s,
        "pair": pair_index,
        "position": position,
        "condition": scenario["id"],
    }
    trace = llm.build_llm_trace(final, base, run_meta)
    trace["pair"] = pair_index
    trace["position"] = position
    trace["condition"] = scenario["id"]
    metrics = _side_metrics(final["summary"], final)
    metrics["query_sequence"] = _query_sequence(final)
    metrics["request_count"] = int(request_counter[0])
    metrics["elapsed_s"] = elapsed_s
    metrics["model"] = llm_cfg.get("model")
    metrics["endpoint_host"] = host_only(str(llm_cfg.get("base_url")))
    return {
        "final": final,
        "trace": trace,
        "metrics": metrics,
        "events": final.get("events") or [],
        "elapsed_s": elapsed_s,
    }


def execute_paired(
    config_bundle: Mapping[str, Any],
    output_root: Path,
    run_dir: Path,
    started_wall: datetime,
) -> dict[str, Any]:
    paired = config_bundle.get("paired")
    if not paired:
        raise ScenarioError("当前配置没有 [paired] 段")
    base = config_bundle["base"]
    scenarios = config_bundle["scenarios"]
    sources = replay.load_sources(base)
    _validate_scenarios_against_sources(scenarios, sources, config_bundle["mode"])
    _write_json(run_dir / "sources.json", sources["manifest"])
    (run_dir / "scenario_config.toml").write_text(
        config_bundle["path"].read_text(encoding="utf-8"), encoding="utf-8"
    )
    (run_dir / "base_config.toml").write_text(
        config_bundle["base_path"].read_text(encoding="utf-8"), encoding="utf-8"
    )
    self_check = run_self_check(config_bundle)
    _write_json(run_dir / "self_check.json", self_check)
    if self_check["status"] != "pass":
        raise ScenarioError(f"本地替身检查未通过：{self_check['failed']} 项失败")

    # 成对复验只重复真实 LLM；本地/固定策略从 reference_run 的冻结基线读取。
    frozen = load_reference_frozen(config_bundle, sources, run_dir)
    _write_json(run_dir / "frozen" / "manifest.json", frozen["manifest"])
    reference_root = (PROJECT_ROOT / frozen["reference_run"]).resolve()
    frozen_paths = {
        f"{scenario_id}.{side}": reference_root
        / frozen["manifest"]["scenarios"][scenario_id][side]["path"]
        for scenario_id in ("A", "B")
        for side in ("local", "fixed")
    }
    frozen_before = _hash_map(frozen_paths)
    source_before = _verify_hashes(sources["files_read"])
    if source_before["changed"] or _verify_hashes(frozen_before)["changed"]:
        raise ScenarioError("参照基线或输入在回放前发生改写")

    llm_cfg = _scenario_llm_config(base)
    order = paired["order"]
    pair_results: list[dict[str, Any]] = []
    global_request_counter = [0]
    started_workflows = 0
    for pair_index in range(1, paired["pairs"] + 1):
        pair_dir = run_dir / "pairs" / f"pair-{pair_index}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        entries: dict[str, dict[str, Any]] = {}
        finals: dict[str, Any] = {}
        order_slice = order[(pair_index - 1) * 2 : pair_index * 2]
        for position, condition in enumerate(order_slice, start=1):
            scenario = scenarios[condition]
            condition_dir = pair_dir / condition
            condition_dir.mkdir(parents=True, exist_ok=True)
            started_workflows += 1
            workflow = None
            request_counter = [0]
            try:
                workflow = _run_llm_workflow(
                    base=base,
                    records=sources["records"],
                    scenario=scenario,
                    llm_cfg=llm_cfg,
                    pair_index=pair_index,
                    position=position,
                    request_counter=request_counter,
                    global_request_counter=global_request_counter,
                    max_total_requests=int(paired["max_total_requests"]),
                )
                _write_json(condition_dir / "llm_trace.json", workflow["trace"])
                events = workflow["events"]
                (condition_dir / "events.jsonl").write_text(
                    "\n".join(json.dumps(event, ensure_ascii=False) for event in events)
                    + ("\n" if events else ""),
                    encoding="utf-8",
                )
                metrics = workflow["metrics"]
                _write_json(condition_dir / "metrics.json", metrics)
                finals[condition] = workflow["final"]
                entries[condition] = {
                    "condition": condition,
                    "position": position,
                    "status": metrics.get("status"),
                    "metrics": metrics,
                    "trace": "llm_trace.json",
                    "events": "events.jsonl",
                    "request_count": int(request_counter[0]),
                    "error": None,
                }
            except Exception as error:  # noqa: BLE001 - 单组失败保留证据，不补跑
                message = llm._safe_error(error, llm_cfg)
                workflow_requests = int(request_counter[0])
                _write_json(
                    condition_dir / "blocked.json",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": "blocked",
                        "pair": pair_index,
                        "condition": condition,
                        "request_count": workflow_requests,
                        "reason": message,
                    },
                )
                entries[condition] = {
                    "condition": condition,
                    "position": position,
                    "status": "blocked",
                    "metrics": None,
                    "trace": None,
                    "events": None,
                    "request_count": workflow_requests,
                    "error": message,
                }
                finals[condition] = None
            finally:
                if condition in entries:
                    entries[condition]["request_count"] = int(request_counter[0])
        judgment = judge_pair(finals.get("A"), finals.get("B"))
        pair_metrics = {
            "pair": pair_index,
            "order": order_slice,
            "workflows": [entries[condition] for condition in order_slice],
            "judgment": judgment,
        }
        _write_json(pair_dir / "pair_metrics.json", pair_metrics)
        pair_results.append(pair_metrics)

    total_requests = int(global_request_counter[0])
    source_after = _verify_hashes(sources["files_read"])
    frozen_after = _verify_hashes(frozen_before)
    if source_after["changed"] or frozen_after["changed"]:
        raise ScenarioError("真实 LLM 阶段来源或冻结基线发生改写")
    if total_requests > int(paired["max_total_requests"]):
        raise ScenarioError(f"六次工作流模型请求合计 {total_requests} 超过 {paired['max_total_requests']}")

    judgeable = sum(1 for pair in pair_results if pair["judgment"]["status"] != "not_judgeable")
    reproduced = sum(1 for pair in pair_results if pair["judgment"]["status"] == "reproduced")
    status_counts: dict[str, int] = {}
    for pair in pair_results:
        status = pair["judgment"]["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    finished_wall = _utc_now()
    command = " ".join(
        shlex.quote(part)
        for part in [
            "python",
            "-m",
            "recsys.experiments.offline_replay_scenarios",
            "--paired",
            "--config",
            str(config_bundle["path"]),
            "--output-root",
            str(output_root),
        ]
    )
    run_meta = {
        "schema_version": SCHEMA_VERSION,
        "phase": "Phase 13 Paired Replication",
        "command": command,
        "argv": sys.argv[1:],
        "started_at": _iso(started_wall),
        "finished_at": _iso(finished_wall),
        "elapsed_s": round((finished_wall - started_wall).total_seconds(), 3),
        "model": llm_cfg.get("model"),
        "endpoint_host": host_only(str(llm_cfg.get("base_url"))),
        "output_dir": str(run_dir.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_files_read": sources["files_read"],
        "source_hashes_unchanged": not source_after["changed"],
        "source_hashes_after": source_after["after"],
        "frozen_files": frozen_before,
        "frozen_reference_run": frozen["reference_run"],
        "frozen_hashes_unchanged": not frozen_after["changed"],
        "frozen_hashes_after": frozen_after["after"],
        "workflows": len(pair_results) * 2,
        "workflows_started": started_workflows,
        "llm_request_total": total_requests,
        "pair_order": order,
        "scope": {
            "training": 0,
            "recall": 0,
            "real_evaluation": 0,
            "llm_requests": total_requests,
            "network_requests": total_requests,
            "other_network_requests": 0,
            "dataset_files_read": 0,
            "candidate_detail_files_read": 0,
            "cache_files_read": 0,
            "memory_index_read": 0,
        },
        "write_scope": str(run_dir.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    replication = {
        "planned_pairs": paired["pairs"],
        "judgeable_pairs": judgeable,
        "reproduced_pairs": reproduced,
        "reproduced_over_judgeable": f"{reproduced}/{judgeable}",
        "judgeable_over_planned": f"{judgeable}/{paired['pairs']}",
        "status_counts": status_counts,
    }
    overall = {
        "schema_version": SCHEMA_VERSION,
        "phase": "Phase 13 Paired Replication",
        "status": "completed",
        "pairs": pair_results,
        "replication": replication,
        "references": {
            "A": {
                "local": frozen["traces"]["A"]["local"]["summary"],
                "fixed": frozen["traces"]["A"]["fixed"]["summary"],
            },
            "B": {
                "local": frozen["traces"]["B"]["local"]["summary"],
                "fixed": frozen["traces"]["B"]["fixed"]["summary"],
            },
        },
        "frozen": frozen["manifest"],
        "self_check": {
            "path": "self_check.json",
            "status": self_check["status"],
            "passed": self_check["passed"],
            "total": self_check["total"],
        },
        "source_manifest": "sources.json",
        "run": run_meta,
        "scope": run_meta["scope"],
    }
    public_overall = copy.deepcopy(overall)
    for pair in public_overall["pairs"]:
        for workflow in pair["workflows"]:
            if isinstance(workflow.get("metrics"), dict):
                workflow["metrics"].pop("_llm_steps", None)
    _write_json(run_dir / "metrics.json", public_overall)
    _write_json(run_dir / "run_meta.json", run_meta)
    report = build_paired_report(
        config_bundle, sources, frozen, self_check, overall, run_meta
    )
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    return {
        "overall": overall,
        "self_check": self_check,
        "frozen": frozen,
        "run_meta": run_meta,
        "report": report,
    }


def build_paired_report(
    config_bundle: Mapping[str, Any],
    sources: Mapping[str, Any],
    frozen: Mapping[str, Any],
    self_check: Mapping[str, Any],
    overall: Mapping[str, Any],
    run_meta: Mapping[str, Any],
) -> str:
    paired = config_bundle["paired"]
    replication = overall["replication"]
    pairs = overall["pairs"]
    scenarios = config_bundle["scenarios"]

    status_names = {
        "reproduced": "复现",
        "tie": "持平",
        "degraded_budget_gain": "退化（B 少查但指标下降）",
        "same_best_budget_difference": "最佳相同但揭晓不同",
        "not_judgeable": "不可判定",
        "other": "其他",
    }

    def wf_row(condition: str, workflow: Mapping[str, Any]) -> str:
        metrics = workflow.get("metrics") or {}
        request_count = workflow.get("request_count") or metrics.get("request_count") or 0
        if not metrics:
            return (
                f"| {condition} | `blocked` | — | — | — | — | — | — | {request_count} | — |"
            )
        sequence = " → ".join(f"`{q}`" for q in (metrics.get("query_sequence") or [])) or "—"
        return (
            "| {condition} | `{status}` | `{first}` | {sequence} | `{best}` | {hits} | {recall:.6f} | "
            "{reveals} | {requests} | {stop} |".format(
                condition=condition,
                status=metrics.get("status"),
                first=metrics.get("first_choice") or "—",
                sequence=sequence,
                best=metrics.get("best_quota"),
                hits=metrics.get("hits"),
                recall=float(metrics.get("recall_at_100") or 0),
                reveals=metrics.get("reveals"),
                requests=request_count,
                stop=_stop_text(metrics.get("stop_reason"), metrics.get("stop_code")),
            )
        )

    pair_summary_rows = []
    pair_blocks = []
    for pair in pairs:
        p = pair["pair"]
        workflows = {item["condition"]: item for item in pair["workflows"]}
        a = workflows.get("A", {})
        b = workflows.get("B", {})
        a_req = a.get("request_count") or ((a.get("metrics") or {}).get("request_count") or 0)
        b_req = b.get("request_count") or ((b.get("metrics") or {}).get("request_count") or 0)
        judgment = pair["judgment"]
        pair_summary_rows.append(
            "| {p} | {order} | {a} | {b} | {judge} |".format(
                p=p,
                order=" → ".join(pair["order"]),
                a=(
                    f"`{a.get('status')}` / `{(a.get('metrics') or {}).get('best_quota')}` / "
                    f"{(a.get('metrics') or {}).get('reveals')} 揭晓 / "
                    f"{a_req} 请求"
                ),
                b=(
                    f"`{b.get('status')}` / `{(b.get('metrics') or {}).get('best_quota')}` / "
                    f"{(b.get('metrics') or {}).get('reveals')} 揭晓 / "
                    f"{b_req} 请求"
                ),
                judge=status_names.get(judgment["status"], judgment["status"]),
            )
        )
        detail_rows = "\n".join(
            wf_row("A · 对照组" if condition == "A" else "B · 历史组", workflows[condition])
            for condition in pair["order"]
        )
        a_reason = str((a.get("metrics") or {}).get("stop_reason") or "")
        b_reason = str((b.get("metrics") or {}).get("stop_reason") or "")
        pair_blocks.append(
            f"""### Pair {p} · {(' → '.join(pair['order']))}

| 条件 | 状态 | 首选 | 查询序列 | 最佳 | 命中 | Recall@100 | 揭晓 | 请求 | 停止原因 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{detail_rows}

- 判定：**{status_names.get(judgment['status'], judgment['status'])}**。{judgment['reason']}
- 差异：A {judgment.get('a_reveals')} 次揭晓 / B {judgment.get('b_reveals')} 次揭晓；B 相对 A 的节省值为 {judgment.get('saved_reveals')}。
- 首轮可见：A 仅 `50/50`；B 含 `70/30`（仅起始上下文）；两组可查询列表均不含 `70/30`。
- 历史使用证据：B 最终回复{'提及' if '70/30' in b_reason else '未提及'} `70/30`；A 最终回复{'提及' if '70/30' in a_reason else '未提及'} `70/30`。
- 轨迹：`pairs/pair-{p}/A/llm_trace.json`、`pairs/pair-{p}/B/llm_trace.json`。
"""
        )
    reference_rows = []
    for condition in ("A", "B"):
        local_trace = frozen["traces"][condition]["local"]
        fixed_trace = frozen["traces"][condition]["fixed"]
        reference_rows.append(
            "| {condition} | {local_seq} / `{local_best}` / {local_reveals} | {fixed_seq} / `{fixed_best}` / {fixed_reveals} |".format(
                condition=condition,
                local_seq=" → ".join(f"`{q}`" for q in _query_sequence(local_trace)) or "—",
                local_best=local_trace["summary"]["best_quota"],
                local_reveals=local_trace["summary"]["reveals"],
                fixed_seq=" → ".join(f"`{q}`" for q in _query_sequence(fixed_trace)) or "—",
                fixed_best=fixed_trace["summary"]["best_quota"],
                fixed_reveals=fixed_trace["summary"]["reveals"],
            )
        )
    source_rows = "\n".join(
        f"- `{path}`（sha256 `{payload['sha256']}`）"
        for path, payload in sorted(run_meta["source_files_read"].items())
    )
    frozen_rows = "\n".join(
        "| {scenario} | {side} | `{path}` | `{sha}` |".format(
            scenario=scenario_id,
            side=side,
            path=(
                f"{frozen['reference_run']}/{frozen['manifest']['scenarios'][scenario_id][side]['path']}"
                if frozen.get("reference_run")
                else frozen["manifest"]["scenarios"][scenario_id][side]["path"]
            ),
            sha=frozen["manifest"]["scenarios"][scenario_id][side]["sha256"],
        )
        for scenario_id in ("A", "B")
        for side in ("local", "fixed")
    )
    self_rows = "\n".join(
        f"| {item['id']} | {'通过' if item['status'] == 'pass' else '失败'} | {_cell(item['detail'], 160)} |"
        for item in self_check["checks"][:10]
    )
    judgeable = replication["judgeable_pairs"]
    reproduced = replication["reproduced_pairs"]
    if reproduced == 3:
        conclusion = "3 / 3 对复现：在本次配置下观察到重复出现的“历史组少揭晓 1 次、最佳不变”。"
    elif reproduced > 0:
        conclusion = f"部分复现：{reproduced} / {judgeable} 对可判定结果复现；其余按实际状态记录。"
    else:
        conclusion = f"未观察到复现：0 / {judgeable} 对可判定结果满足复现定义。"
    status_counts_text = "、".join(
        f"{status_names.get(key, key)} {value}" for key, value in sorted(replication["status_counts"].items())
    )
    return f"""# 2026-11-09 · Phase 13 · 历史信息节省查询的成对复验

> 复用 `70/30` 历史消融配置与 `kuasar_small` 汇总 JSON，新增 3 对真实 LLM 回放。每对两组各启动一次，顺序固定为 A→B、B→A、A→B；上一轮结果不计入本轮。没有训练、重新召回或真实评测，除已配置 LLM 服务外没有其他网络请求。

## 结论

- **{conclusion}**
- 统计口径：复现对数 / 可判定对数 = **{replication['reproduced_over_judgeable']}**；可判定对数 / 计划 3 对 = **{replication['judgeable_over_planned']}**。
- 逐对状态：{status_counts_text}。
- 复现定义：同一对内 A/B 均 `verified`、最佳均为 `30/70`（5 / 0.013672）、A 揭晓 2 次且 B 揭晓 1 次；其他结果按持平、退化或不可判定记录。
- 结论只覆盖本次 3 对回放与现有 5 组配额，不扩展到正式测试集或其他配置。

## 1. 对照条件

| 组别 | 起始历史 | 可查询配额 | 预算 | 模型请求上限 |
| --- | --- | --- | ---: | ---: |
| A · 对照组 | `50/50` | `30/70`、`80/20`、`20/80` | 2 | 3 |
| B · 历史组 | `50/50`、`70/30` | `30/70`、`80/20`、`20/80` | 2 | 3 |

固定执行顺序：`A → B`、`B → A`、`A → B`；`70/30` 在两组都不可查询，历史不扣预算。

## 2. 来源与冻结参照

- 来源：`kuasar_small` / `validation`，128 用户 / 362 目标，每用户 100 候选；5 组来源共享缓存身份 `sha256:{sources['manifest']['recall_cache']['version']}`。
- 本轮沿用上一轮冻结的本地规则与固定策略轨迹，不重跑；参照运行：`{frozen.get('reference_run', '—')}`。

| 条件 | 本地规则 | 固定策略 |
| --- | --- | --- |
{chr(10).join(reference_rows)}

| 场景 | 方 | 文件 | sha256 |
| --- | --- | --- | --- |
{frozen_rows}

- 来源工件（仅汇总 JSON）：

{source_rows}

## 3. 三对结果

| Pair | 顺序 | A 状态 / 最佳 / 揭晓 / 请求 | B 状态 / 最佳 / 揭晓 / 请求 | 判定 |
| --- | --- | --- | --- | --- |
{chr(10).join(pair_summary_rows)}

## 4. 逐对明细

{chr(10).join(pair_blocks)}

## 5. 本地自检

- 状态：**{self_check['status']}**；通过 **{self_check['passed']} / {self_check['total']}**。
- 覆盖：原有状态机 13 项 + 消融场景检查 + 配对顺序与复现/持平/退化/不可判定规则；全部合成、不联网。
- 前 10 条：

| 断言 | 状态 | 说明 |
| --- | --- | --- |
{self_rows}

完整结果见 [self_check.json](self_check.json)。

## 6. 执行范围与合规

| 项 | 实际 |
| --- | --- |
| 执行命令 | `{run_meta['command']}` |
| 模型 / 端点 | `{run_meta['model']}` / `{run_meta['endpoint_host']}` |
| 工作流启动 | **{run_meta.get('workflows_started', 6)} / 6**（3 对 × 2 组，各 1 次） |
| 模型请求 | **{run_meta['llm_request_total']} / 18** |
| 训练 / 重新召回 / 真实评测 | 0 / 0 / 0 |
| 其他网络请求 | 0（仅已配置 LLM 服务） |
| 数据集 / 候选明细 / 缓存 / 实验记忆索引 | 未读取 |
| 来源与冻结基线哈希 | 运行前后一致（见 `run_meta.json`） |
| 旧工件 | 未改写；只写入新的运行子目录 |

## 7. 验收对照

| 验收项 | 结论 | 证据 |
| --- | --- | --- |
| 运行可复核 | 通过 | 固定顺序、配置、来源哈希、6 组轨迹和每对指标均落盘 |
| 对照一致 | 通过 | 两组候选/预算一致；首轮仅 B 多 70/30 历史；70/30 不可查询 |
| 预算可核验 | 通过 | 6 次工作流，模型请求 {run_meta['llm_request_total']} / 18；各组 ≤3、揭晓 ≤2 |
| 结论可核验 | 通过 | 按预定义规则统计复现/持平/退化/不可判定，未调整顺序或追加配对 |

## 8. 结论边界

本轮只验证这套配置下的 3 对重复回放；每组只有一次真实 LLM 运行，复现次数由结果决定，不据此估计总体概率或泛化能力。完整 Phase 13 正式验证仍保留为待办。
"""
    parts = ["python", "-m", "recsys.experiments.offline_replay_scenarios"]
    if config_path.resolve() != DEFAULT_CONFIG_PATH.resolve():
        parts.extend(["--config", str(config_path)])
    if output_root.resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        parts.extend(["--output-root", str(output_root)])
    return " ".join(shlex.quote(part) for part in parts)


def _command(config_path: Path, output_root: Path) -> str:
    parts = ["python", "-m", "recsys.experiments.offline_replay_scenarios"]
    if config_path.resolve() != DEFAULT_CONFIG_PATH.resolve():
        parts.extend(["--config", str(config_path)])
    if output_root.resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        parts.extend(["--output-root", str(output_root)])
    return " ".join(shlex.quote(part) for part in parts)


def _make_run_dir(output_root: Path) -> Path:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    candidate = output_root / f"run-{stamp}-{os.getpid()}"
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"run-{stamp}-{os.getpid()}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _fmt(record: Mapping[str, Any]) -> str:
    if not record or record.get("hits") is None:
        return "—"
    return f"{record['hits']} / {float(record['recall_at_100']):.6f} / {float(record['coverage']):.6f}"


def _cell(value: Any, limit: int = 120) -> str:
    return _safe(value, limit).replace("\n", " ").replace("|", "\\|")


def _stop_text(reason: Any, code: Any) -> str:
    if code == "budget_exhausted_final":
        return "预算用尽后的最终回复（见下）"
    if code == "model_stop":
        return "模型主动停止（见下）"
    if code == "budget_exhausted":
        return "预算用完"
    return _cell(reason, 90)


def _visible(visible: Mapping[str, Any]) -> str:
    starts = visible.get("starting_histories")
    if starts is None:
        starts = [visible["starting_history"]]
    parts = [f"{record['quota']}: {_fmt(record)}" for record in starts]
    for result in visible.get("revealed") or []:
        parts.append(f"{result['quota']}: {_fmt(result)}")
    return "<br>".join(parts)


def _trace_rows(trace: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> str:
    rows = []
    phases = {"first": "首轮选择", "feedback": "反馈后决策", "final": "预算用尽后的最终回复"}
    steps = trace.get("steps", []) if isinstance(trace, Mapping) else trace
    for step in steps:
        decision = step.get("decision") or {}
        if decision.get("action") == "choose":
            action = f"查 `{decision.get('quota')}`"
        elif step.get("stop") or decision.get("action") == "stop":
            action = "停止（最终回复见下）"
        else:
            action = f"停止：{_cell(decision.get('reason'), 100)}"
        returned = step.get("tool_return") or {}
        if returned.get("status") == "ok":
            ret = f"ok `{returned.get('quota')}`: {_fmt(returned.get('metrics') or returned.get('result') or {})}"
        elif returned:
            ret = f"error `{returned.get('code')}`"
        else:
            ret = "—"
        rows.append(
            "| {request} | {phase} | {visible} | {action} | {call} | {ret} | {remaining} | {best} |".format(
                request=step.get("model_request_index", step.get("step")),
                phase=phases.get(step.get("phase"), step.get("phase", "—")),
                visible=_visible(step.get("visible_before") or step.get("visible") or {}),
                action=action.replace("|", "\\|"),
                call=(step.get("tool_call") or {}).get("id") or "—",
                ret=ret.replace("|", "\\|"),
                remaining=step.get("remaining_budget_after"),
                best=(step.get("best_after") or {}).get("quota") or "—",
            )
        )
    return "\n".join(rows)


def build_report(
    config_bundle: Mapping[str, Any],
    sources: Mapping[str, Any],
    frozen: Mapping[str, Any],
    self_check: Mapping[str, Any],
    overall: Mapping[str, Any],
    run_meta: Mapping[str, Any],
) -> str:
    scenarios = config_bundle["scenarios"]
    scenario_blocks = []
    for scenario_id in ("A", "B"):
        metrics = (overall.get("scenarios") or {}).get(scenario_id)
        if not metrics:
            scenario_blocks.append(
                f"## 场景 {scenario_id}：blocked\n\n本场景未产出三方指标；见对应目录的 `blocked.json`。\n"
            )
            continue
        llm = metrics["llm"]
        local = metrics["local"]
        fixed = metrics["fixed"]
        comparison = metrics["comparison"]
        status = llm["status"]
        if comparison["status"] == "judged":
            judge = (
                f"{'真实 LLM 胜出' if comparison['llm_outcome'] == 'win' else ('真实 LLM 与最佳持平' if comparison['llm_outcome'] == 'tie' else '真实 LLM 输给最佳')}"
                f"；三方最佳 {comparison['best_quota']}，并列方 {', '.join(comparison['names_at_best'])}。"
            )
        else:
            judge = f"真实 LLM 状态 `{status}`，本场景记为 **not_judged**，不按完整结果判胜负。"
        def saved_label(value: Any) -> str:
            if value is None:
                return "不适用"
            return f"节省 {value} 次" if value >= 0 else f"多 {abs(value)} 次"

        saved_local = saved_label(comparison["saved_reveals_vs_local"])
        saved_fixed = saved_label(comparison["saved_reveals_vs_fixed"])
        scenario_blocks.append(
            f"""## 场景 {scenario_id} · {scenarios[scenario_id]['label']}

- 起始历史：{', '.join(f'`{q}`' for q in metrics['starting_quotas'])}；可揭晓：{', '.join(f'`{q}`' for q in metrics['allowed_quotas'])}；每方预算 {metrics['budget']}。
- 观察点：{scenarios[scenario_id]['observation']}。

| 方 | 首选 | 最佳配额 | 命中 | Recall@100 | Coverage | 揭晓 / 预算 | 首次最佳 | 停止原因 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 真实 LLM | `{llm['first_choice'] or '—'}` | `{llm['best_quota']}` | {llm['hits']} | {float(llm['recall_at_100']):.6f} | {float(llm['coverage']):.6f} | {llm['reveals']} / {metrics['budget']} | {llm['first_best_reveal']} | {_stop_text(llm['stop_reason'], llm['stop_code'])} |
| 本地规则 Agent | `{local['first_choice'] or '—'}` | `{local['best_quota']}` | {local['hits']} | {float(local['recall_at_100']):.6f} | {float(local['coverage']):.6f} | {local['reveals']} / {metrics['budget']} | {local['first_best_reveal']} | {_stop_text(local['stop_reason'], local['stop_code'])} |
| 固定策略 | `{fixed['first_choice'] or '—'}` | `{fixed['best_quota']}` | {fixed['hits']} | {float(fixed['recall_at_100']):.6f} | {float(fixed['coverage']):.6f} | {fixed['reveals']} / {metrics['budget']} | {fixed['first_best_reveal']} | {_stop_text(fixed['stop_reason'], fixed['stop_code'])} |

- 判定：{judge}
- 真实 LLM 请求 **{llm['request_count']} / {scenarios[scenario_id]['max_model_requests']}**，耗时 **{llm['elapsed_s']} s**；相同最佳下相对本地规则查询 **{saved_local}**，相对固定策略 **{saved_fixed}**。
- LLM 相对本地规则：命中 {comparison['deltas']['llm_vs_local']['hits']:+d}、Recall@100 {comparison['deltas']['llm_vs_local']['recall_at_100']:+.6f}、揭晓 {comparison['deltas']['llm_vs_local']['reveals']:+d}；相对固定策略：命中 {comparison['deltas']['llm_vs_fixed']['hits']:+d}、Recall@100 {comparison['deltas']['llm_vs_fixed']['recall_at_100']:+.6f}、揭晓 {comparison['deltas']['llm_vs_fixed']['reveals']:+d}。

**真实 LLM 最终回复（截断，最多 400 字符）**：

> {_safe(llm['stop_reason'], 400).replace(chr(10), chr(10) + '> ')}

### 真实 LLM 轨迹

| 请求 | 阶段 | 当时可见 | 模型决定 | 工具调用 ID | 工具返回 | 剩余预算 | 最佳 |
| --- | --- | --- | --- | --- | --- | --- | --- |
{_trace_rows(metrics.get('_llm_steps') or []) or '| — | — | — | — | — | — | — | — |'}
"""
        )
    source_rows = "\n".join(
        f"- `{path}`（sha256 `{payload['sha256']}`，{payload['role']}）"
        for path, payload in sorted(run_meta["source_files_read"].items())
    )
    frozen_rows = "\n".join(
        "| {scenario} | {side} | `{path}` | `{sha}` |".format(
            scenario=scenario_id,
            side=side,
            path=frozen["manifest"]["scenarios"][scenario_id][side]["path"],
            sha=frozen["manifest"]["scenarios"][scenario_id][side]["sha256"],
        )
        for scenario_id in ("A", "B")
        for side in ("local", "fixed")
    )
    self_rows = "\n".join(
        f"| {item['id']} | {'通过' if item['status'] == 'pass' else '失败'} | {_cell(item['detail'], 160)} |"
        for item in self_check["checks"][:8]
    )

    def conclusion_line(scenario_id: str) -> str:
        metrics = (overall.get("scenarios") or {}).get(scenario_id)
        if not metrics:
            return f"场景 {scenario_id}：blocked，未产出三方指标。"
        llm_metrics = metrics["llm"]
        comparison = metrics["comparison"]
        if comparison["status"] == "judged":
            outcome = {
                "win": "真实 LLM 胜出",
                "tie": "真实 LLM 与最佳持平",
                "lose": "真实 LLM 输给最佳",
            }[comparison["llm_outcome"]]
            judge = f"{outcome}；三方最佳 {comparison['best_quota']}（并列：{', '.join(comparison['names_at_best'])}）"
        else:
            judge = f"not_judged（真实 LLM 状态 `{llm_metrics['status']}`）"
        return (
            f"场景 {scenario_id} · {scenarios[scenario_id]['label']}：真实 LLM 首选 "
            f"`{llm_metrics['first_choice'] or '—'}`、最佳 `{llm_metrics['best_quota']}`，{judge}；"
            f"请求 {llm_metrics['request_count']} 次 / 揭晓 {llm_metrics['reveals']} 次。"
        )

    mode = config_bundle.get("mode", "multi_scenario")
    is_ablation = mode == "history_ablation"
    title = "Phase 13 · `70/30` 历史信息消融" if is_ablation else "Phase 13 小规模多场景离线对照"
    scenario_rows = "\n".join(
        "| {sid} · {label} | {starts} | {allowed} | {budget} | {max_requests} |".format(
            sid=sid,
            label=scenarios[sid]["label"],
            starts="、".join(f"`{q}`" for q in scenarios[sid]["starting_quotas"]),
            allowed="、".join(f"`{q}`" for q in scenarios[sid]["allowed_quotas"]),
            budget=scenarios[sid]["budget"],
            max_requests=scenarios[sid]["max_model_requests"],
        )
        for sid in ("A", "B")
    )
    fixed_text = " → ".join(f"`{q}`" for q in config_bundle["fixed_order"])
    detail_sec = "4" if is_ablation else "3"
    self_sec = "5" if is_ablation else "4"
    scope_sec = "6" if is_ablation else "5"
    accept_sec = "7" if is_ablation else "6"
    boundary_sec = "8" if is_ablation else "7"
    ablation_section = ""
    ablation_conclusion = ""
    ablation_line = ""
    if is_ablation:
        a = (overall.get("scenarios") or {}).get("A") or {}
        b = (overall.get("scenarios") or {}).get("B") or {}
        if a and b:
            rows = []
            for strategy, strategy_label in (
                ("llm", "真实 LLM"),
                ("local", "本地规则 Agent"),
                ("fixed", "固定策略"),
            ):
                for sid, condition in (("A", "对照（无 70/30）"), ("B", "历史（可见 70/30）")):
                    metrics = (overall.get("scenarios") or {}).get(sid) or {}
                    side = metrics.get(strategy) or {}
                    if strategy == "llm":
                        trace = {"steps": metrics.get("_llm_steps") or []}
                    else:
                        trace = frozen["traces"][sid][strategy]
                    sequence = _query_sequence(trace)
                    rows.append(
                        "| {strategy} | {condition} | `{first}` | {sequence} | `{best}` | {hits} | {recall:.6f} | {reveals} | {first_best} | {stop} |".format(
                            strategy=strategy_label,
                            condition=condition,
                            first=side.get("first_choice") or "—",
                            sequence=" → ".join(f"`{q}`" for q in sequence) or "—",
                            best=side.get("best_quota"),
                            hits=side.get("hits"),
                            recall=float(side.get("recall_at_100") or 0),
                            reveals=side.get("reveals"),
                            first_best=side.get("first_best_reveal"),
                            stop=_stop_text(side.get("stop_reason"), side.get("stop_code")),
                        )
                    )
            behavior = []
            for strategy, strategy_label in (
                ("llm", "真实 LLM"),
                ("local", "本地规则 Agent"),
                ("fixed", "固定策略"),
            ):
                side_a = a[strategy]
                side_b = b[strategy]
                seq_a = _query_sequence(
                    {"steps": a.get("_llm_steps") or []}
                    if strategy == "llm"
                    else frozen["traces"]["A"][strategy]
                )
                seq_b = _query_sequence(
                    {"steps": b.get("_llm_steps") or []}
                    if strategy == "llm"
                    else frozen["traces"]["B"][strategy]
                )
                same_behavior = seq_a == seq_b
                same_best = side_a["best_quota"] == side_b["best_quota"]
                same_reveals = side_a["reveals"] == side_b["reveals"]
                if same_behavior and same_best and same_reveals:
                    behavior.append(f"{strategy_label}：两组首选/查询序列/最佳/揭晓次数相同，未观察到行为差异。")
                else:
                    behavior.append(
                        f"{strategy_label}：查询序列 `{' → '.join(seq_a) or '—'}` → "
                        f"`{' → '.join(seq_b) or '—'}`；最佳 {side_a['best_quota']} → "
                        f"{side_b['best_quota']}；揭晓 {side_a['reveals']} → {side_b['reveals']}。"
                    )
            saved_text = "不适用"
            if a["llm"]["best_quota"] == b["llm"]["best_quota"]:
                saved = int(a["llm"]["reveals"]) - int(b["llm"]["reveals"])
                saved_text = f"历史组少揭晓 {saved} 次" if saved > 0 else (
                    f"历史组多揭晓 {-saved} 次" if saved < 0 else "两组揭晓次数相同"
                )
            history_visible = "70/30" in b["starting_quotas"]
            history_mentioned = "70/30" in str(b["llm"].get("stop_reason") or "") or "0.009115" in str(
                b["llm"].get("stop_reason") or ""
            )
            control_mentioned = "70/30" in str(a["llm"].get("stop_reason") or "")
            ablation_conclusion = (
                f"历史消融：真实 LLM 首选 {a['llm']['first_choice'] or '—'} → "
                f"{b['llm']['first_choice'] or '—'}；最佳 {a['llm']['best_quota']} → "
                f"{b['llm']['best_quota']}；揭晓 {a['llm']['reveals']} → {b['llm']['reveals']}，{saved_text}。"
            )
            ablation_section = f"""## 3. 历史消融对照

| 策略 | 条件 | 首选 | 查询序列 | 最佳 | 命中 | Recall@100 | 揭晓 | 首次最佳 | 停止原因 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

- 行为差异：{'；'.join(x.rstrip('。') for x in behavior)}。
- 首轮输入差异：历史组仅多一条 `70/30` 起始历史；去掉该条后两组首轮 payload 一致（自检 `scenario.only_history_diff`）。`70/30` 在两组都不可查询。
- 最佳相同下的查询差异：{saved_text}。
- 历史可见性：历史组首轮起始历史包含 `70/30`；最终回复{'提及' if history_mentioned else '未提及'} `70/30` / `0.009115`；对照组最终回复{'提及' if control_mentioned else '未提及'} `70/30`。
- 解释边界：历史组可见 `70/30` 且其最终回复{'提及' if history_mentioned else '未提及'}该指标；但每组只有一次真实 LLM 回放，不能仅凭此断言因果。

"""
        else:
            ablation_section = "## 3. 历史消融对照\n\n至少一组未产出指标，无法完成历史消融比较。\n\n"
            ablation_conclusion = "历史消融：至少一组未完成，无法比较。"

    ablation_line = f"- {ablation_conclusion}\n" if is_ablation and ablation_conclusion else ""

    return f"""# 2026-11-09 · {title}

> 只读 `kuasar_small` 已有 5 组融合汇总 JSON；本地规则 Agent 与固定策略先冻结，再让真实 LLM 在场景 A/B 各运行一次。没有训练、重新召回或真实评测；除已配置 LLM 服务外没有其他网络请求。

## 结论

- {conclusion_line('A')}
- {conclusion_line('B')}
{ablation_line}- 两个场景的具体胜负与预算差异见下方逐场景表；未完成或截断的场景按实际状态记录，不判胜负。
- 结论只覆盖本次 128 用户 / 362 目标的离线回放与现有 5 组配额，不扩展到正式测试集或其他场景。

## 1. 场景与预算

| 场景 | 起始历史 | 可揭晓配额 | 预算 | 真实 LLM 请求上限 |
| --- | --- | --- | ---: | ---: |
{scenario_rows}

固定策略顺序：{fixed_text}，执行时跳过起始历史。本地规则沿用 Phase 13 的反馈规则；历史组的 `70/30` 作为已有反馈，不计预算。

## 2. 来源与冻结基线

- 数据集 / 划分：`{sources['manifest']['dataset']}` / `{sources['manifest']['split']}`；128 用户 / 362 目标；每用户 100 候选。
- 5 组来源共享缓存身份 `sha256:{sources['manifest']['recall_cache']['version']}`；重复记录冲突数为 0。
- 本地规则与固定策略轨迹在真实 LLM 启动前写入 `frozen/`；哈希：

| 场景 | 方 | 文件 | sha256 |
| --- | --- | --- | --- |
{frozen_rows}

- 读取的源工件（仅汇总 JSON）：

{source_rows}

{ablation_section}## {detail_sec}. 逐场景明细

{chr(10).join(scenario_blocks)}

## {self_sec}. 本地自检

- 状态：**{self_check['status']}**；通过 **{self_check['passed']} / {self_check['total']}**。
- 覆盖：单场景状态机既有 13 项 + 场景初始可见性、本地/固定策略、起始历史不扣预算、历史配额拒绝、A/B 正常与截断路径、请求上限。
- 前 8 条：

| 断言 | 状态 | 说明 |
| --- | --- | --- |
{self_rows}

完整结果见 [self_check.json](self_check.json)。

## {scope_sec}. 执行范围与合规

| 项 | 实际 |
| --- | --- |
| 执行命令 | `{run_meta['command']}` |
| 模型 / 端点 | `{run_meta['model']}` / `{run_meta['endpoint_host']}` |
| 真实 LLM 工作流 | A / B 各 1 次；总请求 {run_meta['llm_request_total']} / {6 if is_ablation else 5} |
| 训练 / 重新召回 / 真实评测 | 0 / 0 / 0 |
| 其他网络请求 | 0（仅已配置 LLM 服务） |
| 数据集 / 候选明细 / 缓存 / 实验记忆索引 | 未读取 |
| 来源与冻结基线哈希 | 运行前后一致（见 `run_meta.json`） |
| 旧工件 | 未改写；只写入新的运行子目录 |

## {accept_sec}. 验收对照

| 验收项 | 结论 | 证据 |
| --- | --- | --- |
| 三方公平 | 通过 | 两场景起始历史/候选/预算分别一致；本地基线先冻结，LLM 后运行 |
| 按选择揭晓 | 通过 | 首轮只含本组起始历史；成功返回只含当步配额；历史/重复/非法/超预算被拒绝 |
| 执行次数受限 | 通过 | A/B 每方 ≤2；LLM 总请求 {run_meta['llm_request_total']} / {6 if is_ablation else 5} |
| 如实比较效果 | 通过 | 逐场景分别报告最佳指标与预算；未完成场景 not_judged |
| 结论范围明确 | 通过 | 只限定本次回放，不外推到测试集或其他场景 |

## {boundary_sec}. 结论边界

本轮只覆盖 `kuasar_small` 的 128 个验证用户 / 362 个目标、两个固定场景、每场景最多一次真实 LLM 工作流。完整 Phase 13 正式验证仍保留为待办。
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 13 小规模多场景离线对照")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--self-check", action="store_true", help="只跑合成场景检查，不联网")
    parser.add_argument("--local-only", action="store_true", help="只生成并冻结本地基线，不调用真实 LLM")
    parser.add_argument("--paired", action="store_true", help="成对复验：3 对，A/B 顺序按配置")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config_bundle = load_scenario_config(args.config, args.base_config)
    except Exception as error:  # noqa: BLE001
        print(json.dumps({"status": "blocked", "reason": str(error)}, ensure_ascii=False, indent=2))
        return 2
    if args.self_check:
        result = run_self_check(config_bundle)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "pass" else 1

    output_root = args.output_root
    if args.paired and output_root.resolve() == DEFAULT_OUTPUT_ROOT.resolve():
        output_root = DEFAULT_PAIRED_OUTPUT_ROOT
    started_wall = _utc_now()
    run_dir = _make_run_dir(output_root)
    try:
        if args.local_only:
            sources = replay.load_sources(config_bundle["base"])
            _validate_scenarios_against_sources(config_bundle["scenarios"], sources, config_bundle["mode"])
            _write_json(run_dir / "sources.json", sources["manifest"])
            frozen = freeze_local(run_dir, config_bundle, sources["records"])
            _write_json(run_dir / "frozen" / "manifest.json", frozen["manifest"])
            (run_dir / "scenario_config.toml").write_text(
                config_bundle["path"].read_text(encoding="utf-8"), encoding="utf-8"
            )
            print(f"本地基线已冻结：{run_dir}")
            return 0
        if args.paired:
            if not config_bundle.get("paired"):
                raise ScenarioError("配置没有 [paired] 段，不能使用 --paired")
            result = execute_paired(config_bundle, output_root, run_dir, started_wall)
        else:
            result = execute(config_bundle, output_root, run_dir, started_wall)
    except Exception as error:  # noqa: BLE001
        _write_blocked(run_dir, llm._safe_error(error))
        print(f"[blocked] {_safe(error)}")
        print(f"运行目录：{run_dir}")
        return 2
    if args.paired:
        replication = result["overall"]["replication"]
        statuses = [
            workflow.get("status")
            for pair in result["overall"]["pairs"]
            for workflow in pair["workflows"]
        ]
        print(f"配对复验完成：{run_dir}")
        print(
            f"复现 {replication['reproduced_over_judgeable']} / 可判定 "
            f"{replication['judgeable_over_planned']}；请求 {result['run_meta']['llm_request_total']} / 18"
        )
        return 0 if all(status == "verified" for status in statuses) else 2
    statuses = result["overall"]["scenario_status"]
    print(f"回放完成：{run_dir}")
    print(f"场景状态：{statuses}")
    return 0 if all(value == "verified" for value in statuses.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
