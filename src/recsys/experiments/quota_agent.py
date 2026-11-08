"""Research Agent MVP：用真实评测工具做「融合配额」的决策闭环。

Agent 的任务：

> 每个用户最多 100 个候选，双塔和 MIND 怎么分配配额，才能保留更多验证目标？

它只有两种能力（Phase 11 的「不随意改代码」原则）：

1. 读实验上下文（聚合指标，不含用户 ID / 候选明细 / 标签）；
2. 调用 ``evaluate_fusion_quota`` 工具。

程序负责执行工具、校验预算、拒绝重复配置；LLM（若配置了）只负责选下一组并解释理由；
没有 LLM 时用确定性本地策略跑通同一闭环，并在报告里标记「真实 LLM 待验收」。

    python -m recsys.experiments.quota_agent --experiment-dir experiments/2026-11-08-kuasar_small-fusion-quota-agent
    python -m recsys.experiments.quota_agent --self-check        # 只跑边界用例，不访问缓存

LLM 配置（OpenAI 兼容接口，全部走环境变量，不落盘）：

    DSH_AGENT_LLM_BASE_URL=https://api.example.com/v1
    DSH_AGENT_LLM_API_KEY=...
    DSH_AGENT_LLM_MODEL=...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from recsys.experiments.fusion_quota import (
    ALLOWED_QUOTAS,
    DEFAULT_CACHE,
    DEFAULT_OUTPUT,
    QuotaError,
    evaluate_fusion_quota,
    fuse_by_quota,
    validate_quota,
)

#: 任务给定的聚合上下文（来自上一轮 MVP 报告与 run_meta，不含任何原始 ID）。
CONTEXT = {
    "dataset": "kuasar_small",
    "fixed_users": 128,
    "validation_targets": 362,
    "candidate_budget_per_user": 100,
    "k_recall_per_channel": 100,
    "previous_mvp": {
        "fusion_rule": "无配额限制的 50/50 轮流（两路各 100 条源候选）",
        "fusion_pool_hits": 4,
        "union_hits_at_100": 9,
        "two_tower_hits_at_100": 2,
        "mind_hits_at_100": 7,
    },
    "objective": "命中目标数优先，Recall@100 次之；持平时保留基准",
}


class BudgetExceeded(RuntimeError):
    """超过时间或实验次数预算。"""


class DuplicateExperiment(RuntimeError):
    """同一配额被重复评测。"""


@dataclass
class AgentRuntime:
    max_evals: int = 3
    budget_seconds: float = 180.0
    started: float = field(default_factory=time.perf_counter)
    evaluated: list[tuple[int, int]] = field(default_factory=list)

    def remaining_seconds(self) -> float:
        return self.budget_seconds - (time.perf_counter() - self.started)

    def check_can_run(self, quota: tuple[int, int]) -> None:
        if len(self.evaluated) >= self.max_evals:
            raise BudgetExceeded(f"已达到实验次数上限 {self.max_evals}")
        if self.remaining_seconds() <= 0:
            raise BudgetExceeded("已达到时间预算")
        if quota in self.evaluated:
            raise DuplicateExperiment(f"配额 {quota} 已经评测过，拒绝重复运行")
        if quota not in ALLOWED_QUOTAS:
            raise QuotaError(f"不支持的配额 {quota}")

    def record(self, quota: tuple[int, int]) -> None:
        self.evaluated.append(quota)


# --------------------------------------------------------------------------- LLM 客户端


def llm_available() -> bool:
    return all(
        os.environ.get(name)
        for name in ("DSH_AGENT_LLM_BASE_URL", "DSH_AGENT_LLM_API_KEY", "DSH_AGENT_LLM_MODEL")
    )


def ask_llm(messages: list[dict]) -> str:
    """最小 OpenAI 兼容 chat/completions 调用；失败时抛异常由调用方回退。"""
    base_url = os.environ["DSH_AGENT_LLM_BASE_URL"].rstrip("/")
    payload = json.dumps(
        {
            "model": os.environ["DSH_AGENT_LLM_MODEL"],
            "messages": messages,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['DSH_AGENT_LLM_API_KEY']}",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM 输出里没有 JSON")
    return json.loads(text[start : end + 1])


# --------------------------------------------------------------------------- 决策


def choose_next(step: int, results: list[dict], runtime: AgentRuntime, use_llm: bool) -> tuple[dict, str]:
    """返回 ({"quota": [t,m], "reason": "..."} , "llm"/"local")。"""
    baseline = results[0]
    if step == 1:
        if use_llm:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是推荐系统研究 Agent。只输出 JSON："
                        '{"quota": [双塔配额, MIND 配额], "reason": "中文一句话"}。'
                        "可选配额只有 70/30、30/70、80/20、20/80，总和必须 100。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"context": CONTEXT, "baseline_result": baseline, "allowed": ALLOWED_QUOTAS},
                        ensure_ascii=False,
                    ),
                },
            ]
            try:
                decision = extract_json(ask_llm(messages))
                quota = validate_quota(*decision["quota"])
                if quota == (50, 50):
                    raise QuotaError("LLM 选了与基准相同的 50/50")
                return {"quota": list(quota), "reason": str(decision.get("reason", ""))}, "llm"
            except Exception as error:  # noqa: BLE001 - 回退到本地策略并记录原因
                results[0].setdefault("llm_error", str(error)[:200])

        # 本地策略：用上一轮 MVP 的通道级命中做方向判断（聚合信息，不看标签明细）
        two_hits = CONTEXT["previous_mvp"]["two_tower_hits_at_100"]
        mind_hits = CONTEXT["previous_mvp"]["mind_hits_at_100"]
        quota = (70, 30) if two_hits > mind_hits else (30, 70)
        reason = (
            f"上一轮 MVP 的 Top-100 命中里 MIND={mind_hits}、双塔={two_hits}，"
            f"先把更多配额给更强的 MIND，取 {quota[0]}/{quota[1]} 做对照"
        )
        return {"quota": list(quota), "reason": reason}, "local"

    # step >= 2：决定第三组或停止
    best = max(results, key=lambda row: (row["hits"], row["recall_at_100"]))
    if len(runtime.evaluated) >= runtime.max_evals or runtime.remaining_seconds() <= 0:
        return {"quota": None, "reason": "预算已用完，停止"}, "program"
    if use_llm:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是推荐系统研究 Agent。根据已评测结果决定是否再测一组："
                    '输出 JSON：{"action": "run" 或 "stop", "quota": [t,m] 或 null, "reason": "..."}。'
                    "若继续，只能从未评测过的允许配额里选；结论按命中数优先、Recall@100 次之。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"context": CONTEXT, "results": results, "evaluated": runtime.evaluated},
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            decision = extract_json(ask_llm(messages))
            if decision.get("action") == "stop":
                return {"quota": None, "reason": str(decision.get("reason", "LLM 选择停止"))}, "llm"
            quota = validate_quota(*decision["quota"])
            return {"quota": list(quota), "reason": str(decision.get("reason", ""))}, "llm"
        except Exception as error:  # noqa: BLE001
            results[-1].setdefault("llm_error", str(error)[:200])

    # 本地策略：第二组若严格优于基准，沿同方向再探一档；否则停止
    improved = best["hits"] > baseline["hits"] or (
        best["hits"] == baseline["hits"] and best["recall_at_100"] > baseline["recall_at_100"]
    )
    if not improved:
        return {"quota": None, "reason": "第二组没有超过基准（命中数/Recall@100），停止并保留基准"}, "local"
    # 沿「更强通道」的方向再走一档：(30,70) -> (20,80)；(70,30) -> (80,20)
    if best["quota"]["mind"] > best["quota"]["two_tower"]:
        direction = (20, 80)
    else:
        direction = (80, 20)
    if direction in runtime.evaluated:
        return {"quota": None, "reason": "同方向更深一档已评测或不可用，停止"}, "local"
    reason = (
        f"第二组 {best['quota']} 已在命中数上超过基准（{best['hits']} vs {baseline['hits']}），"
        f"再沿同一方向测 {direction[0]}/{direction[1]} 确认趋势"
    )
    return {"quota": list(direction), "reason": reason}, "local"


# --------------------------------------------------------------------------- 自检


def run_self_check() -> dict:
    """用合成候选验证边界：非法配额、重复调用、次数/时间预算、去重与配额补位。"""
    checks: list[str] = []
    for bad in ((60, 40), (50, 40), (101, -1), (-1, 101)):
        try:
            validate_quota(*bad)
        except QuotaError:
            checks.append(f"拒绝非法配额 {bad}")
        else:
            raise AssertionError(f"非法配额 {bad} 未被拒绝")

    two = [{"item": i, "rank": i + 1} for i in range(10)]
    mind = [{"item": i, "rank": i + 1} for i in range(5, 15)]
    pool, stats = fuse_by_quota(two, mind, 5, 5, budget=10)
    items = [row["item"] for row in pool]
    assert len(items) == len(set(items)) == 10, "融合池出现重复或数量不对"
    assert all(row["sources"] for row in pool), "存在没有来源的候选"
    checks.append("合成候选：无重复、来源完整、配额 5/5 共 10 条")

    pool2, stats2 = fuse_by_quota(two[:3], mind, 5, 5, budget=10)
    assert len(pool2) <= 10 and stats2["fills"] > 0, "补位没有被记录"
    checks.append(f"合成候选：一路耗尽时补位 {stats2['fills']} 条")

    runtime = AgentRuntime(max_evals=2, budget_seconds=60)
    runtime.check_can_run((50, 50))
    runtime.record((50, 50))
    try:
        runtime.check_can_run((50, 50))
    except DuplicateExperiment:
        checks.append("拒绝重复配额")
    else:
        raise AssertionError("重复配额未被拒绝")
    runtime.record((30, 70))
    try:
        runtime.check_can_run((70, 30))
    except BudgetExceeded:
        checks.append("拒绝超次数预算")
    else:
        raise AssertionError("超次数预算未被拒绝")

    short = AgentRuntime(max_evals=3, budget_seconds=0.0)
    try:
        short.check_can_run((50, 50))
    except BudgetExceeded:
        checks.append("拒绝超时间预算")
    else:
        raise AssertionError("超时间预算未被拒绝")

    return {"passed": True, "checks": checks}


# --------------------------------------------------------------------------- 主流程


def run_agent(
    experiment_dir: Path,
    *,
    max_evals: int = 3,
    budget_seconds: float = 180.0,
    cache_dir: Path = DEFAULT_CACHE,
    output_dir: Path = DEFAULT_OUTPUT,
    force_local: bool = False,
) -> dict:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    events_path = experiment_dir / "events.jsonl"
    events_path.write_text("", encoding="utf-8")

    def log_event(payload: dict) -> None:
        with events_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    runtime = AgentRuntime(max_evals=max_evals, budget_seconds=budget_seconds)
    use_llm = llm_available() and not force_local
    log_event(
        {
            "event": "start",
            "context": CONTEXT,
            "llm_available": use_llm,
            "max_evals": max_evals,
            "budget_seconds": budget_seconds,
        }
    )
    (experiment_dir / "context.json").write_text(
        json.dumps(CONTEXT, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    results: list[dict] = []
    decisions: list[dict] = []

    # 第 1 组：基准 50/50（协议要求，不由 LLM 决定）
    runtime.check_can_run((50, 50))
    log_event({"event": "decision", "step": 1, "quota": [50, 50], "reason": "协议要求先跑基准 50/50", "by": "program"})
    result = evaluate_fusion_quota(50, 50, cache_dir=cache_dir, output_dir=output_dir, tag="50_50")
    runtime.record((50, 50))
    results.append(result)
    log_event({"event": "tool_result", "step": 1, "result": result})

    step = 1
    while len(results) < max_evals and runtime.remaining_seconds() > 0:
        step += 1
        try:
            decision, by = choose_next(step - 1, results, runtime, use_llm)
        except Exception as error:  # noqa: BLE001
            log_event({"event": "decision_error", "step": step, "error": str(error)[:300]})
            break
        if decision["quota"] is None:
            decisions.append({"step": step, "action": "stop", "reason": decision["reason"], "by": by})
            log_event({"event": "decision", "step": step, "action": "stop", "reason": decision["reason"], "by": by})
            break
        quota = tuple(int(x) for x in decision["quota"])
        try:
            runtime.check_can_run(quota)
        except (BudgetExceeded, DuplicateExperiment, QuotaError) as error:
            decisions.append({"step": step, "action": "stop", "reason": f"程序拦截：{error}", "by": "program"})
            log_event({"event": "guard", "step": step, "error": str(error)})
            break
        decisions.append({"step": step, "action": "run", "quota": list(quota), "reason": decision["reason"], "by": by})
        log_event({"event": "decision", "step": step, "quota": list(quota), "reason": decision["reason"], "by": by})
        result = evaluate_fusion_quota(*quota, cache_dir=cache_dir, output_dir=output_dir, tag=f"{quota[0]}_{quota[1]}")
        runtime.record(quota)
        results.append(result)
        log_event({"event": "tool_result", "step": step, "result": result})

    # 结论：命中数优先，Recall@100 次之，持平保留基准
    def score(row: dict) -> tuple:
        return (row["hits"], row["recall_at_100"], -abs(row["quota"]["two_tower"] - 50))

    best = max(results, key=score)
    baseline = results[0]
    if decisions and decisions[-1]["action"] == "run":
        stop_reason = f"达到实验次数上限（{len(results)} 组），按规则选最优"
    elif decisions:
        stop_reason = decisions[-1]["reason"]
    else:
        stop_reason = "未产生决策记录"
    conclusion = {
        "best_by_rule": best["quota"],
        "rule": "命中目标数优先，Recall@100 次之；持平时保留基准",
        "baseline": {
            "quota": baseline["quota"],
            "hits": baseline["hits"],
            "recall_at_100": baseline["recall_at_100"],
        },
        "delta_vs_baseline": {
            "hits": best["hits"] - baseline["hits"],
            "recall_at_100": round(best["recall_at_100"] - baseline["recall_at_100"], 6),
        },
        "evaluated": [row["quota"] for row in results],
        "elapsed_s": round(time.perf_counter() - runtime.started, 2),
        "stop_reason": stop_reason,
        "llm_used": use_llm,
        "llm_note": (
            "有真实 LLM 决策轨迹" if use_llm else "未配置 LLM，使用本地确定性策略；真实 LLM 待验收"
        ),
    }
    summary = {
        "context": CONTEXT,
        "results": results,
        "decisions": decisions,
        "conclusion": conclusion,
        "events_file": str(events_path),
        "llm_used": use_llm,
    }
    (experiment_dir / "agent_result.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log_event({"event": "finish", "conclusion": conclusion})
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/2026-11-08-kuasar_small-fusion-quota-agent"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-evals", type=int, default=3)
    parser.add_argument("--budget-seconds", type=float, default=180.0)
    parser.add_argument("--force-local", action="store_true", help="忽略 LLM 配置，强制本地策略")
    parser.add_argument("--self-check", action="store_true", help="只跑边界用例，不访问缓存")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_check:
        result = run_self_check()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    summary = run_agent(
        args.experiment_dir,
        max_evals=args.max_evals,
        budget_seconds=args.budget_seconds,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        force_local=args.force_local,
    )
    print(json.dumps(summary["conclusion"], indent=2, ensure_ascii=False))
    for row in summary["results"]:
        print(
            f"{row['quota']['two_tower']}/{row['quota']['mind']}: "
            f"hits={row['hits']} recall@100={row['recall_at_100']} "
            f"candidates={row['candidates_total']} fills={row['fills_total']} {row['elapsed_s']}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
