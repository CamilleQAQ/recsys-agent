"""真实 LLM 原生 tool_calls 验收入口（Phase 11 验收）。

与 ``quota_agent.py`` 的区别：这里**不接受**「模型返回 JSON 再由 Python 解析」的路线，
而是要求服务商返回结构化的 ``tool_calls``，由程序按 ``tool_call_id`` 执行工具，
再用同一个 ID 把工具结果回传给模型。没有真实 LLM 配置时显式失败（不回落本地策略）。

    $env:DSH_AGENT_LLM_BASE_URL="https://api.deepseek.com/v1"
    $env:DSH_AGENT_LLM_API_KEY="sk-..."
    $env:DSH_AGENT_LLM_MODEL="deepseek-chat"
    python -m recsys.experiments.llm_tool_call --experiment-dir experiments/2026-11-08-kuasar_small-llm-tool-call

约束：

* 只读上一轮 MVP 缓存与已落盘的 50/50 基准，训练 0 次、重新召回 0 次；
* 最多 2 次模型请求、恰好 1 次由 LLM 发起的工具调用；
* 单次运行 180 秒预算，超时/失败不自动重试；
* 发给模型的内容只有聚合指标与配置；原始 ID、候选明细、标签、密钥都不入上下文与日志。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from recsys.experiments.fusion_quota import (
    ALLOWED_QUOTAS,
    DEFAULT_CACHE,
    QuotaError,
    evaluate_fusion_quota,
    load_context,
    tool_schema,
    validate_quota,
)

EXPERIMENT_DEFAULT = Path("experiments/2026-11-08-kuasar_small-llm-tool-call")
BASELINE_RUN = Path("experiments/2026-11-08-kuasar_small-fusion-quota-agent/runs/50_50.json")
TOOL_NAME = "evaluate_fusion_quota"
ALLOWED_AFTER_BASELINE = tuple(q for q in ALLOWED_QUOTAS if q != (50, 50))
BUDGET_SECONDS = 180.0
LLM_ENV = ("DSH_AGENT_LLM_BASE_URL", "DSH_AGENT_LLM_API_KEY", "DSH_AGENT_LLM_MODEL")


class VerificationError(RuntimeError):
    """验收条件不满足；调用方应记录停点并标记未验收。"""


def sha_json(payload: dict) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def load_dotenv(path: Path | None = None) -> bool:
    """读取仓库根目录的 ``.env``（若存在），只补未设置的环境变量。

    只用于本地开发便利：不覆盖已存在的环境变量，不打印任何值。
    返回是否成功读取到文件。
    """
    env_path = path or Path(__file__).resolve().parents[3] / ".env"
    if not env_path.exists():
        return False
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
    return True


def llm_config() -> dict[str, str] | None:
    values = {name: os.environ.get(name, "").strip() for name in LLM_ENV}
    if not all(values.values()):
        return None
    return {
        "base_url": values["DSH_AGENT_LLM_BASE_URL"],
        "api_key": values["DSH_AGENT_LLM_API_KEY"],
        "model": values["DSH_AGENT_LLM_MODEL"],
    }


def host_only(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def post_chat(
    config: dict[str, str],
    messages: list[dict],
    *,
    tools: list[dict] | None,
    timeout: float,
) -> dict:
    payload: dict[str, Any] = {
        "model": config["model"],
        "messages": messages,
        "temperature": 0.0,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    request = urllib.request.Request(
        f"{config['base_url'].rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config['api_key']}",
        },
    )
    with urllib.request.urlopen(request, timeout=max(1.0, timeout)) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_single_tool_call(response: dict, *, baseline_quota: tuple[int, int]) -> dict:
    """从服务商响应里取出唯一合法的工具调用；普通文本 JSON 一律不接受。"""
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message", {})
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        raise VerificationError("响应里没有 tool_calls（普通文本/JSON 回复不计入验收）")
    if len(tool_calls) != 1:
        raise VerificationError(f"tool_calls 数量为 {len(tool_calls)}，要求恰好 1 条")
    call = tool_calls[0]
    function = call.get("function") or {}
    if (call.get("type") or "function") != "function":
        raise VerificationError(f"不支持的 tool call 类型：{call.get('type')}")
    name = function.get("name")
    if name != TOOL_NAME:
        raise VerificationError(f"未知工具：{name}")
    if not call.get("id"):
        raise VerificationError("tool_call 缺少 id")
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError as error:
        raise VerificationError(f"工具参数不是合法 JSON：{error}") from error
    for key in ("two_tower_quota", "mind_quota"):
        if key not in arguments or not isinstance(arguments[key], int):
            raise VerificationError(f"工具参数缺少整数 {key}")
    quota = validate_quota(arguments["two_tower_quota"], arguments["mind_quota"])
    if quota == baseline_quota:
        raise VerificationError("模型选择了基准配额；本轮要求对照组")
    if quota not in ALLOWED_AFTER_BASELINE:
        raise VerificationError(f"配额不在允许集合：{quota}")
    return {"id": call["id"], "name": name, "arguments": arguments, "quota": quota, "message": message}


def run_self_check() -> dict:
    """用合成响应验证解析与守卫（不访问网络、不评测缓存）。"""
    checks: list[str] = []
    valid = {
        "id": "resp-1",
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": TOOL_NAME,
                                "arguments": json.dumps({"two_tower_quota": 30, "mind_quota": 70}),
                            },
                        }
                    ],
                }
            }
        ],
    }
    parsed = extract_single_tool_call(valid, baseline_quota=(50, 50))
    assert parsed["quota"] == (30, 70)
    checks.append("接受合法的单条 tool_call 并解析参数")

    plain = {"id": "r", "choices": [{"message": {"content": '{"two_tower_quota": 30, "mind_quota": 70}'}}]}
    try:
        extract_single_tool_call(plain, baseline_quota=(50, 50))
    except VerificationError:
        checks.append("拒绝普通文本 JSON（无 tool_calls）")
    else:
        raise AssertionError("普通 JSON 回复未被拒绝")

    two_calls = json.loads(json.dumps(valid))
    two_calls["choices"][0]["message"]["tool_calls"].append(two_calls["choices"][0]["message"]["tool_calls"][0])
    try:
        extract_single_tool_call(two_calls, baseline_quota=(50, 50))
    except VerificationError:
        checks.append("拒绝多条 tool_calls")
    else:
        raise AssertionError("多条 tool_calls 未被拒绝")

    unknown = json.loads(json.dumps(valid))
    unknown["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "train_model"
    try:
        extract_single_tool_call(unknown, baseline_quota=(50, 50))
    except VerificationError:
        checks.append("拒绝未知工具")
    else:
        raise AssertionError("未知工具未被拒绝")

    illegal = json.loads(json.dumps(valid))
    illegal["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"two_tower_quota": 60, "mind_quota": 40}
    )
    try:
        extract_single_tool_call(illegal, baseline_quota=(50, 50))
    except QuotaError:
        checks.append("拒绝非法配额 60/40")
    else:
        raise AssertionError("非法配额未被拒绝")

    baseline_choice = json.loads(json.dumps(valid))
    baseline_choice["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"two_tower_quota": 50, "mind_quota": 50}
    )
    try:
        extract_single_tool_call(baseline_choice, baseline_quota=(50, 50))
    except VerificationError:
        checks.append("拒绝重复基准 50/50")
    else:
        raise AssertionError("重复基准未被拒绝")

    return {"passed": True, "checks": checks}
    """读取已落盘的 50/50 基准，并核对缓存指纹仍与当前一致。"""
    if not BASELINE_RUN.exists():
        raise VerificationError(f"缺少基准文件：{BASELINE_RUN}（先跑 2.23 的本地 Agent）")
    baseline = json.loads(BASELINE_RUN.read_text(encoding="utf-8"))
    context = load_context(DEFAULT_CACHE)  # 会重新核对 128 用户 / 362 目标与缓存完整性
    current = context.fingerprints
    recorded = baseline.get("evidence", {}).get("fingerprints", {})
    keys = ("users", "items", "split_strategy", "split_sha256", "id_mapping_sha256", "routing_seed", "k_recall")
    mismatched = [key for key in keys if recorded.get(key) != current.get(key)]
    if mismatched:
        raise VerificationError(f"基准指纹与当前缓存不一致：{mismatched}")
    based_on = {
        "quota": baseline["quota"],
        "hits": baseline["hits"],
        "recall_at_100": baseline["recall_at_100"],
        "coverage": baseline["coverage"],
        "candidates_total": baseline["candidates_total"],
        "users": baseline["users"],
        "targets": baseline["targets"],
        "fingerprints_verified": True,
    }
    return based_on, current


def build_messages(baseline: dict) -> list[dict]:
    """只发送配置与聚合指标：无用户 ID、无候选明细、无标签、无密钥。"""
    return [
        {
            "role": "system",
            "content": (
                "你是推荐系统研究 Agent。你必须调用提供的工具来评测一组融合配额，"
                "不要用普通文本代替工具调用。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": (
                        "每个用户最多保留 100 个候选时，双塔和 MIND 怎么分配配额，"
                        "才能保留更多验证目标？"
                    ),
                    "candidate_budget_per_user": 100,
                    "quota_order": "双塔 / MIND",
                    "allowed_quotas": [list(pair) for pair in ALLOWED_AFTER_BASELINE],
                    "baseline_quota": baseline["quota"],
                    "baseline_aggregate_metrics": {
                        "users": baseline["users"],
                        "validation_targets": baseline["targets"],
                        "hits": baseline["hits"],
                        "recall_at_100": baseline["recall_at_100"],
                        "coverage": baseline["coverage"],
                    },
                    "selection_rule": "命中目标数优先，Recall@100 次之；持平时保留基准",
                    "instruction": "从 allowed_quotas 中选一组，调用 evaluate_fusion_quota 工具。",
                },
                ensure_ascii=False,
            ),
        },
    ]


def run(experiment_dir: Path, *, cache_dir: Path = DEFAULT_CACHE) -> dict:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    events_path = experiment_dir / "events.jsonl"
    events_path.write_text("", encoding="utf-8")

    def log_event(payload: dict) -> None:
        with events_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    started = time.perf_counter()

    def remaining() -> float:
        return BUDGET_SECONDS - (time.perf_counter() - started)

    load_dotenv()  # 本地 .env 只补未设置的环境变量
    config = llm_config()
    if config is None:
        missing = [name for name in LLM_ENV if not os.environ.get(name)]
        reason = f"缺少真实 LLM 配置：{', '.join(missing)}"
        log_event({"event": "blocked", "reason": reason, "elapsed_s": round(time.perf_counter() - started, 2)})
        result = {"status": "blocked", "reason": reason, "llm_env": list(LLM_ENV)}
        (experiment_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result

    log_event({"event": "llm_config", "model": config["model"], "endpoint_host": host_only(config["base_url"])})
    try:
        baseline, fingerprints = load_baseline()
    except (VerificationError, QuotaError) as error:
        log_event({"event": "blocked", "reason": str(error)})
        result = {"status": "blocked", "reason": str(error)}
        (experiment_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result
    log_event({"event": "baseline_loaded", "baseline": baseline})
    (experiment_dir / "baseline.json").write_text(
        json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    tools = [tool_schema()]
    log_event({"event": "tools_registered", "tools": [TOOL_NAME], "tool_choice": "auto"})

    messages = build_messages(baseline)
    request_timeout = min(30.0, max(1.0, remaining()))
    log_event({"event": "llm_request", "request": 1, "timeout_s": round(request_timeout, 1), "tools": True})
    try:
        response = post_chat(config, messages, tools=tools, timeout=request_timeout)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        log_event({"event": "blocked", "reason": f"第 1 次模型请求失败：{error}"})
        result = {"status": "blocked", "reason": f"第 1 次模型请求失败：{error}"}
        (experiment_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result

    response_id = response.get("id")
    raw_message = (response.get("choices") or [{}])[0].get("message", {})
    raw_calls = raw_message.get("tool_calls") or []
    log_event(
        {
            "event": "llm_response",
            "request": 1,
            "response_id": response_id,
            "tool_calls": [
                {
                    "id": call.get("id"),
                    "type": call.get("type"),
                    "name": (call.get("function") or {}).get("name"),
                    "arguments": (call.get("function") or {}).get("arguments"),
                }
                for call in raw_calls
            ],
            "content_preview": (raw_message.get("content") or "")[:200],
        }
    )

    baseline_quota = (int(baseline["quota"]["two_tower"]), int(baseline["quota"]["mind"]))
    try:
        parsed = extract_single_tool_call(response, baseline_quota=baseline_quota)
    except (VerificationError, QuotaError) as error:
        log_event({"event": "blocked", "reason": str(error), "response_id": response_id})
        result = {"status": "blocked", "reason": str(error), "response_id": response_id}
        (experiment_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result

    call_id = parsed["id"]
    quota = parsed["quota"]
    message = parsed["message"]

    log_event({"event": "tool_call_accepted", "tool_call_id": call_id, "quota": list(quota)})
    tool_started = time.perf_counter()
    output_dir = Path("artifacts/kuasar_small/llm_tool_call")
    tag = f"{quota[0]}_{quota[1]}"
    result = evaluate_fusion_quota(quota[0], quota[1], cache_dir=cache_dir, output_dir=output_dir, tag=tag)
    tool_elapsed = round(time.perf_counter() - tool_started, 2)
    log_event(
        {
            "event": "tool_executed",
            "tool_call_id": call_id,
            "quota": list(quota),
            "result": {
                key: result[key]
                for key in (
                    "users",
                    "targets",
                    "candidates_total",
                    "candidates_mean",
                    "fills_total",
                    "hits",
                    "recall_at_100",
                    "coverage",
                    "elapsed_s",
                    "diagnostic_union_hits_at_100",
                )
            },
            "tool_elapsed_s": tool_elapsed,
            "evidence": result["evidence"]["fusion"],
        }
    )

    followup = messages + [
        {"role": "assistant", "content": message.get("content") or "", "tool_calls": tool_calls},
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(
                {
                    "quota": result["quota"],
                    "hits": result["hits"],
                    "recall_at_100": result["recall_at_100"],
                    "coverage": result["coverage"],
                    "candidates_total": result["candidates_total"],
                    "fills_total": result["fills_total"],
                },
                ensure_ascii=False,
            ),
        },
        {
            "role": "user",
            "content": (
                "这是真实工具结果。请对比基准与该配额，用一句话给出结论和停止理由；"
                "不要再调用工具，也不要输出 JSON 代码块。"
            ),
        },
    ]
    request2_timeout = min(30.0, max(1.0, remaining()))
    log_event(
        {"event": "tool_result_sent", "tool_call_id": call_id, "request": 2, "timeout_s": round(request2_timeout, 1)}
    )
    final_text = ""
    final_response_id = None
    try:
        response2 = post_chat(config, followup, tools=None, timeout=request2_timeout)
        final_response_id = response2.get("id")
        final_text = ((response2.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        log_event({"event": "llm_final", "response_id": final_response_id, "content": final_text})
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        log_event({"event": "final_request_failed", "error": str(error)})

    baseline_hits = baseline["hits"]
    best_is_new = result["hits"] > baseline_hits or (
        result["hits"] == baseline_hits and result["recall_at_100"] > baseline["recall_at_100"]
    )
    conclusion = {
        "baseline_quota": baseline["quota"],
        "chosen_quota": result["quota"],
        "baseline_hits": baseline_hits,
        "chosen_hits": result["hits"],
        "delta_hits": result["hits"] - baseline_hits,
        "delta_recall_at_100": round(result["recall_at_100"] - baseline["recall_at_100"], 6),
        "winner": "chosen" if best_is_new else "baseline",
        "final_model_reply": final_text,
    }
    summary = {
        "status": "verified",
        "response_ids": {"tool_call": response_id, "final": final_response_id},
        "tool_call_id": call_id,
        "baseline": baseline,
        "chosen": {
            key: result[key]
            for key in (
                "quota",
                "users",
                "targets",
                "candidates_total",
                "candidates_mean",
                "fills_total",
                "hits",
                "recall_at_100",
                "coverage",
                "elapsed_s",
            )
        },
        "conclusion": conclusion,
        "budget": {
            "model_requests": 2 if final_response_id else 1,
            "tool_calls": 1,
            "elapsed_s": round(time.perf_counter() - started, 2),
            "limit_seconds": BUDGET_SECONDS,
        },
        "evidence": {
            "events": str(events_path),
            "candidates": result["evidence"]["fusion"],
            "fingerprints": result["evidence"]["fingerprints"],
        },
    }
    log_event({"event": "finish", "conclusion": conclusion})
    (experiment_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT_DEFAULT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--self-check", action="store_true", help="用合成响应验证解析与守卫（不访问网络）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_check:
        print(json.dumps(run_self_check(), indent=2, ensure_ascii=False))
        return 0
    try:
        summary = run(args.experiment_dir, cache_dir=args.cache_dir)
    except VerificationError as error:
        print(f"[未验收] {error}", file=sys.stderr)
        return 2
    if summary["status"] != "verified":
        print(f"[未验收] {summary['reason']}", file=sys.stderr)
        return 2
    print(json.dumps(summary["conclusion"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
