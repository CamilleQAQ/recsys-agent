"""Phase 12 发布门槛的独立离线回归。

从 ``history_first_agent.py`` 移出；只负责编排模拟响应、只读夹具和断言，
不复制发布判断。它调用现有 ``run(...)`` / LangGraph / 索引逻辑：

* 模拟模型：历史查询 → 评测 → 最终回复超时；
* 评测工具返回已落盘的 ``80_20/metrics.json`` 只读夹具；
* 真实 LangGraph、真实历史查询/index 读取；
* 每次运行创建唯一输出目录，不删除、不覆盖旧证据。

独立运行::

    python -m recsys.experiments.regression_publish_gate --cache-dir artifacts/kuasar_small/mvp

旧 CLI 入口仍然可用（委托到本模块）::

    python -m recsys.experiments.history_first_agent --regression-publish-gate \
      --cache-dir artifacts/kuasar_small/mvp --quota 80/20
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from recsys.experiments.agent_tools import (
    TOOL_EVALUATE,
    TOOL_SEARCH_HISTORY,
    build_registry,
)
from recsys.experiments.experiment_history import (
    DEFAULT_HISTORY_ROOT,
    search_experiment_history,
)
from recsys.experiments.experiment_memory import (
    build_identity,
    identity_key,
    load_index,
)
from recsys.experiments.fusion_quota import (
    DEFAULT_CACHE,
    load_context,
    validate_quota,
)
from recsys.experiments.history_first_agent import parse_quota, run as run_agent

DEFAULT_OUTPUT_ROOT = Path("artifacts/kuasar_small/experiment_memory/regression_publish_gate")
FIXTURE = Path("artifacts/kuasar_small/experiment_memory/cache_eval/80_20/metrics.json")
PHASE12_MEMORY_DIR = Path("experiments/2026-11-08-kuasar_small-experiment-memory/memory")
DEFAULT_TARGET_QUOTA = (80, 20)
REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return REPO_ROOT / path


def _unique_run_dir(parent: Path) -> Path:
    """在父目录下创建唯一子目录；冲突换名，不做任何删除。"""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"run-{stamp}-{os.getpid()}"
    for index in range(100):
        candidate = parent / (base if index == 0 else f"{base}-{index}")
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"无法在 {parent} 下创建唯一回归目录（冲突过多）")


def _prepare_history(run_dir: Path) -> Path:
    """复制配置与 50/50 基准，但不放 80/20 记录，保证测试历史 miss。"""
    history_root = run_dir / "history"
    runs = history_root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    source_config = DEFAULT_HISTORY_ROOT / "config.toml"
    source_baseline = DEFAULT_HISTORY_ROOT / "runs" / "50_50.json"
    if not source_config.exists() or not source_baseline.exists():
        raise RuntimeError("缺少 Phase 11 历史配置或 50/50 基准，无法构造测试历史")
    shutil.copyfile(source_config, history_root / "config.toml")
    shutil.copyfile(source_baseline, runs / "50_50.json")
    return history_root


def _fake_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": f"resp-{call_id}",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    ],
                }
            }
        ],
    }


def _fake_reply(content: str) -> dict:
    return {
        "id": f"resp-final-{abs(hash(content)) % 100000}",
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }


def run(
    experiment_dir: Path | None = None,
    *,
    cache_dir: Path = DEFAULT_CACHE,
    cache_dir_source: str = "default",
    target_quota: tuple[int, int] = DEFAULT_TARGET_QUOTA,
) -> dict:
    """执行一次发布门槛回归；``experiment_dir`` 是输出父目录，不是固定覆盖目录。"""
    target_quota = validate_quota(*target_quota)
    parent = Path(experiment_dir) if experiment_dir is not None else DEFAULT_OUTPUT_ROOT
    run_dir = _unique_run_dir(parent)
    history_root = _prepare_history(run_dir)
    memory_dir = run_dir / "memory"
    output_dir = run_dir / "output"

    fixture_path = _resolve(FIXTURE)
    if not fixture_path.exists():
        raise RuntimeError(f"缺少只读评测夹具：{fixture_path}")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if (fixture.get("quota") or {}) != {
        "two_tower": target_quota[0],
        "mind": target_quota[1],
    }:
        raise RuntimeError("夹具配额与回归目标不一致")
    if not (fixture.get("evidence") or {}).get("fusion"):
        raise RuntimeError("夹具缺少候选证据路径")

    calls = {"chat": 0, "eval": 0}
    script = [
        _fake_call(
            "call-gate-history",
            TOOL_SEARCH_HISTORY,
            {"two_tower_quota": target_quota[0], "mind_quota": target_quota[1]},
        ),
        _fake_call(
            "call-gate-eval",
            TOOL_EVALUATE,
            {"two_tower_quota": target_quota[0], "mind_quota": target_quota[1]},
        ),
    ]

    def fake_chat(config, messages, *, tools, timeout):
        calls["chat"] += 1
        if script:
            return script.pop(0)
        raise TimeoutError("simulated timeout after evaluation result")

    def fake_eval(arguments, context) -> dict:
        calls["eval"] += 1
        return json.loads(json.dumps(fixture, ensure_ascii=False))

    registry = build_registry(evaluate_execute=fake_eval)
    summary = run_agent(
        run_dir,
        cache_dir=cache_dir,
        cache_dir_source=cache_dir_source,
        history_root=history_root,
        output_dir=output_dir,
        target_quota=target_quota,
        memory_dir=memory_dir,
        policy="local",
        chat_fn=fake_chat,
        registry=registry,
    )
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_names = [event.get("event") for event in events]
    index, index_path = load_index(memory_dir)
    context = load_context(cache_dir)
    key = identity_key(build_identity(context, target_quota))
    evaluation = summary.get("evaluation") or {}
    memory = summary.get("memory") or {}

    requery = search_experiment_history(
        target_quota[0],
        target_quota[1],
        cache_dir=cache_dir,
        history_root=history_root,
        memory_dir=memory_dir,
    )
    existing = search_experiment_history(
        target_quota[0],
        target_quota[1],
        cache_dir=cache_dir,
        history_root=DEFAULT_HISTORY_ROOT,
        memory_dir=PHASE12_MEMORY_DIR,
    )

    checks = {
        "status_partial": summary["status"] == "partial",
        "simulated_cli_exit_2": (0 if summary["status"] == "verified" else 2) == 2,
        "evaluation_executed_once": summary["evaluation_count"] == 1,
        "tool_result_kept": bool(evaluation.get("quota")),
        "candidate_evidence_kept": bool((evaluation.get("evidence") or {}).get("fusion")),
        "timeout_reason_kept": "simulated timeout" in str(summary.get("error") or ""),
        "no_memory_saved_event": "memory_saved" not in event_names,
        "memory_publish_skipped_event": "memory_publish_skipped" in event_names,
        "memory_not_saved": memory.get("saved") is False,
        "memory_publish_skipped": memory.get("publish") == "skipped",
        "eval_result_not_returned": memory.get("eval_result_returned") is False,
        "index_empty": index["entries"] == {},
        "key_not_indexed": key not in index["entries"],
        "requery_miss": requery.get("status") == "miss",
        "requery_without_second_eval": calls["eval"] == 1,
        "existing_verified_hit": existing.get("status") == "hit",
        "existing_verified_source": existing.get("source") == "memory_index",
        "existing_verified_key": bool(existing.get("memory_key")),
    }
    result = {
        "mode": "publish_gate_final_timeout",
        "policy": "local-deterministic",
        "fixture": str(fixture_path),
        "cache_dir": str(cache_dir),
        "history_root": str(history_root),
        "memory_dir": str(memory_dir),
        "evidence_dir": str(run_dir),
        "status": summary["status"],
        "simulated_cli_exit": 0 if summary["status"] == "verified" else 2,
        "request_count": summary["request_count"],
        "history_query_count": summary["history_query_count"],
        "evaluation_count": summary["evaluation_count"],
        "error": summary.get("error"),
        "evaluation": {
            "quota": evaluation.get("quota"),
            "hits": evaluation.get("hits"),
            "recall_at_100": evaluation.get("recall_at_100"),
            "evidence": evaluation.get("evidence"),
        },
        "memory": memory,
        "key": key,
        "index_path": str(index_path),
        "index_entry_count": len(index["entries"]),
        "events": event_names,
        "requery": {
            "status": requery.get("status"),
            "source": requery.get("source"),
            "reason": requery.get("reason"),
        },
        "existing_verified": {
            "status": existing.get("status"),
            "source": existing.get("source"),
            "memory_key": existing.get("memory_key"),
        },
        "fake_eval_calls": calls["eval"],
        "fake_chat_calls": calls["chat"],
        "checks": checks,
        "passed": all(checks.values()),
        "metrics_file": str(run_dir / "metrics.json"),
        "events_file": str(run_dir / "events.jsonl"),
    }
    (run_dir / "regression.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="输出父目录；省略时写 artifacts/kuasar_small/experiment_memory/regression_publish_gate",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--quota", type=parse_quota, default=DEFAULT_TARGET_QUOTA)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache_dir = Path(args.cache_dir) if args.cache_dir is not None else DEFAULT_CACHE
    cache_source = "cli" if args.cache_dir is not None else "default"
    try:
        result = run(
            args.experiment_dir,
            cache_dir=cache_dir,
            cache_dir_source=cache_source,
            target_quota=args.quota,
        )
    except Exception as error:  # noqa: BLE001 - 独立回归失败时给出清晰退出码
        print(f"[regression] {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[evidence] {result['evidence_dir']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
