"""成对复验请求计数回归（本地替身，不联网、不调用真实 LLM）。

    $env:PYTHONPATH = "src"
    python scripts/check_paired_request_accounting.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.experiments import offline_replay_llm as llm  # noqa: E402
from recsys.experiments import offline_replay_scenarios as sc  # noqa: E402
from recsys.experiments.offline_replay import utc_now  # noqa: E402

BUNDLE_PATH = ROOT / "configs" / "experiments" / "offline_replay_70_30_ablation.toml"
TEMP_ROOT = ROOT / "artifacts" / "kuasar_small" / "paired_accounting_regression"


def fake_chat(config, messages, *, tools, timeout):
    first_user = next(item for item in messages if item.get("role") == "user")
    payload = json.loads(first_user["content"])
    starts = payload.get("starting_histories") or [payload.get("starting_history")]
    is_history = any(item["quota"] == "70/30" for item in starts)
    tool_messages = [item for item in messages if item.get("role") == "tool"]
    if not tool_messages:
        return llm._tool_response("call-1", 30, 70)
    if is_history:
        return llm._text_response("历史组停止")
    if len(tool_messages) == 1:
        return llm._tool_response("call-2", 80, 20)
    return llm._text_response("对照组最终停止")


def run_case(name: str, failure: str | None) -> dict[str, Any]:
    root = TEMP_ROOT / name
    if root.exists():
        shutil.rmtree(root)
    run_dir = sc._make_run_dir(root)
    bundle = sc.load_scenario_config(BUNDLE_PATH)
    original_post = llm.post_chat
    original_build = llm.build_llm_trace
    original_write = sc._write_json
    hit = {"done": False}
    llm.post_chat = fake_chat
    if failure == "build_trace":
        def failing_build(final, base, meta):
            if meta.get("pair") == 1 and meta.get("condition") == "A" and not hit["done"]:
                hit["done"] = True
                raise RuntimeError("simulated build_llm_trace failure")
            return original_build(final, base, meta)

        llm.build_llm_trace = failing_build
    elif failure == "write_artifact":
        def failing_write(path, payload):
            if not hit["done"] and str(path).replace("\\", "/").endswith(
                "pairs/pair-1/A/llm_trace.json"
            ):
                hit["done"] = True
                raise OSError("simulated artifact write failure")
            return original_write(path, payload)

        sc._write_json = failing_write

    try:
        result = sc.execute_paired(bundle, root, run_dir, utc_now())
        pair1 = result["overall"]["pairs"][0]
        entries = {item["condition"]: item for item in pair1["workflows"]}
        group_counts = [
            item["request_count"]
            for pair in result["overall"]["pairs"]
            for item in pair["workflows"]
        ]
        outcome = {
            "case": name,
            "failure": failure,
            "total_request_count": result["run_meta"]["llm_request_total"],
            "group_request_counts": group_counts,
            "group_sum": sum(group_counts),
            "pair1_a_status": entries["A"]["status"],
            "pair1_a_request_count": entries["A"]["request_count"],
            "pair1_judgment": pair1["judgment"]["status"],
            "replication": result["overall"]["replication"],
        }
        assert outcome["total_request_count"] == 15, outcome
        assert outcome["group_sum"] == 15, outcome
        if failure:
            assert entries["A"]["request_count"] == 3, outcome
            assert entries["A"]["status"] == "blocked", outcome
            assert pair1["judgment"]["status"] == "not_judgeable", outcome
        else:
            assert group_counts == [3, 2, 2, 3, 3, 2], outcome
            assert result["overall"]["replication"]["reproduced_pairs"] == 3, outcome
        return outcome
    finally:
        llm.post_chat = original_post
        llm.build_llm_trace = original_build
        sc._write_json = original_write
        if root.exists():
            shutil.rmtree(root)


def main() -> int:
    cases = []
    try:
        cases.append(run_case("normal", None))
        cases.append(run_case("failure_build_trace", "build_trace"))
        cases.append(run_case("failure_write_artifact", "write_artifact"))
    except Exception as error:  # noqa: BLE001
        print(json.dumps({"status": "fail", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False, indent=2))
        return 1
    print(
        json.dumps(
            {
                "status": "pass",
                "cases": cases,
                "note": "fake model only; no real LLM/network/evaluation",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
