"""Code-editing research adapted from karpathy/autoresearch's program.md.

The agent writes rank(groups, context), not a choice from a configuration menu.
The fixed evaluator owns labels. Each attempt has its own source, predictions,
feedback and shared-memory record. The caller's budget is used directly.
"""
from __future__ import annotations

import difflib
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from recsys.evaluation.metrics import compute_ranking_metrics
from recsys.experiments.llm_tool_call import load_dotenv, llm_config, post_chat
from recsys.platform.budget import BudgetExceeded, BudgetPolicy, TaskBudget
from recsys.platform.shared_experiments import (
    SharedExperimentStore, build_record, reuse_reasons, resolve_path,
)

BUSINESS_MODULES = {
    "recommendation": "recsys.experiments.rec_agent",
    "search": "recsys.experiments.search_agent",
    "ads": "recsys.ads.agent",
}
IDENTITY = '''def rank(groups, context):
    return {str(g["id"]): [r["item"] for r in g["candidates"]] for g in groups}
'''
SUBMIT = {"type": "function", "function": {
    "name": "submit_algorithm",
    "description": "提交完整 Python 算法源码并执行真实验证；不是配置选择。",
    "parameters": {"type": "object", "properties": {
        "hypothesis": {"type": "string"},
        "feedback_analysis": {"type": "string", "description": "上一轮哪些证据改变了判断；首轮分析基线和历史"},
        "code": {"type": "string", "description": "完整源码，实现 rank(groups, context) 返回组ID到有序item列表"},
    }, "required": ["hypothesis", "feedback_analysis", "code"], "additionalProperties": False},
}}


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate(case, rankings):
    """Validate identities, then use existing ranking metrics and cold/warm breakdown."""
    expected = {str(g["id"]): g for g in case["groups"]}
    if set(rankings) != set(expected):
        raise ValueError("算法输出的组 ID 与实验输入不一致")
    for group_id, items in rankings.items():
        available = {r["item"] for r in expected[group_id]["candidates"]}
        if len(items) != len(set(items)) or not set(items) <= available:
            raise ValueError(f"{group_id}: 重复或未知候选")
        if len(items) < min(max(case["ks"]), len(available)):
            raise ValueError(f"{group_id}: 输出候选不足，不能通过少输出规避评估")
    truth = {str(g["id"]): set(g["positives"]) for g in case["groups"] if g["positives"]}
    metrics = compute_ranking_metrics(rankings, truth, tuple(case["ks"]))
    cold_ids = set(case.get("cold_items", [r["item"] for g in case["groups"] for r in g["candidates"] if r.get("cold", False)]))
    segments = {}
    for cold in (False, True):
        part = {}
        for group in case["groups"]:
            positives = {i for i in group["positives"] if (i in cold_ids) == cold}
            if positives:
                part[str(group["id"])] = positives
        segments["cold" if cold else "warm"] = {
            "positive_groups": len(part), "positive_count": sum(map(len, part.values())),
            "metrics": compute_ranking_metrics(rankings, part, tuple(case["ks"])) if part else {},
        }
    k = min(case["ks"])
    details = []
    for group in case["groups"]:
        group_id = str(group["id"])
        targets = set(group["positives"])
        details.append({"group": group_id, "positive_count": len(targets),
                        "hits": len(set(rankings[group_id][:k]) & targets)})
    return {"metrics": metrics, "segments": segments, "groups": len(expected),
            "positive_count": sum(map(len, truth.values())),
            "positives_outside_candidates": sum(len(set(g["positives"]) - {r["item"] for r in g["candidates"]}) for g in case["groups"]),
            "groups_without_positives": len(expected) - len(truth), "per_group": details}


def execute(algorithm_path, public_input, output_dir, timeout):
    """One process, one timeout, no polling/retry loop. Labels are not worker input."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "input.json"
    output_path = output_dir / "rankings.json"
    write_json(input_path, public_input)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    with (output_dir / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-B", "-m", "recsys.platform.autoresearch",
                        "--worker", str(algorithm_path), str(input_path), str(output_path)],
                       stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True, env=env)
    return json.loads(output_path.read_text(encoding="utf-8"))


def run_research(task, *, business, case_path, work_dir, scope=None, store=None,
                 rounds=2, client=None, baseline_feedback=None):
    """Run the existing business agent's bounded code/experiment/feedback loop."""
    if rounds < 1:
        raise ValueError("研究至少需要一次代码修改实验")
    case_path = resolve_path(case_path)
    case_bytes = case_path.read_bytes()
    case = json.loads(case_bytes)
    if case["business"] != business or case["split"] != "validation":
        raise ValueError("业务必须与实验匹配，研究反馈只能使用 validation")
    protocol_hash = hashlib.sha256(case_bytes + inspect.getsource(evaluate).encode()).hexdigest()
    run_id = f"research-{business}-{uuid.uuid4().hex[:10]}"
    work_dir = Path(work_dir).resolve() / run_id
    work_dir.mkdir(parents=True)
    scope = scope or TaskBudget(run_id, BudgetPolicy(
        max_llm_requests=rounds, max_experiments=rounds+1,
        max_tool_calls=rounds+3, max_history_lookups=1, max_seconds=180))
    store = store if isinstance(store, SharedExperimentStore) else SharedExperimentStore(
        resolve_path(store) if store else work_dir / "memory")
    load_dotenv()
    config = llm_config()
    if client is None and not config:
        raise ValueError("算法研究需要真实 LLM；未配置时不使用脚本冒充")
    public = {"groups": [{k: v for k, v in g.items() if k != "positives"} for g in case["groups"]],
              "context": case.get("context", {})}
    # Keep a local protocol snapshot; metric/labels stay in the evaluator process.
    write_json(work_dir / "case.json", case)
    scope.spend("tool")
    scope.spend("history")
    history = []
    for record in store.latest():
        if (record.get("business") == business
                and record.get("config", {}).get("protocol_sha256") == protocol_hash
                and not reuse_reasons(record, {"dataset": case["dataset"], "split": "validation"})):
            history.append({"record_id": record["record_id"], "result": record["result"],
                            "code": Path(record["artifacts"]["algorithm"]).read_text(encoding="utf-8")})
    history = history[-8:]
    write_json(work_dir / "history.json", history)
    program = (Path(__file__).parent / "research_program.md").read_text(encoding="utf-8")
    source = {name: resolve_path(name).read_text(encoding="utf-8") for name in case.get("source_files", [])}
    messages = [{"role": "system", "content": program + f"\n当前角色：{business} 业务 Agent。"},
                {"role": "user", "content": json.dumps({"task": task, "objective": case["objective"],
                    "rounds": rounds, "context": public["context"], "source_files": source,
                    "candidate_examples": public["groups"][:2], "history": history}, ensure_ascii=False)}]
    attempts = []
    best = None
    previous_code = case.get("baseline_code", IDENTITY)
    stopped = "round_limit"
    started = time.perf_counter()
    for iteration in range(rounds + 1):
        cached_baseline = iteration == 0 and baseline_feedback is not None
        trial = work_dir / f"trial_{iteration:02d}"
        trial.mkdir()
        hypothesis = "固定基线"
        analysis = "执行相同协议的原始算法，建立基线"
        code = previous_code
        try:
            scope.check("seconds")
            if not cached_baseline:
                scope.check("experiment")
                scope.check("tool")
            if iteration:
                scope.spend("llm")
                response = (client or post_chat)(config, messages, tools=[SUBMIT],
                    timeout=max(1, min(60, scope.remaining("seconds"))))
                write_json(trial / "llm_response.json", response)
                message = response["choices"][0]["message"]
                calls = message.get("tool_calls", [])
                if not calls:
                    stopped = "llm_stop"
                    (trial / "conclusion.md").write_text(message.get("content") or "", encoding="utf-8")
                    break
                call = calls[0]
                if call["function"]["name"] != "submit_algorithm":
                    raise ValueError("未知研究工具")
                proposal = json.loads(call["function"]["arguments"])
                hypothesis, analysis, code = (proposal[k] for k in ("hypothesis", "feedback_analysis", "code"))
                messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": [call]})
            if not cached_baseline:
                scope.spend("experiment")
                scope.spend("tool")
        except BudgetExceeded as error:
            stopped = "budget_exhausted"
            write_json(trial / "budget_stop.json", error.as_dict())
            break
        algorithm = trial / "algorithm.py"
        algorithm.write_text(code, encoding="utf-8")
        (trial / "changes.diff").write_text("".join(difflib.unified_diff(
            previous_code.splitlines(True), code.splitlines(True), fromfile="previous.py", tofile="algorithm.py")), encoding="utf-8")
        attempt = {"iteration": iteration, "hypothesis": hypothesis, "feedback_analysis": analysis,
                   "algorithm": str(algorithm), "code_sha256": hashlib.sha256(code.encode()).hexdigest()}
        try:
            if cached_baseline:
                feedback = baseline_feedback
                attempt["source"] = "shared_fixed_baseline_cache"
            else:
                ranks = execute(algorithm, public, trial, timeout=max(.1, min(60, scope.remaining("seconds"))))
                feedback = evaluate(case, ranks)
            value = feedback["metrics"][case["objective"]]
            keep = best is None or value > best["value"]
            attempt.update(status="keep" if keep else "discard", value=value, feedback=feedback,
                           delta=0 if best is None else value-best["value"])
            if keep:
                best = {"value": value, "algorithm": str(algorithm), "iteration": iteration,
                        "code": code, "feedback": feedback}
        except (subprocess.SubprocessError, OSError, ValueError, KeyError, TypeError) as error:
            attempt.update(status="crash", error=f"{type(error).__name__}: {error}",
                           log=(trial / "run.log").read_text(encoding="utf-8")[-4000:])
        previous_code = code
        write_json(trial / "feedback.json", attempt)
        record = store.append(build_record(task_id=run_id, business=business,
            status="success" if attempt["status"] != "crash" else "failed",
            config={"protocol_sha256": protocol_hash, "code_sha256": attempt["code_sha256"], "iteration": iteration},
            data={"dataset": case["dataset"], "split": "validation"},
            model={"models": case.get("models", []), "weights_source": "existing_only"},
            result={**attempt, "metrics": attempt.get("feedback", {}).get("metrics", {}),
                    "summary": f"{hypothesis}: {attempt['status']}"}, budget=scope.snapshot(),
            artifacts={"algorithm": str(algorithm), "feedback": str(trial / "feedback.json"),
                       "case": str(work_dir / "case.json")},
            evidence={"validated": attempt["status"] != "crash", "decision_driver": "llm" if iteration else "fixed_baseline",
                      "simulation": case.get("simulation", {}), "notes": "code-editing validation research; no training"}))
        attempt["record_id"] = record["record_id"]
        attempts.append(attempt)
        # Per-user result deltas and cold/warm metrics are actual observations, not guessed diagnoses.
        feedback_view = {**attempt, "feedback": {k: v for k, v in attempt.get("feedback", {}).items() if k != "per_group"}}
        write_json(work_dir / "attempts.json", attempts)
        content = json.dumps({"last_experiment": feedback_view, "best_so_far": best,
                              "budget": scope.snapshot()}, ensure_ascii=False)
        if iteration:
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        else:
            messages.append({"role": "user", "content": content})
        write_json(work_dir / "messages.json", messages)
    if best:
        (work_dir / "best_algorithm.py").write_text(best["code"], encoding="utf-8")
    successful = sum(a["iteration"] > 0 and a["status"] != "crash" for a in attempts)
    status = "success" if successful == rounds else "partial" if successful else "failed"
    result = {"ok": status == "success", "status": status,
              "business": business, "research_mode": "code_editing", "stop_reason": stopped,
              "completed_algorithm_experiments": successful, "attempts": attempts,
              "best": best, "metrics": best["feedback"]["metrics"] if best else {},
              "record_id": attempts[-1]["record_id"] if attempts else None,
              "history_records_read": [h["record_id"] for h in history],
              "artifacts": {"summary": str(work_dir / "summary.json"), "best_algorithm": str(work_dir / "best_algorithm.py") if best else None},
              "budget": scope.snapshot(), "elapsed_s": round(time.perf_counter()-started, 3)}
    write_json(work_dir / "summary.json", result)
    return result


if __name__ == "__main__":
    import importlib.util
    if len(sys.argv) != 5 or sys.argv[1] != "--worker":
        raise SystemExit("Use the Research Agent or business agent run_research entry.")
    spec = importlib.util.spec_from_file_location("research_candidate", sys.argv[2])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    public = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
    ranks = module.rank(public["groups"], public["context"])
    write_json(sys.argv[4], ranks)
