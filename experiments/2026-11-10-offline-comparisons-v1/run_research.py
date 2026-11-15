from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch
import json

from recsys.pipeline import ranking
from recsys.platform.research_agent import ResearchAgent, ResearchConfig
from recsys.platform.shared_experiments import SharedExperimentStore, reuse_reasons

root = Path("experiments/2026-11-10-offline-comparisons-v1/research")
root.mkdir(parents=True, exist_ok=False)
def write(name, value):
    (root/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")

base = ResearchConfig.load()
workloads = [
    ("recommendation", {"users":4, "recall_models":["popularity"], "ranker":"lightgbm_ranker","split":"validation"}),
    ("recommendation", {"users":8, "recall_models":["popularity"], "ranker":"lightgbm_ranker","split":"validation"}),
    ("recommendation", {"users":4, "recall_models":["two_tower"], "ranker":"lightgbm_ranker","split":"validation"}),
    ("search", {"mode":"bm25","session_id":"11208"}),
    ("search", {"mode":"bm25","session_id":"8490"}),
    ("search", {"mode":"hybrid","session_id":"4402"}),
    ("ads", {"model":"din","split":"test","requests":2}),
    ("ads", {"model":"dcnv2","split":"test","requests":3}),
]
def intent_for(business, options):
    lookup = {"business":business,"dataset":"kuasar_small"}
    if business == "search":
        lookup.update({"split":"not_applicable","session_id":options["session_id"],"mode":options["mode"]})
    else:
        lookup.update({"split":options["split"],"model":options.get("model",options.get("ranker"))})
    return lookup

def run_case(group, store, business, options, text, *, history, budget, fixture_task=None):
    cfg = deepcopy(base)
    cfg.experiment_root = root/group
    cfg.store_dir = store.root
    cfg.subagent_intent[business].update(options)
    cfg.subagent_intent["recommendation"]["artifact_policy"] = "existing_only"
    cfg.budget.max_experiments = budget
    cfg.budget.max_llm_requests = 0
    cfg.budget.max_tool_calls = 12
    intent = intent_for(business,options)
    if fixture_task:
        intent["task_id_filter"] = fixture_task
    started = perf_counter()
    summary = ResearchAgent(cfg, planner="scripted", store=store).run(
        text,intent={**intent,"history_first":history},
    )
    return summary, perf_counter()-started

def business_records(store,summary,business):
    return [store.get(item["record_id"]) for item in summary["tool_results"]
        if item["business"]==business and item["tool"].startswith("run_") and item["record_id"]]

def signature(business, metrics):
    if business == "recommendation":
        return metrics.get("run_pipeline")
    if business == "search":
        return {k:metrics.get(k) for k in ("status","return_counts","tool_sequence")}
    ad = metrics.get("simulate_auction") or {}
    return {"ctr":(metrics.get("predict_ad_ctr") or {}).get("metrics"),
            "auction":[v.get("summary") for v in ad.get("summaries",[])]}

seed = SharedExperimentStore(root/"fixture"/"shared")
stores = {name: SharedExperimentStore(root/name/"shared") for name in ("memory_on","memory_off")}
fixture_records = []
cases = []
with patch.object(ranking,"train_lightgbm",side_effect=AssertionError("Training forbidden")):
    for idx,(business,opts) in enumerate(workloads):
        s,elapsed = run_case("fixture",seed,business,opts,f"准备已验证的 {business} 历史记录 {idx+1}",
                             history=False,budget=1)
        records = business_records(seed,s,business)
        if s["status"]!="verified" or len(records)!=1 or reuse_reasons(records[0]):
            raise RuntimeError(f"fixture {idx+1} failed: {s['status']} {s['stop_detail']} {s['tool_results']}")
        record = records[0]
        fixture_records.append(record)
        print("fixture",idx+1,business,round(elapsed,2),flush=True)
        exact = dict(opts)
        near = dict(opts)
        if business == "search":
            near["mode"] = "hybrid" if opts["mode"]=="bm25" else "bm25"
        else:
            near["split"] = "test" if opts["split"]=="validation" else "validation"
        for kind, options, budget in (("exact",exact,1),("mismatch",near,1),("no_budget",near,0)):
            selector = intent_for(business,options)
            cases.append({"id":f"{idx+1:02d}-{kind}","kind":kind,"business":business,
                          "options":options,"budget":budget,"fixture_task":s["task_id"],
                          "fixture_record":record["record_id"],
                          "task":f"获取 {business} 的实验结果。口径 {json.dumps(selector,ensure_ascii=False)}，"
                                 f"对照历史任务 {s['task_id']}，需要时按当前预算处理。"})
    for store in stores.values():
        store.append_many(fixture_records)
    write("workload.json",{"fixture_experiments":len(fixture_records),"cases":cases,
        "method":"Existing Research Agent scripted planner in both arms; real business tools. Both stores have identical verified initial records. The memory-off arm skips the history node. Pairs alternate order. Timed from ResearchAgent.run through persisted outputs; fixture setup excluded."})
    rows = []
    for idx,case in enumerate(cases):
        paired = {}
        order = ("memory_on","memory_off") if idx%2==0 else ("memory_off","memory_on")
        for group in order:
            enabled = group=="memory_on"
            store = stores[group]
            s,elapsed = run_case(group,store,case["business"],case["options"],case["task"],
                                 history=enabled,budget=case["budget"],fixture_task=case["fixture_task"])
            history = s.get("history") or {}
            records = business_records(store,s,case["business"])
            reused = s["reuse_mode"]=="history"
            payload = history.get("metrics") if reused else ((records[0]["result"]["metrics"]) if len(records)==1 else {})
            count = s["new_experiments"]
            if case["kind"]=="no_budget":
                correct = s["stop_reason"]=="budget_exhausted" and count==0 and s["status"]!="verified" and not reused
            elif case["kind"]=="exact" and enabled:
                correct = s["status"]=="verified" and reused and count==0 and history["record_id"]==case["fixture_record"]
            else:
                correct = s["status"]=="verified" and not reused and count==1 and len(records)==1 and records[0]["status"]=="success"
                if enabled:
                    correct = correct and history.get("status")=="miss"
            counters = s["budget"].get("counters") or {}
            paired[group] = {"status":s["status"],"stop":s["stop_reason"],"experiments":count,
                "wall_s":round(elapsed,5),"tool_calls":counters.get("tool"),"history_lookups":counters.get("history"),
                "history_status":history.get("status"),"history_reason":history.get("reason"),
                "reused":reused,"history_record_id":history.get("record_id"),"correct":correct,
                "result_signature":signature(case["business"],payload or {}),
                "summary":s["artifacts"]["summary"]}
        a,b=paired["memory_on"],paired["memory_off"]
        row = {"case":case["id"],"kind":case["kind"],"business":case["business"],"order":order,**paired,
               "observed_results_equal": a["result_signature"]==b["result_signature"] if case["budget"] else None}
        rows.append(row)
        with (root/"pairs.jsonl").open("a",encoding="utf-8") as file:
            file.write(json.dumps(row,ensure_ascii=False)+"\n")
        print("pair",case["id"],[(k,paired[k]["experiments"],paired[k]["wall_s"],paired[k]["correct"]) for k in order],flush=True)

def aggregate(pairs):
    result = {}
    for group in stores:
        cells = [row[group] for row in pairs]
        result[group] = {"tasks":len(cells),"correct":sum(cell["correct"] for cell in cells),
                         "verified":sum(cell["status"]=="verified" for cell in cells),
                         "experiments":sum(cell["experiments"] for cell in cells),
                         "tool_calls":sum(cell["tool_calls"] or 0 for cell in cells),
                         "history_lookups":sum(cell["history_lookups"] or 0 for cell in cells),
                         "reuses":sum(cell["reused"] for cell in cells),
                         "wall_s":round(sum(cell["wall_s"] for cell in cells),3),
                         "median_task_s":round(median(cell["wall_s"] for cell in cells),3)}
    result["matching_result_signatures"] = sum(row["observed_results_equal"] is True for row in pairs)
    result["comparable_pairs"] = sum(row["observed_results_equal"] is not None for row in pairs)
    return result
result = {"planner":"scripted_stand_in; both arms executed real business tools",
    "measurement":"Sum of measured task wall times including output persistence; excludes fixture preparation.",
    "fixture_experiments":len(fixture_records),"training_calls":0,
    "overall":aggregate(rows),
    "by_scenario":{kind:aggregate([r for r in rows if r["kind"]==kind])
                   for kind in ("exact","mismatch","no_budget")}}
write("comparison.json",result)
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
