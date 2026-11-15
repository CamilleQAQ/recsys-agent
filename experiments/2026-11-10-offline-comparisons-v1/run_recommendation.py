from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json
import time

import torch
from recsys.experiments import rec_agent as agent, rec_tools as rt
from recsys.data.loader import prepare_dataset
from recsys.pipeline import ranking
from recsys.pipeline.chain import RecommendationPipeline, metrics_for
from recsys.pipeline.config import load_pipeline_config, RecallChannelConfig, FusionConfig

root = Path("experiments/2026-11-10-offline-comparisons-v1/recommendation")
root.mkdir(parents=True, exist_ok=False)
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))
def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")

agent.load_dotenv()
llm = agent.llm_config()
if llm is None:
    raise SystemExit("Missing real LLM configuration")
ctx = rt.ToolContext(data_dir=Path("data"), artifacts_dir=Path("artifacts"))
recall_paths = {model: rt.resolve_checkpoint(ctx, model) for model in rt.RECALL_MODELS}
recall_paths = {model: path for model, path in recall_paths.items() if path}
recall_paths["popularity"] = None
rank_paths = {model: rt.resolve_ranker_checkpoint(ctx, model) for model in ("lightgbm_ranker", "dcnv2")}
rank_paths = {model: path for model, path in rank_paths.items() if path}
data = prepare_dataset("kuasar_small", Path("data"))
val = agent.AgentSettings(users=256, seed=42, max_tool_calls=7, max_train_calls=0, max_seconds=240)
profiles = {}
for model, path in recall_paths.items():
    if path is None:
        profiles[model] = {"checkpoint": None, "note": "Train-click frequency baseline, no weight file."}
        continue
    weight = torch.load(path, map_location="cpu", weights_only=False)
    cfg = weight.get("train_config") or {}
    prov = weight.get("provenance") or {}
    counts = (prov.get("data") or {}).get("counts") or {}
    profiles[model] = {
        "checkpoint": str(path), "training_epochs": cfg.get("epochs", cfg.get("max_epochs")),
        "best_epoch": weight.get("best_epoch"), "max_batches_per_epoch": cfg.get("max_batches_per_epoch"),
        "historical_validation_users": cfg.get("validation_users") or "full/unspecified",
        "historical_validation_clicks": counts.get("validation"),
        "historical_validation_Recall@20": (weight.get("best_validation_metrics") or {}).get("Recall@20"),
    }
    del weight
candidate_context = {
    "recall": profiles,
    "rankers": {m:str(p) for m,p in rank_paths.items()},
    "notes": [
        "Historical validation metrics use different sample sizes and are clues only.",
        "Old DIN, DIEN, Ultra-HSTU defaults with history leakage are out of scope.",
    ],
}
write(root / "candidate_profiles.json", candidate_context)
base = rt.build_registry()
registry = rt.ToolRegistry()
registry.register(rt.ToolSpec(
    "list_artifacts", "Available methods and actual training/validation metadata.",
    rt._object({}), lambda ctx, args: {"ok": True, **candidate_context},
))
source = base.get("run_pipeline")
schema = {
    k: source.parameters["properties"][k] for k in ("models", "ranker", "fusion", "quota")
}
schema["models"] = {"type":"array","items":{"type":"string","enum":list(recall_paths)},"minItems":1,"maxItems":3}
schema["ranker"] = {"type":"string","enum":list(rank_paths)}
evaluations = {}
attempts = 0
selected = None
def validate(ctx, args):
    global attempts
    if attempts == 2:
        return {"ok":False,"error":"Two validation evaluations already attempted."}
    attempts += 1
    candidate_id = f"candidate-{attempts}"
    print("validation", candidate_id, args, flush=True)
    result = source.execute(ctx, {
        **args, "artifact_policy":"existing_only", "split":"validation", "k_recall":100, "k_final":20,
        "output_name":candidate_id,
    })
    if result.get("ok"):
        hits = result["diagnostics"]
        p = ctx.state["last_pipeline"]
        val_users = set(p.users)
        targets = [r for r in data.validation_interactions if r.user_idx in val_users]
        result["validation_fusion_Recall@100"] = metrics_for({
            u: [item.item for item in rows] for u,rows in p.fused.items()
        }, targets, (100,))["Recall@100"]
        evaluations[candidate_id] = {"candidate_id":candidate_id, "args":args,
            "validation_fusion_Recall@100":result["validation_fusion_Recall@100"],
            "validation_NDCG@20":result["metrics"]["rerank"]["NDCG@20"],
            "validation_recall_union_hits":hits["union_hits_at_k_recall"]}
    return {"candidate_id":candidate_id, **result}
def lock(ctx,args):
    global selected
    selected = evaluations.get(args["candidate_id"])
    if selected is None:
        return {"ok":False,"error":"Candidate did not complete validation."}
    selected = {**selected,"reason":args["reason"]}
    write(root / "locked_plan.json", selected)
    return {"ok":True,"locked_plan":selected}
registry.register(replace(source, execute=validate,
    parameters=rt._object(schema,["models","ranker","fusion"])))
registry.register(rt.ToolSpec("lock_plan", "Freeze one successfully evaluated plan before any test data.",
    rt._object({"candidate_id":{"type":"string","enum":["candidate-1","candidate-2"]},
                "reason":{"type":"string"}},["candidate_id","reason"]),lock))
def live(config, messages, *, tools, timeout):
    response = agent.post_chat(config, messages, tools=tools, timeout=timeout)
    message = response["choices"][0]["message"]
    trace = {"id":response.get("id"),"model":response.get("model"),"message":message}
    with (root / "llm_calls.jsonl").open("a",encoding="utf-8") as file:
        file.write(json.dumps(trace,ensure_ascii=False)+"\n")
    print("LLM",trace["model"],[c["function"]["name"] for c in message.get("tool_calls") or []],flush=True)
    return response
task = (
    "Choose one defensible recommendation strategy. Call list_artifacts to review actual checkpoint "
    "training depth and available validation evidence. Compare one or two configurations on the same 256 "
    "validation users via run_pipeline. The fixed reference is two_tower + mind / round_robin / lightgbm_ranker. "
    "Consider an alternative (e.g. fusion or reranker, or different recall if evidence supports it). "
    "Choose freely, including the fixed reference if warranted by validation. Max two candidate evaluations, "
    "use existing weights only. Prefer final NDCG@20, breaking near ties with fusion Recall@100 and raw counts. "
    "If quota fusion, provide quotas that total 100. Then call lock_plan with the chosen candidate ID "
    "and a brief Chinese rationale, and stop. Do not request, infer or claim test results."
)
with patch.object(agent,"build_registry",return_value=registry), patch.object(
    ranking,"train_lightgbm",side_effect=AssertionError("Training forbidden")
):
    val_summary = agent.run_agent(val,task=task,plan=[],experiment_dir=root/"validation",chat_fn=live)
print("validation finished",val_summary["status"],val_summary["errors"],selected,flush=True)
if selected is None or not val_summary["llm_used"]:
    raise SystemExit("No real Agent selection; no test run made")

p1 = Path("experiments/2026-11-10-kuasar-small-test-512/run-20261110-134155-d7bc6c")
p2 = Path("experiments/2026-11-10-kuasar-small-test-all/run-20261110-135118-7a06aa")
cfg = load_pipeline_config(p1/"config.toml")
cfg2 = load_pipeline_config(p2/"config.toml")
assert cfg.recall == cfg2.recall and cfg.fusion == cfg2.fusion and cfg.ranking == cfg2.ranking and cfg.rerank == cfg2.rerank
baseline_users = [set(read(path/"run_meta.json")["users"]) for path in (p1,p2)]
assert not baseline_users[0] & baseline_users[1]
users = sorted(set.union(*baseline_users))
indices = [int(data.mappings.user_to_index[u]) for u in users]
assert len(indices) == 2000
assert set(indices) == {r.user_idx for r in data.test_interactions}
provenance = [read(p/"run_meta.json")["provenance"] for p in (p1,p2)]
assert provenance[0]["id_mapping"]["sha256"] == provenance[1]["id_mapping"]["sha256"]
assert provenance[0]["data"]["split_sha256"] == provenance[1]["data"]["split_sha256"]
combined = read(p2/"combined_metrics.json")
def stages(fused,final,diag,seconds):
    target_users = {r.user_idx for r in data.test_interactions}
    assert set(fused) == set(final) == target_users
    return {"fused_recall_at_100":metrics_for(fused,data.test_interactions,(100,))["Recall@100"],
            **metrics_for(final,data.test_interactions,(20,)),
            "fused_hit_targets":sum(len(set(items[:100]) & target_by_user.get(u,set())) for u,items in fused.items()),
            "final_hit_targets":sum(len(set(items[:20]) & target_by_user.get(u,set())) for u,items in final.items()),
            "union_hit_targets":diag["union_hits"],"pipeline_s":seconds}
target_by_user = {}
for row in data.test_interactions:
    target_by_user.setdefault(row.user_idx,set()).add(row.item_idx)
def baseline_items(file):
    out = {}
    for path in (p1,p2):
        payload = read(path/file)
        for u,row in payload.items():
            items = row["items"] if isinstance(row,dict) else row
            out[int(u)] = [item["item_idx"] for item in items]
    return out
baseline = stages(baseline_items("fusion.json"), baseline_items("final.json"),
    {"union_hits":combined["combined"]["union_at100"]}, combined["combined_pipeline_s"])
assert baseline["fused_hit_targets"] == combined["combined"]["fusion_at100"]
assert baseline["final_hit_targets"] == combined["combined"]["final_at20"]
args = selected["args"]
config = replace(cfg, run=replace(cfg.run,users=2000,user_ids=(),output_name="agent-selected-test"),
    recall=tuple(RecallChannelConfig(name=m,model=m,checkpoint=recall_paths[m]) for m in args["models"]),
    fusion=FusionConfig(mode=args["fusion"],limit=100,quota=args.get("quota") or {}),
    ranking=replace(cfg.ranking, model=args["ranker"],checkpoint=rank_paths[args["ranker"]],train_if_missing=False))
if (cfg.recall,cfg.fusion,cfg.ranking,cfg.rerank) == (config.recall,config.fusion,config.ranking,config.rerank):
    evaluated = baseline
    test_mode = "identical configuration; reused fixed results"
else:
    print("test locked plan",args,flush=True)
    with patch.object(ranking,"train_lightgbm",side_effect=AssertionError("Training forbidden")):
        run = RecommendationPipeline(config,data=data).run(split="test",users=indices,output_dir=root/"test")
    test_mode = "single existing-weight test run"
    evaluated = stages({u:[c.item for c in rows] for u,rows in run.fused.items()},
        run.final_items, {"union_hits":run.diagnostics["union_hits_at_k_recall"]},run.timings["total_s"])
result = {"dataset":"kuasar_small","test_users":len(users),"test_targets":sum(map(len,target_by_user.values())),
          "validation_users":256,"validation_candidates":list(evaluations.values()),"selected":selected,
          "test_mode":test_mode,"fixed_baseline":baseline,"agent_selected":evaluated,
          "baseline_sources":[str(p1),str(p2)],"training_calls":val_summary["budget"]["train_calls"]}
write(root/"comparison.json",result)
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
