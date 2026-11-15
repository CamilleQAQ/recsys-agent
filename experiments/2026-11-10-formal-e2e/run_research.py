from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import json

from recsys.platform import research_agent as research
from recsys.platform.shared_experiments import SharedExperimentStore, reuse_reasons
from recsys.pipeline import ranking

root=Path("experiments/2026-11-10-formal-e2e")
out=root/"research"
out.mkdir(parents=True,exist_ok=False)
store=SharedExperimentStore(out/"shared")
config=research.ResearchConfig.load()
config.experiment_root=out
config.store_dir=store.root
config.max_steps=12
config.budget.max_llm_requests=12
config.budget.max_tool_calls=28
config.budget.max_experiments=4
config.budget.max_history_lookups=2
config.budget.max_artifact_reads=2
config.budget.max_seconds=840
config.subagent_intent["recommendation"].update({
    "users":64,"split":"validation","recall_models":["two_tower","mind"],
    "ranker":"lightgbm_ranker","k_recall":100,"k_final":20,
    "artifact_policy":"existing_only",
})
config.subagent_intent["search"].update({"mode":"rerank","session_id":"11208","require_existing_index":True})
config.subagent_intent["ads"].update({"model":"din","split":"test","requests":4,"limit":100,"simulate_auction":True})
post=research.post_chat
def live(cfg,messages,*,tools,timeout):
    result=post(cfg,messages,tools=tools,timeout=timeout)
    msg=result["choices"][0]["message"]
    event={"id":result.get("id"),"model":result.get("model"),"finish_reason":result["choices"][0].get("finish_reason"),
           "tool_calls":msg.get("tool_calls") or [],"content":msg.get("content")}
    with (out/"native_llm.jsonl").open("a",encoding="utf-8") as file:
        file.write(json.dumps(event,ensure_ascii=False)+"\n")
    print("native LLM",event["model"],[(c["function"]["name"],c["function"]["arguments"]) for c in event["tool_calls"]],flush=True)
    return result

task = (
    "研究问题：对于 KuaiSAR 短视频，怎样在不重训的情况下增加最终推荐命中？请用真实工具执行一项跨推荐、"
    "搜索、广告的研究。先从已有 30/70 双塔/MIND 推荐历史或 RAG 找到一条有来源的先验，指出先验适用的范围。"
    "已有 30/70 是融合配额历史，只能作为先验，不可当成本次 Top-20 结果。用 64 个 validation 用户做两次"
    "推荐实验，固定 LightGBM 精排、k_final=20，召回模型仅从 two_tower、mind、popularity 中选择，"
    "候选深度可在 20 到 200 之间选。由你选择首轮配置，看到实际指标后再决定第二轮的一个策略变化"
    "（候选深度或者召回模型，具体由你决定），最终解释第二次参数如何受第一轮反馈影响。"
    "同时用搜索子智能体验证一次会话 11208 的真实混合检索和精排，用广告子智能体验证一次 DIN "
    "的 CTR 推理、排序和投放模拟，并区分模拟金额和真实点击。"
    "总预算最多 4 次新实验，其中推荐两次、搜索一次、广告一次；使用现有权重。"
    "按需用 read_experiment_result 取回已入库记录，核对关键指标和记录 ID。"
    "完成后用中文给出你的假设、逐轮依据、指标、经验和下一步，停止时不要再调用业务。"
)
with patch.object(research,"post_chat",side_effect=live), patch.object(
    ranking,"train_lightgbm",side_effect=AssertionError("existing weights only")
):
    summary=research.ResearchAgent(config,planner="llm",store=store).run(
        task,intent={"businesses":["recommendation","search","ads"],"history_first":False}
    )
result={k:summary.get(k) for k in ("task_id","status","planner","stop_reason","stop_detail","llm_requests",
        "businesses","record_ids","task_record_id","budget","new_experiments","elapsed_s","artifacts","final_answer")}
result["records"]=[
    {"record_id":rec["record_id"],"business":rec["business"],"status":rec["status"],
     "config":rec.get("config"),"reuse_rejections":reuse_reasons(rec),"metrics":rec.get("result",{}).get("metrics")}
    for rec in store.latest() if rec["business"] in research.BUSINESSES
]
(out/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps({k:v for k,v in result.items() if k not in {"records","final_answer"}},ensure_ascii=False),flush=True)
