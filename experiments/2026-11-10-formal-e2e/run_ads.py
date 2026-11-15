from collections import Counter
from pathlib import Path
from time import perf_counter
import gzip
import hashlib
import json

import numpy as np

from recsys.ads.auction import new_budget_state,run_request_auction,summarize_auction
from recsys.ads.config import load_ads_config
from recsys.ads.features import build_click_metrics
from recsys.ads.ranking import rank_candidates,summarize_rankings
from recsys.ads.runtime import AdsRuntime

root=Path("experiments/2026-11-10-formal-e2e")
out=root/"ads"
out.mkdir(parents=True,exist_ok=False)
started=perf_counter()
config=load_ads_config()
runtime=AdsRuntime()
model="din"
adapter=runtime.model(model)
weight=runtime.weight_info(model)
with open(weight["path"],"rb") as file:
    weight["sha256"]=hashlib.file_digest(file,"sha256").hexdigest()
by_split={}
queue=[]
for split_index,split in enumerate(("train","validation","test")):
    view=runtime.view(split)
    batch=runtime.batch(split)
    scores=np.asarray(adapter.score(batch.as_dict())["ctr"],dtype=np.float64)
    assert scores.shape==view.labels.shape and np.isfinite(scores).all() and np.all((scores>=0)&(scores<=1))
    starts=view.starts()
    advertisers=runtime.advertiser_ids(view.items)
    for i,(a,b) in enumerate(zip(starts[:-1],starts[1:])):
        queue.append((int(view.timestamps[i]),split_index,i,int(view.users[a]),view.items[a:b],scores[a:b],view.labels[a:b],advertisers[a:b]))
    by_split[split]={"rows":len(view),"requests":len(view.groups),"multi_ad_requests":int(np.count_nonzero(view.groups>1)),
                    "clicks":int(np.sum(view.labels)),
                    "metrics":build_click_metrics(view.labels,scores)}
queue.sort(key=lambda row:(row[0],row[1],row[2]))
assert len(queue)==5276
total=500.0
budgets={"total_only":new_budget_state(config.budget,config.auction,total=total,per_advertiser=total),
         "with_advertiser_cap":new_budget_state(config.budget,config.auction,total=total,per_advertiser=config.budget.per_advertiser)}
outcomes={name:[] for name in budgets}
first_unfilled={name:None for name in budgets}
rankings=[]
with gzip.open(out/"request_replay.jsonl.gz","wt",encoding="utf-8") as file:
    for j,(ts,split,i,user,items,scores,labels,advertisers) in enumerate(queue):
        ranking=rank_candidates(items,scores,labels=labels,advertiser_ids=advertisers,top_k=len(items))
        rankings.append({"candidate_count":len(items),"ranking":ranking})
        item={"timestamp":ts,"split":("train","validation","test")[split],"request_index":i,"candidate_count":len(items),
              "ranking":ranking,"strategies":{}}
        for name,budget in budgets.items():
            result=run_request_auction(request_index=i,user=user,timestamp=ts,items=items,pctr=scores,labels=labels,
                                       advertiser_ids=advertisers,auction=config.auction,budget=budget,top_k=len(items))
            assert len(result["winners"])<=config.auction.slots
            assert budget.spent<=budget.total+1e-9 and budget.remaining>=0
            outcomes[name].append(result)
            if result.get("no_winner_reason") and first_unfilled[name] is None:
                first_unfilled[name]={"position":j+1,"timestamp":ts,"reason":result["no_winner_reason"]}
            item["strategies"][name]={k:result.get(k) for k in ("winner","charge","no_winner_reason")}
        file.write(json.dumps(item,separators=(",",":"))+"\n")
reports={}
for name,budget in budgets.items():
    records=outcomes[name]
    stats=summarize_auction(records,budget)
    reasons=Counter(stats.pop("no_winner_reasons"))
    total_spend=budget.spent
    assert abs(stats["spend"]-total_spend)<=0.00005*len(records)+0.0001
    assert abs(sum(budget.advertiser_spent.values())-total_spend)<1e-7
    assert all(value<=budget.per_advertiser+1e-9 for value in budget.advertiser_spent.values())
    spends=sorted(budget.advertiser_spent.values(),reverse=True)
    multi=[r for r in records if r["candidate_count"]>1]
    reports[name]={**stats,"no_winner_reasons":dict(reasons),"fill_rate":1-stats["no_winner_requests"]/len(records),
                   "first_unfilled_request":first_unfilled[name],
                   "first_total_budget_exhausted":next(({"position":j+1,"timestamp":r["timestamp"]}
                     for j,r in enumerate(records) if r.get("no_winner_reason")=="budget_exhausted_total"),None),
                   "advertisers_charged":len(spends),"top10_advertiser_spend_share":sum(spends[:10])/total_spend if total_spend else None,
                   "multi_ad_requests":len(multi),"multi_ad_requests_filled":sum(bool(r["winners"]) for r in multi)}
payload={"dataset":runtime.dataset,"model":model,"weight":weight,"splits":by_split,
         "requests":len(queue),"ad_exposure_rows":sum(n["rows"] for n in by_split.values()),
         "multi_ad_requests":sum(n["multi_ad_requests"] for n in by_split.values()),
         "ranking":summarize_rankings(rankings),"auction_config":config.auction.as_dict(),"replay":reports,
         "money":"simulated","click_labels":"KuaiSAR logged clicks","conversion_labels":"unavailable",
         "protocol":"One pCTR inference pass per split, pCTR ranking and globally chronological replay across the three splits. Logged clicks reused only for offline billing. Existing DIN interface-test checkpoint is not a trained CTR model.",
         "wall_s":round(perf_counter()-started,3)}
(out/"result.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps(payload,ensure_ascii=False),flush=True)
