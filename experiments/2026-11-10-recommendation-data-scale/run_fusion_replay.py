from pathlib import Path
from time import perf_counter
import json
import numpy as np
from recsys.pipeline.recall import Candidate
from recsys.pipeline.fusion import fuse_channels
from recsys.pipeline.config import FusionConfig
from recsys.evaluation.metrics import recall_at_k,ndcg_at_k

root=Path("experiments/2026-11-10-recommendation-data-scale")
p=Path("experiments/2026-11-10-formal-e2e/recommendation")
splits=Path("data/kuasar_small/processed/split")
started=perf_counter()
channels={}
for name in ("two_tower","mind"):
    blob=json.loads((p/f"recall_{name}.json").read_text(encoding="utf-8"))
    channels[name]={int(u):[Candidate(int(x["item_idx"]),x["score"],int(x["rank"])) for x in rows] for u,rows in blob.items()}
truth={}
for u,i in zip(np.load(splits/"test_users.npy"),np.load(splits/"test_items.npy")):
    truth.setdefault(int(u),set()).add(int(i))
def metrics(recs):
    return {"Recall@20":recall_at_k(recs,truth,20),"Recall@100":recall_at_k(recs,truth,100),"NDCG@20":ndcg_at_k(recs,truth,20),
         "hit_targets_20":sum(len(set(recs[u][:20])&g) for u,g in truth.items()),
         "hit_targets_100":sum(len(set(recs[u][:100])&g) for u,g in truth.items())}
def items(d):
    return {u:[x.item for x in rows] for u,rows in d.items()}
comparison={name:metrics(items(ch)) for name,ch in channels.items()}
for mode in ("round_robin","rrf","quota"):
    cfg=FusionConfig(mode=mode,limit=100,quota={"two_tower":30,"mind":70} if mode=="quota" else {})
    recs=items(fuse_channels(channels,cfg))
    assert len(recs)==2000 and all(len(row)==100 for row in recs.values())
    comparison[mode]=metrics(recs)
    if mode=="round_robin":
        original=json.loads((p/"fusion.json").read_text(encoding="utf-8"))
        assert all(recs[int(u)]==[x["item_idx"] for x in rows] for u,rows in original.items())
out={"source":str(p),"protocol":"Reuse cached Two-Tower and MIND rankings; same test set and 100 candidates; no reranking or test tuning. RRF uses repo default k=60. Quota 30/70 previously selected on validation.",
     "methods":comparison,"elapsed_s":round(perf_counter()-started,3)}
(root/"fusion_comparison.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(out,ensure_ascii=False,indent=2))
