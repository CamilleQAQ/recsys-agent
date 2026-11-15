from pathlib import Path
from time import perf_counter
import gzip,json,hashlib,gc
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from recsys.data.schema import ArrayInteractions,IdMappings,PreparedData,SortedIndexLookup
from recsys.evaluation.metrics import compute_ranking_metrics
from recsys.models.two_tower import TwoTowerModel
from recsys.training.trainer import train_one_epoch
from recsys.utils.seed import set_random_seed

root=Path("experiments/2026-11-13-kuairec-small")
root.mkdir(parents=True,exist_ok=True)
raw=Path("data/kuairec_small/raw/small_matrix.csv")
assert raw.exists()
torch.set_num_threads(4)
t0=perf_counter()
df=pd.read_csv(raw,usecols=["user_id","video_id","timestamp","watch_ratio"],
    dtype={"user_id":"int32","video_id":"int32","timestamp":"float64","watch_ratio":"float64"})
raw_rows=len(df)
valid_time=np.isfinite(df.timestamp.to_numpy())
assert np.isfinite(df.watch_ratio).all()
missing_time_rows=int((~valid_time).sum())
user_ids=np.sort(df.user_id.unique())
item_ids=np.sort(df.video_id.unique())
nu,ni=len(user_ids),len(item_ids)
assert (nu,ni)==(1411,3327),(nu,ni)
us=np.searchsorted(user_ids,df.user_id.to_numpy())
it=np.searchsorted(item_ids,df.video_id.to_numpy())
ts=df.timestamp.to_numpy()
pos=df.watch_ratio.to_numpy()>=1.0
keys=us*ni+it
unknown_time_pairs=np.unique(keys[~valid_time])
# First exposure labels, matching the project's rule. Exclude contradictory labels at equal earliest times.
first=np.full(nu*ni,np.inf)
np.minimum.at(first,keys[valid_time],ts[valid_time])
at_first=ts==first[keys]
has_pos=np.zeros(nu*ni,dtype=bool)
has_neg=np.zeros(nu*ni,dtype=bool)
has_pos[keys[at_first & pos]]=True
has_neg[keys[at_first & ~pos]]=True
ambiguous=has_pos & has_neg
positive=(has_pos & ~ambiguous).reshape(nu,ni)
first[unknown_time_pairs]=np.inf
positive.reshape(-1)[unknown_time_pairs]=False
first=first.reshape(nu,ni)
del df,us,it,ts,pos,keys,at_first,has_pos,has_neg
gc.collect()
val_times=np.empty(nu)
test_times=np.empty(nu)
for u in range(nu):
    times=np.unique(first[u][np.isfinite(first[u])])
    val_times[u]=times[int(.8*len(times))]
    test_times[u]=times[int(.9*len(times))]
assert np.all(val_times<test_times)
seen_val=first<val_times[:,None]
seen_test=first<test_times[:,None]
masks={"train":positive & seen_val,"validation":positive & ~seen_val & seen_test,
       "test":positive & ~seen_test}
arrays={}
for split,mask in masks.items():
    u,i=np.nonzero(mask)
    t=first[u,i]
    order=np.lexsort((i,t,u))
    arrays[split]=ArrayInteractions(u[order].astype(np.int32),i[order].astype(np.int32),t[order])
    np.savez_compressed(root/f"{split}.npz",users=u[order],items=i[order],timestamps=t[order])
mapping=IdMappings(SortedIndexLookup(user_ids),user_ids,SortedIndexLookup(item_ids),item_ids)
data=PreparedData(arrays["train"],arrays["validation"],arrays["test"],mapping,seen_val,seen_test,
                  split_strategy="per-user-first-exposure-time-80-10-10")
train_u,train_i=arrays["train"]._users,arrays["train"]._items
counts=np.bincount(train_i,minlength=ni)
test=arrays["test"]
truth={}
for u,i in zip(test._users,test._items):
    truth.setdefault(int(u),set()).add(int(i))
users=list(range(nu))
seen={u:set(np.flatnonzero(seen_test[u]).tolist()) for u in users}
assert all(not(g & seen[u]) for u,g in truth.items())
protocol={
 "dataset":"KuaiRec 2.0 official small_matrix","raw_rows":raw_rows,"users":nu,"items":ni,
 "unique_first_exposures":int(np.isfinite(first).sum()),"ambiguous_first_labels_excluded":int(ambiguous.sum()),
 "missing_timestamp_rows":missing_time_rows,"unknown_time_pairs_excluded":len(unknown_time_pairs),
 "positive_rule":"first playback watch_ratio >= 1.0, frozen before modeling",
 "split":"Per-user distinct first-exposure timestamps: earliest 80% train, next 10% validation, last 10% test; equal timestamps kept together; boundary times from exposure times, without reading labels.",
 "seen_filter":"All first playbacks strictly before each evaluation cutoff, including negatives. Validation observations used only as known history/mask in test, not for optimizer updates.",
 "rationale":"Block holdout avoids an almost fully exposed user's last single positive creating only a handful of available candidates.",
 "split_counts":{k:len(v) for k,v in arrays.items()},"eval_users":len(truth),
 "cold_test_targets":int((counts[test._items]==0).sum()),
 "train_items":int((counts>0).sum()),"train_item_frequency_quantiles":np.quantile(counts[counts>0],[0,.5,.9,1]).tolist(),
 "test_unseen_candidate_quantiles":np.quantile([ni-len(seen[u]) for u in users],[0,.5,1]).tolist(),
 "two_tower":{"embedding_dim":32,"output_dim":32,"tower_type":"embedding","temperature":.1,"lr":.01,"batch_size":1024,
              "epochs":5,"seed":42,"selection":"best validation NDCG@20","mask_train_positive_false_negatives":True},
 "training_history":"Only this dataset; new initialization and independent mappings.",
}
(root/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
print("DATA",json.dumps(protocol),flush=True)

def top(scores,seen_map,cap=100,positive_only=False):
    rec={}
    for row,u in enumerate(users):
        s=np.asarray(scores[row],dtype=float).copy()
        s[list(seen_map[u])]=-np.inf
        eligible=np.flatnonzero(np.isfinite(s) & ((s>0) if positive_only else True))
        order=np.lexsort((eligible,-s[eligible]))[:cap]
        rec[u]=eligible[order].astype(int).tolist()
    return rec
def truth_for(split):
    g={}
    a=arrays[split]
    for u,i in zip(a._users,a._items):g.setdefault(int(u),set()).add(int(i))
    return g
val_truth=truth_for("validation")
# Score the full user mapping; each metric uses users with positives in its own split.
val_seen={u:set(np.flatnonzero(seen_val[u]).tolist()) for u in users}
metrics={}
def evaluate(name,recs):
    assert all(not(set(r)&seen[u]) and len(set(r))==len(r) for u,r in recs.items())
    m=compute_ranking_metrics(recs,truth,(20,100))
    metrics[name]={k:float(m[k]) for k in ("Recall@20","Recall@100","NDCG@20")}
    metrics[name]["hits@20"]=sum(len(set(recs[u][:20])&g) for u,g in truth.items())
    metrics[name]["hits@100"]=sum(len(set(recs[u])&g) for u,g in truth.items())
    with gzip.open(root/f"{name}.json.gz","wt",encoding="utf-8") as file:json.dump(recs,file)
    print("TEST",name,json.dumps(metrics[name]),flush=True)
evaluate("popularity",top(np.broadcast_to(counts,(len(users),ni)),seen))
inv=np.zeros(ni,dtype=np.float32)
np.divide(1,np.sqrt(counts),out=inv,where=counts>0)
z=csr_matrix((inv[train_i],(train_u,train_i)),shape=(nu,ni))
gram=z@z.T
# User self term only scores already-seen positives, which are filtered in evaluation.
scores=(gram[users]@z).toarray()
evaluate("itemcf",top(scores,seen,positive_only=True))
del z,gram,scores
gc.collect()
set_random_seed(42)
model=TwoTowerModel(nu,ni,embedding_dim=32,hidden_dim=64,output_dim=32,temperature=.1,tower_type="embedding")
optimizer=torch.optim.Adam(model.parameters(),lr=.01,fused=True)
train_positive_matrix=torch.from_numpy(masks["train"])
rng=np.random.default_rng(42)
history=[]
best_score=-1
best_state=None
for epoch in range(1,6):
    order=rng.permutation(len(train_u))
    def batches():
        for start in range(0,len(order),1024):
            idx=order[start:start+1024]
            yield {"user_idx":torch.from_numpy(train_u[idx].astype(np.int64)),
                   "item_idx":torch.from_numpy(train_i[idx].astype(np.int64))}
    loss=train_one_epoch(model,batches(),optimizer,torch.device("cpu"),train_positive_matrix=train_positive_matrix)
    with torch.no_grad():
        model.eval()
        user_vectors=model.encode_users(torch.tensor(users)).numpy()
        item_vectors=model.encode_items(torch.arange(ni)).numpy()
    scores=user_vectors@item_vectors.T
    val_recs=top(scores,val_seen)
    vm=compute_ranking_metrics(val_recs,val_truth,(20,100))
    history.append({"epoch":epoch,"loss":loss,"val_NDCG@20":vm["NDCG@20"]})
    if vm["NDCG@20"]>best_score:
        best_score=vm["NDCG@20"]
        best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        best_epoch=epoch
    print("TRAIN",history[-1],flush=True)
model.load_state_dict(best_state)
with torch.no_grad():
    scores=model.encode_users(torch.tensor(users)).numpy()@model.encode_items(torch.arange(ni)).numpy().T
evaluate("two_tower",top(scores,seen))
torch.save({"model_state_dict":best_state,"model_config":protocol["two_tower"],"dataset":"kuairec_small","best_epoch":best_epoch,
            "positive_threshold":1.0,"protocol":protocol},root/"two_tower.pt")
out={"protocol":protocol,"metrics":metrics,"training_history":history,"best_epoch":best_epoch,"wall_s":round(perf_counter()-t0,3)}
(root/"result.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print("FINISHED",json.dumps(out),flush=True)
