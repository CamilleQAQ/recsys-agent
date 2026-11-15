from pathlib import Path
import gzip
import json
import math
import numpy as np

root=Path("experiments/2026-11-10-recommendation-data-scale")
base=Path("data/kuasar_small/processed/split")
def load(part,col):
    return np.load(base/f"{part}_{col}.npy",mmap_mode="r")
u_test,i_test=load("test","users"),load("test","items")
all_users=sorted(set(map(int,u_test)))
test_truth={u:set() for u in all_users}
for u,i in zip(u_test,i_test):
    test_truth[int(u)].add(int(i))
frequencies=np.bincount(load("train","items"),minlength=454717)
discount=np.array([1/math.log2(i+2) for i in range(20)])
key_metrics=("Recall@20","Recall@100","NDCG@20")
known=json.loads((root/"results.json").read_text(encoding="utf-8"))["metrics"]
per_user={}
exposure={}
for name,result in known.items():
    with gzip.open(root/f"{name}.json.gz","rt",encoding="utf-8") as f:
        recs=json.load(f)
    rows=[]
    cold_recs=0
    total_recs=0
    for u in all_users:
        r=recs[str(u)]
        target=test_truth[u]
        y=np.asarray([i in target for i in r],dtype=np.float64)
        rows.append([y[:20].sum()/len(target),y[:100].sum()/len(target),
                     float(discount[:len(y[:20])]@y[:20])/discount[:len(target)].sum()])
        cold_recs+=int((frequencies[r[:100]]==0).sum()) if r else 0
        total_recs+=len(r[:100])
    a=np.asarray(rows)
    assert np.allclose(a.mean(axis=0),[result["overall"][m] for m in key_metrics])
    per_user[name]=a
    exposure[name]={"cold_recommendation_slots":cold_recs,"total_recommendation_slots":total_recs,
                    "cold_share":cold_recs/total_recs if total_recs else None}
pairs=[
    ("two_tower_100","two_tower_25"),("two_tower_100","two_tower_50"),
    ("itemcf_100","itemcf_25"),("popularity_100","popularity_25"),
    ("popularity_100","two_tower_100"),("itemcf_100","old_final"),
    ("popularity_100","old_final"),("old_final","old_round_robin"),
]
rng=np.random.default_rng(20261110)
sample=rng.integers(0,len(all_users),size=(4000,len(all_users)),dtype=np.int32)
contrasts=[]
for x,y in pairs:
    d=per_user[x]-per_user[y]
    stats=d[sample].mean(axis=1)
    contrasts.append({"first":x,"second":y,"metrics":{
        m:{"difference":float(d[:,j].mean()),"paired_user_bootstrap_95pct_interval":np.quantile(stats[:,j],[.025,.975]).tolist()}
        for j,m in enumerate(key_metrics) if m!='Recall@100' or 'old_final' not in (x,y)}})
out={"cold_exposure_using_full_small_train":exposure,
     "paired_uncertainty":{"protocol":"4000 paired test-user resamples; single fixed trained seed. Intervals reflect test sampling only.","comparisons":contrasts}}
(root/"analysis.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"cold_exposure":{m:exposure[m] for m in ("old_two_tower","old_mind","old_final","two_tower_100")},
                  "contrasts":contrasts},ensure_ascii=False,indent=2))
