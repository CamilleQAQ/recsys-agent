from pathlib import Path
from time import perf_counter
from collections import Counter
import gzip
import hashlib
import json

import faiss
import numpy as np
from scipy.sparse import csr_matrix
import torch

from recsys.data.loader import prepare_dataset
from recsys.data.schema import seen_items_for_split
from recsys.evaluation.metrics import recall_at_k,ndcg_at_k
from recsys.models.two_tower import TwoTowerModel
from recsys.retrieval.faiss_index import build_faiss_index,faiss_top_k_from_embeddings
from recsys.training.trainer import build_item_feature_tensors,train_one_epoch
from recsys.utils.seed import set_random_seed

root=Path("experiments/2026-11-10-recommendation-data-scale")
source=Path("data/kuasar_small/processed/split")
historic=Path("experiments/2026-11-10-formal-e2e/recommendation")
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))
def write(name,payload):
    (root/name).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
def load(part,column):
    return np.load(source/f"{part}_{column}.npy",mmap_mode="r")
torch.set_num_threads(4)
faiss.omp_set_num_threads(4)
t0=perf_counter()
seed=42
data=prepare_dataset("kuasar_small",Path("data"))
n=data.num_items
u_count=data.num_users
train_u,train_i,test_u,test_i=(load(split,col) for split in ("train","test") for col in ("users","items"))
users=sorted(map(int,np.unique(test_u)))
view=seen_items_for_split(data,"test")
seen={u:view.get(u,set()) for u in users}
all_truth={u:set() for u in users}
for u,i in zip(test_u,test_i):
    all_truth[int(u)].add(int(i))
truth_count=sum(map(len,all_truth.values()))
assert len(users)==2000 and truth_count==4275 and len(train_i)==748674
full_counts=np.bincount(train_i,minlength=n)
device=torch.device("cpu")
features=build_item_feature_tensors(data)
reference=read(root/"data_audit.json")["used_checkpoints"]["two_tower"]
model_config=reference["model_config"]
batch=int(reference["train_config"]["batch_size"])
lr=float(reference["train_config"]["learning_rate"])
steps=(len(train_i)+batch-1)//batch
exposures=steps*batch
fractions=(25,50,100)
sizes={pct:len(train_i)*pct//100 for pct in fractions}
subset_order=np.random.default_rng(seed).permutation(len(train_i))
protocol={
    "test_split":"unchanged KuaiSAR-small held-out last request; all 2000 test users; original test exposure mask used for every arm",
    "counts":{"users":len(users),"positives":truth_count,"catalog":n,"masked_test_pairs":sum(len(g & seen[u]) for u,g in all_truth.items())},
    "training":{"nested_fraction_percent":list(fractions),"unique_positive_rows":sizes,
        "sampling":"One fixed permutation of the original train split; 25% prefix contained in 50%, contained in 100%. Each arm cycles shuffled subsets to equalize updates, full subset covered once before repetition.",
        "same_optimizer_steps":steps,"same_positive_row_exposures":exposures,"batch_size":batch,"learning_rate":lr,
        "optimizer":"torch Adam (fused CPU implementation); no weight decay; project in-batch cross-entropy, same init seed",
        "negative_mask":False,"model_config":model_config,"seed":seed,"device":"CPU, 4 torch and FAISS threads",
        "checkpoint_selection":"Final fixed step; no tuning on validation or test. Prior production checkpoint had 20 epochs; treated as historical reference only."},
    "index":"Exact FAISS IndexFlatIP, same entire ordered item ID vocabulary for all Two-Tower arms, including zero-training-click items.",
    "itemcf":"Exact sum of cosine co-click similarities to each user's train items; sparse factorization avoids item-by-item matrix. Positive-neighbor candidates only.",
    "popularity":"Descending train click count, ties ascending item ID; only positive-count candidates.",
    "cold_breakdown":"Group membership fixed using the full small training split, for fair comparison. Per-arm cold counts also recorded.",
}
write("protocol.json",protocol)
all_metrics={}
scale={}
def view_truth(item_filter):
    return {u:h for u,g in all_truth.items() if (h:={i for i in g if item_filter(i)})}
groups={"cold":view_truth(lambda i:full_counts[i]==0),"warm":view_truth(lambda i:full_counts[i]>0),
        "one_click":view_truth(lambda i:full_counts[i]==1),
        "two_to_five":view_truth(lambda i:2<=full_counts[i]<=5),
        "six_to_twenty":view_truth(lambda i:6<=full_counts[i]<=20),
        "twenty_one_plus":view_truth(lambda i:full_counts[i]>=21)}
def compute(recs,g):
    return {"users":len(g),"targets":sum(map(len,g.values())),
            "Recall@20":recall_at_k(recs,g,20),"Recall@100":recall_at_k(recs,g,100),
            "NDCG@20":ndcg_at_k(recs,g,20),
            "hit_targets_20":sum(len(set(recs[u][:20]) & target) for u,target in g.items()),
            "hit_targets_100":sum(len(set(recs[u][:100]) & target) for u,target in g.items())}
def record(name,recs,elapsed=None):
    assert set(recs)==set(users)
    assert all(len(set(row))==len(row) and not (set(row)&seen[u]) for u,row in recs.items())
    overall=compute(recs,all_truth)
    all_metrics[name]={"overall":overall,"full_train_frequency_groups":{g:compute(recs,y) for g,y in groups.items()},
                       "users_with_full_top100":sum(len(r)==100 for r in recs.values()),"wall_s":elapsed}
    with gzip.open(root/f"{name}.json.gz","wt",encoding="utf-8") as file:
        json.dump(recs,file,separators=(",",":"))
    print(name,json.dumps(overall),elapsed,flush=True)
    write("results.json",{"metrics":all_metrics,"scales":scale,"elapsed_s":round(perf_counter()-t0,3)})
def popularity(counts):
    eligible=np.flatnonzero(counts)
    top=eligible[np.argsort(-counts[eligible],kind="stable")]
    from itertools import islice
    return {u:list(islice((int(i) for i in top if int(i) not in seen[u]),100)) for u in users}

def itemcf(us,it,counts):
    # (X D^-1/2) (X D^-1/2)^T (X D^-1/2), row by row, is sum of
    # cosine-normalized item co-click scores over a user's training history.
    inv=np.zeros(n,dtype=np.float32)
    np.divide(1,np.sqrt(counts),where=counts>0,out=inv)
    z=csr_matrix((inv[it],(us,it)),shape=(u_count,n),dtype=np.float32)
    gram=(z @ z.T).tocsr()
    gram.setdiag(0)
    gram.eliminate_zeros()
    recs={}
    for start in range(0,len(users),16):
        block=users[start:start+16]
        scores=(gram[block] @ z).tocsr()
        for j,u in enumerate(block):
            a,b=scores.indptr[j:j+2]
            ids=scores.indices[a:b]
            vals=scores.data[a:b]
            keep=~np.isin(ids,np.fromiter(seen[u],dtype=np.int32))
            ids,vals=ids[keep],vals[keep]
            order=np.lexsort((ids,-vals))[:100]
            recs[u]=ids[order].astype(int).tolist()
    return recs

def train_tower(pct,rows):
    set_random_seed(seed)
    model=TwoTowerModel(**model_config).to(device)
    model.item_cat_features.copy_(torch.from_numpy(features["cat"]))
    model.item_numeric_features.copy_(torch.from_numpy(features["numeric"]))
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,fused=True)
    rng=np.random.default_rng(seed+1)
    positions=np.concatenate([rng.permutation(rows) for _ in range((exposures+len(rows)-1)//len(rows))])[:exposures]
    assert len(np.unique(positions))==len(rows)
    def batches():
        for step,idx in enumerate(positions.reshape(steps,batch),1):
            yield {"user_idx":torch.from_numpy(np.asarray(train_u[idx],dtype=np.int64)),
                   "item_idx":torch.from_numpy(np.asarray(train_i[idx],dtype=np.int64))}
            if step%60==0 or step==steps:
                print("TRAIN",pct,step,"/",steps,"wall",round(perf_counter()-t0,1),flush=True)
    started=perf_counter()
    loss=train_one_epoch(model,batches(),optimizer,device)
    return model,loss,perf_counter()-started

@torch.inference_mode()
def retrieve(model,counts):
    started=perf_counter()
    model.eval()
    item_vectors=np.empty((n,model_config["output_dim"]),dtype=np.float32)
    for a in range(0,n,8192):
        b=min(a+8192,n)
        item_vectors[a:b]=model.encode_items(torch.arange(a,b)).numpy()
    user_vectors=model.encode_users(torch.tensor(users)).numpy()
    index=build_faiss_index(item_vectors,np.arange(n),data.mappings.index_to_item)
    pos=index.position_to_item_idx
    in_index=counts[pos]
    assert index.num_items==n and np.array_equal(pos,np.arange(n))
    coverage={"faiss_backend":type(index.index).__name__,"faiss_ntotal":index.num_items,
              "training_items_in_index":int((in_index>0).sum()),"zero_train_click_items_in_index":int((in_index==0).sum()),
              "test_targets_in_index":int(np.isin(test_i,pos).sum())}
    recs=faiss_top_k_from_embeddings(users,user_vectors,index,seen,k=100,initial_candidates=256)
    return recs,coverage,perf_counter()-started

# Tiny unit of evidence for the ItemCF math, independent of the benchmark.
z=np.array([[1.,1.,0.],[1.,0.,1.]],dtype=np.float32)
weights=np.divide(1,np.sqrt(z.sum(axis=0)))
z=z*weights
assert np.allclose((z@z.T)@z,z@(z.T@z))

for pct in fractions:
    rows=subset_order[:sizes[pct]]
    us,it=train_u[rows],train_i[rows]
    counts=np.bincount(it,minlength=n)
    scale[str(pct)]={"unique_train_rows":len(rows),"active_train_users":len(np.unique(us)),
                     "train_items":int((counts>0).sum()),"test_cold_targets":int((counts[test_i]==0).sum()),
                     "indices_sha256":hashlib.sha256(np.sort(rows).astype("<i8").tobytes()).hexdigest()}
    print("ARM",pct,scale[str(pct)],flush=True)
    for method,fun in (("popularity",lambda:popularity(counts)),("itemcf",lambda:itemcf(us,it,counts))):
        t=perf_counter()
        recs=fun()
        record(f"{method}_{pct}",recs,round(perf_counter()-t,3))
    model,loss,t_train=train_tower(pct,rows)
    recs,coverage,t_index=retrieve(model,counts)
    scale[str(pct)].update(coverage,training_loss=loss,training_s=round(t_train,3),encode_index_and_retrieve_s=round(t_index,3))
    record(f"two_tower_{pct}",recs,round(t_train+t_index,3))
    del model

for file,name in (("recall_two_tower.json","old_two_tower"),("recall_mind.json","old_mind"),
                  ("fusion.json","old_round_robin"),("ranked.json","old_lightgbm"),("final.json","old_final")):
    blob=read(historic/file)
    recs={int(u):[int(x["item_idx"]) for x in (row["items"] if isinstance(row,dict) else row)] for u,row in blob.items()}
    record(name,recs)
print("FINISHED",round(perf_counter()-t0,3),flush=True)
