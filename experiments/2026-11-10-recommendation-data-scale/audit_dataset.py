from pathlib import Path
import json
import numpy as np
import torch
from recsys.data.loader import prepare_dataset
from recsys.data.schema import seen_items_for_split

root=Path("experiments/2026-11-10-recommendation-data-scale")
root.mkdir(parents=True,exist_ok=True)
def bins(values):
    return {"0":int((values==0).sum()),"1":int((values==1).sum()),"2-5":int(((values>=2)&(values<=5)).sum()),
            "6-20":int(((values>=6)&(values<=20)).sum()),"21+":int((values>=21).sum())}
results={}
for name in ("kuasar_small","kuasar"):
    p=Path("data")/name/"processed"
    train_users,train_items,valid_users,valid_items,test_users,test_items=(
        np.load(p/"split"/f"{part}_{field}.npy",mmap_mode="r")
        for part in ("train","validation","test") for field in ("users","items")
    )
    vocab_users=np.load(p/"ids"/"user_ids.npy",mmap_mode="r")
    vocab_items=np.load(p/"ids"/"item_ids.npy",mmap_mode="r")
    user_counts=np.bincount(train_users,minlength=len(vocab_users))
    counts=np.bincount(train_items,minlength=len(vocab_items))
    positives=counts[counts>0]
    by_test=counts[test_items]
    by_test_unique=counts[np.unique(test_items)]
    before_test=counts+np.bincount(valid_items,minlength=len(vocab_items))
    item_frequency=bins(counts)
    results[name]={"train_interactions":len(train_items),"validation_interactions":len(valid_items),
        "test_interactions":len(test_items),"catalog_items":len(vocab_items),"users_in_mapping":len(vocab_users),
        "active_train_users":int((user_counts>0).sum()),"train_clicked_items":len(positives),
        "test_positive_users":len(np.unique(test_users)),"test_unique_target_items":len(by_test_unique),
        "train_item_interaction_distribution":item_frequency,"train_clicked_item_quantiles":dict(zip(
            ["p25","p50","p75","p90","p95","p99","max"],[float(x) for x in np.quantile(positives,[.25,.5,.75,.9,.95,.99,1])])),
        "train_interactions_per_clicked_item":float(np.mean(positives)),
        "train_interactions_per_active_user":float(np.mean(user_counts[user_counts>0])),
        "matrix_density":len(train_items)/(len(vocab_items)*len(vocab_users)),
        "test_targets_by_train_item_frequency":bins(by_test),"unique_test_items_by_train_item_frequency":bins(by_test_unique),
        "test_cold_share":float((by_test==0).mean()),
        "test_targets_without_train_or_validation_click":int((before_test[test_items]==0).sum()),
        "catalog_items_without_train_click":item_frequency["0"]}
    if name=="kuasar_small":
        data=prepare_dataset(name,Path("data"))
        seen=seen_items_for_split(data,"test")
        by_user={int(u):seen.get(int(u),set()) for u in np.unique(test_users)}
        results[name]["test_targets_filtered_as_seen"]=sum(int(i) in by_user[int(u)] for u,i in zip(test_users,test_items))
        paths=Path("artifacts/kuasar_small")
        results["used_checkpoints"]={}
        for model,filename in (("two_tower","two_tower_features.pt"),("mind","mind20.pt")):
            w=torch.load(paths/filename,map_location="cpu",weights_only=False)
            prov=w.get("provenance") or {}
            results["used_checkpoints"][model]={"path":str(paths/filename),"dataset":w.get("dataset"),
                "model_config":w.get("model_config"),"train_config":w.get("train_config"),
                "counts":(prov.get("data") or {}).get("counts"),"split_sha256":(prov.get("data") or {}).get("split_sha256"),
                "id_mapping_sha256":(prov.get("id_mapping") or {}).get("sha256")}
            del w
search=Path("artifacts/search/kuasar_vector/vector/doc_ids.json")
ids=json.loads(search.read_text(encoding="utf-8"))
if isinstance(ids,list):
    small=Path("data/kuasar_small/processed")
    original=np.load(small/"ids"/"item_ids.npy",mmap_mode="r")
    trained=original[np.unique(np.load(small/"split"/"train_items.npy",mmap_mode="r"))]
    results["separate_search_faiss"]={"documents":len(ids),"overlap_with_small_training_items":len(set(map(int,ids))&set(trained))}
meta=Path("experiments/2026-11-10-formal-e2e/recommendation/run_meta.json")
results["current_pipeline"]={"artifact":str(meta),"backend":"numpy exact dot product for Two-Tower; MIND score_users exact dot product",
    "uses_recommendation_faiss":False,"data":json.loads(meta.read_text(encoding="utf-8"))["provenance"]["data"]}
(root/"data_audit.json").write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps(results,ensure_ascii=False,indent=2))
