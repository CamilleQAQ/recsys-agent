from collections import Counter
from pathlib import Path
from random import Random
from time import perf_counter
from unittest.mock import patch
import csv
import json
import gzip

import numpy as np
from recsys.evaluation.metrics import compute_ranking_metrics
from recsys.search.bm25 import BM25Index, IndexedItem
from recsys.search.encoders import build_encoder
from recsys.search.prepare import scan_sessions, parse_token_ids
from recsys.search.query import normalize_query
from recsys.search.service import SearchService
from recsys.search.vector_index import SearchVectorIndex

root=Path("experiments/2026-11-10-formal-e2e")
out=root/"search"
out.mkdir(parents=True,exist_ok=False)
raw=Path("data/kuasar/raw/KuaiSAR_v2")
started=perf_counter()
csv.field_size_limit(2**31-1)
prefix,scan=scan_sessions(raw/"src_inter.csv",50_000)
eligible=[s for s in prefix if s.keyword and s.keyword_consistent and
          all(i>0 for i in s.keyword) and len({r.item_id for r in s.exposures})>=2]
picked=Random(42).sample(eligible,min(1000,len(eligible)))
sessions={(s.user_id,s.session_id):{"session":s,"items":set(),"targets":set(),"query_consistent":True,"rows":0} for s in picked}
# Complete the sampled sessions even if they have later/non-contiguous records.
with (raw/"src_inter.csv").open(encoding="utf-8-sig",newline="") as file:
    reader=csv.reader(file)
    header={name:i for i,name in enumerate(next(reader))}
    a,b,c,d,e=[header[x] for x in ("user_id","search_session_id","keyword","item_id","click_cnt")]
    for raw_rows,row in enumerate(reader,1):
        target=sessions.get((row[a].strip(),row[b].strip()))
        if target is None:
            continue
        if parse_token_ids(row[c])!=target["session"].keyword:
            target["query_consistent"]=False
        item=int(row[d])
        target["rows"]+=1
        target["items"].add(item)
        if int(float(row[e]))>0:
            target["targets"].add(item)
valid=sorted((s for s in sessions.values() if s["query_consistent"]),
             key=lambda s:(float(s["session"].timestamp),s["session"].order))
assert valid
exposed=set.union(*(s["items"] for s in valid))
wanted=set(Random(42).sample(sorted(exposed),min(15000,len(exposed))))
print("Search sessions",len(valid),"exposed unique",len(exposed),"index input cap",len(wanted),flush=True)
catalog={}
seen_feature_ids=set()
with (raw/"item_features.csv").open(encoding="utf-8-sig",newline="") as file:
    reader=csv.reader(file)
    header={name:i for i,name in enumerate(next(reader))}
    item_col,caption_col,type_col,author_col=[header[x] for x in ("item_id","caption","item_type","author_id")]
    cat_cols=[header[f"{level}_level_category_name_en"] for level in ("first","second","third","fourth")]
    for feature_rows,row in enumerate(reader,1):
        item=int(row[item_col])
        if item not in wanted:
            continue
        seen_feature_ids.add(item)
        tokens=parse_token_ids(row[caption_col])
        if tokens:
            category=" / ".join(row[c].strip() for c in cat_cols if row[c].strip())
            catalog[item]=IndexedItem(item,tokens,category,item_type=row[type_col],author_id=int(float(row[author_col] or -1)))
        if seen_feature_ids==wanted:
            break
print("Search captions",len(catalog),"feature rows",feature_rows,flush=True)
assert len(catalog)>120
docs=[catalog[i] for i in sorted(catalog)]
encoder=build_encoder({"name":"token_hash","dim":4096,"seed":20261109,"include_category":True,"category_weight":0.25})
index_path=out/"index.json"
vector_dir=out/"vector"
BM25Index(docs).save(index_path)
SearchVectorIndex.build(docs,encoder,cache_dir=vector_dir,backend="faiss")
service=SearchService.load(index_path,vector_dir,encoder)
assert service.vector.faiss_active and service.vector.backend_actual=="faiss.IndexFlatIP"
marks={}
outputs={m:{} for m in ("bm25","vector","hybrid","rerank")}
elapsed={m:[] for m in outputs}
times=Counter()
for i,row in enumerate(valid):
    form=normalize_query(row["session"].keyword)
    if row["targets"]:
        marks[i]=row["targets"]
    t=perf_counter()
    bm=service.search_bm25(form.tokens,top_k=50)
    bm_s=perf_counter()-t
    t=perf_counter()
    vector=service.search_vector(form.tokens,top_k=50)
    vector_s=perf_counter()-t
    sources={"bm25":[(hit.item_id,hit.score) for hit in bm],"vector":[(hit.item_id,hit.score) for hit in vector]}
    with patch.object(service.hybrid,"rankings",return_value=sources):
        t=perf_counter()
        hybrid=service.reranker._base.fetch(form.tokens,50)
        fusion_s=perf_counter()-t
        t=perf_counter()
        ranked=service.rerank(form.tokens,top_k=20)
        rerank_s=perf_counter()-t
    for name,items,sec in (
        ("bm25",bm,bm_s),("vector",vector,vector_s),
        ("hybrid",hybrid,bm_s+vector_s+fusion_s),
        ("rerank",ranked,bm_s+vector_s+fusion_s+rerank_s),
    ):
        outputs[name][i]=[hit.item_id for hit in items[:20]]
        elapsed[name].append(sec*1000)
        assert len(outputs[name][i])==len(set(outputs[name][i]))
    assert set(outputs["rerank"][i]) <= {h.item_id for h in hybrid}
    times[form.normalization]+=1
assert marks
result={}
for name,recs in outputs.items():
    m=compute_ranking_metrics(recs,marks,ks=(10,20))
    result[name]={"Recall@20":m["Recall@20"],"NDCG@10":m["NDCG@10"],
                  "HitRate@10":m["HitRate@10"],"empty_sessions":sum(not r for r in recs.values()),
                  "median_ms":float(np.median(elapsed[name])),"p95_ms":float(np.quantile(elapsed[name],.95))}
with gzip.open(out/"query_results.jsonl.gz","wt",encoding="utf-8") as file:
    for i,row in enumerate(valid):
        session=row["session"]
        record={"session_id":session.session_id,"user_id":session.user_id,"timestamp":session.timestamp,
                "query_tokens":session.keyword,"target_items":sorted(row["targets"]),
                "exposures":row["rows"],"top20":{name:recs[i] for name,recs in outputs.items()}}
        file.write(json.dumps(record,separators=(",",":"))+"\n")
total=sum(len(g) for g in marks.values())
covered=sum(len(g & catalog.keys()) for g in marks.values())
payload={"sessions":len(valid),"sessions_with_click":len(marks),"clicked_targets":total,"in_index_click_targets":covered,
         "target_coverage":covered/total,"unique_exposed_items":len(exposed),"index_items":len(catalog),
         "sample_input_scan":scan,"completed_session_scan_rows":raw_rows,"feature_rows":feature_rows,
         "query_normalization":dict(times),"faiss_backend":service.vector.backend_actual,
         "faiss_calls":service.vector.faiss_search_calls,"encoder":"training-free token hash",
         "protocol":"Seed 42 over complete, valid query sessions in first 50k exposure rows. Selected sessions completed across raw log. Shared label-blind exposure corpus with caption availability; binary logged clicks as relevance; missing documents count as misses; metrics over clicked sessions.",
         "timing_protocol":"Per-query measured stages with cached shared upstream rankings, summed for composite modes; excludes index load/build.",
         "metrics":result,"wall_s":round(perf_counter()-started,3)}
(out/"result.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(payload,ensure_ascii=False),flush=True)
