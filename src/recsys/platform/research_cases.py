"""Create explicit, dataset-independent research snapshots from existing interfaces.

These functions do inference only. A snapshot fixes the candidate pool, labels,
features and baseline for a code-editing experiment; it is not a training API.
"""
from pathlib import Path
import gzip
import json
import numpy as np

from recsys.platform.autoresearch import write_json


def recommendation_case(channels_path, split_dir, output, *, models, users=32):
    """Read saved validation recall channels, preserving labels outside the pool."""
    with gzip.open(channels_path, "rt", encoding="utf-8") as f:
        channels = json.load(f)
    split_dir = Path(split_dir)
    train = np.load(split_dir / "train.npz")
    val = np.load(split_dir / "validation.npz")
    maximum = max(int(val["items"].max()), max(i for name in models for row in channels[name].values() for i in row))
    counts = np.bincount(train["items"], minlength=maximum+1)
    truth = {}
    for u, i in zip(val["users"], val["items"]):
        truth.setdefault(str(int(u)), set()).add(int(i))
    groups = []
    for user in sorted(truth, key=int)[:users]:
        rows = {}
        # Stable interleaving is the same fixed multi-channel baseline as the project.
        for rank in range(max(len(channels[n][user]) for n in models)):
            for name in models:
                values = channels[name][user]
                if rank < len(values):
                    item = values[rank]
                    row = rows.setdefault(item, {"item": item, "cold": bool(item >= len(counts) or counts[item] == 0),
                                                "train_count": int(counts[item]) if item < len(counts) else 0,
                                                "ranks": {}})
                    row["ranks"][name] = rank+1
        groups.append({"id": user, "candidates": list(rows.values()), "positives": sorted(truth[user])})
    case = {"business": "recommendation", "dataset": split_dir.name, "split": "validation",
            "objective": "NDCG@20", "ks": [20, 100], "models": list(models), "groups": groups,
            "cold_items": np.flatnonzero(counts == 0).tolist(),
            "context": {"protocol": "Existing validation split and frozen recall snapshots; not a fresh blind test",
                        "scope": "Fusion/candidate-order algorithm research; fixed retrieved pool",
                        "channels_path": str(Path(channels_path).resolve())},
            "source_files": ["src/recsys/pipeline/fusion.py"]}
    write_json(output, case)
    return case


def search_case(output, *, config_path="configs/experiments/search_agent.toml", sessions=8):
    """Use Search Agent's existing service, BM25/FAISS and real click labels."""
    import tomllib
    from recsys.search.service import SearchService
    from recsys.search.encoders import build_encoder
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    agent, backend = cfg["search_agent"], cfg["search_backend"]
    encoder = build_encoder({"encoder": backend["encoder"], "dim": backend["encoder_dim"],
        "seed": backend["encoder_seed"], "include_category": backend["include_category"],
        "category_weight": backend["category_weight"]})
    service = SearchService.load(agent["index_path"], backend["vector_index_dir"], encoder)
    sample = json.loads(Path(agent["sample_path"]).read_text(encoding="utf-8"))
    catalog = {x.item_id: x for x in service.bm25.items}
    groups = []
    for session in sample["sessions"][:sessions]:
        tokens = session["keyword"]
        rankings = {name: service.search(name, tokens, top_k=len(catalog)) for name in ["bm25", "vector", "hybrid"]}
        rows = {}
        for name, hits in rankings.items():
            for hit in hits:
                item = catalog[hit.item_id]
                row = rows.setdefault(hit.item_id, {"item": hit.item_id,
                    "coverage": len(set(tokens) & set(item.caption_tokens))/max(1,len(set(tokens))),
                    "category": item.category, "ranks": {}, "scores": {}})
                row["ranks"][name] = hit.rank
                row["scores"][name] = hit.score
        order = [h.item_id for h in rankings["hybrid"]]
        order.extend(i for i in rows if i not in order)
        groups.append({"id": str(session["session_id"]), "query_tokens": tokens,
                       "candidates": [rows[i] for i in order], "positives": session["clicked_items"]})
    case = {"business": "search", "dataset": "kuasar_search_sample", "split": "validation",
            "objective": "NDCG@5", "ks": [5, 20], "models": ["bm25", "faiss_token_hash"],
            "groups": groups, "context": {"scope": "Existing query sample, real clicks; validation-only development snapshot, not a held-out search benchmark",
             "catalog_size": len(catalog), "candidate_scope": "Existing indexed catalog, label coverage reported separately"},
            "source_files": ["src/recsys/search/rerank.py"]}
    write_json(output, case)
    return case


def ads_case(output, *, dataset="kuasar_small", models=("din",), requests=16, min_candidates=2):
    """Reuse AdsRuntime to score real request candidates; never create weights."""
    from recsys.ads.runtime import AdsRuntime
    runtime = AdsRuntime(dataset=dataset)
    view = runtime.view("validation")
    chosen, rows = runtime.request_rows("validation", count=requests, min_candidates=min_candidates, prefer_large=False)
    batch = runtime.batch("validation", rows)
    scores = {name: np.asarray(runtime.model(name).score(batch.as_dict())["ctr"]) for name in models}
    weights = {name: runtime.weight_info(name) for name in models}
    groups = []
    offset = 0
    for index in chosen:
        count = int(view.groups[index])
        candidates = [{"item": int(rows[p]), "video_item": int(batch.items[p]),
                       "scores": {n: float(s[p]) for n,s in scores.items()}}
                      for p in range(offset, offset+count)]
        # row ID is unique even if a request contains repeated video IDs.
        candidates.sort(key=lambda x: (-x["scores"][models[0]], x["item"]))
        positives = [int(rows[p]) for p in range(offset, offset+count) if batch.labels[p] > 0]
        groups.append({"id": str(int(index)), "timestamp": int(view.timestamps[index]),
                       "candidates": candidates, "positives": positives})
        offset += count
    case = {"business": "ads", "dataset": dataset, "split": "validation", "objective": "NDCG@5",
            "ks": [5, 20], "models": list(models), "groups": groups,
            "context": {"scope": "Offline real-click request ranking, not CVR or revenue optimization",
                        "minimum_candidates": min_candidates,
                        "candidate_selection": "first eligible requests by stored request order, no click-label selection",
                        "weights": weights}, "source_files": ["src/recsys/ads/ranking.py"]}
    write_json(output, case)
    return case
