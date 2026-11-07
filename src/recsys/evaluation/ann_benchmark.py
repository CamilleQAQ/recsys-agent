"""精确检索 vs 近似检索对照（Phase 2 工作清单第 10 项）。

比较三种索引在同一份用户/物品向量上的表现：

* **暴力检索**：torch 全量 matmul，精确，作为质量与速度的参照；
* **IndexFlatIP**：FAISS 精确内积，复杂度同样是 O(N)；
* **IndexIVFFlat**：倒排 + 聚类，需要训练，nprobe 控制精度/速度；
* **IndexHNSWFlat**：图索引，efSearch 控制精度/速度。

指标：

* **质量**：与暴力检索 Top-M 的候选重合率、过滤曝光后的 Recall/NDCG/HR、Coverage；
* **速度**：2000 个用户的检索耗时（重复取中位数），只算检索、不算编码；
* **资源**：索引序列化后的字节数（Flat / IVF / HNSW 的内存近似）。

用法：

    python -m recsys.evaluation.ann_benchmark --checkpoint artifacts/kuasar_small/two_tower.pt
    python -m recsys.evaluation.ann_benchmark --checkpoint ... --output artifacts/ann.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch

from recsys.data.loader import (
    prepare_dataset,
    resolve_checkpoint_dataset,
    resolve_positive_threshold,
)
from recsys.data.schema import seen_items_for_split
from recsys.evaluation.evaluator import (
    brute_force_top_k,
    evaluate_recommendations,
    get_evaluation_interactions,
    group_items_by_user,
    load_model,
)
from recsys.retrieval.faiss_index import (
    as_float32_numpy,
    check_l2_normalized,
    encode_users_and_items,
)
from recsys.training.trainer import resolve_device

#: 检索时候选的池子大小；要大于「每个用户的曝光数 + K」，否则过滤后可能不足 K 个。
DEFAULT_K_SEARCH = 600


def _brute_force(
    user_indices: list[int],
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    k: int,
    batch_size: int = 512,
) -> dict[int, list[int]]:
    """精确 Top-K（不做曝光过滤，用于候选重合率比较）。"""
    tensor = torch.from_numpy(item_embeddings)
    recommendations: dict[int, list[int]] = {}
    for start in range(0, len(user_indices), batch_size):
        batch_users = user_indices[start : start + batch_size]
        scores = torch.from_numpy(user_embeddings[start : start + batch_size]) @ tensor.T
        top = torch.topk(scores, k=k, dim=1).indices.cpu().tolist()
        recommendations.update({user: items for user, items in zip(batch_users, top, strict=True)})
    return recommendations


def _index_search(
    index: faiss.Index, user_embeddings: np.ndarray, k: int, batch_size: int = 512
) -> tuple[list[list[int]], float]:
    """批量检索，返回 (每用户的内部 item_idx, 总耗时)。"""
    results: list[list[int]] = []
    started = time.perf_counter()
    for start in range(0, len(user_embeddings), batch_size):
        _, positions = index.search(user_embeddings[start : start + batch_size], k)
        results.extend(positions.tolist())
    elapsed = time.perf_counter() - started
    return results, elapsed


def _filter_and_recommend(
    user_indices: list[int],
    raw_items: list[list[int]],
    seen_items: Any,
    k: int,
) -> dict[int, list[int]]:
    """按曝光过滤后取前 K 个；不足 K 个的用户记录下来。"""
    recommendations: dict[int, list[int]] = {}
    for user, items in zip(user_indices, raw_items, strict=True):
        filtered = [
            item
            for item in items
            if item >= 0 and item not in seen_items.get(user, set())
        ][:k]
        recommendations[user] = filtered
    return recommendations


def _overlap(reference: dict[int, list[int]], candidate: dict[int, list[int]]) -> float:
    """候选重合率：|交集| / K，对用户取平均。"""
    overlaps = []
    for user, items in reference.items():
        target = set(candidate.get(user, []))
        overlaps.append(len(target.intersection(items)) / len(items) if items else 0.0)
    return float(np.mean(overlaps)) if overlaps else 0.0


def _index_size_mb(index: faiss.Index) -> float:
    return len(faiss.serialize_index(index)) / 1048576


def run(
    checkpoint_path: Path,
    data_dir: Path,
    *,
    split: str = "test",
    ks: tuple[int, ...] = (10, 20),
    k_search: int = DEFAULT_K_SEARCH,
    device_name: str = "auto",
    repeats: int = 3,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    model, checkpoint = load_model(checkpoint_path, device)
    dataset = resolve_checkpoint_dataset(checkpoint)
    threshold = resolve_positive_threshold(dataset, checkpoint.get("positive_threshold"))
    prepared = prepare_dataset(dataset, data_dir, positive_threshold=threshold)

    targets = get_evaluation_interactions(prepared, split)
    evaluation_users = sorted(group_items_by_user(targets))
    seen = seen_items_for_split(prepared, split)

    user_embeddings_t, item_embeddings_t = encode_users_and_items(
        model, evaluation_users, prepared.num_items, device
    )
    user_embeddings = as_float32_numpy(user_embeddings_t)
    item_embeddings = as_float32_numpy(item_embeddings_t)
    check_l2_normalized(item_embeddings, "item embeddings")
    check_l2_normalized(user_embeddings, "user embeddings")
    dim = int(item_embeddings.shape[1])
    k_search = min(k_search, prepared.num_items)
    print(
        f"users={len(evaluation_users):,} items={prepared.num_items:,} dim={dim} "
        f"k_search={k_search}",
        flush=True,
    )

    print("building brute-force reference …", flush=True)
    brute_reference = _brute_force(evaluation_users, user_embeddings, item_embeddings, k_search)

    results: dict[str, Any] = {
        "checkpoint": checkpoint_path.as_posix(),
        "dataset": dataset,
        "split": split,
        "users": len(evaluation_users),
        "items": prepared.num_items,
        "dimension": int(dim),
        "k_search": k_search,
        "indexes": {},
    }

    def record(name: str, index: faiss.Index | None, raw: list[list[int]], build_s: float,
               elapsed_s: float, index_mb: float, extra: dict[str, Any] | None = None) -> None:
        candidate = {user: items for user, items in zip(evaluation_users, raw, strict=True)}
        filtered = _filter_and_recommend(evaluation_users, raw, seen, max(ks))
        metrics = evaluate_recommendations(filtered, targets, ks)
        entry = {
            "overlap_at_k_search": round(_overlap(brute_reference, candidate), 6),
            "metrics": {key: round(value, 6) for key, value in metrics.items()},
            "build_seconds": round(build_s, 3),
            "search_seconds": round(elapsed_s, 3),
            "ms_per_user": round(elapsed_s * 1000 / len(evaluation_users), 3),
            "index_mb": round(index_mb, 2),
        }
        if extra:
            entry.update(extra)
        results["indexes"][name] = entry
        print(f"  {name:28s} overlap={entry['overlap_at_k_search']:.3f} "
              f"R@20={metrics.get('Recall@20', 0):.6f} "
              f"{entry['ms_per_user']:.2f} ms/user {entry['index_mb']:.1f} MB", flush=True)

    # 1) 暴力检索（reference 本身）
    timed = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _brute_force(evaluation_users, user_embeddings, item_embeddings, k_search)
        timed.append(time.perf_counter() - t0)
    record("brute_force", None, [brute_reference[u] for u in evaluation_users], 0.0, statistics.median(timed), 0.0)

    # 2) FAISS IndexFlatIP（精确）
    flat = faiss.IndexFlatIP(dim)
    t0 = time.perf_counter()
    flat.add(item_embeddings)
    flat_build = time.perf_counter() - t0
    flat_mb = _index_size_mb(flat)
    raw, _ = _index_search(flat, user_embeddings, k_search)
    timed = []
    for _ in range(repeats):
        timed.append(_index_search(flat, user_embeddings, k_search)[1])
    record("faiss_flat", flat, raw, flat_build, statistics.median(timed), flat_mb)

    # 3) IVF-Flat：nprobe 扫描
    nlist = min(4096, max(64, prepared.num_items // 128))
    quantizer = faiss.IndexFlatIP(dim)
    t0 = time.perf_counter()
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(item_embeddings)
    ivf.add(item_embeddings)
    ivf_build = time.perf_counter() - t0
    ivf_mb = _index_size_mb(ivf)
    for nprobe in (1, 8, 32, 128):
        ivf.nprobe = nprobe
        raw, _ = _index_search(ivf, user_embeddings, k_search)
        timed = []
        for _ in range(repeats):
            timed.append(_index_search(ivf, user_embeddings, k_search)[1])
        record(f"ivf_nprobe_{nprobe}", ivf, raw, ivf_build, statistics.median(timed), ivf_mb,
               {"nlist": nlist, "nprobe": nprobe})

    # 4) HNSW：efSearch 扫描
    hnsw = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 40
    t0 = time.perf_counter()
    hnsw.add(item_embeddings)
    hnsw_build = time.perf_counter() - t0
    hnsw_mb = _index_size_mb(hnsw)
    for ef in (16, 64, 256):
        hnsw.hnsw.efSearch = ef
        raw, _ = _index_search(hnsw, user_embeddings, k_search)
        timed = []
        for _ in range(repeats):
            timed.append(_index_search(hnsw, user_embeddings, k_search)[1])
        record(f"hnsw_ef_{ef}", hnsw, raw, hnsw_build, statistics.median(timed), hnsw_mb,
               {"ef_search": ef})

    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--k-search", type=int, default=DEFAULT_K_SEARCH)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = run(
        args.checkpoint,
        args.data_dir,
        split=args.split,
        ks=tuple(args.ks),
        k_search=args.k_search,
        device_name=args.device,
        repeats=args.repeats,
    )
    output = args.output or Path("artifacts") / f"ann_benchmark_{results['dataset']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[ok] 写入 {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
