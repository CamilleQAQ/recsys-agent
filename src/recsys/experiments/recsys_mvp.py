"""kuasar_small 离线推荐 MVP：双路召回 → 名次融合 → LightGBM 排序 → 类目重排。

一条命令跑完（仓库根目录）：

    python -m recsys.experiments.recsys_mvp --dataset kuasar_small --users 128 --seed 42

设计约束（对应任务书）：

* 只对 seed 42 抽出的至多 128 个验证用户打分，**不**为全部 2,000 用户算分；
* 两路召回各只算一次精确 Top-100，分批处理并缓存；MIND 路由种子固定 1234；
* 融合按两路原始名次轮流取、去重，最多 100 个，来源与名次可追溯；
* LightGBM 只训练一次：从父数据集 ``data/kuasar/processed/ranking/`` 只读 train 工件，
  按原始 ID 对齐父子词表，保留请求边界，最多 10,000 行 / 20 轮；
* 重排：Top-20 首遍每个已知一级类目最多 5 个，缺失类目不占配额，不足时按序补齐并计数。

产物落在 ``artifacts/kuasar_small/mvp/``，指标与阶段耗时写 ``metrics.json``。
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import time
from pathlib import Path

# 线程数必须在 numpy / torch 真正开始计算前限制住（命令行也会设置同名环境变量）。
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from recsys.adapters.ranking import LightGBMRankerAdapter  # noqa: E402
from recsys.adapters.registry import load_adapter  # noqa: E402
from recsys.config import DEFAULT_DATA_DIR  # noqa: E402
from recsys.data.dataset import build_observed_histories  # noqa: E402
from recsys.data.loader import prepare_dataset  # noqa: E402
from recsys.data.schema import seen_items_for_split  # noqa: E402
from recsys.evaluation.evaluator import (  # noqa: E402
    evaluate_recommendations,
    group_items_by_user,
)
from recsys.evaluation.metrics import catalog_coverage  # noqa: E402
from recsys.models.mind import rank_users  # noqa: E402
from recsys.utils.provenance import build_provenance  # noqa: E402
from recsys.data.kuasar_ranking import ITEM_FEATURE_COLUMNS  # noqa: E402

TWO_TOWER_CHECKPOINT = Path("artifacts/kuasar_small/two_tower_features.pt")
MIND_CHECKPOINT = Path("artifacts/kuasar_small/mind20.pt")
ROUTING_SEED = 1234
DEFAULT_OUTPUT = Path("artifacts/kuasar_small/mvp")


# --------------------------------------------------------------------------- 工具


def peak_rss_mib() -> float:
    """当前进程的峰值工作集（Windows）。"""

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    psapi = ctypes.windll.psapi
    kernel32 = ctypes.windll.kernel32
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        return -1.0
    return round(counters.PeakWorkingSetSize / 1048576, 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------- 召回


def select_users(data, count: int, seed: int) -> np.ndarray:
    """从验证集用户里用固定种子抽至多 count 个（排序后保存，保证可复现）。"""
    validation_users = sorted({row.user_idx for row in data.validation_interactions})
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(validation_users), size=min(count, len(validation_users)), replace=False)
    return np.sort(chosen.astype(np.int64))


def top_k_from_scores(
    scores: np.ndarray, seen: set[int], k: int
) -> tuple[list[int], list[float]]:
    """单个用户的精确 Top-K；已曝光物品置 -inf，返回 (item_idx, score)。"""
    row = scores.copy()
    if seen:
        row[list(seen)] = -np.inf
    k = min(k, row.size)
    top = np.argpartition(-row, k - 1)[:k]
    top = top[np.argsort(-row[top])]
    return [int(x) for x in top], [round(float(row[x]), 6) for x in top]


def recall_two_tower(data, users: np.ndarray, k: int, batch_size: int = 32) -> dict[int, list[dict]]:
    adapter = load_adapter("two_tower", TWO_TOWER_CHECKPOINT, device_name="cpu")
    item_vectors = adapter.encode_items(range(data.num_items))
    user_vectors = adapter.encode_users([int(u) for u in users])
    seen = seen_items_for_split(data, "validation")
    results: dict[int, list[dict]] = {}
    for start in range(0, len(users), batch_size):
        batch = users[start : start + batch_size]
        scores = user_vectors[start : start + batch_size] @ item_vectors.T
        for offset, user in enumerate(batch):
            items, values = top_k_from_scores(scores[offset], seen.get(int(user), set()), k)
            results[int(user)] = [
                {"item": item, "score": value, "rank": rank + 1}
                for rank, (item, value) in enumerate(zip(items, values))
            ]
    return results


def recall_mind(data, users: np.ndarray, k: int) -> dict[int, list[dict]]:
    adapter = load_adapter("mind", MIND_CHECKPOINT, device_name="cpu")
    histories = build_observed_histories(data, "validation")  # 验证输入只用训练历史
    seen = seen_items_for_split(data, "validation")
    recommendations = rank_users(
        adapter.model,
        [int(u) for u in users],
        histories,
        data.num_items,
        max_length=int(adapter.checkpoint["model_config"]["max_length"]),
        device=adapter.device,
        k=k,
        seen_items=seen,
        routing_seed=ROUTING_SEED,
    )
    return {
        int(user): [
            {"item": item, "score": None, "rank": rank + 1}
            for rank, item in enumerate(items)
        ]
        for user, items in recommendations.items()
    }


# --------------------------------------------------------------------------- 融合


def round_robin_fusion(
    two_tower: dict[int, list[dict]], mind: dict[int, list[dict]], limit: int
) -> dict[int, list[dict]]:
    """按两路原始名次轮流取、去重，最多 limit 个；记录每路的原始名次。"""
    fused: dict[int, list[dict]] = {}
    for user in sorted(two_tower):
        first, second = two_tower[user], mind.get(user, [])
        pool: list[dict] = []
        seen_items: set[int] = set()
        rank = 0
        while len(pool) < limit and (rank < len(first) or rank < len(second)):
            for channel, rows in (("two_tower", first), ("mind", second)):
                if len(pool) >= limit:
                    break
                if rank >= len(rows):
                    continue
                row = rows[rank]
                if row["item"] in seen_items:
                    continue
                seen_items.add(row["item"])
                pool.append(
                    {
                        "item": row["item"],
                        "fusion_rank": len(pool) + 1,
                        "sources": {channel: {"rank": row["rank"], "score": row["score"]}},
                    }
                )
            rank += 1
        fused[user] = pool
    return fused


# --------------------------------------------------------------------------- 排序训练


def _lookup(parent_ids: np.ndarray, original_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    """parent 索引 → 原始 ID → target 索引；不可映射返回 -1。"""
    original = original_ids[parent_ids]
    positions = np.searchsorted(target_ids, original)
    clipped = np.minimum(positions, max(target_ids.size - 1, 0))
    valid = (positions < target_ids.size) & (target_ids[clipped] == original)
    return np.where(valid, positions, -1).astype(np.int64)


def build_ranking_train(
    data,
    users: np.ndarray,
    max_rows: int,
    parent_ranking: Path,
) -> dict:
    """从父数据集 train 排序工件里取「子集用户 + 可映射物品」的完整请求。"""
    parent_users = np.load(parent_ranking.parent / "ids" / "user_ids.npy")
    parent_items = np.load(parent_ranking.parent / "ids" / "item_ids.npy")
    subset_users = np.asarray(data.mappings.index_to_user)
    subset_items = np.asarray(data.mappings.index_to_item)

    user_map = _lookup(np.arange(parent_users.size), parent_users, subset_users)
    item_map = _lookup(np.arange(parent_items.size), parent_items, subset_items)

    users_array = np.load(parent_ranking / "train_users.npy", mmap_mode="r")
    items_array = np.load(parent_ranking / "train_items.npy", mmap_mode="r")
    labels_array = np.load(parent_ranking / "train_labels.npy", mmap_mode="r")
    groups = np.load(parent_ranking / "train_groups.npy", mmap_mode="r")

    starts = np.concatenate(([0], np.cumsum(groups)))
    request_users = users_array[starts[:-1]]
    selected_mask = np.isin(user_map[request_users], users)  # 该请求的用户是否被选中
    row_user_keep = np.repeat(selected_mask, groups)
    row_item_map = item_map[items_array][row_user_keep]
    row_keep = row_user_keep.copy()
    row_keep[row_user_keep] = row_item_map >= 0

    labels = labels_array.astype(np.int64)
    row_indices = starts[:-1]
    positive = np.add.reduceat(labels * row_keep, row_indices)
    negative = np.add.reduceat((1 - labels) * row_keep, row_indices)
    counts = np.add.reduceat(row_keep.astype(np.int64), row_indices)

    keep_request = selected_mask & (positive > 0) & (negative > 0) & (counts > 0)
    cumulative = np.cumsum(np.where(keep_request, counts, 0))
    chosen_requests = np.flatnonzero(keep_request & (cumulative <= max_rows))
    if chosen_requests.size == 0:
        raise RuntimeError("没有满足「同时含正负标签」的请求，无法训练排序模型")

    selected_rows = np.zeros(len(users_array), dtype=bool)
    for request in chosen_requests:
        selected_rows[starts[request] : starts[request + 1]] = True
    rows = np.flatnonzero(selected_rows & row_keep)

    subset_item_idx = item_map[items_array[rows]]
    subset_user_idx = user_map[users_array[rows]]
    group_sizes = counts[chosen_requests].astype(np.int32)
    if int(group_sizes.sum()) != rows.size:
        raise AssertionError("请求分组长度与样本行数不一致")

    return {
        "rows": rows,
        "items": subset_item_idx,
        "users": subset_user_idx,
        "labels": labels_array[rows].astype(np.int8),
        "groups": group_sizes,
        "requests": int(chosen_requests.size),
        "parent_rows": int(len(users_array)),
    }


def build_features(data, item_indices: np.ndarray) -> dict[str, np.ndarray]:
    """LightGBM 的特征矩阵：只取 ItemFeatures 里已有的元数据列，列序固定。"""
    if data.item_features is None:
        raise RuntimeError("数据集没有 item_features，无法构建排序特征")
    return {
        name: np.asarray(data.item_features.column(name))[item_indices].astype(np.float32)
        for name in ITEM_FEATURE_COLUMNS
    }


# --------------------------------------------------------------------------- 重排


def category_rerank(
    ranked_items: list[int],
    ranked_scores: list[float],
    categories: np.ndarray,
    k: int = 20,
    cap: int = 5,
) -> tuple[list[int], int]:
    """首遍每个已知一级类目最多 cap 个；未知类目不占配额；不足按原序补齐。"""
    picked: list[int] = []
    used: dict[int, int] = {}
    for item in ranked_items:
        if len(picked) >= k:
            break
        category = int(categories[item])
        if category >= 0:
            if used.get(category, 0) >= cap:
                continue
            used[category] = used.get(category, 0) + 1
        picked.append(item)
    relaxed = 0
    if len(picked) < k:
        remaining = [item for item in ranked_items if item not in picked]
        for item in remaining:
            if len(picked) >= k:
                break
            picked.append(item)
            relaxed += 1
    return picked, relaxed


# --------------------------------------------------------------------------- 主流程


def metrics_for(
    recommendations: dict[int, list[int]], interactions, ks=(20,)
) -> dict[str, float]:
    metrics = evaluate_recommendations(recommendations, interactions, ks=ks)
    return {key: round(float(value), 6) for key, value in metrics.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="kuasar_small")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--users", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-recall", type=int, default=100)
    parser.add_argument("--k-final", type=int, default=20)
    parser.add_argument("--lgb-rounds", type=int, default=20)
    parser.add_argument("--lgb-max-rows", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--category-cap", type=int, default=5)
    args = parser.parse_args(argv)

    if args.dataset != "kuasar_small":
        raise SystemExit("本入口只允许 --dataset kuasar_small（避免误跑全量 kuasar）")
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    timings: dict[str, float] = {}

    log("加载 kuasar_small 与两份 checkpoint 元信息 …")
    data = prepare_dataset("kuasar_small", args.data_dir)
    parent_ranking = args.data_dir / "kuasar" / "processed" / "ranking"
    provenance = build_provenance(data, dataset="kuasar_small", positive_threshold=1.0)
    fingerprints = {
        "dataset": "kuasar_small",
        "users": int(data.num_users),
        "items": int(data.num_items),
        "split_strategy": data.split_strategy,
        "split_sha256": provenance["data"]["split_sha256"],
        "id_mapping_sha256": provenance["id_mapping"]["sha256"],
        "two_tower_checkpoint": {"path": str(TWO_TOWER_CHECKPOINT), "sha256": sha256_file(TWO_TOWER_CHECKPOINT)},
        "mind_checkpoint": {"path": str(MIND_CHECKPOINT), "sha256": sha256_file(MIND_CHECKPOINT)},
        "routing_seed": ROUTING_SEED,
        "k_recall": args.k_recall,
        "k_final": args.k_final,
    }

    users = select_users(data, args.users, args.seed)
    user_ids = [int(data.mappings.index_to_user[int(u)]) for u in users]
    json_dump(args.artifacts_dir / "users.json", {"seed": args.seed, "users": user_ids})
    log(f"选中 {len(users)} 个验证用户（原始 ID 已落盘）")

    stage = time.perf_counter()
    two_tower = recall_two_tower(data, users, args.k_recall)
    timings["recall_two_tower_s"] = round(time.perf_counter() - stage, 1)

    stage = time.perf_counter()
    mind = recall_mind(data, users, args.k_recall)
    timings["recall_mind_s"] = round(time.perf_counter() - stage, 1)

    json_dump(args.artifacts_dir / "recall_two_tower.json", {str(k): v for k, v in two_tower.items()})
    json_dump(args.artifacts_dir / "recall_mind.json", {str(k): v for k, v in mind.items()})

    stage = time.perf_counter()
    fused = round_robin_fusion(two_tower, mind, args.k_recall)
    timings["fusion_s"] = round(time.perf_counter() - stage, 2)
    json_dump(args.artifacts_dir / "fusion.json", {str(k): v for k, v in fused.items()})

    stage = time.perf_counter()
    train_data = build_ranking_train(data, users, args.lgb_max_rows, parent_ranking)
    timings["ranking_data_s"] = round(time.perf_counter() - stage, 1)

    train_features = build_features(data, train_data["items"])
    ranker = LightGBMRankerAdapter(
        params={"num_threads": args.threads, "num_leaves": 31, "learning_rate": 0.05, "seed": args.seed, "verbosity": -1}
    )
    stage = time.perf_counter()
    ranker.fit(
        train_features,
        train_data["labels"],
        groups=train_data["groups"],
        num_boost_round=args.lgb_rounds,
    )
    timings["lgb_fit_s"] = round(time.perf_counter() - stage, 1)
    model_path = args.artifacts_dir / "lgbm_ranker.txt"
    ranker.save(model_path)
    feature_names = list(ranker.feature_names)
    json_dump(
        args.artifacts_dir / "features.json",
        {
            "names": feature_names,
            "dtype": "float32",
            "missing_rule": "保留原始编码，缺失为 -1（类目/作者）或 0（计数/标记）",
        },
    )
    json_dump(
        args.artifacts_dir / "ranking_train_meta.json",
        {
            "requests": train_data["requests"],
            "rows": int(len(train_data["rows"])),
            "positives": int(train_data["labels"].sum()),
            "negatives": int((1 - train_data["labels"]).sum()),
            "groups": int(train_data["groups"].size),
            "parent_rows": train_data["parent_rows"],
            "num_boost_round": args.lgb_rounds,
        },
    )
    np.savez(
        args.artifacts_dir / "ranking_train_rows.npz",
        users=train_data["users"],
        items=train_data["items"],
        labels=train_data["labels"],
        groups=train_data["groups"],
    )

    stage = time.perf_counter()
    ranked: dict[int, list[dict]] = {}
    categories = np.asarray(data.item_features.column("category_first_code"))
    for user in sorted(fused):
        pool = fused[user]
        items = np.asarray([row["item"] for row in pool], dtype=np.int64)
        scores = ranker.predict(build_features(data, items))
        order = sorted(range(len(pool)), key=lambda i: (-scores[i], pool[i]["fusion_rank"]))
        ranked[user] = [
            {
                **pool[i],
                "lgbm_score": round(float(scores[i]), 6),
                "lgbm_rank": position + 1,
            }
            for position, i in enumerate(order)
        ]
    timings["lgbm_predict_s"] = round(time.perf_counter() - stage, 1)
    json_dump(args.artifacts_dir / "ranked.json", {str(k): v for k, v in ranked.items()})

    stage = time.perf_counter()
    final: dict[int, list[int]] = {}
    relaxations = 0
    for user in sorted(ranked):
        items = [row["item"] for row in ranked[user]]
        scores = [row["lgbm_score"] for row in ranked[user]]
        picked, relaxed = category_rerank(items, scores, categories, k=args.k_final, cap=args.category_cap)
        final[user] = picked
        relaxations += relaxed
    timings["rerank_s"] = round(time.perf_counter() - stage, 2)

    target_interactions = [row for row in data.validation_interactions if row.user_idx in set(users.tolist())]
    stages = {
        "two_tower_top20": {user: [row["item"] for row in two_tower[user][: args.k_final]] for user in users},
        "fusion_top20": {user: [row["item"] for row in fused[user][: args.k_final]] for user in users},
        "lgbm_top20": {user: [row["item"] for row in ranked[user][: args.k_final]] for user in users},
        "rerank_top20": final,
    }
    metrics = {
        name: {
            **metrics_for(recs, target_interactions, ks=(args.k_final,)),
            "Coverage": round(float(catalog_coverage(recs, data.num_items)), 6),
        }
        for name, recs in stages.items()
    }
    # 诊断：128 个用户是极小样本，Top-20 命中很可能为 0；同时记录 Top-100 候选命中，
    # 用来证明「召回链路确实捞到了目标」，而不是管线坏了。
    target_by_user = group_items_by_user(target_interactions)
    hits = {
        "two_tower@100": sum(len({r["item"] for r in two_tower[u]} & target_by_user.get(u, set())) for u in users),
        "mind@100": sum(len({r["item"] for r in mind[u]} & target_by_user.get(u, set())) for u in users),
        "union@100": sum(
            len(({r["item"] for r in two_tower[u]} | {r["item"] for r in mind[u]}) & target_by_user.get(u, set()))
            for u in users
        ),
        "fusion@20": sum(len({r["item"] for r in fused[u][: args.k_final]} & target_by_user.get(u, set())) for u in users),
        "rerank@20": sum(len(set(final[u]) & target_by_user.get(u, set())) for u in users),
    }
    diagnostics = {
        "users": int(len(users)),
        "targets": int(sum(len(v) for v in target_by_user.values())),
        "hits": {k: int(v) for k, v in hits.items()},
        "note": "128 用户以内验证样本，Top-20 命中为 0 属样本量限制；Top-100 命中用于核对链路",
    }
    json_dump(args.artifacts_dir / "metrics.json", {"stages": metrics, "diagnostics": diagnostics})

    # 断言：无重复、无曝光泄漏、候选来源可追溯、排序/重排都来自融合池
    seen = seen_items_for_split(data, "validation")
    assertions = []
    for user in users:
        pool_items = [row["item"] for row in fused[user]]
        assert len(pool_items) == len(set(pool_items)), f"user {user} 融合候选有重复"
        assert len(pool_items) <= args.k_recall
        assert all(item not in seen.get(int(user), set()) for item in pool_items), f"user {user} 融合含已曝光物品"
        ranked_items = [row["item"] for row in ranked[user]]
        assert all(item in pool_items for item in ranked_items), "排序结果超出融合池"
        assert set(final[user]) <= set(pool_items), "重排结果超出融合池"
        assert len(final[user]) <= args.k_final
        lock = [row for row in fused[user]]
        assert all("sources" in row and row["sources"] for row in lock), "融合候选缺少来源"
    assertions.append("无重复 / 无曝光泄漏 / 来源可追溯 / 排序与重排均来自融合池：通过")
    assert int(train_data["groups"].sum()) == len(train_data["labels"]), "分组长度与样本数不一致"
    assertions.append("排序训练分组长度与行数一致：通过")
    assert list(train_features) and all(name in train_features for name in feature_names)
    assertions.append("训练/推理特征列一致：通过")

    # 复验：重载模型，对缓存候选重新打分，核对与保存分数一致
    reloaded = LightGBMRankerAdapter.load(model_path)
    check_users = sorted(fused)[:5]
    max_delta = 0.0
    for user in check_users:
        items = np.asarray([row["item"] for row in fused[user]], dtype=np.int64)
        expected = np.asarray([row["lgbm_score"] for row in ranked[user]])
        replayed = reloaded.predict(build_features(data, items))
        order = sorted(range(len(items)), key=lambda i: (-replayed[i], fused[user][i]["fusion_rank"]))
        replayed_sorted = replayed[order]
        max_delta = max(max_delta, float(np.abs(replayed_sorted - expected).max()))
    assert max_delta < 1e-5, f"重载复验分数不一致：{max_delta}"
    assertions.append(f"LightGBM 重载复验通过（最大分数差 {max_delta:.2e}）")

    samples = []
    for user in sorted(fused)[:3]:
        raw_user = int(data.mappings.index_to_user[int(user)])
        samples.append(
            {
                "user": raw_user,
                "items": [
                    {
                        "item": int(data.mappings.index_to_item[int(row["item"])]),
                        "sources": {k: v for k, v in row["sources"].items()},
                        "fusion_rank": row["fusion_rank"],
                        "lgbm_score": row["lgbm_score"],
                        "lgbm_rank": row["lgbm_rank"],
                        "final_rank": final[user].index(row["item"]) + 1 if row["item"] in final[user] else None,
                    }
                    for row in ranked[user][:5]
                ],
            }
        )
    json_dump(args.artifacts_dir / "samples.json", samples)
    json_dump(args.artifacts_dir / "final.json", {str(k): v for k, v in final.items()})

    elapsed = time.perf_counter() - started
    run_meta = {
        "budget": {
            "users": args.users,
            "k_recall": args.k_recall,
            "k_final": args.k_final,
            "lgb_rounds": args.lgb_rounds,
            "lgb_max_rows": args.lgb_max_rows,
            "threads": args.threads,
            "category_cap": args.category_cap,
        },
        "actual": {
            "users": int(len(users)),
            "rows": int(len(train_data["rows"])),
            "requests": int(train_data["requests"]),
            "positives": int(train_data["labels"].sum()),
            "negatives": int((1 - train_data["labels"]).sum()),
            "fusion_size_avg": round(float(np.mean([len(v) for v in fused.values()])), 2),
            "relaxations": relaxations,
        },
        "timings": {**timings, "total_s": round(elapsed, 1)},
        "peak_rss_mib": peak_rss_mib(),
        "fingerprints": fingerprints,
        "assertions": assertions,
        "metrics": {"stages": metrics, "diagnostics": diagnostics},
        "lightgbm": __import__("lightgbm").__version__,
    }
    json_dump(args.artifacts_dir / "run_meta.json", run_meta)
    log(f"完成：{json.dumps({k: v for k, v in timings.items()}, ensure_ascii=False)}")
    log(f"总耗时 {elapsed:.1f}s，峰值 RSS {run_meta['peak_rss_mib']} MiB")
    for name, values in metrics.items():
        log(f"{name}: " + json.dumps(values, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
