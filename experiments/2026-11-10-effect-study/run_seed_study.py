"""多种子召回主表：Two-Tower 与 MIND 各训练 N 个种子，逐阶段评估。

每个种子独立训练、独立召回、独立融合、独立排序与重排；同一种子内所有阶段
共用同一份候选，因此阶段之间的差异是「同一批用户同一批候选」的成对差异。

用法
----
    python experiments/2026-11-10-effect-study/run_seed_study.py 42,43,44
    python experiments/2026-11-10-effect-study/run_seed_study.py 42,43,44,45,46
"""

from __future__ import annotations

import gc
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_ranker_recall_features as rrf  # noqa: E402
from common import (  # noqa: E402
    REPO,
    ROOT,
    load_data,
    load_fusion,
    per_user_metrics,
    summarize,
    targets_by_user,
)

from recsys.adapters.registry import load_adapter  # noqa: E402
from recsys.data.dataset import build_observed_histories  # noqa: E402
from recsys.data.schema import seen_items_for_split  # noqa: E402
from recsys.models.mind import load_mind_from_checkpoint, rank_users  # noqa: E402
from recsys.pipeline.config import FusionConfig  # noqa: E402
from recsys.pipeline.fusion import fuse_channels  # noqa: E402
from recsys.pipeline.recall import Candidate, run_channel  # noqa: E402
from recsys.training.train_mind import MindTrainConfig  # noqa: E402
from recsys.training.train_mind import train as train_mind  # noqa: E402
from recsys.training.trainer import TrainConfig  # noqa: E402
from recsys.training.trainer import train as train_two_tower  # noqa: E402

ARTIFACTS = REPO / "artifacts" / "effect_study"
OUT = ROOT / "seed_study.json"
CAND_DIR = ROOT / "seed_candidates"
VALIDATION_USERS = 512  # 仅影响每轮验证的评测用户数，不改变训练数据与指标口径
EPOCHS = 20


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def train_models(seed: int) -> tuple[Path, Path]:
    two_tower_path = ARTIFACTS / "kuasar_small" / f"two_tower_seed{seed}.pt"
    mind_path = ARTIFACTS / "kuasar_small" / f"mind_seed{seed}.pt"
    if not two_tower_path.exists():
        log(f"seed={seed} 训练 Two-Tower（{EPOCHS} epoch）…")
        begin = time.perf_counter()
        train_two_tower(
            TrainConfig(
                dataset="kuasar_small",
                data_dir=REPO / "data",
                artifacts_dir=ARTIFACTS,
                epochs=EPOCHS,
                batch_size=2048,
                learning_rate=0.01,
                embedding_dim=64,
                hidden_dim=128,
                output_dim=64,
                tower_type="embedding",
                mask_false_negatives=False,
                use_item_features=True,
                feature_dim=16,
                checkpoint_name=f"two_tower_seed{seed}",
                seed=seed,
                validation_users=VALIDATION_USERS,
            )
        )
        log(f"seed={seed} Two-Tower 完成（{time.perf_counter() - begin:.0f}s）")
    if not mind_path.exists():
        log(f"seed={seed} 训练 MIND（{EPOCHS} epoch）…")
        begin = time.perf_counter()
        train_mind(
            MindTrainConfig(
                dataset="kuasar_small",
                data_dir=REPO / "data",
                artifacts_dir=ARTIFACTS,
                epochs=EPOCHS,
                batch_size=1024,
                learning_rate=1e-2,
                embed_dim=64,
                max_length=20,
                interest_num=4,
                temperature=0.1,
                checkpoint_name=f"mind_seed{seed}",
                seed=seed,
                validation_users=VALIDATION_USERS,
            )
        )
        log(f"seed={seed} MIND 完成（{time.perf_counter() - begin:.0f}s）")
    return two_tower_path, mind_path


def mind_topk(checkpoint: Path, data, users: list[int], k: int = 100):
    """与 ``_mind_channel`` 相同的打分（max over K interests），但按批算分不落全矩阵。"""
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = load_mind_from_checkpoint(blob, torch.device("cpu"))
    histories = build_observed_histories(data, "test", True)
    seen = seen_items_for_split(data, "test")
    recommendations = rank_users(
        model,
        users,
        histories,
        data.num_items,
        max_length=int(blob["model_config"]["max_length"]),
        device=torch.device("cpu"),
        k=k,
        seen_items=seen,
        user_batch_size=64,
    )
    return {
        int(user): [
            Candidate(int(item), None, rank + 1)
            for rank, item in enumerate(recommendations[int(user)])
        ]
        for user in users
    }


def itemcf_channel(data, users: list[int], k: int = 100):
    """与数据规模实验同一实现：训练点击的余弦共现，按用户历史求和。"""
    counts = np.bincount(
        np.fromiter((row.item_idx for row in data.train_interactions), dtype=np.int64,
                    count=len(data.train_interactions)),
        minlength=data.num_items,
    )
    us = np.fromiter((row.user_idx for row in data.train_interactions), dtype=np.int64,
                     count=len(data.train_interactions))
    it = np.fromiter((row.item_idx for row in data.train_interactions), dtype=np.int64,
                     count=len(data.train_interactions))
    inv = np.zeros(data.num_items, dtype=np.float32)
    np.divide(1, np.sqrt(counts), where=counts > 0, out=inv)
    z = csr_matrix((inv[it], (us, it)), shape=(data.num_users, data.num_items), dtype=np.float32)
    gram = (z @ z.T).tocsr()
    gram.setdiag(0)
    gram.eliminate_zeros()
    seen = seen_items_for_split(data, "test")
    output = {}
    for start in range(0, len(users), 16):
        block = users[start:start + 16]
        scores = (gram[block] @ z).tocsr()
        for row, user in enumerate(block):
            begin, end = scores.indptr[row:row + 2]
            ids = scores.indices[begin:end]
            values = scores.data[begin:end]
            keep = ~np.isin(ids, np.fromiter(seen.get(user, set()), dtype=np.int32))
            ids, values = ids[keep], values[keep]
            order = np.lexsort((ids, -values))[:k]
            output[int(user)] = [
                Candidate(int(item), float(score), rank + 1)
                for rank, (item, score) in enumerate(zip(ids[order], values[order]))
            ]
    return output


def popularity_channel(data, users: list[int], k: int = 100):
    counts = np.zeros(data.num_items, dtype=np.int64)
    for row in data.train_interactions:
        counts[row.item_idx] += 1
    seen = seen_items_for_split(data, "test")
    eligible = np.flatnonzero(counts)
    top = eligible[np.argsort(-counts[eligible], kind="stable")]
    output = {}
    for user in users:
        blocked = seen.get(user, set())
        picked = [int(item) for item in top if int(item) not in blocked][:k]
        output[int(user)] = [
            Candidate(item, float(counts[item]), rank + 1) for rank, item in enumerate(picked)
        ]
    return output


def measure(label: str, candidates: dict, ground_truth: dict, users: list[int]) -> dict:
    metrics = per_user_metrics(candidates, ground_truth, ks=(20, 100))
    hits20 = sum(len(set(candidates[u][:20]) & ground_truth[u]) for u in users)
    hits100 = sum(len(set(candidates[u][:100]) & ground_truth[u]) for u in users)
    return {
        "stage": label,
        "metrics": {key: round(float(value.mean()), 6) for key, value in metrics.items()},
        "hit_targets_at20": hits20,
        "hit_targets_at100": hits100,
        "per_user": {key: value for key, value in metrics.items()},
    }


def strip(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k != "per_user"}


def main() -> int:
    positional = [value for value in sys.argv[1:] if not value.startswith("--")]
    seeds = [int(value) for value in (positional[0].split(",") if positional else ["42"])]
    eval_only = "--eval-only" in sys.argv
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = load_data()
    ground_truth = targets_by_user(data, "test")
    users = sorted(ground_truth)
    log(f"测试用户 {len(users)} / 目标 {sum(len(v) for v in ground_truth.values())}")

    # 断点续跑：已有结果按种子合并，确定性基线不重算（约 170 秒）。
    previous = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    per_seed: dict[str, dict] = dict(previous.get("per_seed") or {})
    deterministic: dict[str, dict] = dict(previous.get("deterministic") or {})

    # 确定性基线只需算一次
    for name, channel in (("popularity", popularity_channel), ("itemcf", itemcf_channel)):
        if name in deterministic:
            log(f"{name}: 复用已有结果 Recall@20={deterministic[name]['metrics']['Recall@20']:.6f}")
            continue
        begin = time.perf_counter()
        candidates = channel(data, users)
        deterministic[name] = measure(name, {u: [c.item for c in rows] for u, rows in candidates.items()},
                                      ground_truth, users)
        log(f"{name}: Recall@20={deterministic[name]['metrics']['Recall@20']:.6f} "
            f"Recall@100={deterministic[name]['metrics']['Recall@100']:.6f} "
            f"({time.perf_counter() - begin:.0f}s)")

    ranker_path = ROOT / "ranker_R5.txt"
    if not ranker_path.exists():
        raise SystemExit("缺少 ranker_R5.txt：先跑 run_ranker_recall_features.py")
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(ranker_path))
    priors = rrf.InteractionPriors(data)
    earliest = {}
    for row in data.test_interactions:
        user, timestamp = int(row.user_idx), int(row.timestamp)
        if user not in earliest or timestamp < earliest[user]:
            earliest[user] = timestamp

    for seed in seeds:
        if str(seed) in per_seed:
            log(f"seed={seed} 已有结果，跳过")
            continue
        begin = time.perf_counter()
        two_tower_path = ARTIFACTS / "kuasar_small" / f"two_tower_seed{seed}.pt"
        mind_path = ARTIFACTS / "kuasar_small" / f"mind_seed{seed}.pt"
        if not eval_only:
            train_models(seed)

        # 召回
        tower_adapter = load_adapter("two_tower", two_tower_path, device_name="cpu")
        two_tower_candidates = run_channel(
            tower_adapter, data, users, split="test", k=100,
            include_validation_in_test_history=True,
        )
        mind_candidates = mind_topk(mind_path, data, users, k=100)
        del tower_adapter
        gc.collect()

        channels = {"two_tower": two_tower_candidates, "mind": mind_candidates}
        stages: dict[str, dict] = {}
        for name, mapping in channels.items():
            stages[name] = measure(
                name, {u: [c.item for c in mapping[u]] for u in users}, ground_truth, users
            )

        fused = fuse_channels(channels, FusionConfig(mode="round_robin", limit=100))
        fused_items = {u: [c.item for c in fused[u]] for u in users}
        stages["fusion"] = measure("fusion", fused_items, ground_truth, users)
        union_hits = sum(
            len({c.item for mapping in channels.values() for c in mapping[u]} & ground_truth[u])
            for u in users
        )

        # 排序：R5（物品特征 + 交互先验 + 两路召回分数）
        pool = {u: [{"item": c.item, "fusion_rank": c.fusion_rank} for c in fused[u]] for u in users}
        test_items = {u: np.asarray([row["item"] for row in pool[u]], dtype=np.int64) for u in users}
        two_tower_scorer = rrf.TwoTowerScorer(two_tower_path)
        blob = torch.load(mind_path, map_location="cpu", weights_only=False)
        mind_model = load_mind_from_checkpoint(blob, torch.device("cpu"))
        from recsys.models.mind import encode_item_matrix

        mind_items = encode_item_matrix(mind_model, data.num_items, torch.device("cpu")).astype(np.float32)
        test_histories = build_observed_histories(data, "test", True)
        keys, vectors = rrf.mind_interest_vectors(
            mind_model, {(u, earliest[u]): test_histories.get(u, []) for u in users},
            torch.device("cpu"), max_length=int(blob["model_config"]["max_length"]),
        )
        position = {key[0]: index for index, key in enumerate(keys)}
        base = rrf.item_features(data, np.zeros(1, dtype=np.int64))  # 仅取列名
        scores = {}
        for user in users:
            items = test_items[user]
            block = {
                **{name: np.asarray(data.item_features.column(name))[items].astype(np.float32)
                   for name in base},
                **priors.build(
                    np.full(items.size, user, dtype=np.int64), items,
                    np.full(items.size, earliest[user], dtype=np.int64),
                ),
                "two_tower_score": two_tower_scorer.score(np.full(items.size, user, dtype=np.int64), items),
                "mind_score": np.max(vectors[position[user]] @ mind_items[items].T, axis=0).astype(np.float32),
            }
            names = sorted(block)
            scores[user] = np.asarray(booster.predict(np.stack([block[n] for n in names], axis=1)),
                                      dtype=np.float64)
        ordered = rrf.order_by_scores(users, pool, scores)
        stages["ranked"] = measure("ranked", ordered, ground_truth, users)
        stages["final"] = measure("final", rrf.rerank(data, ordered), ground_truth, users)

        del two_tower_scorer, mind_model, mind_items, vectors
        gc.collect()

        # 逐用户指标与候选都落盘：报告里的成对区间和以后的重分析都基于它们。
        (ROOT / "peruser").mkdir(exist_ok=True)
        np.savez_compressed(
            ROOT / "peruser" / f"seed{seed}.npz",
            **{f"{stage}__{metric}": value for stage, payload in stages.items()
               for metric, value in payload["per_user"].items()},
        )
        with gzip.open(CAND_DIR / f"seed{seed}.json.gz", "wt", encoding="utf-8") as handle:
            json.dump({
                "two_tower": {u: [c.item for c in rows] for u, rows in channels["two_tower"].items()},
                "mind": {u: [c.item for c in rows] for u, rows in channels["mind"].items()},
                "fusion": fused_items,
                "ranked": ordered,
                "final": rrf.rerank(data, ordered),
            }, handle, separators=(",", ":"))

        per_seed[str(seed)] = {
            "stages": {name: strip(value) for name, value in stages.items()},
            "union_hits_at_100": int(union_hits),
            "checkpoints": {"two_tower": str(two_tower_path), "mind": str(mind_path)},
            "seconds": round(time.perf_counter() - begin, 1),
            "metrics": {name: value["metrics"] for name, value in stages.items()},
        }
        log(f"seed={seed} 完成：fusion Recall@20={stages['fusion']['metrics']['Recall@20']:.6f} "
            f"final Recall@20={stages['final']['metrics']['Recall@20']:.6f} "
            f"（{per_seed[str(seed)]['seconds']:.0f}s）")
        OUT.write_text(json.dumps({
            "seeds": seeds,
            "deterministic": {name: strip(value) for name, value in deterministic.items()},
            "per_seed": per_seed,
            "elapsed_s": round(time.perf_counter() - started, 1),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- 聚合
    all_seeds = sorted(per_seed, key=int)
    log(f"聚合种子：{', '.join(all_seeds)}")
    stage_names = ["two_tower", "mind", "fusion", "ranked", "final"]
    aggregate = {}
    for name in stage_names:
        values = [per_seed[str(seed)]["metrics"][name] for seed in all_seeds]
        aggregate[name] = {}
        for metric in values[0]:
            array = np.asarray([row[metric] for row in values], dtype=np.float64)
            aggregate[name][metric] = {
                "mean": round(float(array.mean()), 6),
                "std": round(float(array.std(ddof=1)), 6) if array.size > 1 else 0.0,
                "per_seed": [round(float(value), 6) for value in array],
                "spread_ratio": round(float(array.std(ddof=1) / array.mean()), 4)
                if array.size > 1 and array.mean() else None,
            }

    payload = json.loads(OUT.read_text(encoding="utf-8"))
    payload["seeds"] = [int(seed) for seed in all_seeds]
    payload["aggregate"] = aggregate
    payload["note"] = (
        "跨种子 std 只反映训练随机性与推理顺序；用户抽样不确定性另见 paired_bootstrap。"
        "所有种子共用同一份测试目标与同一候选预算。"
    )
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"写出 {OUT}（{payload['elapsed_s']}s）")
    for name in stage_names:
        row = aggregate[name]["Recall@20"]
        log(f"{name:10s} Recall@20 {row['mean']:.6f} ± {row['std']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
