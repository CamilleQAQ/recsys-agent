"""排序器加「召回分数」特征：验证「排序器坏在看不到用户-物品相关性」这一归因。

对照
----
R0  融合顺序（参照）
R5  9 列物品特征 + 5 列交互先验 + Two-Tower 分数 + MIND 分数
R5b 9 列物品特征 + Two-Tower 分数 + MIND 分数（去掉交互先验，隔离加法来源）

召回分数一律按「该行请求时间戳之前的历史」计算，训练行与推理行同一实现：
* Two-Tower 用户塔只用用户 ID，分数与时间无关；
* MIND 用 ``histories_for_requests`` 构造严格早于请求的序列；推理侧用
  ``build_observed_histories``（训练 + 验证历史），与链路里 MIND 通道一致。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    BASELINE_DIR,
    RANKING_DIR,
    ROOT,
    REPO,
    InteractionPriors,
    assert_matches_official,
    load_data,
    load_fusion,
    paired_bootstrap,
    per_user_metrics,
    targets_by_user,
)

import lightgbm as lgb  # noqa: E402

from recsys.data.dataset import build_observed_histories, pad_item_sequence  # noqa: E402
from recsys.data.kuasar_ranking import ITEM_FEATURE_COLUMNS  # noqa: E402
from recsys.models.mind import (  # noqa: E402
    EVAL_ROUTING_SEED,
    deterministic_routing,
    encode_item_matrix,
    load_mind_from_checkpoint,
)
from recsys.models.two_tower import TwoTowerModel  # noqa: E402
from recsys.pipeline.rerank import diversify  # noqa: E402
from recsys.pipeline.ranking import RankedCandidate  # noqa: E402
from recsys.ranking.samples import (  # noqa: E402
    build_history_source,
    histories_for_requests,
    load_ranking_view,
)

OUT = ROOT / "ranker_recall_features.json"
TWO_TOWER_CKPT = REPO / "artifacts" / "kuasar_small" / "two_tower_features.pt"
MIND_CKPT = REPO / "artifacts" / "kuasar_small" / "mind20.pt"
NUM_THREADS = 8
CHUNK = 100_000


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def item_features(data, items: np.ndarray) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(data.item_features.column(name))[items].astype(np.float32)
        for name in ITEM_FEATURE_COLUMNS
    }


def order_by_scores(users, pool, scores):
    out = {}
    for user in users:
        rows = pool[user]
        order = np.lexsort(
            (np.asarray([row["fusion_rank"] for row in rows]), -scores[user])
        )
        out[user] = [rows[int(index)]["item"] for index in order]
    return out


def rerank(data, recommendations, k=20, cap=5):
    categories = np.asarray(data.item_features.column("category_first_code"))
    out = {}
    for user, items in recommendations.items():
        candidates = [
            RankedCandidate(item=item, score=0.0, rank=index + 1)
            for index, item in enumerate(items)
        ]
        picked, _ = diversify(candidates, categories=categories, authors=None, k=k, category_cap=cap)
        out[user] = [candidate.item for candidate in picked]
    return out


class TwoTowerScorer:
    """只加载一次，缓存两侧 embedding；分数 = 归一化向量的点积。"""

    def __init__(self, checkpoint: Path) -> None:
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = TwoTowerModel(**blob["model_config"]).eval()
        self.model.load_state_dict(blob["model_state_dict"])
        with torch.inference_mode():
            self.item_embeddings = self.model.encode_items(
                torch.arange(self.model.item_tower.embedding.num_embeddings)
            ).numpy()
            self.user_embeddings = self.model.encode_users(
                torch.arange(self.model.user_tower.embedding.num_embeddings)
            ).numpy()

    def score(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        out = np.empty(items.size, dtype=np.float32)
        for start in range(0, items.size, CHUNK):
            end = min(start + CHUNK, items.size)
            out[start:end] = np.sum(
                self.user_embeddings[users[start:end]] * self.item_embeddings[items[start:end]],
                axis=1,
            )
        return out


def _pad_history(tokens, max_length: int, already_tokenized: bool) -> np.ndarray:
    """``histories_for_requests`` 返回的序列已经是 1-based token（含 0 补齐），
    不能再走 ``pad_item_sequence``（它会再 +1）。"""
    if already_tokenized:
        sequence = np.asarray(tokens[-max_length:], dtype=np.int64)
        padded = np.zeros(max_length, dtype=np.int64)
        padded[-sequence.size:] = sequence
        return padded
    return np.asarray(pad_item_sequence(tokens, max_length), dtype=np.int64)


def mind_interest_vectors(
    model,
    histories: dict,
    device,
    *,
    max_length: int,
    batch: int = 128,
    already_tokenized: bool = False,
):
    """按固定路由顺序批量算每个 (user, 时间戳) 的 K 个兴趣向量。"""
    keys = sorted(histories)
    blocks = []
    model.eval()
    model.mode = "user"  # torch-rechub 的 user_tower 在 mode="item" 时直接返回 None
    with torch.inference_mode(), deterministic_routing(EVAL_ROUTING_SEED):
        for start in range(0, len(keys), batch):
            block = keys[start:start + batch]
            hist = np.stack([
                _pad_history(histories[key], max_length, already_tokenized) for key in block
            ])
            inputs = {
                "user_id": torch.as_tensor([key[0] for key in block], dtype=torch.long, device=device) + 1,
                "hist_item": torch.as_tensor(hist, dtype=torch.long, device=device),
            }
            blocks.append(model.user_tower(inputs, interest_mask=None).float().cpu().numpy())
    return keys, np.concatenate(blocks, axis=0)


def mind_scores(vectors, item_embeddings, items, row_index) -> np.ndarray:
    """max over K interests of dot(interest, item_embedding)，分块避免大矩阵。"""
    out = np.empty(items.size, dtype=np.float32)
    for start in range(0, items.size, CHUNK):
        end = min(start + CHUNK, items.size)
        interests = vectors[row_index[start:end]]          # [n, K, D]
        target = item_embeddings[items[start:end]]         # [n, D]
        out[start:end] = np.max(np.einsum("nkd,nd->nk", interests, target), axis=1)
    return out


def main() -> int:
    started = time.perf_counter()
    device = torch.device("cpu")
    data = load_data()
    ground_truth = targets_by_user(data, "test")
    pool = load_fusion()
    users = sorted(pool)
    official = json.loads((BASELINE_DIR / "result.json").read_text(encoding="utf-8"))["metrics"]

    train_view = load_ranking_view(data, RANKING_DIR, split="train", max_rows=10**9)
    valid_view = load_ranking_view(data, RANKING_DIR, split="validation", max_rows=10**9)
    log(f"训练 {train_view.requests:,} 请求 / {len(train_view):,} 行；验证 {valid_view.requests:,} 请求")

    # ---------------------------------------------------------------- 召回分数
    log("计算 Two-Tower 分数 …")
    tower = TwoTowerScorer(TWO_TOWER_CKPT)
    tt_train = tower.score(train_view.users, train_view.items)
    tt_valid = tower.score(valid_view.users, valid_view.items)

    log("计算 MIND 分数 …")
    mind_blob = torch.load(MIND_CKPT, map_location="cpu", weights_only=False)
    mind = load_mind_from_checkpoint(mind_blob, device).eval()
    mind_max_length = int(mind_blob["model_config"]["max_length"])
    mind_item_embeddings = encode_item_matrix(mind, data.num_items, device).astype(np.float32)

    train_pairs = sorted(set(zip(train_view.users.tolist(), train_view.timestamps.tolist())))
    source = build_history_source(data, "train")
    train_histories = histories_for_requests(source, train_pairs, max_history=mind_max_length)
    train_keys, train_vectors = mind_interest_vectors(
        mind, train_histories, device, max_length=mind_max_length, already_tokenized=True
    )
    train_index = {key: position for position, key in enumerate(train_keys)}
    train_rows = np.fromiter(
        (train_index[(int(u), int(t))] for u, t in zip(train_view.users, train_view.timestamps)),
        dtype=np.int64, count=len(train_view),
    )
    mind_train = mind_scores(train_vectors, mind_item_embeddings, train_view.items, train_rows)

    valid_pairs = sorted(set(zip(valid_view.users.tolist(), valid_view.timestamps.tolist())))
    valid_histories = histories_for_requests(source, valid_pairs, max_history=mind_max_length)
    valid_keys, valid_vectors = mind_interest_vectors(
        mind, valid_histories, device, max_length=mind_max_length, already_tokenized=True
    )
    valid_index = {key: position for position, key in enumerate(valid_keys)}
    valid_rows = np.fromiter(
        (valid_index[(int(u), int(t))] for u, t in zip(valid_view.users, valid_view.timestamps)),
        dtype=np.int64, count=len(valid_view),
    )
    mind_valid = mind_scores(valid_vectors, mind_item_embeddings, valid_view.items, valid_rows)
    log(f"训练侧 MIND 分数完成（{len(train_pairs):,} 个 (user, 时间戳) 序列，"
        f"单序列 {train_vectors.shape[1]} 个兴趣 × {train_vectors.shape[2]} 维）")

    # ---------------------------------------------------------------- 推理侧分数
    test_items = {user: np.asarray([row["item"] for row in pool[user]], dtype=np.int64) for user in users}
    earliest = {}
    for row in data.test_interactions:
        user, timestamp = int(row.user_idx), int(row.timestamp)
        if user not in earliest or timestamp < earliest[user]:
            earliest[user] = timestamp
    test_histories = build_observed_histories(data, "test", True)
    test_keys, test_vectors = mind_interest_vectors(
        mind, {(user, earliest[user]): test_histories.get(user, []) for user in users},
        device, max_length=mind_max_length,
    )
    test_index = {key[0]: position for position, key in enumerate(test_keys)}
    tt_test, mind_test = {}, {}
    for user in users:
        items = test_items[user]
        tt_test[user] = tower.score(np.full(items.size, user, dtype=np.int64), items)
        position = test_index[user]
        mind_test[user] = np.max(
            test_vectors[position] @ mind_item_embeddings[items].T, axis=0
        ).astype(np.float32)
    log("推理侧召回分数完成")

    priors = InteractionPriors(data)
    extra_train = priors.build(train_view.users, train_view.items, train_view.timestamps)
    extra_valid = priors.build(valid_view.users, valid_view.items, valid_view.timestamps)
    extra_test = {
        user: priors.build(
            np.full(test_items[user].size, user, dtype=np.int64),
            test_items[user],
            np.full(test_items[user].size, earliest[user], dtype=np.int64),
        )
        for user in users
    }

    base_train = item_features(data, train_view.items)
    base_valid = item_features(data, valid_view.items)
    base_test = {user: item_features(data, test_items[user]) for user in users}
    recall_train = {"two_tower_score": tt_train, "mind_score": mind_train}
    recall_valid = {"two_tower_score": tt_valid, "mind_score": mind_valid}
    recall_test = {
        user: {"two_tower_score": tt_test[user], "mind_score": mind_test[user]} for user in users
    }

    def fit(features_train, features_valid=None, rounds=600):
        """早停用「训练请求内 5% 随机留出」，而不是父表 validation。

        父表 validation 只有 1,087 个请求、平均 4.3 个候选，NDCG 噪声大到
        best_iteration 会停在 1–4 轮；同分布留出既大得多也稳定。
        """
        names = sorted(features_train)
        groups = train_view.groups
        starts = np.concatenate(([0], np.cumsum(groups)))
        request_count = groups.size
        rng = np.random.default_rng(20261110)
        hold_requests = np.zeros(request_count, dtype=bool)
        hold_requests[rng.choice(request_count, size=max(1, request_count // 20), replace=False)] = True
        row_holdout = np.repeat(hold_requests, groups)
        train_rows = np.flatnonzero(~row_holdout)
        holdout_rows = np.flatnonzero(row_holdout)
        params = {
            "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [20],
            "boosting_type": "gbdt", "num_leaves": 63, "learning_rate": 0.05,
            "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1,
            "min_data_in_leaf": 50, "verbosity": -1, "num_threads": NUM_THREADS,
        }
        train_set = lgb.Dataset(
            np.stack([features_train[n][train_rows] for n in names], axis=1),
            label=train_view.labels[train_rows],
            group=groups[~hold_requests].tolist(),
            feature_name=names, free_raw_data=False,
        )
        valid_set = lgb.Dataset(
            np.stack([features_train[n][holdout_rows] for n in names], axis=1),
            label=train_view.labels[holdout_rows],
            group=groups[hold_requests].tolist(),
            feature_name=names, reference=train_set,
        )
        booster = lgb.train(params, train_set, num_boost_round=rounds,
                            valid_sets=[valid_set], valid_names=["holdout"],
                            callbacks=[lgb.early_stopping(50, verbose=False)])
        importance = {
            name: int(value) for name, value in zip(
                booster.feature_name(), booster.feature_importance(importance_type="gain")
            )
        }
        return booster, importance

    def predict(booster, features_by_user):
        scores = {}
        for user in users:
            block = features_by_user[user]
            names = sorted(block)
            matrix = np.stack([block[name] for name in names], axis=1)
            scores[user] = np.asarray(booster.predict(matrix), dtype=np.float64)
        return scores

    def evaluate(label, recommendations, official_metrics=None):
        metrics = per_user_metrics(recommendations, ground_truth, ks=(20, 100))
        if official_metrics is not None:
            assert_matches_official(metrics, official_metrics)
        hits = sum(len(set(recommendations[u][:20]) & ground_truth[u]) for u in users)
        log(f"{label}: Recall@20={metrics['Recall@20'].mean():.6f} "
            f"NDCG@20={metrics['NDCG@20'].mean():.6f} hits@20={hits}")
        return {
            "arm": label,
            "metrics": {key: round(float(value.mean()), 6) for key, value in metrics.items()},
            "hit_targets_at20": hits,
            "_per_user": metrics,
        }

    fusion_recs = {user: [row["item"] for row in pool[user]] for user in users}
    results = {"R0": evaluate("R0_fusion_order", fusion_recs, official["fusion"])}

    variants = {
        "R5": (
            {**base_train, **extra_train, **recall_train},
            {**base_valid, **extra_valid, **recall_valid},
            {user: {**base_test[user], **extra_test[user], **recall_test[user]} for user in users},
        ),
        "R5b": (
            {**base_train, **recall_train},
            {**base_valid, **recall_valid},
            {user: {**base_test[user], **recall_test[user]} for user in users},
        ),
    }
    for name, (fit_train, fit_valid, test_block) in variants.items():
        log(f"训练 {name}（{len(fit_train)} 列特征）…")
        booster, importance = fit(fit_train, fit_valid)
        log(f"{name}: best_iteration={booster.best_iteration}")
        ordered = order_by_scores(users, pool, predict(booster, test_block))
        results[name] = evaluate(f"{name}_retrained", ordered)
        results[f"{name}_rerank"] = evaluate(f"{name}_retrained+rerank", rerank(data, ordered))
        results[name]["importance"] = importance
        results[name]["best_iteration"] = int(booster.best_iteration)
        booster.save_model(str(ROOT / f"ranker_{name}.txt"))

    baseline = results["R0"]["_per_user"]
    comparisons = {
        key: {
            metric: paired_bootstrap(results[key]["_per_user"][metric], baseline[metric])
            for metric in ("Recall@20", "NDCG@20")
        }
        for key in ("R5", "R5b")
    }
    output = {
        "protocol": {
            "dataset": "kuasar_small",
            "split": "test (2000 users / 4275 targets)",
            "candidate_pool": str(BASELINE_DIR / "fusion.json"),
            "features": {
                "two_tower_score": "dot(user_emb, item_emb)，用户塔只用 ID，与时间无关",
                "mind_score": "max over K interests of dot(interest, item_emb)，历史严格早于请求",
            },
            "leakage_note": "排序训练行来自父表 train 请求；召回分数用该请求时间戳之前的历史，"
                            "与推理侧同一构造；两个召回模型本身是用 train 切分训练的。",
        },
        "arms": {key: {k: v for k, v in value.items() if not k.startswith("_per_user")}
                 for key, value in results.items()},
        "comparisons_vs_R0": comparisons,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"写出 {OUT}（{output['elapsed_s']}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
