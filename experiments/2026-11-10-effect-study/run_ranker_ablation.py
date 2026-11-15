"""排序器 2×2（+ 两个补充组）消融：同一批缓存融合候选，只换排序器。

组别
----
R0  融合顺序（不排序，作为上界/底线）
R1  现行排序器：artifacts/kuasar_small/mvp/lgbm_ranker.txt（9,997 行 / 20 轮）
R2  重训·同特征：父表 train 全部子集用户请求（约 156 万行），9 列物品特征
R3  重训·加交互先验：9 列 + 5 列「严格早于请求」的交互特征
R4  融合顺序 + R3 分数按名次百分位等权混合（不调参，w=0.5 先验固定）

用法
----
    python experiments/2026-11-10-effect-study/run_ranker_ablation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    BASELINE_DIR,
    RANKING_DIR,
    ROOT,
    InteractionPriors,
    assert_matches_official,
    load_data,
    load_fusion,
    paired_bootstrap,
    per_user_metrics,
    summarize,
    targets_by_user,
)

import lightgbm as lgb  # noqa: E402

from recsys.adapters.ranking import LightGBMRankerAdapter  # noqa: E402
from recsys.data.kuasar_ranking import ITEM_FEATURE_COLUMNS  # noqa: E402
from recsys.pipeline.rerank import diversify  # noqa: E402
from recsys.pipeline.ranking import RankedCandidate  # noqa: E402
from recsys.ranking.samples import load_ranking_view  # noqa: E402

OUT = ROOT / "ranker_ablation.json"
BASE_RANKER = ROOT.parents[1] / "artifacts" / "kuasar_small" / "mvp" / "lgbm_ranker.txt"
MAX_ROUNDS = 600
EARLY_STOP = 50
NUM_THREADS = 8


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def item_features(data, items: np.ndarray) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(data.item_features.column(name))[items].astype(np.float32)
        for name in ITEM_FEATURE_COLUMNS
    }


def order_by_scores(
    users: list[int],
    pool: dict[int, list[dict]],
    scores: dict[int, np.ndarray],
) -> dict[int, list[int]]:
    """按分数降序、融合名次升序打破并列；返回 item 序列。"""
    out: dict[int, list[int]] = {}
    for user in users:
        rows = pool[user]
        score = scores[user]
        order = np.lexsort(
            (np.asarray([row["fusion_rank"] for row in rows]), -score)
        )
        out[user] = [rows[int(index)]["item"] for index in order]
    return out


def rerank(data, recommendations: dict[int, list[int]], k: int = 20, cap: int = 5):
    categories = np.asarray(data.item_features.column("category_first_code"))
    out: dict[int, list[int]] = {}
    for user, items in recommendations.items():
        candidates = [
            RankedCandidate(item=item, score=0.0, rank=index + 1)
            for index, item in enumerate(items)
        ]
        picked, _ = diversify(
            candidates, categories=categories, authors=None, k=k, category_cap=cap
        )
        out[user] = [candidate.item for candidate in picked]
    return out


def evaluate(
    label: str,
    recommendations: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    official: dict[str, float] | None = None,
) -> dict:
    metrics = per_user_metrics(recommendations, ground_truth, ks=(20, 100))
    if official is not None:
        assert_matches_official(metrics, official)
    payload = {
        "arm": label,
        "users": len(recommendations),
        "metrics": {
            key: round(float(values.mean()), 6) for key, values in metrics.items()
        },
        "metric_summary": {key: {k: round(v, 6) for k, v in summarize(values).items()}
                           for key, values in metrics.items()},
    }
    hit20 = sum(len(set(recommendations[u][:20]) & ground_truth[u]) for u in recommendations)
    hit100 = sum(len(set(recommendations[u][:100]) & ground_truth[u]) for u in recommendations)
    payload["hit_targets"] = {"at20": hit20, "at100": hit100}
    payload["_per_user"] = metrics
    log(f"{label}: Recall@20={payload['metrics']['Recall@20']:.6f} "
        f"NDCG@20={payload['metrics']['NDCG@20']:.6f} "
        f"Recall@100={payload['metrics']['Recall@100']:.6f} hits@{20}={hit20}")
    return payload


def main() -> int:
    started = time.perf_counter()
    data = load_data()
    ground_truth = targets_by_user(data, "test")
    pool = load_fusion()
    users = sorted(pool)
    if set(users) != set(ground_truth):
        raise AssertionError("融合候选用户集合与测试目标用户集合不一致")
    log(f"测试用户 {len(users)} / 目标 {sum(len(v) for v in ground_truth.values())} / "
        f"候选池 {sum(len(v) for v in pool.values())}")

    official = json.loads((BASELINE_DIR / "result.json").read_text(encoding="utf-8"))["metrics"]
    results: dict[str, dict] = {}

    # ---------------------------------------------------------------- R0 融合顺序
    r0 = {user: [row["item"] for row in pool[user]] for user in users}
    results["R0"] = evaluate("R0_fusion_order", r0, ground_truth, official["fusion"])
    results["R0_rerank"] = evaluate("R0_fusion_order+rerank", rerank(data, r0), ground_truth)

    # ---------------------------------------------------------------- R1 现行排序器
    legacy = LightGBMRankerAdapter.load(BASE_RANKER)
    legacy_scores: dict[int, np.ndarray] = {}
    for user in users:
        items = np.asarray([row["item"] for row in pool[user]], dtype=np.int64)
        legacy_scores[user] = legacy.predict(item_features(data, items))
    r1 = order_by_scores(users, pool, legacy_scores)
    results["R1"] = evaluate("R1_legacy_ranker", r1, ground_truth, official["ranked"])
    results["R1_rerank"] = evaluate(
        "R1_legacy_ranker+rerank", rerank(data, r1), ground_truth, official["rerank"]
    )

    # ---------------------------------------------------------------- 训练视图
    log("加载父表 train 全部子集用户请求（这是 R2/R3 的唯一改动：数据量与时间覆盖）")
    train_view = load_ranking_view(data, RANKING_DIR, split="train", max_rows=10**9)
    log(f"训练视图：{train_view.requests:,} 请求 / {len(train_view):,} 行 / "
        f"{int(train_view.labels.sum()):,} 正例")
    valid_view = load_ranking_view(data, RANKING_DIR, split="validation", max_rows=10**9)
    log(f"验证视图：{valid_view.requests:,} 请求 / {len(valid_view):,} 行")

    priors = InteractionPriors(data)
    base_train = item_features(data, train_view.items)
    base_valid = item_features(data, valid_view.items)
    extra_train = priors.build(train_view.users, train_view.items, train_view.timestamps)
    extra_valid = priors.build(valid_view.users, valid_view.items, valid_view.timestamps)

    def train_ranker(name: str, train_features: dict, valid_features: dict, rounds: int):
        # 早停用训练请求内 5% 随机留出：父表 validation 只有 1,087 个请求，
        # 噪声大到 best_iteration 会停在个位数。
        groups = train_view.groups
        request_count = groups.size
        rng = np.random.default_rng(20261110)
        hold_requests = np.zeros(request_count, dtype=bool)
        hold_requests[rng.choice(request_count, size=max(1, request_count // 20), replace=False)] = True
        row_holdout = np.repeat(hold_requests, groups)
        train_rows = np.flatnonzero(~row_holdout)
        holdout_rows = np.flatnonzero(row_holdout)
        names = sorted(train_features)
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [20],
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "min_data_in_leaf": 50,
            "verbosity": -1,
            "num_threads": NUM_THREADS,
        }
        train_set = lgb.Dataset(
            np.stack([train_features[n][train_rows] for n in names], axis=1),
            label=train_view.labels[train_rows],
            group=groups[~hold_requests].tolist(),
            feature_name=names,
            free_raw_data=False,
        )
        valid_set = lgb.Dataset(
            np.stack([train_features[n][holdout_rows] for n in names], axis=1),
            label=train_view.labels[holdout_rows],
            group=groups[hold_requests].tolist(),
            feature_name=names,
            reference=train_set,
        )
        begin = time.perf_counter()
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=rounds,
            valid_sets=[valid_set],
            valid_names=["holdout"],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(100)],
        )
        elapsed = time.perf_counter() - begin
        log(f"{name}: best_iteration={booster.best_iteration} 训练 {elapsed:.1f}s")
        importance = {
            name_: int(value)
            for name_, value in zip(
                booster.feature_name(), booster.feature_importance(importance_type="gain")
            )
        }
        return booster, importance, elapsed

    def score_with(booster, features_by_user: dict[int, dict[str, np.ndarray]]):
        out: dict[int, np.ndarray] = {}
        for user in users:
            matrix = np.stack([features_by_user[user][n] for n in sorted(features_by_user[user])], axis=1)
            out[user] = np.asarray(booster.predict(matrix), dtype=np.float64)
        return out

    test_items = {user: np.asarray([row["item"] for row in pool[user]], dtype=np.int64) for user in users}
    earliest: dict[int, int] = {}
    for row in data.test_interactions:
        user = int(row.user_idx)
        timestamp = int(row.timestamp)
        if user not in earliest or timestamp < earliest[user]:
            earliest[user] = timestamp
    test_ts = {user: earliest[user] for user in users}
    test_base = {user: item_features(data, test_items[user]) for user in users}
    test_extra = {
        user: priors.build(
            np.full(test_items[user].size, user, dtype=np.int64),
            test_items[user],
            np.full(test_items[user].size, test_ts[user], dtype=np.int64),
        )
        for user in users
    }
    log("推理侧交互先验特征构建完成")

    # ---------------------------------------------------------------- R2 同特征重训
    booster2, importance2, time2 = train_ranker("R2", base_train, base_valid, MAX_ROUNDS)
    scores2 = score_with(booster2, test_base)
    r2 = order_by_scores(users, pool, scores2)
    results["R2"] = evaluate("R2_retrained_9features", r2, ground_truth)
    results["R2_rerank"] = evaluate("R2_retrained_9features+rerank", rerank(data, r2), ground_truth)
    booster2.save_model(str(ROOT / "ranker_R2.txt"))

    # ---------------------------------------------------------------- R3 加交互先验
    booster3, importance3, time3 = train_ranker(
        "R3", {**base_train, **extra_train}, {**base_valid, **extra_valid}, MAX_ROUNDS
    )
    scores3 = score_with(
        booster3, {user: {**test_base[user], **test_extra[user]} for user in users}
    )
    r3 = order_by_scores(users, pool, scores3)
    results["R3"] = evaluate("R3_retrained_14features", r3, ground_truth)
    results["R3_rerank"] = evaluate("R3_retrained_14features+rerank", rerank(data, r3), ground_truth)
    booster3.save_model(str(ROOT / "ranker_R3.txt"))

    # ---------------------------------------------------------------- R4 名次百分位混合
    def percentile(values: np.ndarray) -> np.ndarray:
        order = np.argsort(np.argsort(values))  # 0 = 最小
        return 1.0 - order / max(len(values) - 1, 1)

    blends: dict[str, dict] = {}
    for weight in (0.25, 0.5, 0.75):
        # 百分位统一约定「越大越好」：排序分数本身就是越大越好，融合名次要取负。
        mixed = {
            user: weight * percentile(scores3[user]) + (1 - weight) * percentile(-np.asarray(
                [row["fusion_rank"] for row in pool[user]], dtype=np.float64
            ))
            for user in users
        }
        ordered = order_by_scores(
            users, pool, {user: mixed[user] for user in users}
        )
        arm = f"R4_blend_w{weight}"
        payload = evaluate(arm, ordered, ground_truth)
        payload["rerank"] = evaluate(f"{arm}+rerank", rerank(data, ordered), ground_truth)["metrics"]
        blends[arm] = payload

    # ---------------------------------------------------------------- 随机重排锚点
    # 同一候选池随机打乱 50 次：判断各组「到底比随机好还是差」，没有这个锚点，
    # 0.002 和 0.005 之间的差别无法解释。
    rng = np.random.default_rng(20261110)
    random_draws = []
    random_per_user = []
    for _ in range(50):
        shuffled = {}
        for user in users:
            order = rng.permutation(len(pool[user]))
            shuffled[user] = [pool[user][int(index)]["item"] for index in order]
        metrics = per_user_metrics(shuffled, ground_truth, ks=(20,))
        random_draws.append(float(metrics["Recall@20"].mean()))
        random_per_user.append(metrics["Recall@20"])
    random_per_user = np.stack(random_per_user, axis=0)  # [50, users]
    random_anchor = {
        "arm": "R_random_permutation",
        "draws": len(random_draws),
        "Recall@20_mean": float(np.mean(random_draws)),
        "Recall@20_std_across_permutations": float(np.std(random_draws, ddof=1)),
        "Recall@20_p95": float(np.percentile(random_draws, 95)),
        "hit_targets_at20_mean": float(np.mean(random_draws) * sum(
            len(v) for v in ground_truth.values()
        )),
        "_per_user_mean": random_per_user.mean(axis=0),
    }
    log(f"随机重排锚点: Recall@20={random_anchor['Recall@20_mean']:.6f} "
        f"± {random_anchor['Recall@20_std_across_permutations']:.6f}")

    # ---------------------------------------------------------------- 诊断
    fusion_rank = {user: np.asarray([row["fusion_rank"] for row in pool[user]]) for user in users}

    def spearman(a: np.ndarray, b: np.ndarray) -> float:
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    diagnostics = {
        "spearman_legacy_vs_fusion": float(np.mean([
            spearman(legacy_scores[user], -fusion_rank[user]) for user in users
        ])),
        "spearman_r2_vs_fusion": float(np.mean([
            spearman(scores2[user], -fusion_rank[user]) for user in users
        ])),
        "spearman_r3_vs_fusion": float(np.mean([
            spearman(scores3[user], -fusion_rank[user]) for user in users
        ])),
        "top20_overlap_with_fusion": {
            label: round(float(np.mean([
                len(set(recs[user][:20]) & set(r0[user][:20])) / 20 for user in users
            ])), 4)
            for label, recs in (("R1", r1), ("R2", r2), ("R3", r3))
        },
        "importance_R2": importance2,
        "importance_R3": importance3,
        "train_seconds": {"R2": round(time2, 1), "R3": round(time3, 1)},
        "train_rows": int(len(train_view)),
        "train_requests": int(train_view.requests),
    }

    # ---------------------------------------------------------------- 成对区间
    baseline = results["R0"]["_per_user"]
    comparisons = {}
    for label in ("R1", "R2", "R3"):
        comparisons[label] = {
            key: paired_bootstrap(values[key], baseline[key])
            for key in ("Recall@20", "NDCG@20", "Recall@100")
            for values in (results[label]["_per_user"],)
        }
    comparisons["R4_blend_w0.5"] = {
        key: paired_bootstrap(blends["R4_blend_w0.5"]["_per_user"][key], baseline[key])
        for key in ("Recall@20", "NDCG@20", "Recall@100")
    }
    random_per_user_mean = random_anchor["_per_user_mean"]
    comparisons["R1_rerank"] = {
        metric: paired_bootstrap(
            results["R1_rerank"]["_per_user"][metric], results["R0_rerank"]["_per_user"][metric]
        )
        for metric in ("Recall@20", "NDCG@20")
    }
    # 「精排和重排一起去掉」相对现行最终输出的差值
    comparisons["no_ranker_no_rerank_vs_current_final"] = {
        metric: paired_bootstrap(
            results["R0"]["_per_user"][metric], results["R1_rerank"]["_per_user"][metric]
        )
        for metric in ("Recall@20", "NDCG@20")
    }
    comparisons_vs_random = {
        label: paired_bootstrap(results[label]["_per_user"]["Recall@20"], random_per_user_mean)
        for label in ("R0", "R1", "R2", "R3")
    }
    comparisons_vs_random["R4_blend_w0.5"] = paired_bootstrap(
        blends["R4_blend_w0.5"]["_per_user"]["Recall@20"], random_per_user_mean
    )

    def strip(payload: dict) -> dict:
        return {k: v for k, v in payload.items() if not k.startswith("_per_user")}

    output = {
        "protocol": {
            "dataset": "kuasar_small",
            "split": "test (2000 users / 4275 targets / 454,717 items)",
            "candidate_pool": str(BASELINE_DIR / "fusion.json"),
            "candidate_pool_note": "全部组别共用同一批 100 条融合候选，只改排序顺序",
            "ranker_training": {
                "legacy": "2,418 requests / 9,997 rows / 20 rounds (artifacts/kuasar_small/mvp)",
                "R2_R3": f"{train_view.requests:,} requests / {len(train_view):,} rows, "
                          f"Lambdarank, early stop on parent validation NDCG@20",
            },
            "features": {
                "base": list(ITEM_FEATURE_COLUMNS),
                "extra": ["item_pop_before", "author_pop_before", "user_activity_before",
                          "user_category_before", "user_author_before"],
                "extra_note": "只使用严格早于请求时间戳的 train 点击；训练行与推理行同一实现",
            },
            "blend": "w * rank_percentile(ranker) + (1-w) * rank_percentile(fusion)，w=0.5 先验固定",
        },
        "arms": {key: strip(value) for key, value in results.items()},
        "random_anchor": strip(random_anchor),
        "blend_sensitivity": {key: strip(value) for key, value in blends.items()},
        "comparisons_vs_R0": comparisons,
        "comparisons_vs_random": comparisons_vs_random,
        "diagnostics": diagnostics,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"写出 {OUT}（{output['elapsed_s']}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
