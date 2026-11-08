"""融合配额评测工具：双塔 / MIND 在 100 个候选预算下怎么分配。

可独立运行（不需要 LLM）：

    python -m recsys.experiments.fusion_quota --two-tower 50 --mind 50
    python -m recsys.experiments.fusion_quota --two-tower 30 --mind 70 --tag 30_70

只读上一轮 MVP 的缓存（不重新召回、不训练排序器）：

* ``artifacts/kuasar_small/mvp/users.json``
* ``artifacts/kuasar_small/mvp/recall_two_tower.json``
* ``artifacts/kuasar_small/mvp/recall_mind.json``
* ``artifacts/kuasar_small/mvp/run_meta.json``（指纹与预算）

融合规则（固定顺序，双塔在前）：

1. 两路各自按原始名次往后走，按配额交替取；**只有成功加入新物品才消耗该路配额**；
2. 遇到重复物品时不消耗配额，在该路继续往后取；
3. 一路配额用完或耗尽时由另一路补位，记录实际占用与补位数；
4. 候选总数不超过 100，来源与原始名次全部保留。

指标：命中目标数（按 ``(user_idx, item_idx)`` 对计数）、Recall@100（按用户平均，
复用 ``evaluate_recommendations``）、Coverage、用户/目标数、实际候选数与耗时。
两路**未截断并集**的命中只作诊断，不与同预算融合结果混为一谈。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from recsys.config import DEFAULT_DATA_DIR
from recsys.data.loader import prepare_dataset
from recsys.data.schema import seen_items_for_split
from recsys.evaluation.evaluator import evaluate_recommendations, group_items_by_user
from recsys.evaluation.metrics import catalog_coverage

#: 允许的配额组合（双塔, MIND），总预算固定 100。
ALLOWED_QUOTAS: tuple[tuple[int, int], ...] = ((50, 50), (70, 30), (30, 70), (80, 20), (20, 80))
DEFAULT_CACHE = Path("artifacts/kuasar_small/mvp")
DEFAULT_OUTPUT = Path("artifacts/kuasar_small/quota_agent")
TOTAL_BUDGET = 100


def tool_schema() -> dict:
    """给 LLM 的原生工具定义（OpenAI tools 格式），参数为两个必填整数配额。"""
    return {
        "type": "function",
        "function": {
            "name": "evaluate_fusion_quota",
            "description": (
                "在固定的 128 个验证用户、每用户 100 个候选的预算下，"
                "评测双塔与 MIND 的融合配额，返回命中目标数、Recall@100、Coverage 等真实指标。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "two_tower_quota": {
                        "type": "integer",
                        "description": "双塔配额（0-100），与 mind_quota 之和必须为 100",
                    },
                    "mind_quota": {
                        "type": "integer",
                        "description": "MIND 配额（0-100），与 two_tower_quota 之和必须为 100",
                    },
                },
                "required": ["two_tower_quota", "mind_quota"],
                "additionalProperties": False,
            },
        },
    }


class QuotaError(ValueError):
    """配额非法、缓存不一致或预算越界。"""


@dataclass(frozen=True)
class FusionContext:
    data: Any
    users: np.ndarray
    user_ids: list[int]
    targets: dict[int, set[int]]
    target_interactions: list
    seen: Any
    two_tower: dict[int, list[dict]]
    mind: dict[int, list[dict]]
    fingerprints: dict[str, Any]
    cache_dir: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_quota(two_tower_quota: int, mind_quota: int) -> tuple[int, int]:
    """只允许固定组合；总预算必须为 100，且不能出现负配额。"""
    if two_tower_quota < 0 or mind_quota < 0:
        raise QuotaError("配额不能为负")
    if two_tower_quota + mind_quota != TOTAL_BUDGET:
        raise QuotaError(f"配额之和必须为 {TOTAL_BUDGET}，收到 {two_tower_quota}+{mind_quota}")
    pair = (int(two_tower_quota), int(mind_quota))
    if pair not in ALLOWED_QUOTAS:
        raise QuotaError(f"不支持的配额 {pair}；允许：{ALLOWED_QUOTAS}")
    return pair


def load_context(cache_dir: Path = DEFAULT_CACHE, data_dir: Path = DEFAULT_DATA_DIR) -> FusionContext:
    """读取并核对缓存：用户名单、两路召回、指纹；不一致时直接报错，不做静默取交集。"""
    users_path = cache_dir / "users.json"
    meta_path = cache_dir / "run_meta.json"
    for path in (users_path, meta_path, cache_dir / "recall_two_tower.json", cache_dir / "recall_mind.json"):
        if not path.exists():
            raise QuotaError(f"缺少缓存文件：{path}（请先运行 recsys_mvp，且不要清空 artifacts）")

    user_payload = json.loads(users_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("fingerprints", {}).get("dataset") != "kuasar_small":
        raise QuotaError("run_meta 里的 dataset 不是 kuasar_small")

    data = prepare_dataset("kuasar_small", data_dir)
    if int(meta["fingerprints"]["users"]) != data.num_users or int(meta["fingerprints"]["items"]) != data.num_items:
        raise QuotaError("run_meta 的用户/物品数与当前子集不一致")

    user_ids = [int(value) for value in user_payload["users"]]
    try:
        users = np.asarray([int(data.mappings.user_to_index[user_id]) for user_id in user_ids], dtype=np.int64)
    except KeyError as error:
        raise QuotaError(f"users.json 里的用户不在当前子集词表：{error}") from error
    if len(set(users.tolist())) != len(users):
        raise QuotaError("users.json 存在重复用户")

    two_tower = {
        int(user): rows
        for user, rows in json.loads((cache_dir / "recall_two_tower.json").read_text(encoding="utf-8")).items()
    }
    mind = {
        int(user): rows
        for user, rows in json.loads((cache_dir / "recall_mind.json").read_text(encoding="utf-8")).items()
    }
    expected = set(users.tolist())
    for name, payload in (("two_tower", two_tower), ("mind", mind)):
        if set(payload) != expected:
            missing = expected - set(payload)
            extra = set(payload) - expected
            raise QuotaError(f"{name} 缓存的用户集合与 users.json 不一致：缺 {len(missing)}，多 {len(extra)}")
        for user, rows in payload.items():
            items = [int(row["item"]) for row in rows]
            if len(items) != len(set(items)):
                raise QuotaError(f"{name} 用户 {user} 的缓存候选有重复")
            if len(items) > TOTAL_BUDGET:
                raise QuotaError(f"{name} 用户 {user} 的缓存候选超过 {TOTAL_BUDGET}")
            if not rows or [int(row["rank"]) for row in rows] != list(range(1, len(rows) + 1)):
                raise QuotaError(f"{name} 用户 {user} 的原始名次不是 1..n")

    target_interactions = [row for row in data.validation_interactions if row.user_idx in expected]
    targets = group_items_by_user(target_interactions)
    if len(users) != 128 or len(target_interactions) != 362:
        raise QuotaError(f"验收口径不符：期望 128 用户 / 362 目标，实际 {len(users)} / {len(target_interactions)}")

    seen = seen_items_for_split(data, "validation")
    for user in users:
        blocked = seen.get(int(user), set())
        for name, payload in (("two_tower", two_tower), ("mind", mind)):
            leaked = {int(row["item"]) for row in payload[int(user)]} & blocked
            if leaked:
                raise QuotaError(f"{name} 用户 {user} 的缓存包含已曝光物品：{sorted(leaked)[:3]}")

    fingerprints = {
        "dataset": "kuasar_small",
        "users": int(data.num_users),
        "items": int(data.num_items),
        "split_strategy": meta["fingerprints"].get("split_strategy"),
        "split_sha256": meta["fingerprints"].get("split_sha256"),
        "id_mapping_sha256": meta["fingerprints"].get("id_mapping_sha256"),
        "checkpoints": {
            "two_tower": meta["fingerprints"].get("two_tower_checkpoint"),
            "mind": meta["fingerprints"].get("mind_checkpoint"),
        },
        "routing_seed": meta["fingerprints"].get("routing_seed"),
        "k_recall": meta["fingerprints"].get("k_recall"),
        "users_file_sha256": sha256_file(users_path),
        "recall_two_tower_sha256": sha256_file(cache_dir / "recall_two_tower.json"),
        "recall_mind_sha256": sha256_file(cache_dir / "recall_mind.json"),
    }
    return FusionContext(
        data=data,
        users=users,
        user_ids=user_ids,
        targets=targets,
        target_interactions=target_interactions,
        seen=seen,
        two_tower=two_tower,
        mind=mind,
        fingerprints=fingerprints,
        cache_dir=cache_dir,
    )


def _source_map(rows: list[dict]) -> dict[int, int]:
    return {int(row["item"]): int(row["rank"]) for row in rows}


def fuse_by_quota(
    two_tower_rows: list[dict],
    mind_rows: list[dict],
    two_tower_quota: int,
    mind_quota: int,
    budget: int = TOTAL_BUDGET,
) -> tuple[list[dict], dict[str, int]]:
    """按配额交替融合；返回候选池（含来源与名次）与占用统计。"""
    order = ("two_tower", "mind")
    lists = {"two_tower": two_tower_rows, "mind": mind_rows}
    quota = {"two_tower": two_tower_quota, "mind": mind_quota}
    used = {"two_tower": 0, "mind": 0}
    pointer = {"two_tower": 0, "mind": 0}
    pool: list[dict] = []
    items_in_pool: set[int] = set()
    fills = 0

    def take_next(channel: str) -> int | None:
        """取下一条未入池候选；重复项不消耗配额，直接跳过。"""
        rows = lists[channel]
        while pointer[channel] < len(rows):
            item = int(rows[pointer[channel]]["item"])
            pointer[channel] += 1
            if item not in items_in_pool:
                return item
        return None

    while len(pool) < budget and (used["two_tower"] < quota["two_tower"] or used["mind"] < quota["mind"]):
        progressed = False
        for channel in order:
            if len(pool) >= budget:
                break
            if used[channel] >= quota[channel]:
                continue
            item = take_next(channel)
            if item is None:
                continue
            items_in_pool.add(item)
            pool.append({"item": item, "sources": {}, "added_by": channel})
            used[channel] += 1
            progressed = True
        if not progressed:
            break

    # 补位：某一路耗尽时，另一路可以在配额之外补齐剩余预算
    while len(pool) < budget:
        progressed = False
        for channel in order:
            if len(pool) >= budget:
                break
            item = take_next(channel)
            if item is None:
                continue
            items_in_pool.add(item)
            pool.append({"item": item, "sources": {}, "added_by": channel})
            fills += 1
            progressed = True
        if not progressed:
            break

    # 补全来源：每个候选在所有通道里的原始名次（可查，不只记录加入时那一路）
    for row in pool:
        item = row["item"]
        sources: dict[str, dict[str, int]] = {}
        for channel in order:
            rank = _source_map(lists[channel]).get(item)
            if rank is not None:
                sources[channel] = {"rank": rank}
        row["sources"] = sources
    stats = {
        "two_tower_used": used["two_tower"],
        "mind_used": used["mind"],
        "fills": fills,
        "candidates": len(pool),
    }
    return pool, stats


def evaluate_fusion_quota(
    two_tower_quota: int,
    mind_quota: int,
    *,
    cache_dir: Path = DEFAULT_CACHE,
    output_dir: Path = DEFAULT_OUTPUT,
    tag: str | None = None,
) -> dict[str, Any]:
    """评测一组配额：这是 Agent 唯一的“实验工具”，也可以脱离 LLM 直接调用。"""
    quota = validate_quota(two_tower_quota, mind_quota)
    started = time.perf_counter()
    context = load_context(cache_dir)

    fused: dict[int, list[dict]] = {}
    stats: dict[str, dict[str, int]] = {}
    for user in context.users:
        pool, user_stats = fuse_by_quota(
            context.two_tower[int(user)], context.mind[int(user)], quota[0], quota[1]
        )
        fused[int(user)] = pool
        stats[int(user)] = user_stats

    recommendations = {user: [row["item"] for row in pool] for user, pool in fused.items()}
    metric = evaluate_recommendations(recommendations, context.target_interactions, ks=(TOTAL_BUDGET,))
    hits = sum(
        len({row["item"] for row in fused[int(user)]} & context.targets.get(int(user), set()))
        for user in context.users
    )
    union_hits = sum(
        len(
            (
                {int(row["item"]) for row in context.two_tower[int(user)]}
                | {int(row["item"]) for row in context.mind[int(user)]}
            )
            & context.targets.get(int(user), set())
        )
        for user in context.users
    )
    elapsed = round(time.perf_counter() - started, 2)

    label = tag or f"{quota[0]}_{quota[1]}"
    run_dir = output_dir / "cache_eval" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    fusion_payload = {
        "quota": {"two_tower": quota[0], "mind": quota[1]},
        "budget": TOTAL_BUDGET,
        "fingerprints": context.fingerprints,
        "users": len(context.users),
        "stats_by_user": {str(user): value for user, value in stats.items()},
        "pool": {str(user): pool for user, pool in fused.items()},
    }
    (run_dir / "fusion.json").write_text(
        json.dumps(fusion_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    result = {
        "tool": "evaluate_fusion_quota",
        "quota": {"two_tower": quota[0], "mind": quota[1]},
        "users": len(context.users),
        "targets": len(context.target_interactions),
        "candidates_total": int(sum(row["candidates"] for row in stats.values())),
        "candidates_mean": round(
            float(np.mean([row["candidates"] for row in stats.values()])), 3
        ),
        "two_tower_used_total": int(sum(row["two_tower_used"] for row in stats.values())),
        "mind_used_total": int(sum(row["mind_used"] for row in stats.values())),
        "fills_total": int(sum(row["fills"] for row in stats.values())),
        "hits": int(hits),
        "recall_at_100": round(float(metric["Recall@100"]), 6),
        "coverage": round(float(catalog_coverage(recommendations, context.data.num_items)), 6),
        "elapsed_s": elapsed,
        "diagnostic_union_hits_at_100": int(union_hits),
        "evidence": {
            "fusion": str(run_dir / "fusion.json"),
            "cache": str(cache_dir),
            "fingerprints": context.fingerprints,
        },
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-tower", type=int, required=True, dest="two_tower")
    parser.add_argument("--mind", type=int, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tag", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = evaluate_fusion_quota(
        args.two_tower,
        args.mind,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        tag=args.tag,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
