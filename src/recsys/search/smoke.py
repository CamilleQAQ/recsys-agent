"""Search MVP 冒烟验收：一次进程内检查「有匹配 / 空查询 / 无匹配」。

    python -m recsys.search.smoke --top-k 5 \\
        --output experiments/2026-11-09-kuasar-search-bm25/smoke.json

检查项：

1. 从 ``sample.json`` 里选一个能命中的会话查询，核对 Top-K 数量、rank 连续、
   score 降序、同分 item_id 升序、返回字段齐全；
2. ``search([], top_k)`` 返回空列表；
3. 用一个不在索引词表里的 token 查询，返回空列表。

脚本只读已建好的索引与样本，不重建、不训练、不评测全量数据。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from recsys.search.bm25 import BM25Index, SearchHit

DEFAULT_INDEX_PATH = Path("artifacts/search/kuasar_bm25/index.json")
DEFAULT_SAMPLE_PATH = Path("artifacts/search/kuasar_bm25/sample.json")
_REQUIRED_FIELDS = {"rank", "item_id", "score", "category"}


def _find_match(
    index: BM25Index,
    sessions: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for session in sessions:
        query = [int(token) for token in session.get("keyword", [])]
        if not query:
            continue
        all_hits = index.search(query, top_k=len(index))
        if not all_hits:
            continue
        top = all_hits[:top_k]
        clicked = {
            int(item_id)
            for item_id in session.get("clicked_joined_items")
            or session.get("clicked_items")
            or []
        }
        clicked_ranks = [hit.rank for hit in top if hit.item_id in clicked]
        clicked_hit = bool(clicked_ranks)
        clicked_rank = min(clicked_ranks) if clicked_ranks else None
        # 优先「点击物品在 Top-K 内」且排名靠前、query 更长的会话，
        # 这样冒烟展示的是真正的多 token 查询而不是单 token 特例。
        key = (
            1 if clicked_hit else 0,
            min(len(all_hits), top_k),
            -(clicked_rank if clicked_rank is not None else 999),
            len(query),
            len(all_hits),
        )
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "session": session,
                "query": query,
                "all_hits": all_hits,
                "top": top,
                "clicked": sorted(clicked),
                "clicked_hit": clicked_hit,
                "clicked_rank": clicked_rank,
            }
    return best


def _fields_ok(hits: list[SearchHit]) -> bool:
    for hit in hits:
        payload = hit.as_dict()
        if not _REQUIRED_FIELDS.issubset(payload):
            return False
        if not isinstance(payload["rank"], int) or payload["rank"] < 1:
            return False
        if not isinstance(payload["item_id"], int):
            return False
        if not isinstance(payload["score"], float):
            return False
        if not isinstance(payload["category"], str) or not payload["category"]:
            return False
    return True


def _score_order_ok(hits: list[SearchHit]) -> bool:
    return all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))


def _tie_order_ok(hits: list[SearchHit]) -> bool:
    for left, right in zip(hits, hits[1:]):
        if left.score == right.score and left.item_id > right.item_id:
            return False
    return True


def run_smoke(index_path: Path, sample_path: Path, top_k: int = 5) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k 必须 > 0")
    index = BM25Index.load(index_path)
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    sessions = sample.get("sessions") or []
    if not sessions:
        raise ValueError(f"{sample_path} 里没有 sessions")

    match = _find_match(index, sessions, top_k)
    empty_hits = index.search([], top_k=top_k)
    no_match_token = 10**12
    no_match_hits = index.search([no_match_token], top_k=top_k)

    checks: dict[str, bool] = {
        "index_loaded": index.num_docs > 0 and index.vocab_size > 0,
        "match_found": match is not None,
        "top_k_respected": False,
        "rank_contiguous": False,
        "score_order": False,
        "tie_order": False,
        "fields_present": False,
        "empty_query": empty_hits == [],
        "no_match": no_match_hits == [],
    }

    match_payload: dict[str, Any] | None = None
    if match is not None:
        top = match["top"]
        all_hits = match["all_hits"]
        checks["top_k_respected"] = len(top) == min(top_k, len(all_hits))
        checks["rank_contiguous"] = [hit.rank for hit in top] == list(range(1, len(top) + 1))
        checks["score_order"] = _score_order_ok(top)
        checks["tie_order"] = _tie_order_ok(top)
        checks["fields_present"] = _fields_ok(top)
        session = match["session"]
        match_payload = {
            "session_id": session.get("session_id"),
            "query_tokens": match["query"],
            "top_k": top_k,
            "available_matches": len(all_hits),
            "clicked_items": match["clicked"],
            "clicked_in_top_k": match["clicked_hit"],
            "clicked_best_rank": match["clicked_rank"],
            "results": [hit.as_dict() for hit in top],
        }

    passed = all(checks.values())
    return {
        "format": "recsys.search.smoke.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "passed": passed,
        "checks": checks,
        "index": {
            "path": str(index_path),
            "num_docs": index.num_docs,
            "vocab_size": index.vocab_size,
            "avgdl": round(index.avgdl, 6),
            "k1": index.k1,
            "b": index.b,
        },
        "match": match_payload,
        "empty_query": {"query_tokens": [], "results": []},
        "no_match": {
            "query_tokens": [no_match_token],
            "results": [hit.as_dict() for hit in no_match_hits],
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None, help="把验收结果写成 JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.index, args.sample):
        if not path.exists():
            print(f"[error] 找不到 {path}；先运行 python -m recsys.search.prepare", file=sys.stderr)
            return 2
    try:
        result = run_smoke(args.index, args.sample, top_k=args.top_k)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"[error] 冒烟失败：{error}", file=sys.stderr)
        return 1

    print(
        f"[smoke] index docs={result['index']['num_docs']} "
        f"vocab={result['index']['vocab_size']} k1={result['index']['k1']} b={result['index']['b']}"
    )
    match = result["match"]
    if match is None:
        print("[smoke] no matching query found in sample")
    else:
        print(
            f"[smoke] match session={match['session_id']} query={match['query_tokens']} "
            f"available={match['available_matches']} clicked_in_top_k={match['clicked_in_top_k']}"
        )
        for hit in match["results"]:
            print(
                f"        rank={hit['rank']} item_id={hit['item_id']} "
                f"score={hit['score']:.6f} category={hit['category']}"
            )
    print(f"[smoke] empty query -> {len(result['empty_query']['results'])} results")
    print(f"[smoke] no-match token -> {len(result['no_match']['results'])} results")
    for name, ok in result["checks"].items():
        print(f"[smoke] {name}: {'ok' if ok else 'FAIL'}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[smoke] wrote {args.output}")

    if result["passed"]:
        print("[ok] search smoke passed (match / empty / no-match)")
        return 0
    print("[fail] search smoke failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
