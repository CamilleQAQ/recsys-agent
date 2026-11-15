"""搜索查询 CLI（独立进程）。

保留原有 BM25 用法，并新增向量 / 混合 / 精排三种模式：

    python -m recsys.search --tokens "3345744,3936332,3847879" --top-k 5
    python -m recsys.search --mode vector --tokens 3345744 3936332 --top-k 5
    python -m recsys.search --mode hybrid --tokens "..." --top-k 5
    python -m recsys.search --mode rerank --tokens "..." --top-k 5 --json
    python -m recsys.search --top-k 5                       # 空查询 → []

默认 ``--mode bm25``，只读 BM25 索引；``vector``/``hybrid``/``rerank`` 需要
已构建的搜索向量索引（``python -m recsys.search.build_vector_index``）。
查询 token 就是 KuaiSAR 原始 token ID；输出物品 ID、分数和可读类目，
混合/精排模式额外显示来源（``bm25#1``、``vector#3``）。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from recsys.search.bm25 import BM25Index
from recsys.search.build_vector_index import DEFAULT_CONFIG_PATH, load_config
from recsys.search.encoders import SearchEncoder, build_encoder
from recsys.search.query import QueryForm, normalize_query, rewrite_query
from recsys.search.rerank import RERANKER_NAME
from recsys.search.service import MODES, SearchService, SearchServiceError

DEFAULT_INDEX_PATH = Path("artifacts/search/kuasar_bm25/index.json")
DEFAULT_VECTOR_DIR = Path("artifacts/search/kuasar_vector/vector")


def parse_tokens(values: list[str]) -> list[int]:
    tokens: list[int] = []
    for value in values:
        for part in value.replace(",", " ").split():
            try:
                tokens.append(int(part))
            except ValueError as error:
                raise ValueError(f"不是合法的 token ID：{part!r}") from error
    return tokens


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--vector-index", type=Path, default=DEFAULT_VECTOR_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", choices=list(MODES), default="bm25")
    parser.add_argument(
        "--tokens",
        nargs="*",
        default=[],
        help=(
            '查询 token ID；逗号或空格分隔，例如 "3345744,3936332" 或 3345744 3936332；'
            "省略或传空字符串表示空查询"
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--rewrite",
        default="none",
        help="受约束改写：none / drop_last / drop_first / drop_token:<id>",
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON（默认表格）")
    return parser.parse_args(argv)


def _build_service(args: argparse.Namespace) -> tuple[SearchService, SearchEncoder]:
    config = load_config(args.config)
    encoder = build_encoder(config["encoder"])
    hybrid_cfg = config.get("hybrid", {})
    rerank_cfg = config.get("rerank", {})
    service = SearchService.load(
        args.index,
        args.vector_index,
        encoder,
        rrf_k=int(hybrid_cfg.get("rrf_k", 60)),
        hybrid_pool=int(hybrid_cfg.get("pool", 50)),
        rerank_method=rerank_cfg.get("method", RERANKER_NAME),
        rerank_pool=int(rerank_cfg.get("pool", 50)),
        prior_weight=float(rerank_cfg.get("prior_weight", 0.35)),
        coverage_weight=float(rerank_cfg.get("coverage_weight", 0.65)),
    )
    return service, encoder


def _query_form(args: argparse.Namespace, tokens: list[int]) -> QueryForm:
    form = normalize_query(tokens)
    if args.rewrite and args.rewrite != "none":
        form = replace(rewrite_query(form.normalized, args.rewrite), original=form.original)
    return form


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        query_tokens = parse_tokens(args.tokens)
        form = _query_form(args, query_tokens)
    except ValueError as error:
        print(f"[error] {error}", file=sys.stderr)
        return 2

    if not args.index.exists():
        print(
            f"[error] 找不到索引 {args.index}；先运行 python -m recsys.search.prepare",
            file=sys.stderr,
        )
        return 2

    if args.mode == "bm25":
        try:
            index = BM25Index.load(args.index)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"[error] 索引加载失败：{error}", file=sys.stderr)
            return 2
        hits = index.search(form.normalized, top_k=args.top_k)
        payload = {
            "mode": "bm25",
            "query": form.as_dict(),
            "top_k": args.top_k,
            "count": len(hits),
            "results": [hit.as_dict() for hit in hits],
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=True, indent=2))
            return 0
        print(f"mode=bm25  query={list(form.normalized)}  归一化={form.normalization}")
        if not hits:
            print("[]")
            return 0
        print(f"{'rank':>4}  {'item_id':>10}  {'score':>12}  category")
        for hit in hits:
            print(f"{hit.rank:>4}  {hit.item_id:>10}  {hit.score:>12.6f}  {hit.category}")
        return 0

    try:
        service, _encoder = _build_service(args)
    except (OSError, ValueError, SearchServiceError) as error:
        print(f"[error] 搜索服务初始化失败：{error}", file=sys.stderr)
        return 2
    hits = service.search(args.mode, form.normalized, top_k=args.top_k)
    payload = {
        "mode": args.mode,
        "query": form.as_dict(),
        "top_k": args.top_k,
        "count": len(hits),
        "results": [hit.as_dict(with_sources=True) for hit in hits],
        "backend": {
            "vector": service.vector.backend_actual,
            "faiss_active": service.vector.faiss_active,
            "faiss_search_calls": service.vector.faiss_search_calls,
        },
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return 0

    print(
        f"mode={args.mode}  query={list(form.normalized)}  归一化={form.normalization}  "
        f"vector_backend={service.vector.backend_actual}"
    )
    if not hits:
        print("[]")
        return 0
    header = f"{'rank':>4}  {'item_id':>10}  {'score':>12}  {'sources':<24}  category"
    print(header)
    for hit in hits:
        sources = ",".join(f"{s.source}#{s.rank}" for s in hit.sources)
        print(
            f"{hit.rank:>4}  {hit.item_id:>10}  {hit.score:>12.6f}  "
            f"{sources:<24}  {hit.category}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
