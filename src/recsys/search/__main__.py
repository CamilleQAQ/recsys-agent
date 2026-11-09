"""Search MVP 查询 CLI（独立进程）。

    python -m recsys.search --tokens "3345744,3936332,3847879" --top-k 5
    python -m recsys.search --tokens 3345744 3936332 3847879 --top-k 5 --json
    python -m recsys.search --top-k 5                       # 空查询 → []

查询 token 就是 KuaiSAR 原始 token ID；输出物品 ID、BM25 分数和可读类目。
默认索引在 ``artifacts/search/kuasar_bm25/index.json``，可用 ``--index`` 覆盖。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from recsys.search.bm25 import BM25Index

DEFAULT_INDEX_PATH = Path("artifacts/search/kuasar_bm25/index.json")


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
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON（默认表格）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        query_tokens = parse_tokens(args.tokens)
    except ValueError as error:
        print(f"[error] {error}", file=sys.stderr)
        return 2

    if not args.index.exists():
        print(
            f"[error] 找不到索引 {args.index}；先运行 python -m recsys.search.prepare",
            file=sys.stderr,
        )
        return 2
    try:
        index = BM25Index.load(args.index)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"[error] 索引加载失败：{error}", file=sys.stderr)
        return 2

    hits = index.search(query_tokens, top_k=args.top_k)
    if args.json:
        print(
            json.dumps(
                {
                    "query_tokens": query_tokens,
                    "top_k": args.top_k,
                    "count": len(hits),
                    "results": [hit.as_dict() for hit in hits],
                },
                ensure_ascii=True,
                indent=2,
            )
        )
        return 0

    if not hits:
        print("[]")
        return 0
    print(f"{'rank':>4}  {'item_id':>10}  {'score':>10}  category")
    for hit in hits:
        print(f"{hit.rank:>4}  {hit.item_id:>10}  {hit.score:>10.6f}  {hit.category}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
