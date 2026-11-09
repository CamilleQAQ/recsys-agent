"""可序列化的 BM25 索引。

只依赖标准库：索引小（本轮最多 1,000 个物品），保存成 JSON 即可跨进程加载。
排序规则固定为 **score 降序、同分按 item_id 升序**；空查询和无匹配都返回空列表。

BM25 采用常见的 Robertson/Sparck Jones IDF 形式：

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    score(q, d) = Σ_t qtf(t) * idf(t) * tf(t,d) * (k1 + 1)
                  / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

其中 ``qtf`` 是查询里 token 的重复次数。重复 token 会被计入，不做去重。
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

INDEX_FORMAT = "recsys.bm25.v1"


@dataclass(frozen=True)
class IndexedItem:
    """索引里的一个物品。caption_tokens 就是 KuaiSAR 原始 token ID。"""

    item_id: int
    caption_tokens: tuple[int, ...]
    category: str = ""
    category_zh: str = ""
    item_type: str = ""
    author_id: int = -1

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "caption_tokens": list(self.caption_tokens),
            "category": self.category,
            "category_zh": self.category_zh,
            "item_type": self.item_type,
            "author_id": self.author_id,
        }


@dataclass(frozen=True)
class SearchHit:
    """一条检索结果；rank 从 1 开始。"""

    rank: int
    item_id: int
    score: float
    category: str
    category_zh: str = ""
    item_type: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "item_id": self.item_id,
            "score": self.score,
            "category": self.category,
            "category_zh": self.category_zh,
            "item_type": self.item_type,
        }


class BM25Index:
    """内存 BM25 索引；构造后不可变，可保存/加载。"""

    def __init__(
        self,
        items: Iterable[IndexedItem],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._items = tuple(items)
        if not self._items:
            raise ValueError("BM25Index 至少需要一个物品")
        if k1 < 0:
            raise ValueError(f"k1 必须 >= 0，实际 {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b 必须在 [0, 1]，实际 {b}")
        if len({item.item_id for item in self._items}) != len(self._items):
            raise ValueError("item_id 不能重复")

        self._k1 = float(k1)
        self._b = float(b)
        self._doc_len = [len(item.caption_tokens) for item in self._items]
        total_len = sum(self._doc_len)
        self._avgdl = total_len / len(self._items) if total_len else 1.0

        postings: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for doc_idx, item in enumerate(self._items):
            for token, tf in Counter(item.caption_tokens).items():
                postings[token].append((doc_idx, tf))
        self._postings = {token: tuple(rows) for token, rows in postings.items()}
        num_docs = len(self._items)
        self._idf = {
            token: math.log(1.0 + (num_docs - len(rows) + 0.5) / (len(rows) + 0.5))
            for token, rows in self._postings.items()
        }

    @property
    def items(self) -> tuple[IndexedItem, ...]:
        return self._items

    @property
    def num_docs(self) -> int:
        return len(self._items)

    @property
    def vocab_size(self) -> int:
        return len(self._postings)

    @property
    def avgdl(self) -> float:
        return self._avgdl

    @property
    def k1(self) -> float:
        return self._k1

    @property
    def b(self) -> float:
        return self._b

    def __len__(self) -> int:
        return len(self._items)

    def search(self, query_tokens: Iterable[int] | None, top_k: int = 10) -> list[SearchHit]:
        """按 BM25 分数返回 Top-K。

        - ``query_tokens`` 为空 / ``None`` → ``[]``；
        - 没有任何 token 命中 → ``[]``；
        - 结果按 ``(-score, item_id)`` 排序，同分时 item_id 小的在前。
        """
        if top_k <= 0:
            return []
        if query_tokens is None:
            return []
        query_counts = Counter(int(token) for token in query_tokens)
        if not query_counts:
            return []

        scores = [0.0] * len(self._items)
        matched: set[int] = set()
        for token, qtf in query_counts.items():
            rows = self._postings.get(token)
            if not rows:
                continue
            idf = self._idf[token]
            for doc_idx, tf in rows:
                doc_len = self._doc_len[doc_idx]
                norm = self._k1 * (1.0 - self._b + self._b * doc_len / self._avgdl)
                scores[doc_idx] += qtf * idf * (tf * (self._k1 + 1.0)) / (tf + norm)
                matched.add(doc_idx)

        ranked = sorted(
            ((scores[doc_idx], self._items[doc_idx].item_id, doc_idx) for doc_idx in matched),
            key=lambda row: (-row[0], row[1]),
        )[:top_k]
        hits: list[SearchHit] = []
        for rank, (score, _item_id, doc_idx) in enumerate(ranked, start=1):
            item = self._items[doc_idx]
            hits.append(
                SearchHit(
                    rank=rank,
                    item_id=item.item_id,
                    score=float(score),
                    category=item.category,
                    category_zh=item.category_zh,
                    item_type=item.item_type,
                )
            )
        return hits

    def summary(self) -> dict[str, Any]:
        lengths = self._doc_len
        return {
            "format": INDEX_FORMAT,
            "num_docs": self.num_docs,
            "vocab_size": self.vocab_size,
            "avgdl": round(self._avgdl, 6),
            "k1": self._k1,
            "b": self._b,
            "total_tokens": int(sum(lengths)),
            "min_doc_len": min(lengths) if lengths else 0,
            "max_doc_len": max(lengths) if lengths else 0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": INDEX_FORMAT,
            "bm25": {
                "k1": self._k1,
                "b": self._b,
                "avgdl": self._avgdl,
                "num_docs": self.num_docs,
                "vocab_size": self.vocab_size,
            },
            "items": [item.as_dict() for item in self._items],
            "doc_len": list(self._doc_len),
            "postings": {
                str(token): [[doc_idx, tf] for doc_idx, tf in rows]
                for token, rows in self._postings.items()
            },
            "idf": {str(token): value for token, value in self._idf.items()},
        }

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(target)
        return target

    @classmethod
    def load(cls, path: Path | str) -> BM25Index:
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        fmt = payload.get("format")
        if fmt != INDEX_FORMAT:
            raise ValueError(f"{source} 不是 {INDEX_FORMAT} 索引（format={fmt!r}）")
        bm25 = payload.get("bm25") or {}
        items = [
            IndexedItem(
                item_id=int(row["item_id"]),
                caption_tokens=tuple(int(token) for token in row.get("caption_tokens", [])),
                category=str(row.get("category", "")),
                category_zh=str(row.get("category_zh", "")),
                item_type=str(row.get("item_type", "")),
                author_id=int(row.get("author_id", -1)),
            )
            for row in payload.get("items", [])
        ]
        index = cls(items, k1=float(bm25.get("k1", 1.5)), b=float(bm25.get("b", 0.75)))
        # 保存的统计量用于快速人工核对；加载后仍以从 items 重算的结果为准。
        expected = {
            "num_docs": index.num_docs,
            "vocab_size": index.vocab_size,
        }
        for key, value in expected.items():
            if key in bm25 and int(bm25[key]) != value:
                raise ValueError(f"{source} 索引不一致：{key} 记录 {bm25[key]}，重算 {value}")
        if "avgdl" in bm25 and abs(float(bm25["avgdl"]) - index.avgdl) > 1e-6:
            raise ValueError(
                f"{source} 索引不一致：avgdl 记录 {bm25['avgdl']}，重算 {index.avgdl}"
            )
        return index


def load_index(path: Path | str) -> BM25Index:
    """便捷入口：``from recsys.search import load_index``。"""
    return BM25Index.load(path)
