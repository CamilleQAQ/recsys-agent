"""Phase 4 本地 RAG：资料解析、索引、检索、引用和增量更新。

* 独立索引目录（默认 ``artifacts/rag/phase4``），与推荐物品向量索引、搜索侧
  FAISS 索引分开；
* 只使用本地资料：项目文档、模型说明、开源仓库来源、历史实验报告与失败记录；
* 每个片段保留 ``path + start_line + end_line + heading``，检索结果可直接定位原文；
* 索引按文件 sha256 增量更新：只有内容变化的文件重新切块，其余复用；
* 提供 ``retrieve_knowledge`` 工具 schema，Research Agent 与三个业务 Agent 共用。

文档检索只负责“查背景、实现和来源”；历史实验能否复用仍由
:mod:`recsys.platform.shared_experiments` 的校验逻辑判定。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tomllib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from recsys.platform.locks import file_lock

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = Path("configs/platform/rag.toml")
DEFAULT_INDEX_DIR = Path("artifacts/rag/phase4")
INDEX_FILENAME = "index.json"
INDEX_FORMAT = "recsys.platform.rag.index.v1"
SCHEMA_VERSION = 1

MAX_TOP_K = 10
DEFAULT_TOP_K = 3

#: 不在配置里重复声明时使用的本地资料集合（少量、可复现，不下载任何权重）。
DEFAULT_SOURCES: tuple[dict[str, str], ...] = (
    {"path": "README.md", "kind": "project"},
    {"path": "PROJECT_MASTER_v2.md", "kind": "manual"},
    {"path": "docs/architecture.md", "kind": "project"},
    {"path": "docs/rec-agent.md", "kind": "project"},
    {"path": "docs/search-agent.md", "kind": "project"},
    {"path": "docs/ads-agent.md", "kind": "project"},
    {"path": "docs/model-zoo.md", "kind": "model_registry"},
    {"path": "docs/models/two_tower.md", "kind": "model"},
    {"path": "docs/models/mind.md", "kind": "model"},
    {"path": "docs/models/faiss.md", "kind": "model"},
    {"path": "third_party/SOURCES.toml", "kind": "open_source"},
    {
        "path": "experiments/2026-11-08-kuasar_small-fusion-quota-agent/report.md",
        "kind": "experiment",
    },
    {
        "path": "experiments/2026-11-08-kuasar_small-history-first-agent/report.md",
        "kind": "experiment",
    },
    {
        "path": "experiments/2026-11-09-kuasar-search-agent/report.md",
        "kind": "experiment",
    },
    {
        "path": "experiments/2026-11-09-kuasar_small-ads-phase3/report.md",
        "kind": "experiment",
    },
    {
        "path": "experiments/2026-11-09-kuasar_small-offline-replay-llm-regression/report.md",
        "kind": "experiment_failure",
    },
    {
        "path": "experiments/2026-11-08-kuasar_small-experiment-memory/report.md",
        "kind": "experiment",
    },
)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


class RagError(RuntimeError):
    """索引缺失、损坏或参数不合法。"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def display_path(value: str | Path) -> str:
    path = Path(value)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except (ValueError, OSError):
        return str(path)


def resolve_path(value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize(text: str) -> list[str]:
    """英文/数字按词，中文按二元组；对匿名 token 和项目术语都可用。"""

    tokens: list[str] = []
    for match in _CJK_RE.finditer(text.lower()):
        part = match.group(0)
        if part[0].isascii():
            tokens.append(part)
            continue
        if len(part) == 1:
            tokens.append(part)
        else:
            tokens.extend(part[index : index + 2] for index in range(len(part) - 1))
    return tokens


def load_rag_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = resolve_path(path or DEFAULT_CONFIG_PATH)
    payload: dict[str, Any] = {}
    if config_path and config_path.exists():
        try:
            payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise RagError(f"RAG 配置无法读取：{config_path}（{error}）") from error
    section = dict(payload.get("rag") or {})
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        sources = [dict(item) for item in DEFAULT_SOURCES]
    normalized_sources = []
    for item in sources:
        if not isinstance(item, Mapping) or not (item.get("path") or item.get("dir")):
            continue
        normalized_sources.append(
            {
                "path": str(item["path"]) if item.get("path") else None,
                "dir": str(item["dir"]) if item.get("dir") else None,
                "kind": str(item.get("kind") or "project"),
            }
        )
    return {
        "config_path": str(config_path) if config_path and config_path.exists() else None,
        "index_dir": str(section.get("index_dir") or DEFAULT_INDEX_DIR),
        "top_k": int(section.get("top_k", DEFAULT_TOP_K)),
        "max_chunk_chars": int(section.get("max_chunk_chars", 1400)),
        "chunk_lines": int(section.get("chunk_lines", 24)),
        "overlap_lines": int(section.get("overlap_lines", 4)),
        "sources": normalized_sources,
    }


# ---------------------------------------------------------------------- 切块


def _chunk_markdown(
    text: str,
    *,
    path: str,
    kind: str,
    max_chars: int,
) -> list[dict[str, Any]]:
    lines = text.splitlines()
    chunks: list[dict[str, Any]] = []
    heading = ""
    buffer: list[str] = []
    start_line = 1

    def flush(end_line: int) -> None:
        nonlocal buffer
        body = "\n".join(buffer).strip()
        if body:
            chunks.append(
                {
                    "path": path,
                    "kind": kind,
                    "heading": heading,
                    "start_line": start_line,
                    "end_line": max(start_line, end_line),
                    "text": body,
                }
            )
        buffer = []

    for index, line in enumerate(lines, start=1):
        match = _HEADING_RE.match(line)
        if match:
            flush(index - 1)
            heading = line.strip()
            start_line = index
            buffer = [line]
            continue
        buffer.append(line)
        current_chars = sum(len(item) + 1 for item in buffer)
        if current_chars >= max_chars:
            flush(index)
            start_line = index + 1
    flush(len(lines))
    return chunks


def _chunk_windows(
    text: str,
    *,
    path: str,
    kind: str,
    chunk_lines: int,
    overlap_lines: int,
) -> list[dict[str, Any]]:
    lines = text.splitlines()
    if not lines:
        return []
    chunk_lines = max(4, int(chunk_lines))
    overlap_lines = max(0, min(int(overlap_lines), chunk_lines - 1))
    step = max(1, chunk_lines - overlap_lines)
    chunks: list[dict[str, Any]] = []
    for start in range(0, len(lines), step):
        end = min(len(lines), start + chunk_lines)
        if end <= start:
            break
        body = "\n".join(lines[start:end]).strip()
        if body:
            chunks.append(
                {
                    "path": path,
                    "kind": kind,
                    "heading": "",
                    "start_line": start + 1,
                    "end_line": end,
                    "text": body,
                }
            )
        if end >= len(lines):
            break
    return chunks


def _chunk_file(path: Path, *, kind: str, max_chars: int, chunk_lines: int, overlap_lines: int) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rel = display_path(path)
    if path.suffix.lower() in {".md", ".markdown"}:
        return _chunk_markdown(text, path=rel, kind=kind, max_chars=max_chars)
    return _chunk_windows(
        text, path=rel, kind=kind, chunk_lines=chunk_lines, overlap_lines=overlap_lines
    )


def _chunk_id(chunk: Mapping[str, Any]) -> str:
    payload = f"{chunk.get('path')}|{chunk.get('start_line')}|{chunk.get('end_line')}|{chunk.get('text')}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _snippet(text: str, limit: int = 700) -> str:
    text = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ---------------------------------------------------------------------- 索引


class RagIndex:
    """本地资料索引；``build_or_update`` 增量更新，``search`` 返回可定位片段。"""

    def __init__(
        self,
        index_dir: Path | str | None = None,
        *,
        config_path: Path | str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = dict(config or load_rag_config(config_path))
        self.index_dir = resolve_path(index_dir or self.config["index_dir"])
        assert self.index_dir is not None
        self.index_path = self.index_dir / INDEX_FILENAME
        self.lock_path = self.index_dir / ".rag.lock"
        self._payload: dict[str, Any] | None = None

    # ------------------------------------------------------------------ 构建

    def build_or_update(self, *, force: bool = False) -> dict[str, Any]:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_path):
            old = self._read_existing()
            old_files = {
                str(item.get("path")): item
                for item in (old or {}).get("sources") or []
                if isinstance(item, Mapping)
            }
            old_chunks_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for chunk in (old or {}).get("chunks") or []:
                if isinstance(chunk, Mapping):
                    old_chunks_by_path[str(chunk.get("path"))].append(dict(chunk))

            sources: list[dict[str, Any]] = []
            chunks: list[dict[str, Any]] = []
            added: list[str] = []
            changed: list[str] = []
            unchanged: list[str] = []
            missing: list[str] = []
            indexed_paths: set[str] = set()
            directory_suffixes = {".md", ".markdown", ".toml", ".txt", ".jsonl"}

            def index_file(source: Mapping[str, Any], path: Path) -> None:
                rel = display_path(path)
                digest = sha256_file(path)
                previous = old_files.get(rel)
                if previous and previous.get("sha256") == digest and not force:
                    reused = old_chunks_by_path.get(rel) or []
                    if reused:
                        chunks.extend(reused)
                        sources.append(
                            {
                                **previous,
                                "kind": str(source["kind"]),
                                "updated_at": previous.get("updated_at") or utc_now(),
                            }
                        )
                        unchanged.append(rel)
                        indexed_paths.add(rel)
                        return
                fresh = _chunk_file(
                    path,
                    kind=str(source["kind"]),
                    max_chars=int(self.config["max_chunk_chars"]),
                    chunk_lines=int(self.config["chunk_lines"]),
                    overlap_lines=int(self.config["overlap_lines"]),
                )
                for chunk in fresh:
                    chunk["chunk_id"] = _chunk_id(chunk)
                sources.append(
                    {
                        "path": rel,
                        "kind": str(source["kind"]),
                        "sha256": digest,
                        "size": path.stat().st_size,
                        "chunks": len(fresh),
                        "updated_at": utc_now(),
                    }
                )
                chunks.extend(fresh)
                indexed_paths.add(rel)
                if previous:
                    changed.append(rel)
                else:
                    added.append(rel)

            for source in self.config["sources"]:
                if source.get("dir"):
                    root = resolve_path(source.get("dir"))
                    if root is None or not root.exists() or not root.is_dir():
                        missing.append(str(source.get("dir")))
                        continue
                    for path in sorted(root.rglob("*")):
                        if path.is_file() and path.suffix.lower() in directory_suffixes:
                            index_file(source, path)
                    continue
                path = resolve_path(source.get("path"))
                if path is None or not path.exists() or not path.is_file():
                    missing.append(str(source.get("path")))
                    continue
                index_file(source, path)

            removed = sorted(set(old_files) - indexed_paths)
            postings, df, lengths = self._build_postings(chunks)
            stats = {
                "files": len(sources),
                "chunks": len(chunks),
                "vocab": len(postings),
                "avgdl": round(sum(lengths) / len(lengths), 3) if lengths else 0.0,
            }
            payload = {
                "format": INDEX_FORMAT,
                "schema_version": SCHEMA_VERSION,
                "created_at": (old or {}).get("created_at") or utc_now(),
                "updated_at": utc_now(),
                "index_dir": display_path(self.index_dir),
                "config": {
                    "max_chunk_chars": self.config["max_chunk_chars"],
                    "chunk_lines": self.config["chunk_lines"],
                    "overlap_lines": self.config["overlap_lines"],
                    "tokenizer": "ascii_word+cjk_bigram",
                },
                "sources": sources,
                "chunks": chunks,
                "postings": {token: values for token, values in sorted(postings.items())},
                "df": dict(sorted(df.items())),
                "lengths": lengths,
                "stats": stats,
            }
            self._write(payload)
            self._payload = payload
            return {
                "ok": True,
                "index_dir": display_path(self.index_dir),
                "index_path": str(self.index_path),
                "stats": stats,
                "added": added,
                "changed": changed,
                "unchanged": unchanged,
                "removed": removed,
                "missing": missing,
                "force": bool(force),
            }

    def _build_postings(
        self, chunks: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, list[list[int]]], Counter[str], list[int]]:
        postings: dict[str, list[list[int]]] = defaultdict(list)
        df: Counter[str] = Counter()
        lengths: list[int] = []
        for index, chunk in enumerate(chunks):
            heading = str(chunk.get("heading") or "")
            body = str(chunk.get("text") or "")
            tokens = tokenize(f"{heading}\n{body}")
            counts = Counter(tokens)
            lengths.append(len(tokens))
            if not counts:
                continue
            chunk["length"] = len(tokens)
            for token, tf in counts.items():
                postings[token].append([index, int(tf)])
                df[token] += 1
        return postings, df, lengths

    def _read_existing(self) -> dict[str, Any] | None:
        if not self.index_path.exists():
            return None
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or payload.get("format") != INDEX_FORMAT:
            return None
        return payload

    def _write(self, payload: Mapping[str, Any]) -> None:
        temp = self.index_path.with_name(f"{INDEX_FILENAME}.tmp.{os.getpid()}")
        temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, self.index_path)

    # ------------------------------------------------------------------ 检索

    def load(self) -> dict[str, Any]:
        if self._payload is not None:
            return self._payload
        payload = self._read_existing()
        if payload is None:
            raise RagError(f"RAG 索引不存在或格式不兼容：{display_path(self.index_path)}")
        self._payload = payload
        return payload

    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        path_prefix: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise RagError("query 必须是非空字符串")
        top_k = max(1, min(MAX_TOP_K, int(top_k)))
        payload = self.load()
        chunks: list[dict[str, Any]] = list(payload.get("chunks") or [])
        postings: dict[str, list[list[int]]] = payload.get("postings") or {}
        df: dict[str, int] = payload.get("df") or {}
        lengths: list[int] = list(payload.get("lengths") or [])
        total = len(chunks)
        avgdl = (sum(lengths) / total) if total and lengths else 1.0
        k1, b = 1.5, 0.75
        query_counts = Counter(tokenize(query))
        scores: dict[int, float] = defaultdict(float)
        matched_tokens: dict[int, set[str]] = defaultdict(set)
        for token, qtf in query_counts.items():
            posting = postings.get(token)
            if not posting:
                continue
            token_df = max(1, int(df.get(token, len(posting))))
            idf = math.log(1 + (total - token_df + 0.5) / (token_df + 0.5))
            for chunk_index, tf in posting:
                if chunk_index >= total:
                    continue
                dl = lengths[chunk_index] if chunk_index < len(lengths) else 0
                denom = tf + k1 * (1 - b + b * dl / max(1.0, avgdl))
                scores[chunk_index] += idf * (tf * (k1 + 1)) / denom * (1.0 + 0.15 * min(qtf, 3))
                matched_tokens[chunk_index].add(token)
        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], str(chunks[item[0]].get("path")), int(chunks[item[0]].get("start_line", 0))),
        )
        hits: list[dict[str, Any]] = []
        for chunk_index, score in ranked:
            chunk = chunks[chunk_index]
            path = str(chunk.get("path") or "")
            if path_prefix and not path.startswith(path_prefix):
                continue
            start = int(chunk.get("start_line") or 1)
            end = int(chunk.get("end_line") or start)
            hits.append(
                {
                    "rank": len(hits) + 1,
                    "score": round(float(score), 6),
                    "path": path,
                    "kind": chunk.get("kind"),
                    "heading": chunk.get("heading") or None,
                    "start_line": start,
                    "end_line": end,
                    "citation": f"{path}:{start}-{end}",
                    "snippet": _snippet(str(chunk.get("text") or "")),
                    "matched_terms": sorted(matched_tokens.get(chunk_index, set()))[:8],
                }
            )
            if len(hits) >= top_k:
                break
        return {
            "ok": True,
            "query": query,
            "top_k": top_k,
            "hits": hits,
            "index": self.describe(include_sources=False),
        }

    def describe(self, *, include_sources: bool = True) -> dict[str, Any]:
        exists = self.index_path.exists()
        payload: dict[str, Any] | None = None
        if exists:
            try:
                payload = self.load()
            except RagError:
                payload = None
        description: dict[str, Any] = {
            "ok": bool(payload),
            "index_dir": display_path(self.index_dir),
            "index_path": display_path(self.index_path),
            "exists": exists,
            "format": (payload or {}).get("format"),
            "updated_at": (payload or {}).get("updated_at"),
            "stats": (payload or {}).get("stats") or {},
            "config": self.config.get("config_path") or "built-in default sources",
        }
        if include_sources and payload:
            description["sources"] = [
                {
                    "path": item.get("path"),
                    "kind": item.get("kind"),
                    "chunks": item.get("chunks"),
                    "sha256": (item.get("sha256") or "")[:12],
                }
                for item in payload.get("sources") or []
            ]
        return description

    def read_citation(self, citation: str) -> dict[str, Any]:
        """按 ``path:start-end`` 读回原文，用于验证引用可定位。"""

        match = re.match(r"^(.*):(\d+)-(\d+)$", str(citation))
        if not match:
            raise RagError(f"引用格式不合法：{citation}")
        path_value, start, end = match.group(1), int(match.group(2)), int(match.group(3))
        path = resolve_path(path_value)
        if path is None or not path.exists():
            raise RagError(f"引用文件不存在：{path_value}")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if start < 1 or end < start or end > len(lines):
            raise RagError(f"引用行号越界：{citation}（文件共 {len(lines)} 行）")
        return {
            "path": path_value,
            "start_line": start,
            "end_line": end,
            "text": "\n".join(lines[start - 1 : end]),
        }


# ---------------------------------------------------------------------- 工具


def retrieve_knowledge(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    index_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    path_prefix: str | None = None,
) -> dict[str, Any]:
    """给 Agent 的 RAG 工具实现：返回片段、分数和可定位引用。"""

    try:
        index = RagIndex(index_dir, config_path=config_path)
    except RagError as error:
        return {"ok": False, "tool": "retrieve_knowledge", "error": str(error)}
    if not index.index_path.exists():
        return {
            "ok": False,
            "tool": "retrieve_knowledge",
            "error": f"RAG 索引不存在：{display_path(index.index_path)}",
            "hint": "先运行 python -m recsys.platform --rag-build",
        }
    try:
        return {"tool": "retrieve_knowledge", **index.search(query, top_k=top_k, path_prefix=path_prefix)}
    except RagError as error:
        return {"ok": False, "tool": "retrieve_knowledge", "error": str(error)}


class KnowledgeTool:
    """业务 Agent 侧的最小 RAG 工具封装。"""

    def __init__(
        self,
        *,
        index_dir: Path | str | None = None,
        default_top_k: int = DEFAULT_TOP_K,
        path_prefix: str | None = None,
    ) -> None:
        self.index_dir = index_dir
        self.default_top_k = max(1, min(MAX_TOP_K, int(default_top_k)))
        self.path_prefix = path_prefix

    def retrieve(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        return retrieve_knowledge(
            query,
            top_k=top_k if top_k is not None else self.default_top_k,
            index_dir=self.index_dir,
            path_prefix=self.path_prefix,
        )

    def describe(self) -> dict[str, Any]:
        try:
            return RagIndex(self.index_dir).describe(include_sources=False)
        except RagError as error:
            return {"ok": False, "error": str(error)}


KNOWLEDGE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "retrieve_knowledge",
        "description": (
            "检索本地项目知识库（项目文档、模型说明、开源仓库来源、历史实验与失败记录）。"
            "返回相关片段及可定位来源 path:start-end；只用于查背景/实现/来源，"
            "历史实验能否复用必须再用实验历史工具校验。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "description": "检索问题或关键词"},
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_TOP_K,
                    "default": DEFAULT_TOP_K,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def knowledge_model_view(result: Mapping[str, Any]) -> dict[str, Any]:
    """给模型看的 RAG 结果：保留引用，去掉索引内部结构。"""

    view: dict[str, Any] = {
        "tool": "retrieve_knowledge",
        "ok": bool(result.get("ok")),
        "query": result.get("query"),
        "hits": [
            {
                "rank": hit.get("rank"),
                "score": hit.get("score"),
                "citation": hit.get("citation"),
                "path": hit.get("path"),
                "start_line": hit.get("start_line"),
                "end_line": hit.get("end_line"),
                "heading": hit.get("heading"),
                "snippet": hit.get("snippet"),
            }
            for hit in result.get("hits") or []
        ],
    }
    if not result.get("ok"):
        view["error"] = result.get("error")
        view["hint"] = result.get("hint")
    return view


def default_index_dir() -> Path:
    env = os.environ.get("RECSYS_RAG_INDEX", "").strip()
    if env:
        return Path(env)
    try:
        config = load_rag_config()
        return resolve_path(config["index_dir"]) or (REPO_ROOT / DEFAULT_INDEX_DIR)
    except RagError:
        return REPO_ROOT / DEFAULT_INDEX_DIR


def load_sources_for_update(path: Path | str | None = None) -> dict[str, Any]:
    config = load_rag_config(path)
    return config


__all__ = [
    "DEFAULT_INDEX_DIR",
    "DEFAULT_TOP_K",
    "KNOWLEDGE_TOOL_SCHEMA",
    "KnowledgeTool",
    "MAX_TOP_K",
    "RagError",
    "RagIndex",
    "default_index_dir",
    "display_path",
    "knowledge_model_view",
    "load_rag_config",
    "load_sources_for_update",
    "retrieve_knowledge",
    "sha256_file",
    "tokenize",
]
