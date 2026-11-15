"""构建搜索侧 FAISS 向量索引（只读现有 BM25 小样本索引）。

    python -m recsys.search.build_vector_index
    python -m recsys.search.build_vector_index --force
    python -m recsys.search.build_vector_index --check

输入：``artifacts/search/kuasar_bm25/index.json``（Search 1 产物，120 物品）。
输出：``artifacts/search/kuasar_vector/``

    vector/doc_ids.json   物品 ID（字符串，上游存法）
    vector/meta.json      backend / 编码器配置 / 维度 / 配置哈希（最后写入）
    vector/ann.faiss      FAISS IndexFlatIP payload
    manifest.json         来源、编码器、后端、sha256 与计时

不训练、不微调、不扫原始 CSV；BM25 索引与样本保持只读。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import tomllib
from pathlib import Path
from typing import Any

from recsys.search.bm25 import IndexedItem, load_index
from recsys.search.encoders import build_encoder
from recsys.search.upstream import UPSTREAM_SHORT, faiss_version
from recsys.search.vector_index import (
    FAISS_BACKEND,
    NUMPY_BACKEND,
    SearchVectorIndex,
    VectorIndexError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "search" / "kuasar_search_phase2.toml"


class BuildError(RuntimeError):
    pass


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        config = tomllib.load(file)
    for section in ("search", "encoder", "vector"):
        if section not in config:
            raise BuildError(f"配置缺少 [{section}]：{path}")
    return config


def paths_for(config: dict[str, Any]) -> dict[str, Path]:
    search = config["search"]
    root = _resolve(search["artifact_root"])
    return {
        "bm25_index": _resolve(search["bm25_index"]),
        "bm25_sample": _resolve(search["bm25_sample"]),
        "root": root,
        "vector_dir": root / search["vector_dir"],
        "manifest": root / search["manifest_file"],
    }


def build_assets(
    config: dict[str, Any], *, force: bool = False, persist: bool = True
) -> dict[str, Any]:
    paths = paths_for(config)
    bm25_path = paths["bm25_index"]
    if not bm25_path.exists():
        raise BuildError(
            f"找不到 BM25 索引 {_display(bm25_path)}；先运行 python -m recsys.search.prepare"
        )
    if not force and paths["manifest"].exists() and paths["vector_dir"].exists():
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if manifest.get("bm25_index", {}).get("sha256") == _sha256_file(bm25_path):
            print(f"[reuse] {_display(paths['vector_dir'])} 已是最新（--force 可重建）")
            return manifest
        print("[stale] BM25 索引已变化，重建搜索向量索引", flush=True)

    encoder = build_encoder(config["encoder"])
    backend = str(config["vector"].get("backend", FAISS_BACKEND)).strip().lower()
    if backend not in (FAISS_BACKEND, NUMPY_BACKEND):
        raise BuildError(f"未知向量后端 {backend!r}；可选 faiss / numpy")

    started = time.perf_counter()
    bm25 = load_index(bm25_path)
    items: list[IndexedItem] = list(bm25.items)
    print(
        f"[1/2] 读取 BM25 索引 {_display(bm25_path)}：{len(items)} 物品 / "
        f"{bm25.vocab_size} token；编码器 {encoder.name} dim={encoder.dim}",
        flush=True,
    )
    try:
        index = SearchVectorIndex.build(
            items,
            encoder,
            cache_dir=paths["vector_dir"] if persist else None,
            backend=backend,
            extra_meta={
                "dataset": config["search"].get("dataset", "kuasar"),
                "source": "bm25_index",
                "bm25_index_sha256": _sha256_file(bm25_path),
                "upstream": f"skim-search-agent@{UPSTREAM_SHORT}",
            },
            persist=persist,
        )
    except VectorIndexError as error:
        raise BuildError(str(error)) from error
    elapsed = time.perf_counter() - started

    payload = paths["vector_dir"] / "ann.faiss"
    if backend == NUMPY_BACKEND:
        payload = paths["vector_dir"] / "embeddings.npy"
    manifest = {
        "format": "recsys.search.vector_manifest.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(elapsed, 3),
        "source": {
            "bm25_index": _display(bm25_path),
            "num_items": bm25.num_docs,
            "scanned_raw_csv": False,
            "trained": False,
        },
        "bm25_index": {
            "path": _display(bm25_path),
            "sha256": _sha256_file(bm25_path),
            "num_docs": bm25.num_docs,
            "vocab_size": bm25.vocab_size,
        },
        "encoder": encoder.describe(),
        "encoder_config_hash": encoder.config_hash(),
        "backend": {
            "requested": backend,
            "actual": index.backend_actual,
            "faiss_active": index.faiss_active,
            "faiss_version": faiss_version(),
        },
        "index": {
            "num_items": index.num_items,
            "dim": index.dim,
            "dir": _display(paths["vector_dir"]),
            "payload": _display(payload),
            "payload_sha256": _sha256_file(payload) if payload.exists() else None,
            "payload_bytes": payload.stat().st_size if payload.exists() else None,
        },
        "component": f"skim-search-agent@{UPSTREAM_SHORT} vector_index + faiss.IndexFlatIP",
        "notes": [
            "只读 Search 1 的 BM25 索引，未扫描原始 CSV、未重建 BM25。",
            "查询与文档使用同一套 token 编码配置（同一编码器实例）。",
            "搜索向量索引独立于推荐侧 FAISS 索引。",
        ],
    }
    if persist:
        paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
        paths["manifest"].write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[2/2] 写入 {_display(paths['vector_dir'])}（{index.backend_actual}）")
        print(f"[ok] manifest：{_display(paths['manifest'])}")
    return manifest


def check_assets(config: dict[str, Any]) -> int:
    paths = paths_for(config)
    if not paths["manifest"].exists():
        print(f"[error] 找不到 {_display(paths['manifest'])}；先运行 build_vector_index")
        return 2
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    encoder = build_encoder(config["encoder"])
    try:
        index = SearchVectorIndex.load(paths["vector_dir"], encoder)
    except (VectorIndexError, OSError, ValueError) as error:
        print(f"[error] 搜索向量索引重载失败：{error}")
        return 2
    sample_path = paths["bm25_sample"]
    tokens: list[int] = []
    if sample_path.exists():
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        for session in sample.get("sessions", []):
            keyword = [int(token) for token in session.get("keyword") or []]
            if keyword:
                tokens = keyword
                break
    hits = index.search_scored(tokens, top_k=3) if tokens else []
    print(
        f"[ok] {_display(paths['vector_dir'])}：{index.num_items} 物品 / dim {index.dim} / "
        f"backend {index.backend_actual}；查询 {tokens[:4]} → "
        + (", ".join(f"{hit.item_id}:{hit.score:.4f}" for hit in hits) or "无结果")
    )
    if manifest.get("backend", {}).get("actual") != index.backend_actual:
        print("[fail] manifest 记录的后端与重载结果不一致")
        return 1
    if not index.faiss_active:
        print("[fail] 重载后的索引没有走 FAISS")
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--force", action="store_true", help="忽略已有产物，重建向量索引")
    parser.add_argument("--check", action="store_true", help="只校验索引可重载并查询")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.check:
        return check_assets(config)
    try:
        manifest = build_assets(config, force=args.force)
    except (BuildError, VectorIndexError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"[error] 构建失败：{error}")
        return 2
    print(
        json.dumps(
            {
                "items": manifest["index"]["num_items"],
                "dim": manifest["index"]["dim"],
                "backend": manifest["backend"]["actual"],
                "faiss_version": manifest["backend"]["faiss_version"],
                "encoder": manifest["encoder"]["encoder"],
                "elapsed_s": manifest["elapsed_s"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
