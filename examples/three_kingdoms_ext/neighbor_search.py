# -*- coding: utf-8 -*-
"""
人物相关 chunk 检索：对查询做 ``query:`` 向量后 kNN，并可从命中片段中用 jieba 词性聚合「人名」共现。

另支持对人物索引（``entity_index`` 写入）做 kNN，直接返回最相近的人物名。

用法::

    cd C:\\Users\\Asus\\Desktop\\ai\\14\\es2vec
    python -m es2vec.three_kingdoms_ext.neighbor_search chunks \\
        --index es2vec_three_kingdoms_chunks --query 曹操 --k 8

    python -m es2vec.three_kingdoms_ext.neighbor_search chunks \\
        --index es2vec_three_kingdoms_chunks --query 曹操 --aggregate-names --name-hits 15

    python -m es2vec.three_kingdoms_ext.neighbor_search entities \\
        --index es2vec_three_kingdoms_entities --query 曹操 --k 10
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _bootstrap_es2vec_path() -> None:
    p = Path(__file__).resolve().parent
    for _ in range(12):
        script = p / "core" / "_install_path.py"
        if p.name == "es2vec" and script.is_file():
            spec = importlib.util.spec_from_file_location("_es2vec_install_path", script)
            mod = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(mod)
            mod.install(__file__)
            return
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError("无法定位 es2vec 包（需要 es2vec/core/_install_path.py）")


_bootstrap_es2vec_path()


import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

_ip = Path(__file__).resolve().parent
for _ in range(12):
    _script = _ip / "_install_path.py"
    if _ip.name == "es2vec" and _script.is_file():
        _spec = importlib.util.spec_from_file_location("es2vec_install_path", _script)
        _mod = importlib.util.module_from_spec(_spec)
        assert _spec.loader is not None
        _spec.loader.exec_module(_mod)
        _mod.install(__file__)
        break
    if _ip.parent == _ip:
        raise SystemExit("无法定位 es2vec/_install_path.py")
    _ip = _ip.parent
else:
    raise SystemExit("无法定位 es2vec/_install_path.py")

from es2vec.core.es_client import get_es
from es2vec.core.config import (
    CHAPTER_ID_FIELD,
    CHUNK_INDEX_FIELD,
    CHARACTER_NAME_FIELD,
    CHARACTER_PROFILE_FIELD,
    DEFAULT_INDEX_NAME,
    TEXT_FIELD,
    VECTOR_FIELD,
)
from es2vec.core.local_embedder import get_local_embedder
from es2vec.core.openai_compatible_embedder import get_openai_compatible_embedder


def _jieba_nr_counter(texts: Iterable[str], *, exclude: frozenset[str]) -> Counter[str]:
    """从文本集合中统计 jieba 标注为专有名词（nr / nz）的片段。"""
    import jieba.posseg as pseg

    c: Counter[str] = Counter()
    for t in texts:
        for w, f in pseg.cut(t):
            w = w.strip()
            if len(w) < 2:
                continue
            if f not in ("nr", "nz"):
                continue
            if w in exclude:
                continue
            c[w] += 1
    return c


def _encode_query_vec(
    q: str,
    *,
    local_model: str,
    use_openai: bool,
    openai_base_url: str,
    openai_api_key: str,
    openai_embedding_model: str,
) -> list[float]:
    if use_openai:

        def _p(s: str) -> str | None:
            t = (s or "").strip()
            return t if t else None

        emb = get_openai_compatible_embedder(
            base_url=_p(openai_base_url),
            api_key=_p(openai_api_key),
            model=_p(openai_embedding_model),
        )
        return emb.encode_queries([q], batch_size=1)[0]
    emb = get_local_embedder(local_model.strip() or None)
    return emb.encode_queries([q], batch_size=1)[0]


def _knn_search(
    *,
    index: str,
    query_vector: list[float],
    k: int,
    num_candidates: int,
    source_fields: list[str],
) -> list[dict[str, Any]]:
    es = get_es()
    knn = {
        "field": VECTOR_FIELD,
        "query_vector": query_vector,
        "k": k,
        "num_candidates": num_candidates,
    }
    resp = es.search(
        index=index,
        knn=knn,
        size=k,
        source_includes=source_fields,
    )
    body = resp.body if hasattr(resp, "body") else resp
    return (body.get("hits") or {}).get("hits") or []


def cmd_chunks(args: argparse.Namespace) -> None:
    qv = _encode_query_vec(
        args.query,
        local_model=args.local_model,
        use_openai=bool(args.use_openai_compatible_embedding),
        openai_base_url=args.openai_base_url,
        openai_api_key=args.openai_api_key,
        openai_embedding_model=args.openai_embedding_model,
    )
    src = [TEXT_FIELD, CHAPTER_ID_FIELD, CHUNK_INDEX_FIELD]
    hits = _knn_search(
        index=args.index,
        query_vector=qv,
        k=args.k,
        num_candidates=args.num_candidates,
        source_fields=src,
    )
    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=2, default=str))
        return
    print(f"索引={args.index!r}  query={args.query!r}  命中={len(hits)}")
    hit_texts: list[str] = []
    for i, h in enumerate(hits, start=1):
        src_d = h.get("_source") or {}
        text = str(src_d.get(TEXT_FIELD, ""))
        hit_texts.append(text)
        ch = src_d.get(CHAPTER_ID_FIELD)
        ci = src_d.get(CHUNK_INDEX_FIELD)
        meta = f" chapter={ch!r} chunk={ci}" if ch is not None or ci is not None else ""
        prev = text[:280] + ("..." if len(text) > 280 else "")
        print(f"{i}. score={h.get('_score')}{meta}\n   {prev}\n")
    if args.aggregate_names:
        exclude = frozenset({args.query.strip(), args.query.strip()[:2]})
        topn = max(1, int(args.name_hits))
        counter = _jieba_nr_counter(hit_texts[:topn], exclude=exclude)
        print("--- jieba nr/nz 共现（前若干条命中内统计，排除查询词）---")
        for name, cnt in counter.most_common(25):
            print(f"  {name}\t{cnt}")


def cmd_entities(args: argparse.Namespace) -> None:
    qv = _encode_query_vec(
        args.query,
        local_model=args.local_model,
        use_openai=bool(args.use_openai_compatible_embedding),
        openai_base_url=args.openai_base_url,
        openai_api_key=args.openai_api_key,
        openai_embedding_model=args.openai_embedding_model,
    )
    hits = _knn_search(
        index=args.index,
        query_vector=qv,
        k=args.k,
        num_candidates=args.num_candidates,
        source_fields=[CHARACTER_NAME_FIELD, CHARACTER_PROFILE_FIELD],
    )
    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=2, default=str))
        return
    print(f"人物索引={args.index!r}  query={args.query!r}  命中={len(hits)}")
    for i, h in enumerate(hits, start=1):
        src_d = h.get("_source") or {}
        name = src_d.get(CHARACTER_NAME_FIELD, "")
        prof = str(src_d.get(CHARACTER_PROFILE_FIELD, ""))[:200]
        print(f"{i}. {name!r} score={h.get('_score')}\n   {prof}...\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="三国：chunk / 人物向量近邻检索")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_chunks = sub.add_parser("chunks", help="在 chunk 索引上做 kNN，可选 jieba 人名聚合")
    p_chunks.add_argument("--index", default=DEFAULT_INDEX_NAME)
    p_chunks.add_argument("--query", required=True)
    p_chunks.add_argument("--k", type=int, default=10)
    p_chunks.add_argument("--num-candidates", type=int, default=100)
    p_chunks.add_argument("--local-model", default="")
    p_chunks.add_argument(
        "--use-openai-compatible-embedding",
        action="store_true",
    )
    p_chunks.add_argument("--openai-base-url", default="")
    p_chunks.add_argument("--openai-api-key", default="")
    p_chunks.add_argument("--openai-embedding-model", default="")
    p_chunks.add_argument("--aggregate-names", action="store_true", help="对前 N 条命中做 jieba nr/nz 统计")
    p_chunks.add_argument("--name-hits", type=int, default=12, help="参与人名统计的命中条数上限")
    p_chunks.add_argument("--json", action="store_true")
    p_chunks.set_defaults(func=cmd_chunks)

    p_ent = sub.add_parser("entities", help="在人物 profile 索引上做 kNN")
    p_ent.add_argument("--index", default="es2vec_three_kingdoms_entities")
    p_ent.add_argument("--query", required=True)
    p_ent.add_argument("--k", type=int, default=10)
    p_ent.add_argument("--num-candidates", type=int, default=100)
    p_ent.add_argument("--local-model", default="")
    p_ent.add_argument("--use-openai-compatible-embedding", action="store_true")
    p_ent.add_argument("--openai-base-url", default="")
    p_ent.add_argument("--openai-api-key", default="")
    p_ent.add_argument("--openai-embedding-model", default="")
    p_ent.add_argument("--json", action="store_true")
    p_ent.set_defaults(func=cmd_entities)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
