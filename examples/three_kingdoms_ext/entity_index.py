# -*- coding: utf-8 -*-
"""
为若干「主要人物」聚合包含其人名的 chunk 文本，编码后写入独立 ES 索引，便于人物级向量近邻。

字段：``character``（keyword）、``profile_text``（text）、``vector``（dense_vector，与主索引同源编码）。

用法::

    cd C:\\Users\\Asus\\Desktop\\ai\\14\\es2vec
    python -m es2vec.three_kingdoms_ext.entity_index \\
        --input three_kingdoms_ext/out/three_kingdoms_chunks.jsonl \\
        --index es2vec_three_kingdoms_entities --recreate
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
from pathlib import Path
from typing import Any, Iterator

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

from elasticsearch.helpers import bulk

from es2vec.core.es_client import ensure_index, get_es
from es2vec.core.config import (
    CHARACTER_NAME_FIELD,
    CHARACTER_PROFILE_FIELD,
    DEFAULT_VECTOR_DIMS,
    VECTOR_FIELD,
)
from es2vec.core.local_embedder import get_local_embedder
from es2vec.core.openai_compatible_embedder import (
    ApiEmbedder,
    get_openai_compatible_embedder,
)
from es2vec.core.synonym_api import build_index_settings_with_synonyms
from es2vec.core.inference_utils import cluster_has_smartcn, infer_text_embeddings


def _default_characters_file() -> Path:
    return Path(__file__).resolve().parent / "data" / "major_characters.txt"


def load_character_names(path: Path) -> list[str]:
    names: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                names.append(s)
    return names


def _iter_chunk_texts(jsonl: Path) -> Iterator[str]:
    with open(jsonl, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("text")
            if isinstance(t, str) and t.strip():
                yield t.strip()


def build_profiles(
    jsonl: Path,
    names: list[str],
    *,
    max_profile_chars: int,
) -> dict[str, str]:
    """每人聚合若干包含该人名的片段，总长不超过 max_profile_chars。"""
    profiles: dict[str, list[str]] = {n: [] for n in names}
    lengths: dict[str, int] = {n: 0 for n in names}
    for chunk in _iter_chunk_texts(jsonl):
        for n in names:
            if n not in chunk:
                continue
            if lengths[n] >= max_profile_chars:
                continue
            take = chunk
            room = max_profile_chars - lengths[n]
            if len(take) > room:
                take = take[:room]
            profiles[n].append(take)
            lengths[n] += len(take)
    return {n: "\n\n".join(parts).strip() for n, parts in profiles.items() if parts}


def _entity_mapping_props(vector_dims: int) -> dict[str, Any]:
    return {
        CHARACTER_NAME_FIELD: {"type": "keyword"},
        CHARACTER_PROFILE_FIELD: {"type": "text", "analyzer": "standard"},
        VECTOR_FIELD: {
            "type": "dense_vector",
            "dims": vector_dims,
            "index": True,
            "similarity": "cosine",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="人物 profile 向量 -> ES 索引")
    ap.add_argument("--input", type=Path, required=True, help="chunk JSONL")
    ap.add_argument(
        "--index",
        default="es2vec_three_kingdoms_entities",
        help="目标索引名",
    )
    ap.add_argument(
        "--characters-file",
        type=Path,
        default=None,
        help="人物名单 UTF-8 一行一名；默认 three_kingdoms_ext/data/major_characters.txt",
    )
    ap.add_argument("--max-profile-chars", type=int, default=28_000, help="每人 profile 最大字符数")
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--local-model", default="")
    ap.add_argument(
        "--use-openai-compatible-embedding",
        action="store_true",
    )
    ap.add_argument("--openai-base-url", default="")
    ap.add_argument("--openai-api-key", default="")
    ap.add_argument("--openai-embedding-model", default="")
    ap.add_argument("--use-es-inference", action="store_true")
    ap.add_argument("--inference-id", default="es2vec_multilingual_e5")
    ap.add_argument("--vector-dims", type=int, default=0)
    ap.add_argument("--no-smartcn", action="store_true")
    args = ap.parse_args()

    if args.use_es_inference and args.use_openai_compatible_embedding:
        raise SystemExit("--use-es-inference 与 --use-openai-compatible-embedding 不能同时使用")

    if not args.input.is_file():
        raise SystemExit(f"文件不存在: {args.input}")

    cf = args.characters_file or _default_characters_file()
    if not cf.is_file():
        raise SystemExit(f"人物名单不存在: {cf}")
    names = load_character_names(cf)
    if not names:
        raise SystemExit("人物名单为空")

    profiles = build_profiles(
        args.input,
        names,
        max_profile_chars=int(args.max_profile_chars),
    )
    rows = [(n, profiles[n]) for n in names if n in profiles and profiles[n]]
    if not rows:
        raise SystemExit("没有任何人物在语料中命中；请检查名单与 JSONL")

    es = get_es()
    use_smartcn = (not args.no_smartcn) and cluster_has_smartcn(es)
    settings, text_idx, text_search, _, _ = build_index_settings_with_synonyms(
        use_smartcn=use_smartcn,
        synonyms_set_id=None,
        include_jieba_token_field=False,
    )
    # profile 使用 standard；settings 里主要为 shards，analyzer 对 standard 字段无强依赖
    _ = text_idx, text_search

    embedder: Any | None = None
    openai_embedder: ApiEmbedder | None = None

    if args.use_es_inference:
        first_text = rows[0][1]
        first_vecs = infer_text_embeddings(
            es, args.inference_id, [first_text], input_type="ingest"
        )
        dims = args.vector_dims or len(first_vecs[0])
    elif args.use_openai_compatible_embedding:

        def _p(s: str) -> str | None:
            t = (s or "").strip()
            return t if t else None

        openai_embedder = get_openai_compatible_embedder(
            base_url=_p(args.openai_base_url),
            api_key=_p(args.openai_api_key),
            model=_p(args.openai_embedding_model),
        )
        dims = args.vector_dims or openai_embedder.embedding_dim or DEFAULT_VECTOR_DIMS
        dims = len(openai_embedder.encode_passages([rows[0][1]], batch_size=1)[0])
    else:
        embedder = get_local_embedder((args.local_model or "").strip() or None)
        dims = args.vector_dims or embedder.embedding_dim or DEFAULT_VECTOR_DIMS
        dims = len(embedder.encode_passages([rows[0][1]], batch_size=1)[0])

    props = _entity_mapping_props(dims)
    ensure_index(
        es,
        args.index,
        properties=props,
        settings=settings,
        recreate=args.recreate,
    )

    def encode_batch(texts: list[str]) -> list[list[float]]:
        if args.use_es_inference:
            return infer_text_embeddings(es, args.inference_id, texts, input_type="ingest")
        if args.use_openai_compatible_embedding:
            assert openai_embedder is not None
            return openai_embedder.encode_passages(
                texts, batch_size=min(len(texts), args.batch_size)
            )
        assert embedder is not None
        return embedder.encode_passages(texts, batch_size=min(len(texts), args.batch_size))

    acts: list[dict[str, Any]] = []
    offset = 0
    pairs = list(rows)
    while offset < len(pairs):
        batch = pairs[offset : offset + args.batch_size]
        offset += len(batch)
        texts = [t for _, t in batch]
        vecs = encode_batch(texts)
        for (name, text), vec in zip(batch, vecs, strict=True):
            acts.append(
                {
                    "_op_type": "index",
                    "_index": args.index,
                    "_id": name,
                    "_source": {
                        CHARACTER_NAME_FIELD: name,
                        CHARACTER_PROFILE_FIELD: text,
                        VECTOR_FIELD: vec,
                    },
                }
            )

    ok, err = bulk(
        es.options(request_timeout=300),
        acts,
        chunk_size=min(50, len(acts)),
    )
    es.indices.refresh(index=args.index)
    elist = err or []
    print(f"实体索引完成: {args.index}，成功 {ok} 条，错误 {len(elist)} 条")
    if elist:
        print("错误示例:", elist[:2])


if __name__ == "__main__":
    main()