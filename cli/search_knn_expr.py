# -*- coding: utf-8 -*-
"""
对若干词/短语的向量做加减后再做纯 kNN 检索（不与全文 RRF 混合）。

说明（重要）
------------
- 索引里的向量来自 ``passage:`` 前缀的句子编码；本脚本默认用 ``query:`` 侧编码参与加减，
  与 ``search_hybrid.py`` 里查询向量方式一致，便于和已建索引做近邻检索。
- 句向量模型（如 multilingual-e5）**不是** word2vec：``曹操 + 刘备`` 的向量和
  没有稳定的语言学意义，结果可当探索性参考，不要当作「词类比 =」的严格答案。
- 需要 **词级** ``most_similar``（如 Gensim）时，见 ``es2vec.three_kingdoms_ext.train_w2v``。
- 若索引时用了 ``--jieba``，本脚本只搜向量腿，不依赖 ``text_tokens``。

用法（在仓库根 ``ai\\14`` 下，与 ``index_corpus`` 相同）::

    cd C:\\Users\\Asus\\Desktop\\ai\\14\\es2vec
    python search_knn_expr.py --index es2vec_corpus --plus 曹操 --plus 刘备

带减法项（示意：A + B - C）::

    python search_knn_expr.py --index es2vec_corpus --plus 曹操 --plus 刘备 --minus 袁绍

用与文档相同的 passage 编码做加减（实验用）::

    python search_knn_expr.py --index es2vec_corpus --plus 曹操 --plus 刘备 --embed-mode passage

若索引用 ``--use-openai-compatible-embedding`` 构建，检索向量须同源::

    python search_knn_expr.py --index my_ix --plus 曹操 --use-openai-compatible-embedding
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
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

import sys
from pathlib import Path

from es2vec.core.es_client import get_es
from es2vec.core.config import DEFAULT_INDEX_NAME, TEXT_FIELD, VECTOR_FIELD
from es2vec.core.local_embedder import get_local_embedder
from es2vec.core.openai_compatible_embedder import get_openai_compatible_embedder


def _encode_one(
    embedder,
    text: str,
    *,
    embed_mode: str,
) -> np.ndarray:
    if embed_mode == "passage":
        return np.asarray(embedder.encode_passages([text], batch_size=1)[0], dtype=np.float64)
    if embed_mode == "query":
        return np.asarray(embedder.encode_queries([text], batch_size=1)[0], dtype=np.float64)
    raise ValueError("embed_mode 须为 query 或 passage")


def _openai_kw(
    base_url: str,
    api_key: str,
    embedding_model: str,
) -> tuple[str | None, str | None, str | None]:
    def _p(s: str) -> str | None:
        t = (s or "").strip()
        return t if t else None

    return _p(base_url), _p(api_key), _p(embedding_model)


def combine_expression_vector(
    pluses: Sequence[str],
    minuses: Sequence[str],
    *,
    embed_mode: str,
    local_model: str,
    use_openai_compatible_embedding: bool = False,
    openai_base_url: str = "",
    openai_api_key: str = "",
    openai_embedding_model: str = "",
) -> list[float]:
    if not pluses:
        raise ValueError("至少指定一个 --plus")
    if use_openai_compatible_embedding:
        bu, ak, md = _openai_kw(openai_base_url, openai_api_key, openai_embedding_model)
        emb = get_openai_compatible_embedder(base_url=bu, api_key=ak, model=md)
    else:
        emb = get_local_embedder(local_model.strip() or None)
    v = np.zeros(emb.embedding_dim, dtype=np.float64)
    for t in pluses:
        v += _encode_one(emb, t, embed_mode=embed_mode)
    for t in minuses:
        v -= _encode_one(emb, t, embed_mode=embed_mode)
    n = float(np.linalg.norm(v))
    if n <= 0:
        raise ValueError("组合后向量为零，无法检索")
    v /= n
    return v.astype(float).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description="向量加减（启发式）后 kNN 检索")
    ap.add_argument("--index", default=DEFAULT_INDEX_NAME)
    ap.add_argument("--plus", action="append", default=[], metavar="TEXT", help="相加项，可重复")
    ap.add_argument("--minus", action="append", default=[], metavar="TEXT", help="相减项，可重复")
    ap.add_argument(
        "--embed-mode",
        choices=("query", "passage"),
        default="query",
        help="query：与 hybrid 查询向量一致；passage：与索引文档向量前缀一致（实验）",
    )
    ap.add_argument("--local-model", default="", help="覆盖默认本地模型")
    ap.add_argument(
        "--use-openai-compatible-embedding",
        action="store_true",
        help="与 index_corpus 同源：用 OpenAI 兼容 /v1/embeddings 生成组合向量",
    )
    ap.add_argument("--openai-base-url", default="", help="覆盖 ES2VEC_OPENAI_BASE_URL")
    ap.add_argument(
        "--openai-api-key",
        default="",
        help="覆盖环境变量 DASHSCOPE_API_KEY（OpenAI 兼容 embeddings）",
    )
    ap.add_argument(
        "--openai-embedding-model",
        default="",
        help="覆盖 ES2VEC_OPENAI_EMBEDDING_MODEL",
    )
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--num-candidates", type=int, default=100)
    ap.add_argument("--json", action="store_true", help="打印原始 JSON")
    args = ap.parse_args()

    if not args.plus:
        raise SystemExit("请至少指定一次 --plus，例如: --plus 曹操 --plus 刘备")

    qv = combine_expression_vector(
        args.plus,
        args.minus,
        embed_mode=args.embed_mode,
        local_model=args.local_model,
        use_openai_compatible_embedding=args.use_openai_compatible_embedding,
        openai_base_url=args.openai_base_url,
        openai_api_key=args.openai_api_key,
        openai_embedding_model=args.openai_embedding_model,
    )

    es = get_es()
    label = " + ".join(args.plus)
    if args.minus:
        label += " - (" + " + ".join(args.minus) + ")"
    knn = {
        "field": VECTOR_FIELD,
        "query_vector": qv,
        "k": args.k,
        "num_candidates": args.num_candidates,
    }
    resp = es.search(
        index=args.index,
        knn=knn,
        size=args.k,
        source_includes=[TEXT_FIELD],
    )
    body = resp.body if hasattr(resp, "body") else resp
    if args.json:
        import json

        print(json.dumps(body, ensure_ascii=False, indent=2, default=str))
        return

    hits = (body.get("hits") or {}).get("hits") or []
    print(f"索引: {args.index}  组合: {label!r}  embed_mode={args.embed_mode}  命中: {len(hits)}")
    for i, h in enumerate(hits, start=1):
        src = h.get("_source") or {}
        text = src.get(TEXT_FIELD, "")
        score = h.get("_score")
        preview = str(text)[:300]
        print(f"{i}. score={score}\n   {preview}{'...' if len(str(text)) > 300 else ''}\n")


if __name__ == "__main__":
    main()
