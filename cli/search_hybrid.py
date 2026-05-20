# -*- coding: utf-8 -*-
"""
对查询字符串生成本地或 ES 端 query 向量，再合并全文与向量腿：

- **默认**：使用 ``retriever`` 里的 **RRF**（需集群许可证支持 RRF；否则会 403
  ``non-compliant for [Reciprocal Rank Fusion (RRF)]``）。
- **``--no-rrf``**（默认推荐本地/基础许可）：**加权 script_score**
  ``w_vec * sim + w_kwd * kwNorm``（权重见 ``--vec-weight`` / ``--kw-weight``）。
  ``sim`` 为余弦相似度（负值截断为 0，上界截断为 1）；``kwNorm`` 由 ``--kw-norm``
  决定：默认 **saturation** ``kw/(kw+kw_sat)`` 把 BM25 压到 (0,1)，避免高分长尾
  压死向量腿；``log1p`` 为对数压缩；``raw`` 为原始 ``_score``（与旧版行为一致）。
  内层为 ``bool``：``exists`` 向量 + 可选 ``match``，``minimum_should_match: 0``。

子能力：

  - standard：全文 match（默认字段 text，或索引时启用的 text_tokens）
  - knn：向量近邻

用法（在项目根目录；默认本地模型，无需 Inference 许可）::

    python cli/search_hybrid.py --index es2vec_corpus --q \"孙悟空与唐僧\"

若索引时使用了 --jieba，检索时需加::

    --match-field text_tokens

使用 ES Inference（需许可）::

    python cli/search_hybrid.py --index ... --q ... --use-es-inference --inference-id ...

OpenAI 兼容 Embeddings（与本地、ES Inference 三选一）::

    python cli/search_hybrid.py --index ... --q ... --use-openai-compatible-embedding

无 RRF 许可（遇 403）时::

    python cli/search_hybrid.py --index es2vec_corpus --q \"孙悟空与唐僧\" --no-rrf
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
import json
import sys
from pathlib import Path
from typing import Any, Final, Literal

# script_score 内 Painless 分支：勿与 ES 参数名冲突
_KW_NORM_RAW: Final[int] = 0
_KW_NORM_SATURATION: Final[int] = 1
_KW_NORM_LOG1P: Final[int] = 2

KeywordNormMode = Literal["raw", "saturation", "log1p"]

_KEYWORD_NORM_CODE: dict[KeywordNormMode, int] = {
    "raw": _KW_NORM_RAW,
    "saturation": _KW_NORM_SATURATION,
    "log1p": _KW_NORM_LOG1P,
}

import sys
from pathlib import Path

from elasticsearch import BadRequestError

from es2vec.core.es_client import get_es, get_index_vector_dims
from es2vec.core.config import (
    CHAPTER_ID_FIELD,
    CHUNK_INDEX_FIELD,
    DEFAULT_HYBRID_KW_SAT,
    DEFAULT_HYBRID_KW_WEIGHT,
    DEFAULT_HYBRID_VEC_WEIGHT,
    DEFAULT_INDEX_NAME,
    DEFAULT_INFERENCE_ID,
    DEFAULT_NAME_RERANK_POOL,
    TEXT_FIELD,
    TEXT_TOKEN_FIELD,
    VECTOR_FIELD,
    env_float,
    env_int,
    resolve_name_rerank,
)
from es2vec.core.doc_kind_filter import es_filter_chunks_only
from es2vec.core.search_rerank import apply_name_rerank_to_search_body
from es2vec.core.inference_utils import infer_text_embeddings
from es2vec.core.local_embedder import LocalEmbedder, get_local_embedder
from es2vec.core.openai_compatible_embedder import (
    ApiEmbedder,
    get_openai_compatible_embedder,
)


def build_rrf_retriever(
    query: str,
    query_vector: list[float],
    *,
    text_field: str = TEXT_FIELD,
    vector_field: str = VECTOR_FIELD,
    k: int = 10,
    num_candidates: int = 100,
    rank_window_size: int = 50,
    rank_constant: int = 60,
) -> dict[str, Any]:
    """构造 RRF retriever：standard match + knn（仅 chunk 文档）。"""
    chunk_filter = es_filter_chunks_only()
    knn: dict[str, Any] = {
        "field": vector_field,
        "query_vector": query_vector,
        "k": k,
        "num_candidates": num_candidates,
        "filter": chunk_filter,
    }
    standard_query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            text_field: {
                                "query": query,
                            }
                        }
                    }
                ],
                "filter": [chunk_filter],
            }
        }
    }
    return {
        "rrf": {
            "retrievers": [
                {"standard": standard_query},
                {"knn": knn},
            ],
            "rank_window_size": rank_window_size,
            "rank_constant": rank_constant,
        }
    }


def _keyword_norm_mode_to_code(mode: KeywordNormMode) -> int:
    return _KEYWORD_NORM_CODE[mode]


def _weighted_hybrid_script_source(vector_field: str) -> str:
    """
    Painless：子查询 BM25 得到 ``_score``，经 ``kwNorm`` 与 ``dense_vector`` 余弦
    ``sim`` 按权重线性组合。

    ``vector_field`` 仅允许来自映射常量（如 ``vector``），勿拼接用户输入。

    归一策略（由 ``params.kw_norm_code`` 选择）原因简述：
    - **sim 上界**：余弦在浮点误差下可能略大于 1，截断后与 ``kwNorm∈[0,1]`` 更可加权。
    - **saturation**：``kw/(kw+s)`` 单调有界，BM25 再高也趋近 1，量纲与 sim 接近，
      避免极少数极高 BM25 主导排序。
    - **log1p**：``min(1, log1p(kw)/cap)`` 压缩长尾，cap 大则整体更「平」。
    - **raw**：不做关键词侧变换，便于对照实验或与历史行为对齐。
    """
    # cosineSimilarity 第二参为字段名；与 index_corpus 中 similarity=cosine 一致
    return f"""
double sim = cosineSimilarity(params.query_vector, '{vector_field}');
if (sim < 0.0) {{
  sim = 0.0;
}}
sim = Math.min(1.0, sim);
double kw = _score;
if (kw < 0.0) {{
  kw = 0.0;
}}
double kwNorm;
int mode = params.kw_norm_code;
if (mode == {_KW_NORM_RAW}) {{
  kwNorm = kw;
}} else if (mode == {_KW_NORM_SATURATION}) {{
  kwNorm = kw / (kw + params.kw_sat);
}} else {{
  kwNorm = Math.min(1.0, Math.log1p(kw) / params.kw_log_cap);
}}
return params.w_vec * sim + params.w_kwd * kwNorm;
""".strip()


def build_weighted_hybrid_query(
    query: str,
    query_vector: list[float],
    *,
    text_field: str = TEXT_FIELD,
    vector_field: str = VECTOR_FIELD,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    keyword_norm_mode: KeywordNormMode = "saturation",
    kw_sat: float = 15.0,
    kw_log_cap: float = 10.0,
) -> dict[str, Any]:
    """
    单阶段 ``script_score``：子查询提供 BM25 ``_score``，经 ``keyword_norm_mode`` 变为
    ``kwNorm`` 后与 ``cosineSimilarity`` 按 ``w_vec`` / ``w_kwd`` 线性加权求和。

    子查询为 ``bool``：``filter`` 保证文档带向量；``should`` 为全文 ``match`` 且
    ``minimum_should_match: 0``，这样无关键词命中时仍可仅凭向量侧排序。
    """
    if kw_sat <= 0:
        raise ValueError("kw_sat 必须为正数（saturation 分母 kw+kw_sat）")
    if kw_log_cap <= 0:
        raise ValueError("kw_log_cap 必须为正数（log1p 归一除数）")
    inner: dict[str, Any] = {
        "bool": {
            "filter": [
                {"exists": {"field": vector_field}},
                es_filter_chunks_only(),
            ],
            "should": [
                {
                    "match": {
                        text_field: {
                            "query": query,
                        }
                    }
                }
            ],
            "minimum_should_match": 0,
        }
    }
    return {
        "script_score": {
            "query": inner,
            "script": {
                "source": _weighted_hybrid_script_source(vector_field),
                "params": {
                    "query_vector": query_vector,
                    "w_vec": float(vector_weight),
                    "w_kwd": float(keyword_weight),
                    "kw_norm_code": _keyword_norm_mode_to_code(keyword_norm_mode),
                    "kw_sat": float(kw_sat),
                    "kw_log_cap": float(kw_log_cap),
                },
            },
        }
    }


def _wrap_es_search_error(
    exc: BadRequestError,
    *,
    index: str,
    query_dims: int,
) -> Exception:
    """将 ES script_score / knn 的 runtime error 转为更易排查的提示。"""
    msg = str(exc).lower()
    if "runtime error" in msg or "search_phase_execution_exception" in msg:
        return ValueError(
            f"Elasticsearch 检索脚本执行失败（常见原因：查询向量维度 {query_dims} 与索引 "
            f"{index!r} 中 vector 字段不一致，或 match 字段不存在）。"
            "请执行 GET /{index}/_mapping 核对 vector.dims，并确认建索引与检索使用同一嵌入模型。"
        )
    return exc


def resolve_query_vector(
    es: Any,
    query: str,
    *,
    use_es_inference: bool,
    use_openai_compatible_embedding: bool,
    inference_id: str,
    embedder: LocalEmbedder | ApiEmbedder | None,
    local_model: str,
    openai_base_url: str = "",
    openai_api_key: str = "",
    openai_embedding_model: str = "",
) -> list[float]:
    """单条查询 -> 向量。"""
    if use_es_inference:
        return infer_text_embeddings(es, inference_id, [query], input_type="search")[0]
    if use_openai_compatible_embedding:

        def _ov(s: str) -> str | None:
            t = (s or "").strip()
            return t if t else None

        emb = embedder or get_openai_compatible_embedder(
            base_url=_ov(openai_base_url),
            api_key=_ov(openai_api_key),
            model=_ov(openai_embedding_model),
        )
        return emb.encode_queries([query], batch_size=1)[0]
    emb_local = embedder if isinstance(embedder, LocalEmbedder) else None
    emb = emb_local or get_local_embedder(local_model.strip() or None)
    return emb.encode_queries([query], batch_size=1)[0]


def hybrid_search(
    es: Any,
    index: str,
    query: str,
    *,
    use_es_inference: bool = False,
    use_openai_compatible_embedding: bool = False,
    inference_id: str = DEFAULT_INFERENCE_ID,
    embedder: LocalEmbedder | ApiEmbedder | None = None,
    local_model: str = "",
    openai_base_url: str = "",
    openai_api_key: str = "",
    openai_embedding_model: str = "",
    match_field: str = TEXT_FIELD,
    k: int = 10,
    num_candidates: int = 100,
    rank_window_size: int = 50,
    rank_constant: int = 60,
    use_rrf: bool = True,
    vector_weight: float | None = None,
    keyword_weight: float | None = None,
    keyword_norm_mode: KeywordNormMode = "saturation",
    kw_sat: float | None = None,
    kw_log_cap: float = 10.0,
    name_rerank: bool | None = None,
    name_rerank_pool: int | None = None,
) -> dict[str, Any]:
    w_vec = (
        vector_weight
        if vector_weight is not None
        else env_float("ES2VEC_VEC_WEIGHT", DEFAULT_HYBRID_VEC_WEIGHT)
    )
    w_kwd = (
        keyword_weight
        if keyword_weight is not None
        else env_float("ES2VEC_KW_WEIGHT", DEFAULT_HYBRID_KW_WEIGHT)
    )
    kw_sat_val = (
        kw_sat if kw_sat is not None else env_float("ES2VEC_KW_SAT", DEFAULT_HYBRID_KW_SAT)
    )
    do_name_rerank = resolve_name_rerank(query, name_rerank)
    rerank_pool = (
        name_rerank_pool
        if name_rerank_pool is not None
        else env_int("ES2VEC_NAME_RERANK_POOL", DEFAULT_NAME_RERANK_POOL)
    )
    fetch_size = max(k, rerank_pool) if do_name_rerank else k

    qv = resolve_query_vector(
        es,
        query,
        use_es_inference=use_es_inference,
        use_openai_compatible_embedding=use_openai_compatible_embedding,
        inference_id=inference_id,
        embedder=embedder,
        local_model=local_model,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        openai_embedding_model=openai_embedding_model,
    )
    index_dims = get_index_vector_dims(es, index, vector_field=VECTOR_FIELD)
    if index_dims is not None and len(qv) != index_dims:
        embed_src = (
            "OpenAI 兼容 API"
            if use_openai_compatible_embedding
            else ("ES Inference" if use_es_inference else "本地 SentenceTransformer")
        )
        raise ValueError(
            f"查询向量维度 {len(qv)} 与索引 {index!r} 中 {VECTOR_FIELD} 字段维度 "
            f"{index_dims} 不一致（当前嵌入来源：{embed_src}）。"
            "建索引与检索须使用同一套向量模型；若索引用本地 multilingual-e5-small（384 维），"
            "检索勿开启 ES2VEC_USE_OPENAI_COMPATIBLE_EMBEDDING；"
            "若索引用 --use-openai-compatible-embedding（qwen3-vl-embedding 常见 1024 维，默认 2560），"
            "检索须设 ES2VEC_USE_OPENAI_COMPATIBLE_EMBEDDING=1 并配置相同 API/模型，"
            "或加 --recreate 用当前模型重建索引。"
        )
    src_fields = [TEXT_FIELD, VECTOR_FIELD, CHAPTER_ID_FIELD, CHUNK_INDEX_FIELD]
    if match_field not in src_fields:
        src_fields.append(match_field)

    if not use_rrf:
        weighted_q = build_weighted_hybrid_query(
            query,
            qv,
            text_field=match_field,
            vector_field=VECTOR_FIELD,
            vector_weight=w_vec,
            keyword_weight=w_kwd,
            keyword_norm_mode=keyword_norm_mode,
            kw_sat=kw_sat_val,
            kw_log_cap=kw_log_cap,
        )
        try:
            resp = es.search(
                index=index,
                size=fetch_size,
                query=weighted_q,
                source=src_fields,
            )
        except BadRequestError as exc:
            raise _wrap_es_search_error(exc, index=index, query_dims=len(qv)) from exc
    else:
        retriever = build_rrf_retriever(
            query,
            qv,
            text_field=match_field,
            k=fetch_size,
            num_candidates=max(num_candidates, fetch_size),
            rank_window_size=rank_window_size,
            rank_constant=rank_constant,
        )
        try:
            resp = es.search(
                index=index,
                size=fetch_size,
                retriever=retriever,
                source=src_fields,
            )
        except BadRequestError as exc:
            raise _wrap_es_search_error(exc, index=index, query_dims=len(qv)) from exc

    if do_name_rerank:
        body = resp.body if hasattr(resp, "body") else resp
        if isinstance(body, dict):
            apply_name_rerank_to_search_body(
                body,
                query,
                text_field=match_field,
                k=k,
                rerank_pool=rerank_pool,
            )

    return resp


def main() -> None:
    ap = argparse.ArgumentParser(
        description="混合检索：默认 RRF；--no-rrf 为向量+关键词线性加权（免 RRF 许可）"
    )
    ap.add_argument("--index", default=DEFAULT_INDEX_NAME)
    ap.add_argument(
        "--use-es-inference",
        action="store_true",
        help="使用 ES Inference API 生成查询向量（需许可证）",
    )
    ap.add_argument(
        "--use-openai-compatible-embedding",
        action="store_true",
        help="使用 OpenAI 兼容 /v1/embeddings 生成查询向量（与 --use-es-inference 互斥）",
    )
    ap.add_argument(
        "--openai-base-url",
        default="",
        help="覆盖 ES2VEC_OPENAI_BASE_URL",
    )
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
    ap.add_argument("--inference-id", default=DEFAULT_INFERENCE_ID)
    ap.add_argument(
        "--local-model",
        default="",
        help="本地 SentenceTransformer 模型 id 或路径（默认环境变量或 intfloat/multilingual-e5-small）",
    )
    ap.add_argument(
        "--match-field",
        default=TEXT_FIELD,
        help=f"全文 match 字段，默认 {TEXT_FIELD}；jieba 索引时用 {TEXT_TOKEN_FIELD}",
    )
    ap.add_argument("--q", required=True, help="查询文本")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--num-candidates", type=int, default=100)
    ap.add_argument("--rank-window-size", type=int, default=50)
    ap.add_argument("--rank-constant", type=int, default=60)
    ap.add_argument(
        "--no-rrf",
        action="store_true",
        help="不使用 RRF；改用 script_score：w_vec*sim + w_kwd*kwNorm（见 --kw-norm）",
    )
    ap.add_argument(
        "--vec-weight",
        type=float,
        default=0.7,
        dest="vec_weight",
        metavar="W",
        help="加权混合时向量（余弦相似度）权重，默认 0.7",
    )
    ap.add_argument(
        "--kw-weight",
        type=float,
        default=0.3,
        dest="kw_weight",
        metavar="W",
        help="加权混合时关键词腿权重（乘在 kwNorm 上），默认 0.3",
    )
    ap.add_argument(
        "--kw-norm",
        choices=("raw", "saturation", "log1p"),
        default="saturation",
        dest="kw_norm",
        help="仅 --no-rrf：BM25 归一方式。saturation=kw/(kw+kw_sat)（默认）；"
        "log1p=min(1,log1p(kw)/cap)；raw=原始 _score（旧行为）",
    )
    ap.add_argument(
        "--kw-sat",
        type=float,
        default=15.0,
        dest="kw_sat",
        metavar="S",
        help="仅 saturation：越大则高分 BM25 越难贴近 1，关键词腿整体越「平」，默认 15",
    )
    ap.add_argument(
        "--kw-log-cap",
        type=float,
        default=10.0,
        dest="kw_log_cap",
        metavar="C",
        help="仅 log1p：除数越大同 BM25 下 kwNorm 越小，默认 10",
    )
    ap.add_argument(
        "--name-rerank",
        action="store_true",
        help="对命中按查询词在正文中的密度二阶段重排（默认：≤4 字且无空格时自动开启）",
    )
    ap.add_argument(
        "--no-name-rerank",
        action="store_true",
        help="关闭密度重排（覆盖 ES2VEC_NAME_RERANK 与自动规则）",
    )
    ap.add_argument(
        "--name-rerank-pool",
        type=int,
        default=None,
        metavar="N",
        help="重排候选池大小，默认 50（环境变量 ES2VEC_NAME_RERANK_POOL）",
    )
    ap.add_argument("--json", action="store_true", help="输出原始 JSON（便于调试）")
    # ap.parse_args()作用是将命令行参数解析为对象，并返回这个对象
    args = ap.parse_args()

    if args.use_es_inference and args.use_openai_compatible_embedding:
        raise SystemExit("--use-es-inference 与 --use-openai-compatible-embedding 不能同时使用")

    name_rerank: bool | None = None
    if args.name_rerank:
        name_rerank = True
    elif args.no_name_rerank:
        name_rerank = False

    es = get_es()
    resp = hybrid_search(
        es,
        args.index,
        args.q,
        use_es_inference=args.use_es_inference,
        use_openai_compatible_embedding=args.use_openai_compatible_embedding,
        inference_id=args.inference_id,
        local_model=args.local_model,
        openai_base_url=args.openai_base_url,
        openai_api_key=args.openai_api_key,
        openai_embedding_model=args.openai_embedding_model,
        match_field=args.match_field,
        k=args.k,
        num_candidates=args.num_candidates,
        rank_window_size=args.rank_window_size,
        rank_constant=args.rank_constant,
        use_rrf=not args.no_rrf,
        vector_weight=args.vec_weight,
        keyword_weight=args.kw_weight,
        keyword_norm_mode=args.kw_norm,
        kw_sat=args.kw_sat,
        kw_log_cap=args.kw_log_cap,
        name_rerank=name_rerank,
        name_rerank_pool=args.name_rerank_pool,
    )
    body = resp.body if hasattr(resp, "body") else resp
    if args.json:
        print(json.dumps(body, ensure_ascii=False, indent=2, default=str))
        return

    hits = (body.get("hits") or {}).get("hits") or []
    print(f"索引: {args.index}  查询: {args.q!r}  命中数: {len(hits)}")
    for i, h in enumerate(hits, start=1):
        src = h.get("_source") or {}
        text = src.get(TEXT_FIELD, "")
        score = h.get("_score")
        print(f"{i}. score={score}\t{text[:200]}{'...' if len(str(text)) > 200 else ''}")


if __name__ == "__main__":
    main()
