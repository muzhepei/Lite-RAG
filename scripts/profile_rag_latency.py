# -*- coding: utf-8 -*-
"""RAG / 混合检索各阶段耗时剖析（在项目根目录或容器内运行）。"""
from __future__ import annotations

import json
import sys
import time
from typing import Any

from es2vec.cli.search_hybrid import hybrid_search, resolve_query_vector
from es2vec.core.config import (
    DEFAULT_INDEX_NAME,
    env_flag_true,
)
from es2vec.core.es_client import get_es
from es2vec.core.openai_compatible_chat import default_chat_client
from es2vec.core.openai_compatible_embedder import get_openai_compatible_embedder
from es2vec.core.rag_prompt import build_context_from_hits, build_rag_messages
from es2vec.core.rag_service import _search_request_from_rag
from es2vec.core.search_service import execute_hybrid_search


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def profile_search(query: str, index: str, k: int = 10) -> dict[str, Any]:
    es = get_es()
    use_api = env_flag_true("ES2VEC_USE_OPENAI_COMPATIBLE_EMBEDDING")
    out: dict[str, Any] = {"query": query, "index": index, "use_openai_embedding": use_api}

    t0 = time.perf_counter()
    t_emb = time.perf_counter()
    qv = resolve_query_vector(
        es,
        query,
        use_es_inference=False,
        use_openai_compatible_embedding=use_api,
        inference_id="",
        embedder=None,
        local_model="",
    )
    out["1_query_embedding_ms"] = _ms(t_emb)

    t_es = time.perf_counter()
    hybrid_search(
        es,
        index,
        query,
        use_openai_compatible_embedding=use_api,
        k=k,
        use_rrf=False,
    )
    out["2_es_hybrid_search_ms"] = _ms(t_es)
    out["2_note"] = "含 1 的 embedding（未单独去重）"

    t_emb2 = time.perf_counter()
    resolve_query_vector(
        es,
        query,
        use_es_inference=False,
        use_openai_compatible_embedding=use_api,
        inference_id="",
        embedder=None,
        local_model="",
    )
    out["1b_embedding_cached_ms"] = _ms(t_emb2)

    t_es_only = time.perf_counter()
    from es2vec.cli.search_hybrid import build_weighted_hybrid_query
    from es2vec.core.config import TEXT_FIELD, VECTOR_FIELD

    wq = build_weighted_hybrid_query(query, qv, text_field=TEXT_FIELD, vector_field=VECTOR_FIELD)
    es.search(index=index, size=k, query=wq, source=[TEXT_FIELD, VECTOR_FIELD])
    out["3_es_only_ms"] = _ms(t_es_only)

    out["search_total_ms"] = _ms(t0)
    return out


def profile_rag(query: str, index: str, top_k: int = 3) -> dict[str, Any]:
    out: dict[str, Any] = {"query": query, "index": index, "top_k": top_k}
    t_all = time.perf_counter()

    from es2vec.core.rag_service import RagRequest

    req = RagRequest(query=query, index=index, top_k=top_k)
    t_s = time.perf_counter()
    search_resp = execute_hybrid_search(_search_request_from_rag(req))
    out["1_hybrid_search_ms"] = _ms(t_s)

    hits = [h for h in search_resp.hits if isinstance(h, dict)]
    t_ctx = time.perf_counter()
    context, _ = build_context_from_hits(hits, max_chars=12000)
    out["2_build_context_ms"] = _ms(t_ctx)
    out["context_chars"] = len(context)

    messages = build_rag_messages(query=query, context=context, system_prompt="你是助手。")
    client = default_chat_client()
    out["chat_model"] = client.model

    t_llm = time.perf_counter()
    t_first: float | None = None
    parts: list[str] = []
    for chunk in client.stream_complete(messages, max_tokens=512):
        if t_first is None:
            t_first = time.perf_counter()
        parts.append(chunk)
    out["3_llm_stream_total_ms"] = _ms(t_llm)
    out["3a_llm_time_to_first_token_ms"] = (
        round((t_first - t_llm) * 1000, 1) if t_first else None
    )
    out["answer_chars"] = len("".join(parts))

    t_ser = time.perf_counter()
  # 模拟 debug SSE 的 retrieval 序列化体积
    from es2vec.core.search_service import search_response_to_dict

    payload = search_response_to_dict(search_resp)
    blob = json.dumps(payload, ensure_ascii=False)
    out["4_serialize_retrieval_ms"] = _ms(t_ser)
    out["retrieval_json_kb"] = round(len(blob.encode("utf-8")) / 1024, 1)

    out["rag_total_ms"] = _ms(t_all)
    return out


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else "刘备是谁"
    index = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_INDEX_NAME
    print("=== 仅混合检索 ===")
    print(json.dumps(profile_search(query, index), ensure_ascii=False, indent=2))
    print("\n=== RAG 全流程 ===")
    print(json.dumps(profile_rag(query, index), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
