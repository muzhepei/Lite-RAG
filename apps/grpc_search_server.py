# -*- coding: utf-8 -*-
"""
gRPC 混合检索服务。

https://grpc.io/docs/languages/python/

启动::

    python apps/grpc_search_server.py

环境变量::

    ES2VEC_GRPC_HOST=0.0.0.0
    ES2VEC_GRPC_PORT=50051
"""
from __future__ import annotations

import importlib.util
import os
import sys
from concurrent import futures
from pathlib import Path
from typing import Any


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

import grpc

from es2vec.core.config import DEFAULT_INDEX_NAME
from es2vec.core.search_service import (
    SearchExecutionError,
    SearchRequest,
    execute_hybrid_search,
    search_response_to_dict,
)
from es2vec.grpc_gen import es2vec_search_pb2, es2vec_search_pb2_grpc

_DEFAULT_HOST = os.environ.get("ES2VEC_GRPC_HOST", "0.0.0.0").strip() or "0.0.0.0"
_DEFAULT_PORT = int(os.environ.get("ES2VEC_GRPC_PORT", "50051"))
_MAX_WORKERS = int(os.environ.get("ES2VEC_GRPC_WORKERS", "4"))


def _proto_request_to_search(req: es2vec_search_pb2.HybridSearchRequest) -> SearchRequest:
    query = (req.query or "").strip()
    if not query:
        raise ValueError("查询不能为空")

    kwargs: dict[str, Any] = {
        "query": query,
        "index": req.index.strip() if req.index else DEFAULT_INDEX_NAME,
        "k": req.k if req.k > 0 else 10,
    }
    if req.match_field:
        kwargs["match_field"] = req.match_field.strip()
    if req.HasField("use_rrf"):
        kwargs["use_rrf"] = req.use_rrf
    if req.HasField("vector_weight"):
        kwargs["vector_weight"] = req.vector_weight
    if req.HasField("keyword_weight"):
        kwargs["keyword_weight"] = req.keyword_weight
    if req.HasField("kw_sat"):
        kwargs["kw_sat"] = req.kw_sat
    if req.HasField("keyword_norm_mode"):
        mode = req.keyword_norm_mode.strip().lower()
        if mode in ("raw", "saturation", "log1p"):
            kwargs["keyword_norm_mode"] = mode
    if req.HasField("name_rerank"):
        kwargs["name_rerank"] = req.name_rerank

    return SearchRequest(**kwargs)


def _dict_to_proto_response(data: dict[str, Any]) -> es2vec_search_pb2.HybridSearchResponse:
    resp = es2vec_search_pb2.HybridSearchResponse(
        query=str(data.get("query", "")),
        index=str(data.get("index", "")),
        total=int(data.get("total", 0)),
        returned=int(data.get("returned", 0)),
    )
    for h in data.get("hits") or []:
        if not isinstance(h, dict):
            continue
        hit = es2vec_search_pb2.SearchHit(
            rank=int(h.get("rank", 0)),
            id=str(h.get("id") or ""),
            score=float(h.get("score") or 0.0),
            text=str(h.get("text", "")),
            text_truncated=bool(h.get("text_truncated", False)),
        )
        chapter_id = h.get("chapter_id")
        if chapter_id is not None:
            hit.chapter_id = str(chapter_id)
        chunk_index = h.get("chunk_index")
        if chunk_index is not None:
            hit.chunk_index = int(chunk_index)
        resp.hits.append(hit)
    return resp


class SearchServiceServicer(es2vec_search_pb2_grpc.SearchServiceServicer):
    def HybridSearch(self, request, context):  # noqa: N802
        try:
            search_req = _proto_request_to_search(request)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return es2vec_search_pb2.HybridSearchResponse()

        try:
            result = execute_hybrid_search(search_req)
        except SearchExecutionError as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"检索失败: {exc}")
            return es2vec_search_pb2.HybridSearchResponse()

        return _dict_to_proto_response(search_response_to_dict(result))

    def Health(self, request, context):  # noqa: N802
        return es2vec_search_pb2.HealthResponse(status="ok")


def serve() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS))
    es2vec_search_pb2_grpc.add_SearchServiceServicer_to_server(
        SearchServiceServicer(), server
    )
    listen_addr = f"{_DEFAULT_HOST}:{_DEFAULT_PORT}"
    server.add_insecure_port(listen_addr)
    server.start()
    print(f"es2vec gRPC 检索: {listen_addr}")
    print(f"默认索引: {DEFAULT_INDEX_NAME}")
    server.wait_for_termination()


def main() -> int:
    try:
        serve()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
