# -*- coding: utf-8 -*-
"""
Web 混合检索服务：提供 REST API 与静态搜索页面。

在项目根目录启动::

    pip install fastapi uvicorn
    python apps/web_search_server.py

浏览器打开 http://127.0.0.1:8765/

环境变量（可选）::

    ES2VEC_WEB_HOST=0.0.0.0
    ES2VEC_WEB_PORT=8765
    ES2VEC_INDEX=es2vec_corpus
    ES2VEC_MATCH_FIELD=text
    ES2VEC_VEC_WEIGHT=0.85
    ES2VEC_KW_WEIGHT=0.15
    ES2VEC_KW_SAT=25
    ES2VEC_NAME_RERANK=1
    ES2VEC_NAME_RERANK_AUTO=1

RAG 问答需配置对话 API（与 Embeddings 共用 Key）::

    DASHSCOPE_API_KEY=...
    ES2VEC_CHAT_MODEL=qwen-turbo
"""
from __future__ import annotations

import importlib.util
import os
import sys
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

# https://fastapi.tiangolo.com/
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from es2vec.core.config import DEFAULT_INDEX_NAME, RAG_DEFAULT_TOP_K, TEXT_TOKEN_FIELD
from es2vec.core.rag_service import (
    RagExecutionError,
    RagRequest,
    RagResponse,
    execute_rag,
    rag_response_to_dict,
    rag_with_retrieval_payload,
)
from es2vec.core.search_service import (
    SearchExecutionError,
    SearchRequest,
    SearchResponse,
    _default_match_field,
    execute_hybrid_search,
    search_response_to_dict,
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_DEFAULT_HOST = os.environ.get("ES2VEC_WEB_HOST", "127.0.0.1").strip() or "127.0.0.1"
_DEFAULT_PORT = int(os.environ.get("ES2VEC_WEB_PORT", "8765"))


def _run_search(req: SearchRequest) -> dict[str, Any]:
    try:
        resp = execute_hybrid_search(req)
    except SearchExecutionError as exc:
        raise HTTPException(status_code=502, detail=f"检索失败: {exc}") from exc
    return search_response_to_dict(resp)


def _run_rag(req: RagRequest, *, debug: bool = False) -> dict[str, Any]:
    try:
        if debug:
            return rag_with_retrieval_payload(req)
        return rag_response_to_dict(execute_rag(req))
    except RagExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


app = FastAPI(
    title="es2vec 混合检索与 RAG",
    description="Elasticsearch 全文 + 向量混合检索；/api/v1/rag 为检索增强生成问答",
    version="1.2.0",
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version, "rag": "enabled"}


@app.get("/api/search", response_model=SearchResponse)
def api_search_legacy(
    q: str = Query(..., min_length=1, description="查询文本"),
    index: str = Query(default=DEFAULT_INDEX_NAME, description="Elasticsearch 索引名"),
    k: int = Query(default=10, ge=1, le=50, description="返回条数"),
    match_field: str = Query(
        default="",
        description=f"全文 match 字段，jieba 索引时一般为 {TEXT_TOKEN_FIELD}",
    ),
) -> dict[str, Any]:
    """执行混合检索（兼容旧版 GET 接口）。"""
    req = SearchRequest(
        query=q,
        index=index,
        k=k,
        **({"match_field": match_field} if match_field.strip() else {}),
    )
    return _run_search(req)


@app.get("/api/v1/search", response_model=SearchResponse)
def api_search_v1_get(
    query: str = Query(..., min_length=1, alias="q", description="查询文本"),
    index: str = Query(default=DEFAULT_INDEX_NAME),
    k: int = Query(default=10, ge=1, le=50),
    match_field: str = Query(default=""),
    use_rrf: bool | None = Query(default=None),
    vector_weight: float | None = Query(default=None),
    keyword_weight: float | None = Query(default=None),
    kw_sat: float | None = Query(default=None),
    keyword_norm_mode: str | None = Query(default=None),
    name_rerank: bool | None = Query(default=None),
) -> dict[str, Any]:
    """v1 混合检索（GET，参数与 POST body 一致）。"""
    extra: dict[str, Any] = {}
    if match_field.strip():
        extra["match_field"] = match_field.strip()
    if keyword_norm_mode and keyword_norm_mode in ("raw", "saturation", "log1p"):
        extra["keyword_norm_mode"] = keyword_norm_mode
    req = SearchRequest(
        query=query,
        index=index,
        k=k,
        use_rrf=use_rrf,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        kw_sat=kw_sat,
        name_rerank=name_rerank,
        **extra,
    )
    return _run_search(req)


@app.post("/api/v1/search", response_model=SearchResponse)
def api_search_v1_post(req: SearchRequest) -> dict[str, Any]:
    """v1 混合检索（POST JSON，推荐其它系统集成）。"""
    return _run_search(req)


@app.post("/api/v1/rag", response_model=RagResponse)
def api_rag_v1_post(
    req: RagRequest,
    debug: bool = Query(default=False, description="为 true 时额外返回 retrieval 原始 hits"),
) -> dict[str, Any]:
    """RAG 问答：混合检索 + LLM 生成（需配置对话 API Key 与 ES2VEC_CHAT_MODEL）。"""
    return _run_rag(req, debug=debug)


@app.get("/api/v1/rag", response_model=RagResponse)
def api_rag_v1_get(
    query: str = Query(..., min_length=1, alias="q"),
    index: str = Query(default=DEFAULT_INDEX_NAME),
    top_k: int = Query(default=RAG_DEFAULT_TOP_K, ge=1, le=20),
    debug: bool = Query(default=False),
) -> dict[str, Any]:
    """RAG 问答（GET，参数子集）。"""
    req = RagRequest(query=query, index=index, top_k=top_k)
    return _run_rag(req, debug=debug)


@app.get("/")
def index_page() -> FileResponse:
    """搜索首页。"""
    page = _STATIC_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="未找到 static/index.html")
    return FileResponse(page, media_type="text/html; charset=utf-8")


if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


def main() -> int:
    # https://www.uvicorn.org/
    try:
        import uvicorn
    except ImportError:
        print("请先安装: pip install fastapi uvicorn", file=sys.stderr)
        return 1

    print(f"es2vec Web 检索: http://{_DEFAULT_HOST}:{_DEFAULT_PORT}/")
    print(f"OpenAPI 文档: http://{_DEFAULT_HOST}:{_DEFAULT_PORT}/docs")
    print(f"默认索引: {DEFAULT_INDEX_NAME}  match 字段: {_default_match_field()}")
    uvicorn.run(
        app,
        host=_DEFAULT_HOST,
        port=_DEFAULT_PORT,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
