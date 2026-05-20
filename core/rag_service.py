# -*- coding: utf-8 -*-
"""RAG 编排：混合检索 → 上下文组装 → LLM 生成。"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from es2vec.core.config import (
    DEFAULT_INDEX_NAME,
    RAG_CHAPTER_SCORE_ALPHA,
    RAG_CHAT_MAX_TOKENS,
    RAG_CHAT_TEMPERATURE,
    RAG_DEFAULT_TOP_K,
    RAG_FETCH_K,
    RAG_MAX_CONTEXT_CHARS,
    RAG_MULTI_HIT_THRESHOLD,
)
from es2vec.core.chapter_enrich import enrich_rag_sources_with_chapter_full_text
from es2vec.core.es_client import get_es
from es2vec.core.openai_compatible_chat import OpenAICompatibleChat, default_chat_client
from es2vec.core.rag_aggregate import aggregate_hits_for_rag
from es2vec.core.rag_prompt import (
    DEFAULT_RAG_SYSTEM_PROMPT,
    build_context_from_units,
    build_rag_messages,
)
from es2vec.core.search_service import (
    SearchExecutionError,
    SearchRequest,
    SearchResponse,
    execute_hybrid_search,
    search_response_to_dict,
)

KeywordNormModeLiteral = Literal["raw", "saturation", "log1p"]


class RagRequest(BaseModel):
    """RAG 问答请求（含检索与生成参数）。"""

    query: str = Field(..., min_length=1, description="用户问题")
    index: str = Field(default_factory=lambda: DEFAULT_INDEX_NAME, description="ES 索引名")
    top_k: int = Field(
        default=RAG_DEFAULT_TOP_K,
        ge=1,
        le=20,
        description="章级去重后送入 LLM 的参考资料条数（chapter_k）",
    )
    fetch_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="ES 检索 chunk 池大小；None 时读 ES2VEC_RAG_FETCH_K",
    )
    multi_hit_threshold: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="同章命中 chunk 数≥此值则用整章；None 时读 ES2VEC_RAG_MULTI_HIT_THRESHOLD",
    )
    chapter_score_alpha: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="章级分数命中数 boost 系数；None 时读 ES2VEC_RAG_CHAPTER_SCORE_ALPHA",
    )
    match_field: str | None = Field(default=None, description="全文 match 字段；None 用环境默认")
    use_rrf: bool | None = Field(default=None, description="是否 RRF 融合")
    vector_weight: float | None = Field(default=None)
    keyword_weight: float | None = Field(default=None)
    kw_sat: float | None = Field(default=None)
    keyword_norm_mode: KeywordNormModeLiteral | None = Field(default=None)
    name_rerank: bool | None = Field(default=None)
    max_context_chars: int | None = Field(
        default=None,
        ge=500,
        le=100_000,
        description="参考资料最大字符数",
    )
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=64, le=8192)
    system_prompt: str | None = Field(default=None, description="覆盖默认 system 提示词")
    include_sources: bool = Field(default=True, description="响应中是否包含引用片段元数据")

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("查询不能为空")
        return s


class RagSource(BaseModel):
    ref: int
    id: str | None = None
    rank: int | None = None
    score: float | None = None
    chapter_id: str | int | None = None
    chunk_index: int | None = None
    source_kind: Literal["chapter", "chunk", "chapter_summary"] | None = None
    chapter_title: str | None = None
    hit_count: int | None = None
    text: str = ""
    text_for_llm: str | None = None
    text_truncated: bool = False
    text_total_chars: int | None = None
    text_preview: str = ""
    chapter: dict[str, Any] | None = None


class RagResponse(BaseModel):
    query: str
    answer: str
    model: str
    index: str
    retrieval_total: int
    retrieval_returned: int
    sources: list[RagSource] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)


class RagExecutionError(Exception):
    """RAG 流程失败（检索、LLM 等）。"""


def _friendly_llm_error(exc: Exception) -> str:
    """将百炼等内容审核错误转为可操作提示。"""
    msg = str(exc)
    lower = msg.lower()
    if "datainspectionfailed" in lower or "inappropriate content" in lower:
        return (
            "大模型输出未通过百炼内容安全审核（《三国演义》中的战争、杀戮等描写可能被误判）。"
            "可尝试：① 换更简短的问题；② 减小 ES2VEC_RAG_MAX_CONTEXT_CHARS；"
            "③ 在 .env 中设置 ES2VEC_DASHSCOPE_DATA_INSPECTION=disable（仅用于合规的自建语料，"
            "须符合阿里云账号与内容安全策略）。原始错误: "
            + msg
        )
    return f"LLM 生成失败: {msg}"


def _resolve_fetch_k(req: RagRequest) -> int:
    return req.fetch_k if req.fetch_k is not None else RAG_FETCH_K


def _resolve_multi_hit_threshold(req: RagRequest) -> int:
    return (
        req.multi_hit_threshold
        if req.multi_hit_threshold is not None
        else RAG_MULTI_HIT_THRESHOLD
    )


def _resolve_chapter_score_alpha(req: RagRequest) -> float:
    return (
        req.chapter_score_alpha
        if req.chapter_score_alpha is not None
        else RAG_CHAPTER_SCORE_ALPHA
    )


def _search_request_from_rag(req: RagRequest) -> SearchRequest:
    extra: dict[str, Any] = {
        "query": req.query,
        "index": req.index,
        "k": _resolve_fetch_k(req),
    }
    if req.match_field and req.match_field.strip():
        extra["match_field"] = req.match_field.strip()
    if req.use_rrf is not None:
        extra["use_rrf"] = req.use_rrf
    if req.vector_weight is not None:
        extra["vector_weight"] = req.vector_weight
    if req.keyword_weight is not None:
        extra["keyword_weight"] = req.keyword_weight
    if req.kw_sat is not None:
        extra["kw_sat"] = req.kw_sat
    if req.keyword_norm_mode is not None:
        extra["keyword_norm_mode"] = req.keyword_norm_mode
    if req.name_rerank is not None:
        extra["name_rerank"] = req.name_rerank
    return SearchRequest(**extra)


def _build_rag_context(
    req: RagRequest,
    search_resp: SearchResponse,
) -> tuple[str, list[dict[str, Any]]]:
    """章级聚合后组装 LLM 参考资料与 sources 元数据。"""
    hits = [h for h in search_resp.hits if isinstance(h, dict)]
    max_ctx = (
        req.max_context_chars
        if req.max_context_chars is not None
        else RAG_MAX_CONTEXT_CHARS
    )
    es = get_es()
    units = aggregate_hits_for_rag(
        hits,
        es=es,
        index=req.index,
        chapter_k=req.top_k,
        multi_hit_threshold=_resolve_multi_hit_threshold(req),
        score_alpha=_resolve_chapter_score_alpha(req),
    )
    context, source_meta = build_context_from_units(units, max_chars=max_ctx)
    source_meta = enrich_rag_sources_with_chapter_full_text(
        source_meta, es, req.index
    )
    return context, source_meta


def execute_rag(
    req: RagRequest,
    *,
    chat: OpenAICompatibleChat | None = None,
    search_result: Any | None = None,
) -> RagResponse:
    """
    执行完整 RAG：检索 → 拼 context → Chat Completions。

    Args:
        search_result: 可选，已执行的 ``SearchResponse``，避免重复检索。

    Raises:
        RagExecutionError: 检索或生成失败。
    """
    if search_result is None:
        try:
            search_resp = execute_hybrid_search(_search_request_from_rag(req))
        except SearchExecutionError as exc:
            raise RagExecutionError(f"检索失败: {exc}") from exc
    else:
        search_resp = search_result

    context, source_meta = _build_rag_context(req, search_resp)

    if not context.strip():
        return RagResponse(
            query=req.query,
            answer="资料中未找到相关信息，无法根据语料回答该问题。",
            model=(chat.model if chat else ""),
            index=req.index,
            retrieval_total=search_resp.total,
            retrieval_returned=search_resp.returned,
            sources=[],
        )

    system = (req.system_prompt or DEFAULT_RAG_SYSTEM_PROMPT).strip()
    messages = build_rag_messages(query=req.query, context=context, system_prompt=system)

    client = chat or default_chat_client()
    temp = req.temperature if req.temperature is not None else RAG_CHAT_TEMPERATURE
    max_tok = req.max_tokens if req.max_tokens is not None else RAG_CHAT_MAX_TOKENS

    try:
        answer, usage = client.complete_with_usage(
            messages,
            temperature=temp,
            max_tokens=max_tok,
        )
    except Exception as exc:
        raise RagExecutionError(_friendly_llm_error(exc)) from exc

    if not answer:
        answer = "模型未返回有效内容，请稍后重试或检查 ES2VEC_CHAT_MODEL 与 API Key。"

    sources: list[RagSource] = []
    if req.include_sources:
        sources = [RagSource.model_validate(s) for s in source_meta]

    return RagResponse(
        query=req.query,
        answer=answer,
        model=client.model,
        index=req.index,
        retrieval_total=search_resp.total,
        retrieval_returned=search_resp.returned,
        sources=sources,
        usage=usage,
    )


def execute_rag_stream(
    req: RagRequest,
    *,
    chat: OpenAICompatibleChat | None = None,
    include_retrieval: bool = False,
) -> Iterator[dict[str, Any]]:
    """
    流式 RAG：检索完成后以 SSE 事件形式逐块产出 LLM 文本。

    事件类型:
      - meta: 检索元信息
      - retrieval: 原始 hits（仅 include_retrieval=True）
      - delta: 文本片段
      - done: 完整 answer、sources、usage
    """
    try:
        search_resp = execute_hybrid_search(_search_request_from_rag(req))
    except SearchExecutionError as exc:
        raise RagExecutionError(f"检索失败: {exc}") from exc

    context, source_meta = _build_rag_context(req, search_resp)

    client = chat or default_chat_client()

    yield {
        "event": "meta",
        "query": req.query,
        "model": client.model,
        "index": req.index,
        "retrieval_total": search_resp.total,
        "retrieval_returned": search_resp.returned,
    }

    if include_retrieval:
        yield {
            "event": "retrieval",
            "retrieval": search_response_to_dict(search_resp),
        }

    if not context.strip():
        msg = "资料中未找到相关信息，无法根据语料回答该问题。"
        yield {"event": "delta", "content": msg}
        yield {
            "event": "done",
            "answer": msg,
            "sources": [],
            "usage": {},
        }
        return

    system = (req.system_prompt or DEFAULT_RAG_SYSTEM_PROMPT).strip()
    messages = build_rag_messages(query=req.query, context=context, system_prompt=system)
    temp = req.temperature if req.temperature is not None else RAG_CHAT_TEMPERATURE
    max_tok = req.max_tokens if req.max_tokens is not None else RAG_CHAT_MAX_TOKENS

    full_parts: list[str] = []
    try:
        for chunk in client.stream_complete(
            messages,
            temperature=temp,
            max_tokens=max_tok,
        ):
            full_parts.append(chunk)
            yield {"event": "delta", "content": chunk}
    except Exception as exc:
        raise RagExecutionError(_friendly_llm_error(exc)) from exc

    answer = "".join(full_parts).strip()
    if not answer:
        answer = "模型未返回有效内容，请稍后重试或检查 ES2VEC_CHAT_MODEL 与 API Key。"
        yield {"event": "delta", "content": answer}

    sources: list[dict[str, Any]] = []
    if req.include_sources:
        sources = [
            RagSource.model_validate(s).model_dump(mode="json") for s in source_meta
        ]

    yield {
        "event": "done",
        "answer": answer,
        "sources": sources,
        "usage": {},
    }


def rag_response_to_dict(resp: RagResponse) -> dict[str, Any]:
    return resp.model_dump(mode="json")


def rag_with_retrieval_payload(
    req: RagRequest,
    *,
    chat: OpenAICompatibleChat | None = None,
) -> dict[str, Any]:
    """返回 RAG 结果，并附带原始检索 hits（便于调试）。"""
    try:
        search_resp = execute_hybrid_search(_search_request_from_rag(req))
    except SearchExecutionError as exc:
        raise RagExecutionError(f"检索失败: {exc}") from exc

    rag_resp = execute_rag(req, chat=chat, search_result=search_resp)
    out = rag_response_to_dict(rag_resp)
    out["retrieval"] = search_response_to_dict(search_resp)
    return out
