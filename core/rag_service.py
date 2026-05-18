# -*- coding: utf-8 -*-
"""RAG 编排：混合检索 → 上下文组装 → LLM 生成。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from es2vec.core.config import (
    DEFAULT_INDEX_NAME,
    RAG_CHAT_MAX_TOKENS,
    RAG_CHAT_TEMPERATURE,
    RAG_DEFAULT_TOP_K,
    RAG_MAX_CONTEXT_CHARS,
)
from es2vec.core.openai_compatible_chat import OpenAICompatibleChat, default_chat_client
from es2vec.core.rag_prompt import (
    DEFAULT_RAG_SYSTEM_PROMPT,
    build_context_from_hits,
    build_rag_messages,
)
from es2vec.core.search_service import (
    SearchExecutionError,
    SearchRequest,
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
        description="检索返回条数（送入 LLM 的片段数上限）",
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
    text_preview: str = ""


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


def _search_request_from_rag(req: RagRequest) -> SearchRequest:
    extra: dict[str, Any] = {"query": req.query, "index": req.index, "k": req.top_k}
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

    hits = [h for h in search_resp.hits if isinstance(h, dict)]
    max_ctx = (
        req.max_context_chars
        if req.max_context_chars is not None
        else RAG_MAX_CONTEXT_CHARS
    )
    context, source_meta = build_context_from_hits(hits, max_chars=max_ctx)

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
        raise RagExecutionError(f"LLM 生成失败: {exc}") from exc

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
