# -*- coding: utf-8 -*-
"""统一混合检索服务层，供 REST / gRPC 等入口复用。"""
from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from es2vec.cli.search_hybrid import KeywordNormMode, hybrid_search
from es2vec.core.config import (
    DEFAULT_INDEX_NAME,
    TEXT_FIELD,
    WEB_HYBRID_KW_SAT,
    WEB_HYBRID_KW_WEIGHT,
    WEB_HYBRID_VEC_WEIGHT,
    env_flag_true,
    env_float,
)
from es2vec.core.es_client import get_es
from es2vec.core.chapter_enrich import (
    enrich_hits_with_chapters,
    fetch_chapters_from_index,
)
from es2vec.core.config import CHAPTER_ID_FIELD, search_include_chapter_default
from es2vec.core.search_response import format_search_response

KeywordNormModeLiteral = Literal["raw", "saturation", "log1p"]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _default_use_rrf() -> bool:
    return _env_bool("ES2VEC_USE_RRF", False)


def _default_kw_norm() -> KeywordNormModeLiteral:
    raw = os.environ.get("ES2VEC_KW_NORM", "saturation").strip().lower()
    if raw in ("raw", "saturation", "log1p"):
        return raw  # type: ignore[return-value]
    return "saturation"


def _default_match_field() -> str:
    return os.environ.get("ES2VEC_MATCH_FIELD", TEXT_FIELD).strip() or TEXT_FIELD


class SearchRequest(BaseModel):
    """混合检索请求参数。"""

    query: str = Field(..., min_length=1, description="查询文本")
    index: str = Field(default_factory=lambda: DEFAULT_INDEX_NAME, description="ES 索引名")
    k: int = Field(default=10, ge=1, le=50, description="返回条数")
    match_field: str = Field(
        default_factory=_default_match_field,
        description="全文 match 字段",
    )
    use_rrf: bool | None = Field(
        default=None,
        description="是否使用 RRF；None 时读 ES2VEC_USE_RRF（默认 false）",
    )
    vector_weight: float | None = Field(default=None, description="向量权重")
    keyword_weight: float | None = Field(default=None, description="关键词权重")
    kw_sat: float | None = Field(default=None, description="BM25 saturation 分母")
    keyword_norm_mode: KeywordNormModeLiteral | None = Field(
        default=None,
        description="关键词归一化：raw | saturation | log1p",
    )
    name_rerank: bool | None = Field(
        default=None,
        description="密度重排；None 时按查询自动判断",
    )
    include_chapter: bool | None = Field(
        default=None,
        description="为命中附加整章信息（chapter_id 聚合）；None 时读 ES2VEC_SEARCH_INCLUDE_CHAPTER",
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("查询不能为空")
        return s


class SearchResponse(BaseModel):
    """与 format_search_response 输出一致的稳定 JSON 结构。"""

    query: str
    index: str
    total: int
    returned: int
    hits: list[dict[str, Any]]
    chapters: list[dict[str, Any]] = Field(
        default_factory=list,
        description="include_chapter 时按章去重后的完整章节列表",
    )


class SearchExecutionError(Exception):
    """检索执行失败（ES、嵌入等）。"""


def execute_hybrid_search(req: SearchRequest) -> SearchResponse:
    """
    执行混合检索并返回格式化结果。

    Raises:
        SearchExecutionError: ES 或嵌入失败。
        ValueError: 参数无效。
    """
    use_rrf = req.use_rrf if req.use_rrf is not None else _default_use_rrf()
    kw_norm: KeywordNormMode = (
        req.keyword_norm_mode
        if req.keyword_norm_mode is not None
        else _default_kw_norm()
    )

    vec_w = (
        req.vector_weight
        if req.vector_weight is not None
        else env_float("ES2VEC_VEC_WEIGHT", WEB_HYBRID_VEC_WEIGHT)
    )
    kw_w = (
        req.keyword_weight
        if req.keyword_weight is not None
        else env_float("ES2VEC_KW_WEIGHT", WEB_HYBRID_KW_WEIGHT)
    )
    kw_sat_val = (
        req.kw_sat if req.kw_sat is not None else env_float("ES2VEC_KW_SAT", WEB_HYBRID_KW_SAT)
    )

    try:
        es = get_es()
        resp = hybrid_search(
            es,
            req.index,
            req.query,
            match_field=req.match_field,
            k=req.k,
            use_openai_compatible_embedding=env_flag_true(
                "ES2VEC_USE_OPENAI_COMPATIBLE_EMBEDDING"
            ),
            use_rrf=use_rrf,
            keyword_norm_mode=kw_norm,
            vector_weight=vec_w,
            keyword_weight=kw_w,
            kw_sat=kw_sat_val,
            name_rerank=req.name_rerank,
        )
    except Exception as exc:
        raise SearchExecutionError(str(exc)) from exc

    body = resp.body if hasattr(resp, "body") else resp
    if not isinstance(body, dict):
        raise SearchExecutionError("Elasticsearch 返回格式异常")

    payload = format_search_response(
        body,
        query=req.query,
        index=req.index,
        text_field=req.match_field,
    )
    include_chapter = (
        req.include_chapter
        if req.include_chapter is not None
        else search_include_chapter_default()
    )
    if include_chapter:
        chapter_ids: list[str] = []
        for h in payload.get("hits") or []:
            if isinstance(h, dict):
                cid = h.get(CHAPTER_ID_FIELD)
                if cid is not None:
                    chapter_ids.append(str(cid))
        if chapter_ids:
            chapters = fetch_chapters_from_index(es, req.index, chapter_ids)
            enrich_hits_with_chapters(payload, chapters)

    return SearchResponse.model_validate(payload)


def search_response_to_dict(resp: SearchResponse) -> dict[str, Any]:
    return resp.model_dump(mode="json")
