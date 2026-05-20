# -*- coding: utf-8 -*-
"""ES 查询过滤：混合检索仅命中 chunk 文档，兼容无 doc_kind 字段的旧索引。"""
from __future__ import annotations

from typing import Any

from es2vec.core.config import (
    DOC_KIND_CHAPTER_SUMMARY,
    DOC_KIND_CHUNK,
    DOC_KIND_FIELD,
)


def es_filter_chunks_only() -> dict[str, Any]:
    """
    仅检索 ``doc_kind=chunk``；无 ``doc_kind`` 字段的旧文档视为 chunk。

    用于混合检索与按章拉取 chunk 拼接整章正文，避免 ``chapter_summary`` 文档混入。
    """
    return {
        "bool": {
            "should": [
                {"term": {DOC_KIND_FIELD: DOC_KIND_CHUNK}},
                {"bool": {"must_not": {"exists": {"field": DOC_KIND_FIELD}}}},
            ],
            "minimum_should_match": 1,
        }
    }


def es_filter_chapter_summaries_only() -> dict[str, Any]:
    """仅命中 ``doc_kind=chapter_summary`` 文档。"""
    return {"term": {DOC_KIND_FIELD: DOC_KIND_CHAPTER_SUMMARY}}
