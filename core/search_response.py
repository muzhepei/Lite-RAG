# -*- coding: utf-8 -*-
"""将 Elasticsearch 检索响应格式化为 API / 前端可用的 JSON 结构。"""
from __future__ import annotations

from typing import Any

from es2vec.core.config import (
    CHAPTER_ID_FIELD,
    CHUNK_INDEX_FIELD,
    TEXT_FIELD,
)


def _extract_total(hits_block: dict[str, Any], fallback: int) -> int:
    total = hits_block.get("total")
    if isinstance(total, dict):
        value = total.get("value")
        if isinstance(value, int):
            return value
    if isinstance(total, int):
        return total
    return fallback


def format_search_response(
    body: dict[str, Any],
    *,
    query: str,
    index: str,
    text_field: str = TEXT_FIELD,
    max_text_len: int = 4000,
) -> dict[str, Any]:
    """
    把 ``hybrid_search`` 返回的 body 转为稳定 JSON 结构。

    Returns:
        ``query``、``index``、``total``、``returned``、``hits`` 等字段。
    """
    hits_block = body.get("hits") or {}
    hits_raw = hits_block.get("hits") or []
    if not isinstance(hits_raw, list):
        hits_raw = []

    items: list[dict[str, Any]] = []
    for rank, h in enumerate(hits_raw, start=1):
        if not isinstance(h, dict):
            continue
        src = h.get("_source") or {}
        if not isinstance(src, dict):
            src = {}
        text = str(src.get(text_field, ""))
        truncated = len(text) > max_text_len
        if truncated:
            text = text[:max_text_len]

        item: dict[str, Any] = {
            "rank": rank,
            "id": h.get("_id"),
            "score": h.get("_score"),
            "text": text,
            "text_truncated": truncated,
        }
        chapter_id = src.get(CHAPTER_ID_FIELD)
        if chapter_id is not None:
            item[CHAPTER_ID_FIELD] = chapter_id
        chunk_index = src.get(CHUNK_INDEX_FIELD)
        if chunk_index is not None:
            item[CHUNK_INDEX_FIELD] = chunk_index
        items.append(item)

    returned = len(items)
    total = _extract_total(hits_block, returned)

    return {
        "query": query,
        "index": index,
        "total": total,
        "returned": returned,
        "hits": items,
    }
