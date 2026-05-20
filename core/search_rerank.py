# -*- coding: utf-8 -*-
"""按查询词在正文中的出现密度对 ES 命中结果做二阶段重排（缓解人名误排）。"""
from __future__ import annotations

import math
from typing import Any

from es2vec.core.config import TEXT_FIELD


def query_term_density_score(query: str, text: str) -> float:
    """
    查询词在段落中的「主题性」启发分：出现次数 + 相对密度。

    提及次数越多、相对篇幅越短，分数越高；用于抑制「整章仅提一次人名」的误排。
    """
    q = query.strip()
    if not q or not text:
        return 0.0
    tf = text.count(q)
    if tf <= 0:
        return 0.0
    length = max(len(text), 1)
    density = tf / (length / 200.0)
    return math.log1p(tf) + min(density, 4.0)


def rerank_es_hits(
    hits: list[dict[str, Any]],
    query: str,
    *,
    text_field: str = TEXT_FIELD,
    blend: float = 0.35,
) -> list[dict[str, Any]]:
    """
    在 ES 分数基础上按词密度重排。

    ``combined = es_score * (1 + blend * density_score)``；无 ES 分时仅用密度分。
    """
    if not hits or not query.strip():
        return hits

    scored: list[tuple[float, dict[str, Any]]] = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        src = h.get("_source") or {}
        if not isinstance(src, dict):
            src = {}
        text = str(src.get(text_field, src.get(TEXT_FIELD, "")))
        dens = query_term_density_score(query, text)
        es_score = h.get("_score")
        base = float(es_score) if es_score is not None else 0.0
        combined = base * (1.0 + blend * dens) if base > 0 else dens
        scored.append((combined, h))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    for combined, h in scored:
        # 写回 _score，使 API/前端展示的「相关度」与排序一致
        h["_score"] = combined
        out.append(h)
    return out


def apply_name_rerank_to_search_body(
    body: dict[str, Any],
    query: str,
    *,
    text_field: str = TEXT_FIELD,
    k: int = 10,
    rerank_pool: int = 50,
    blend: float = 0.35,
) -> dict[str, Any]:
    """就地重排 ``body['hits']['hits']`` 并截断为 top ``k``。"""
    hits_block = body.get("hits")
    if not isinstance(hits_block, dict):
        return body
    hits_raw = hits_block.get("hits")
    if not isinstance(hits_raw, list) or not hits_raw:
        return body

    pool = hits_raw[: max(k, min(rerank_pool, len(hits_raw)))]
    reranked = rerank_es_hits(pool, query, text_field=text_field, blend=blend)
    hits_block["hits"] = reranked[:k]
    return body


def should_auto_name_rerank(query: str, *, max_chars: int = 4) -> bool:
    """短查询且无空格时视为可能的人名检索，启用密度重排。"""
    q = query.strip()
    if not q or " " in q or "\t" in q:
        return False
    return len(q) <= max_chars
