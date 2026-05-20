# -*- coding: utf-8 -*-
"""按 chapter_id 从 ES 聚合整章正文，供检索结果展示。"""
from __future__ import annotations

import re
from typing import Any

from es2vec.core.config import (
    CHAPTER_ID_FIELD,
    CHAPTER_TITLE_FIELD,
    CHUNK_INDEX_FIELD,
    TEXT_FIELD,
)
from es2vec.core.doc_kind_filter import (
    es_filter_chapter_summaries_only,
    es_filter_chunks_only,
)

# 如：第一回 宴桃园豪杰三结义 斩黄巾英雄首立功
_CHAPTER_TITLE_RE = re.compile(
    r"(第[零一二三四五六七八九十百千万\d]+回[^\n。！？]{0,80})"
)


def extract_chapter_title(text: str) -> str:
    """从章首片段文本中提取「第 X 回 …」标题。"""
    s = (text or "").strip()
    if not s:
        return ""
    m = _CHAPTER_TITLE_RE.search(s)
    if m:
        return m.group(1).strip()
    first = s.split("\n", 1)[0].strip()
    return first[:120] if len(first) > 120 else first


def chapter_id_to_number(chapter_id: str) -> int | None:
    """``chapter_0001`` -> ``1``。"""
    m = re.match(r"chapter_(\d+)$", (chapter_id or "").strip(), re.I)
    if not m:
        return None
    try:
        return int(m.group(1), 10)
    except ValueError:
        return None


def fetch_chapters_from_index(
    es: Any,
    index: str,
    chapter_ids: list[str],
    *,
    max_chunks_per_chapter: int = 500,
) -> dict[str, dict[str, Any]]:
    """
    批量拉取各章全部 chunk，按 ``chunk_index`` 拼接为整章文本。

    Returns:
        ``{chapter_id: {chapter_id, chapter_number, title, full_text, chunk_count}}``
    """
    ids = sorted({c.strip() for c in chapter_ids if (c or "").strip()})
    if not ids:
        return {}

    try:
        resp = es.search(
            index=index,
            size=min(len(ids) * max_chunks_per_chapter, 10_000),
            query={
                "bool": {
                    "filter": [
                        {"terms": {CHAPTER_ID_FIELD: ids}},
                        es_filter_chunks_only(),
                    ],
                },
            },
            sort=[{CHAPTER_ID_FIELD: "asc"}, {CHUNK_INDEX_FIELD: "asc"}],
            _source=[TEXT_FIELD, CHAPTER_ID_FIELD, CHUNK_INDEX_FIELD],
        )
    except Exception:
        return {}

    body = resp.body if hasattr(resp, "body") else resp
    hits_raw = (body.get("hits") or {}).get("hits") or []
    if not isinstance(hits_raw, list):
        return {}

    by_chapter: dict[str, list[tuple[int, str]]] = {}
    for h in hits_raw:
        if not isinstance(h, dict):
            continue
        src = h.get("_source") or {}
        if not isinstance(src, dict):
            continue
        cid = str(src.get(CHAPTER_ID_FIELD) or "").strip()
        if not cid:
            continue
        text = str(src.get(TEXT_FIELD) or "")
        raw_idx = src.get(CHUNK_INDEX_FIELD)
        try:
            idx = int(raw_idx) if raw_idx is not None else len(by_chapter.get(cid, []))
        except (TypeError, ValueError):
            idx = len(by_chapter.get(cid, []))
        by_chapter.setdefault(cid, []).append((idx, text))

    out: dict[str, dict[str, Any]] = {}
    for cid, pairs in by_chapter.items():
        pairs.sort(key=lambda x: x[0])
        parts = [t for _, t in pairs if t]
        full_text = "\n".join(parts).strip()
        title = extract_chapter_title(parts[0] if parts else "")
        out[cid] = {
            "chapter_id": cid,
            "chapter_number": chapter_id_to_number(cid),
            "title": title,
            "full_text": full_text,
            "chunk_count": len(parts),
        }
    return out


def fetch_chapter_summaries_from_index(
    es: Any,
    index: str,
    chapter_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """
    批量拉取各章 ``doc_kind=chapter_summary`` 文档。

    Returns:
        ``{chapter_id: {chapter_id, title, summary}}``
    """
    ids = sorted({c.strip() for c in chapter_ids if (c or "").strip()})
    if not ids:
        return {}

    try:
        resp = es.search(
            index=index,
            size=min(len(ids), 500),
            query={
                "bool": {
                    "filter": [
                        {"terms": {CHAPTER_ID_FIELD: ids}},
                        es_filter_chapter_summaries_only(),
                    ],
                },
            },
            _source=[TEXT_FIELD, CHAPTER_ID_FIELD, CHAPTER_TITLE_FIELD],
        )
    except Exception:
        return {}

    body = resp.body if hasattr(resp, "body") else resp
    hits_raw = (body.get("hits") or {}).get("hits") or []
    if not isinstance(hits_raw, list):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for h in hits_raw:
        if not isinstance(h, dict):
            continue
        src = h.get("_source") or {}
        if not isinstance(src, dict):
            continue
        cid = str(src.get(CHAPTER_ID_FIELD) or "").strip()
        if not cid or cid in out:
            continue
        summary = str(src.get(TEXT_FIELD) or "").strip()
        if not summary:
            continue
        title = str(src.get(CHAPTER_TITLE_FIELD) or "").strip()
        if not title:
            title = extract_chapter_title(summary)
        out[cid] = {
            "chapter_id": cid,
            "title": title,
            "summary": summary,
        }
    return out


def enrich_rag_sources_with_chapter_full_text(
    sources: list[dict[str, Any]],
    es: Any,
    index: str,
) -> list[dict[str, Any]]:
    """
    为 RAG 引用列表从 ES 拉取整章 ``full_text``，供前端展示（不受 prompt 长度限制）。

    ``source_kind=chapter`` 时 ``text`` 替换为整章；``chunk`` 时保留命中片段，并附加 ``chapter`` 供展开。
    """
    if not sources:
        return sources

    chapter_ids: list[str] = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        cid = s.get(CHAPTER_ID_FIELD)
        if cid is not None:
            chapter_ids.append(str(cid).strip())
    if not chapter_ids:
        return sources

    chapters = fetch_chapters_from_index(es, index, chapter_ids)
    if not chapters:
        return sources

    for s in sources:
        if not isinstance(s, dict):
            continue
        cid = str(s.get(CHAPTER_ID_FIELD) or "").strip()
        if not cid or cid not in chapters:
            continue
        ch = chapters[cid]
        s["chapter"] = ch
        full = str(ch.get("full_text") or "").strip()
        if not full:
            continue
        title = str(ch.get("title") or "").strip()
        if title:
            s["chapter_title"] = title
        if s.get("source_kind") == "chapter":
            s["text"] = full
            s["text_total_chars"] = len(full)
            cap = 4000
            s["text_preview"] = full[:cap] + ("…" if len(full) > cap else "")
    return sources


def enrich_hits_with_chapters(
    payload: dict[str, Any],
    chapters: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """为每条 hit 附加 ``chapter`` 对象（若可解析 ``chapter_id``）。"""
    hits = payload.get("hits")
    if not isinstance(hits, list) or not chapters:
        return payload

    for h in hits:
        if not isinstance(h, dict):
            continue
        cid = h.get(CHAPTER_ID_FIELD)
        if cid is None:
            continue
        key = str(cid).strip()
        if key in chapters:
            h["chapter"] = chapters[key]
    payload["chapters"] = list(chapters.values())
    return payload
