# -*- coding: utf-8 -*-
"""按 chapter_id 从 ES 聚合整章正文，供检索结果展示。"""
from __future__ import annotations

import re
from typing import Any

from es2vec.core.config import CHAPTER_ID_FIELD, CHUNK_INDEX_FIELD, TEXT_FIELD

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
            query={"terms": {CHAPTER_ID_FIELD: ids}},
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
