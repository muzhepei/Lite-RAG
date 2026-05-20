# -*- coding: utf-8 -*-
"""RAG 章级聚合：大 pool chunk 检索 → 章级打分去重 → 多命中用整章（方案 B）。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from es2vec.core.chapter_enrich import fetch_chapters_from_index
from es2vec.core.config import CHAPTER_ID_FIELD

RagContextKind = Literal["chapter", "chunk"]


@dataclass(frozen=True)
class RagContextUnit:
    """送入 LLM 的一条参考资料单元。"""

    kind: RagContextKind
    text: str
    chapter_id: str | None
    title: str | None
    score: float
    hit_count: int
    rank: int | None
    chunk_index: int | None
    id: str | None


def _hit_score(h: dict[str, Any]) -> float:
    raw = h.get("score")
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _chapter_id_from_hit(h: dict[str, Any]) -> str | None:
    raw = h.get(CHAPTER_ID_FIELD)
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


def _chapter_score(max_score: float, hit_count: int, alpha: float) -> float:
    return max_score * (1.0 + alpha * math.log1p(hit_count))


def _hit_to_chunk_unit(h: dict[str, Any], *, hit_count: int = 1) -> RagContextUnit | None:
    text = str(h.get("text") or "").strip()
    if not text:
        return None
    cid = _chapter_id_from_hit(h)
    raw_idx = h.get("chunk_index")
    chunk_index: int | None = None
    if raw_idx is not None:
        try:
            chunk_index = int(raw_idx)
        except (TypeError, ValueError):
            chunk_index = None
    return RagContextUnit(
        kind="chunk",
        text=text,
        chapter_id=cid,
        title=None,
        score=_hit_score(h),
        hit_count=hit_count,
        rank=h.get("rank") if isinstance(h.get("rank"), int) else None,
        chunk_index=chunk_index,
        id=str(h["id"]) if h.get("id") is not None else None,
    )


def _hits_without_chapter(
    hits: list[dict[str, Any]],
    *,
    limit: int,
) -> list[RagContextUnit]:
    orphan = sorted(
        [h for h in hits if _chapter_id_from_hit(h) is None],
        key=_hit_score,
        reverse=True,
    )
    out: list[RagContextUnit] = []
    for h in orphan:
        if len(out) >= limit:
            break
        u = _hit_to_chunk_unit(h)
        if u is not None:
            out.append(u)
    return out


def aggregate_hits_for_rag(
    hits: list[dict[str, Any]],
    *,
    es: Any,
    index: str,
    chapter_k: int,
    multi_hit_threshold: int,
    score_alpha: float,
) -> list[RagContextUnit]:
    """
    将 chunk 级 hits 聚合为最多 ``chapter_k`` 条上下文单元。

    方案 B：同章 ``hit_count >= multi_hit_threshold`` 时用整章 ``full_text``；
    否则用该章最高分单 chunk。无 ``chapter_id`` 时走兼容路径。
    """
    if chapter_k <= 0:
        return []

    valid = [h for h in hits if isinstance(h, dict)]
    if not valid:
        return []

    has_chapter = any(_chapter_id_from_hit(h) is not None for h in valid)
    if not has_chapter:
        return _hits_without_chapter(valid, limit=chapter_k)

    by_chapter: dict[str, list[dict[str, Any]]] = {}
    for h in valid:
        cid = _chapter_id_from_hit(h)
        if cid is None:
            continue
        by_chapter.setdefault(cid, []).append(h)

    scored: list[tuple[str, float, list[dict[str, Any]]]] = []
    for cid, group in by_chapter.items():
        scores = [_hit_score(x) for x in group]
        max_s = max(scores) if scores else 0.0
        ch_score = _chapter_score(max_s, len(group), score_alpha)
        scored.append((cid, ch_score, group))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_chapters = scored[:chapter_k]

    full_chapter_ids: list[str] = []
    for cid, _, group in top_chapters:
        if len(group) >= multi_hit_threshold:
            full_chapter_ids.append(cid)

    chapters_map: dict[str, dict[str, Any]] = {}
    if full_chapter_ids:
        chapters_map = fetch_chapters_from_index(es, index, full_chapter_ids)

    units: list[RagContextUnit] = []
    for cid, ch_score, group in top_chapters:
        hit_count = len(group)
        best = max(group, key=_hit_score)
        use_full = hit_count >= multi_hit_threshold
        if use_full:
            ch = chapters_map.get(cid) or {}
            full_text = str(ch.get("full_text") or "").strip()
            if full_text:
                units.append(
                    RagContextUnit(
                        kind="chapter",
                        text=full_text,
                        chapter_id=cid,
                        title=str(ch.get("title") or "").strip() or None,
                        score=ch_score,
                        hit_count=hit_count,
                        rank=best.get("rank") if isinstance(best.get("rank"), int) else None,
                        chunk_index=None,
                        id=None,
                    )
                )
                continue
        u = _hit_to_chunk_unit(best, hit_count=hit_count)
        if u is not None:
            units.append(
                RagContextUnit(
                    kind=u.kind,
                    text=u.text,
                    chapter_id=cid,
                    title=u.title,
                    score=ch_score,
                    hit_count=hit_count,
                    rank=u.rank,
                    chunk_index=u.chunk_index,
                    id=u.id,
                )
            )

    if len(units) < chapter_k:
        units.extend(
            _hits_without_chapter(valid, limit=chapter_k - len(units)),
        )

    return units[:chapter_k]
