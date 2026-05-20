# -*- coding: utf-8 -*-
"""RAG 提示词与检索片段上下文组装。"""
from __future__ import annotations

from typing import Any

from es2vec.core.rag_aggregate import RagContextUnit

DEFAULT_RAG_SYSTEM_PROMPT = """你是基于《三国演义》语料库的问答助手。请仅根据用户提供的「参考资料」回答问题。
若资料不足以回答，请明确说明「资料中未找到相关信息」，不要编造情节或人物关系。
回答请使用中文，条理清晰；在引用原文依据时，可在句末用 [编号] 标注来源（编号对应参考资料中的 [1]、[2] 等）。"""


def build_context_from_hits(
    hits: list[dict[str, Any]],
    *,
    max_chars: int,
) -> tuple[str, list[dict[str, Any]]]:
    """
    将检索命中格式化为带编号的参考资料块。

    Returns:
        (context_text, sources) — sources 为实际写入上下文的片段元数据。
    """
    if max_chars <= 0:
        return "", []

    parts: list[str] = []
    sources: list[dict[str, Any]] = []
    used = 0

    for i, h in enumerate(hits, start=1):
        text = str(h.get("text") or "").strip()
        if not text:
            continue

        header_bits: list[str] = [f"[{i}]"]
        doc_id = h.get("id")
        if doc_id is not None:
            header_bits.append(f"id={doc_id}")
        chapter_id = h.get("chapter_id")
        if chapter_id is not None:
            header_bits.append(f"chapter_id={chapter_id}")
        chunk_index = h.get("chunk_index")
        if chunk_index is not None:
            header_bits.append(f"chunk={chunk_index}")

        block = f"{' '.join(header_bits)}\n{text}\n"
        if used + len(block) > max_chars:
            remain = max_chars - used
            if remain < 80:
                break
            block = block[:remain] + "\n…（片段已截断）\n"

        parts.append(block)
        used += len(block)
        sources.append(
            {
                "ref": i,
                "id": doc_id,
                "rank": h.get("rank"),
                "score": h.get("score"),
                "chapter_id": chapter_id,
                "chunk_index": chunk_index,
                "text_preview": text[:240] + ("…" if len(text) > 240 else ""),
            }
        )
        if used >= max_chars:
            break

    return "\n".join(parts).strip(), sources


def _order_units_for_context(units: list[RagContextUnit]) -> list[RagContextUnit]:
    """章摘要/整章单元优先占用上下文预算，避免先写入短 chunk 挤占空间。"""
    summaries = [u for u in units if u.kind == "chapter_summary"]
    chapters = [u for u in units if u.kind == "chapter"]
    chunks = [u for u in units if u.kind == "chunk"]
    summaries.sort(key=lambda u: u.score, reverse=True)
    chapters.sort(key=lambda u: u.score, reverse=True)
    chunks.sort(key=lambda u: u.score, reverse=True)
    return summaries + chapters + chunks


def build_context_from_units(
    units: list[RagContextUnit],
    *,
    max_chars: int,
) -> tuple[str, list[dict[str, Any]]]:
    """
    将章级聚合后的上下文单元格式化为带编号的参考资料块。

    注意：``max_chars`` 为**全部**参考资料共享上限（默认见 ``ES2VEC_RAG_MAX_CONTEXT_CHARS``）。
    同章多段命中后虽会拉取整章，若多章合计超长仍会截断；整章优先于单 chunk 写入。

    Returns:
        (context_text, sources) — sources 含 source_kind、chapter_title、hit_count、text。
    """
    if max_chars <= 0:
        return "", []

    parts: list[str] = []
    sources: list[dict[str, Any]] = []
    used = 0
    ref = 0

    for u in _order_units_for_context(units):
        text = u.text.strip()
        if not text:
            continue

        ref += 1
        header_bits: list[str] = [f"[{ref}]"]
        if u.chapter_id is not None:
            header_bits.append(f"chapter_id={u.chapter_id}")
        if u.kind in ("chapter", "chapter_summary") and u.title:
            header_bits.append(f"title={u.title}")
        if u.kind == "chapter_summary":
            header_bits.append("kind=chapter_summary")
        if u.kind == "chunk" and u.chunk_index is not None:
            header_bits.append(f"chunk={u.chunk_index}")
        if u.id is not None:
            header_bits.append(f"id={u.id}")

        header = f"{' '.join(header_bits)}\n"
        suffix_chapter = "\n…（整章正文已因上下文长度上限截断，可调大 ES2VEC_RAG_MAX_CONTEXT_CHARS）\n"
        suffix_summary = "\n…（章回摘要已因上下文长度上限截断）\n"
        suffix_chunk = "\n…（片段已截断）\n"
        if u.kind == "chapter_summary":
            suffix = suffix_summary
        elif u.kind == "chapter":
            suffix = suffix_chapter
        else:
            suffix = suffix_chunk

        block_full = f"{header}{text}\n"
        truncated = False
        included_text = text

        if used + len(block_full) > max_chars:
            remain = max_chars - used
            min_room = 120 if u.kind in ("chapter", "chapter_summary") else 80
            if remain < min_room:
                break
            body_budget = remain - len(header) - len(suffix)
            if body_budget < 60:
                break
            included_text = text[:body_budget]
            truncated = True
            block_full = f"{header}{included_text}{suffix}"

        parts.append(block_full)
        used += len(block_full)
        preview_cap = 4000 if u.kind in ("chapter", "chapter_summary") else 240
        sources.append(
            {
                "ref": ref,
                "id": u.id,
                "rank": u.rank,
                "score": u.score,
                "chapter_id": u.chapter_id,
                "chunk_index": u.chunk_index,
                "source_kind": u.kind,
                "chapter_title": u.title,
                "hit_count": u.hit_count,
                # 引用展示：ES 整章/整段原文（不受 max_chars 截断）
                "text": text,
                "text_for_llm": included_text,
                "text_truncated": truncated,
                "text_total_chars": len(text),
                "text_preview": text[:preview_cap]
                + ("…" if len(text) > preview_cap else ""),
            }
        )
        if used >= max_chars:
            break

    return "\n".join(parts).strip(), sources


def build_rag_messages(
    *,
    query: str,
    context: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    """构造 OpenAI Chat Completions 的 messages 列表。"""
    user_body = (
        f"## 参考资料\n\n{context}\n\n"
        f"## 用户问题\n\n{query.strip()}\n\n"
        "请根据参考资料回答用户问题。"
    )
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_body},
    ]
