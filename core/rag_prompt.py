# -*- coding: utf-8 -*-
"""RAG 提示词与检索片段上下文组装。"""
from __future__ import annotations

from typing import Any

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
