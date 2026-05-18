# -*- coding: utf-8 -*-
"""
将「按回分章」的 JSONL 再切为较小 chunk，优先在句号类标点处断开，控制每块字数区间。

输入：与 index_corpus 一致，每行 JSON 含 ``text``、可选 ``_id``（如 chapter_0001）。
输出：每行含 ``text``、``_id``（chapter_0001_c001）、``chapter_id``、``chunk_index``。

用法（在项目根目录）::

    python examples/three_kingdoms_ext/chunk_corpus.py \\
        --input examples/data/three_kingdoms_by_chapter.jsonl \\
        --output examples/three_kingdoms_ext/out/three_kingdoms_chunks.jsonl

参数 ``--min-chars`` / ``--max-chars`` 控制目标块大小；``--overlap-chars`` 在相邻块间回卷若干字。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

# 句末标点：优先在此处结束一块（与计划「尽量以。」一致，并兼容常见句末）
_SENT_END = frozenset("。！？…")


def _segment_by_sentence_boundaries(text: str) -> list[str]:
    """
    按句末标点切分为若干片段（每段以句末标点结尾，最后一段可无句末标点）。

    空输入返回空列表。
    """
    s = text.strip()
    if not s:
        return []
    parts: list[str] = []
    start = 0
    for i, ch in enumerate(s):
        if ch in _SENT_END:
            piece = s[start : i + 1].strip()
            if piece:
                parts.append(piece)
            start = i + 1
    tail = s[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _hard_split_long_segment(seg: str, max_chars: int) -> list[str]:
    """单句超过 max_chars 时按固定长度硬切。"""
    if len(seg) <= max_chars:
        return [seg]
    out: list[str] = []
    i = 0
    while i < len(seg):
        out.append(seg[i : i + max_chars])
        i += max_chars
    return out


def pack_segments_to_chunks(
    segments: list[str],
    *,
    min_chars: int,
    max_chars: int,
) -> list[str]:
    """
    将句段合并为若干 chunk：长度尽量落在 [min_chars, max_chars]；
    超长单句先硬切；缓冲满 min_chars 即输出，若再加下一句会超 max 则先输出缓冲。
    """
    if min_chars <= 0 or max_chars < min_chars:
        raise ValueError("须满足 0 < min_chars <= max_chars")

    expanded: list[str] = []
    for seg in segments:
        expanded.extend(_hard_split_long_segment(seg, max_chars))

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if buf:
            t = "".join(buf).strip()
            if t:
                chunks.append(t)
            buf = []
            buf_len = 0

    for seg in expanded:
        if not seg:
            continue
        # 当前缓冲非空且加入本句会超长：先吐出缓冲
        if buf_len > 0 and buf_len + len(seg) > max_chars:
            flush()
        # 仍超长（空缓冲但单句满 max）：整句写入再立即吐出
        if buf_len == 0 and len(seg) >= max_chars:
            buf.append(seg)
            buf_len = len(seg)
            flush()
            continue
        buf.append(seg)
        buf_len += len(seg)
        if buf_len >= min_chars:
            flush()

    flush()
    if not chunks:
        return ["".join(expanded).strip()] if expanded else []
    if len(chunks) >= 2 and len(chunks[-1]) < max(1, min_chars // 2):
        last = chunks.pop()
        chunks[-1] = (chunks[-1] + last).strip()
    return [c for c in chunks if c]


def apply_overlap(chunks: list[str], overlap_chars: int) -> list[str]:
    """在相邻 chunk 间加入重叠前缀（复制上一块尾部若干字）。"""
    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        cur = chunks[i]
        tail = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
        merged = (tail + cur).strip()
        out.append(merged)
    return out


def iter_chapter_records(path: Path) -> Iterator[tuple[str | None, str]]:
    """Yield (_id, text) 每章一行。"""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 第 {line_no} 行解析失败: {e}") from e
            text = obj.get("text")
            if not text or not isinstance(text, str):
                raise ValueError(f"JSONL 第 {line_no} 行缺少字符串字段 text")
            _id = obj.get("_id")
            if _id is not None and not isinstance(_id, str):
                _id = str(_id)
            yield _id, text.strip()


def chapter_base_id(maybe_id: str | None, line_index: int) -> str:
    if maybe_id and maybe_id.strip():
        return maybe_id.strip()
    return f"chapter_{line_index:04d}"


def chunk_chapter_text(
    chapter_text: str,
    *,
    min_chars: int,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    segs = _segment_by_sentence_boundaries(chapter_text)
    chunks = pack_segments_to_chunks(segs, min_chars=min_chars, max_chars=max_chars)
    return apply_overlap(chunks, overlap_chars)


def write_chunks_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    min_chars: int,
    max_chars: int,
    overlap_chars: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(output_path, "w", encoding="utf-8", newline="\n") as out_f:
        for idx, (maybe_id, text) in enumerate(iter_chapter_records(input_path), start=1):
            base = chapter_base_id(maybe_id, idx)
            chunks = chunk_chapter_text(
                text,
                min_chars=min_chars,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
            for ci, chunk_text in enumerate(chunks):
                doc_id = f"{base}_c{ci + 1:03d}"
                row = {
                    "text": chunk_text,
                    "_id": doc_id,
                    "chapter_id": base,
                    "chunk_index": ci,
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="章 JSONL -> 句界导向 chunk JSONL")
    ap.add_argument("--input", type=Path, required=True, help="按章的 JSONL")
    ap.add_argument("--output", type=Path, required=True, help="输出 chunk JSONL")
    ap.add_argument("--min-chars", type=int, default=300, help="目标块最小字符数")
    ap.add_argument("--max-chars", type=int, default=800, help="目标块最大字符数")
    ap.add_argument("--overlap-chars", type=int, default=0, help="块间重叠字符数")
    args = ap.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"文件不存在: {args.input}")

    total = write_chunks_jsonl(
        args.input,
        args.output,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )
    print(f"已写入 {total} 条 chunk -> {args.output}")


if __name__ == "__main__":
    main()
