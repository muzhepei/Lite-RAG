# -*- coding: utf-8 -*-
"""
将「分节阅读 …」开头、以单独一行 ------------ 结尾的 TXT 语料切分为 ES / es2vec 可用的 JSONL。

输出约定（与 index_corpus.py 一致）：
  - 每行一个 JSON 对象；
  - 必须字段 \"text\"；
  - 可选 \"_id\"（本脚本按小节编号生成：sanguo_section_{编号}）。

用法示例::

    python preprocess/split_txt_to_es_jsonl.py
    python preprocess/split_txt_to_es_jsonl.py --input examples/data/three_kingdoms.txt --out-dir examples/output
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator

# 小节标题：整行匹配「分节阅读」+ 可选空白 + 数字
_SECTION_HEADER = re.compile(r"^分节阅读\s*(\d+)\s*$")

# 小节结束：整行（strip 后）恰好为 12 个连字符（与 three_kingdoms.txt 中一致）
_END_MARKER = "------------"


def _iter_section_blocks(lines: list[str]) -> Iterator[tuple[int, list[str]]]:
    """
    从已 split 的行列表中，按「分节阅读 N」到下一个「------------」或文件末尾，产出 (小节号, 原始行块)。

    行块包含：标题行 + 正文行（不含结束分隔行 ------------）。
    """
    i = 0
    n = len(lines)
    while i < n:
        m = _SECTION_HEADER.match(lines[i])
        if not m:
            i += 1
            continue
        section_no = int(m.group(1))
        block: list[str] = [lines[i]]
        i += 1
        while i < n:
            line = lines[i]
            if line.strip() == _END_MARKER:
                i += 1
                break
            block.append(line)
            i += 1
        yield section_no, block


def split_txt_to_segments(
    text: str,
    *,
    prepend_preamble_to_first: bool = True,
) -> list[tuple[str, str]]:
    """
    切分全文。

    参数:
        text: 原始 UTF-8 文本。
        prepend_preamble_to_first: 若首条「分节阅读」前有书名/作者等前言，是否拼到第一节 text 前。

    返回:
        [( _id, text ), ...]，顺序与文件中小节出现顺序一致。
    """
    lines = text.splitlines()
    # 找到第一个小节标题行下标
    first_hdr = next(
        (j for j, ln in enumerate(lines) if _SECTION_HEADER.match(ln)),
        None,
    )
    if first_hdr is None:
        raise ValueError("文件中未找到以「分节阅读」开头的行")

    preamble_lines = lines[:first_hdr] if prepend_preamble_to_first else []
    preamble = "\n".join(preamble_lines).strip()

    segments: list[tuple[str, str]] = []
    for section_no, block in _iter_section_blocks(lines):
        body_core = "\n".join(block).strip()
        if preamble and not segments:
            # 仅第一节前附加前言（书名、作者等）
            full_text = f"{preamble}\n\n{body_core}" if preamble else body_core
            preamble = ""  # 只用一次
        else:
            full_text = body_core

        doc_id = f"sanguo_section_{section_no:03d}"
        segments.append((doc_id, full_text))

    return segments


def write_jsonl(
    segments: list[tuple[str, str]],
    out_path: Path,
) -> None:
    """写入 JSONL：每行 {\"_id\":..., \"text\":...}，中文不转义为 \\uXXXX。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for doc_id, txt in segments:
            row = {"_id": doc_id, "text": txt}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _self_test(jsonl_path: Path) -> None:
    """快速校验：每行可解析 JSON，且含 text（及本脚本写入的 _id）。"""
    count = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            assert "text" in obj and obj["text"].strip(), count
            count += 1
    print(f"[OK] 自测通过：共 {count} 条 JSONL 文档 -> {jsonl_path}")


def main() -> int:
    pkg = Path(__file__).resolve().parent.parent
    default_input = pkg / "examples" / "data" / "three_kingdoms.txt"
    default_out_dir = pkg / "examples" / "output"

    ap = argparse.ArgumentParser(description="TXT 按分节阅读 / ------------ 切分为 ES JSONL")
    ap.add_argument("--input", type=Path, default=default_input, help="输入 TXT 路径")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=default_out_dir,
        help="输出目录（将创建 three_kingdoms.jsonl）",
    )
    ap.add_argument(
        "--no-preamble",
        action="store_true",
        help="不把「第一个分节阅读」之前的内容并入第一节",
    )
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"找不到输入文件: {args.input}", file=sys.stderr)
        return 1

    raw = args.input.read_text(encoding="utf-8", errors="replace")
    try:
        segments = split_txt_to_segments(
            raw,
            prepend_preamble_to_first=not args.no_preamble,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    out_jsonl = args.out_dir / "three_kingdoms.jsonl"
    write_jsonl(segments, out_jsonl)
    _self_test(out_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
