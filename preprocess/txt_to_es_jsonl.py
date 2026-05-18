# -*- coding: utf-8 -*-
"""
将 TXT 语料清洗后按「正文」章节切分，输出 Elasticsearch / es2vec 可用的 JSONL。

处理规则
--------
1. 删除整行仅为连字符分隔符的行（常见为 12 个连字符 ``------------``，也兼容其它长度、仅含 ``-`` 的整行）。
2. 删除「分节阅读 …」类小节标题行（支持 ``分节阅读 1``、``分节阅读 1行`` 等变体）。
3. 按正文分段：以单独一行且以「正文」开头（后接空白）作为新文档起点；该行及其后直至下一
   条「正文」标题前的内容合并为一条 ``text``。若文件开头在第一条「正文」前有书名、作者等，
   默认并入第一节 ``text`` 前部。

输出约定（与项目 index_corpus.py 一致）
----------------------------------------
- 每行一个 JSON 对象；
- 必须字段 ``text``（字符串）；
- 可选 ``_id``：本脚本默认按顺序生成 ``chapter_0001``、``chapter_0002`` …；若需完全由 ES
  自动生成，可使用 ``--no-id``，则 JSON 中不写 ``_id`` 字段。

输出路径默认与脚本所在目录同级：``<本脚本目录>/<输入文件名>.jsonl``。

用法示例::

    python preprocess/txt_to_es_jsonl.py --input examples/data/three_kingdoms.txt
    python preprocess/txt_to_es_jsonl.py --input examples/data/three_kingdoms.txt --output examples/data/three_kingdoms_by_chapter.jsonl
    python preprocess/txt_to_es_jsonl.py --input book.txt --no-preamble --no-id
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# 行级过滤：分节阅读（数字后可跟「行」字）
# ---------------------------------------------------------------------------
# 示例：「分节阅读 1」「分节阅读 12」「分节阅读 3行」
_RE_SECTION_READ = re.compile(r"^\s*分节阅读\s*.+?\s*$")


def _is_dash_separator_line(line: str) -> bool:
    """
    判断是否为「仅由连字符构成」的分隔行（strip 后非空且每个字符都是 '-'）。
    这样可同时去掉 ``------------`` 以及变体长度的分隔线。
    """
    s = line.strip()
    if len(s) < 2:
        return False
    return s[0] == "-" and set(s) == {"-"}


def _should_drop_line(line: str) -> bool:
    """需要整行丢弃的行：分节阅读标题、连字符分隔线。"""
    if _is_dash_separator_line(line):
        return True
    if _RE_SECTION_READ.match(line):
        return True
    return False


def _iter_kept_lines(text: str) -> Iterator[str]:
    """按行遍历，跳过应删除的行，保留换行结构由后续 join 处理。"""
    for line in text.splitlines():
        if _should_drop_line(line):
            continue
        yield line


def _is_body_heading_line(line: str) -> bool:
    """单独一行以「正文」开头且其后为空白，作为一章起点（与 three_kingdoms.txt 一致）。"""
    return line.startswith("正文") and (len(line) == 2 or line[2].isspace())


def split_into_chapters(
    lines: list[str],
    *,
    prepend_preamble_to_first: bool = True,
) -> list[str]:
    """
    将已过滤后的行列表按「正文」标题切成多段，每段为一个字符串（章节全文）。

    :param lines: 已去掉分节阅读与 ------------ 的行列表（无末尾换行符）。
    :param prepend_preamble_to_first: 第一条「正文」之前的行是否拼到第一节开头。
    :return: 各章 text 列表，顺序与文件中「正文」出现顺序一致。
    """
    # 定位第一条正文标题
    first_body_idx: int | None = None
    for i, ln in enumerate(lines):
        if _is_body_heading_line(ln):
            first_body_idx = i
            break

    if first_body_idx is None:
        # 没有「正文」行：整份文件作为单条文档
        whole = "\n".join(lines).strip()
        return [whole] if whole else []

    preamble_lines = lines[:first_body_idx] if prepend_preamble_to_first else []
    preamble = "\n".join(preamble_lines).strip()

    chapters: list[str] = []
    current_start = first_body_idx

    for j in range(first_body_idx, len(lines)):
        if j == first_body_idx:
            continue
        if _is_body_heading_line(lines[j]):
            chunk = "\n".join(lines[current_start:j]).strip()
            if chunk:
                chapters.append(chunk)
            current_start = j

    last = "\n".join(lines[current_start:]).strip()
    if last:
        chapters.append(last)

    if preamble and chapters:
        chapters[0] = f"{preamble}\n\n{chapters[0]}"

    return chapters


def write_jsonl(
    chapters: list[str],
    out_path: Path,
    *,
    emit_id: bool,
    id_prefix: str,
) -> None:
    """
    写入 JSONL。ensure_ascii=False 便于中文直接写入文件。

    :param emit_id: 为 False 时不写 _id，由 ES 端自动生成。
    :param id_prefix: emit_id 为 True 时，_id 为 ``{id_prefix}{序号:04d}``。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for idx, txt in enumerate(chapters, start=1):
            if not txt.strip():
                continue
            row: dict[str, str] = {"text": txt}
            if emit_id:
                row["_id"] = f"{id_prefix}{idx:04d}"
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _self_validate(jsonl_path: Path) -> None:
    """校验每行为合法 JSON，且含非空 text；若存在 _id 则须为非空字符串。"""
    n = 0
    with open(jsonl_path, encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" not in obj or not str(obj["text"]).strip():
                raise ValueError(f"第 {line_no} 行缺少有效 text 字段")
            if "_id" in obj and obj["_id"] is not None:
                if not str(obj["_id"]).strip():
                    raise ValueError(f"第 {line_no} 行 _id 为空")
            n += 1
    print(f"[OK] 校验通过：共 {n} 条文档 -> {jsonl_path}")


def main() -> int:
    pkg = Path(__file__).resolve().parent.parent
    default_input = pkg / "examples" / "data" / "three_kingdoms.txt"
    script_dir = pkg / "examples" / "data"
    ap = argparse.ArgumentParser(
        description="清洗 TXT（去掉分节阅读/连字符行）后按「正文」分章输出 ES JSONL",
    )
    ap.add_argument(
        "--input",
        "-i",
        type=Path,
        default=default_input,
        help="输入 UTF-8 TXT 路径",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="输出 JSONL 路径；默认写到脚本同目录下，文件名为「输入主名.jsonl」",
    )
    ap.add_argument(
        "--no-preamble",
        action="store_true",
        help="不把第一条「正文」之前的内容并入第一节",
    )
    ap.add_argument(
        "--no-id",
        action="store_true",
        help="不写 _id 字段（由 Elasticsearch 写入时自动生成）",
    )
    ap.add_argument(
        "--id-prefix",
        default="chapter_",
        help="生成 _id 时使用的前缀，默认 chapter_",
    )
    args = ap.parse_args()

    in_path: Path = args.input
    if not in_path.is_file():
        print(f"找不到输入文件: {in_path}", file=sys.stderr)
        return 1

    out_path = args.output
    if out_path is None:
        out_path = script_dir / f"{in_path.stem}.jsonl"

    raw = in_path.read_text(encoding="utf-8", errors="replace")
    kept = list(_iter_kept_lines(raw))
    chapters = split_into_chapters(
        kept,
        prepend_preamble_to_first=not args.no_preamble,
    )

    if not chapters:
        print("过滤后无有效正文，未生成输出。", file=sys.stderr)
        return 1

    write_jsonl(
        chapters,
        out_path,
        emit_id=not args.no_id,
        id_prefix=args.id_prefix,
    )
    _self_validate(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
