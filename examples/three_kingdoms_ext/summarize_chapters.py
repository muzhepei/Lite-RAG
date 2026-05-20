# -*- coding: utf-8 -*-
"""
按回 JSONL 离线生成章摘要，供 index_corpus --merge-chapter-summaries 写入 ES。

输入：与 chunk_corpus 相同的按回 JSONL（每行 text、可选 _id）。
输出：每回一行 chapter_summary 文档字段（见方案 A）。

用法（项目根目录）::

    python examples/three_kingdoms_ext/summarize_chapters.py \\
        --input examples/data/three_kingdoms_by_chapter.jsonl \\
        --output examples/three_kingdoms_ext/out/chapter_summaries.jsonl

百炼对话 API 文档:
https://help.aliyun.com/zh/model-studio/developer-reference/compatibility-of-openai-with-dashscope
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

_ext_dir = Path(__file__).resolve().parent
if str(_ext_dir) not in sys.path:
    sys.path.insert(0, str(_ext_dir))


def _bootstrap_es2vec_path() -> None:
    p = Path(__file__).resolve().parent
    for _ in range(12):
        script = p / "core" / "_install_path.py"
        if p.name == "es2vec" and script.is_file():
            spec = importlib.util.spec_from_file_location("_es2vec_install_path", script)
            mod = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(mod)
            mod.install(__file__)
            return
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError("无法定位 es2vec 包（需要 es2vec/core/_install_path.py）")


_bootstrap_es2vec_path()

from chunk_corpus import chapter_base_id, iter_chapter_records  # noqa: E402

from es2vec.core.chapter_enrich import extract_chapter_title  # noqa: E402
from es2vec.core.config import (  # noqa: E402
    CHAPTER_SUMMARY_CHUNK_INDEX,
    DOC_KIND_CHAPTER_SUMMARY,
)
from es2vec.core.openai_compatible_chat import default_chat_client  # noqa: E402

_SUMMARY_SYSTEM = """你是《三国演义》章回摘要助手。请仅根据用户给出的该回原文写摘要。
要求：中文；300～500 字；概括主要人物、事件与结局；不要编造原文没有的情节；不要输出标题以外的多余格式。"""


def _load_existing_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = obj.get("chapter_id")
            if isinstance(cid, str) and cid.strip():
                done.add(cid.strip())
    return done


def summarize_one(
    client: object,
    chapter_id: str,
    title: str,
    full_text: str,
    *,
    max_summary_chars: int,
) -> str:
    from es2vec.core.openai_compatible_chat import OpenAICompatibleChat

    assert isinstance(client, OpenAICompatibleChat)
    body = full_text
    if len(body) > 12000:
        body = body[:12000] + "\n…（原文已截断供摘要）"
    user = (
        f"章节 ID：{chapter_id}\n"
        f"章节标题：{title or '（未知）'}\n\n"
        f"## 该回原文\n\n{body}\n\n"
        f"请写 {max_summary_chars} 字以内的章回摘要。"
    )
    messages = [
        {"role": "system", "content": _SUMMARY_SYSTEM},
        {"role": "user", "content": user},
    ]
    text = client.complete(messages, temperature=0.2, max_tokens=1024)
    return (text or "").strip()


def write_summaries(
    input_path: Path,
    output_path: Path,
    *,
    max_summary_chars: int,
    sleep_sec: float,
    skip_existing: bool,
    append: bool,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = _load_existing_ids(output_path) if skip_existing else set()
    chat = default_chat_client()
    mode = "a" if append and output_path.is_file() else "w"
    n = 0

    with open(output_path, mode, encoding="utf-8", newline="\n") as out_f:
        for idx, (maybe_id, text) in enumerate(iter_chapter_records(input_path), start=1):
            base = chapter_base_id(maybe_id, idx)
            if skip_existing and base in done_ids:
                continue
            title = extract_chapter_title(text)
            try:
                summary = summarize_one(
                    chat,
                    base,
                    title,
                    text,
                    max_summary_chars=max_summary_chars,
                )
            except Exception as exc:
                print(f"跳过 {base}: {exc}", file=sys.stderr)
                continue
            if not summary:
                print(f"跳过 {base}: 模型未返回摘要", file=sys.stderr)
                continue
            row = {
                "_id": f"{base}_summary",
                "text": summary,
                "chapter_id": base,
                "doc_kind": DOC_KIND_CHAPTER_SUMMARY,
                "chapter_title": title,
                "chunk_index": CHAPTER_SUMMARY_CHUNK_INDEX,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            print(f"OK {base} ({len(summary)} 字)")
            if sleep_sec > 0:
                time.sleep(sleep_sec)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="按回 JSONL -> 章摘要 JSONL（方案 A）")
    ap.add_argument("--input", type=Path, required=True, help="按回 JSONL")
    ap.add_argument("--output", type=Path, required=True, help="输出 chapter_summaries.jsonl")
    ap.add_argument(
        "--max-chars",
        type=int,
        default=500,
        help="提示模型摘要目标字数上限",
    )
    ap.add_argument("--sleep", type=float, default=1.0, help="每回请求间隔秒数")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="输出文件中已有 chapter_id 则跳过",
    )
    ap.add_argument(
        "--append",
        action="store_true",
        help="追加写入（与 --skip-existing 联用断点续跑）",
    )
    args = ap.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"文件不存在: {args.input}")

    total = write_summaries(
        args.input,
        args.output,
        max_summary_chars=max(100, args.max_chars),
        sleep_sec=max(0.0, args.sleep),
        skip_existing=args.skip_existing,
        append=args.append or args.skip_existing,
    )
    print(f"已写入 {total} 条章摘要 -> {args.output}")


if __name__ == "__main__":
    main()
