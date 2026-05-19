# -*- coding: utf-8 -*-
"""
RAG 命令行问答：混合检索 + OpenAI 兼容 Chat。

在项目根目录执行::

    python cli/rag_chat.py --q "诸葛亮草船借箭是怎么回事？"
    python cli/rag_chat.py

需在 local_test.env 中配置 DASHSCOPE_API_KEY 或 MODELSCOPE_API_KEY，以及 ES 连接。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


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

from es2vec.core.config import DEFAULT_INDEX_NAME, RAG_DEFAULT_TOP_K
from es2vec.core.rag_service import (
    RagExecutionError,
    RagRequest,
    execute_rag,
    execute_rag_stream,
    rag_response_to_dict,
)


def _print_rag(data: dict) -> None:
    print("\n" + "=" * 60)
    print(f"问题: {data.get('query')}")
    print(f"模型: {data.get('model')}  索引: {data.get('index')}")
    print(f"检索: 返回 {data.get('retrieval_returned')} / 约 {data.get('retrieval_total')} 条")
    usage = data.get("usage") or {}
    if usage:
        print(f"Token: {usage}")
    print("-" * 60)
    print(data.get("answer") or "")
    sources = data.get("sources") or []
    if sources:
        print("-" * 60)
        print("引用片段:")
        for s in sources:
            ref = s.get("ref")
            preview = (s.get("text_preview") or "")[:120]
            print(f"  [{ref}] {preview}")
    print("=" * 60 + "\n")


def _collect_rag_stream(req: RagRequest) -> dict:
    """静默收集流式 RAG 事件，返回汇总 dict。"""
    meta: dict = {}
    done: dict = {}
    for event in execute_rag_stream(req):
        kind = event.get("event")
        if kind == "meta":
            meta = event
        elif kind == "done":
            done = event
        elif kind == "error":
            raise RagExecutionError(event.get("message") or "RAG 生成失败")
    return {**meta, **done}


def _print_rag_stream(req: RagRequest) -> None:
    """流式打印 RAG 回答。"""
    meta: dict = {}
    print("\n" + "=" * 60)
    for event in execute_rag_stream(req):
        kind = event.get("event")
        if kind == "meta":
            meta = event
            print(f"问题: {event.get('query')}")
            print(f"模型: {event.get('model')}  索引: {event.get('index')}")
            print(
                f"检索: 返回 {event.get('retrieval_returned')} "
                f"/ 约 {event.get('retrieval_total')} 条"
            )
            print("-" * 60)
        elif kind == "delta":
            print(event.get("content") or "", end="", flush=True)
        elif kind == "done":
            print()
            sources = event.get("sources") or []
            if sources:
                print("-" * 60)
                print("引用片段:")
                for s in sources:
                    ref = s.get("ref")
                    preview = (s.get("text_preview") or "")[:120]
                    print(f"  [{ref}] {preview}")
            print("=" * 60 + "\n")
        elif kind == "error":
            raise RagExecutionError(event.get("message") or "RAG 生成失败")


def main() -> int:
    ap = argparse.ArgumentParser(description="es2vec RAG 问答")
    ap.add_argument("--q", "--query", dest="query", default="", help="单次提问；省略则进入交互")
    ap.add_argument("--index", default=DEFAULT_INDEX_NAME, help="ES 索引名")
    ap.add_argument("--top-k", type=int, default=RAG_DEFAULT_TOP_K, help="检索条数")
    ap.add_argument("--json", action="store_true", help="输出完整 JSON")
    ap.add_argument(
        "--no-stream",
        action="store_true",
        help="禁用流式输出，等待完整 JSON 后再打印",
    )
    args = ap.parse_args()

    def ask_once(q: str) -> int:
        req = RagRequest(query=q, index=args.index, top_k=args.top_k)
        try:
            if args.json:
                if args.no_stream:
                    data = rag_response_to_dict(execute_rag(req))
                else:
                    data = _collect_rag_stream(req)
                print(json.dumps(data, ensure_ascii=False, indent=2))
            elif args.no_stream:
                data = rag_response_to_dict(execute_rag(req))
                _print_rag(data)
            else:
                _print_rag_stream(req)
        except RagExecutionError as exc:
            print(f"RAG 失败: {exc}", file=sys.stderr)
            return 1
        return 0

    q0 = (args.query or "").strip()
    if q0:
        return ask_once(q0)

    print("es2vec RAG 交互模式（空行退出）")
    print(f"索引: {args.index}  top_k: {args.top_k}")
    while True:
        try:
            line = input("\n问> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        if ask_once(line) != 0:
            continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
