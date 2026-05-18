# -*- coding: utf-8 -*-
"""
gRPC 调用 es2vec 混合检索示例。

https://grpc.io/docs/languages/python/

用法::

    python examples/clients/search_grpc.py --host 127.0.0.1 --port 50051 --query 刘备
"""
from __future__ import annotations

import argparse
import importlib.util
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
    raise RuntimeError("无法定位 es2vec 包")


_bootstrap_es2vec_path()

import grpc

from es2vec.grpc_gen import es2vec_search_pb2, es2vec_search_pb2_grpc


def main() -> int:
    parser = argparse.ArgumentParser(description="es2vec gRPC 检索示例")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--query", "-q", required=True)
    parser.add_argument("--index", default="es2vec_corpus_chunks")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    stub = es2vec_search_pb2_grpc.SearchServiceStub(channel)
    request = es2vec_search_pb2.HybridSearchRequest(
        query=args.query,
        index=args.index,
        k=args.k,
    )
    try:
        response = stub.HybridSearch(request, timeout=120)
    except grpc.RpcError as exc:
        print(f"gRPC 失败: {exc.code()} {exc.details()}", file=sys.stderr)
        return 1

    print(f"query={response.query!r} index={response.index!r} total={response.total}")
    for hit in response.hits:
        text_preview = hit.text[:80] + ("…" if len(hit.text) > 80 else "")
        print(f"  #{hit.rank} score={hit.score:.4f} id={hit.id} {text_preview!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
