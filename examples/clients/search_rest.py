# -*- coding: utf-8 -*-
"""
REST 调用 es2vec 混合检索示例。

https://fastapi.tiangolo.com/

用法::

    python examples/clients/search_rest.py --url http://127.0.0.1:8765 --query 刘备
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="es2vec REST 检索示例")
    parser.add_argument("--url", default="http://127.0.0.1:8765", help="Web 服务根 URL")
    parser.add_argument("--query", "-q", required=True, help="查询文本")
    parser.add_argument("--index", default="es2vec_corpus_chunks", help="ES 索引名")
    parser.add_argument("-k", type=int, default=5, help="返回条数")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    payload = {
        "query": args.query,
        "index": args.index,
        "k": args.k,
    }
    req = urllib.request.Request(
        f"{base}/api/v1/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"请求失败: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
