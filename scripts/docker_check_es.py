# -*- coding: utf-8 -*-
"""Docker 环境：检查 Elasticsearch 是否可连接。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _bootstrap() -> None:
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
    # Docker 镜像内已设置 PYTHONPATH=/opt
    parent = Path(__file__).resolve().parent.parent.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))


_bootstrap()


def main() -> int:
    from es2vec.core.es_client import get_es

    es = get_es()
    info = es.info()
    name = info.get("cluster_name", info)
    print(f"Elasticsearch OK: cluster_name={name}")
    health = es.cluster.health()
    print(f"cluster health: {health.get('status')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Elasticsearch check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
