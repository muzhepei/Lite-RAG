# -*- coding: utf-8 -*-
"""
冒烟测试：
  --offline  不连 ES，仅校验 embedding 响应解析逻辑。
  默认        连接 ES：本地向量建临时索引 + 一次混合检索（无需 Inference 许可）。
  --inference  使用 ES Inference 全流程（需许可证与 ml 角色）。

在项目根目录执行::

    python cli/smoke_demo.py --offline
    python cli/smoke_demo.py
    python cli/smoke_demo.py --inference
"""
from __future__ import annotations

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
    raise RuntimeError("无法定位 es2vec 包（需要 es2vec/core/_install_path.py）")


_bootstrap_es2vec_path()


import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PKG_DIR

from es2vec.core.inference_utils import parse_text_embedding_inference_response

# 冒烟语料：三国演义片段（与 split/three_kingdoms.jsonl 主题一致）
_SMOKE_DOCS: list[dict[str, str]] = [
    {
        "_id": "1",
        "text": "刘备、关羽、张飞于桃园焚香结义，不求同年同月同日生，只愿同年同月同日死。",
    },
    {
        "_id": "2",
        "text": "曹操青梅煮酒论英雄，言天下英雄惟使君与操耳。",
    },
    {
        "_id": "3",
        "text": "诸葛亮隆中对策，三分天下之计，先取荆州为家基，益州成鼎足。",
    },
]
_SMOKE_QUERY = "桃园结义 刘备关羽张飞"


def _write_smoke_jsonl(path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for doc in _SMOKE_DOCS:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def _offline() -> None:
    fake_a = {
        "inference_results": [
            {"predicted_value": [0.1, 0.2, 0.3]},
            {"predicted_value": [1.0, 0.0, 0.0]},
        ]
    }
    fake_b = {"text_embedding": [{"embedding": [0.5, 0.5]}, {"embedding": [2.0, 3.0]}]}
    r1 = parse_text_embedding_inference_response(fake_a)
    r2 = parse_text_embedding_inference_response(fake_b)
    assert r1 == [[0.1, 0.2, 0.3], [1.0, 0.0, 0.0]], r1
    assert r2 == [[0.5, 0.5], [2.0, 3.0]], r2
    print("offline（三国演义冒烟配置已加载）: parse_text_embedding_inference_response OK")


def _any_ml_role(probe: dict) -> bool:
    for n in probe.get("nodes_roles") or []:
        if "ml" in (n.get("roles") or []):
            return True
    return False


def _handle_es_connect_error(exc: BaseException) -> None:
    from elastic_transport import ConnectionError as EsConnectionError
    from elasticsearch.exceptions import ApiError

    if isinstance(exc, ApiError) and getattr(exc, "meta", None) and exc.meta.status == 401:
        print(
            "在线冒烟跳过：ES 返回 401，请设置环境变量 ES_PASSWORD（及 ES_USER）。"
            "离线校验请使用: python smoke_demo.py --offline"
        )
        raise SystemExit(2) from exc
    if isinstance(exc, EsConnectionError):
        print(
            "在线冒烟跳过：无法连接 Elasticsearch（请确认 ES 已启动，并检查 "
            "local_test.env 中 ES_HOST）。离线校验: python smoke_demo.py --offline"
        )
        raise SystemExit(2) from exc
    raise


def _online_local() -> None:
    """本地向量 + index_corpus + hybrid_search，不调用 _inference。"""
    from es2vec.core.es_client import get_es
    from es2vec.core.config import DEFAULT_INDEX_NAME
    from es2vec.cli.search_hybrid import hybrid_search

    try:
        es = get_es()
        es.info()
    except BaseException as e:
        _handle_es_connect_error(e)

    py = sys.executable
    root = str(PROJECT_ROOT)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _write_smoke_jsonl(tmp_path)
        subprocess.check_call(
            [
                py,
                str(PKG_DIR / "cli" / "index_corpus.py"),
                "--input",
                str(tmp_path),
                "--index",
                DEFAULT_INDEX_NAME,
                "--recreate",
                "--batch-size",
                "8",
                "--no-smartcn",
            ],
            cwd=root,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    print(f"=== smoke: 查询（三国演义）=== {_SMOKE_QUERY}")
    resp = hybrid_search(
        es,
        DEFAULT_INDEX_NAME,
        _SMOKE_QUERY,
        use_es_inference=False,
        k=3,
        use_rrf=False,
    )
    body = resp.body if hasattr(resp, "body") else resp
    hits = (body.get("hits") or {}).get("hits") or []
    print("=== smoke: local embed + hybrid hits ===", len(hits))
    for h in hits:
        print(h.get("_source"))


def _online_inference() -> None:
    """需 Inference 许可：创建端点、索引、检索。"""
    from es2vec.core.es_client import get_es
    from es2vec.core.config import DEFAULT_INDEX_NAME, DEFAULT_INFERENCE_ID, DEFAULT_MODEL_ID
    from es2vec.core.inference_utils import (
        ensure_inference_endpoint,
        probe_cluster,
        wait_trained_model_allocated,
    )
    from es2vec.cli.search_hybrid import hybrid_search

    try:
        es = get_es()
        probe = probe_cluster(es)
    except BaseException as e:
        _handle_es_connect_error(e)
    print("=== smoke: cluster ===\n", json.dumps(probe, ensure_ascii=False, indent=2))
    if not _any_ml_role(probe):
        print("跳过 Inference 全流程：集群无 ml 角色。请改用默认本地冒烟: python smoke_demo.py")
        raise SystemExit(2)

    ensure_inference_endpoint(es, inference_id=DEFAULT_INFERENCE_ID)
    wait = wait_trained_model_allocated(es, DEFAULT_MODEL_ID, max_wait_seconds=120.0)
    if not wait.get("ready"):
        print(
            "模型在 120s 内未完全就绪，跳过索引与检索；"
            "请运行: python bootstrap_inference.py --max-wait 1800"
        )
        raise SystemExit(2)

    py = sys.executable
    root = str(PROJECT_ROOT)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _write_smoke_jsonl(tmp_path)
        subprocess.check_call(
            [
                py,
                str(PKG_DIR / "cli" / "index_corpus.py"),
                "--input",
                str(tmp_path),
                "--index",
                DEFAULT_INDEX_NAME,
                "--use-es-inference",
                "--inference-id",
                DEFAULT_INFERENCE_ID,
                "--recreate",
                "--batch-size",
                "8",
                "--no-smartcn",
            ],
            cwd=root,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    print(f"=== smoke: 查询（三国演义）=== {_SMOKE_QUERY}")
    resp = hybrid_search(
        es,
        DEFAULT_INDEX_NAME,
        _SMOKE_QUERY,
        use_es_inference=True,
        inference_id=DEFAULT_INFERENCE_ID,
        k=3,
        use_rrf=False,
    )
    body = resp.body if hasattr(resp, "body") else resp
    hits = (body.get("hits") or {}).get("hits") or []
    print("=== smoke: hybrid hits (ES inference) ===", len(hits))
    for h in hits:
        print(h.get("_source"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="仅解析逻辑，不连 ES")
    ap.add_argument(
        "--inference",
        action="store_true",
        help="使用 ES Inference 全流程（需许可证）；默认走本地向量",
    )
    args = ap.parse_args()
    if args.offline:
        _offline()
        return
    if args.inference:
        _online_inference()
        return
    _online_local()


if __name__ == "__main__":
    main()
