# -*- coding: utf-8 -*-
"""
将同义词规则写入 Elasticsearch Synonyms API（``PUT /_synonyms/{id}``）。

典型用法（先写同义词集，再带相同 ``--synonyms-set-id`` 跑 ``index_corpus.py`` 建索引）::

    python cli/put_synonyms_set.py --set-id es2vec_cn --file examples/data/synonyms_example.txt

词表文件格式见 :func:`es2vec.core.synonym_api.load_synonym_rules_from_file`：
每行一条 Solr 风格等价同义词，例如 ``孔明, 诸葛亮``；``#`` 行为注释。

环境变量连接方式见 ``core/es_client.py``（``ES_HOST``、``ES_PASSWORD`` 等）。
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
import sys
from pathlib import Path

from es2vec.core.es_client import get_es
from es2vec.core.config import DEFAULT_SYNONYMS_SET_ID
from es2vec.core.synonym_api import load_synonym_rules_from_file, put_synonyms_set


def main() -> None:
    ap = argparse.ArgumentParser(description="上传/覆盖 Elasticsearch 同义词集（Synonyms API）")
    ap.add_argument(
        "--set-id",
        default=DEFAULT_SYNONYMS_SET_ID,
        help="同义词集 ID（默认环境变量 ES2VEC_SYNONYMS_SET，须与建索引时一致）",
    )
    ap.add_argument("--file", required=True, type=Path, help="UTF-8 词表路径，一行一条规则")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将提交的规则 JSON，不调用 ES",
    )
    args = ap.parse_args()

    sid = (args.set_id or "").strip()
    if not sid:
        raise SystemExit("请指定 --set-id 或设置环境变量 ES2VEC_SYNONYMS_SET")

    path: Path = args.file
    if not path.is_file():
        raise SystemExit(f"文件不存在: {path}")

    rules = load_synonym_rules_from_file(path)
    if args.dry_run:
        print(json.dumps({"synonyms_set": rules}, ensure_ascii=False, indent=2))
        return

    es = get_es()
    resp = put_synonyms_set(es, sid, rules)
    print(f"已写入同义词集 {sid!r}，规则数={len(rules)}")
    if getattr(resp, "body", None) is not None:
        print(json.dumps(resp.body, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
