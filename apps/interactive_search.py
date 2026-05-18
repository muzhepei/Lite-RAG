# -*- coding: utf-8 -*-
"""
交互式混合检索：在终端输入查询语句，在 Elasticsearch 中检索已索引语料。

依赖与前提
----------
- 已按 ``index_corpus.py`` 建好索引（默认使用本地向量，无需 ``bootstrap_inference.py``）。
- 环境变量配置好 ES 连接（见 ``core/es_client.py``、``local_test.env.example``）。

在项目根目录执行::

    python apps/interactive_search.py

若索引时使用了 ``--jieba``，请加上 ``--match-field text_tokens``。

使用 ES Inference 生成查询向量（需许可证）::

    python apps/interactive_search.py --use-es-inference --inference-id es2vec_multilingual_e5

OpenAI 兼容 Embeddings（与 index_corpus 同源）::

    python apps/interactive_search.py --use-openai-compatible-embedding

退出：输入空行，或 ``q`` / ``quit`` / ``exit``（不区分大小写）。

默认使用 **加权 script_score**（向量 0.7 + 关键词 0.3；关键词侧默认 BM25 **saturation**
归一，无需 RRF）。若集群已授权 RRF，可加 ``--rrf``。非 RRF 下可用 ``--kw-norm`` 等调节。
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
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

_ip = Path(__file__).resolve().parent
for _ in range(12):
    _script = _ip / "_install_path.py"
    if _ip.name == "es2vec" and _script.is_file():
        _spec = importlib.util.spec_from_file_location("es2vec_install_path", _script)
        _mod = importlib.util.module_from_spec(_spec)
        assert _spec.loader is not None
        _spec.loader.exec_module(_mod)
        _mod.install(__file__)
        break
    if _ip.parent == _ip:
        raise SystemExit("无法定位 es2vec/_install_path.py")
    _ip = _ip.parent
else:
    raise SystemExit("无法定位 es2vec/_install_path.py")

from es2vec.core.es_client import get_es
from es2vec.core.config import (
    DEFAULT_INDEX_NAME,
    DEFAULT_INFERENCE_ID,
    DEFAULT_LOCAL_MODEL,
    TEXT_FIELD,
)
from es2vec.cli.search_hybrid import KeywordNormMode, hybrid_search


def _print_hits(body: dict[str, Any], *, query: str, index: str, max_text_len: int) -> None:
    """将 ES 返回的 hits 格式化为人类可读文本。"""
    hits = (body.get("hits") or {}).get("hits") or []
    print(f"\n索引: {index}  查询: {query!r}  命中数: {len(hits)}")
    print("-" * 60)
    for i, h in enumerate(hits, start=1):
        src = h.get("_source") or {}
        text = src.get(TEXT_FIELD, "")
        score = h.get("_score")
        preview = str(text)
        if len(preview) > max_text_len:
            preview = preview[:max_text_len] + "..."
        print(f"{i}. score={score}\n   {preview}\n")


def _should_exit(line: str) -> bool:
    """判断是否应结束交互循环。"""
    s = line.strip().lower()
    if not s:
        return True
    return s in ("q", "quit", "exit", "bye")


def run_loop(
    *,
    index: str,
    use_es_inference: bool,
    use_openai_compatible_embedding: bool,
    inference_id: str,
    local_model: str,
    openai_base_url: str,
    openai_api_key: str,
    openai_embedding_model: str,
    match_field: str,
    k: int,
    num_candidates: int,
    rank_window_size: int,
    rank_constant: int,
    json_mode: bool,
    max_text_len: int,
    use_rrf: bool,
    vector_weight: float,
    keyword_weight: float,
    keyword_norm_mode: KeywordNormMode,
    kw_sat: float,
    kw_log_cap: float,
    name_rerank: bool | None,
    name_rerank_pool: int | None,
) -> None:
    """
    主循环：读取用户输入 → 调用 hybrid_search → 打印结果。

    参数说明与 ``search_hybrid.py`` 中命令行含义一致。
    ``use_rrf`` 默认 False（加权混合，免 RRF 许可）；与命令行 ``--rrf`` 一致。
    """
    fusion = "RRF retriever" if use_rrf else "加权 script_score（向量+BM25）"
    print(f"交互式混合检索（全文 match + 向量 kNN；{fusion}）")
    if use_es_inference:
        print(f"索引: {index}  查询向量: ES Inference 端点 {inference_id!r}")
    elif use_openai_compatible_embedding:
        print(f"索引: {index}  查询向量: OpenAI 兼容 Embeddings  match 字段: {match_field}")
    else:
        lm = local_model.strip() or DEFAULT_LOCAL_MODEL
        print(f"索引: {index}  查询向量: 本地模型 {lm!r}  match 字段: {match_field}")
    print("请输入查询内容；空行或 q/quit/exit 退出。\n")

    es = get_es()

    while True:
        try:
            line = input("查询> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if _should_exit(line):
            print("已退出。")
            break

        try:
            resp = hybrid_search(
                es,
                index,
                line,
                use_es_inference=use_es_inference,
                use_openai_compatible_embedding=use_openai_compatible_embedding,
                inference_id=inference_id,
                local_model=local_model,
                openai_base_url=openai_base_url,
                openai_api_key=openai_api_key,
                openai_embedding_model=openai_embedding_model,
                match_field=match_field,
                k=k,
                num_candidates=num_candidates,
                rank_window_size=rank_window_size,
                rank_constant=rank_constant,
                use_rrf=use_rrf,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                keyword_norm_mode=keyword_norm_mode,
                kw_sat=kw_sat,
                kw_log_cap=kw_log_cap,
                name_rerank=name_rerank,
                name_rerank_pool=name_rerank_pool,
            )
        except Exception as e:
            # 网络、认证、索引不存在、推理失败等统一提示，便于排查
            print(f"检索失败: {e}", file=sys.stderr)
            continue

        body = resp.body if hasattr(resp, "body") else resp
        if json_mode:
            print(json.dumps(body, ensure_ascii=False, indent=2, default=str))
        else:
            _print_hits(body, query=line, index=index, max_text_len=max_text_len)


def main() -> int:
    ap = argparse.ArgumentParser(description="终端输入查询 → ES 混合检索")
    ap.add_argument("--index", default=DEFAULT_INDEX_NAME, help="目标索引名")
    ap.add_argument(
        "--use-es-inference",
        action="store_true",
        help="使用 ES Inference API 生成查询向量（需许可证）",
    )
    ap.add_argument(
        "--use-openai-compatible-embedding",
        action="store_true",
        help="使用 OpenAI 兼容 /v1/embeddings（与 --use-es-inference 互斥）",
    )
    ap.add_argument("--openai-base-url", default="", help="覆盖 ES2VEC_OPENAI_BASE_URL")
    ap.add_argument(
        "--openai-api-key",
        default="",
        help="覆盖环境变量 DASHSCOPE_API_KEY（OpenAI 兼容 embeddings）",
    )
    ap.add_argument(
        "--openai-embedding-model",
        default="",
        help="覆盖 ES2VEC_OPENAI_EMBEDDING_MODEL",
    )
    ap.add_argument("--inference-id", default=DEFAULT_INFERENCE_ID, help="与 --use-es-inference 联用")
    ap.add_argument(
        "--local-model",
        default="",
        help="本地 SentenceTransformer 模型（默认见 ES2VEC_LOCAL_MODEL）",
    )
    ap.add_argument(
        "--match-field",
        default=TEXT_FIELD,
        help="全文 match 字段名（jieba 索引时一般为 text_tokens）",
    )
    ap.add_argument("--k", type=int, default=10, help="返回条数")
    ap.add_argument("--num-candidates", type=int, default=100, help="kNN 候选数")
    ap.add_argument("--rank-window-size", type=int, default=50)
    ap.add_argument("--rank-constant", type=int, default=60)
    ap.add_argument(
        "--rrf",
        action="store_true",
        help="使用 RRF retriever 融合（需集群许可证；默认用加权 script_score）",
    )
    ap.add_argument(
        "--vec-weight",
        type=float,
        default=0.7,
        dest="vec_weight",
        metavar="W",
        help="非 RRF 时向量项权重（默认 0.7）",
    )
    ap.add_argument(
        "--kw-weight",
        type=float,
        default=0.3,
        dest="kw_weight",
        metavar="W",
        help="非 RRF 时关键词腿权重（默认 0.3）",
    )
    ap.add_argument(
        "--kw-norm",
        choices=("raw", "saturation", "log1p"),
        default="saturation",
        dest="kw_norm",
        help="非 RRF：BM25 归一（与 search_hybrid.py 一致，默认 saturation）",
    )
    ap.add_argument(
        "--kw-sat",
        type=float,
        default=15.0,
        dest="kw_sat",
        metavar="S",
        help="非 RRF 且 saturation：kw/(kw+S)，默认 15",
    )
    ap.add_argument(
        "--kw-log-cap",
        type=float,
        default=10.0,
        dest="kw_log_cap",
        metavar="C",
        help="非 RRF 且 log1p：除数，默认 10",
    )
    ap.add_argument(
        "--name-rerank",
        action="store_true",
        help="强制按查询词密度二阶段重排",
    )
    ap.add_argument(
        "--no-name-rerank",
        action="store_true",
        help="关闭密度重排",
    )
    ap.add_argument(
        "--name-rerank-pool",
        type=int,
        default=None,
        metavar="N",
        help="重排候选池大小（默认 50）",
    )
    ap.add_argument("--json", action="store_true", help="输出原始 JSON")
    ap.add_argument(
        "--max-text-len",
        type=int,
        default=500,
        help="每条结果正文预览最大字符数",
    )
    args = ap.parse_args()

    if args.use_es_inference and args.use_openai_compatible_embedding:
        raise SystemExit("--use-es-inference 与 --use-openai-compatible-embedding 不能同时使用")

    name_rerank: bool | None = None
    if args.name_rerank:
        name_rerank = True
    elif args.no_name_rerank:
        name_rerank = False

    try:
        run_loop(
            index=args.index,
            use_es_inference=args.use_es_inference,
            use_openai_compatible_embedding=args.use_openai_compatible_embedding,
            inference_id=args.inference_id,
            local_model=args.local_model,
            openai_base_url=args.openai_base_url,
            openai_api_key=args.openai_api_key,
            openai_embedding_model=args.openai_embedding_model,
            match_field=args.match_field,
            k=args.k,
            num_candidates=args.num_candidates,
            rank_window_size=args.rank_window_size,
            rank_constant=args.rank_constant,
            json_mode=args.json,
            max_text_len=args.max_text_len,
            use_rrf=args.rrf,
            vector_weight=args.vec_weight,
            keyword_weight=args.kw_weight,
            keyword_norm_mode=args.kw_norm,
            kw_sat=args.kw_sat,
            kw_log_cap=args.kw_log_cap,
            name_rerank=name_rerank,
            name_rerank_pool=args.name_rerank_pool,
        )
    except SystemExit:
        raise
    except Exception as e:
        print(f"启动失败: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
