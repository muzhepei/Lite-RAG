# -*- coding: utf-8 -*-
"""
将语料文本写入 Elasticsearch：默认在本地用多语言 E5 生成向量后 bulk 索引；
可选 Elasticsearch Inference API（需许可证）；或 OpenAI 兼容 ``/v1/embeddings`` 网关。

输入格式（二选一）：
  1) JSONL：每行一个 JSON 对象，必须含 \"text\"；可选 \"_id\"（否则自动生成）。
     若加 ``--chunk-fields``，还可含 ``chapter_id``、``chunk_index``（与 ``three_kingdoms_ext.chunk_corpus`` 输出一致）。
  2) 纯文本：非 .jsonl 后缀时按「每行一条文档」，_id 为行号。

用法（在项目根目录；默认本地向量，无需 bootstrap_inference）::

    pip install -r requirements.txt
    python cli/index_corpus.py --input path/to/corpus.jsonl --index es2vec_corpus --recreate

有 Inference 许可且已创建端点时::

    python cli/index_corpus.py --input ... --use-es-inference --inference-id es2vec_multilingual_e5

OpenAI 兼容 Embeddings 网关（与本地 SentenceTransformer、ES Inference **三选一**）::

    python cli/index_corpus.py --input ... --index my_ix --recreate --use-openai-compatible-embedding

环境变量见 core/config.py（ES_HOST、ES2VEC_LOCAL_MODEL、DASHSCOPE_API_KEY、ES2VEC_OPENAI_*、ES2VEC_JIEBA、ES2VEC_SYNONYMS_SET 等）。

同义词（Synonyms API，集群 8.10+）：先用 ``python cli/put_synonyms_set.py`` 或 ``index_corpus.py --synonyms-file`` 上传词表，
建索引时指定相同 ``--synonyms-set-id``，全文 match 即会扩展查询。示例词表：``examples/data/synonyms_example.txt``。
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
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from elasticsearch.helpers import bulk

from es2vec.core.es_client import ensure_index, get_es
from es2vec.core.config import (
    CHAPTER_ID_FIELD,
    CHUNK_INDEX_FIELD,
    DEFAULT_INDEX_NAME,
    DEFAULT_INFERENCE_ID,
    DEFAULT_SYNONYMS_SET_ID,
    DEFAULT_VECTOR_DIMS,
    OPENAI_COMPATIBLE_BASE_URL,
    OPENAI_COMPATIBLE_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_ROUTE_AUTO_TO_DASHSCOPE,
    TEXT_FIELD,
    TEXT_TOKEN_FIELD,
    VECTOR_FIELD,
    env_flag_true,
)
from es2vec.core.inference_utils import cluster_has_smartcn, infer_text_embeddings
from es2vec.core.local_embedder import get_local_embedder
from es2vec.core.config import should_use_dashscope_multimodal_embedding
from es2vec.core.openai_compatible_embedder import (
    ApiEmbedder,
    describe_api_embedder_route,
    get_openai_compatible_embedder,
)
from es2vec.core.synonym_api import (
    build_index_settings_with_synonyms,
    load_synonym_rules_from_file,
    put_synonyms_set,
)


def _jieba_join(text: str) -> str:
    import jieba

    return " ".join(jieba.cut(text.strip(), cut_all=False))


@dataclass(frozen=True)
class IndexDoc:
    """单条待索引文档（JSONL 解析结果）。"""

    doc_id: str | None
    text: str
    chapter_id: str | None = None
    chunk_index: int | None = None


def _mapping_properties(
    vector_dims: int,
    text_index_analyzer: str,
    *,
    text_search_analyzer: str | None = None,
    include_token_field: bool,
    text_tokens_index_analyzer: str | None = None,
    text_tokens_search_analyzer: str | None = None,
    include_chunk_fields: bool = False,
) -> dict[str, Any]:
    text_spec: dict[str, Any] = {"type": "text", "analyzer": text_index_analyzer}
    if text_search_analyzer:
        text_spec["search_analyzer"] = text_search_analyzer
    props: dict[str, Any] = {
        TEXT_FIELD: text_spec,
        VECTOR_FIELD: {
            "type": "dense_vector",
            "dims": vector_dims,
            "index": True,
            "similarity": "cosine",
        },
    }
    if include_token_field:
        tok: dict[str, Any] = {
            "type": "text",
            "analyzer": text_tokens_index_analyzer or "standard",
        }
        if text_tokens_search_analyzer:
            tok["search_analyzer"] = text_tokens_search_analyzer
        props[TEXT_TOKEN_FIELD] = tok
    if include_chunk_fields:
        props[CHAPTER_ID_FIELD] = {"type": "keyword"}
        props[CHUNK_INDEX_FIELD] = {"type": "integer"}
    return props


def iter_index_docs(path: Path, *, parse_chunk_fields: bool) -> Iterator[IndexDoc]:
    """
    从 JSONL 或纯文本迭代待索引文档。

    当 ``parse_chunk_fields`` 为 True 时，读取可选字段 ``chapter_id``、``chunk_index``（整数或可解析整数的字符串）。
    """
    suffix = path.suffix.lower()
    if suffix == ".jsonl" or path.name.lower().endswith(".jsonl"):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSONL 第 {line_no} 行解析失败: {e}") from e
                text = obj.get("text")
                if not text or not isinstance(text, str):
                    raise ValueError(f"JSONL 第 {line_no} 行缺少字符串字段 text")
                _id = obj.get("_id")
                if _id is not None and not isinstance(_id, str):
                    _id = str(_id)
                chapter_id: str | None = None
                chunk_index: int | None = None
                if parse_chunk_fields:
                    raw_ch = obj.get("chapter_id")
                    if isinstance(raw_ch, str) and raw_ch.strip():
                        chapter_id = raw_ch.strip()
                    raw_i = obj.get("chunk_index")
                    if isinstance(raw_i, int):
                        chunk_index = raw_i
                    elif isinstance(raw_i, str) and raw_i.strip().lstrip("-").isdigit():
                        chunk_index = int(raw_i.strip())
                yield IndexDoc(
                    doc_id=_id,
                    text=text.strip(),
                    chapter_id=chapter_id,
                    chunk_index=chunk_index,
                )
        return

    with open(path, encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            t = line.strip()
            if not t:
                continue
            yield IndexDoc(doc_id=str(line_no), text=t)


def main() -> None:
    ap = argparse.ArgumentParser(description="语料 -> 本地或 ES 推理向量 -> ES bulk 索引")
    ap.add_argument("--input", required=True, help="JSONL 或纯文本路径")
    ap.add_argument("--index", default=DEFAULT_INDEX_NAME, help="目标索引名")
    ap.add_argument(
        "--use-es-inference",
        action="store_true",
        help="使用 Elasticsearch Inference API（需许可证与已部署端点）",
    )
    ap.add_argument(
        "--use-openai-compatible-embedding",
        action="store_true",
        help="使用 OpenAI 兼容 POST /v1/embeddings 网关生成向量（与 --use-es-inference 互斥）",
    )
    ap.add_argument(
        "--openai-base-url",
        default="",
        help="覆盖 ES2VEC_OPENAI_BASE_URL（须含 /v1）",
    )
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
    ap.add_argument(
        "--inference-id",
        default=DEFAULT_INFERENCE_ID,
        help="与 --use-es-inference 联用：text_embedding 端点 ID",
    )
    ap.add_argument("--local-model", default="", help="覆盖默认本地模型（默认见 ES2VEC_LOCAL_MODEL）")
    ap.add_argument("--recreate", action="store_true", help="若索引已存在则删除后重建")
    ap.add_argument("--batch-size", type=int, default=32, help="每批编码与 bulk 的文档数")
    ap.add_argument(
        "--vector-dims",
        type=int,
        default=0,
        help="dense_vector 维度；0 表示由本地模型或首批向量自动探测",
    )
    ap.add_argument(
        "--force-smartcn",
        action="store_true",
        help="强制使用 smartcn 分析器（未装插件将导致建索引失败）",
    )
    ap.add_argument("--no-smartcn", action="store_true", help="禁用 smartcn，全文腿使用 standard")
    ap.add_argument(
        "--jieba",
        action="store_true",
        help="写入 text_tokens 字段（jieba 分词后空格拼接，analyzer=standard）",
    )
    ap.add_argument(
        "--no-jieba",
        action="store_true",
        help="显式关闭 jieba（优先级高于环境变量 ES2VEC_JIEBA）",
    )
    ap.add_argument(
        "--synonyms-set-id",
        default=DEFAULT_SYNONYMS_SET_ID,
        help="Synonyms API 同义词集 id（默认 ES2VEC_SYNONYMS_SET）；与 put_synonyms_set.py 的 --set-id 一致",
    )
    ap.add_argument(
        "--synonyms-file",
        type=Path,
        default=None,
        help="若指定：建索引前用该文件内容 PUT 覆盖上述同义词集（UTF-8，一行一条，见 examples/data/synonyms_example.txt）",
    )
    ap.add_argument(
        "--chunk-fields",
        action="store_true",
        help="mapping 增加 chapter_id、chunk_index，并从 JSONL 读取写入（见 three_kingdoms_ext.chunk_corpus 输出）",
    )
    args = ap.parse_args()

    if args.use_es_inference and args.use_openai_compatible_embedding:
        raise SystemExit("--use-es-inference 与 --use-openai-compatible-embedding 不能同时使用")

    path = Path(args.input)
    if not path.is_file():
        raise SystemExit(f"文件不存在: {path}")

    use_jieba = False
    if args.no_jieba:
        use_jieba = False
    elif args.jieba:
        use_jieba = True
    else:
        use_jieba = env_flag_true("ES2VEC_JIEBA")

    es = get_es()
    use_smartcn = False
    if args.force_smartcn:
        use_smartcn = True
    elif not args.no_smartcn:
        use_smartcn = cluster_has_smartcn(es)

    synonyms_set_id = (args.synonyms_set_id or "").strip() or None
    if args.synonyms_file is not None:
        if not synonyms_set_id:
            raise SystemExit("使用 --synonyms-file 时必须同时指定非空的 --synonyms-set-id（或环境变量 ES2VEC_SYNONYMS_SET）")
        sf = Path(args.synonyms_file)
        if not sf.is_file():
            raise SystemExit(f"同义词文件不存在: {sf}")
        rules = load_synonym_rules_from_file(sf)
        put_synonyms_set(es, synonyms_set_id, rules)
        print(f"已上传同义词集 {synonyms_set_id!r}，规则数={len(rules)}（来源 {sf}）")
    elif synonyms_set_id:
        print(
            f"提示: 已指定同义词集 {synonyms_set_id!r}，假定集群中已存在该集；"
            "若尚未创建，请先运行: python cli/put_synonyms_set.py --set-id ... --file ..."
        )

    (
        settings,
        text_idx,
        text_search,
        tok_idx,
        tok_search,
    ) = build_index_settings_with_synonyms(
        use_smartcn=use_smartcn,
        synonyms_set_id=synonyms_set_id,
        include_jieba_token_field=use_jieba,
    )
    print(
        f"全文字段 index_analyzer={text_idx!r} search_analyzer={text_search!r} "
        f"(smartcn={use_smartcn}, synonyms_set={synonyms_set_id!r}) jieba={use_jieba} "
        f"text_tokens_index={tok_idx!r} text_tokens_search={tok_search!r}"
    )

    docs = list(iter_index_docs(path, parse_chunk_fields=bool(args.chunk_fields)))
    if not docs:
        raise SystemExit("未读取到任何有效行")

    model_name = args.local_model.strip() or None

    embedder: Any | None = None
    openai_embedder: ApiEmbedder | None = None

    if args.use_es_inference:
        first_text = docs[0].text
        first_vecs = infer_text_embeddings(
            es, args.inference_id, [first_text], input_type="ingest"
        )
        dims = args.vector_dims or len(first_vecs[0])
        if args.vector_dims and args.vector_dims != len(first_vecs[0]):
            print(
                f"警告: --vector-dims={args.vector_dims} 与推理输出维度 {len(first_vecs[0])} 不一致，"
                f"将使用推理输出维度 {len(first_vecs[0])}。"
            )
            dims = len(first_vecs[0])
    elif args.use_openai_compatible_embedding:
        def _cli(s: str) -> str | None:
            t = (s or "").strip()
            return t if t else None

        _eff_model = _cli(args.openai_embedding_model) or OPENAI_COMPATIBLE_EMBEDDING_MODEL
        _route_msg = describe_api_embedder_route(_eff_model)
        if (
            OPENAI_EMBEDDING_ROUTE_AUTO_TO_DASHSCOPE
            and not _cli(args.openai_base_url)
            and not should_use_dashscope_multimodal_embedding(_eff_model)
        ):
            _route_msg += "（已根据 DASHSCOPE_API_KEY 自动切换百炼 compatible-mode）"
        print(_route_msg)

        openai_embedder = get_openai_compatible_embedder(
            base_url=_cli(args.openai_base_url),
            api_key=_cli(args.openai_api_key),
            model=_cli(args.openai_embedding_model),
        )
        dims = args.vector_dims or openai_embedder.embedding_dim or DEFAULT_VECTOR_DIMS
        first_dim = len(
            openai_embedder.encode_passages([docs[0].text], batch_size=1)[0]
        )
        if args.vector_dims and args.vector_dims != first_dim:
            print(
                f"警告: --vector-dims={args.vector_dims} 与 API 输出维度 {first_dim} 不一致，将使用 {first_dim}。"
            )
        dims = first_dim
    else:
        # 先建索引、后加载 SentenceTransformer，避免 2GB ECS 上与 ES 同时占满内存导致 create 超时
        dims = args.vector_dims or DEFAULT_VECTOR_DIMS
        if not args.vector_dims:
            print(
                f"提示: 本地模型尚未加载，将先用 vector_dims={dims} 创建索引"
                f"（可用 --vector-dims 或 ES2VEC_VECTOR_DIMS 覆盖）；建索引后再加载模型并校验维度。"
            )

    props = _mapping_properties(
        dims,
        text_idx,
        text_search_analyzer=text_search,
        include_token_field=use_jieba,
        text_tokens_index_analyzer=tok_idx,
        text_tokens_search_analyzer=tok_search,
        include_chunk_fields=bool(args.chunk_fields),
    )
    ensure_index(es, args.index, properties=props, settings=settings, recreate=args.recreate)

    if (
        not args.use_es_inference
        and not args.use_openai_compatible_embedding
        and embedder is None
    ):
        embedder = get_local_embedder(model_name)
        first_dim = len(embedder.encode_passages([docs[0].text], batch_size=1)[0])
        if first_dim != dims:
            raise SystemExit(
                f"向量维度与索引 mapping 不一致: mapping={dims}, 模型输出={first_dim}。"
                f"请加 --vector-dims {first_dim} 与 --recreate 后重跑。"
            )

    def actions_for_batch(batch: list[IndexDoc]) -> list[dict[str, Any]]:
        texts = [d.text for d in batch]
        if args.use_es_inference:
            vecs = infer_text_embeddings(es, args.inference_id, texts, input_type="ingest")
        elif args.use_openai_compatible_embedding:
            assert openai_embedder is not None
            vecs = openai_embedder.encode_passages(
                texts, batch_size=min(len(texts), args.batch_size)
            )
        else:
            assert embedder is not None
            vecs = embedder.encode_passages(texts, batch_size=min(len(texts), args.batch_size))
        acts: list[dict[str, Any]] = []
        for doc, vec in zip(batch, vecs, strict=True):
            doc_id = doc.doc_id or str(uuid.uuid4())
            src: dict[str, Any] = {TEXT_FIELD: doc.text, VECTOR_FIELD: vec}
            if use_jieba:
                src[TEXT_TOKEN_FIELD] = _jieba_join(doc.text)
            if args.chunk_fields:
                if doc.chapter_id is not None:
                    src[CHAPTER_ID_FIELD] = doc.chapter_id
                if doc.chunk_index is not None:
                    src[CHUNK_INDEX_FIELD] = doc.chunk_index
            acts.append(
                {
                    "_op_type": "index",
                    "_index": args.index,
                    "_id": doc_id,
                    "_source": src,
                }
            )
        return acts

    total_ok = 0
    errs: list[Any] = []
    offset = 0
    while offset < len(docs):
        batch = docs[offset : offset + args.batch_size]
        offset += len(batch)
        acts = actions_for_batch(batch)
        ok, e = bulk(
            es.options(request_timeout=300),
            acts,
            chunk_size=min(100, len(acts)),
        )
        total_ok += ok
        errs.extend(e or [])

    es.indices.refresh(index=args.index)
    print(f"索引完成: {args.index}，成功写入约 {total_ok} 条，错误 {len(errs)} 条")
    if errs:
        print("错误示例:", errs[:3])


if __name__ == "__main__":
    main()
