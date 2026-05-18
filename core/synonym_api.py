# -*- coding: utf-8 -*-
"""
Elasticsearch **Synonyms API**（8.10.0+）：用 ``PUT /_synonyms/{id}`` 管理同义词集，
在索引 settings 里用 ``synonym_graph`` + ``synonyms_set`` 引用（支持 ``updateable``，后续改词表可自动重载检索分析器）。

**重要**：托管同义词集（``synonyms_set``）的 ``synonym_graph`` **不能**挂在字段默认 ``analyzer`` 上（那会参与索引写入），
Elasticsearch 会报 ``not allowed to run in index time mode``。正确做法是：索引用不带同义词的 ``analyzer``，
检索用 ``search_analyzer`` 指向带 ``synonym_graph`` 的分析器（见 :func:`build_index_settings_with_synonyms`）。

所需集群权限：``manage_search_synonyms``（以及建索引的 ``manage_index_templates`` 等常规权限）。

本模块不负责调用 Inference；仅与同义词集与索引 analysis 配置相关。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from elasticsearch import Elasticsearch

# 与 index_corpus 中 filter 名称保持一致，便于排查
SYNONYM_FILTER_NAME = "es2vec_synonyms_graph"


def load_synonym_rules_from_file(path: Path) -> list[dict[str, str]]:
    """
    从文本文件加载同义词规则，供 ``put_synonym`` 使用。

    文件格式（Solr 等价写法，一行一条规则）：
    - 空行、仅空白行：忽略
    - 以 ``#`` 开头的行：注释
    - 其余整行作为 ``synonyms`` 字符串，例如：``孔明, 诸葛亮``

    等价表示「孔明」「诸葛亮」在检索时互相扩展（与双向同义接近，细节见官方 synonym_graph 文档）。

    Args:
        path: UTF-8 文本路径。

    Returns:
        形如 ``[{"synonyms": "孔明, 诸葛亮"}, ...]`` 的列表，可直接传给 :func:`put_synonyms_set`。
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    rules: list[dict[str, str]] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rules.append({"synonyms": stripped})
    if not rules:
        raise ValueError(f"同义词文件未解析到任何规则: {path}")
    return rules


def put_synonyms_set(
    es: Elasticsearch,
    synonym_set_id: str,
    synonyms_set: Sequence[Mapping[str, Any] | str],
) -> Any:
    """
    创建或**整体替换**指定 ID 的同义词集（与文件「增量行」不同，PUT 为整集覆盖）。

    Args:
        es: Elasticsearch 客户端。
        synonym_set_id: 同义词集 ID，与索引 analyzer 里 ``synonyms_set`` 一致。
        synonyms_set: 规则列表。元素可为 ``{"synonyms": "a, b"}``，或简写为字符串 ``"a, b"``。

    Returns:
        客户端返回的响应对象（可 ``.body`` 查看）。
    """
    normalized: list[dict[str, Any]] = []
    for item in synonyms_set:
        if isinstance(item, str):
            normalized.append({"synonyms": item.strip()})
        else:
            normalized.append(dict(item))
    return es.synonyms.put_synonym(id=synonym_set_id, synonyms_set=normalized)


def build_index_settings_with_synonyms(
    *,
    use_smartcn: bool,
    synonyms_set_id: str | None,
    include_jieba_token_field: bool,
) -> tuple[
    dict[str, Any],
    str,
    str | None,
    str | None,
    str | None,
]:
    """
    构造写入 ``indices.create(..., settings=...)`` 的 settings，并给出 ``text`` / ``text_tokens`` 的
    **索引** 与 **检索** analyzer 名称。

    当使用托管同义词集时，``synonym_graph`` 只能出现在 ``search_analyzer`` 对应链中；
    ``analyzer``（索引时）必须为不含该 filter 的平行链（分词方式一致）。

    Args:
        use_smartcn: 是否安装并启用 smartcn（与现逻辑一致）。
        synonyms_set_id: 若为非空字符串，则配置 ``synonym_graph`` 并引用该同义词集。
        include_jieba_token_field: 是否写入 ``text_tokens``。

    Returns:
        ``(settings, text_index_analyzer, text_search_analyzer, text_tokens_index_analyzer, text_tokens_search_analyzer)``
        - ``text_search_analyzer`` 为 ``None`` 表示 mapping 不写 ``search_analyzer``（与 ``analyzer`` 相同）。
        - 无 jieba 时，``text_tokens_*`` 两项均为 ``None``。
    """
    settings: dict[str, Any] = {"number_of_replicas": 0}
    filters: dict[str, Any] = {}
    analyzers: dict[str, Any] = {}

    if synonyms_set_id:
        filters[SYNONYM_FILTER_NAME] = {
            "type": "synonym_graph",
            "synonyms_set": synonyms_set_id,
            "updateable": True,
        }

    text_index_analyzer: str
    text_search_analyzer: str | None = None

    if use_smartcn:
        if synonyms_set_id:
            analyzers["es2vec_cn_smart_idx"] = {"tokenizer": "smartcn_tokenizer"}
            analyzers["es2vec_cn_smart_search"] = {
                "tokenizer": "smartcn_tokenizer",
                "filter": [SYNONYM_FILTER_NAME],
            }
            text_index_analyzer = "es2vec_cn_smart_idx"
            text_search_analyzer = "es2vec_cn_smart_search"
        else:
            analyzers["cn_smart"] = {"tokenizer": "smartcn_tokenizer"}
            text_index_analyzer = "cn_smart"
    else:
        if synonyms_set_id:
            analyzers["es2vec_text_std_idx"] = {
                "tokenizer": "standard",
                "filter": ["lowercase"],
            }
            analyzers["es2vec_text_std_syn"] = {
                "tokenizer": "standard",
                "filter": ["lowercase", SYNONYM_FILTER_NAME],
            }
            text_index_analyzer = "es2vec_text_std_idx"
            text_search_analyzer = "es2vec_text_std_syn"
        else:
            text_index_analyzer = "standard"

    text_tokens_index_analyzer: str | None = None
    text_tokens_search_analyzer: str | None = None
    if include_jieba_token_field:
        if synonyms_set_id:
            analyzers["es2vec_jieba_tokens_idx"] = {"tokenizer": "whitespace"}
            analyzers["es2vec_jieba_tokens_search"] = {
                "tokenizer": "whitespace",
                "filter": [SYNONYM_FILTER_NAME],
            }
            text_tokens_index_analyzer = "es2vec_jieba_tokens_idx"
            text_tokens_search_analyzer = "es2vec_jieba_tokens_search"
        else:
            text_tokens_index_analyzer = "standard"
            text_tokens_search_analyzer = None

    if filters or analyzers:
        settings["analysis"] = {}
        if filters:
            settings["analysis"]["filter"] = filters
        if analyzers:
            settings["analysis"]["analyzer"] = analyzers

    return (
        settings,
        text_index_analyzer,
        text_search_analyzer,
        text_tokens_index_analyzer,
        text_tokens_search_analyzer,
    )
