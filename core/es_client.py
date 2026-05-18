# -*- coding: utf-8 -*-
"""
Elasticsearch 8.x 客户端封装（本项目内置，不依赖外部 word2vec 目录）。

通过环境变量配置连接；导入本模块前会自动加载同目录 ``local_test.env``（见 ``load_local_test_env``）。

环境变量（均为可选，除密码外有合理默认值）：
  ES_HOST          默认 https://localhost:9200
  ES_USER          默认 elastic
  ES_PASSWORD      默认空；ES 8 开启安全时通常必填
  ES_VERIFY_CERTS  默认 true；本地自签证书可设为 false
  ES_CA_CERTS      CA 证书路径；生产环境推荐设置并开启校验
"""
from __future__ import annotations

import os
from typing import Any, Mapping

from elasticsearch import Elasticsearch

from es2vec.core.load_local_test_env import load_local_test_env

load_local_test_env()


def _env_bool(name: str, default: bool) -> bool:
    """解析环境变量为布尔：1/true/yes 为真（不区分大小写）。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def get_es() -> Elasticsearch:
    """
    构造 Elasticsearch 客户端。

    Returns:
        已配置好 hosts / 认证 / TLS 的 Elasticsearch 实例。
    """
    host = os.environ.get("ES_HOST", "https://localhost:9200").strip()
    user = os.environ.get("ES_USER", "elastic").strip()
    password = os.environ.get("ES_PASSWORD", "")
    verify_certs = _env_bool("ES_VERIFY_CERTS", default=True)
    ca_certs = os.environ.get("ES_CA_CERTS")
    ca_certs = ca_certs.strip() if ca_certs else None

    kwargs: dict[str, Any] = {
        "hosts": [host],
        "verify_certs": verify_certs,
    }
    if not verify_certs:
        kwargs["ssl_show_warn"] = False
    if ca_certs:
        kwargs["ca_certs"] = ca_certs
    if password:
        kwargs["basic_auth"] = (user, password)

    return Elasticsearch(**kwargs)


def ensure_index(
    es: Elasticsearch,
    name: str,
    *,
    properties: Mapping[str, Any],
    settings: Mapping[str, Any] | None = None,
    recreate: bool = True,
) -> None:
    """
    确保索引存在且 mapping 符合预期。

    Args:
        es: Elasticsearch 客户端。
        name: 索引名。
        properties: mappings.properties 字段定义（不含外层 mappings 壳）。
        settings: 可选索引 settings；默认仅关闭副本便于单机开发。
        recreate: 为 True 时若索引已存在则先删除再创建（幂等重跑脚本）。
    """
    if recreate and es.indices.exists(index=name):
        es.indices.delete(index=name)

    if not es.indices.exists(index=name):
        idx_settings: Mapping[str, Any] = (
            dict(settings) if settings is not None else {"number_of_replicas": 0}
        )
        es.indices.create(
            index=name,
            mappings={"properties": dict(properties)},
            settings=dict(idx_settings),
        )
