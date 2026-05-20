# -*- coding: utf-8 -*-
"""
OpenAI 兼容 Embeddings API（自建网关、DashScope compatible-mode 等）。

``qwen3-vl-embedding`` 由 ``get_openai_compatible_embedder`` 自动路由至
``DashScopeMultimodalEmbedder``（原生 multimodal-embedding）。

与 ``LocalEmbedder`` 对齐的对外接口：``embedding_dim``、``encode_passages``、
``encode_queries``。

连接参数默认值见 ``es2vec.core.config``。
"""
from __future__ import annotations

import os
import time
from typing import Any, Sequence, Union

from es2vec.core.config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_EMBEDDING_MAX_BATCH,
    OPENAI_COMPATIBLE_BASE_URL,
    OPENAI_COMPATIBLE_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_DIMS,
    normalize_openai_compatible_api_key,
    should_use_dashscope_multimodal_embedding,
)
from es2vec.core.dashscope_multimodal_embedder import (
    DashScopeMultimodalEmbedder,
    clear_dashscope_multimodal_embedder_cache,
    get_dashscope_multimodal_embedder,
)
from es2vec.core.embedding_utils import l2_normalize, merge_embedding_indexed_rows


class OpenAICompatibleEmbedder:
    """
    使用 ``openai`` 官方客户端访问 ``POST {base}/embeddings``。

    Args:
        base_url: OpenAI 兼容根 URL，须含 ``/v1`` 后缀（与 OpenAI SDK 约定一致）。
        api_key: API Key；可为占位字符串若网关不要求鉴权。
        model: 网关上的 embedding 模型名。
        dims_hint: 若已知维度可省略启动时探测；0 表示首次请求时探测。
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dims_hint: int = 0,
    ) -> None:
        b = base_url.strip()
        k = normalize_openai_compatible_api_key(api_key.strip() if api_key else "")
        m = model.strip()
        if not b:
            raise RuntimeError(
                "OpenAI 兼容向量：base_url 为空。请在 es2vec.config 中设置 OPENAI_COMPATIBLE_BASE_URL "
                "或环境变量 ES2VEC_OPENAI_BASE_URL / CLI --openai-base-url（须含 /v1）。"
            )
        if not m:
            raise RuntimeError(
                "OpenAI 兼容向量：model 为空。请设置 ES2VEC_DASHSCOPE_EMBEDDING_MODEL（百炼）"
                "或 ES2VEC_OPENAI_EMBEDDING_MODEL / CLI --openai-embedding-model。"
            )
        self._base_url = b.rstrip("/")
        self._api_key = k
        self._model = m
        self._embedding_dimensions = int(dims_hint) if dims_hint and dims_hint > 0 else 0
        self._cached_dim: int | None = self._embedding_dimensions or None
        _bu = self._base_url.lower()
        if "dashscope.aliyuncs.com" in _bu or "dashscope-intl.aliyuncs.com" in _bu:
            self._embed_inputs_cap = max(1, int(DASHSCOPE_EMBEDDING_MAX_BATCH))
        else:
            self._embed_inputs_cap = 0

        from openai import OpenAI

        self._client = OpenAI(base_url=self._base_url, api_key=k or "EMPTY")

    @property
    def embedding_dim(self) -> int:
        if self._cached_dim is not None:
            return int(self._cached_dim)
        row = self._embed_batch(["."])[0]
        self._cached_dim = len(row)
        return int(self._cached_dim)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                create_kwargs: dict[str, Any] = {
                    "model": self._model,
                    "input": texts,
                }
                if self._embedding_dimensions > 0:
                    create_kwargs["dimensions"] = self._embedding_dimensions
                resp = self._client.embeddings.create(**create_kwargs)
                raw = merge_embedding_indexed_rows(len(texts), resp.data)
                return [l2_normalize(r) for r in raw]
            except Exception as exc:  # noqa: BLE001
                from openai import AuthenticationError

                if isinstance(exc, AuthenticationError):
                    _bu = self._base_url.lower()
                    if "modelscope.cn" in _bu:
                        raise RuntimeError(
                            "ModelScope 鉴权失败（401）。请逐项核对：\n"
                            "• 本地址需要魔搭 Token（MODELSCOPE_API_KEY 等），不是百炼 DASHSCOPE_API_KEY；\n"
                            "• 若你只有百炼密钥：不要设置 ES2VEC_OPENAI_BASE_URL 为魔搭，或删除该变量，"
                            "让程序在检测到 DASHSCOPE_API_KEY 时自动使用 dashscope 兼容网关；\n"
                            "• Token 在 https://modelscope.cn 申请；值勿加「Bearer 」前缀；确认未过期、无多余空格。\n"
                            f"base_url={self._base_url!r} model={self._model!r}"
                        ) from exc
                    if "dashscope.aliyuncs.com" in _bu or "dashscope-intl.aliyuncs.com" in _bu:
                        raise RuntimeError(
                            "百炼（DashScope）鉴权失败（401）。请逐项核对：\n"
                            "• 环境变量 DASHSCOPE_API_KEY 须为百炼控制台申请的 API Key（与魔搭 Token 不同）；\n"
                            "• 值勿加「Bearer 」前缀；国内与国际域的 base_url 与账号地域需一致；\n"
                            "• 可用 ES2VEC_DASHSCOPE_BASE_URL 指定兼容模式根路径（须含 /v1）。\n"
                            f"base_url={self._base_url!r} model={self._model!r}"
                        ) from exc
                last_err = exc
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        assert last_err is not None
        raise RuntimeError(
            "OpenAI 兼容 embeddings 请求失败（已重试）。请检查 base_url、model、"
            "网络与网关日志。勿在日志中打印 api_key。\n"
            f"base_url={self._base_url!r} model={self._model!r}\n"
            f"原始异常: {last_err!r}"
        ) from last_err

    def _encode_chunked(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> list[list[float]]:
        out: list[list[float]] = []
        buf: list[str] = []
        bs = max(1, int(batch_size))
        if self._embed_inputs_cap > 0:
            bs = min(bs, self._embed_inputs_cap)
        for t in texts:
            buf.append(t or "")
            if len(buf) >= bs:
                out.extend(self._embed_batch(buf))
                buf.clear()
        if buf:
            out.extend(self._embed_batch(buf))
        return out

    def encode_passages(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        del show_progress_bar
        return self._encode_chunked(texts, batch_size=batch_size)

    def encode_queries(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        del show_progress_bar
        return self._encode_chunked(texts, batch_size=batch_size)


ApiEmbedder = Union[OpenAICompatibleEmbedder, DashScopeMultimodalEmbedder]

_cache: dict[str, OpenAICompatibleEmbedder] = {}


def _cache_key(base_url: str, api_key: str, model: str, dims: int) -> str:
    return f"oa\0{base_url.strip().rstrip('/')}\0{api_key.strip()}\0{model.strip()}\0{dims}"


def describe_api_embedder_route(model: str) -> str:
    """供 CLI 打印：当前模型将走的路径。"""
    if should_use_dashscope_multimodal_embedding(model):
        from es2vec.core.config import dashscope_multimodal_api_base

        return (
            f"百炼多模态嵌入: native multimodal-embedding "
            f"api_base={dashscope_multimodal_api_base()!r} model={model!r}"
        )
    b = OPENAI_COMPATIBLE_BASE_URL
    return f"OpenAI 兼容嵌入: base_url={b!r} model={model!r}"


def get_openai_compatible_embedder(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    dims_hint: int | None = None,
) -> ApiEmbedder:
    """
    按连接参数缓存单例；``qwen3-vl-embedding`` 自动走 DashScope 原生多模态 API。

    参数为 None 时使用 ``es2vec.core.config`` 中模块级默认值。
    """
    b = (base_url if base_url is not None else OPENAI_COMPATIBLE_BASE_URL).strip()
    k = normalize_openai_compatible_api_key(
        (api_key if api_key is not None else DASHSCOPE_API_KEY).strip()
    )
    m = (model if model is not None else OPENAI_COMPATIBLE_EMBEDDING_MODEL).strip()
    if "modelscope.cn" in b.lower() and (not k or k == "----"):
        dashscope_hint = (os.environ.get("DASHSCOPE_API_KEY") or "").strip()
        extra = ""
        if dashscope_hint and dashscope_hint != "----":
            extra = (
                "\n检测到已设置 DASHSCOPE_API_KEY，但网关仍为魔搭。"
                "请从 .env 中删除 ES2VEC_OPENAI_BASE_URL（或改为百炼地址），"
                "并确认 docker compose 已加载 .env（cp .env.example .env）。"
            )
        raise RuntimeError(
            "当前 base_url 指向 ModelScope 推理 API，需要魔搭个人访问令牌（与百炼 DASHSCOPE_API_KEY 不是同一种密钥）。\n"
            "请设置 MODELSCOPE_API_KEY（或兼容名 API_KEY），在 https://modelscope.cn 个人中心创建 Token；"
            "值勿加「Bearer 」前缀。若你只有百炼密钥：\n"
            "  1) 在项目根目录 cp .env.example .env，写入 DASHSCOPE_API_KEY=sk-...\n"
            "  2) 不要设置 ES2VEC_OPENAI_BASE_URL 为魔搭地址\n"
            "  3) 可选 ES2VEC_DASHSCOPE_EMBEDDING_MODEL=qwen3-vl-embedding\n"
            "  4) 重新 docker compose run ... index_corpus --use-openai-compatible-embedding"
            f"{extra}\n"
            "也可在 index_corpus / search 上传入 --openai-api-key。勿将密钥写入代码或提交到 Git。"
        )
    dh = dims_hint if dims_hint is not None else OPENAI_EMBEDDING_DIMS

    if should_use_dashscope_multimodal_embedding(m):
        return get_dashscope_multimodal_embedder(
            api_key=k,
            model=m,
            dims_hint=dh,
        )

    key = _cache_key(b, k, m, dh)
    if key not in _cache:
        _cache[key] = OpenAICompatibleEmbedder(
            base_url=b,
            api_key=k,
            model=m,
            dims_hint=dh,
        )
    return _cache[key]


def clear_openai_embedder_cache() -> None:
    """测试或切换网关时清空缓存。"""
    _cache.clear()
    clear_dashscope_multimodal_embedder_cache()
