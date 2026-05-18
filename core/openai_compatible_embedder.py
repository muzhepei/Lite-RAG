# -*- coding: utf-8 -*-
"""
OpenAI 兼容 Embeddings API（自建网关、DashScope compatible-mode 等）。

与 ``LocalEmbedder`` 对齐的对外接口：``embedding_dim``、``encode_passages``、
``encode_queries``（当前实现二者均调用同一 endpoint，便于 Qwen3 等模型在网关侧统一处理）。

连接参数默认值见 ``es2vec.config``（仅设 ``DASHSCOPE_API_KEY`` 时会自动改走百炼 compatible-mode；魔搭网关需 ``MODELSCOPE_API_KEY`` 等）。
"""
from __future__ import annotations

import math
import time
from typing import Any, Sequence

from es2vec.core.config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_EMBEDDING_MAX_BATCH,
    OPENAI_COMPATIBLE_BASE_URL,
    OPENAI_COMPATIBLE_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_DIMS,
    normalize_openai_compatible_api_key,
)


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2 归一化，与 ES dense_vector cosine 常见假设对齐。"""
    if not vec:
        return vec
    s = math.sqrt(sum(x * x for x in vec))
    if s <= 0.0:
        return vec
    inv = 1.0 / s
    return [float(x * inv) for x in vec]


def _merge_embedding_batches(
    inputs: list[str],
    data_objects: Sequence[Any],
) -> list[list[float]]:
    """按 ``index`` 将多批 ``embedding`` 合并为与 ``inputs`` 同序的矩阵行。"""
    n = len(inputs)
    rows: list[list[float] | None] = [None] * n
    for obj in data_objects:
        idx = int(obj.index)
        emb = list(obj.embedding)
        if not (0 <= idx < n):
            raise RuntimeError(f"embedding 响应 index={idx} 超出批次长度 {n}")
        rows[idx] = emb
    missing = [i for i, r in enumerate(rows) if r is None]
    if missing:
        raise RuntimeError(f"embedding 响应缺少 index: {missing[:10]}...")
    return [r for r in rows if r is not None]  # type: ignore[return-value]


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
                "OpenAI 兼容向量：model 为空。请设置 OPENAI_COMPATIBLE_EMBEDDING_MODEL "
                "或 ES2VEC_OPENAI_EMBEDDING_MODEL / CLI --openai-embedding-model。"
            )
        self._base_url = b.rstrip("/")
        self._api_key = k
        self._model = m
        self._dims_hint = int(dims_hint) if dims_hint and dims_hint > 0 else 0
        self._cached_dim: int | None = self._dims_hint or None
        # 百炼 compatible-mode embeddings 单次 input 条数上限（默认 10），见 ES2VEC_DASHSCOPE_EMBEDDING_BATCH_MAX
        _bu = self._base_url.lower()
        if "dashscope.aliyuncs.com" in _bu or "dashscope-intl.aliyuncs.com" in _bu:
            self._embed_inputs_cap = max(1, int(DASHSCOPE_EMBEDDING_MAX_BATCH))
        else:
            self._embed_inputs_cap = 0

        from openai import OpenAI

        self._client = OpenAI(base_url=self._base_url, api_key=k or "EMPTY")

    @property
    def embedding_dim(self) -> int:
        """向量维度；未缓存时发一次极小请求探测。"""
        if self._cached_dim is not None:
            return int(self._cached_dim)
        row = self._embed_batch(["."])[0]
        self._cached_dim = len(row)
        return int(self._cached_dim)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """单次 API；含简单重试。"""
        if not texts:
            return []
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                )
                raw = _merge_embedding_batches(texts, resp.data)
                return [_l2_normalize(r) for r in raw]
            except Exception as exc:  # noqa: BLE001 — 网关错误类型不定
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
        del show_progress_bar  # 与 LocalEmbedder 签名对齐，API 路径无进度条
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


_cache: dict[str, OpenAICompatibleEmbedder] = {}


def _cache_key(base_url: str, api_key: str, model: str) -> str:
    return f"{base_url.strip().rstrip('/')}\0{api_key.strip()}\0{model.strip()}"


def get_openai_compatible_embedder(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    dims_hint: int | None = None,
) -> OpenAICompatibleEmbedder:
    """
    按连接参数缓存单例，避免重复构造客户端。

    参数为 None 时使用 ``es2vec.config`` 中模块级默认值。
    """
    b = (base_url if base_url is not None else OPENAI_COMPATIBLE_BASE_URL).strip()
    k = normalize_openai_compatible_api_key(
        (api_key if api_key is not None else DASHSCOPE_API_KEY).strip()
    )
    m = (model if model is not None else OPENAI_COMPATIBLE_EMBEDDING_MODEL).strip()
    # ModelScope 推理网关必须带个人 Token；占位符或未设会在服务端 401，此处提前说明
    if "modelscope.cn" in b.lower() and (not k or k == "----"):
        raise RuntimeError(
            "当前 base_url 指向 ModelScope 推理 API，需要魔搭个人访问令牌（与百炼 DASHSCOPE_API_KEY 不是同一种密钥）。\n"
            "请设置 MODELSCOPE_API_KEY（或兼容名 API_KEY），在 https://modelscope.cn 个人中心创建 Token；"
            "值勿加「Bearer 」前缀。若你只有百炼密钥，请删除 ES2VEC_OPENAI_BASE_URL 环境变量（不要指向魔搭），"
            "程序会在设置了 DASHSCOPE_API_KEY 时自动改用百炼 compatible-mode 网关。\n"
            "也可在 index_corpus / search 上传入 --openai-api-key。勿将密钥写入代码或提交到 Git。"
        )
    dh = dims_hint if dims_hint is not None else OPENAI_EMBEDDING_DIMS
    key = _cache_key(b, k, m)
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
