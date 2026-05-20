# -*- coding: utf-8 -*-
"""
百炼 DashScope 原生 Multimodal-Embedding API（``qwen3-vl-embedding`` 等）。

文档：https://help.aliyun.com/zh/model-studio/multimodal-embedding-api-reference

首版仅支持纯文本批量向量化（``input.contents`` 每项为 ``{"text": "..."}``），
与 ``OpenAICompatibleEmbedder`` 对外接口一致。
"""
from __future__ import annotations

import time
from typing import Any, Sequence

from es2vec.core.config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_MULTIMODAL_EMBEDDING_MAX_BATCH,
    DASHSCOPE_MULTIMODAL_EMBEDDING_PATH,
    OPENAI_EMBEDDING_DIMS,
    dashscope_multimodal_api_base,
    normalize_openai_compatible_api_key,
)
from es2vec.core.embedding_utils import l2_normalize, merge_embedding_indexed_rows

class DashScopeMultimodalEmbedder:
    """
    调用 ``POST {api_base}/services/embeddings/multimodal-embedding/multimodal-embedding``。

    Args:
        api_base: 含 ``/api/v1`` 的根路径。
        api_key: 百炼 API Key。
        model: 如 ``qwen3-vl-embedding``。
        dims_hint: ``parameters.dimension``；0 表示使用模型默认维（2560）。
    """

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        dims_hint: int = 0,
    ) -> None:
        base = api_base.strip().rstrip("/")
        k = normalize_openai_compatible_api_key(api_key.strip() if api_key else "")
        m = model.strip()
        if not base:
            raise RuntimeError(
                "百炼多模态向量：api_base 为空。请设置 ES2VEC_DASHSCOPE_MULTIMODAL_API_BASE "
                "或 ES2VEC_DASHSCOPE_BASE_URL（compatible-mode 将自动推导为 /api/v1）。"
            )
        if not m:
            raise RuntimeError(
                "百炼多模态向量：model 为空。请设置 ES2VEC_DASHSCOPE_EMBEDDING_MODEL=qwen3-vl-embedding。"
            )
        if not k or k == "----":
            raise RuntimeError(
                "百炼多模态向量：未配置 DASHSCOPE_API_KEY。请在 local_test.env 或环境中设置。"
            )
        self._api_base = base
        self._api_key = k
        self._model = m
        self._embedding_dimensions = int(dims_hint) if dims_hint and dims_hint > 0 else 0
        self._cached_dim: int | None = self._embedding_dimensions or None
        self._embed_url = f"{base}{DASHSCOPE_MULTIMODAL_EMBEDDING_PATH}"
        self._embed_inputs_cap = max(1, int(DASHSCOPE_MULTIMODAL_EMBEDDING_MAX_BATCH))

    @property
    def embedding_dim(self) -> int:
        if self._cached_dim is not None:
            return int(self._cached_dim)
        row = self._embed_batch(["."])[0]
        self._cached_dim = len(row)
        return int(self._cached_dim)

    def _build_payload(self, texts: list[str]) -> dict[str, Any]:
        contents = [{"text": t or ""} for t in texts]
        payload: dict[str, Any] = {
            "model": self._model,
            "input": {"contents": contents},
            "parameters": {"enable_fusion": False},
        }
        if self._embedding_dimensions > 0:
            payload["parameters"]["dimension"] = self._embedding_dimensions
        return payload

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = self._build_payload(texts)
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=120.0) as client:
                    resp = client.post(self._embed_url, headers=headers, json=body)
                if resp.status_code == 401:
                    raise RuntimeError(
                        "百炼多模态 Embedding 鉴权失败（401）。请核对 DASHSCOPE_API_KEY、"
                        "ES2VEC_DASHSCOPE_MULTIMODAL_API_BASE 与账号地域是否一致。\n"
                        f"url={self._embed_url!r} model={self._model!r}"
                    )
                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"百炼多模态 Embedding 请求失败 HTTP {resp.status_code}: "
                        f"{resp.text[:500]}"
                    )
                data = resp.json()
                if not isinstance(data, dict):
                    raise RuntimeError(f"百炼多模态 Embedding 响应非 JSON 对象: {type(data)!r}")
                code = data.get("code")
                if code:
                    raise RuntimeError(
                        f"百炼多模态 Embedding 业务错误 code={code!r} "
                        f"message={data.get('message')!r}"
                    )
                output = data.get("output")
                if not isinstance(output, dict):
                    raise RuntimeError(
                        f"百炼多模态 Embedding 响应缺少 output: {list(data.keys())}"
                    )
                emb_list = output.get("embeddings")
                if not isinstance(emb_list, list):
                    raise RuntimeError("百炼多模态 Embedding 响应 output.embeddings 非列表")
                raw = merge_embedding_indexed_rows(len(texts), emb_list)
                return [l2_normalize(r) for r in raw]
            except RuntimeError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        assert last_err is not None
        raise RuntimeError(
            "百炼多模态 Embedding 请求失败（已重试）。请检查 model、网络与配额。\n"
            f"url={self._embed_url!r} model={self._model!r}\n"
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
        bs = max(1, min(int(batch_size), self._embed_inputs_cap))
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


_MULTIMODAL_EMBEDDER_CACHE: dict[str, DashScopeMultimodalEmbedder] = {}


def _multimodal_cache_key(api_base: str, api_key: str, model: str, dims: int) -> str:
    return f"mm\0{api_base.strip().rstrip('/')}\0{api_key.strip()}\0{model.strip()}\0{dims}"


def get_dashscope_multimodal_embedder(
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str,
    dims_hint: int | None = None,
) -> DashScopeMultimodalEmbedder:
    base = (api_base if api_base is not None else dashscope_multimodal_api_base()).strip()
    k = normalize_openai_compatible_api_key(
        (api_key if api_key is not None else DASHSCOPE_API_KEY).strip()
    )
    dh = dims_hint if dims_hint is not None else OPENAI_EMBEDDING_DIMS
    key = _multimodal_cache_key(base, k, model, dh)
    if key not in _MULTIMODAL_EMBEDDER_CACHE:
        _MULTIMODAL_EMBEDDER_CACHE[key] = DashScopeMultimodalEmbedder(
            api_base=base,
            api_key=k,
            model=model.strip(),
            dims_hint=dh,
        )
    return _MULTIMODAL_EMBEDDER_CACHE[key]


def clear_dashscope_multimodal_embedder_cache() -> None:
    _MULTIMODAL_EMBEDDER_CACHE.clear()
