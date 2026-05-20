# -*- coding: utf-8 -*-
"""
OpenAI 兼容 Chat Completions API（百炼 compatible-mode、魔搭推理网关等）。

文档: https://platform.openai.com/docs/api-reference/chat/create
百炼: https://help.aliyun.com/zh/model-studio/developer-reference/compatibility-of-openai-with-dashscope
深度思考: https://help.aliyun.com/zh/model-studio/deep-thinking
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Sequence

from es2vec.core.config import (
    API_KEY,
    OPENAI_COMPATIBLE_BASE_URL,
    OPENAI_COMPATIBLE_CHAT_MODEL,
    RAG_CHAT_MAX_TOKENS,
    RAG_CHAT_TEMPERATURE,
    chat_enable_thinking,
    normalize_openai_compatible_api_key,
)

# 复用 HTTP 连接，避免每次 RAG 新建 OpenAI 客户端
_CHAT_CLIENT_CACHE: dict[str, OpenAICompatibleChat] = {}


class OpenAICompatibleChat:
    """使用 ``openai`` SDK 调用 ``POST {base}/chat/completions``。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = RAG_CHAT_TEMPERATURE,
        max_tokens: int = RAG_CHAT_MAX_TOKENS,
        enable_thinking: bool | None = None,
    ) -> None:
        b = base_url.strip()
        k = normalize_openai_compatible_api_key((api_key or "").strip())
        m = model.strip()
        if not b:
            raise RuntimeError(
                "OpenAI 兼容对话：base_url 为空。请设置 ES2VEC_OPENAI_BASE_URL 或 DASHSCOPE_API_KEY（自动路由百炼）。"
            )
        if not m:
            raise RuntimeError(
                "OpenAI 兼容对话：model 为空。请设置 ES2VEC_CHAT_MODEL 或 ES2VEC_DASHSCOPE_CHAT_MODEL。"
            )
        if not k or k == "----":
            raise RuntimeError(
                "OpenAI 兼容对话：未配置 API Key。请在 local_test.env 中设置 "
                "DASHSCOPE_API_KEY、MODELSCOPE_API_KEY 或 OPENAI_API_KEY。"
            )

        # https://github.com/openai/openai-python
        from openai import OpenAI

        self._client = OpenAI(base_url=b, api_key=k)
        self._model = m
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._enable_thinking = (
            chat_enable_thinking() if enable_thinking is None else enable_thinking
        )

    @property
    def model(self) -> str:
        return self._model

    def _completion_kwargs(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if "dashscope.aliyuncs.com" in OPENAI_COMPATIBLE_BASE_URL.lower():
            kwargs["extra_body"] = {"enable_thinking": self._enable_thinking}
        return kwargs

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """同步生成一条 assistant 回复文本。"""
        temp = self._temperature if temperature is None else temperature
        mt = self._max_tokens if max_tokens is None else max_tokens

        resp = self._client.chat.completions.create(
            **self._completion_kwargs(
                messages, temperature=temp, max_tokens=mt, stream=False
            ),
        )
        choice = resp.choices[0] if resp.choices else None
        if choice is None or choice.message is None:
            return ""
        content = choice.message.content
        return (content or "").strip()

    def complete_with_usage(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """与 ``complete`` 相同，并返回 usage 字典（若网关提供）。"""
        temp = self._temperature if temperature is None else temperature
        mt = self._max_tokens if max_tokens is None else max_tokens

        resp = self._client.chat.completions.create(
            **self._completion_kwargs(
                messages, temperature=temp, max_tokens=mt, stream=False
            ),
        )
        choice = resp.choices[0] if resp.choices else None
        text = ""
        if choice is not None and choice.message is not None:
            text = (choice.message.content or "").strip()

        usage: dict[str, Any] = {}
        if resp.usage is not None:
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }
        return text, usage

    def stream_complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        yield_reasoning: bool = False,
    ) -> Iterator[str]:
        """
        流式生成 assistant 回复，逐块 yield 文本片段。

        Args:
            yield_reasoning: 为 True 时同时 yield ``reasoning_content``（深度思考链）。
                默认仅 yield 正文 ``content``，且 ``enable_thinking`` 默认关闭以缩短首字延迟。
        """
        temp = self._temperature if temperature is None else temperature
        mt = self._max_tokens if max_tokens is None else max_tokens

        stream = self._client.chat.completions.create(
            **self._completion_kwargs(
                messages, temperature=temp, max_tokens=mt, stream=True
            ),
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            if yield_reasoning:
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield reasoning
            if delta.content:
                yield delta.content


def _chat_cache_key(base_url: str, api_key: str, model: str, enable_thinking: bool) -> str:
    return (
        f"chat\0{base_url.strip().rstrip('/')}\0{api_key.strip()}\0"
        f"{model.strip()}\0{int(enable_thinking)}"
    )


def default_chat_client() -> OpenAICompatibleChat:
    """按 ``config`` 默认构造对话客户端（进程内单例缓存）。"""
    thinking = chat_enable_thinking()
    key = _chat_cache_key(OPENAI_COMPATIBLE_BASE_URL, API_KEY, OPENAI_COMPATIBLE_CHAT_MODEL, thinking)
    if key not in _CHAT_CLIENT_CACHE:
        _CHAT_CLIENT_CACHE[key] = OpenAICompatibleChat(
            base_url=OPENAI_COMPATIBLE_BASE_URL,
            api_key=API_KEY,
            model=OPENAI_COMPATIBLE_CHAT_MODEL,
            enable_thinking=thinking,
        )
    return _CHAT_CLIENT_CACHE[key]
