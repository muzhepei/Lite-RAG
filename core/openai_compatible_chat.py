# -*- coding: utf-8 -*-
"""
OpenAI 兼容 Chat Completions API（百炼 compatible-mode、魔搭推理网关等）。

文档: https://platform.openai.com/docs/api-reference/chat/create
百炼: https://help.aliyun.com/zh/model-studio/developer-reference/compatibility-of-openai-with-dashscope
"""
from __future__ import annotations

from typing import Any, Sequence

from es2vec.core.config import (
    API_KEY,
    OPENAI_COMPATIBLE_BASE_URL,
    OPENAI_COMPATIBLE_CHAT_MODEL,
    RAG_CHAT_MAX_TOKENS,
    RAG_CHAT_TEMPERATURE,
    normalize_openai_compatible_api_key,
)


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

    @property
    def model(self) -> str:
        return self._model

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
            model=self._model,
            messages=list(messages),
            temperature=temp,
            max_tokens=mt,
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
            model=self._model,
            messages=list(messages),
            temperature=temp,
            max_tokens=mt,
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


def default_chat_client() -> OpenAICompatibleChat:
    """按 ``config`` 默认构造对话客户端。"""
    return OpenAICompatibleChat(
        base_url=OPENAI_COMPATIBLE_BASE_URL,
        api_key=API_KEY,
        model=OPENAI_COMPATIBLE_CHAT_MODEL,
    )
