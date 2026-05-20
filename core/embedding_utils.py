# -*- coding: utf-8 -*-
"""向量后处理与 API 响应合并（OpenAI 兼容 / DashScope 多模态共用）。"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def l2_normalize(vec: list[float]) -> list[float]:
    """L2 归一化，与 ES dense_vector cosine 常见假设对齐。"""
    if not vec:
        return vec
    s = math.sqrt(sum(x * x for x in vec))
    if s <= 0.0:
        return vec
    inv = 1.0 / s
    return [float(x * inv) for x in vec]


def merge_embedding_indexed_rows(
    batch_size: int,
    items: Sequence[Any],
) -> list[list[float]]:
    """
    按 ``index`` 将带 embedding 字段的对象合并为与批次同序的矩阵行。

    支持 OpenAI SDK 对象（``.index`` / ``.embedding``）或 dict（``index`` / ``embedding``）。
    """
    n = batch_size
    rows: list[list[float] | None] = [None] * n
    for obj in items:
        if isinstance(obj, Mapping):
            idx = int(obj["index"])
            emb = list(obj["embedding"])
        else:
            idx = int(obj.index)
            emb = list(obj.embedding)
        if not (0 <= idx < n):
            raise RuntimeError(f"embedding 响应 index={idx} 超出批次长度 {n}")
        rows[idx] = emb
    missing = [i for i, r in enumerate(rows) if r is None]
    if missing:
        raise RuntimeError(f"embedding 响应缺少 index: {missing[:10]}...")
    return [r for r in rows if r is not None]  # type: ignore[return-value]
