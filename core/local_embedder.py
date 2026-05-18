# -*- coding: utf-8 -*-
"""
本地多语言 E5 向量编码（Sentence-Transformers），用于无 Elasticsearch Inference 许可的场景。

与 Elastic 内置 multilingual-e5-small 对齐的约定：
  - 文档/段落侧使用前缀 ``passage: ``，查询侧使用 ``query: ``；
  - L2 归一化后与 ES ``dense_vector`` + ``cosine`` 相似度一致。

环境变量（可选，默认值见 ``es2vec.config``）：
  ES2VEC_LOCAL_MODEL  HuggingFace 模型 id 或本地目录路径。
  ES2VEC_HF_LOCAL_FILES_ONLY  未设置时默认仅本地；需联网下载时显式设为 0/false。
  ES2VEC_HF_CACHE_FOLDER  自定义 HF 缓存目录。
  HF_ENDPOINT  例如 ``https://hf-mirror.com``，在访问 huggingface.co 不稳定时使用镜像。
"""
from __future__ import annotations

import os
from typing import Any, Sequence

from es2vec.core.config import DEFAULT_LOCAL_MODEL, env_hf_local_files_only

# intfloat 系列 E5 在检索任务中的通用前缀（与官方推理脚本一致）
_PREFIX_PASSAGE = "passage: "
_PREFIX_QUERY = "query: "


def _offline_env_true(name: str) -> bool:
    """识别 HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE 等常见取值。"""
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _sentence_transformer_load_kwargs() -> dict[str, Any]:
    """
    从环境变量构造 ``SentenceTransformer`` 的加载参数。

    ``local_files_only`` 为 True 时不访问网络，要求模型已在 HF 缓存或
    ``ES2VEC_LOCAL_MODEL`` 指向本地目录。未设置 ``ES2VEC_HF_LOCAL_FILES_ONLY`` 时默认 True。
    """
    kwargs: dict[str, Any] = {}
    if (
        env_hf_local_files_only()
        or _offline_env_true("HF_HUB_OFFLINE")
        or _offline_env_true("TRANSFORMERS_OFFLINE")
    ):
        kwargs["local_files_only"] = True
    cache_folder = os.environ.get("ES2VEC_HF_CACHE_FOLDER", "").strip()
    if cache_folder:
        kwargs["cache_folder"] = cache_folder
    return kwargs


class LocalEmbedder:
    """封装 ``SentenceTransformer``，按批编码并返回 Python ``list[float]`` 列表。"""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = (model_name or DEFAULT_LOCAL_MODEL).strip()
        # 延迟 import，便于仅跑 ES 脚本且未安装 torch 时其它模块仍可加载
        from sentence_transformers import SentenceTransformer

        load_kw = _sentence_transformer_load_kwargs()
        try:
            self._model = SentenceTransformer(self.model_name, **load_kw)
        except Exception as exc:
            # Windows / 国内网络下常见：SSL UNEXPECTED_EOF、代理中断；重试耗尽后 httpx 可能报 client closed
            hint = (
                "加载本地向量模型失败。\n"
                "若报错含 SSL、huggingface、EOF、client has been closed，多为访问 Hugging Face 不稳定。\n"
                "可任选其一：\n"
                "  (1) PowerShell 使用镜像后再运行："
                "$env:HF_ENDPOINT='https://hf-mirror.com'\n"
                "  (2) 用浏览器或另一台机器下载模型目录后，设置 ES2VEC_LOCAL_MODEL 为该文件夹路径。\n"
                "  (3) 未设置 ES2VEC_HF_LOCAL_FILES_ONLY 时默认仅读本地；若需联网拉模型请设："
                "$env:ES2VEC_HF_LOCAL_FILES_ONLY='0'\n"
                f"当前 model={self.model_name!r}，local_files_only={load_kw.get('local_files_only', False)}。\n"
                "原始异常："
            )
            raise RuntimeError(hint + repr(exc)) from exc

    @property
    def embedding_dim(self) -> int:
        """当前模型输出维度（如 multilingual-e5-small 为 384）。"""
        m = self._model
        # sentence-transformers 新版本将方法重命名为 get_embedding_dimension
        newer = getattr(m, "get_embedding_dimension", None)
        if callable(newer):
            return int(newer())
        return int(m.get_sentence_embedding_dimension())

    def encode_passages(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        """对语料/文档批量编码（带 passage 前缀 + 归一化）。"""
        if not texts:
            return []
        prefixed = [_PREFIX_PASSAGE + (t or "") for t in texts]
        return self._encode_prefixed(prefixed, batch_size=batch_size, show_progress_bar=show_progress_bar)

    def encode_queries(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        """对查询批量编码（带 query 前缀 + 归一化）。"""
        if not texts:
            return []
        prefixed = [_PREFIX_QUERY + (t or "") for t in texts]
        return self._encode_prefixed(prefixed, batch_size=batch_size, show_progress_bar=show_progress_bar)

    def _encode_prefixed(
        self,
        prefixed: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        emb = self._model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return [row.astype(float).tolist() for row in emb]


_cache: dict[str, LocalEmbedder] = {}


def get_local_embedder(model_name: str | None = None) -> LocalEmbedder:
    """
    按模型名缓存单例，避免重复加载权重。

    Args:
        model_name: 覆盖默认；为 None 时使用 ``DEFAULT_LOCAL_MODEL``。
    """
    key = (model_name or DEFAULT_LOCAL_MODEL).strip()
    if key not in _cache:
        _cache[key] = LocalEmbedder(key)
    return _cache[key]


def clear_embedder_cache() -> None:
    """测试或切换模型时可清空缓存。"""
    _cache.clear()
