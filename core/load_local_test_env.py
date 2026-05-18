# -*- coding: utf-8 -*-
"""
加载 ``es2vec`` 包根目录下的 ``local_test.env``，供本地测试使用。

规则：
  - 仅当文件存在时加载；
  - 不覆盖已在进程环境中设置的变量（便于 CI / 临时覆盖）；
  - 支持 ``#`` 行注释与空行；``KEY=value``，首尾空白会去掉。

见包根 ``local_test.env``（勿提交真实密码时可改用 ``local_test.env.example`` 复制）。
"""
from __future__ import annotations

import os
from pathlib import Path


def _pkg_dir() -> Path:
    """``es2vec`` 包根目录（含 ``local_test.env``）。"""
    p = Path(__file__).resolve().parent
    return p.parent if p.name == "core" else p


def load_local_test_env(filename: str = "local_test.env") -> Path | None:
    """
    将包根目录下 ``<filename>`` 中的键值写入 ``os.environ``（不覆盖已有键）。

    Returns:
        已加载的文件路径；若文件不存在则返回 None。
    """
    path = _pkg_dir() / filename
    if not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8-sig")
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        key, _, rest = s.partition("=")
        key = key.strip()
        val = rest.strip()
        if not key:
            continue
        if key not in os.environ:
            os.environ[key] = val
    return path
