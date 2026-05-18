# -*- coding: utf-8 -*-
"""
``es2vec`` 包路径常量；在已能 ``import es2vec`` 之后使用。

入口脚本请先用 ``core._install_path.install(__file__)`` 完成路径引导。
"""
from __future__ import annotations

from pathlib import Path

# 本文件位于 es2vec/core/，上一级为包根目录（即项目根）
PKG_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PKG_DIR
