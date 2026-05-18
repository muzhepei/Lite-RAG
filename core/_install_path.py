# -*- coding: utf-8 -*-
"""
仅使用标准库：在导入 ``es2vec`` 包之前把项目根目录加入 ``sys.path``。

各入口脚本在文件顶部调用 :func:`bootstrap_script`，避免「未加入 path 就无法 import es2vec」的循环依赖。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def install(caller_file: str | Path) -> tuple[Path, Path]:
    """
    定位 ``es2vec`` 包根目录，并将包的上级目录加入 ``sys.path``（供 ``import es2vec``）。

    入口脚本可在**项目根目录**直接执行，例如 ``python cli/index_corpus.py``。

    Returns:
        (project_root, pkg_dir)，二者均为包根目录
    """
    p = Path(caller_file).resolve().parent
    for _ in range(12):
        if p.name == "es2vec" and (p / "__init__.py").is_file():
            pkg_dir = p
            import_root = pkg_dir.parent
            if str(import_root) not in sys.path:
                sys.path.insert(0, str(import_root))
            return pkg_dir, pkg_dir
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError(
        "无法定位 es2vec 包目录（请从含 es2vec/__init__.py 的目录结构运行脚本）"
    )


def bootstrap_script(caller_file: str | Path) -> tuple[Path, Path]:
    """
    供 ``cli/``、``apps/``、``examples/`` 等子目录下的脚本在顶部调用。

    与 :func:`install` 等价，名称更贴近「入口脚本引导」语义。
    """
    return install(caller_file)


def bootstrap_from_caller_module() -> tuple[Path, Path]:
    """
    从 ``core/_install_path.py`` 被 importlib 加载后，由入口脚本调用并传入 ``__file__`` 更稳妥；
    本函数保留给需要零参数引导的场景（一般不直接使用）。
    """
    raise RuntimeError("请使用 bootstrap_script(__file__)")


def load_and_bootstrap(caller_file: str | Path) -> tuple[Path, Path]:
    """在无法 ``from es2vec.core._install_path import install`` 时，用 importlib 加载本模块并引导。"""
    p = Path(caller_file).resolve().parent
    for _ in range(12):
        script = p / "core" / "_install_path.py"
        if p.name == "es2vec" and script.is_file():
            spec = importlib.util.spec_from_file_location("_es2vec_install_path", script)
            mod = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(mod)
            return mod.install(caller_file)
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError("无法定位 es2vec/core/_install_path.py")
