# -*- coding: utf-8 -*-
"""
从 proto 生成 Python gRPC stub。

grpcio-tools: https://grpc.github.io/grpc/python/grpc_tools.html

用法（项目根目录）::

    pip install grpcio-tools
    python scripts/gen_grpc.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    proto = root / "proto" / "es2vec_search.proto"
    out_dir = root / "grpc_gen"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not proto.is_file():
        print(f"未找到 proto: {proto}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{root / 'proto'}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        str(proto),
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=str(root))

    grpc_py = out_dir / "es2vec_search_pb2_grpc.py"
    if grpc_py.is_file():
        text = grpc_py.read_text(encoding="utf-8")
        text = text.replace(
            "import es2vec_search_pb2 as es2vec__search__pb2",
            "from es2vec.grpc_gen import es2vec_search_pb2 as es2vec__search__pb2",
        )
        grpc_py.write_text(text, encoding="utf-8")

    init_py = out_dir / "__init__.py"
    if not init_py.exists():
        init_py.write_text(
            '# -*- coding: utf-8 -*-\n"""gRPC 生成代码（勿手改 *_pb2*.py）。"""\n',
            encoding="utf-8",
        )

    print(f"已生成: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
