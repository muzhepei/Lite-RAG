# -*- coding: utf-8 -*-
"""
在三国演义（chunk 或章 JSONL）上训练 **词级** Word2Vec，用于 ``most_similar`` 类比（如 刘备+诸葛亮）。

与 ES 中 E5 句向量无关；模型存为 ``KeyedVectors``（``.kv``）。

用法::

    cd C:\\Users\\Asus\\Desktop\\ai\\14\\es2vec
    python -m es2vec.three_kingdoms_ext.train_w2v train \\
        --input three_kingdoms_ext/out/three_kingdoms_chunks.jsonl \\
        --output three_kingdoms_ext/out/sg_w2v.kv

    python -m es2vec.three_kingdoms_ext.train_w2v analog \\
        --model three_kingdoms_ext/out/sg_w2v.kv \\
        --plus 刘备 --plus 诸葛亮 --topn 12
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _bootstrap_es2vec_path() -> None:
    p = Path(__file__).resolve().parent
    for _ in range(12):
        script = p / "core" / "_install_path.py"
        if p.name == "es2vec" and script.is_file():
            spec = importlib.util.spec_from_file_location("_es2vec_install_path", script)
            mod = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(mod)
            mod.install(__file__)
            return
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError("无法定位 es2vec 包（需要 es2vec/core/_install_path.py）")


_bootstrap_es2vec_path()


import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import importlib.util
import sys
from pathlib import Path

_ip = Path(__file__).resolve().parent
for _ in range(12):
    _script = _ip / "_install_path.py"
    if _ip.name == "es2vec" and _script.is_file():
        _spec = importlib.util.spec_from_file_location("es2vec_install_path", _script)
        _mod = importlib.util.module_from_spec(_spec)
        assert _spec.loader is not None
        _spec.loader.exec_module(_mod)
        _mod.install(__file__)
        break
    if _ip.parent == _ip:
        raise SystemExit("无法定位 es2vec/_install_path.py")
    _ip = _ip.parent
else:
    raise SystemExit("无法定位 es2vec/_install_path.py")


def _iter_jsonl_texts(path: Path) -> Iterator[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 第 {line_no} 行解析失败: {e}") from e
            t = obj.get("text")
            if isinstance(t, str) and t.strip():
                yield t.strip()


def _sentences_from_texts(texts: Iterator[str]) -> list[list[str]]:
    import jieba

    out: list[list[str]] = []
    for t in texts:
        # 去掉单字，减轻「的、之」等对人物类比词的干扰
        words = [w for w in jieba.cut(t, cut_all=False) if len(w.strip()) >= 2]
        if words:
            out.append(words)
    return out


def cmd_train(args: argparse.Namespace) -> None:
    from gensim.models import Word2Vec

    if not args.input.is_file():
        raise SystemExit(f"文件不存在: {args.input}")
    sents = _sentences_from_texts(_iter_jsonl_texts(args.input))
    if not sents:
        raise SystemExit("未从 JSONL 读取到任何 text")
    print(f"句/段条数={len(sents)}，开始训练 Word2Vec ...")
    model = Word2Vec(
        sents,
        vector_size=int(args.vector_size),
        window=int(args.window),
        min_count=int(args.min_count),
        workers=int(args.workers),
        epochs=int(args.epochs),
        sg=int(args.sg),
        seed=42,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.wv.save(str(out))
    print(f"已保存 KeyedVectors -> {out}")


def cmd_analog(args: argparse.Namespace) -> None:
    from gensim.models import KeyedVectors

    mpath = Path(args.model)
    if not mpath.is_file():
        raise SystemExit(f"模型文件不存在: {mpath}")
    kv = KeyedVectors.load(str(mpath), mmap="r")
    plus = [p.strip() for p in args.plus if p.strip()]
    minus = [p.strip() for p in args.minus if p.strip()]
    if not plus:
        raise SystemExit("至少指定一个 --plus")
    for w in plus + minus:
        if w not in kv:
            raise SystemExit(f"词不在词表中: {w!r}")
    res = kv.most_similar(positive=plus, negative=minus, topn=int(args.topn))
    label = " + ".join(plus)
    if minus:
        label += " - (" + " + ".join(minus) + ")"
    print(f"most_similar({label!r}) top {args.topn}:")
    for w, score in res:
        print(f"  {w}\t{score:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="三国演义词向量：训练 / 类比查询")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="从 JSONL 的 text 字段训练 Word2Vec 并保存 .kv")
    pt.add_argument("--input", type=Path, required=True)
    pt.add_argument("--output", type=Path, required=True, help="输出 .kv 路径")
    pt.add_argument("--vector-size", type=int, default=128)
    pt.add_argument("--window", type=int, default=5)
    pt.add_argument("--min-count", type=int, default=2)
    pt.add_argument("--epochs", type=int, default=8)
    pt.add_argument("--workers", type=int, default=4)
    pt.add_argument("--sg", type=int, default=1, help="1=skip-gram，0=CBOW")
    pt.set_defaults(func=cmd_train)

    pa = sub.add_parser("analog", help="KeyedVectors.most_similar 查询")
    pa.add_argument("--model", type=Path, required=True)
    pa.add_argument("--plus", action="append", default=[], metavar="词")
    pa.add_argument("--minus", action="append", default=[], metavar="词")
    pa.add_argument("--topn", type=int, default=15)
    pa.set_defaults(func=cmd_analog)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
