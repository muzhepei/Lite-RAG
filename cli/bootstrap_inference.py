# -*- coding: utf-8 -*-
"""
创建（或确认已存在）Elasticsearch 内置 multilingual E5 的 text_embedding 推理端点，
并轮询直至底层 trained model 部署完成。

本脚本仅在集群具备 **Inference API 许可** 且需使用 **ES 托管推理** 时需要执行。

若无许可证：请改用本地向量流程，**无需**运行本脚本——直接::

    python cli/index_corpus.py --input ... --index ...
    python cli/search_hybrid.py --index ... --q ...

用法（在项目根目录，有许可时）::

    pip install -r requirements.txt
    # 配置 ES_HOST / ES_USER / ES_PASSWORD 等（见 core/es_client.py、local_test.env.example）
    # 若 HTTPS 使用自签证书：PowerShell 可先执行
    #   $env:ES_VERIFY_CERTS='false'
    # 或使用本脚本参数：python cli/bootstrap_inference.py --no-verify-ssl
    python cli/bootstrap_inference.py

首次拉取模型可能较久，脚本使用较长超时；若仍超时，可再次运行（幂等）。

说明：Elasticsearch 的 Inference API（本脚本使用的 text_embedding 服务端点）需要集群许可证包含
对应功能；若出现 ``non-compliant for [inference]``，需由集群管理员升级/续期 Elastic 订阅或
申请试用许可证，无法仅靠本仓库脚本绕过。
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
import os
import sys
from pathlib import Path

from elastic_transport import TlsError
from elasticsearch.exceptions import AuthorizationException

from es2vec.core.es_client import get_es
from es2vec.core.config import DEFAULT_INFERENCE_ID, DEFAULT_MODEL_ID
from es2vec.core.inference_utils import (
    ensure_inference_endpoint,
    infer_text_embeddings,
    probe_cluster,
    wait_trained_model_allocated,
)


def _any_ml_role(probe: dict) -> bool:
    for n in probe.get("nodes_roles") or []:
        if "ml" in (n.get("roles") or []):
            return True
    return False


def main() -> None:
    p = argparse.ArgumentParser(description="创建/确认 ES text_embedding 推理端点（multilingual E5）")
    p.add_argument("--inference-id", default=DEFAULT_INFERENCE_ID, help="推理端点 ID")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="底层 trained model_id")
    p.add_argument("--max-wait", type=float, default=1800.0, help="等待模型部署的最长时间（秒）")
    p.add_argument("--skip-wait", action="store_true", help="仅创建端点，不等待 fully_allocated")
    p.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="不校验 TLS 证书（自签/内网 CA）；等价于环境变量 ES_VERIFY_CERTS=false，仅建议在开发环境使用",
    )
    args = p.parse_args()

    if args.no_verify_ssl:
        os.environ["ES_VERIFY_CERTS"] = "false"

    es = get_es()
    try:
        probe = probe_cluster(es)
    except TlsError as exc:
        print(
            "\nTLS 连接失败（常见于自签证书或企业内网 CA）。可选处理方式：\n"
            "  1) 开发环境：重新运行并加上参数  --no-verify-ssl\n"
            "  2) PowerShell：$env:ES_VERIFY_CERTS='false'  后再运行本脚本\n"
            "  3) 生产推荐：设置环境变量 ES_CA_CERTS 为 PEM 格式的 CA 证书路径，并保持校验开启\n",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    print("=== 集群探测 ===")
    print(json.dumps(probe, ensure_ascii=False, indent=2))
    if not _any_ml_role(probe):
        print(
            "\n警告：未发现带 ml 角色的节点；内置模型推理可能失败。"
            "请在 elasticsearch.yml 中为节点添加 ml 角色并重启后再试。\n"
        )

    print("\n=== 创建或确认 inference 端点 ===")
    try:
        out = ensure_inference_endpoint(
            es,
            inference_id=args.inference_id,
            model_id=args.model_id,
            timeout="30m",
        )
    except AuthorizationException as exc:
        msg = str(exc).lower()
        if "non-compliant" in msg and "inference" in msg:
            print(
                "\n集群返回 403：当前许可证不允许使用 Inference API（text_embedding 服务端点）。\n"
                "这是 Elasticsearch 商业功能与许可证策略限制，不是本脚本或客户端库的缺陷。\n\n"
                "可行方向：\n"
                "  · 由运维在 Kibana「Stack Management → License」查看并升级/续期 Elastic 订阅，"
                "或申请官方试用许可；\n"
                "  · 若仅做 POC：使用带完整功能的试用集群或 Elastic Cloud 对应套餐；\n"
                "  · 若无法获得 Inference 许可：改为在应用侧用本地/外部模型生成向量，"
                "仅把 ES 当作向量存储与检索（与本仓库「ES 内建 E5」路径不同）。\n",
                file=sys.stderr,
            )
        else:
            print(f"\nElasticsearch 返回 403：{exc}\n", file=sys.stderr)
        raise SystemExit(3) from exc
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))

    if args.skip_wait:
        print("\n已跳过部署等待。可用 Kibana Stack Monitoring 或 _ml/trained_models 查看进度。")
        return

    print("\n=== 等待模型部署（fully_allocated）===")
    wait = wait_trained_model_allocated(
        es, args.model_id, max_wait_seconds=args.max_wait
    )
    print(json.dumps(wait, ensure_ascii=False, indent=2, default=str))
    if not wait.get("ready"):
        raise SystemExit("模型在超时时间内未就绪，请增大 --max-wait 或稍后重试本脚本。")

    print("\n=== 试推理（验证端点）===")
    vecs = infer_text_embeddings(es, args.inference_id, ["这是一句中文测试。"], input_type="ingest")
    print("试推理维度:", len(vecs[0]) if vecs else 0)


if __name__ == "__main__":
    main()
