# -*- coding: utf-8 -*-
"""
与 Elasticsearch Inference / ML 相关的辅助函数：集群探测、创建端点、解析向量、阻塞等待部署。
"""
from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError, RequestError

from .config import DEFAULT_INFERENCE_ID, DEFAULT_MODEL_ID


def probe_cluster(es: Elasticsearch) -> dict[str, Any]:
    """
    只读探测：版本号、各节点角色、是否安装 smartcn 插件（用于中文分词全文腿）。

    Returns:
        结构化信息字典，便于打印或记录。
    """
    info = es.info()
    version = (info.get("version") or {}).get("number", "?")
    nodes_resp = es.nodes.info(metric="roles", flat_settings=True)
    nodes = nodes_resp.get("nodes") or {}
    roles_summary: list[dict[str, Any]] = []
    for nid, meta in nodes.items():
        roles = list(meta.get("roles") or [])
        roles_summary.append({"node_id": nid, "name": meta.get("name"), "roles": roles})

    smartcn_any = False
    try:
        plug_resp = es.nodes.info(metric="plugins")
        for _nid, meta in (plug_resp.get("nodes") or {}).items():
            for p in meta.get("plugins", []) or []:
                if "analysis-smartcn" in (p.get("name") or ""):
                    smartcn_any = True
                    break
    except RequestError:
        pass

    ml_ok = False
    ml_detail: str | dict[str, Any] = "skipped"
    try:
        ml_info = es.ml.info()
        ml_ok = True
        ml_detail = {"jobs": (ml_info.get("limits") or {}).get("max_jobs", "?")}
    except RequestError as e:
        ml_detail = str(e)

    return {
        "version": version,
        "nodes_roles": roles_summary,
        "has_ml_info": ml_ok,
        "ml_info_detail": ml_detail,
        "smartcn_plugin": smartcn_any,
    }


def _has_ml_role(probe: Mapping[str, Any]) -> bool:
    for n in probe.get("nodes_roles") or []:
        if "ml" in (n.get("roles") or []):
            return True
    return False


def ensure_inference_endpoint(
    es: Elasticsearch,
    *,
    inference_id: str = DEFAULT_INFERENCE_ID,
    model_id: str = DEFAULT_MODEL_ID,
    num_allocations: int = 1,
    num_threads: int = 1,
    timeout: str = "30m",
) -> dict[str, Any]:
    """
    幂等创建 Elasticsearch service 的 text_embedding 推理端点；已存在则跳过创建。

    首次创建会触发模型下载，可能耗时数分钟；请使用较长 client timeout（见各脚本）。
    """
    try:
        got = es.inference.get(
            task_type="text_embedding", inference_id=inference_id, pretty=True
        )
        return {"status": "exists", "get": dict(got.body if hasattr(got, "body") else got)}
    except NotFoundError:
        pass

    resp = es.inference.put_elasticsearch(
        task_type="text_embedding",
        elasticsearch_inference_id=inference_id,
        service="elasticsearch",
        service_settings={
            "model_id": model_id,
            "num_allocations": num_allocations,
            "num_threads": num_threads,
        },
        timeout=timeout,
    )
    return {"status": "created", "put": dict(resp.body if hasattr(resp, "body") else resp)}


def wait_trained_model_allocated(
    es: Elasticsearch,
    model_id: str,
    *,
    poll_seconds: float = 5.0,
    max_wait_seconds: float = 1800.0,
) -> dict[str, Any]:
    """
    轮询 ML trained model 统计信息，直到 deployment_stats 显示 fully_allocated 或超时。

    说明：大模型首次下载时 allocation 可能长时间处于 starting。
    """
    deadline = time.monotonic() + max_wait_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            stats = es.ml.get_trained_models_stats(model_id=model_id)
            last = dict(stats.body if hasattr(stats, "body") else stats)
            models = last.get("trained_model_stats") or []
            if models:
                deployment = (models[0].get("deployment_stats") or {}) if models else {}
                state = deployment.get("state")
                ac = deployment.get("allocation_count")
                tac = deployment.get("target_allocation_count")
                allocated = (
                    state == "fully_allocated"
                    or (
                        ac is not None
                        and tac is not None
                        and tac > 0
                        and ac == tac
                    )
                )
                if allocated:
                    return {"ready": True, "deployment_stats": deployment, "raw": last}
        except NotFoundError:
            # 模型尚未注册到 ML，继续等
            pass
        time.sleep(poll_seconds)
    return {"ready": False, "timeout_seconds": max_wait_seconds, "last": last}


def parse_text_embedding_inference_response(body: Any) -> list[list[float]]:
    """
    将 Inference API 返回体解析为「每条输入对应一条 embedding」的列表。

    兼容多种返回形状（不同小版本 / 服务实现略有差异）。
    """
    if hasattr(body, "body"):
        body = body.body
    if not isinstance(body, dict):
        return []

    # 形状 A：inference_results[].predicted_value
    ir = body.get("inference_results")
    if isinstance(ir, list) and ir:
        out: list[list[float]] = []
        for row in ir:
            if not isinstance(row, dict):
                continue
            vec = row.get("predicted_value") or row.get("embedding")
            if isinstance(vec, list) and vec and isinstance(vec[0], (int, float)):
                out.append([float(x) for x in vec])
        if out:
            return out

    # 形状 B：text_embedding[].embedding
    te = body.get("text_embedding")
    if isinstance(te, list) and te:
        out2: list[list[float]] = []
        for item in te:
            if not isinstance(item, dict):
                continue
            vec = item.get("embedding") or item.get("predicted_value")
            if isinstance(vec, list) and vec and isinstance(vec[0], (int, float)):
                out2.append([float(x) for x in vec])
        if out2:
            return out2

    return []


def infer_text_embeddings(
    es: Elasticsearch,
    inference_id: str,
    texts: Sequence[str],
    *,
    input_type: str | None = "ingest",
) -> list[list[float]]:
    """
    调用 POST /_inference/text_embedding/<id>，对一批文本生成向量。

    Args:
        es: 客户端。
        inference_id: 推理端点 ID。
        texts: 文本列表（可为单条）。
        input_type: E5 类模型支持 ingest / search 等；索引文档用 ingest，查询用 search。
    """
    if not texts:
        return []
    kwargs: dict[str, Any] = {
        "inference_id": inference_id,
        "input": list(texts),
        "task_type": "text_embedding",
    }
    if input_type:
        kwargs["input_type"] = input_type
    resp = es.inference.inference(**kwargs)
    body = resp.body if hasattr(resp, "body") else resp
    vecs = parse_text_embedding_inference_response(body)
    if len(vecs) != len(texts):
        raise RuntimeError(
            f"推理返回向量条数 {len(vecs)} 与输入条数 {len(texts)} 不一致，原始响应键: "
            f"{list(body.keys()) if isinstance(body, dict) else type(body)}"
        )
    return vecs


def cluster_has_smartcn(es: Elasticsearch) -> bool:
    """若任一节点安装 analysis-smartcn，返回 True。"""
    return bool(probe_cluster(es).get("smartcn_plugin"))
