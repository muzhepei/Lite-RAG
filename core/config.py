# -*- coding: utf-8 -*-
"""
es2vec 默认常量。

可通过环境变量覆盖（便于 CI / 多环境）：
  ES2VEC_INFERENCE_ID   默认 es2vec_multilingual_e5（仅 --use-es-inference 时）
  ES2VEC_MODEL_ID       默认 .multilingual-e5-small-elasticsearch（同上）
  ES2VEC_INDEX           默认 es2vec_corpus
  ES2VEC_LOCAL_MODEL     本地向量模型 id 或路径，默认 intfloat/multilingual-e5-small
  ES2VEC_HF_LOCAL_FILES_ONLY  未设置时默认为 1（仅从本机加载）；设为 0/false 时允许访问 Hub 下载
  ES2VEC_HF_CACHE_FOLDER  可选，覆盖 SentenceTransformer 的 cache_folder（大模型放其它盘）
  ES2VEC_JIEBA           设为 1/true 时，索引脚本默认写入 text_tokens（需安装 jieba）
  ES2VEC_SYNONYMS_SET    可选；非空时作为默认同义词集 ID（与 Synonyms API 的 id 一致，需 8.10+）

  混合检索（--no-rrf / Web 默认加权模式，缓解人名误排）：
  ES2VEC_VEC_WEIGHT        向量余弦权重，Web 未设时默认 0.85；CLI 默认 0.7
  ES2VEC_KW_WEIGHT         关键词 BM25 权重，Web 未设时默认 0.15；CLI 默认 0.3
  ES2VEC_KW_SAT            BM25 saturation 分母，Web 未设时默认 25；CLI 默认 15
  ES2VEC_KW_NORM           saturation | log1p | raw
  ES2VEC_NAME_RERANK       1/true 强制按查询词密度二阶段重排；0/false 关闭
  ES2VEC_NAME_RERANK_AUTO  未设 ES2VEC_NAME_RERANK 时：1（默认）对 ≤4 字且无空格的查询自动重排
  ES2VEC_NAME_RERANK_POOL  重排候选池大小，默认 50

  OpenAI 兼容 Embeddings（--use-openai-compatible-embedding，与本地/ES Inference 三选一）：
  ES2VEC_OPENAI_BASE_URL   未设置时默认魔搭推理网关；若仅设置了 DASHSCOPE_API_KEY（百炼）且未改此项，会自动改为百炼 compatible-mode 地址
  ES2VEC_DASHSCOPE_BASE_URL  可选；自动走百炼时覆盖默认 ``https://dashscope.aliyuncs.com/compatible-mode/v1``（新加坡域见阿里云文档）
  ES2VEC_DASHSCOPE_EMBEDDING_MODEL  百炼 embedding 模型；**走百炼网关时优先于** ``ES2VEC_OPENAI_EMBEDDING_MODEL``；未设时默认 ``qwen3-vl-embedding``
  ES2VEC_DASHSCOPE_EMBEDDING_BATCH_MAX  百炼兼容 ``/embeddings`` 单次 input 条数上限，默认 10（超出会 400）
  ES2VEC_DASHSCOPE_MULTIMODAL_API_BASE  百炼原生多模态 API 根路径（含 ``/api/v1``）；``qwen3-vl-embedding`` 走此端点
  ES2VEC_DASHSCOPE_MULTIMODAL_BATCH_MAX  原生 multimodal-embedding 单次 contents 条数上限，默认 20
  DASHSCOPE_API_KEY  百炼 API Key（与魔搭 Token 不同）；按序兼容 API_KEY、OPENAI_API_KEY、MODELSCOPE_API_KEY
  MODELSCOPE_API_KEY  仅当 ``ES2VEC_OPENAI_BASE_URL`` 指向 ``api-inference.modelscope.cn`` 时使用魔搭 Token
  勿在值前写 ``Bearer ``；未设密钥时回退占位符 ``----``
  ES2VEC_OPENAI_EMBEDDING_MODEL  魔搭/自定义网关 embedding 模型；百炼场景请用 ``ES2VEC_DASHSCOPE_EMBEDDING_MODEL``
  ES2VEC_EMBEDDING_DIMS   默认 1024（qwen3-vl-embedding 亦支持 256/512/768/1536/2048/2560）；设为 0 则 API 用模型默认维（qwen3-vl 为 2560）
  ES2VEC_USE_OPENAI_COMPATIBLE_EMBEDDING  1/true 时 Web/gRPC 混合检索走云端 Embeddings（须与建索引时一致）

  RAG 对话（/api/v1/rag、cli/rag_chat.py；与 Embeddings 共用 ES2VEC_OPENAI_BASE_URL 与 API Key）：
  ES2VEC_CHAT_MODEL            对话模型 ID（chat/completions，非 embedding）；魔搭默认 Qwen/Qwen2.5-7B-Instruct；百炼自动路由默认 qwen3.6-plus
  ES2VEC_DASHSCOPE_CHAT_MODEL  可选；百炼自动路由时覆盖默认对话模型（否则用 ES2VEC_CHAT_MODEL 或上述默认）
  ES2VEC_CHAT_ENABLE_THINKING    1/true 时百炼 Qwen3 等开启深度思考（流式先出 reasoning_content，首字变慢）；RAG 默认 0
  ES2VEC_RAG_TOP_K             送入 LLM 的参考资料条数（章级去重后），默认 3
  ES2VEC_RAG_FETCH_K           ES 检索 chunk 池大小（章级聚合前），默认 20
  ES2VEC_RAG_MULTI_HIT_THRESHOLD  同章命中 chunk 数≥此值则用整章正文，默认 2
  ES2VEC_RAG_CHAPTER_SCORE_ALPHA  章级分数命中数 boost 系数，默认 0.3
  ES2VEC_RAG_MAX_CONTEXT_CHARS 参考资料写入 prompt 的最大字符，超出截断，默认 12000
  ES2VEC_RAG_MAX_TOKENS        chat/completions 的 max_tokens，默认 1024
  ES2VEC_CHAT_TEMPERATURE      生成温度，0 更确定，默认 0.3

  访问 Hugging Face 不稳定时（SSL EOF、超时），可在 shell 中设置官方 Hub 变量，例如：
    HF_ENDPOINT=https://hf-mirror.com
  或与 HF_HUB_OFFLINE=1 配合已缓存的模型使用。
"""
from __future__ import annotations

import os

from es2vec.core.load_local_test_env import load_local_test_env

# 须在读取 ES2VEC_INDEX 等变量之前加载，否则 Web/CLI 导入 config 时仍为默认值
load_local_test_env()

# Inference 端点 ID（勿与底层 model_id 相同）
DEFAULT_INFERENCE_ID = os.environ.get("ES2VEC_INFERENCE_ID", "es2vec_multilingual_e5").strip()

# Elastic 内置多语言 E5（含中文）；首次创建端点时会触发模型下载与部署
DEFAULT_MODEL_ID = os.environ.get(
    "ES2VEC_MODEL_ID", ".multilingual-e5-small-elasticsearch"
).strip()

DEFAULT_INDEX_NAME = os.environ.get("ES2VEC_INDEX", "es2vec_corpus").strip()

# Synonyms API（8.10+）同义词集 ID；空字符串表示不使用托管同义词
DEFAULT_SYNONYMS_SET_ID = os.environ.get("ES2VEC_SYNONYMS_SET", "").strip()

# 多数部署下 multilingual-e5-small 为 384 维；若不确定可在首次 infer 后自动探测
DEFAULT_VECTOR_DIMS = int(os.environ.get("ES2VEC_VECTOR_DIMS", "384"))

TEXT_FIELD = "text"
VECTOR_FIELD = "vector"
# 可选：jieba 分词后用空格拼接，供 standard 分析器做 BM25（无 smartcn 时尤有用）
TEXT_TOKEN_FIELD = "text_tokens"
# 章内 chunk 元数据（three_kingdoms_ext.chunk_corpus 输出；需 index_corpus --chunk-fields）
CHAPTER_ID_FIELD = "chapter_id"
CHUNK_INDEX_FIELD = "chunk_index"
# 人物侧索引（three_kingdoms_ext.entity_index）
CHARACTER_NAME_FIELD = "character"
CHARACTER_PROFILE_FIELD = "profile_text"

# 本地 Sentence-Transformers 模型（与 multilingual-e5-small 维度语义对齐）
DEFAULT_LOCAL_MODEL = os.environ.get(
    "ES2VEC_LOCAL_MODEL", "intfloat/multilingual-e5-small"
).strip()


def normalize_openai_compatible_api_key(raw: str) -> str:
    """
    规范化写入 OpenAI SDK 的 api_key：去首尾空白；若以 ``Bearer `` 开头则去掉
    （否则会变成 ``Bearer Bearer ...``，网关返回 401）。
    """
    s = (raw or "").strip()
    if len(s) >= 7 and s[:7].lower() == "bearer ":
        return s[7:].strip()
    return s


def _first_nonempty_env_with_source(names: tuple[str, ...]) -> tuple[str, str | None]:
    """返回 (值, 命中的环境变量名)；全空时 ``("", None)``。"""
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        t = raw.strip()
        if t:
            return t, name
    return "", None


# OpenAI 兼容 Embeddings：未设置或为空时使用下列默认值（与 local_test.env.example 一致）
_DEF_OPENAI_BASE = "https://api-inference.modelscope.cn/v1"
_DEF_DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEF_OPENAI_KEY = "----"
_DEF_OPENAI_MODEL = "Qwen/Qwen3-Embedding-8B"
# 百炼 OpenAI 兼容 text embedding：https://help.aliyun.com/zh/model-studio/developer-reference/embedding-interfaces-compatible-with-openai
# qwen3-vl-embedding 须走原生 Multimodal-Embedding API（见 dashscope_multimodal_embedder.py）
_DEF_DASHSCOPE_MODEL = "qwen3-vl-embedding"
_DEF_DASHSCOPE_MULTIMODAL_API_BASE = "https://dashscope.aliyuncs.com/api/v1"
_DASHSCOPE_MULTIMODAL_EMBEDDING_MODELS = frozenset({"qwen3-vl-embedding"})

_url_env_raw = os.environ.get("ES2VEC_OPENAI_BASE_URL")
_initial_openai_base = (_url_env_raw or _DEF_OPENAI_BASE).strip()
_url_env_explicit = bool((_url_env_raw or "").strip())

_key_raw, _key_env_source = _first_nonempty_env_with_source(
    ("DASHSCOPE_API_KEY", "API_KEY", "OPENAI_API_KEY", "MODELSCOPE_API_KEY")
)
_API_KEY_RAW = _key_raw or _DEF_OPENAI_KEY
API_KEY = normalize_openai_compatible_api_key(_API_KEY_RAW)
# 与阿里云百炼约定一致：以下为解析后的同一密钥（优先读环境变量 DASHSCOPE_API_KEY）
DASHSCOPE_API_KEY = API_KEY

_model_env_raw = os.environ.get("ES2VEC_OPENAI_EMBEDDING_MODEL")
_initial_openai_model = (_model_env_raw or _DEF_OPENAI_MODEL).strip()
_model_env_explicit = bool((_model_env_raw or "").strip())

_dashscope_key_only = (os.environ.get("DASHSCOPE_API_KEY") or "").strip()
_has_real_dashscope_key = bool(_dashscope_key_only and _dashscope_key_only != "----")


def _is_modelscope_default_base(url: str) -> bool:
    """是否为 config 默认魔搭网关（.env 误填此项时仍可按百炼 Key 自动纠正）。"""
    return url.strip().rstrip("/") == _DEF_OPENAI_BASE.rstrip("/")


# 百炼密钥 +（未显式改网关 或 仍为魔搭默认 URL）→ 自动改走 DashScope compatible-mode
OPENAI_EMBEDDING_ROUTE_AUTO_TO_DASHSCOPE = (
    _has_real_dashscope_key
    and (not _url_env_explicit or _is_modelscope_default_base(_initial_openai_base))
)
OPENAI_COMPATIBLE_BASE_URL = (
    (os.environ.get("ES2VEC_DASHSCOPE_BASE_URL") or _DEF_DASHSCOPE_BASE).strip()
    if OPENAI_EMBEDDING_ROUTE_AUTO_TO_DASHSCOPE
    else _initial_openai_base
)


def _is_dashscope_compatible_base(url: str) -> bool:
    """是否为百炼 OpenAI 兼容网关 base_url。"""
    u = url.strip().lower()
    return "dashscope.aliyuncs.com" in u or "dashscope-intl.aliyuncs.com" in u


def _is_dashscope_host(url: str) -> bool:
    """是否为百炼 DashScope 域名（兼容模式或原生 API）。"""
    return _is_dashscope_compatible_base(url) or "/api/v1" in url.strip().lower()


def is_dashscope_multimodal_embedding_model(model: str) -> bool:
    """是否应走百炼原生 ``multimodal-embedding`` API（首版仅 ``qwen3-vl-embedding``）。"""
    return model.strip() in _DASHSCOPE_MULTIMODAL_EMBEDDING_MODELS


def _derive_multimodal_api_base_from_compatible(compatible_base: str) -> str:
    """将 ``.../compatible-mode/v1`` 转为 ``.../api/v1``。"""
    u = compatible_base.strip().rstrip("/")
    suffix = "/compatible-mode/v1"
    if u.lower().endswith(suffix):
        return u[: -len(suffix)] + "/api/v1"
    if u.lower().endswith("/v1"):
        return u.rsplit("/v1", 1)[0] + "/api/v1"
    return _DEF_DASHSCOPE_MULTIMODAL_API_BASE


def dashscope_multimodal_api_base() -> str:
    """
    百炼原生多模态 Embedding API 根路径（含 ``/api/v1``，不含具体 service 路径）。

    文档：https://help.aliyun.com/zh/model-studio/multimodal-embedding-api-reference
    """
    explicit = (os.environ.get("ES2VEC_DASHSCOPE_MULTIMODAL_API_BASE") or "").strip()
    if explicit:
        return explicit.rstrip("/")
    dash_base = (os.environ.get("ES2VEC_DASHSCOPE_BASE_URL") or "").strip()
    if dash_base:
        return _derive_multimodal_api_base_from_compatible(dash_base)
    if _is_dashscope_compatible_base(OPENAI_COMPATIBLE_BASE_URL):
        return _derive_multimodal_api_base_from_compatible(OPENAI_COMPATIBLE_BASE_URL)
    return _DEF_DASHSCOPE_MULTIMODAL_API_BASE.rstrip("/")


_dsm_bs_raw = (os.environ.get("ES2VEC_DASHSCOPE_MULTIMODAL_BATCH_MAX") or "20").strip()
try:
    _dsm_bs = int(_dsm_bs_raw)
except ValueError:
    _dsm_bs = 20
DASHSCOPE_MULTIMODAL_EMBEDDING_MAX_BATCH = _dsm_bs if _dsm_bs > 0 else 20

DASHSCOPE_MULTIMODAL_EMBEDDING_PATH = (
    "/services/embeddings/multimodal-embedding/multimodal-embedding"
)


def should_use_dashscope_multimodal_embedding(model: str) -> bool:
    """
    是否对给定模型走百炼原生 multimodal-embedding（而非 compatible-mode /embeddings）。
    """
    if not is_dashscope_multimodal_embedding_model(model):
        return False
    if (os.environ.get("ES2VEC_DASHSCOPE_MULTIMODAL_API_BASE") or "").strip():
        return True
    if _has_real_dashscope_key and (
        OPENAI_EMBEDDING_ROUTE_AUTO_TO_DASHSCOPE
        or _is_dashscope_compatible_base(OPENAI_COMPATIBLE_BASE_URL)
    ):
        return True
    return False


def _is_modelscope_style_embedding_model(model: str) -> bool:
    """魔搭推理上的模型 id（如 Qwen/...），不能用于百炼 compatible-mode embeddings。"""
    m = model.strip()
    if m == _DEF_OPENAI_MODEL:
        return True
    return m.startswith("Qwen/")


def _resolve_openai_compatible_embedding_model() -> str:
    """
    解析 OpenAI 兼容 embedding 模型 id。

    百炼网关下优先级：``ES2VEC_DASHSCOPE_EMBEDDING_MODEL`` > 非魔搭风格的
    ``ES2VEC_OPENAI_EMBEDDING_MODEL`` > 百炼默认 ``qwen3-vl-embedding``。
    其它网关：``ES2VEC_OPENAI_EMBEDDING_MODEL`` > 魔搭默认。
    """
    dashscope_model_env = (os.environ.get("ES2VEC_DASHSCOPE_EMBEDDING_MODEL") or "").strip()
    on_dashscope = OPENAI_EMBEDDING_ROUTE_AUTO_TO_DASHSCOPE or _is_dashscope_compatible_base(
        OPENAI_COMPATIBLE_BASE_URL
    )

    if on_dashscope:
        if dashscope_model_env:
            return dashscope_model_env
        if _model_env_explicit and not _is_modelscope_style_embedding_model(_initial_openai_model):
            return _initial_openai_model
        return _DEF_DASHSCOPE_MODEL

    if _model_env_explicit:
        return _initial_openai_model
    return _DEF_OPENAI_MODEL


OPENAI_COMPATIBLE_EMBEDDING_MODEL = _resolve_openai_compatible_embedding_model()

_EDIMS_RAW = (os.environ.get("ES2VEC_EMBEDDING_DIMS") or "1024").strip()
if _EDIMS_RAW == "0":
    _openai_edims = 0
elif not _EDIMS_RAW:
    _openai_edims = 1024
else:
    try:
        _edv = int(_EDIMS_RAW)
    except ValueError:
        _openai_edims = 1024
    else:
        _openai_edims = _edv if _edv > 0 else 1024
OPENAI_EMBEDDING_DIMS = _openai_edims

_dsb_bs_raw = (os.environ.get("ES2VEC_DASHSCOPE_EMBEDDING_BATCH_MAX") or "10").strip()
try:
    _dsb_bs = int(_dsb_bs_raw)
except ValueError:
    _dsb_bs = 10
DASHSCOPE_EMBEDDING_MAX_BATCH = _dsb_bs if _dsb_bs > 0 else 10

# OpenAI 兼容 Chat（RAG 生成层）
_DEF_CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# 百炼 Chat Completions 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
_DEF_DASHSCOPE_CHAT_MODEL = "qwen3.6-plus"

_chat_model_env_raw = os.environ.get("ES2VEC_CHAT_MODEL")
_initial_chat_model = (_chat_model_env_raw or _DEF_CHAT_MODEL).strip()
_chat_model_env_explicit = bool((_chat_model_env_raw or "").strip())

OPENAI_COMPATIBLE_CHAT_MODEL = (
    (os.environ.get("ES2VEC_DASHSCOPE_CHAT_MODEL") or _DEF_DASHSCOPE_CHAT_MODEL).strip()
    if (
        OPENAI_EMBEDDING_ROUTE_AUTO_TO_DASHSCOPE
        and not _chat_model_env_explicit
        and _initial_chat_model == _DEF_CHAT_MODEL
    )
    else _initial_chat_model
)


def env_flag_true(name: str) -> bool:
    """解析 ES2VEC_JIEBA 等 1/true/on/yes。"""
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


def env_flag_false(name: str) -> bool:
    """解析 0/false/no/off。"""
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in ("0", "false", "no", "off")


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


# 混合检索默认权重（CLI；Web 见 apps/web_search_server.py 中的推荐默认）
DEFAULT_HYBRID_VEC_WEIGHT = 0.7
DEFAULT_HYBRID_KW_WEIGHT = 0.3
DEFAULT_HYBRID_KW_SAT = 15.0
# Web / 人名检索推荐
WEB_HYBRID_VEC_WEIGHT = 0.85
WEB_HYBRID_KW_WEIGHT = 0.15
WEB_HYBRID_KW_SAT = 25.0
DEFAULT_NAME_RERANK_POOL = 50

RAG_DEFAULT_TOP_K = env_int("ES2VEC_RAG_TOP_K", 3)
RAG_FETCH_K = env_int("ES2VEC_RAG_FETCH_K", 20)
RAG_MULTI_HIT_THRESHOLD = env_int("ES2VEC_RAG_MULTI_HIT_THRESHOLD", 2)
RAG_CHAPTER_SCORE_ALPHA = env_float("ES2VEC_RAG_CHAPTER_SCORE_ALPHA", 0.3)
RAG_MAX_CONTEXT_CHARS = env_int("ES2VEC_RAG_MAX_CONTEXT_CHARS", 12000)
RAG_CHAT_MAX_TOKENS = env_int("ES2VEC_RAG_MAX_TOKENS", 1024)
RAG_CHAT_TEMPERATURE = env_float("ES2VEC_CHAT_TEMPERATURE", 0.3)

# 仅检索：为命中附加整章正文（默认开启；设 ES2VEC_SEARCH_INCLUDE_CHAPTER=0 关闭）
def search_include_chapter_default() -> bool:
    raw = os.environ.get("ES2VEC_SEARCH_INCLUDE_CHAPTER", "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def chat_enable_thinking() -> bool:
    """
    百炼 Qwen3 深度思考（``enable_thinking``）。

    开启时流式响应先长时间输出 ``reasoning_content``，再输出正文 ``content``，
    若前端只展示 content，会感到「首 token 极慢」。RAG 场景默认关闭。
    """
    return env_flag_true("ES2VEC_CHAT_ENABLE_THINKING")


def resolve_name_rerank(query: str, explicit: bool | None = None) -> bool:
    """
    是否对命中结果做查询词密度重排。

    ``explicit`` 非 None 时优先；否则读 ``ES2VEC_NAME_RERANK``；
    未设置时默认对短人名查询自动启用（可用 ``ES2VEC_NAME_RERANK_AUTO=0`` 关闭）。
    """
    if explicit is not None:
        return explicit
    if env_flag_true("ES2VEC_NAME_RERANK"):
        return True
    if env_flag_false("ES2VEC_NAME_RERANK"):
        return False
    if env_flag_false("ES2VEC_NAME_RERANK_AUTO"):
        return False
    from es2vec.core.search_rerank import should_auto_name_rerank

    return should_auto_name_rerank(query)


def env_hf_local_files_only() -> bool:
    """
    ``ES2VEC_HF_LOCAL_FILES_ONLY``：进程环境里**未配置**（无此键或值为空）时视为 ``1``，
    即 ``SentenceTransformer(..., local_files_only=True)``，不访问 Hugging Face。

    需要首次在线下载时，请显式设为 ``0`` / ``false`` / ``no`` / ``off``。
    """
    raw = os.environ.get("ES2VEC_HF_LOCAL_FILES_ONLY")
    if raw is None:
        return True
    if not raw.strip():
        return True
    return env_flag_true("ES2VEC_HF_LOCAL_FILES_ONLY")
