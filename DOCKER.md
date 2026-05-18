# Docker 使用说明

> 主文档：[README.md — Docker 部署](README.md#docker-部署)。本文档与之同步，便于单独查阅与排错。

使用 Docker Compose 启动 **Elasticsearch 8.13**、**es2vec** CLI 与常驻 **Web** 服务，无需在宿主机单独安装 ES。

## 前置条件

- 已安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/)（Windows）或 Docker Engine + Compose
- 磁盘空间：ES 镜像约 600MB；首次构建 es2vec 镜像会下载 PyTorch 等依赖；首次索引会下载向量模型（写入 `hf_cache` 卷）
- **ECS / Linux 服务器**：`cp .env.example .env` 并编辑后启动；`sysctl -w vm.max_map_count=262144`（详见 [README — 云服务器部署](README.md#云服务器--ecs-部署)）

## 服务说明

| 服务 | 容器名 | 说明 |
|------|--------|------|
| `elasticsearch` | `es2vec-elasticsearch` | 单节点 ES 8.13，HTTP 无 TLS |
| `es2vec` | `es2vec-app` | 项目 Python 环境，通过 `docker compose run` 执行 CLI |
| `web` | `es2vec-web` | Web 混合检索（FastAPI），默认后台常驻 |
| `grpc` | `es2vec-grpc` | gRPC 混合检索，默认后台常驻 |

### 端口与账号（默认）

| 项 | 容器内 | 宿主机 |
|----|--------|--------|
| ES HTTP | `http://elasticsearch:9200` | `http://localhost:9200` |
| Web 检索 | `http://0.0.0.0:8765` | `http://localhost:8765` |
| gRPC 检索 | `0.0.0.0:50051` | `localhost:50051` |
| 用户 | `elastic` | 同左 |
| 密码 | `es2vec_dev` | 同左（可用 `ELASTIC_PASSWORD` 覆盖） |

## 快速开始

在项目根目录（含 `docker-compose.yml` 的 `es2vec` 目录）执行：

```powershell
cd path\to\es2vec

# 构建应用镜像
docker compose build

# 仅启动 Elasticsearch（后台）
docker compose up -d elasticsearch

# 等待 healthy 后检查连接
docker compose run --rm es2vec python scripts/docker_check_es.py

# 离线冒烟（不连 ES）
docker compose run --rm es2vec python cli/smoke_demo.py --offline
```

## 常用命令

### 启动与停止

```powershell
# 启动 ES
docker compose up -d elasticsearch

# 启动 Web 检索页（需 ES 已 healthy，且已建好索引）
docker compose up -d web

# 同时启动 ES + Web
docker compose up -d elasticsearch web

# 查看状态
docker compose ps

# 查看 ES / Web 日志
docker compose logs -f elasticsearch
docker compose logs -f web

# 停止 Web（保留 ES）
docker compose stop web

# 停止并删除容器（保留数据卷）
docker compose down

# 停止并删除容器 + 数据卷（清空索引与模型缓存）
docker compose down -v
```

### 索引与检索

首次建索引会下载 `intfloat/multilingual-e5-small`（写入 `hf_cache` 卷），耗时较长。

**章级索引**（推荐语料：`examples/data/three_kingdoms_by_chapter.jsonl`）：

```powershell
docker compose run --rm es2vec python cli/index_corpus.py `
  --input examples/data/three_kingdoms_by_chapter.jsonl `
  --index es2vec_corpus --recreate
```

**人名 / 片段检索（chunk 级，推荐）**：

```powershell
# 章级 JSONL → chunk JSONL
docker compose run --rm es2vec python examples/three_kingdoms_ext/chunk_corpus.py `
  --input examples/data/three_kingdoms_by_chapter.jsonl `
  --output examples/three_kingdoms_ext/out/three_kingdoms_chunks.jsonl

docker compose run --rm es2vec python cli/index_corpus.py `
  --input examples/three_kingdoms_ext/out/three_kingdoms_chunks.jsonl `
  --index es2vec_corpus_chunks --chunk-fields --recreate
```

**混合检索**（chunk 索引 + 人名查询示例）：

```powershell
docker compose run --rm es2vec python cli/search_hybrid.py `
  --index es2vec_corpus_chunks --q "刘备" --no-rrf `
  --vec-weight 0.85 --kw-weight 0.15 --kw-sat 25
```

**终端交互检索**：

```powershell
docker compose run --rm -it es2vec python apps/interactive_search.py `
  --index es2vec_corpus_chunks --no-rrf
```

**Web 检索页面**（推荐常驻服务）：

```powershell
# 默认索引 es2vec_corpus_chunks，端口 8765
docker compose up -d web

# 指定索引或宿主机端口（可选）
$env:ES2VEC_INDEX = "es2vec_corpus"
$env:ES2VEC_WEB_PORT = "8765"
docker compose up -d web
```

浏览器打开：**http://127.0.0.1:8765/**（健康检查：`/api/health`）。

一次性前台运行（调试用，退出即停）：

```powershell
docker compose run --rm -p 8765:8765 `
  -e ES2VEC_WEB_HOST=0.0.0.0 `
  -e ES2VEC_INDEX=es2vec_corpus_chunks `
  es2vec python apps/web_search_server.py
```

### 预处理语料（可选）

仓库已含清洗后的 `examples/data/three_kingdoms_by_chapter.jsonl`，一般可直接索引。若要从原始 TXT 重新生成：

```powershell
# 按「正文」分章 → examples/data/three_kingdoms_by_chapter.jsonl 同结构
docker compose run --rm es2vec python preprocess/txt_to_es_jsonl.py `
  --input examples/data/three_kingdoms.txt `
  --output examples/data/three_kingdoms_by_chapter.jsonl

# 按「分节阅读」切分 → examples/output/three_kingdoms.jsonl
docker compose run --rm es2vec python preprocess/split_txt_to_es_jsonl.py `
  --input examples/data/three_kingdoms.txt `
  --out-dir examples/output
```

### 同义词集（可选）

```powershell
docker compose run --rm es2vec python cli/put_synonyms_set.py `
  --input examples/data/synonyms_example.txt

# 建索引时加 --synonyms-set-id <上一步输出的 ID>
```

## 自定义密码

```powershell
$env:ELASTIC_PASSWORD = "你的密码"
docker compose up -d elasticsearch
```

`es2vec` 服务会通过 compose 自动使用同一密码连接 ES。

## 宿主机本地 Python 连接 Docker ES

容器内脚本无需额外配置；若在 **Windows 本机** 直接跑 `python cli/...`，可复制环境模板：

```powershell
copy local_test.env.docker.example local_test.env
# 编辑 ES_PASSWORD，与 ELASTIC_PASSWORD 一致
```

`local_test.env.docker.example` 中 `ES_HOST=http://localhost:9200` 指向 compose 映射端口。

## 数据卷

| 卷名 | 用途 |
|------|------|
| `es_data` | Elasticsearch 索引数据 |
| `hf_cache` | Hugging Face / SentenceTransformer 模型缓存 |

删除卷后需重新建索引；模型需重新下载。

## 故障排查

### 端口被占用

若启动报错 `bind: Only one usage of each socket address`：

- 检查 `9200` 是否被占用：`netstat -ano | findstr 9200`
- 或在 `docker-compose.yml` 中修改 `ports` 为 `"9202:9200"`，并同步修改 `local_test.env.docker.example` 中的 `ES_HOST`

### ES 未就绪

`docker compose run es2vec` 会等待 `elasticsearch` 健康检查通过。若长时间不健康：

```powershell
docker compose logs elasticsearch
```

常见原因：内存不足，可适当增大 Docker Desktop 内存上限。

### 客户端版本错误（400 media_type）

确保 `requirements.txt` 中 `elasticsearch` 为 `>=8.13,<9`，与 ES 8.13 服务端匹配。修改后执行：

```powershell
docker compose build --no-cache es2vec
```

## 相关文件

| 文件 | 说明 |
|------|------|
| `Dockerfile` | es2vec 应用镜像（Python 3.12 + CPU PyTorch） |
| `docker-compose.yml` | ES + es2vec 服务编排 |
| `.dockerignore` | 构建上下文排除项 |
| `scripts/docker_check_es.py` | ES 连通性检查脚本 |
| `.env.example` | Compose 部署模板（复制为 `.env`，含密码、RAG Key、端口等） |
| `local_test.env.docker.example` | 宿主机 Python 连 Docker ES（非容器内） |
| `apps/web_search_server.py` | Web 混合检索（FastAPI，默认 8765） |
| `apps/grpc_search_server.py` | gRPC 混合检索（默认 50051） |
| `proto/es2vec_search.proto` | gRPC 接口定义 |

本机 Python 快速开始、模块说明与环境变量见 [README.md](README.md)。

## 对外集成（REST / gRPC）

供其它系统在内网调用（**无鉴权**，勿直接暴露公网；生产请前置 API 网关）。

**前置**：`elasticsearch` 已 healthy，且已完成 `index_corpus`（与 Web 使用相同索引，默认 `es2vec_corpus_chunks`）。

```powershell
docker compose up -d elasticsearch web grpc
```

### REST

| 端点 | 说明 |
|------|------|
| `GET /api/health` | 健康检查 |
| `GET /api/search?q=...` | 兼容旧版查询参数 |
| `POST /api/v1/search` | **推荐**：JSON Body，可调权重等 |
| `GET /docs` | OpenAPI 交互文档 |

```powershell
curl "http://127.0.0.1:8765/api/search?q=刘备&index=es2vec_corpus_chunks&k=5"

curl -X POST "http://127.0.0.1:8765/api/v1/search" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"刘备\",\"index\":\"es2vec_corpus_chunks\",\"k\":5}"
```

示例脚本：`examples/clients/search_rest.py`

### gRPC

- 地址：`localhost:50051`（容器内服务名 `grpc:50051`）
- Proto：`proto/es2vec_search.proto`
- RPC：`SearchService.HybridSearch`、`SearchService.Health`

```powershell
python examples/clients/search_grpc.py --host 127.0.0.1 --port 50051 --query 刘备
```

修改 proto 后重新生成 stub：`python scripts/gen_grpc.py`
