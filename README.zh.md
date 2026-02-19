# ONNX 轻量向量服务（FastAPI + ONNX Runtime + Qdrant）

一个在本地/私有环境即可运行的向量底座，适合 RAG、知识库、搜索、Agent/RPA。全文偏温和、可落地，帮助你快速启用。😊

## 优势亮点
- **CPU 友好**：默认即用 ONNX Runtime + fastembed，小机器也能跑（8GB 内存起步）。
- **开箱即用**：Docker Compose 一条命令启动，默认附带健康检查与指标。
- **全套 API**：向量生成、集合管理、写入、检索、混合重排、游标遍历、payload 维护，一站搞定。
- **可观测 & 鉴权**：Prometheus 指标、/healthz，支持 Embedding API Key 与 Qdrant API Key。
- **灵活调优**：HNSW 参量、量化开关、批大小都可通过环境变量调整。

## 快速开始
1. 启动
   ```bash
   docker compose up -d
   ```
2. 健康检查
   ```bash
   curl http://127.0.0.1:18000/healthz
   ```
3. 指标查看
   ```bash
   curl http://127.0.0.1:18000/metrics
   ```
4. 默认地址
   - Embedding API: `http://127.0.0.1:18000`
   - Qdrant（容器内）: `http://qdrant:6333` （需要调试再映射宿主机端口）

## 部署
- **已构建镜像（推荐生产）**
  ```yaml
  services:
    embedding:
      image: your-registry/embedding:1.0.0
  ```
  ```bash
  docker compose up -d
  ```
- **源码构建（开发）**
  ```bash
  docker compose up -d --build
  ```

## 环境变量速览
- 推理：`MODEL_NAME`，`MAX_LENGTH`，`BATCH_SIZE`，`NORMALIZE`，`OMP_NUM_THREADS`
- Qdrant：`QDRANT_URL`，`QDRANT_API_KEY`，`DEFAULT_COLLECTION`，`ON_DISK_PAYLOAD`
- 索引/量化：`DEFAULT_HNSW_EF`，`ENABLE_SCALAR_QUANTIZATION`，`QUANTIZATION_QUANTILE`，`QUANTIZATION_ALWAYS_RAM`，`UPSERT_BATCH_SIZE`
- 鉴权：`EMBED_API_KEY`（需要时传 `X-Api-Key`）
- 日志：`LOG_LEVEL`
- MCP 桥接：`EMBEDDING_API_URL`（默认 `http://127.0.0.1:18000`），`MCP_SERVER_NAME`

## MCP 支持（供大模型工具调用）
仓库已内置 MCP 服务（`app/mcp_server.py`），可让 Claude Desktop、Cursor、Cherry Studio 等客户端把当前向量 API 作为工具调用。

先安装 MCP 可选依赖：

```bash
pip install -r requirements-mcp.txt
```

先启动 embedding API，再通过 stdio 启动 MCP：

```bash
python -m app.mcp_server
```

可用 MCP 工具：
- `healthz`
- `list_collections`
- `ensure_collection`
- `upsert`
- `search`
- `query_hybrid`
- `retrieve`
- `delete`

示例 MCP 客户端配置：

```json
{
  "mcpServers": {
    "easyqdrant": {
      "command": "python",
      "args": ["-m", "app.mcp_server"],
      "env": {
        "EMBEDDING_API_URL": "http://127.0.0.1:18000",
        "EMBED_API_KEY": ""
      }
    }
  }
}
```

## API 一览
- 健康与指标：`GET /healthz`，`GET /metrics`
- 集合：`GET /collections`，`GET /collections/{name}/stats`，`POST /collections/{name}/ensure`
- 写入：`POST /upsert`，`POST /bulk-upsert-file`（JSONL，返回 task_id），`GET /tasks/{task_id}`
- 检索：`POST /search`，`POST /query-hybrid`，`POST /rerank`
- 数据访问：`POST /retrieve`，`POST /scroll`
- 元数据：`POST /update-payload`，`POST /delete`

## 常用示例
> 如开启鉴权，记得加 `-H "X-Api-Key: $EMBED_API_KEY"`。

**Embed 向量生成**
```bash
curl -X POST 'http://127.0.0.1:18000/embed' \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["你好世界", "embedding test"], "prefix": "passage: ", "strip": true}'
```

**Ensure 集合**
```bash
curl -X POST 'http://127.0.0.1:18000/collections/documents/ensure' \
  -H 'Content-Type: application/json' \
  -d '{"recreate": false, "hnsw_m": 16, "hnsw_ef_construct": 128, "scalar_quantization": true, "quantile": 0.99, "always_ram": false}'
```

**Upsert 写入**
```bash
curl -X POST 'http://127.0.0.1:18000/upsert' \
  -H 'Content-Type: application/json' \
  -d '{"collection": "documents", "prefix": "passage: ", "items": [
    {"id": "doc-1", "text": "今天天气不错", "metadata": {"source": "demo", "lang": "zh"}},
    {"id": "doc-2", "text": "FastAPI with ONNX is lightweight", "metadata": {"source": "demo", "lang": "en"}}
  ]}'
```

**Bulk JSONL 写入**
```bash
curl -X POST 'http://127.0.0.1:18000/bulk-upsert-file' \
  -F 'file=@data.jsonl' -F 'collection=documents' -F 'prefix=passage: ' -F 'strip=true'
# 任务进度
curl http://127.0.0.1:18000/tasks/<task_id>
```

**Search / Hybrid**
```bash
curl -X POST 'http://127.0.0.1:18000/search' \
  -H 'Content-Type: application/json' \
  -d '{"collection": "documents", "query": "轻量向量服务", "prefix": "query: ", "top_k": 5, "hnsw_ef": 64, "with_payload": true}'
```
```bash
curl -X POST 'http://127.0.0.1:18000/query-hybrid' \
  -H 'Content-Type: application/json' \
  -d '{"collection": "documents", "query": "向量检索调优", "top_k": 5, "candidate_k": 30, "alpha": 0.8}'
```

**Rerank（仅重排）**
```bash
curl -X POST 'http://127.0.0.1:18000/rerank' \
  -H 'Content-Type: application/json' \
  -d '{"query": "battery life", "alpha": 0.7, "top_k": 5, "candidates": [{"id": "a", "text": "Long lasting battery", "score": 0.8}, {"id": "b", "text": "Fast charging", "score": 0.6}]}'
```

**Retrieve / Scroll / Update / Delete**
```bash
curl -X POST 'http://127.0.0.1:18000/retrieve' \
  -H 'Content-Type: application/json' \
  -d '{"collection": "documents", "ids": ["doc-1", "doc-2"], "with_payload": true}'
```
```bash
curl -X POST 'http://127.0.0.1:18000/scroll' \
  -H 'Content-Type: application/json' \
  -d '{"collection": "documents", "limit": 10, "with_payload": true}'
```
```bash
curl -X POST 'http://127.0.0.1:18000/update-payload' \
  -H 'Content-Type: application/json' \
  -d '{"collection": "documents", "ids": ["doc-1"], "payload": {"tag": "faq", "updated_by": "ops"}}'
```
```bash
curl -X POST 'http://127.0.0.1:18000/delete' \
  -H 'Content-Type: application/json' \
  -d '{"collection": "documents", "ids": ["doc-2"], "wait": true}'
```

## 性能建议（8GB CPU）
- 模型：`BAAI/bge-small-zh-v1.5`
- `MAX_LENGTH=512`，`BATCH_SIZE=4~8`，`UPSERT_BATCH_SIZE=64~128`
- 线程 `OMP_NUM_THREADS=2~4`
- `DEFAULT_HNSW_EF=64` 起步，压测后再调
- 量化：`ENABLE_SCALAR_QUANTIZATION=true`，`QUANTIZATION_QUANTILE=0.99`
- 单实例 `workers=1`
- 压测：`python3 scripts/bench.py --requests 200 --hnsw-ef 64`

## 安全建议
- 默认不把 Qdrant 暴露公网；需要时自控映射端口。
- 需要鉴权时设置 `EMBED_API_KEY`，调用加 `X-Api-Key`；Qdrant 侧配合 `QDRANT_API_KEY` 与容器内 `QDRANT__SERVICE__API_KEY`。

## FAQ
- **维度不匹配**：模型维度与 collection 不符时，重建 collection。
- **容器访问 qdrant**：用 `http://qdrant:6333`，不是 127.0.0.1。
- **模型下载慢**：挂载缓存目录如 `./models:/root/.cache`。

## 贡献
欢迎 PR / Issue：新模型支持、检索示例、压测与调优心得。
