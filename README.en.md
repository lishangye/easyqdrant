# ONNX Embedding + Qdrant Service (FastAPI + ONNX Runtime + Qdrant)

A gentle, ready-to-run vector service for local/private setups—great for RAG, knowledge bases, search, and agent/RPA tools. Designed to be approachable while staying production-minded.

## Why it helps
- **CPU-friendly**: ONNX Runtime + fastembed works well on modest machines (8GB RAM+).
- **Up in one command**: Docker Compose brings the API, health check, and metrics online quickly.
- **Complete HTTP surface**: embeddings, collection ensure/stats, upsert, bulk JSONL ingest, search, hybrid rerank, scroll/retrieve, payload maintenance.
- **Observability & auth**: Prometheus metrics, `/healthz`, optional Embedding API Key and Qdrant API Key.
- **Tunable**: HNSW knobs, quantization switches, batch sizes, thread count all via env vars.

## Quickstart
1. Start
   ```bash
   docker compose up -d
   ```
2. Health
   ```bash
   curl http://127.0.0.1:18000/healthz
   ```
3. Metrics
   ```bash
   curl http://127.0.0.1:18000/metrics
   ```
4. Base URLs
   - Embedding API: `http://127.0.0.1:18000`
   - Qdrant (in container): `http://qdrant:6333` (map a host port only if you need to debug)

## Deploy
- **Prebuilt image (recommended for prod)**
  ```yaml
  services:
    embedding:
      image: your-registry/embedding:1.0.0
  ```
  ```bash
  docker compose up -d
  ```
- **Build from source (dev)**
  ```bash
  docker compose up -d --build
  ```

## Env vars (at a glance)
- Inference: `MODEL_NAME`, `MAX_LENGTH`, `BATCH_SIZE`, `NORMALIZE`, `OMP_NUM_THREADS`
- Qdrant: `QDRANT_URL`, `QDRANT_API_KEY`, `DEFAULT_COLLECTION`, `ON_DISK_PAYLOAD`
- Index/quantization: `DEFAULT_HNSW_EF`, `ENABLE_SCALAR_QUANTIZATION`, `QUANTIZATION_QUANTILE`, `QUANTIZATION_ALWAYS_RAM`, `UPSERT_BATCH_SIZE`
- Auth: `EMBED_API_KEY` (send `X-Api-Key`)
- RBAC & audit: `ENABLE_RBAC`, `DEFAULT_ROLE`, `AUDIT_MAX_EVENTS`
- Logging: `LOG_LEVEL`
- MCP bridge: `EMBEDDING_API_URL` (default `http://127.0.0.1:18000`), `MCP_SERVER_NAME`

## MCP support (for LLM tool calling)
This repo now includes an MCP server (`app/mcp_server.py`) so LLM clients (Claude Desktop, Cursor, Cherry Studio, etc.) can call your vector APIs as tools.

Install optional MCP dependency first:

```bash
pip install -r requirements-mcp.txt
```

Start the embedding API first, then run MCP via stdio:

```bash
python -m app.mcp_server
```

Available MCP tools include:
- `healthz`
- `list_collections`
- `ensure_collection`
- `upsert`
- `search`
- `query_hybrid`
- `retrieve`
- `delete`

Example MCP client config:

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

## API map
- Health/metrics: `GET /healthz`, `GET /metrics`
- OpenAI-compatible embeddings: `POST /v1/embeddings`
- Collections: `GET /collections`, `GET /collections/{name}/stats`, `POST /collections/{name}/ensure`
- Ingest: `POST /upsert`, `POST /bulk-upsert-file` (JSONL, task_id), `GET /tasks/{task_id}`
- Search: `POST /search`, `POST /query-hybrid`, `POST /rerank`
- Data: `POST /retrieve`, `POST /scroll`
- Metadata: `POST /update-payload`, `POST /delete`
- Audit: `GET /audit/events` (auditor/admin only)
- Agent memory APIs: `POST /memory/spaces/ensure`, `POST /memory/write`, `POST /memory/query`, `POST /memory/get`, `POST /memory/update`, `POST /memory/forget`, `POST /memory/scroll`, `GET /memory/spaces/{collection}/stats`

## Quick examples
> If auth is on, add `-H "X-Api-Key: $EMBED_API_KEY"`.

**Embed**
```bash
curl -X POST 'http://127.0.0.1:18000/embed' \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["hello", "world"], "prefix": "passage: ", "strip": true}'
```

**Ensure collection**
```bash
curl -X POST 'http://127.0.0.1:18000/collections/documents/ensure' \
  -H 'Content-Type: application/json' \
  -d '{"recreate": false, "hnsw_m": 16, "hnsw_ef_construct": 128, "scalar_quantization": true, "quantile": 0.99, "always_ram": false}'
```

**Upsert**
```bash
curl -X POST 'http://127.0.0.1:18000/upsert' \
  -H 'Content-Type: application/json' \
  -d '{"collection": "documents", "prefix": "passage: ", "items": [
    {"id": "doc-1", "text": "Nice weather today", "metadata": {"source": "demo", "lang": "en"}},
    {"id": "doc-2", "text": "FastAPI with ONNX is lightweight", "metadata": {"source": "demo", "lang": "en"}}
  ]}'
```

**Bulk JSONL**
```bash
curl -X POST 'http://127.0.0.1:18000/bulk-upsert-file' \
  -F 'file=@data.jsonl' -F 'collection=documents' -F 'prefix=passage: ' -F 'strip=true'
# poll task
curl http://127.0.0.1:18000/tasks/<task_id>
```

**Search / Hybrid**
```bash
curl -X POST 'http://127.0.0.1:18000/search' \
  -H 'Content-Type: application/json' \
  -d '{"collection": "documents", "query": "lightweight vector service", "prefix": "query: ", "top_k": 5, "hnsw_ef": 64, "with_payload": true}'
```
```bash
curl -X POST 'http://127.0.0.1:18000/query-hybrid' \
  -H 'Content-Type: application/json' \
  -d '{"collection": "documents", "query": "vector tuning", "top_k": 5, "candidate_k": 30, "alpha": 0.8}'
```

**Rerank only**
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

## Tuning (8GB CPU starting point)
- Model: `BAAI/bge-small-zh-v1.5`
- `MAX_LENGTH=512`, `BATCH_SIZE=4-8`, `UPSERT_BATCH_SIZE=64-128`
- Threads: `OMP_NUM_THREADS=2-4`
- `DEFAULT_HNSW_EF=64` to start; raise after benchmarking
- Quantization: `ENABLE_SCALAR_QUANTIZATION=true`, `QUANTIZATION_QUANTILE=0.99`
- Single worker recommended initially (`workers=1`)
- Benchmark: `python3 scripts/bench.py --requests 200 --hnsw-ef 64`

## Security
- Keep Qdrant off the public internet (compose does not expose by default).
- For auth, set `EMBED_API_KEY` and send `X-Api-Key`; pair with `QDRANT_API_KEY` and container `QDRANT__SERVICE__API_KEY` for Qdrant.

## FAQ
- **Vector dim mismatch**: collection vector size must match the current model; recreate the collection after model changes.
- **Accessing qdrant in container**: use `http://qdrant:6333`, not 127.0.0.1.
- **Slow model download**: mount cache such as `./models:/root/.cache`.

## Contributing
PRs/issues welcome—new models, search/filter examples, benchmarks, and tuning tips.

## Testing record
- Detailed test data, iteration logs, and outcomes: `docs/TESTING_REPORT.md`

## Agent long-term memory APIs
- Full API contract, field conventions, workflow and error guide: `docs/AGENT_MEMORY_API.md`
