---
name: embedding-http-client
description: Call the FastAPI embedding service to embed text, ensure/manage Qdrant collections, upsert (JSON or JSONL), search, hybrid rerank, retrieve/scroll, update payloads, and delete points. Use when an agent needs to interact with this HTTP API for RAG/search workflows.
---

# Embedding HTTP Client

## Quick start
- `base_url` default `http://127.0.0.1:18000`. Ask the user if unsure.
- Common headers: `Content-Type: application/json`; add `X-Api-Key: <EMBED_API_KEY>` when EMBED_API_KEY is set.
- Payload prefix helpers: use `"passage: "` for documents, `"query: "` for queries; set `strip=true` to trim whitespace.

## Endpoint checklist
- Health/metrics: `GET /healthz`, `GET /metrics`.
- Collections: `GET /collections`, `GET /collections/{name}/stats`, `POST /collections/{name}/ensure`.
- Ingest: `POST /upsert` (JSON), `POST /bulk-upsert-file` (multipart JSONL, returns task_id), `GET /tasks/{task_id}`.
- Search: `POST /search` (vector only), `POST /query-hybrid` (vector + lexical), `POST /rerank` (local rerank without Qdrant).
- Data access: `POST /retrieve`, `POST /scroll`.
- Metadata maintenance: `POST /update-payload`, `POST /delete`.

## Payload templates
> Replace placeholders; include `-H "X-Api-Key: $EMBED_API_KEY"` when auth is enabled.

**Embed**
```json
POST /embed
{"texts": ["hello", "world"], "prefix": "passage: ", "strip": true}
```

**Ensure collection**
```json
POST /collections/{name}/ensure
{"recreate": false, "hnsw_m": 16, "hnsw_ef_construct": 128, "scalar_quantization": true, "quantile": 0.99, "always_ram": false}
```

**Upsert**
```json
POST /upsert
{"collection": "documents", "prefix": "passage: ", "items": [
  {"id": "doc-1", "text": "今天不错", "metadata": {"source": "demo", "lang": "zh"}},
  {"id": "doc-2", "text": "FastAPI with ONNX", "metadata": {"source": "demo", "lang": "en"}}
], "strip": true}
```

**Bulk upsert (JSONL)**
```
POST /bulk-upsert-file (multipart)
file=@data.jsonl; collection=documents; prefix=passage: ; strip=true
```
Follow with `GET /tasks/{task_id}` to poll status.

**Search**
```json
POST /search
{"collection": "documents", "query": "轻量向量服务", "prefix": "query: ", "top_k": 5, "hnsw_ef": 64, "with_payload": true,
 "filter": {"must": [{"key": "lang", "match": {"value": "zh"}}]}}
```

**Hybrid query**
```json
POST /query-hybrid
{"collection": "documents", "query": "vector tuning", "top_k": 5, "candidate_k": 30, "alpha": 0.8, "with_payload": true}
```

**Rerank candidates (no Qdrant call)**
```json
POST /rerank
{"query": "battery life", "alpha": 0.7, "top_k": 5,
 "candidates": [{"id": "a", "text": "long lasting battery", "score": 0.8}, {"id": "b", "text": "fast charge", "score": 0.6}]}
```

**Retrieve by IDs**
```json
POST /retrieve
{"collection": "documents", "ids": ["doc-1", "doc-2"], "with_payload": true, "with_vectors": false}
```

**Scroll**
```json
POST /scroll
{"collection": "documents", "limit": 20, "offset": null, "with_payload": true, "with_vectors": false}
```

**Update payload**
```json
POST /update-payload
{"collection": "documents", "ids": ["doc-1"], "payload": {"tag": "faq", "updated_by": "agent"}, "wait": true}
```

**Delete**
```json
POST /delete
{"collection": "documents", "ids": ["doc-2"], "wait": true}
```

## Tips & constraints
- Dimension mismatches: collection vector size must match the current `MODEL_NAME`. Recreate collection via `/collections/{name}/ensure` after model changes.
- Filters follow Qdrant JSON Filter schema; pass through as-is in `filter` fields.
- `with_payload` defaults true on search/retrieve/scroll; set false if you only need IDs/scores.
- `exact=true` on `/search` forces exact scoring; otherwise HNSW with `hnsw_ef`.
- Metadata suggestion keys: `source`, `doc_id`, `chunk_id`, `title`, `url`, `created_at`, `lang`, `tenant_id`.
- Bulk file must be JSONL where each line is an upsert item: `{ "id": ..., "text": ..., "metadata": {...} }`.

## Error handling
- 401: set `X-Api-Key` to match `EMBED_API_KEY` env.
- 404 collection: call `/collections/{name}/ensure` first.
- 400 ids/filter required: `update-payload` and `delete` need either `ids` or `filter`, not both.

## Operational notes
- Default base URL assumes Docker Compose on localhost; override if behind a proxy or different port.
- Qdrant API Key is forwarded automatically via env; callers only set `X-Api-Key` for the embedding service.
- Metrics are plain text Prometheus format; scrape `/metrics` if you need observability.
