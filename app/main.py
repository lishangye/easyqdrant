import json
import logging
import os
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastembed import TextEmbedding
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from starlette.responses import PlainTextResponse

APP_NAME = "embedding-service-onnx-lite"
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-small-zh-v1.5")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "128"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
NORMALIZE = os.getenv("NORMALIZE", "true").lower() in ("1", "true", "yes")
API_KEY = os.getenv("EMBED_API_KEY", "")
THREADS = int(os.getenv("OMP_NUM_THREADS", "2"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "documents")
ON_DISK_PAYLOAD = os.getenv("ON_DISK_PAYLOAD", "true").lower() in ("1", "true", "yes")
DEFAULT_HNSW_EF = int(os.getenv("DEFAULT_HNSW_EF", "64"))
ENABLE_SCALAR_QUANTIZATION = os.getenv("ENABLE_SCALAR_QUANTIZATION", "true").lower() in (
    "1",
    "true",
    "yes",
)
QUANTIZATION_QUANTILE = float(os.getenv("QUANTIZATION_QUANTILE", "0.99"))
QUANTIZATION_ALWAYS_RAM = os.getenv("QUANTIZATION_ALWAYS_RAM", "false").lower() in (
    "1",
    "true",
    "yes",
)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(APP_NAME)

app = FastAPI(title=APP_NAME)

model: Optional[TextEmbedding] = None
vector_dim: Optional[int] = None
qdrant: Optional[QdrantClient] = None

metrics_lock = threading.Lock()
request_counter: Dict[Tuple[str, str, str], int] = {}
latency_counter: Dict[Tuple[str, str], List[float]] = {}
error_counter: Dict[Tuple[str, str], int] = {}

tasks_lock = threading.Lock()
tasks_state: Dict[str, Dict[str, Any]] = {}


@app.middleware("http")
async def access_log_and_metrics(request: Request, call_next):
    start = time.perf_counter()
    method = request.method
    path = request.url.path
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        with metrics_lock:
            error_counter[(method, path)] = error_counter.get((method, path), 0) + 1
        logger.exception("request_failed method=%s path=%s", method, path)
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        with metrics_lock:
            key = (method, path, str(status_code))
            request_counter[key] = request_counter.get(key, 0) + 1
            lk = (method, path)
            total_sum, total_count = latency_counter.get(lk, [0.0, 0.0])
            latency_counter[lk] = [total_sum + duration_ms, total_count + 1]
            if status_code >= 500:
                error_counter[(method, path)] = error_counter.get((method, path), 0) + 1
        logger.info("request method=%s path=%s status=%s latency_ms=%.2f", method, path, status_code, duration_ms)


def check_key(x_api_key: Optional[str]) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class EmbedRequest(BaseModel):
    texts: List[str] = Field(min_length=1)
    prefix: Optional[str] = None
    strip: bool = True


class EmbedResponse(BaseModel):
    dim: int
    vectors: List[List[float]]


class EnsureCollectionRequest(BaseModel):
    recreate: bool = False
    on_disk_payload: Optional[bool] = None
    hnsw_m: Optional[int] = Field(default=None, ge=4)
    hnsw_ef_construct: Optional[int] = Field(default=None, ge=8)
    hnsw_full_scan_threshold: Optional[int] = Field(default=None, ge=1)
    scalar_quantization: Optional[bool] = None
    quantile: float = Field(default=QUANTIZATION_QUANTILE, gt=0.0, le=1.0)
    always_ram: bool = QUANTIZATION_ALWAYS_RAM


class EnsureCollectionResponse(BaseModel):
    collection: str
    created: bool
    recreated: bool
    dim: int


class UpsertItem(BaseModel):
    id: Optional[Union[int, str]] = None
    text: str
    metadata: Optional[Dict[str, Any]] = None


class UpsertRequest(BaseModel):
    collection: str = DEFAULT_COLLECTION
    items: List[UpsertItem] = Field(min_length=1)
    prefix: str = "passage: "
    strip: bool = True


class UpsertResponse(BaseModel):
    collection: str
    count: int
    dim: int


class SearchRequest(BaseModel):
    collection: str = DEFAULT_COLLECTION
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    prefix: str = "query: "
    strip: bool = True
    with_payload: bool = True
    score_threshold: Optional[float] = None
    filter: Optional[Dict[str, Any]] = None
    hnsw_ef: Optional[int] = Field(default=None, ge=8)
    exact: bool = False


class SearchHit(BaseModel):
    id: Union[int, str]
    score: float
    payload: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    collection: str
    count: int
    results: List[SearchHit]


class DeleteRequest(BaseModel):
    collection: str = DEFAULT_COLLECTION
    ids: Optional[List[Union[int, str]]] = None
    filter: Optional[Dict[str, Any]] = None
    wait: bool = True


class DeleteResponse(BaseModel):
    collection: str
    mode: str
    requested: int
    status: str


class RetrieveRequest(BaseModel):
    collection: str = DEFAULT_COLLECTION
    ids: List[Union[int, str]] = Field(min_length=1)
    with_payload: bool = True
    with_vectors: bool = False


class ScrollRequest(BaseModel):
    collection: str = DEFAULT_COLLECTION
    limit: int = Field(default=10, ge=1, le=1000)
    offset: Optional[Union[int, str]] = None
    with_payload: bool = True
    with_vectors: bool = False
    filter: Optional[Dict[str, Any]] = None


class UpdatePayloadRequest(BaseModel):
    collection: str = DEFAULT_COLLECTION
    payload: Dict[str, Any]
    ids: Optional[List[Union[int, str]]] = None
    filter: Optional[Dict[str, Any]] = None
    wait: bool = True


class QueryHybridRequest(BaseModel):
    collection: str = DEFAULT_COLLECTION
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    candidate_k: int = Field(default=30, ge=1, le=500)
    prefix: str = "query: "
    with_payload: bool = True
    alpha: float = Field(default=0.8, ge=0.0, le=1.0)
    hnsw_ef: Optional[int] = Field(default=None, ge=8)


class RerankCandidate(BaseModel):
    id: Union[int, str]
    text: str
    payload: Optional[Dict[str, Any]] = None
    score: Optional[float] = None


class RerankRequest(BaseModel):
    query: str
    candidates: List[RerankCandidate] = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=200)
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)


class TaskResponse(BaseModel):
    task_id: str
    status: str


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().replace("\n", " ").split(" ") if t]


def _lexical_score(query: str, text: str) -> float:
    q_tokens = set(_tokenize(query))
    t_tokens = set(_tokenize(text))
    if not q_tokens or not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def _embed_texts(texts: List[str]) -> np.ndarray:
    assert model is not None
    dense = np.array(list(model.embed(texts, batch_size=BATCH_SIZE)), dtype=np.float32)
    if NORMALIZE:
        norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12
        dense = dense / norms
    return dense


def _to_qdrant_point_id(item_id: Optional[Union[int, str]]) -> Union[int, str]:
    if item_id is None:
        return str(uuid4())
    if isinstance(item_id, int):
        if item_id < 0:
            raise HTTPException(status_code=400, detail="id must be unsigned integer, UUID, or string")
        return item_id
    try:
        UUID(item_id)
        return item_id
    except ValueError:
        return str(uuid5(NAMESPACE_URL, item_id))


def _build_hnsw_config(req: EnsureCollectionRequest) -> Optional[models.HnswConfigDiff]:
    if req.hnsw_m is None and req.hnsw_ef_construct is None and req.hnsw_full_scan_threshold is None:
        return None
    return models.HnswConfigDiff(
        m=req.hnsw_m,
        ef_construct=req.hnsw_ef_construct,
        full_scan_threshold=req.hnsw_full_scan_threshold,
    )


def _build_quantization_config(req: EnsureCollectionRequest) -> Optional[models.ScalarQuantization]:
    enable = ENABLE_SCALAR_QUANTIZATION if req.scalar_quantization is None else req.scalar_quantization
    if not enable:
        return None
    return models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=req.quantile,
            always_ram=req.always_ram,
        )
    )


def _create_collection(collection: str, req: Optional[EnsureCollectionRequest] = None) -> None:
    assert qdrant is not None
    assert vector_dim is not None
    ensure_req = req or EnsureCollectionRequest()
    qdrant.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=vector_dim,
            distance=models.Distance.COSINE,
        ),
        hnsw_config=_build_hnsw_config(ensure_req),
        quantization_config=_build_quantization_config(ensure_req),
        on_disk_payload=ON_DISK_PAYLOAD if ensure_req.on_disk_payload is None else ensure_req.on_disk_payload,
    )


def _validate_collection_dim(collection: str) -> None:
    assert qdrant is not None
    assert vector_dim is not None
    info = qdrant.get_collection(collection).config.params.vectors
    if isinstance(info, dict):
        first = next(iter(info.values()))
        size = int(first.size)
    else:
        size = int(info.size)
    if size != vector_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Collection dim mismatch: collection={size}, model={vector_dim}",
        )


def _ensure_collection(collection: str) -> None:
    assert qdrant is not None
    if not qdrant.collection_exists(collection):
        try:
            _create_collection(collection)
        except UnexpectedResponse as exc:
            if exc.status_code != 409:
                raise
    _validate_collection_dim(collection)


def _upsert_items(collection: str, items: List[UpsertItem], prefix: str, strip: bool) -> int:
    assert qdrant is not None
    _ensure_collection(collection)

    texts: List[str] = []
    for item in items:
        text = item.text.strip() if strip else item.text
        texts.append(prefix + text)

    dense = _embed_texts(texts)
    points: List[models.PointStruct] = []

    for idx, item in enumerate(items):
        payload: Dict[str, Any] = dict(item.metadata or {})
        payload["text"] = item.text
        point_id = _to_qdrant_point_id(item.id)
        if isinstance(item.id, str):
            try:
                UUID(item.id)
            except ValueError:
                payload["_external_id"] = item.id
        points.append(models.PointStruct(id=point_id, vector=dense[idx].tolist(), payload=payload))

    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        qdrant.upsert(collection_name=collection, points=points[i : i + UPSERT_BATCH_SIZE], wait=True)

    return len(points)


def _update_task(task_id: str, **kwargs: Any) -> None:
    with tasks_lock:
        if task_id in tasks_state:
            tasks_state[task_id].update(kwargs)


def _process_bulk_file(task_id: str, file_path: str, collection: str, prefix: str, strip: bool) -> None:
    logger.info("bulk_task_start task_id=%s collection=%s file=%s", task_id, collection, file_path)
    processed = 0
    failed = 0
    buffer: List[UpsertItem] = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                    buffer.append(
                        UpsertItem(
                            id=obj.get("id"),
                            text=obj.get("text", ""),
                            metadata=obj.get("metadata"),
                        )
                    )
                except Exception:
                    failed += 1
                    continue

                if len(buffer) >= UPSERT_BATCH_SIZE:
                    processed += _upsert_items(collection, buffer, prefix, strip)
                    _update_task(task_id, processed=processed, failed=failed, line=line_num)
                    buffer = []

        if buffer:
            processed += _upsert_items(collection, buffer, prefix, strip)

        _update_task(task_id, status="completed", processed=processed, failed=failed)
        logger.info("bulk_task_done task_id=%s processed=%s failed=%s", task_id, processed, failed)
    except Exception as exc:
        _update_task(task_id, status="failed", error=str(exc), processed=processed, failed=failed)
        logger.exception("bulk_task_failed task_id=%s", task_id)
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass


@app.on_event("startup")
def startup() -> None:
    global model, vector_dim, qdrant
    model = TextEmbedding(model_name=MODEL_NAME, threads=THREADS, max_length=MAX_LENGTH)
    warmup = list(model.embed(["hello"], batch_size=1))
    vector_dim = int(len(warmup[0]))
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
    logger.info("startup_complete model=%s dim=%s qdrant=%s", MODEL_NAME, vector_dim, QDRANT_URL)


@app.get("/healthz")
def healthz() -> dict:
    qdrant_ok = False
    try:
        assert qdrant is not None
        qdrant.get_collections()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False
    return {"status": "ok", "model": MODEL_NAME, "dim": vector_dim, "qdrant": qdrant_ok}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    lines = [
        "# HELP app_requests_total Total API requests",
        "# TYPE app_requests_total counter",
    ]
    with metrics_lock:
        for (method, path, status), count in sorted(request_counter.items()):
            lines.append(
                f'app_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}'
            )

        lines.append("# HELP app_request_latency_ms_avg Average request latency in ms")
        lines.append("# TYPE app_request_latency_ms_avg gauge")
        for (method, path), (total_sum, total_count) in sorted(latency_counter.items()):
            avg = total_sum / total_count if total_count else 0.0
            lines.append(f'app_request_latency_ms_avg{{method="{method}",path="{path}"}} {avg:.4f}')

        lines.append("# HELP app_errors_total Total failed requests")
        lines.append("# TYPE app_errors_total counter")
        for (method, path), count in sorted(error_counter.items()):
            lines.append(f'app_errors_total{{method="{method}",path="{path}"}} {count}')

    return "\n".join(lines) + "\n"


@app.get("/collections")
def list_collections(x_api_key: Optional[str] = Header(default=None)) -> dict:
    check_key(x_api_key)
    assert qdrant is not None
    data = qdrant.get_collections()
    names = [c.name for c in data.collections]
    return {"count": len(names), "collections": names}


@app.get("/collections/{name}/stats")
def collection_stats(name: str, x_api_key: Optional[str] = Header(default=None)) -> dict:
    check_key(x_api_key)
    assert qdrant is not None
    if not qdrant.collection_exists(name):
        raise HTTPException(status_code=404, detail="Collection not found")
    info = qdrant.get_collection(name)
    payload = info.model_dump(mode="json") if hasattr(info, "model_dump") else json.loads(info.json())
    return {"collection": name, "stats": payload}


@app.post("/collections/{name}/ensure", response_model=EnsureCollectionResponse)
def ensure_collection(name: str, req: EnsureCollectionRequest, x_api_key: Optional[str] = Header(default=None)) -> dict:
    check_key(x_api_key)
    assert qdrant is not None

    created = False
    recreated = False
    exists = qdrant.collection_exists(name)

    if exists and req.recreate:
        qdrant.delete_collection(name)
        _create_collection(name, req)
        created = True
        recreated = True
    elif not exists:
        _create_collection(name, req)
        created = True

    _validate_collection_dim(name)
    assert vector_dim is not None
    logger.info("ensure_collection name=%s created=%s recreated=%s", name, created, recreated)
    return {"collection": name, "created": created, "recreated": recreated, "dim": vector_dim}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest, x_api_key: Optional[str] = Header(default=None)) -> dict:
    check_key(x_api_key)
    prefix = req.prefix or ""
    texts: List[str] = []
    for t in req.texts:
        item = "" if t is None else t
        if req.strip:
            item = item.strip()
        texts.append(prefix + item)
    dense = _embed_texts(texts)
    return {"dim": int(dense.shape[1]), "vectors": dense.tolist()}


@app.post("/upsert", response_model=UpsertResponse)
def upsert(req: UpsertRequest, x_api_key: Optional[str] = Header(default=None)) -> dict:
    check_key(x_api_key)
    count = _upsert_items(req.collection, req.items, req.prefix, req.strip)
    assert vector_dim is not None
    logger.info("upsert collection=%s count=%s", req.collection, count)
    return {"collection": req.collection, "count": count, "dim": int(vector_dim)}


@app.post("/bulk-upsert-file", response_model=TaskResponse)
async def bulk_upsert_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection: str = Form(DEFAULT_COLLECTION),
    prefix: str = Form("passage: "),
    strip: bool = Form(True),
    x_api_key: Optional[str] = Header(default=None),
) -> dict:
    check_key(x_api_key)
    task_id = str(uuid4())

    suffix = ".jsonl"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    with tasks_lock:
        tasks_state[task_id] = {
            "status": "queued",
            "collection": collection,
            "processed": 0,
            "failed": 0,
            "file": file.filename,
            "created_at": time.time(),
        }

    _update_task(task_id, status="running")
    background_tasks.add_task(_process_bulk_file, task_id, tmp_path, collection, prefix, strip)
    logger.info("bulk_task_queued task_id=%s collection=%s file=%s", task_id, collection, file.filename)
    return {"task_id": task_id, "status": "running"}


@app.get("/tasks/{task_id}")
def get_task(task_id: str, x_api_key: Optional[str] = Header(default=None)) -> dict:
    check_key(x_api_key)
    with tasks_lock:
        task = tasks_state.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"task_id": task_id, **task}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, x_api_key: Optional[str] = Header(default=None)) -> dict:
    check_key(x_api_key)
    assert qdrant is not None
    _ensure_collection(req.collection)

    query_text = req.query.strip() if req.strip else req.query
    vector = _embed_texts([req.prefix + query_text])[0].tolist()

    query_filter = None
    if req.filter:
        query_filter = models.Filter.model_validate(req.filter)

    search_params = models.SearchParams(hnsw_ef=req.hnsw_ef or DEFAULT_HNSW_EF, exact=req.exact)

    hits = qdrant.search(
        collection_name=req.collection,
        query_vector=vector,
        limit=req.top_k,
        query_filter=query_filter,
        search_params=search_params,
        with_payload=req.with_payload,
        score_threshold=req.score_threshold,
    )

    results = [
        {
            "id": hit.id,
            "score": float(hit.score),
            "payload": hit.payload if req.with_payload else None,
        }
        for hit in hits
    ]
    return {"collection": req.collection, "count": len(results), "results": results}


@app.post("/query-hybrid", response_model=SearchResponse)
def query_hybrid(req: QueryHybridRequest, x_api_key: Optional[str] = Header(default=None)) -> dict:
    check_key(x_api_key)
    assert qdrant is not None
    _ensure_collection(req.collection)

    vector = _embed_texts([req.prefix + req.query])[0].tolist()
    search_params = models.SearchParams(hnsw_ef=req.hnsw_ef or DEFAULT_HNSW_EF, exact=False)

    hits = qdrant.search(
        collection_name=req.collection,
        query_vector=vector,
        limit=req.candidate_k,
        with_payload=True,
        search_params=search_params,
    )

    reranked: List[Dict[str, Any]] = []
    for hit in hits:
        payload = hit.payload or {}
        text = str(payload.get("text", ""))
        lexical = _lexical_score(req.query, text)
        hybrid_score = req.alpha * float(hit.score) + (1.0 - req.alpha) * lexical
        reranked.append(
            {
                "id": hit.id,
                "score": hybrid_score,
                "payload": payload if req.with_payload else None,
            }
        )

    reranked.sort(key=lambda x: x["score"], reverse=True)
    results = reranked[: req.top_k]
    return {"collection": req.collection, "count": len(results), "results": results}


@app.post("/rerank")
def rerank(req: RerankRequest, x_api_key: Optional[str] = Header(default=None)) -> dict:
    check_key(x_api_key)
    scored = []
    for cand in req.candidates:
        lexical = _lexical_score(req.query, cand.text)
        base = cand.score if cand.score is not None else 0.0
        final_score = req.alpha * base + (1.0 - req.alpha) * lexical
        scored.append(
            {
                "id": cand.id,
                "score": float(final_score),
                "text": cand.text,
                "payload": cand.payload,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"count": min(req.top_k, len(scored)), "results": scored[: req.top_k]}


@app.post("/retrieve")
def retrieve(req: RetrieveRequest, x_api_key: Optional[str] = Header(default=None)) -> dict:
    check_key(x_api_key)
    assert qdrant is not None
    _ensure_collection(req.collection)

    point_ids = [_to_qdrant_point_id(i) for i in req.ids]
    records = qdrant.retrieve(
        collection_name=req.collection,
        ids=point_ids,
        with_payload=req.with_payload,
        with_vectors=req.with_vectors,
    )

    points: List[Dict[str, Any]] = []
    for r in records:
        points.append(
            {
                "id": r.id,
                "payload": r.payload,
                "vector": r.vector if req.with_vectors else None,
            }
        )
    return {"collection": req.collection, "count": len(points), "points": points}


@app.post("/scroll")
def scroll(req: ScrollRequest, x_api_key: Optional[str] = Header(default=None)) -> dict:
    check_key(x_api_key)
    assert qdrant is not None
    _ensure_collection(req.collection)

    scroll_filter = models.Filter.model_validate(req.filter) if req.filter else None
    points, next_offset = qdrant.scroll(
        collection_name=req.collection,
        scroll_filter=scroll_filter,
        limit=req.limit,
        offset=req.offset,
        with_payload=req.with_payload,
        with_vectors=req.with_vectors,
    )

    out = [
        {
            "id": p.id,
            "payload": p.payload,
            "vector": p.vector if req.with_vectors else None,
        }
        for p in points
    ]
    return {"collection": req.collection, "count": len(out), "next_offset": next_offset, "points": out}


@app.post("/update-payload")
def update_payload(req: UpdatePayloadRequest, x_api_key: Optional[str] = Header(default=None)) -> dict:
    check_key(x_api_key)
    assert qdrant is not None
    _ensure_collection(req.collection)

    has_ids = bool(req.ids)
    has_filter = req.filter is not None
    if not has_ids and not has_filter:
        raise HTTPException(status_code=400, detail="Either ids or filter is required")
    if has_ids and has_filter:
        raise HTTPException(status_code=400, detail="Provide ids or filter, not both")

    points_selector: Union[List[Union[int, str]], models.Filter]
    mode: str
    requested: int

    if has_ids:
        points_selector = [_to_qdrant_point_id(i) for i in (req.ids or [])]
        mode = "ids"
        requested = len(points_selector)
    else:
        points_selector = models.Filter.model_validate(req.filter)
        mode = "filter"
        requested = 1

    result = qdrant.set_payload(
        collection_name=req.collection,
        payload=req.payload,
        points=points_selector,
        wait=req.wait,
    )

    status = str(result.status) if result is not None else "unknown"
    return {
        "collection": req.collection,
        "mode": mode,
        "requested": requested,
        "status": status,
    }


@app.post("/delete", response_model=DeleteResponse)
def delete(req: DeleteRequest, x_api_key: Optional[str] = Header(default=None)) -> dict:
    check_key(x_api_key)
    assert qdrant is not None
    _ensure_collection(req.collection)

    has_ids = bool(req.ids)
    has_filter = req.filter is not None
    if not has_ids and not has_filter:
        raise HTTPException(status_code=400, detail="Either ids or filter is required")
    if has_ids and has_filter:
        raise HTTPException(status_code=400, detail="Provide ids or filter, not both")

    if has_ids:
        point_ids = [_to_qdrant_point_id(item_id) for item_id in (req.ids or [])]
        selector = models.PointIdsList(points=point_ids)
        mode = "ids"
        requested = len(point_ids)
    else:
        selector = models.FilterSelector(filter=models.Filter.model_validate(req.filter))
        mode = "filter"
        requested = 1

    result = qdrant.delete(collection_name=req.collection, points_selector=selector, wait=req.wait)
    status = str(result.status) if result is not None else "unknown"
    logger.info("delete collection=%s mode=%s requested=%s status=%s", req.collection, mode, requested, status)
    return {
        "collection": req.collection,
        "mode": mode,
        "requested": requested,
        "status": status,
    }
