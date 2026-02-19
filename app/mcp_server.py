import json
import os
from typing import Any, Dict, List, Optional
from urllib import error, request

from mcp.server.fastmcp import FastMCP

MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", "easyqdrant")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://127.0.0.1:18000")
EMBEDDING_API_KEY = os.getenv("EMBED_API_KEY", "")

mcp = FastMCP(MCP_SERVER_NAME)


def _request_json(method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{EMBEDDING_API_URL.rstrip('/')}{path}"
    headers = {"Content-Type": "application/json"}
    if EMBEDDING_API_KEY:
        headers["X-Api-Key"] = EMBEDDING_API_KEY

    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    req = request.Request(url=url, data=body, method=method, headers=headers)
    try:
        with request.urlopen(req, timeout=60) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        detail = raw
        try:
            detail = json.loads(raw)
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"HTTP {exc.code} {method} {path}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Cannot connect to embedding API at {EMBEDDING_API_URL}: {exc.reason}") from exc


@mcp.tool()
def healthz() -> Dict[str, Any]:
    """Check embedding service health and qdrant connectivity."""
    return _request_json("GET", "/healthz")


@mcp.tool()
def list_collections() -> Dict[str, Any]:
    """List available Qdrant collections."""
    return _request_json("GET", "/collections")


@mcp.tool()
def ensure_collection(name: str, recreate: bool = False) -> Dict[str, Any]:
    """Create collection if needed, optionally recreate it."""
    return _request_json("POST", f"/collections/{name}/ensure", {"recreate": recreate})


@mcp.tool()
def upsert(
    items: List[Dict[str, Any]],
    collection: str = "documents",
    prefix: str = "passage: ",
    strip: bool = True,
) -> Dict[str, Any]:
    """Upsert items to a collection. Each item: {id?, text, metadata?}."""
    payload = {
        "collection": collection,
        "prefix": prefix,
        "strip": strip,
        "items": items,
    }
    return _request_json("POST", "/upsert", payload)


@mcp.tool()
def search(
    query: str,
    collection: str = "documents",
    top_k: int = 5,
    with_payload: bool = True,
    score_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Vector similarity search from a query string."""
    payload: Dict[str, Any] = {
        "collection": collection,
        "query": query,
        "top_k": top_k,
        "with_payload": with_payload,
    }
    if score_threshold is not None:
        payload["score_threshold"] = score_threshold
    return _request_json("POST", "/search", payload)


@mcp.tool()
def query_hybrid(
    query: str,
    collection: str = "documents",
    top_k: int = 5,
    candidate_k: int = 30,
    alpha: float = 0.8,
) -> Dict[str, Any]:
    """Hybrid query: vector recall + lexical rerank."""
    return _request_json(
        "POST",
        "/query-hybrid",
        {
            "collection": collection,
            "query": query,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "alpha": alpha,
        },
    )


@mcp.tool()
def retrieve(ids: List[str], collection: str = "documents", with_payload: bool = True) -> Dict[str, Any]:
    """Retrieve points by IDs."""
    return _request_json(
        "POST",
        "/retrieve",
        {
            "collection": collection,
            "ids": ids,
            "with_payload": with_payload,
            "with_vectors": False,
        },
    )


@mcp.tool()
def delete(ids: List[str], collection: str = "documents", wait: bool = True) -> Dict[str, Any]:
    """Delete points by IDs."""
    return _request_json("POST", "/delete", {"collection": collection, "ids": ids, "wait": wait})


if __name__ == "__main__":
    mcp.run()
