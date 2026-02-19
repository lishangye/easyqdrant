import json
import os
from typing import Any, Dict, List, Optional, Union
from urllib import error, request

MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", "easyqdrant")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://127.0.0.1:18000")
EMBEDDING_API_KEY = os.getenv("EMBED_API_KEY", "")


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
        detail: Union[str, Dict[str, Any]] = raw
        try:
            detail = json.loads(raw)
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"HTTP {exc.code} {method} {path}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Cannot connect to embedding API at {EMBEDDING_API_URL}: {exc.reason}") from exc


def _build_mcp_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'mcp'. Install project dependencies (pip install -r requirements-mcp.txt) "
            "before running `python -m app.mcp_server`."
        ) from exc

    mcp = FastMCP(MCP_SERVER_NAME)

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
        prefix: str = "query: ",
        strip: bool = True,
        with_payload: bool = True,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
        hnsw_ef: Optional[int] = None,
        exact: bool = False,
    ) -> Dict[str, Any]:
        """Vector similarity search from a query string."""
        payload: Dict[str, Any] = {
            "collection": collection,
            "query": query,
            "top_k": top_k,
            "prefix": prefix,
            "strip": strip,
            "with_payload": with_payload,
            "exact": exact,
        }
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        if filter is not None:
            payload["filter"] = filter
        if hnsw_ef is not None:
            payload["hnsw_ef"] = hnsw_ef
        return _request_json("POST", "/search", payload)

    @mcp.tool()
    def query_hybrid(
        query: str,
        collection: str = "documents",
        top_k: int = 5,
        candidate_k: int = 30,
        prefix: str = "query: ",
        with_payload: bool = True,
        alpha: float = 0.8,
        hnsw_ef: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Hybrid query: vector recall + lexical rerank."""
        payload: Dict[str, Any] = {
            "collection": collection,
            "query": query,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "prefix": prefix,
            "with_payload": with_payload,
            "alpha": alpha,
        }
        if hnsw_ef is not None:
            payload["hnsw_ef"] = hnsw_ef
        return _request_json("POST", "/query-hybrid", payload)

    @mcp.tool()
    def retrieve(
        ids: List[Union[int, str]],
        collection: str = "documents",
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> Dict[str, Any]:
        """Retrieve points by IDs."""
        return _request_json(
            "POST",
            "/retrieve",
            {
                "collection": collection,
                "ids": ids,
                "with_payload": with_payload,
                "with_vectors": with_vectors,
            },
        )

    @mcp.tool()
    def delete(
        collection: str = "documents",
        ids: Optional[List[Union[int, str]]] = None,
        filter: Optional[Dict[str, Any]] = None,
        wait: bool = True,
    ) -> Dict[str, Any]:
        """Delete points by ids or filter."""
        payload: Dict[str, Any] = {
            "collection": collection,
            "wait": wait,
        }
        if ids is not None:
            payload["ids"] = ids
        if filter is not None:
            payload["filter"] = filter
        return _request_json("POST", "/delete", payload)

    return mcp


if __name__ == "__main__":
    _build_mcp_server().run()
