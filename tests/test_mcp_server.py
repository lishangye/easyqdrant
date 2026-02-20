import types
import sys

import app.mcp_server as mcp_server


class DummyFastMCP:
    def __init__(self, name: str):
        self.name = name
        self.tools = {}

    def tool(self):
        def deco(func):
            self.tools[func.__name__] = func
            return func

        return deco


def build_server_with_dummy_mcp(monkeypatch):
    monkeypatch.setitem(sys.modules, "mcp", types.ModuleType("mcp"))
    monkeypatch.setitem(sys.modules, "mcp.server", types.ModuleType("mcp.server"))
    fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
    fastmcp_mod.FastMCP = DummyFastMCP
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_mod)
    return mcp_server._build_mcp_server()


def test_build_mcp_server_registers_tools(monkeypatch):
    srv = build_server_with_dummy_mcp(monkeypatch)
    assert set(srv.tools) == {
        "healthz",
        "list_collections",
        "ensure_collection",
        "upsert",
        "search",
        "query_hybrid",
        "retrieve",
        "delete",
    }


def test_search_payload_contains_advanced_options(monkeypatch):
    srv = build_server_with_dummy_mcp(monkeypatch)
    captured = {}

    def fake_request_json(method, path, payload=None):
        captured["method"] = method
        captured["path"] = path
        captured["payload"] = payload
        return {"ok": True}

    monkeypatch.setattr(mcp_server, "_request_json", fake_request_json)

    result = srv.tools["search"](
        query="hello",
        collection="documents",
        top_k=3,
        prefix="query: ",
        strip=False,
        with_payload=False,
        score_threshold=0.9,
        filter={"must": []},
        hnsw_ef=64,
        exact=True,
    )

    assert result == {"ok": True}
    assert captured == {
        "method": "POST",
        "path": "/search",
        "payload": {
            "collection": "documents",
            "query": "hello",
            "top_k": 3,
            "prefix": "query: ",
            "strip": False,
            "with_payload": False,
            "score_threshold": 0.9,
            "filter": {"must": []},
            "hnsw_ef": 64,
            "exact": True,
        },
    }


def test_delete_payload_accepts_filter_without_ids(monkeypatch):
    srv = build_server_with_dummy_mcp(monkeypatch)
    captured = {}

    def fake_request_json(method, path, payload=None):
        captured["payload"] = payload
        return {"status": "ok"}

    monkeypatch.setattr(mcp_server, "_request_json", fake_request_json)

    srv.tools["delete"](collection="documents", filter={"must": []}, wait=False)

    assert captured["payload"] == {
        "collection": "documents",
        "wait": False,
        "filter": {"must": []},
    }
