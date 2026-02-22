import ast
from pathlib import Path

SOURCE = Path('app/main.py').read_text()
MODULE = ast.parse(SOURCE)


def _function_names():
    return {n.name for n in MODULE.body if isinstance(n, ast.FunctionDef)}


def test_required_http_routes_declared():
    required_route_snippets = [
        '@app.post("/embed"',
        '@app.post("/v1/embeddings"',
        '@app.post("/upsert"',
        '@app.post("/search"',
        '@app.post("/query-hybrid"',
        '@app.post("/retrieve"',
        '@app.post("/scroll"',
        '@app.post("/update-payload"',
        '@app.post("/delete"',
        '@app.get("/audit/events"',
    ]
    for snippet in required_route_snippets:
        assert snippet in SOURCE


def test_required_security_helpers_exist():
    names = _function_names()
    assert '_resolve_role' in names
    assert '_authorize' in names
    assert '_audit_event' in names
    assert '_hash_query' in names


def test_openai_models_declared():
    class_names = {n.name for n in MODULE.body if isinstance(n, ast.ClassDef)}
    for name in [
        'OpenAIEmbeddingsRequest',
        'OpenAIEmbeddingItem',
        'OpenAIUsage',
        'OpenAIEmbeddingsResponse',
        'AuditEvent',
        'AuditEventsResponse',
    ]:
        assert name in class_names


def test_openai_endpoint_compatibility_fields_present():
    for token in ['input', 'model', 'encoding_format', 'user', 'prompt_tokens', 'total_tokens']:
        assert token in SOURCE
