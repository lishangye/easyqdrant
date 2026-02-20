import ast
from pathlib import Path


def _get_module():
    return ast.parse(Path('app/main.py').read_text())


def test_openai_embeddings_models_and_route_exist():
    mod = _get_module()
    class_names = {n.name for n in mod.body if isinstance(n, ast.ClassDef)}
    assert 'OpenAIEmbeddingsRequest' in class_names
    assert 'OpenAIEmbeddingsResponse' in class_names

    source = Path('app/main.py').read_text()
    assert '@app.post("/v1/embeddings"' in source


def test_rbac_and_audit_helpers_exist():
    mod = _get_module()
    fn_names = {n.name for n in mod.body if isinstance(n, ast.FunctionDef)}
    for required in ('_resolve_role', '_authorize', '_audit_event'):
        assert required in fn_names

    source = Path('app/main.py').read_text()
    assert 'ENABLE_RBAC' in source
    assert 'ROLE_PERMISSIONS' in source
    assert '@app.get("/audit/events"' in source


def test_sensitive_actions_record_audit_events():
    source = Path('app/main.py').read_text()
    assert '"vectors.delete"' in source
    assert '"collections.ensure"' in source
    assert '_audit_event(' in source
