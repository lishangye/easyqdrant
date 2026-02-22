from pathlib import Path


SOURCE = Path('app/main.py').read_text()


def test_memory_routes_present():
    required = [
        '@app.post("/memory/spaces/ensure")',
        '@app.post("/memory/write")',
        '@app.post("/memory/query")',
        '@app.post("/memory/get")',
        '@app.post("/memory/update")',
        '@app.post("/memory/forget")',
        '@app.post("/memory/scroll")',
        '@app.get("/memory/spaces/{collection}/stats")',
    ]
    for item in required:
        assert item in SOURCE


def test_memory_models_present():
    required_tokens = [
        'class MemorySpaceEnsureRequest',
        'class MemoryWriteRequest',
        'class MemoryQueryRequest',
        'class MemoryGetRequest',
        'class MemoryUpdateRequest',
        'class MemoryForgetRequest',
        'class MemoryScrollRequest',
        'class MemoryPutItem',
    ]
    for token in required_tokens:
        assert token in SOURCE


def test_memory_filter_helper_present():
    assert 'def _memory_filter(' in SOURCE
    for token in ['session_id', 'agent_id', 'tags_any', 'min_importance']:
        assert token in SOURCE
