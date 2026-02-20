import json
from pathlib import Path


def test_topics_json_valid_and_non_empty():
    config = json.loads(Path('.github/topics.json').read_text())
    assert isinstance(config.get('topics'), list)
    assert config['topics']


def test_workflow_references_topics_json_and_contains_expected_logic():
    text = Path('.github/workflows/sync-topics.yml').read_text()
    assert '.github/topics.json' in text
    assert 'JSON.parse' in text
    assert 'PUT /repos/{owner}/{repo}/topics' in text
