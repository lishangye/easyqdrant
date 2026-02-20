import ast
from pathlib import Path


MODULE = ast.parse(Path('app/main.py').read_text())


def _literal_for(name: str):
    for node in MODULE.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f'{name} not found')


def test_role_permissions_have_expected_baseline():
    perms = _literal_for('ROLE_PERMISSIONS')
    assert set(perms) == {'reader', 'writer', 'admin', 'auditor'}

    assert 'vectors.search' in perms['reader']
    assert 'vectors.retrieve' in perms['reader']
    assert 'vectors.delete' not in perms['reader']

    assert 'vectors.upsert' in perms['writer']
    assert 'vectors.delete' in perms['writer']
    assert 'collections.ensure' not in perms['writer']

    assert perms['admin'] == ['*']
    assert perms['auditor'] == ['audit.read', 'collections.list', 'collections.stats']


def test_collection_patterns_defined_for_all_roles():
    patterns = _literal_for('ROLE_COLLECTION_PATTERNS')
    assert set(patterns) == {'reader', 'writer', 'admin', 'auditor'}
    for role, values in patterns.items():
        assert isinstance(values, list)
        assert values, role
