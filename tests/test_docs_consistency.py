from pathlib import Path


README_FILES = ['README.md', 'README.en.md', 'README.zh.md']


def test_readmes_document_openai_and_audit_features():
    for readme in README_FILES:
        text = Path(readme).read_text()
        assert '/v1/embeddings' in text
        assert '/audit/events' in text
        assert 'ENABLE_RBAC' in text
        assert 'AUDIT_MAX_EVENTS' in text


def test_agent_md_includes_governance_commit_requirements():
    text = Path('AGENT.md').read_text()
    for token in [
        'OpenAI 风格 Embeddings API',
        '最小权限',
        '高风险操作',
        '代码改动',
        '测试改动',
        '文档改动',
        '回滚说明',
    ]:
        assert token in text


def test_test_report_exists_and_contains_iterations():
    text = Path('docs/TESTING_REPORT.md').read_text()
    assert 'Round 1' in text
    assert 'Round 2' in text
    assert '测试数据' in text
    assert '环境限制' in text
