# easyqdrant 测试记录与验证报告

> 目标：对当前版本进行尽可能全面的测试与文档校验，覆盖接口兼容性、安全策略、审计能力、配置一致性，并持续迭代到测试稳定通过。

## 环境限制
- 当前执行环境无法通过代理正常下载外部依赖（`pip install` 访问 pypi 返回 403）。
- 因此本轮以 **静态契约测试 + AST/文本一致性测试** 为主，确保在受限环境下仍有高覆盖验证。

## 测试覆盖矩阵
1. **API 契约（静态）**
   - 路由存在性：`/embed`、`/v1/embeddings`、`/search`、`/delete`、`/audit/events` 等。
   - OpenAI Embeddings 字段：`input/model/encoding_format/user`、`usage.prompt_tokens/total_tokens`。
2. **RBAC 策略（静态）**
   - 角色集合：`reader/writer/admin/auditor`。
   - 行为基线：reader 不可 delete；writer 可 upsert/delete；admin 通配；auditor 仅审计读取。
   - collection pattern 映射完整性。
3. **审计能力（静态）**
   - 审计辅助函数存在：`_audit_event`、`_hash_query`、`_authorize`。
   - 高风险动作关键字：`collections.ensure`、`vectors.delete`。
4. **配置与文档一致性**
   - topics/workflow 配置引用正确。
   - README 中英中三份文档覆盖新增接口和安全环境变量。
   - `AGENT.md` 覆盖协作基线、提交规范、安全基线。

## 测试数据
### A. OpenAI Embeddings 请求样例
- A1（单条）
```json
{"input": "hello", "model": "BAAI/bge-small-zh-v1.5", "encoding_format": "float", "user": "u1"}
```
- A2（多条）
```json
{"input": ["你好", "world"], "model": "BAAI/bge-small-zh-v1.5", "encoding_format": "float"}
```

### B. RBAC 权限样例
- B1: `reader -> vectors.search`（应允许）
- B2: `reader -> vectors.delete`（应拒绝）
- B3: `writer -> vectors.upsert`（应允许）
- B4: `writer -> collections.ensure`（应拒绝）
- B5: `auditor -> audit.read`（应允许）

### C. 审计查询样例
- C1: `GET /audit/events?limit=100`
- C2: `GET /audit/events?action=vectors.search&collection=documents`

## 迭代测试记录

## Round 1（发现问题）
执行：
1. `python -m compileall -q app tests scripts && echo COMPILE_ROUND1_OK`
2. `pytest -q`

结果：
- 编译通过（`COMPILE_ROUND1_OK`）。
- 测试：`15 passed, 2 failed`。
- 失败原因：`tests/test_rbac_policy_static.py` 仅处理 `ast.Assign`，未处理 `ROLE_PERMISSIONS`/`ROLE_COLLECTION_PATTERNS` 的 `ast.AnnAssign` 形式。

修复：
- 更新 `_literal_for` 同时支持 `ast.Assign` 与 `ast.AnnAssign`。

## Round 2（修复后复测）
执行：
1. `python -m compileall -q app tests scripts && echo COMPILE_ROUND2_OK`
2. `pytest -q`

结果：
- 编译通过（`COMPILE_ROUND2_OK`）。
- 测试全通过：`17 passed`。

## Round 3（文档与测试最终确认）
执行：
1. `pytest -q`

结果：
- 测试全通过：`17 passed`。

## Round 4（文档补充后再次验证）
执行：
1. `python -m compileall -q app tests scripts`
2. `pytest -q`

结果：
- 测试全通过：`17 passed`。

## 安全性 / 可用性 / 性能平衡分析
- **安全性**：
  - 引入 RBAC 和 collection 级约束，并对关键动作加入审计。
- **可用性**：
  - 保留原 `/embed`，新增 `/v1/embeddings`，兼容存量客户端与 OpenAI 风格客户端。
- **性能**：
  - 审计存储使用有界 `deque(maxlen=AUDIT_MAX_EVENTS)`，防止无限内存增长。

## 结论
- 已完成多轮迭代测试并收敛到稳定通过状态（`17 passed`）。
- 在当前网络受限环境下，静态契约层验证已做到高覆盖。
- 建议在可联网 CI 环境追加：
  1) FastAPI TestClient 集成测试；
  2) Qdrant 联调 E2E；
  3) 并发/压测（search、upsert、audit endpoint）。
