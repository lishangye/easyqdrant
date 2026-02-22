# AGENT.md

本文件定义 easyqdrant 的协作开发基线，面向后续功能扩展（OpenAI Embeddings 风格接口、RBAC、审计查询、MCP 桥接）。

## 1. 项目目标
- 提供轻量、可部署、可观测的向量化与检索服务。
- 在保持现有 `/embed` 与向量库接口能力的同时，逐步兼容 OpenAI 风格 Embeddings API（`/v1/embeddings`）。
- 提供可治理能力：最小权限、按 collection 授权、关键操作审计、审核查询。

## 2. 目录职责
- `app/main.py`：核心 HTTP API（embedding、collection、qdrant CRUD、权限与审计相关能力）。
- `app/mcp_server.py`：MCP 工具桥接层（不承载业务状态）。
- `tests/`：单元测试与配置回归测试。
- `.github/workflows/`：CI 与仓库元数据同步。
- `README*.md`：中英双语用户文档与运维说明。

## 3. 接口兼容目标
### 3.1 Embeddings
- 保留现有：`POST /embed`（项目原生接口）。
- 新增兼容：`POST /v1/embeddings`（OpenAI 风格）。
- v1 目标：功能兼容优先，支持 `input/model/encoding_format/user` 与标准响应结构。
- v2 目标：补充 token 统计精度、错误码对齐、更多兼容字段。

### 3.2 稳定性要求
- 新接口不破坏现有接口语义。
- 所有新增字段必须向后兼容（默认值可用）。

## 4. 提交规范（必须遵守）
每个涉及 API 的提交必须同时包含：
1. **代码改动**。
2. **测试改动**（至少覆盖新增/修改路径）。
3. **文档改动**（README 或 API 说明）。
4. **回滚说明**（PR 描述内给出回滚点/开关）。

建议 commit 前缀：`feat:` `fix:` `refactor:` `docs:` `test:` `chore:`。

## 5. 安全基线
### 5.1 默认最小权限
- 默认角色应为只读（reader）或更低。
- 高风险能力（delete/recreate）必须显式授予。

### 5.2 授权模型
- 采用 `role -> action -> collection_pattern`。
- 至少包含角色：`reader`, `writer`, `admin`, `auditor`。
- collection 访问必须可配置（如 `finance_*`）。

### 5.3 审计要求
- 高风险操作必须审计：`delete`, `ensure(recreate=true)`。
- 查询类操作需可追踪：`search/query-hybrid/retrieve`。
- 审计字段最小集：时间、actor、role、action、collection、query_hash、结果数量、状态。

## 6. 性能与可用性平衡
- 优先保证核心路径低延迟（embed/search）。
- 审计记录默认轻量化（内存环形缓冲或异步写入），避免阻塞主路径。
- 对大批量写入保持批处理（已有 `UPSERT_BATCH_SIZE` 约束）。

## 7. 开发迭代建议
1. 先完成最小闭环（接口 + 测试 + 文档）。
2. 再做细节对齐（错误码、token 统计、策略管理）。
3. 每次迭代必须验证：安全、可用性、性能三者平衡。

## 8. 本地验证建议命令
- `python -m compileall -q app tests scripts`
- `pytest -q`
- `python -m app.main`（或 uvicorn 启动）
- 对 OpenAI 风格接口进行最小 curl 验证

