# Agent Long-term Memory API 设计与使用说明

本文档定义 easyqdrant 的 Agent 长期记忆接口族，目标是让机器人/Agent 能直接理解并使用。

## 1. 设计目标
- 提供稳定、可审计、可权限控制的长期记忆接口。
- 兼容多 Agent / 多会话场景（`agent_id`、`session_id`）。
- 与现有向量能力复用，避免重复实现。

## 2. 安全与权限模型
- 通用请求头：
  - `X-Api-Key`：服务鉴权（可选，按部署配置）。
  - `X-Role`：角色（`reader|writer|admin|auditor`）。
  - `X-Actor`：调用主体标识（用于审计）。
- RBAC 开关：`ENABLE_RBAC=true` 时生效。
- 权限语义：
  - `memory.read`：读记忆（query/get/scroll/stats）
  - `memory.write`：写记忆（write/update/forget）
  - `collections.ensure`：管理 memory space（创建/重建）

## 3. Memory 数据结构约定
每条 memory 写入 payload 建议包含：
- `role`: 语义类别（如 `fact` / `preference` / `plan`）
- `importance`: [0,1]
- `tags`: 标签数组
- `source`: 来源
- `session_id`: 会话
- `agent_id`: 机器人实例
- `created_by`: 调用主体

## 4. API 一览
1. `POST /memory/spaces/ensure`
2. `POST /memory/write`
3. `POST /memory/query`
4. `POST /memory/get`
5. `POST /memory/update`
6. `POST /memory/forget`
7. `POST /memory/scroll`
8. `GET /memory/spaces/{collection}/stats`

---

## 5. 详细接口

## 5.1 Ensure memory space
`POST /memory/spaces/ensure`

请求：
```json
{
  "collection": "agent_memory",
  "recreate": false
}
```

说明：
- 用于初始化长期记忆 collection。
- `recreate=true` 是高风险操作，建议仅 admin 使用。

## 5.2 Write memories
`POST /memory/write`

请求：
```json
{
  "collection": "agent_memory",
  "prefix": "memory: ",
  "items": [
    {
      "id": "mem-1",
      "text": "用户喜欢简洁回答",
      "role": "preference",
      "importance": 0.9,
      "tags": ["style", "user_profile"],
      "source": "conversation",
      "session_id": "sess-001",
      "agent_id": "agent-A",
      "metadata": {"lang": "zh"}
    }
  ]
}
```

## 5.3 Query memories
`POST /memory/query`

请求：
```json
{
  "collection": "agent_memory",
  "query": "用户回答风格偏好",
  "top_k": 10,
  "min_importance": 0.5,
  "tags_any": ["style"],
  "session_id": "sess-001",
  "agent_id": "agent-A",
  "with_payload": true
}
```

## 5.4 Get memories by ids
`POST /memory/get`

请求：
```json
{
  "collection": "agent_memory",
  "ids": ["mem-1", "mem-2"]
}
```

## 5.5 Update memory payload
`POST /memory/update`

请求：
```json
{
  "collection": "agent_memory",
  "ids": ["mem-1"],
  "set_payload": {
    "importance": 0.95,
    "tags": ["style", "priority"]
  },
  "wait": true
}
```

## 5.6 Forget memories
`POST /memory/forget`

两种模式：
- 按 id 删除
- 按过滤删除（`session_id`/`agent_id`/`tags_any`）

示例（按过滤）：
```json
{
  "collection": "agent_memory",
  "session_id": "sess-001",
  "tags_any": ["temporary"],
  "wait": true
}
```

## 5.7 Scroll memories
`POST /memory/scroll`

请求：
```json
{
  "collection": "agent_memory",
  "limit": 20,
  "offset": null,
  "agent_id": "agent-A"
}
```

## 5.8 Memory space stats
`GET /memory/spaces/{collection}/stats`

示例：
```bash
curl -H "X-Role: reader" http://127.0.0.1:18000/memory/spaces/agent_memory/stats
```

## 6. 推荐调用流程（Agent）
1. 启动时 ensure space
2. 对新事实执行 write
3. 推理前 query（带 session_id + agent_id）
4. 任务完成后 update importance / forget 临时记忆
5. 定期审计 `/audit/events`

## 7. 常见错误
- 403：角色无权限（RBAC 开启）
- 400：forget 未提供 ids 或过滤条件
- 404：collection 不存在（需先 ensure）

## 8. 兼容建议
- `collection` 命名按租户/场景划分（如 `agent_mem_prod`, `agent_mem_test`）。
- `tags` 尽量标准化，便于批量过滤和治理。
