# CLAUDE.md

## 契约驱动开发（自动触发）

当你要修改以下目录中的任何文件时，自动执行：

```
触发目录：apps/modelops_api/routers/
         apps/modelops_api/services/
         apps/modelops_api/repositories/
         packages/models/
```

**流程（不需要等用户指令）：**

1. 先读 `doc/前后端接口契约文档_V1.0.md` 和 `doc/接口约束总汇_V1.0.md`，确认接口形状
2. Pydantic 模型 → Repository（SQL）→ Service（业务逻辑）→ Router（HTTP）
3. `python scripts/generate_openapi.py` 刷新 OpenAPI
4. `python -m pytest tests/unit tests/test_openapi.py -q` 验证

**硬规则：**
- Router 请求体必须 Pydantic BaseModel，禁止裸 `body: dict`
- 资源不存在抛 `NotFoundError`（404），禁止 `ValueError`（500）
- 所有响应走统一包络 `{success, code, message, data, trace_id}`
- W4 违规错误码：`DATASET_POLICY_VIOLATION`
- 标签未成熟错误码：`LABEL_NOT_MATURE`
- 改完接口必须跑 `generate_openapi.py`

**架构边界（来自路线图）：**
- PostgreSQL = 业务事实唯一真相源
- MinIO = 文件/快照/模型产物
- Neo4j = 知识图谱（KnowledgeService 独占访问）
- Qdrant = 文档检索投影（可重建，不存唯一事实）
- MLflow = 实验追踪
- LLM = 知识抽取和解释，不直接决策

详细步骤见 `.claude/skills/contract-implement.md`
