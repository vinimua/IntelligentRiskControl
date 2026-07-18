# 契约驱动实现

> **自动触发**：当 AI 计划新增或修改以下目录中的文件时，必须先过本流程。
>
> 触发目录：
> - `apps/modelops_api/routers/`
> - `apps/modelops_api/services/`
> - `apps/modelops_api/repositories/`
> - `packages/models/`
>
> **不需要等用户说"按契约实现"。AI 自己决定改接口的那一刻，就自动走这个流程。**

## 前置：查契约文档

在动手写任何代码之前，先读这两份文件确认接口的形状：

```
doc/前后端接口契约文档_V1.0.md   — 接口定义（方法、路径、请求体、响应体、权限）
doc/接口约束总汇_V1.0.md         — 全局约束（统一包络格式、错误码表 §1.8、时间格式 §1.5）
```

**如果契约文档里没有这个接口** → 停下来告诉用户"契约里没定义这个接口，需要先补契约还是直接实现？"

**如果接口在契约里** → 从 JSON 示例中提取字段名、类型、必填性。然后走下面的流程。

## 执行流程

### 1. Pydantic 模型（`packages/models/`）

检查是否已有可复用的 ContractModel：
- 请求体字段在已有模型中 → 复用，不新建
- 请求体是新结构 → 新增 ContractModel 类
- 响应体字段从已有模型选取，Router 层用 `_envelope()` 包裹

### 2. 三层实现（严格此顺序）

```
Repository → Service → Router
```

| 层 | 目录 | 规则 |
|---|---|---|
| Repository | `apps/modelops_api/repositories/` | 只写 SQL（`text()`），返回 `dict` |
| Service | `apps/modelops_api/services/` | 业务逻辑 + 异常（`NotFoundError` / `ForbiddenError` / `ConflictError`） |
| Router | `apps/modelops_api/routers/` | Pydantic `BaseModel` 请求体 + `@router` 装饰器 + `_envelope()` 返回 |

**强制规则：**
- Router 层请求体 **必须用 Pydantic BaseModel**，禁止裸 `body: dict`
- 资源不存在 → `NotFoundError`（404），禁止 `ValueError`（→ 500）
- 所有响应走 `_envelope(request, data)` 统一包络：`{success, code, message, data, trace_id}`
- 错误码对照契约 §1.8：
  - W4 违规 → `ForbiddenError(code="DATASET_POLICY_VIOLATION")`
  - 标签未成熟 → `ConflictError(code="LABEL_NOT_MATURE")`
  - 参数校验失败 → FastAPI 自动 422，不需要手动处理
- 如果新增了 Router 文件 → 在 `apps/modelops_api/main.py` 中 `include_router`

### 3. 自动生成 OpenAPI 并验证

```bash
python scripts/generate_openapi.py          # 生成 OpenAPI JSON + YAML
python scripts/generate_openapi.py --check  # 验证产物与代码一致
```

### 4. 补充测试

每个接口至少覆盖：
- 正常路径（200）
- 参数校验（422）—— Pydantic 自动拦，但必须有测试证明拦了
- 资源不存在（404）
- 业务规则拒绝（403/409）

### 5. 跑全部测试

```bash
python -m pytest tests/unit tests/test_openapi.py -q
```

### 6. 输出报告

```
新增/修改文件：
  packages/models/xxx/model.py        ← （如有新 Pydantic 模型）
  apps/modelops_api/repositories/xxx_repo.py
  apps/modelops_api/services/xxx_service.py
  apps/modelops_api/routers/xxx.py
  tests/unit/test_xxx.py              ← （如有新测试）

OpenAPI:  python scripts/generate_openapi.py --check → PASS
测试:     python -m pytest → N passed
```

## 禁止事项

- ❌ 改 `doc/` 下的契约源文件（只读源头）
- ❌ skip OpenAPI 生成步骤（改完接口不跑 `generate_openapi.py`）
- ❌ Router 层裸写 SQL（必须走 Repo）
- ❌ 裸 `body: dict`（必须 Pydantic BaseModel）
- ❌ `ValueError` → 500（必须 `NotFoundError` → 404）
- ❌ 非统一包络格式的响应
