"""数据隔离集成测试 — 需要数据库"""

import os

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from apps.modelops_api.config import settings
from apps.modelops_api.repositories.audit_repo import AuditRepo
from apps.modelops_api.services.dataset_access_policy import DatasetAccessPolicy

pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_INTEGRATION", "true") == "true",
    reason="SKIP_INTEGRATION=true",
)


W4_FLAGS = {
    "allows_training": False,
    "allows_iteration_label": False,
    "allows_deployment_label": True,
    "is_frozen": True,
}


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine(settings.database_url, echo=False)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:
        yield s
        await s.rollback()
    await engine.dispose()


class TestDataIsolation:
    async def test_iteration_cannot_read_w4_labels_db(self, session):
        """任务三请求 W4 标签 → 拒绝并携带契约错误码。"""
        policy = DatasetAccessPolicy(session)
        with pytest.raises(Exception) as exc_info:
            await policy.validate_access(
                {**W4_FLAGS, "window_id": "W4_001", "window_name": "W4_OOT"},
                "TASK_3",
                "READ_LABEL",
            )
        assert exc_info.value.status_code == 403
        assert exc_info.value.code == "DATASET_POLICY_VIOLATION"

    async def test_deployment_can_read_w4_db(self, session):
        """任务四可以读取 W4。"""
        policy = DatasetAccessPolicy(session)
        await policy.validate_access(
            {**W4_FLAGS, "window_id": "W4_001", "window_name": "W4_OOT"},
            "TASK_4",
            "READ_LABEL",
        )
        # 不抛异常即通过

    async def test_training_cannot_use_w4_db(self, session):
        """W4 训练 → 违规。"""
        policy = DatasetAccessPolicy(session)
        with pytest.raises(Exception) as exc_info:
            await policy.validate_access(
                {**W4_FLAGS, "window_id": "W4_001", "window_name": "W4_OOT"},
                "TASK_3",
                "TRAINING",
            )
        assert exc_info.value.status_code == 403

    async def test_violation_written_to_audit(self, session):
        """违规写入 audit.data_access_violations（Repo 直写路径）。"""
        audit = AuditRepo(session)
        result = await audit.log_violation(
            task_phase="TASK_3",
            violation_code="W4_LABEL_ACCESS_VIOLATION",
            model_id="test_model",
            window_id="W4_001",
            attempted_operation="READ_LABEL",
        )
        assert result["violation_id"] is not None
        await session.flush()

    async def test_policy_violation_written_to_audit(self, session):
        """回归 RISK-001：经 validate_access 策略路径触发违规，审计表必须出现记录。"""
        policy = DatasetAccessPolicy(session)
        with pytest.raises(Exception):
            await policy.validate_access(
                {**W4_FLAGS, "window_id": "W4_RISK001", "window_name": "W4_OOT"},
                "TASK_3",
                "READ_LABEL",
            )
        result = await session.execute(
            text("""
                SELECT violation_code, task_phase, detail_json
                FROM audit.data_access_violations
                WHERE window_id = 'W4_RISK001'
                ORDER BY created_at DESC
            """)
        )
        row = result.mappings().first()
        assert row is not None, "策略路径未写入审计记录（RISK-001 未修复）"
        assert row["violation_code"] == "W4_LABEL_ACCESS_VIOLATION"
        assert row["task_phase"] == "TASK_3"
