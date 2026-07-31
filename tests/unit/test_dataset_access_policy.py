"""DatasetAccessPolicy 单元测试"""

from datetime import datetime, timezone

import pytest

from apps.modelops_api.core.exceptions import ConflictError, ForbiddenError
from apps.modelops_api.services.dataset_access_policy import DatasetAccessPolicy

W0_WINDOW = {
    "window_id": "W0_001",
    "window_name": "W0-初始训练",
    "allows_training": True,
    "allows_monitoring_label": True,
    "allows_diagnosis_label": True,
    "allows_iteration_label": True,
    "allows_deployment_label": False,
    "is_frozen": False,
}

W3_WINDOW = {
    "window_id": "W3_001",
    "window_name": "W3-近期训练",
    "allows_training": True,
    "allows_monitoring_label": True,
    "allows_diagnosis_label": True,
    "allows_iteration_label": True,
    "allows_deployment_label": False,
    "is_frozen": False,
}

W4_WINDOW = {
    "window_id": "W4_001",
    "window_name": "W4-OOT独立准入",
    "allows_training": False,
    "allows_monitoring_label": False,
    "allows_diagnosis_label": False,
    "allows_iteration_label": False,
    "allows_deployment_label": True,
    "is_frozen": True,
}


class TestDatasetAccessPolicy:
    """测试 W4/OOT 数据隔离规则 — 无需数据库连接。"""

    def test_iteration_cannot_read_w4_labels(self):
        """任务三不得读取 W4/OOT 标签。"""
        policy = DatasetAccessPolicy.__new__(DatasetAccessPolicy)
        assert policy.can_read_labels(W4_WINDOW, "TASK_3") is False

    def test_iteration_cannot_use_w4_training(self):
        """任务三不得使用 W4/OOT 做训练。"""
        policy = DatasetAccessPolicy.__new__(DatasetAccessPolicy)
        assert policy.can_use_for_training(W4_WINDOW, "TASK_3") is False

    def test_training_cannot_use_w4(self):
        """任何任务都不能用 W4/OOT 做训练。"""
        policy = DatasetAccessPolicy.__new__(DatasetAccessPolicy)
        assert policy.can_use_for_training(W4_WINDOW, "TASK_1") is False
        assert policy.can_use_for_training(W4_WINDOW, "TASK_2") is False
        assert policy.can_use_for_training(W4_WINDOW, "TASK_3") is False
        assert policy.can_use_for_training(W4_WINDOW, "TASK_4") is False

    def test_deployment_can_read_w4(self):
        """任务四可以读取 W4/OOT 标签。"""
        policy = DatasetAccessPolicy.__new__(DatasetAccessPolicy)
        assert policy.can_read_labels(W4_WINDOW, "TASK_4") is True

    def test_task3_can_read_w3_labels(self):
        """任务三应该可以读取 W3 标签。"""
        policy = DatasetAccessPolicy.__new__(DatasetAccessPolicy)
        assert policy.can_read_labels(W3_WINDOW, "TASK_3") is True

    def test_task1_cannot_train(self):
        """任务一不做训练。"""
        policy = DatasetAccessPolicy.__new__(DatasetAccessPolicy)
        assert policy.can_use_for_training(W0_WINDOW, "TASK_1") is False

    def test_task3_can_train_w3(self):
        """任务三可以使用 W3 做训练。"""
        policy = DatasetAccessPolicy.__new__(DatasetAccessPolicy)
        assert policy.can_use_for_training(W3_WINDOW, "TASK_3") is True

    def test_w4_required_flags(self):
        """W4 窗口的硬约束字段必须正确。"""
        assert W4_WINDOW["allows_training"] is False
        assert W4_WINDOW["allows_iteration_label"] is False
        assert W4_WINDOW["allows_deployment_label"] is True
        assert W4_WINDOW["is_frozen"] is True


class _StubAuditRepo:
    def __init__(self):
        self.calls = []

    async def log_violation(self, **kwargs):
        self.calls.append(kwargs)
        return {"violation_id": "stub"}


class _StubSession:
    async def flush(self):
        pass


def _policy_with_stub_audit() -> DatasetAccessPolicy:
    policy = DatasetAccessPolicy.__new__(DatasetAccessPolicy)
    policy.audit_repo = _StubAuditRepo()
    policy.session = _StubSession()
    return policy


class TestValidateAccessAudit:
    """validate_access 必须真正 await 审计写入（回归 RISK-001）。"""

    async def test_label_violation_writes_audit_and_uses_contract_code(self):
        policy = _policy_with_stub_audit()
        with pytest.raises(ForbiddenError) as exc_info:
            await policy.validate_access(W4_WINDOW, "TASK_3", "READ_LABEL")
        assert exc_info.value.code == "DATASET_POLICY_VIOLATION"
        assert exc_info.value.status_code == 403
        assert policy.audit_repo.calls[0]["violation_code"] == "W4_LABEL_ACCESS_VIOLATION"
        assert policy.audit_repo.calls[0]["task_phase"] == "TASK_3"

    async def test_training_violation_writes_audit(self):
        policy = _policy_with_stub_audit()
        with pytest.raises(ForbiddenError) as exc_info:
            await policy.validate_access(W4_WINDOW, "TASK_3", "TRAINING")
        assert exc_info.value.code == "DATASET_POLICY_VIOLATION"
        assert policy.audit_repo.calls[0]["violation_code"] == "W4_TRAINING_VIOLATION"

    async def test_allowed_access_writes_no_audit(self):
        policy = _policy_with_stub_audit()
        await policy.validate_access(W4_WINDOW, "TASK_4", "READ_LABEL")
        assert policy.audit_repo.calls == []


class TestLabelMaturity:
    """LABEL_NOT_MATURE 校验（回归 RISK-002）。"""

    def test_immature_label_raises_label_not_mature(self):
        policy = DatasetAccessPolicy.__new__(DatasetAccessPolicy)
        maturity = datetime(2026, 12, 31, tzinfo=timezone.utc)
        as_of = datetime(2026, 7, 18, tzinfo=timezone.utc)
        with pytest.raises(ConflictError) as exc_info:
            policy.validate_label_maturity(maturity, "TASK_2", as_of=as_of)
        assert exc_info.value.code == "LABEL_NOT_MATURE"

    def test_mature_label_passes(self):
        policy = DatasetAccessPolicy.__new__(DatasetAccessPolicy)
        maturity = datetime(2026, 1, 1, tzinfo=timezone.utc)
        as_of = datetime(2026, 7, 18, tzinfo=timezone.utc)
        policy.validate_label_maturity(maturity, "TASK_2", as_of=as_of)

    def test_none_maturity_passes(self):
        policy = DatasetAccessPolicy.__new__(DatasetAccessPolicy)
        policy.validate_label_maturity(None, "TASK_2")

    def test_naive_datetime_treated_as_utc(self):
        policy = DatasetAccessPolicy.__new__(DatasetAccessPolicy)
        maturity = datetime(2026, 12, 31)  # naive
        as_of = datetime(2026, 7, 18, tzinfo=timezone.utc)
        with pytest.raises(ConflictError):
            policy.validate_label_maturity(maturity, "TASK_2", as_of=as_of)
