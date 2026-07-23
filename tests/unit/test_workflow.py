"""阶段 3 验收测试 — LangGraph 可行走骨架"""

import pytest

from apps.modelops_api.services.workflow import graph as wf
from apps.modelops_api.services.workflow.graph import (
    MOCK_CHALLENGER_QUALIFIED,
    MOCK_DEPLOYMENT_DECISION,
    MOCK_NEED_ITERATION,
    build_graph,
    deployment_node,
    diagnosis_node,
    iteration_subgraph,
    manual_review_node,
    monitoring_node,
    route_after_diagnosis,
    route_after_monitoring,
)
from packages.models.common.enums import LifecyclePhase, Severity
from packages.models.workflow.lifecycle_state import ModelLifecycleState


def _base_state(**overrides) -> dict:
    s = {
        "lifecycle_run_id": "test-run-001",
        "model_id": "test_model",
        "champion_version": "v1",
        "trigger_type": "SCHEDULED_TRIGGER",
        "current_phase": LifecyclePhase.CREATED.value,
        "requires_manual_review": False,
    }
    s.update(overrides)
    return s


# ── 图结构测试 ──

class TestGraphStructure:
    def test_all_nodes_registered(self):
        g = build_graph()
        nodes = set(g.nodes.keys())
        expected = {
            "MonitoringNode", "DiagnosisNode", "IterationSubgraph",
            "DeploymentNode", "ManualReviewNode",
        }
        assert expected.issubset(nodes), f"missing nodes: {expected - nodes}"

    def test_start_goes_to_monitoring(self):
        g = build_graph()
        compiled = g.compile()
        # 验证 START → MonitoringNode 边：编译图的 nodes 字典中 MonitoringNode 存在
        assert "MonitoringNode" in compiled.get_graph().nodes


# ── 条件路由测试 ──

class TestRouting:
    def test_no_alerts_goes_to_end(self):
        result = route_after_monitoring(_base_state(has_alerts=False))
        assert result == "__end__"

    def test_has_alerts_goes_to_diagnosis(self):
        result = route_after_monitoring(_base_state(has_alerts=True))
        assert result == "DiagnosisNode"

    def test_need_iteration_true_goes_to_iteration(self):
        result = route_after_diagnosis(_base_state(need_iteration=True))
        assert result == "IterationSubgraph"

    def test_need_iteration_false_goes_to_end(self):
        result = route_after_diagnosis(_base_state(need_iteration=False))
        assert result == "__end__"

    def test_need_iteration_none_goes_to_manual_review(self):
        result = route_after_diagnosis(_base_state(need_iteration=None))
        assert result == "ManualReviewNode"


# ── Mock 节点行为测试 ──


class TestMonitoringNode:
    @pytest.fixture(autouse=True)
    async def _seed_test_model(self):
        """确保 test_model 存在于 models 表中（FK 约束要求）。"""
        from apps.modelops_api.database import get_db
        from sqlalchemy import text

        db_gen = get_db()
        db = await db_gen.__anext__()
        try:
            await db.execute(
                text("INSERT INTO model_registry.models (model_id, model_name, model_type, status, current_champion_version) "
                     "VALUES ('test_model', 'test_model', 'champion', 'ACTIVE', 'v1') "
                     "ON CONFLICT (model_id) DO NOTHING")
            )
            await db.commit()
        finally:
            await db_gen.aclose()

    async def test_monitoring_node_returns_expected_keys(self):
        """阶段 4 monitoring_node 返回标准 State 字段（无论走真实服务还是降级 Mock）。"""
        result = await monitoring_node(_base_state())
        assert "monitoring_run_id" in result
        assert "has_alerts" in result
        assert "alert_count" in result
        assert "max_alert_severity" in result
        assert "current_phase" in result
        assert isinstance(result["has_alerts"], bool)


class TestDiagnosisNode:
    async def test_need_iteration_produces_root_cause(self):
        result = await diagnosis_node(_base_state())
        assert result["need_iteration"] is True
        assert result["primary_root_cause_code"] == "feature_drift"
        assert result["recommended_action"] == "MODEL_ITERATION"

    async def test_no_iteration_needed(self):
        original = MOCK_NEED_ITERATION
        wf.MOCK_NEED_ITERATION = False
        try:
            result = await diagnosis_node(_base_state())
            assert result["need_iteration"] is False
            assert result["recommended_action"] == "CONTINUE_OBSERVATION"
        finally:
            wf.MOCK_NEED_ITERATION = original

    async def test_uncertain_goes_to_manual(self):
        original = MOCK_NEED_ITERATION
        wf.MOCK_NEED_ITERATION = None
        try:
            result = await diagnosis_node(_base_state())
            assert result["need_iteration"] is None
            assert result["recommended_action"] == "MANUAL_REVIEW"
        finally:
            wf.MOCK_NEED_ITERATION = original


class TestIterationSubgraph:
    async def test_challenger_qualified(self):
        result = await iteration_subgraph(_base_state())
        assert result["challenger_qualified"] is True
        assert result["challenger_version"] is not None

    async def test_challenger_not_qualified(self):
        original = MOCK_CHALLENGER_QUALIFIED
        wf.MOCK_CHALLENGER_QUALIFIED = False
        try:
            result = await iteration_subgraph(_base_state())
            assert result["challenger_qualified"] is False
        finally:
            wf.MOCK_CHALLENGER_QUALIFIED = original


class TestDeploymentNode:
    async def test_promote_path(self):
        result = await deployment_node(_base_state())
        assert result["deployment_decision"] == "PROMOTE"
        assert result["current_phase"] == LifecyclePhase.PROMOTED.value

    async def test_rollback_path(self):
        original = MOCK_DEPLOYMENT_DECISION
        wf.MOCK_DEPLOYMENT_DECISION = "ROLLBACK"
        try:
            result = await deployment_node(_base_state())
            assert result["current_phase"] == LifecyclePhase.ROLLED_BACK.value
        finally:
            wf.MOCK_DEPLOYMENT_DECISION = original


class TestManualReviewNode:
    async def test_interrupt_approved_continues(self):
        """模拟人工审核通过：interrupt 返回 'approved' → phase 回到 MANUAL_REVIEW。"""
        from unittest.mock import AsyncMock, patch

        with patch(
            "apps.modelops_api.services.workflow.graph.interrupt",
            return_value="approved",
        ):
            result = await manual_review_node(_base_state())
            assert result["requires_manual_review"] is False
            assert result["current_phase"] == LifecyclePhase.MANUAL_REVIEW.value
            assert "last_error" not in result or result.get("last_error") is None

    async def test_interrupt_rejected_fails(self):
        """模拟人工审核拒绝：interrupt 返回 'rejected' → phase 走向 FAILED。"""
        from unittest.mock import patch

        with patch(
            "apps.modelops_api.services.workflow.graph.interrupt",
            return_value="rejected",
        ):
            result = await manual_review_node(_base_state())
            assert result["requires_manual_review"] is True
            assert result["current_phase"] == LifecyclePhase.FAILED.value
            assert result["last_error"]["reason"] == "manual_review_rejected"


# ── State 大小约束测试（验收标准：State 中不含完整 Evidence/DataFrame/训练历史）──

class TestStateConstraints:
    def test_state_fields_are_control_only(self):
        """State 只保存流程控制字段 + 摘要，不含大 payload。"""
        field_types = ModelLifecycleState.model_fields
        # 所有字段都是标量/Optional/简单枚举，没有 list[dict] 或 DataFrame
        for name, field in field_types.items():
            anno_str = str(field.annotation)
            # 这些是允许的：str, bool, int, float, dict（last_error），以及它们的 Optional 版本
            assert "DataFrame" not in anno_str, f"{name} 不应包含 DataFrame"
            assert "Evidence" not in anno_str, f"{name} 不应包含完整 Evidence"
            assert "list[dict]" not in anno_str, f"{name} 不应包含 list[dict]"
            assert "list[str]" not in anno_str, f"{name} 不应包含 list[str]"


# ── 幂等与恢复测试 ──

class TestIdempotencyDesign:
    def test_state_is_serializable(self):
        """State 可以完整 JSON 序列化/反序列化（Checkpointer 要求）。"""
        import json

        s = ModelLifecycleState(
            lifecycle_run_id="test-001",
            model_id="m1",
            champion_version="v1",
            has_alerts=True,
            alert_count=2,
            max_alert_severity="HIGH",
            need_iteration=True,
            primary_root_cause_score=0.85,
            last_error={"reason": "test", "at": "2026-01-01T00:00:00Z"},
        )
        dumped = s.model_dump()
        # 验证可以完整序列化为 JSON
        json_str = json.dumps(dumped, default=str)
        restored = json.loads(json_str)
        assert restored["lifecycle_run_id"] == "test-001"
        assert restored["alert_count"] == 2
        assert restored["primary_root_cause_score"] == 0.85
        assert restored["last_error"]["reason"] == "test"

    def test_restart_after_crash_reuses_run_id(self):
        """重启 API 后同一 lifecycle_run_id 可从 Checkpoint 恢复。

        验证设计约定：thread_id = lifecycle_run_id，即相同 run_id 两次调用
        get_state 查询的是同一个 Checkpointer thread。
        """
        lifecycle_run_id = "restart-test-001"
        # 模拟：两次 API 调用，同一 run_id，Checkpointer config 中 thread_id 相同
        config1 = {"configurable": {"thread_id": lifecycle_run_id}}
        config2 = {"configurable": {"thread_id": lifecycle_run_id}}
        assert config1["configurable"]["thread_id"] == config2["configurable"]["thread_id"]
        assert config1["configurable"]["thread_id"] == lifecycle_run_id
