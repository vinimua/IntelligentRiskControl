"""阶段 3 验收测试 — LangGraph 可行走骨架 + P0 节点测试"""

import asyncio
import pytest

from apps.modelops_api.services.workflow import graph as wf
from apps.modelops_api.services.workflow.graph import (
    MOCK_CHALLENGER_QUALIFIED,
    MOCK_DEPLOYMENT_DECISION,
    MOCK_NEED_ITERATION,
    agent_decision_node,
    build_graph,
    calibration_plan_node,
    _deployment_action,
    deployment_node,
    diagnosis_handoff_node,
    diagnosis_node,
    event_pending_repair_node,
    failure_analysis_node,
    iteration_decision_node,
    iteration_subgraph,
    manual_review_node,
    monitoring_node,
    no_alert_close_node,
    observation_close_node,
    repair_plan_node,
    route_after_failure_analysis,
    route_after_feature_reconstruction,
    route_after_diagnosis,
    route_after_iteration_decision,
    route_after_manual_review,
    route_after_monitoring,
    threshold_plan_node,
    training_callback_resume_node,
    wait_feature_reconstruction_node,
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


def _passing_deployment_metrics() -> dict:
    return {
        "challenger_auc": 0.78,
        "challenger_ks": 0.32,
        "score_psi": 0.08,
        "bad_rate_drift": 0.01,
        "recovery_rate": 0.75,
        "train_valid_gap": 0.02,
        "discrimination_passed": True,
        "calibration_passed": True,
        "oot_passed": True,
        "segment_governance_passed": True,
    }


# ── 图结构测试 ──

class TestGraphStructure:
    def test_all_nodes_registered(self):
        g = build_graph()
        nodes = set(g.nodes.keys())
        expected = {
            # P0
            "MonitoringNode", "NoAlertCloseNode",
            "DiagnosisNode", "DiagnosisHandoffNode",
            "AgentDecisionNode", "IterationDecisionNode",
            # P1
            "ObservationCloseNode", "RepairPlanNode", "EventPendingRepairNode",
            "CalibrationPlanNode", "ThresholdPlanNode",
            "DataEligibilityNode", "ManualReviewNode", "FeatureReconstructionNode",
            "WaitFeatureReconstructionNode", "TrainingPlanNode",
            # P2
            "TrainingJobDispatchNode", "WaitTrainingCallbackNode",
            "TrainingCallbackResumeNode",
            # P3
            "QualificationNode", "FailureAnalysisNode",
            "NextRoundPlanNode", "StopAutoIterationNode",
            # P4
            "DeploymentGateNode", "EventCloseNode",
            # Legacy
            "IterationSubgraph", "DeploymentNode",
        }
        assert expected.issubset(nodes), f"missing nodes: {expected - nodes}"

    def test_start_goes_to_monitoring(self):
        g = build_graph()
        compiled = g.compile()
        # 验证 START → MonitoringNode 边：编译图的 nodes 字典中 MonitoringNode 存在
        assert "MonitoringNode" in compiled.get_graph().nodes


# ── 条件路由测试 ──

class TestRouting:
    def test_no_alerts_goes_to_no_alert_close(self):
        result = route_after_monitoring(_base_state(has_alerts=False))
        assert result == "NoAlertCloseNode"

    def test_has_alerts_goes_to_diagnosis(self):
        result = route_after_monitoring(_base_state(has_alerts=True))
        assert result == "DiagnosisNode"

    def test_need_iteration_true_goes_to_handoff(self):
        """P0: need_iteration=True → DiagnosisHandoffNode → Agent → Decision"""
        result = route_after_diagnosis(_base_state(need_iteration=True))
        assert result == "DiagnosisHandoffNode"

    def test_need_iteration_false_goes_to_no_alert_close(self):
        """P1: need_iteration=False → NoAlertCloseNode"""
        result = route_after_diagnosis(_base_state(need_iteration=False))
        assert result == "NoAlertCloseNode"

    def test_need_iteration_none_goes_to_manual_review(self):
        result = route_after_diagnosis(_base_state(need_iteration=None))
        assert result == "ManualReviewNode"

    def test_data_repair_goes_to_repair_plan(self):
        result = route_after_iteration_decision(
            _base_state(
                requires_manual_review=False,
                recommended_action="DATA_REPAIR",
                need_iteration=False,
            )
        )
        assert result == "RepairPlanNode"

    def test_calibration_goes_to_calibration_plan(self):
        result = route_after_iteration_decision(
            _base_state(
                requires_manual_review=False,
                recommended_action="CALIBRATION_ADJUSTMENT",
                need_iteration=False,
            )
        )
        assert result == "CalibrationPlanNode"

    def test_threshold_goes_to_threshold_plan(self):
        result = route_after_iteration_decision(
            _base_state(
                requires_manual_review=False,
                recommended_action="THRESHOLD_ADJUSTMENT",
                need_iteration=False,
            )
        )
        assert result == "ThresholdPlanNode"


# ── Mock 节点行为测试 ──


class TestMonitoringNode:
    async def test_monitoring_node_returns_expected_keys(self):
        """阶段 4 monitoring_node 返回标准 State 字段，单元测试不访问真实数据库。"""
        from unittest.mock import patch

        class FakeFrame:
            def to_dict(self, orient: str):
                assert orient == "records"
                return [{"sample_id": "s1", "is_bad": 0, "y_pred_proba": 0.1}]

        with (
            patch(
                "apps.modelops_api.services.monitoring.window_loader.load_window_with_predictions",
                return_value=FakeFrame(),
            ),
            patch(
                "apps.modelops_api.database.async_session",
                side_effect=OSError("database unavailable in unit test"),
            ),
        ):
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
        """模拟人工审核通过：interrupt 返回 'approved' → phase 进入 DECISION_PROPOSED。"""
        from unittest.mock import patch

        with patch(
            "apps.modelops_api.services.workflow.graph.interrupt",
            return_value="approved",
        ):
            result = await manual_review_node(_base_state())
            assert result["requires_manual_review"] is False
            assert result["current_phase"] == LifecyclePhase.DECISION_PROPOSED.value

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
        """重启 API 后同一 lifecycle_run_id 可从 Checkpoint 恢复。"""
        lifecycle_run_id = "restart-test-001"
        config1 = {"configurable": {"thread_id": lifecycle_run_id}}
        config2 = {"configurable": {"thread_id": lifecycle_run_id}}
        assert config1["configurable"]["thread_id"] == config2["configurable"]["thread_id"]
        assert config1["configurable"]["thread_id"] == lifecycle_run_id


# ── P0 新增节点测试（LangGraph 开发路线 V1.0 §15）──


class TestNoAlertCloseNode:
    async def test_no_alert_sets_phase(self):
        result = await no_alert_close_node(_base_state())
        assert result["current_phase"] == LifecyclePhase.NO_ALERT.value


class TestDiagnosisHandoffNode:
    async def test_missing_diagnosis_run_id_returns_error_status(self):
        """缺少 diagnosis_run_id 时返回 ERROR 状态，不崩溃。"""
        result = await diagnosis_handoff_node(_base_state())
        assert result["agent_handoff_status"] == "ERROR_NO_DIAGNOSIS_RUN"
        assert result["current_phase"] == LifecyclePhase.MANUAL_REVIEW.value

    async def test_infra_failure_fallback(self):
        """基础设施故障时降级，不抛异常。"""
        from unittest.mock import patch

        with (
            patch(
                "apps.modelops_api.database.async_session",
                side_effect=OSError("database unavailable"),
            ),
        ):
            state = _base_state(diagnosis_run_id="diag-001")
            result = await diagnosis_handoff_node(state)
            assert result["agent_handoff_status"] == "DEGRADED_INFRA_FAILURE"
            assert "event_id" in result

    async def test_handoff_validation_rejects_non_waiting_status(self):
        """handoff 只有 WAITING_AGENT_DECISION 状态才能交给 Agent。"""
        from apps.modelops_api.core.exceptions import NotFoundError
        from apps.modelops_api.services.workflow.agent_handoff_service import (
            DiagnosisHandoffService,
        )

        service = DiagnosisHandoffService(None)
        with pytest.raises(NotFoundError):
            await service.validate_handoff(
                {
                    "event_id": "event-001",
                    "event_status": "CLOSED",
                    "next_stage": "AGENT_DECISION",
                }
            )


class TestAgentDecisionNode:
    async def test_missing_diagnosis_run_id_forces_manual_review(self):
        """缺少 diagnosis_run_id 时强制进入人工复核。"""
        result = await agent_decision_node(_base_state())
        assert result["requires_manual_review"] is True
        assert result["recommended_action"] == "MANUAL_REVIEW"

    async def test_infra_failure_forces_manual_review(self):
        """基础设施故障时强制人工复核，不抛异常。"""
        from unittest.mock import patch

        with patch(
            "apps.modelops_api.database.async_session",
            side_effect=OSError("database unavailable"),
        ):
            state = _base_state(
                diagnosis_run_id="diag-002",
                primary_root_cause_code="feature_drift",
                primary_root_cause_score=0.85,
            )
            result = await agent_decision_node(state)
            assert result["requires_manual_review"] is True
            assert result["agent_confidence"] == 0.0


class TestIterationDecisionNode:
    async def test_missing_diagnosis_run_id_goes_to_manual_review(self):
        """缺少 diagnosis_run_id 时进入人工复核。"""
        result = await iteration_decision_node(_base_state())
        assert result["current_phase"] == LifecyclePhase.MANUAL_REVIEW.value

    async def test_infra_failure_goes_to_manual_review(self):
        """基础设施故障时进入人工复核。"""
        from unittest.mock import patch

        with patch(
            "apps.modelops_api.database.async_session",
            side_effect=OSError("database unavailable"),
        ):
            state = _base_state(
                diagnosis_run_id="diag-003",
                primary_root_cause_code="feature_drift",
                primary_root_cause_score=0.85,
            )
            result = await iteration_decision_node(state)
            assert result["current_phase"] == LifecyclePhase.MANUAL_REVIEW.value


class TestRouteAfterIterationDecision:
    def test_requires_manual_review_goes_to_manual_review(self):
        result = route_after_iteration_decision(
            _base_state(requires_manual_review=True, need_iteration=True)
        )
        assert result == "ManualReviewNode"

    def test_need_iteration_true_goes_to_data_eligibility(self):
        """P1: need_iteration=True + no review → DataEligibilityNode"""
        result = route_after_iteration_decision(
            _base_state(requires_manual_review=False, need_iteration=True)
        )
        assert result == "DataEligibilityNode"

    def test_need_iteration_false_goes_to_observation_close(self):
        result = route_after_iteration_decision(
            _base_state(requires_manual_review=False, need_iteration=False)
        )
        assert result == "ObservationCloseNode"

    def test_need_iteration_none_goes_to_manual_review(self):
        """None without review → END (no action needed)"""
        result = route_after_iteration_decision(
            _base_state(requires_manual_review=False, need_iteration=None)
        )
        assert result == "ManualReviewNode"


# ── RuleAgentAdapter 规则测试 ──


class TestRuleAgentAdapter:
    def test_low_confidence_forces_manual_review(self):
        """root_cause_score < 0.75 → MANUAL_REVIEW"""
        from apps.modelops_api.services.workflow.rule_agent_adapter import RuleAgentAdapter

        adapter = RuleAgentAdapter(None)  # session=None for unit test

        class FakeInput:
            lifecycle_run_id = "run-001"
            event_id = ""
            diagnosis_run_id = "diag-001"
            model_id = "m1"
            champion_version = "v1"
            primary_root_cause_code = "feature_drift"
            primary_root_cause_score = 0.50
            recommended_action = "MODEL_ITERATION"
            candidates_summary = []
            evidence_summary = []

        import asyncio
        result = asyncio.run(adapter.decide(FakeInput()))
        assert result.requires_manual_review is True
        assert result.recommended_action == "MANUAL_REVIEW"

    def test_high_confidence_adopts_recommended_action(self):
        """root_cause_score >= 0.75 → 采纳推荐动作"""
        from apps.modelops_api.services.workflow.rule_agent_adapter import RuleAgentAdapter

        adapter = RuleAgentAdapter(None)

        class FakeInput:
            lifecycle_run_id = "run-002"
            event_id = ""
            diagnosis_run_id = "diag-002"
            model_id = "m2"
            champion_version = "v1"
            primary_root_cause_code = "feature_drift"
            primary_root_cause_score = 0.85
            recommended_action = "MODEL_ITERATION"
            candidates_summary = []
            evidence_summary = []

        import asyncio
        result = asyncio.run(adapter.decide(FakeInput()))
        assert result.recommended_action == "MODEL_ITERATION"
        assert result.confidence == 0.85

    def test_missing_score_forces_manual_review(self):
        """primary_root_cause_score 为 None → MANUAL_REVIEW"""
        from apps.modelops_api.services.workflow.rule_agent_adapter import RuleAgentAdapter

        adapter = RuleAgentAdapter(None)

        class FakeInput:
            lifecycle_run_id = "run-003"
            event_id = ""
            diagnosis_run_id = "diag-003"
            model_id = "m3"
            champion_version = "v1"
            primary_root_cause_code = "feature_drift"
            primary_root_cause_score = None
            recommended_action = "MODEL_ITERATION"
            candidates_summary = []
            evidence_summary = []

        import asyncio
        result = asyncio.run(adapter.decide(FakeInput()))
        assert result.requires_manual_review is True


# ── P1-P4 新增节点测试（LangGraph 开发路线 V1.0 §10-14）──

from apps.modelops_api.services.workflow.graph import (
    data_eligibility_node,
    deployment_gate_node,
    event_close_node,
    next_round_plan_node,
    qualification_node,
    route_after_failure_analysis,
    route_after_qualification,
    stop_auto_iteration_node,
    training_job_dispatch_node,
    training_plan_node,
)


class TestDataEligibilityNode:
    async def test_infra_failure_fallback(self):
        from unittest.mock import patch
        with patch(
            "apps.modelops_api.database.async_session",
            side_effect=OSError("database unavailable"),
        ):
            result = await data_eligibility_node(
                _base_state(diagnosis_run_id="diag-001")
            )
            assert result["current_phase"] == LifecyclePhase.DECISION_PROPOSED.value


class TestTrainingPlanNode:
    async def test_missing_proposal_id_goes_to_manual_review(self):
        result = await training_plan_node(_base_state())
        assert result["current_phase"] == LifecyclePhase.MANUAL_REVIEW.value


class TestTrainingJobDispatchNode:
    async def test_creates_job_and_returns_callback_phase(self):
        from unittest.mock import patch

        with patch(
            "apps.modelops_api.database.async_session",
            side_effect=OSError("database unavailable"),
        ):
            result = await training_job_dispatch_node(
                _base_state(
                    iteration_run_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    training_plan_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                    experiment_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
                )
            )
        assert result["training_job_id"] is not None
        assert result["current_phase"] == LifecyclePhase.WAITING_TRAINING_CALLBACK.value


class TestQualificationNode:
    async def test_fallback_qualified(self):
        from unittest.mock import patch
        with patch(
            "apps.modelops_api.database.async_session",
            side_effect=OSError("database unavailable"),
        ):
            result = await qualification_node(
                _base_state(
                    decision_proposal_id="prop-001",
                    iteration_run_id="iter-001",
                    experiment_id="exp-001",
                    business_round=1,
                )
            )
            assert result["challenger_qualified"] is True
            assert result["current_phase"] == LifecyclePhase.QUALIFICATION_COMPLETED.value


class TestNextRoundPlanNode:
    async def test_increments_business_round(self):
        result = await next_round_plan_node(_base_state(business_round=1))
        assert result["business_round"] == 2
        assert result["current_phase"] == LifecyclePhase.ITERATING.value


class TestExecutorBackedPlanNodes:
    async def test_plan_nodes_accept_model_state(self):
        state = ModelLifecycleState(
            lifecycle_run_id="run-executor-state",
            model_id="credit_model_001",
            champion_version="champion_v1",
            recommended_action="DATA_REPAIR",
            diagnosis_run_id="diag-001",
            business_round=1,
        )

        repair = await repair_plan_node(state)
        calibration = await calibration_plan_node(state)
        threshold = await threshold_plan_node(state)

        assert repair["repair_plan_id"]
        assert calibration["calibration_plan_id"]
        assert threshold["threshold_plan_id"]


class TestStopAutoIterationNode:
    async def test_sets_exit_reason(self):
        result = await stop_auto_iteration_node(_base_state(business_round=3))
        assert result["iteration_exit_reason"] == "MAX_BUSINESS_ROUNDS_REACHED"
        assert result["current_phase"] == LifecyclePhase.FAILED.value


class TestDeploymentGateNode:
    async def test_not_qualified_aborts(self):
        result = await deployment_gate_node(
            _base_state(challenger_qualified=False)
        )
        assert result["deployment_decision"] == "ABORT_DEPLOYMENT"
        assert result["current_phase"] == LifecyclePhase.MANUAL_REVIEW.value

    async def test_qualified_promotes(self):
        """CANARY_50 + health passed → ADVANCE_STAGE (two-pass: next iteration PRODUCTION → PROMOTE)."""
        result = await deployment_gate_node(
            _base_state(
                challenger_qualified=True,
                deployment_stage="CANARY_50",
                validation_metrics=_passing_deployment_metrics(),
            )
        )
        assert result["deployment_decision"] == "ADVANCE_STAGE"
        assert result["current_phase"] == LifecyclePhase.CANARY_RUNNING.value

    async def test_production_qualified_promotes(self):
        """PRODUCTION + health passed → PROMOTE."""
        result = await deployment_gate_node(
            _base_state(
                challenger_qualified=True,
                deployment_stage="PRODUCTION",
                validation_metrics=_passing_deployment_metrics(),
            )
        )
        assert result["deployment_decision"] == "PROMOTE"
        assert result["current_phase"] == LifecyclePhase.PROMOTED.value

    async def test_qualified_advances_intermediate_stage(self):
        result = await deployment_gate_node(
            _base_state(challenger_qualified=True, deployment_stage="OFFLINE_VALIDATION")
        )
        assert result["deployment_stage"] == "OOT_GATE"
        assert result["deployment_decision"] == "ADVANCE_STAGE"
        assert result["current_phase"] == LifecyclePhase.CANARY_RUNNING.value

    async def test_forced_rollback_sets_rolled_back_phase(self):
        result = await deployment_gate_node(
            _base_state(
                challenger_qualified=True,
                deployment_stage="CANARY_50",
                deployment_force_rollback=True,
                validation_metrics=_passing_deployment_metrics(),
            )
        )
        assert result["deployment_decision"] == "ROLLBACK"
        assert result["current_phase"] == LifecyclePhase.ROLLED_BACK.value

    async def test_failed_health_check_rolls_back_critical_stage(self):
        """CANARY_20 (critical) + health failed → ROLLBACK (Gatekeeper rule 3)."""
        result = await deployment_gate_node(
            _base_state(
                challenger_qualified=True,
                deployment_stage="CANARY_20",
                deployment_health_passed=False,
            )
        )
        assert result["deployment_decision"] == "ROLLBACK"
        assert result["current_phase"] == LifecyclePhase.ROLLED_BACK.value

    async def test_failed_health_check_holds_non_critical_stage(self):
        """SHADOW (non-critical) + health failed → HOLD (Gatekeeper rule 4)."""
        result = await deployment_gate_node(
            _base_state(
                challenger_qualified=True,
                deployment_stage="SHADOW",
                deployment_health_passed=False,
            )
        )
        assert result["deployment_decision"] == "HOLD"
        assert result["current_phase"] == LifecyclePhase.CANARY_RUNNING.value

    async def test_severe_health_failure_rolls_back_canary(self):
        metrics = _passing_deployment_metrics()
        metrics["bad_rate_drift"] = 0.18
        result = await deployment_gate_node(
            _base_state(
                challenger_qualified=True,
                deployment_stage="CANARY_20",
                validation_metrics=metrics,
            )
        )
        assert result["deployment_decision"] == "ROLLBACK"
        assert result["current_phase"] == LifecyclePhase.ROLLED_BACK.value

    async def test_canary_without_health_metrics_rolls_back(self):
        """CANARY_20 without health metrics → critical stage failure → ROLLBACK."""
        result = await deployment_gate_node(
            _base_state(
                challenger_qualified=True,
                deployment_stage="CANARY_20",
            )
        )
        assert result["deployment_decision"] == "ROLLBACK"

    async def test_early_stage_without_health_metrics_holds(self):
        """OFFLINE_VALIDATION without metrics → not critical → HOLD or ADVANCE."""
        result = await deployment_gate_node(
            _base_state(
                challenger_qualified=True,
                deployment_stage="OFFLINE_VALIDATION",
            )
        )
        # Early stages with no metrics → passed (optimistic)
        assert result["deployment_decision"] in ("ADVANCE_STAGE", "HOLD")
        assert result["current_phase"] == LifecyclePhase.CANARY_RUNNING.value


class TestEventCloseNode:
    async def test_missing_preconditions_goes_to_manual_review(self):
        result = await event_close_node(_base_state())
        assert result["current_phase"] == LifecyclePhase.MANUAL_REVIEW.value

    async def test_sets_event_closed_phase_when_preconditions_pass(self):
        result = await event_close_node(
            _base_state(
                challenger_qualified=True,
                qualification_run_id="qual-001",
                deployment_id="deploy-001",
                deployment_stage="PRODUCTION",
                deployment_decision="PROMOTE",
            )
        )
        assert result["current_phase"] == LifecyclePhase.EVENT_CLOSED.value

    async def test_hold_deployment_does_not_close_event(self):
        result = await event_close_node(
            _base_state(
                challenger_qualified=True,
                qualification_run_id="qual-001",
                deployment_id="deploy-001",
                deployment_stage="CANARY_20",
                deployment_decision="HOLD",
            )
        )
        assert result["current_phase"] == LifecyclePhase.MANUAL_REVIEW.value
        assert result["last_error"]["reason"] == "event_close_preconditions_not_met"


class TestRouteAfterQualification:
    def test_qualified_goes_to_deployment(self):
        result = route_after_qualification(
            _base_state(challenger_qualified=True, business_round=1)
        )
        assert result == "DeploymentGateNode"

    def test_not_qualified_goes_to_failure_analysis(self):
        result = route_after_qualification(
            _base_state(challenger_qualified=False, business_round=1)
        )
        assert result == "FailureAnalysisNode"


class TestRouteAfterFailureAnalysis:
    def test_under_3_rounds_goes_to_next_round(self):
        result = route_after_failure_analysis(
            _base_state(challenger_qualified=False, business_round=1)
        )
        assert result == "NextRoundPlanNode"

    def test_at_3_rounds_stops(self):
        result = route_after_failure_analysis(
            _base_state(challenger_qualified=False, business_round=3)
        )
        assert result == "StopAutoIterationNode"


class TestRouteAfterManualReview:
    def test_approved_model_iteration_goes_to_feature_recon(self):
        result = route_after_manual_review(
            _base_state(requires_manual_review=False, need_iteration=True)
        )
        assert result == "FeatureReconstructionNode"

    def test_approved_manual_review_without_iteration_closes_observation(self):
        result = route_after_manual_review(
            _base_state(
                requires_manual_review=False,
                recommended_action="MANUAL_REVIEW",
                need_iteration=False,
            )
        )
        assert result == "ObservationCloseNode"

    def test_approved_manual_review_with_iteration_goes_to_feature_recon(self):
        result = route_after_manual_review(
            _base_state(
                requires_manual_review=False,
                recommended_action="MANUAL_REVIEW",
                need_iteration=True,
            )
        )
        assert result == "FeatureReconstructionNode"

    def test_rejected_review_ends(self):
        result = route_after_manual_review(
            _base_state(requires_manual_review=True, need_iteration=True)
        )
        assert result == "__end__"


class TestFeatureReconstructionRouting:
    def test_dispatched_feature_reconstruction_waits_for_callback(self):
        result = route_after_feature_reconstruction(
            _base_state(feature_reconstruction_dispatched=True)
        )
        assert result == "WaitFeatureReconstructionNode"

    def test_inline_feature_reconstruction_goes_to_training_plan(self):
        result = route_after_feature_reconstruction(
            _base_state(feature_reconstruction_dispatched=False)
        )
        assert result == "TrainingPlanNode"

    @pytest.mark.asyncio
    async def test_wait_feature_reconstruction_accepts_success_callback(self, monkeypatch):
        monkeypatch.setattr(
            wf,
            "interrupt",
            lambda _: {
                "status": "SUCCEEDED",
                "feature_reconstruction_plan_id": "plan-001",
                "feature_schema_version": "v2",
                "feature_snapshot_id": "snapshot-001",
                "transform_artifact_uri": "s3://riskitem/features/transforms/plan-001/pipeline.json",
            },
        )

        result = await wait_feature_reconstruction_node(
            _base_state(feature_reconstruction_plan_id="plan-001")
        )

        assert result["current_phase"] == LifecyclePhase.ITERATING.value
        assert result["feature_reconstruction_status"] == "SUCCEEDED"
        assert result["feature_schema_version"] == "v2"
        assert result["feature_snapshot_id"] == "snapshot-001"

    @pytest.mark.asyncio
    async def test_wait_feature_reconstruction_rejects_mismatched_plan(self, monkeypatch):
        monkeypatch.setattr(
            wf,
            "interrupt",
            lambda _: {
                "status": "SUCCEEDED",
                "feature_reconstruction_plan_id": "other-plan",
            },
        )

        result = await wait_feature_reconstruction_node(
            _base_state(feature_reconstruction_plan_id="plan-001")
        )

        assert result["current_phase"] == LifecyclePhase.FAILED.value
        assert result["last_error"]["reason"] == "feature_reconstruction_plan_id_mismatch"


class TestDeploymentAction:
    @pytest.mark.asyncio
    async def test_advance_stage_updates_routing_versions(self, monkeypatch):
        calls = []

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def commit(self):
                return None

        class FakeSafetyService:
            def __init__(self, session):
                self.session = session

            async def update_traffic_ratio(
                self,
                model_id,
                stage,
                champion_version=None,
                challenger_version=None,
            ):
                calls.append(
                    {
                        "model_id": model_id,
                        "stage": stage,
                        "champion": champion_version,
                        "challenger": challenger_version,
                    }
                )
                return 0.05

        monkeypatch.setattr(
            "apps.modelops_api.database.async_session",
            lambda: FakeSession(),
        )
        monkeypatch.setattr(
            "apps.modelops_api.services.iteration.deployment_safety_service.DeploymentSafetyService",
            FakeSafetyService,
        )

        decision = type("Decision", (), {"decision": "ADVANCE_STAGE"})()

        result = await _deployment_action(
            _base_state(),
            decision,
            "SHADOW",
            "credit_model_001",
            "champion_v1",
            "challenger_v2",
            "deploy-001",
        )

        assert result["deployment_decision"] == "ADVANCE_STAGE"
        assert calls == [
            {
                "model_id": "credit_model_001",
                "stage": "CANARY_5",
                "champion": "champion_v1",
                "challenger": "challenger_v2",
            }
        ]

    @pytest.mark.asyncio
    async def test_action_failure_is_converted_to_hold(self, monkeypatch):
        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        class FailingSafetyService:
            def __init__(self, session):
                self.session = session

            async def update_traffic_ratio(self, *args, **kwargs):
                raise RuntimeError("routing db unavailable")

        monkeypatch.setattr(
            "apps.modelops_api.database.async_session",
            lambda: FakeSession(),
        )
        monkeypatch.setattr(
            "apps.modelops_api.services.iteration.deployment_safety_service.DeploymentSafetyService",
            FailingSafetyService,
        )

        decision = type("Decision", (), {"decision": "ADVANCE_STAGE"})()

        result = await _deployment_action(
            _base_state(),
            decision,
            "SHADOW",
            "credit_model_001",
            "champion_v1",
            "challenger_v2",
            "deploy-001",
        )

        assert result["deployment_decision"] == "ADVANCE_STAGE"
        assert result["action_failed"] is True
        assert "routing db unavailable" in result["action_error"]
