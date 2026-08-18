"""A7 定稿工作流级路由与证据生产者专项测试。"""

import pytest

from apps.modelops_api.services.diagnosis.diagnosis_service import (
    _infer_drift_context,
)


def _fake_state(**overrides) -> dict:
    base = {
        "lifecycle_run_id": "11111111-1111-1111-1111-111111111111",
        "model_id": "credit_model_001",
        "champion_version": "champion_v1",
        "deployment_decision": None,
        "deployment_stage": None,
        "business_round": 1,
    }
    base.update(overrides)
    return base


def test_rollback_routes_to_deployment_outcome_node():
    """P0-1: ROLLBACK 不再直接 END，必须经过 DeploymentOutcomeNode 写观测。"""
    from apps.modelops_api.services.workflow.graph import (
        route_after_deployment_gate,
        route_after_deployment_outcome,
    )
    state = _fake_state(deployment_decision="ROLLBACK",
                        deployment_stage="CANARY_50")

    assert route_after_deployment_gate(state) == "DeploymentOutcomeNode"
    assert route_after_deployment_outcome(state) == "__end__"


def test_promote_routes_outcome_then_event_close():
    from apps.modelops_api.services.workflow.graph import (
        route_after_deployment_gate,
        route_after_deployment_outcome,
    )
    state = _fake_state(deployment_decision="PROMOTE",
                        deployment_stage="PRODUCTION")

    assert route_after_deployment_gate(state) == "DeploymentOutcomeNode"
    assert route_after_deployment_outcome(state) == "EventCloseNode"


def test_graph_edges_round2_reruns_decision():
    """P0-2: NextRoundPlanNode → IterationDecisionNode（第二轮重新决策）。"""
    from apps.modelops_api.services.workflow.graph import build_graph

    graph = build_graph()
    edges = set(graph.edges)
    assert ("NextRoundPlanNode", "IterationDecisionNode") in edges
    # DeploymentGate → DeploymentOutcome 是条件边（P0-1），验证分支存在
    branches = graph.branches.get("DeploymentGateNode") or {}
    targets = {
        target for branch in branches.values() for target in (branch.ends or {}).values()
    }
    assert "DeploymentOutcomeNode" in targets


def test_infer_drift_context_scope_and_pattern():
    """P1-1: 真实漂移证据推导 scope/pattern。"""
    rows = [
        {"feature_name": "f1", "psi": 0.3, "window_id": "W1"},
        {"feature_name": "f2", "psi": 0.25, "window_id": "W1"},
        {"feature_name": "f1", "psi": 0.6, "window_id": "W2"},
        {"feature_name": "f2", "psi": 0.5, "window_id": "W2"},
    ]
    multi = {"W1": rows[:2], "W2": rows[2:]}

    scope, pattern = _infer_drift_context(rows, multi, "FEATURE_DRIFT")
    # 2/2 特征漂移 → GLOBAL；W2 均值 0.55 vs W1 均值 0.275 = 2x → SUDDEN
    assert scope == "GLOBAL"
    assert pattern == "SUDDEN"

    # 非漂移根因不推导
    assert _infer_drift_context(rows, multi, "DATA_QUALITY_ISSUE") == (None, None)

    # 无数据诚实返回 None
    assert _infer_drift_context([], None, "FEATURE_DRIFT") == (None, None)


def test_infer_drift_context_gradual_pattern():
    rows = [
        {"feature_name": "f1", "psi": 0.2, "window_id": "W1"},
        {"feature_name": "f1", "psi": 0.25, "window_id": "W2"},
    ]
    multi = {"W1": rows[:1], "W2": rows[1:]}
    scope, pattern = _infer_drift_context(rows, multi, "CONCEPT_DRIFT")
    # 1/1 特征漂移 → GLOBAL；0.25 < 0.2*1.5=0.3 → GRADUAL
    assert scope == "GLOBAL"
    assert pattern == "GRADUAL"


def test_oot_canary_rollback_chain_produces_natural_against():
    """P0-2 完整链路：OOT_GATE 完成 W4 → CANARY 阶段 ROLLBACK →
    观测层产出 NATURAL AGAINST（W4 证据不被后续阶段清空）。"""
    from packages.models.common.enums import DataTrack, EvidenceDirection
    from apps.modelops_api.services.knowledge_observation_service import (
        KnowledgeObservationService,
    )

    state = _fake_state(
        deployment_decision="ROLLBACK",
        deployment_stage="CANARY_50",
        deployment_id="deploy-1",
        challenger_qualified=True,
        qualification_run_id="qual-1",
        # OOT_GATE 阶段写入的 W4 证据（跨阶段保留）
        oot_validation_completed=True,
        oot_validation_run_id="oot-run-1",
        w4_available=True,
        candidate_frozen_before_oot=True,
        oot_passed=False,
        lifecycle_terminal=True,
        primary_root_cause_code="FEATURE_DRIFT",
        primary_root_cause_score=0.85,
        diagnosis_run_id="diag-1",
        selected_strategy_code="recent_weighted_retrain",
        alert_codes=["HIGH_FEATURE_PSI"],
    )

    observations = KnowledgeObservationService.build_observations(state)
    natural = [o for o in observations if o.data_track == DataTrack.NATURAL]
    assert len(natural) == 2  # 配对 RECOMMENDS + MITIGATES
    assert all(o.direction == EvidenceDirection.AGAINST for o in natural)
    assert {o.relation_key for o in natural} == {
        "FEATURE_DRIFT|RECOMMENDS|recent_weighted_retrain",
        "recent_weighted_retrain|MITIGATES|FEATURE_DRIFT",
    }


def test_deployment_subgraph_result_preserves_w4_evidence_from_state():
    """P0-2: Canary/Production 阶段通过 _deployment_subgraph_result 覆盖
    State 时必须保留已有 W4 证据。"""
    from apps.modelops_api.services.workflow.graph import (
        _deployment_subgraph_result,
    )

    class _Decision:
        decision = "ROLLBACK"
        decision_reasons = ["r1"]
        selected_strategy_code = "recent_weighted_retrain"

    result = _deployment_subgraph_result(
        "deploy-1",
        _Decision(),
        {"deployment_decision": "ROLLBACK", "deployment_stage": "CANARY_50"},
        oot_evidence={
            "completed": True,
            "lifecycle": "oot-run-1",
            "w4_available": True,
            "candidate_frozen_before_oot": True,
            "oot_passed": False,
        },
    )
    assert result["oot_validation_completed"] is True
    assert result["oot_validation_run_id"] == "oot-run-1"
    assert result["w4_available"] is True
    assert result["candidate_frozen_before_oot"] is True
    assert result["oot_passed"] is False
    assert result["lifecycle_terminal"] is True


def test_deployment_outcome_node_rollback_keeps_rolled_back():
    """P1-1: ROLLBACK 经 DeploymentOutcomeNode 后保留 ROLLED_BACK 终态。"""
    import asyncio
    from apps.modelops_api.services.workflow.graph import (
        deployment_outcome_node,
    )

    result = asyncio.run(deployment_outcome_node(_fake_state(
        deployment_decision="ROLLBACK",
        deployment_stage="CANARY_50",
    )))
    assert result["current_phase"] == "ROLLED_BACK"


def _qualification_report(reasons: list[str]) -> "QualificationReport":
    from packages.models.common.enums import QualificationGateCode, QualificationStatus
    from packages.models.iteration.qualification import (
        QualificationGateResult, QualificationReport,
    )
    return QualificationReport(
        qualification_run_id="qual-1",
        iteration_run_id="iter-1",
        experiment_id="exp-1",
        candidate_version="c1",
        status=QualificationStatus.FAILED,
        qualified=False,
        failed_gate_codes=[QualificationGateCode.STABILITY],
        gate_results=[
            QualificationGateResult(
                gate_code=QualificationGateCode.STABILITY,
                gate_order=0,
                status=QualificationStatus.FAILED,
                reasons=reasons,
            ),
        ],
        rule_version="qualification-rules-v1",
    )


def test_failure_attribution_extracts_unstable_features():
    """P1-2: 只有归因确认的特征才授予不稳定特征子集证据。"""
    from apps.modelops_api.services.iteration.failure_attribution import (
        _extract_unstable_features,
    )

    report = _qualification_report(
        ["feature=age PSI 0.42 over threshold", "score_psi too high"],
    )

    features = _extract_unstable_features(report)
    assert features == ["age"]


def test_failure_attribution_no_feature_evidence_returns_empty():
    """稳定性失败但原因里没有特征级证据 → 空列表，不授予证据码。"""
    from apps.modelops_api.services.iteration.failure_attribution import (
        _extract_unstable_features,
    )

    report = _qualification_report(["整体 Score PSI 0.32 超过 0.20 阈值"])

    assert _extract_unstable_features(report) == []


@pytest.mark.asyncio
async def test_oot_gate_runs_even_when_validation_metrics_present():
    """P0-1: 资格完成后 validation_metrics 非空，OOT_GATE 仍必须执行真实 W4。"""
    from unittest.mock import patch
    from apps.modelops_api.services.workflow.graph import _deployment_observe

    state = _fake_state(
        model_id="credit_model_001",
        challenger_version="challenger_v1",
        lifecycle_run_id="11111111-1111-1111-1111-111111111111",
    )
    called = {}

    def fake_frozen(model_id, lifecycle_run_id, candidate_version):
        return {"loaded": True, "model": object(), "feature_cols": ["f1"]}

    def fake_oot(model, feature_cols, **kwargs):
        called["run_oot_validation"] = True
        return {
            "oot_auc": 0.62, "oot_ks": 0.18, "oot_psi": 0.28,
            "w4_available": True, "oot_passed": False,
        }

    with (
        patch(
            "apps.modelops_api.services.deployment.deployment_oot_service.load_frozen_challenger",
            side_effect=fake_frozen,
        ),
        patch(
            "apps.modelops_api.services.deployment.deployment_oot_service.run_oot_validation",
            side_effect=fake_oot,
        ),
    ):
        health_result, alerts, oot_evidence = await _deployment_observe(
            state,
            stage="OOT_GATE",
            health_metrics={"challenger_auc": 0.72, "challenger_ks": 0.25},  # 非空
            lifecycle_run_id="11111111-1111-1111-1111-111111111111",
            deployment_id="deploy-1",
        )

    assert called.get("run_oot_validation") is True  # 真实 W4 被执行
    assert oot_evidence["completed"] is True
    assert oot_evidence["w4_available"] is True
    assert oot_evidence["oot_passed"] is False


def test_feature_psi_exceeded_fails_stability_gate():
    """P0: Score PSI 与 gap 均正常，但单个特征 PSI 超标 → STABILITY 必须失败。"""
    from apps.modelops_api.services.iteration.qualification_service import (
        QualificationService,
    )
    from apps.modelops_api.services.iteration.config_loader import (
        load_iteration_config,
    )
    from packages.models.iteration.qualification import QualificationInput

    qual_input = QualificationInput(
        qualification_run_id="qual-1",
        iteration_run_id="iter-1",
        experiment_id="exp-1",
        candidate_version="c1",
        data_reproducible=True,
        discrimination_passed=True,
        calibration_passed=True,
        score_psi=0.15,          # 正常
        train_valid_gap=0.01,    # 正常
        segment_governance_passed=True,
        oot_window_id="W4",
        candidate_frozen_before_oot=True,
        oot_passed=True,
        feature_psi={"age": 0.42},  # 单个特征超标
    )

    report = QualificationService(load_iteration_config()).evaluate(qual_input)

    stability = next(
        g for g in report.gate_results
        if g.gate_code.value == "STABILITY"
    )
    assert stability.status.value == "FAILED"
    assert stability.unstable_feature_codes == ["age"]
    assert any(r.startswith("FEATURE_PSI_EXCEEDED") for r in stability.reasons)
    assert report.qualified is False


def test_feature_psi_threshold_comes_from_config_not_request():
    """P1: 阈值来自 qualification.yaml 配置，请求模型根本不接受该字段。"""
    from apps.modelops_api.routers.iteration import QualificationRequest

    # 外部请求模型没有 feature_psi_threshold / feature_psi 字段
    assert "feature_psi_threshold" not in QualificationRequest.model_fields
    assert "feature_psi" not in QualificationRequest.model_fields

    from apps.modelops_api.services.iteration.config_loader import (
        load_iteration_config,
    )
    rule = load_iteration_config().qualification
    assert rule.feature_psi_threshold == 0.25


def test_proposal_carries_monitoring_run_id():
    """P0: DecisionInput.monitoring_run_id → DecisionProposal 持久化。"""
    from apps.modelops_api.services.iteration import RepairDecisionService
    from tests.unit.test_iteration_decision import _decision_input, _root
    from packages.models.iteration import MetricDegradation

    request = _decision_input(
        _root("feature_drift", evidence=["D", "I", "C", "T"]),
        [MetricDegradation(metric_code="AUC")],
    ).model_copy(update={"monitoring_run_id": "mon-1"})

    proposal = RepairDecisionService().decide(request)

    assert proposal.monitoring_run_id == "mon-1"


def test_qualification_request_is_identity_only():
    """P1: 外部资格请求只接收身份字段，其他门禁输入一律不接收。"""
    from apps.modelops_api.routers.iteration import QualificationRequest

    fields = set(QualificationRequest.model_fields)
    assert fields == {
        "qualification_run_id", "iteration_run_id", "experiment_id",
        "candidate_version", "data_track",
    }
    for forbidden in (
        "score_psi", "train_valid_gap", "data_reproducible",
        "discrimination_passed", "calibration_passed", "oot_passed",
        "segment_governance_passed", "feature_psi", "feature_psi_threshold",
    ):
        assert forbidden not in fields


def test_qualification_input_built_from_trusted_experiment_fails_stability():
    """接口级：Proposal 的 monitoring_run_id 加载到 age:0.42 →
    服务端构建 QualificationInput → STABILITY 失败。"""
    from apps.modelops_api.routers.iteration import (
        QualificationRequest,
        _qualification_input_from_experiment,
    )
    from apps.modelops_api.services.iteration.qualification_service import (
        QualificationService,
    )
    from apps.modelops_api.services.iteration.config_loader import (
        load_iteration_config,
    )

    body = QualificationRequest(
        qualification_run_id="qual-1",
        iteration_run_id="iter-1",
        experiment_id="exp-1",
        candidate_version="c1",
    )
    # W3 验证结果：Score PSI 与 gap 正常；OOT 写回正常；
    # 完整恢复/健康/Bootstrap 证据（其他门不误失败，只验证特征 PSI 门）
    experiment_json = {
        "validation_metrics": {
            "challenger_auc": 0.72, "champion_auc": 0.60,
            "recovery_auc": 0.7, "challenger_ks": 0.28, "champion_ks": 0.20,
            "recovery_ks": 0.65, "score_psi": 0.15, "train_valid_gap": 0.01,
            "discrimination_passed": True, "calibration_passed": True,
            "original_drop": 0.0, "recovered_amount": 0.12,
            "healthy_lower_bound": 0.72,
            "bootstrap_ci_lower": 0.05, "bootstrap_ci_upper": 0.20,
            "ks_healthy_lower_bound": 0.15,
            "ks_bootstrap_ci_lower": 0.01, "ks_bootstrap_ci_upper": 0.10,
        },
        "segment_metrics": {"segment_governance_passed": True},
        "data_reproducible": True,
        "candidate_frozen_before_oot": True,
        "oot_passed": True,
    }
    # 服务端从 proposal.monitoring_run_id 加载的特征漂移
    feature_psi = {"age": 0.42}

    qual_input = _qualification_input_from_experiment(
        body, experiment_json, feature_psi,
    )
    report = QualificationService(load_iteration_config()).evaluate(qual_input)

    stability = next(
        g for g in report.gate_results
        if g.gate_code.value == "STABILITY"
    )
    assert stability.status.value == "FAILED"
    assert stability.unstable_feature_codes == ["age"]
    assert report.qualified is False


def test_target_recovery_not_fixed_failed_when_evidence_complete():
    """P0: 恢复字段完整时（含 AUC/KS 健康区间与 Bootstrap），
    TARGET_RECOVERY 必须 PASSED，整体资格 PASSED。"""
    from apps.modelops_api.services.iteration.qualification_service import (
        QualificationService,
        build_qualification_input,
    )
    from apps.modelops_api.services.iteration.config_loader import (
        load_iteration_config,
    )

    experiment_json = {
        "validation_metrics": {
            "challenger_auc": 0.79, "champion_auc": 0.72,
            # 恢复率 ≥ 0.90（严格 A7 资格口径 qualification-rules-v2）
            "recovery_auc": 0.92, "challenger_ks": 0.36, "champion_ks": 0.30,
            "recovery_ks": 0.95, "score_psi": 0.12, "train_valid_gap": 0.01,
            "discrimination_passed": True, "calibration_passed": True,
            "original_drop": 0.0, "recovered_amount": 0.07,
            "healthy_lower_bound": 0.72,
            "healthy_upper_bound": None,
            "bootstrap_ci_lower": 0.03,
            "bootstrap_ci_upper": 0.11,
            # KS 专属证据（Worker 真实计算）
            "ks_healthy_lower_bound": 0.20,
            "ks_bootstrap_ci_lower": 0.02,
            "ks_bootstrap_ci_upper": 0.08,
        },
        "segment_metrics": {"segment_governance_passed": True},
        "data_reproducible": True,
        "candidate_frozen_before_oot": True,
        "frozen_identity_checksum": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "oot_passed": True,
    }

    qual_input = build_qualification_input(
        qualification_run_id="qual-1",
        iteration_run_id="iter-1",
        experiment_id="exp-1",
        candidate_version="c1",
        experiment_json=experiment_json,
        feature_psi={},
    )
    report = QualificationService(load_iteration_config()).evaluate(qual_input)

    target = next(
        g for g in report.gate_results
        if g.gate_code.value == "TARGET_RECOVERY"
    )
    # 验收断言：门状态 PASSED，且整体资格通过（AUC/KS 均不固定失败）
    assert target.status.value == "PASSED", target.reasons
    assert report.qualified is True


def test_qualification_evidence_incomplete_raises():
    """P1: 缺少 train_valid_gap 等必填证据 → 拒绝评估，禁止 fail-open。"""
    import pytest as _pytest
    from apps.modelops_api.services.iteration.qualification_service import (
        QualificationEvidenceIncompleteError,
        build_qualification_input,
    )

    experiment_json = {
        "validation_metrics": {
            "challenger_auc": 0.79, "challenger_ks": 0.36,
            "score_psi": 0.12,  # train_valid_gap 缺失
            "discrimination_passed": True, "calibration_passed": True,
        },
        "data_reproducible": True,
        "candidate_frozen_before_oot": True,
        "oot_passed": True,
    }

    with _pytest.raises(QualificationEvidenceIncompleteError) as exc_info:
        build_qualification_input(
            qualification_run_id="qual-1",
            iteration_run_id="iter-1",
            experiment_id="exp-1",
            candidate_version="c1",
            experiment_json=experiment_json,
        )
    assert "train_valid_gap" in exc_info.value.missing_fields


def test_oot_window_comes_from_config():
    """P1: oot_window_id 来自 qualification.yaml 配置，不硬编码 W4。"""
    from apps.modelops_api.services.iteration.qualification_service import (
        build_qualification_input,
    )
    from apps.modelops_api.services.iteration.config_loader import (
        load_iteration_config,
    )

    experiment_json = {
        "validation_metrics": {
            "challenger_auc": 0.79, "challenger_ks": 0.36,
            "recovery_auc": 0.8, "recovery_ks": 0.65,
            "score_psi": 0.12, "train_valid_gap": 0.01,
            "discrimination_passed": True, "calibration_passed": True,
            "healthy_lower_bound": 0.72,
            "bootstrap_ci_lower": 0.03, "bootstrap_ci_upper": 0.11,
            "ks_healthy_lower_bound": 0.20,
            "ks_bootstrap_ci_lower": 0.02, "ks_bootstrap_ci_upper": 0.08,
        },
        "segment_metrics": {"segment_governance_passed": True},
        "data_reproducible": True,
        "candidate_frozen_before_oot": True,
        "oot_passed": True,
    }

    qual_input = build_qualification_input(
        qualification_run_id="qual-1",
        iteration_run_id="iter-1",
        experiment_id="exp-1",
        candidate_version="c1",
        experiment_json=experiment_json,
    )
    rule = load_iteration_config().qualification
    assert qual_input.oot_window_id == rule.required_oot_window_id
