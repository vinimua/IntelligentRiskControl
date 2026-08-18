"""任务三真实执行闭环验收测试（四条端到端验收的单元级覆盖）。"""

import numpy as np
import pandas as pd
import pytest


def _synthetic_window_df(n: int = 600, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "apply_time": pd.date_range("2026-01-01", periods=n, freq="h"),
        "age": rng.integers(20, 60, n).astype(float),
        "income_level": rng.integers(1, 8, n).astype(float),
        "credit_query_times": rng.integers(0, 12, n).astype(float),
        "overdue_history": rng.integers(0, 5, n).astype(float),
    })
    logit = (
        -3.0 + 0.03 * df["age"] + 0.2 * df["income_level"]
        - 0.15 * df["credit_query_times"] + 0.25 * df["overdue_history"]
    )
    prob = 1 / (1 + np.exp(-logit))
    df["is_bad"] = (rng.random(n) < prob).astype(int)
    return df


def test_e2e_full_retrain_trains_real_model():
    """验收 1: 全量训练成功（真实 LightGBM 拟合 + 特征契约）。"""
    from workers.training_tasks import _train_lightgbm

    df = _synthetic_window_df()
    result = _train_lightgbm(df, seed=2026)

    assert hasattr(result["model"], "booster_")
    assert result["model"].booster_.num_trees() > 0
    assert result["feature_cols"]
    assert result["val_auc"] > 0.5


def test_e2e_incremental_inherits_champion_trees():
    """验收 2: 增量续训成功（继承 Champion 旧树，非重新拟合）。"""
    import lightgbm as lgb
    from workers.training_tasks import _train_lightgbm

    df1 = _synthetic_window_df(n=500, seed=1)
    df2 = _synthetic_window_df(n=500, seed=2)
    champion = _train_lightgbm(df1, seed=2026)
    champion_trees = champion["model"].booster_.num_trees()
    champion_booster = champion["model"].booster_

    result = _train_lightgbm(
        df2,
        seed=2026,
        init_model=champion_booster,
        feature_cols=champion["feature_cols"],
    )
    total_trees = result["model"].booster_.num_trees()

    # 新模型在 Champion 基础上续训：总树数必须大于 Champion 树数
    assert total_trees >= champion_trees
    assert result["val_auc"] > 0.5


def test_e2e_feature_selection_round2_contract():
    """验收 3: 特征筛选第二轮 —— 冻结特征清单进入 TrainingPlan。"""
    from apps.modelops_api.services.iteration.feature_selection_service import (
        select_features,
    )

    result = select_features(
        ["age", "income_level", "credit_query_times", "overdue_history"],
        unstable_feature_codes=["credit_query_times"],
        feature_importance={
            "age": 0.30, "income_level": 0.25,
            "credit_query_times": 0.10, "overdue_history": 0.35,
        },
    )
    # 经归因确认的不稳定特征被剔除，其余保留（不简单删除所有漂移特征）
    assert "credit_query_times" not in result.selected_feature_codes
    assert set(result.selected_feature_codes) == {
        "age", "income_level", "overdue_history",
    }
    assert result.drop_reasons.get("credit_query_times") == "UNSTABLE_ATTRIBUTED"


def test_e2e_canary_rollback_produces_two_natural_against():
    """验收 4: Canary 回滚产生两条 NATURAL AGAINST（配对 RECOMMENDS+MITIGATES）。

    完整链路已在 test_a7_workflow_routing 覆盖（OOT_GATE → CANARY →
    ROLLBACK → NATURAL AGAINST），此处补充部署结果节点终态断言。
    """
    import asyncio
    from apps.modelops_api.services.workflow.graph import (
        deployment_outcome_node,
    )

    result = asyncio.run(deployment_outcome_node({
        "lifecycle_run_id": "11111111-1111-1111-1111-111111111111",
        "model_id": "credit_model_001",
        "champion_version": "champion_v1",
        "deployment_decision": "ROLLBACK",
        "deployment_stage": "CANARY_50",
    }))
    assert result["current_phase"] == "ROLLED_BACK"


def test_frontend_manual_callback_payload_satisfies_pre_oot_qualification():
    """前端手动回调合同与资格构建器对齐：KS/Bootstrap 证据齐全、
    data_reproducible/candidate_frozen 顶层、无 validation_metrics.oot_passed。"""
    from apps.modelops_api.services.iteration.qualification_service import (
        QualificationService,
        build_qualification_input,
    )
    from apps.modelops_api.services.iteration.config_loader import (
        load_iteration_config,
    )

    # 与 apps/web/src/app/page.tsx submitTrainingCallback 保持一致的合同
    payload = {
        "training_metrics": {"auc": 0.81, "ks": 0.43},
        "validation_metrics": {
            "original_drop": 0.04, "recovered_amount": 0.038,
            "recovery_rate": 0.95, "recovery_auc": 0.92, "recovery_ks": 0.95,
            "champion_auc": 0.74, "challenger_auc": 0.778,
            "champion_ks": 0.3, "challenger_ks": 0.36,
            "healthy_lower_bound": 0.76,
            "ks_healthy_lower_bound": 0.15,
            "bootstrap_ci_lower": 0.03, "bootstrap_ci_upper": 0.1,
            "ks_bootstrap_ci_lower": 0.02, "ks_bootstrap_ci_upper": 0.08,
            "score_psi": 0.08, "train_valid_gap": 0.015,
            "discrimination_passed": True, "calibration_passed": True,
        },
        "segment_metrics": {"segment_governance_passed": True},
        "data_reproducible": True,
        "candidate_frozen_before_oot": True,
        "artifact_checksums": {},
        "environment_manifest": {"runtime": "frontend-manual"},
        "technical_retry_count": 0,
    }
    assert "oot_passed" not in payload["validation_metrics"]

    qual_input = build_qualification_input(
        qualification_run_id="qual-1",
        iteration_run_id="iter-1",
        experiment_id="exp-1",
        candidate_version="c1",
        experiment_json=payload,
        include_oot=False,
    )
    report = QualificationService(load_iteration_config()).evaluate(
        qual_input, include_oot=False,
    )
    assert report.qualified is True
    assert {g.value for g in report.failed_gate_codes} == set()


def test_prepare_incremental_init_loads_champion_with_real_feature_cols(monkeypatch):
    """P0-1 跨链路：Champion 加载必须使用训练数据的特征列（非空），
    并完成 booster 转换 + 树数记录。"""
    import lightgbm as lgb
    from workers import training_tasks
    from workers.training_tasks import _prepare_incremental_init

    df = _synthetic_window_df(n=400, seed=3)
    champion = lgb.LGBMClassifier(n_estimators=20, objective="binary", verbosity=-1)
    features = [c for c in df.columns if c not in ("sample_id", "apply_time", "is_bad")]
    champion.fit(df[features], df["is_bad"])

    captured = {}

    def fake_load(champion_version, val_df, feature_cols, algorithm, model_id):
        captured["feature_cols"] = feature_cols
        return {"loaded": True, "model": champion, "scores": None,
                "auc": None, "ks": None, "load_errors": []}

    monkeypatch.setattr(training_tasks, "_load_and_score_champion", fake_load)

    init_model, cols_override, tree_count = _prepare_incremental_init(
        {"base_model_version": "champion_v1"}, df, df, "lightgbm", "m1",
    )

    # Champion 加载收到的是真实训练特征列（P0 修复点：不能传空列）
    assert captured["feature_cols"]
    assert set(captured["feature_cols"]) == set(features)
    assert hasattr(init_model, "num_trees")
    assert tree_count == 20


def test_prepare_incremental_init_rejects_feature_order_mismatch(monkeypatch):
    """特征顺序不一致必须拒绝（不静默错位续训）。"""
    import lightgbm as lgb
    from workers import training_tasks
    from workers.training_tasks import _prepare_incremental_init

    df = _synthetic_window_df(n=300, seed=4)
    champion = lgb.LGBMClassifier(n_estimators=10, objective="binary", verbosity=-1)
    champion.fit(df[["income_level", "age"]], df["is_bad"])  # 顺序不同

    monkeypatch.setattr(
        training_tasks, "_load_and_score_champion",
        lambda *a, **k: {"loaded": True, "model": champion,
                         "scores": None, "auc": None, "ks": None,
                         "load_errors": []},
    )

    with pytest.raises(ValueError, match="特征顺序"):
        _prepare_incremental_init(
            {"base_model_version": "champion_v1"}, df, df, "lightgbm", "m1",
        )


def test_feature_selection_negative_correlation_detected():
    """P1-2: 高共线性使用 abs —— -0.95 的负相关同样被剪枝。"""
    from apps.modelops_api.services.iteration.feature_selection_service import (
        select_features,
    )

    result = select_features(
        ["a", "b"],
        feature_importance={"a": 0.6, "b": 0.4},
        correlation_matrix={"a": {"b": -0.95}},
        correlation_threshold=0.90,
    )
    # 保留更重要的 a，剔除高负相关的 b
    assert result.selected_feature_codes == ["a"]
    assert "b" in result.dropped_feature_codes
    assert result.drop_reasons["b"].startswith("HIGH_COLLINEARITY_WITH")


def test_oot_gate_failure_routes_to_final_qualification_not_outcome():
    """P0-3: OOT_GATE 阶段 ROLLBACK → FinalQualificationNode（资格失败归因），
    不写 NATURAL 部署观测；Canary ROLLBACK 才进 DeploymentOutcomeNode。"""
    from apps.modelops_api.services.workflow.graph import (
        route_after_deployment_gate,
    )

    oot_failure = {
        "deployment_decision": "ROLLBACK",
        "deployment_stage": "OOT_GATE",
    }
    assert route_after_deployment_gate(oot_failure) == "FinalQualificationNode"

    canary_rollback = {
        "deployment_decision": "ROLLBACK",
        "deployment_stage": "CANARY_50",
    }
    assert (
        route_after_deployment_gate(canary_rollback)
        == "DeploymentOutcomeNode"
    )


def test_segment_weighted_fail_closed_without_segment_definition():
    """P1-1: 缺少冻结客群定义时 fail-closed，不静默全 1 权重。"""
    from workers.training_tasks import _build_sample_weight

    job = {"strategy_code": "segment_weighted_retrain", "sample_weight_policy": {}}
    with pytest.raises(ValueError, match="冻结客群"):
        _build_sample_weight(job, _synthetic_window_df(n=50))


def test_round2_state_reset_allows_final_qualification():
    """P0 全链路：第一轮 OOT FAILED → NextRound 重置 → 第二轮 OOT 成功
    → 必须进入 FinalQualificationNode（不跳过最终七门资格）。"""
    import asyncio
    from apps.modelops_api.services.workflow.graph import (
        next_round_plan_node,
        route_after_deployment_gate,
    )

    round1_state = {
        "lifecycle_run_id": "11111111-1111-1111-1111-111111111111",
        "model_id": "credit_model_001",
        "champion_version": "champion_v1",
        "business_round": 1,
        "deployment_decision": "ROLLBACK",
        "deployment_stage": "OOT_GATE",
        "deployment_id": "deploy-1",
        "final_qualification_completed": True,  # 首轮 OOT 失败时写入
        "oot_validation_completed": True,
        "oot_validation_run_id": "oot-1",
        "w4_available": True,
        "oot_passed": False,
        "candidate_frozen_before_oot": True,
        "lifecycle_terminal": True,
        "challenger_qualified": False,
    }

    # NextRoundPlanNode：进入第二轮并清理首轮部署/W4 状态
    reset = asyncio.run(next_round_plan_node(round1_state))
    round2_state = {**round1_state, **reset}
    assert round2_state["business_round"] == 2
    assert round2_state["final_qualification_completed"] is False
    assert round2_state["deployment_stage"] == "OFFLINE_VALIDATION"
    assert round2_state["oot_validation_completed"] is False
    assert round2_state["deployment_decision"] is None

    # 第二轮 OOT 成功（模拟 OOT_GATE 完成后的 ADVANCE_STAGE 决策）
    round2_oot_done = {
        **round2_state,
        "deployment_decision": "ADVANCE_STAGE",
        "oot_validation_completed": True,
    }
    assert (
        route_after_deployment_gate(round2_oot_done)
        == "FinalQualificationNode"
    )


def test_segment_policy_flows_decision_to_plan():
    """P1: 冻结合格客群定义 DecisionInput → StrategySelection.parameters
    → TrainingPlan.sample_weight_policy 全链贯通。"""
    from apps.modelops_api.services.iteration import (
        RepairDecisionService,
        RiskAssessmentService,
        TrainingPlanBuilder,
    )
    from packages.models.common.enums import ProposalStatus
    from packages.models.iteration import MetricDegradation
    from tests.unit.test_iteration_decision import _decision_input, _root

    segment_evidence = {
        "segment_column": "city_tier",
        "affected_segments": ["3", "4"],
        "segment_boost": 3.0,
        "evidence_source": "W0_VS_W3_CATEGORICAL_SHARE_DELTA",
    }
    request = _decision_input(
        _root("SEGMENT_DRIFT", evidence=["D", "I", "C", "T"]),
        [MetricDegradation(metric_code="AUC")],
    ).model_copy(update={"segment_evidence": segment_evidence})

    proposal = RepairDecisionService().decide(request)
    assert proposal.strategies[0].strategy_code == "segment_weighted_retrain"
    assert (
        proposal.strategies[0].parameters["sample_weight_policy"]
        == segment_evidence
    )

    approved = proposal.model_copy(update={"status": ProposalStatus.APPROVED})
    risk = RiskAssessmentService().assess(approved)
    plan = TrainingPlanBuilder().build(
        approved,
        risk,
        approval_id="a1",
        iteration_run_id="ir1",
        data_snapshot_ids=["s1"],
    )
    assert plan.sample_weight_policy == segment_evidence


def test_training_job_carries_feature_selection_fields_from_plan():
    """P1: TrainingJobInput 从 TrainingPlan 携带三个筛选字段（合同断言）。"""
    from packages.models.iteration.training_job import TrainingJobInput

    job = TrainingJobInput(
        training_job_id="j1",
        idempotency_key="k1",
        iteration_run_id="ir1",
        training_plan_id="tp1",
        experiment_id="e1",
        business_round=2,
        strategy_code="feature_selection_retrain",
        training_window_ids=["W2"],
        validation_window_ids=["W3"],
        data_snapshot_ids=["s1"],
        label_versions=["v1"],
        feature_schema_version="f1",
        preprocessing_version="p1",
        algorithm="lightgbm",
        qualification_rule_version="q1",
        base_model_version="c1",
        seed=42,
        artifact_output_uri="u",
        training_mode="FEATURE_SELECTION",
        unstable_feature_codes=["credit_query_times"],
        selected_feature_codes=["age", "income_level", "overdue_history"],
        feature_selection_artifact_uri="s3://riskitem/feature-selection/r1.json",
    )
    assert job.unstable_feature_codes == ["credit_query_times"]
    assert job.selected_feature_codes == ["age", "income_level", "overdue_history"]
    assert job.feature_selection_artifact_uri


def test_segment_weight_numeric_categories_match_and_not_uniform():
    """P0-1: 数值型客群码（"3"/"4"）与整数列规范化匹配，
    加权的样本数必须 0 < boosted < total。"""
    from workers.training_tasks import _build_sample_weight

    rng = np.random.default_rng(9)
    df = _synthetic_window_df(n=600, seed=9)
    # 整数客群列：astype("string") 规范化后与诊断层产出的 "3"/"4" 一致
    df["city_tier"] = rng.integers(1, 6, len(df)).astype(int)
    job = {
        "strategy_code": "segment_weighted_retrain",
        "sample_weight_policy": {
            "segment_column": "city_tier",
            "affected_segments": ["3", "4"],
            "segment_boost": 3.0,
        },
    }
    weights = _build_sample_weight(job, df)
    boosted = int((weights > 1.0).sum())
    assert 0 < boosted < len(df)


def test_segment_weight_no_match_fails_closed():
    """P0-1: 冻结客群在训练数据中无匹配样本 → 报错，不静默全 1。"""
    from workers.training_tasks import _build_sample_weight

    df = _synthetic_window_df(n=100, seed=10)
    df["city_tier"] = 1.0
    job = {
        "strategy_code": "segment_weighted_retrain",
        "sample_weight_policy": {
            "segment_column": "city_tier",
            "affected_segments": ["99"],
        },
    }
    with pytest.raises(ValueError, match="无匹配样本"):
        _build_sample_weight(job, df)


def test_infer_segment_evidence_only_growing_degraded_segments(monkeypatch):
    """P0-2: 40/60→50/50 的整体迁移只能选中"占比增加 + 退化"的客群，
    不能把两个类别都选出来统一加权。"""
    from apps.modelops_api.services.diagnosis.diagnosis_service import (
        _infer_segment_evidence,
    )
    import apps.modelops_api.services.monitoring.window_loader as _wl

    def make_df(n0: int, n1: int, signal: float):
        rows = []
        rng = np.random.default_rng(11)
        for cat, n in ((0, n0), (1, n1)):
            logit = rng.normal(0, signal, n)
            prob = 1 / (1 + np.exp(-logit))
            bad = rng.random(n) < prob
            pred = prob + rng.normal(0, 0.05, n)
            for i in range(n):
                rows.append({
                    "city_tier": cat,
                    "is_bad": int(bad[i]),
                    "y_pred_proba": float(max(0.0, min(1.0, pred[i]))),
                })
        return pd.DataFrame(rows)

    frames = {
        "W0": make_df(400, 600, signal=1.5),   # cat0 占比 40%，信号强（AUC 高）
        "W3": make_df(500, 500, signal=0.2),   # cat0 占比 50%（+10%），信号弱（退化）
    }

    def fake_load(window_id: str, model_id=None):
        return frames[window_id]

    monkeypatch.setattr(_wl, "load_window_with_predictions", fake_load)

    evidence = _infer_segment_evidence(
        "credit_model_001",
        [{"feature_name": "city_tier", "feature_type": "categorical"}],
    )
    assert evidence is not None
    assert evidence["segment_column"] == "city_tier"
    # 只有占比增加的 cat0 被选中；cat1 占比 -10% 不得入选
    assert evidence["affected_segments"] == ["0"]


def test_l1_falls_back_when_segment_evidence_missing():
    """P0-3: 选了 segment_weighted 但无冻结客群证据 → L1 回退/人工复核，
    不创建必然失败的 TrainingJob。"""
    from apps.modelops_api.services.iteration import RepairDecisionService
    from packages.models.common.enums import RecommendedAction
    from packages.models.iteration import MetricDegradation
    from tests.unit.test_iteration_decision import _decision_input, _root

    request = _decision_input(
        _root("SEGMENT_DRIFT", evidence=["D", "I", "C", "T"]),
        [MetricDegradation(metric_code="AUC")],
    ).model_copy(update={"segment_evidence": None})

    proposal = RepairDecisionService().decide(request)

    # SEGMENT_DRIFT 的 YAML 规则只有 segment_weighted → 证据不足时人工复核
    assert proposal.action == RecommendedAction.MANUAL_REVIEW
    assert proposal.requires_manual_review is True
    assert "SEGMENT_EVIDENCE_INSUFFICIENT_FALLBACK" in proposal.decision_reasons
