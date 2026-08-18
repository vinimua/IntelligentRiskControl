from packages.models.common.enums import DataTrack, EvidenceDirection

from apps.modelops_api.services.knowledge_observation_service import (
    KnowledgeObservationService,
)
from scripts import apply_kg_weights_to_neo4j, run_kg_calibration


def test_lifecycle_observations_include_persisted_context_fields():
    """A7 §10: 诊断观测走 AUDIT 轨道；策略选择不产生观测；
    只有生命周期冻结的部署结果才产生 NATURAL。"""
    observations = KnowledgeObservationService.build_observations(
        {
            "lifecycle_run_id": "11111111-1111-1111-1111-111111111111",
            "diagnosis_run_id": "diag-1",
            "primary_root_cause_code": "FEATURE_DRIFT",
            "primary_root_cause_score": 0.85,
            "decision_proposal_id": "proposal-1",
            "selected_strategy_code": "recent_weighted_retrain",
            "decision_reasons": [
                "KG_STRATEGY:recent_weighted_retrain",
                "SUPPORT_CASES:25",
            ],
            # 生命周期结果冻结（挑战者合格 + W4 FINAL-OOT 完成证据 + 终态）
            "deployment_id": "deploy-1",
            "deployment_decision": "PROMOTE",
            "deployment_stage": "PRODUCTION",
            "challenger_qualified": True,
            "qualification_run_id": "qual-1",
            "oot_validation_completed": True,
            "oot_validation_run_id": "oot-1",
            "w4_available": True,
            "candidate_frozen_before_oot": True,
            "oot_passed": True,
            "lifecycle_terminal": True,
            "max_alert_severity": "HIGH",
            "alert_codes": ["HIGH_FEATURE_PSI"],  # 从真实 monitoring alerts 传入
        }
    )

    # 诊断 INDICATES（AUDIT）+ 部署配对 RECOMMENDS/MITIGATES（NATURAL）
    # —— 策略选择不再写观测
    assert len(observations) == 3
    assert {obs.relation_key for obs in observations} == {
        "HIGH_FEATURE_PSI|INDICATES|FEATURE_DRIFT",
        "FEATURE_DRIFT|RECOMMENDS|recent_weighted_retrain",
        "recent_weighted_retrain|MITIGATES|FEATURE_DRIFT",
    }
    assert all(
        obs.lifecycle_run_id == "11111111-1111-1111-1111-111111111111"
        for obs in observations
    )
    assert all(obs.evidence_detail for obs in observations)
    assert observations[0].direction == EvidenceDirection.SUPPORT
    # 诊断置信度不是真实执行结果 → 纯审计轨道
    assert observations[0].data_track == DataTrack.AUDIT
    # 终态部署结果（含配对 RECOMMENDS）是真实 W4 结果 → NATURAL
    assert observations[1].data_track == DataTrack.NATURAL
    assert observations[2].data_track == DataTrack.NATURAL


def test_decision_stage_writes_no_natural_observations():
    """决策阶段（策略被选择但生命周期未冻结）不产生任何 NATURAL 观测。"""
    observations = KnowledgeObservationService.build_observations(
        {
            "lifecycle_run_id": "11111111-1111-1111-1111-111111111111",
            "diagnosis_run_id": "diag-1",
            "primary_root_cause_code": "FEATURE_DRIFT",
            "primary_root_cause_score": 0.85,
            "decision_proposal_id": "proposal-1",
            "selected_strategy_code": "recent_weighted_retrain",
            "decision_reasons": ["KG_STRATEGY:recent_weighted_retrain"],
            "alert_codes": ["HIGH_FEATURE_PSI"],
        }
    )

    natural = [
        obs for obs in observations
        if obs.data_track == DataTrack.NATURAL
    ]
    assert natural == []


def test_bayesian_shrinkage_replaces_weak_prior_rule():
    result = run_kg_calibration._bayesian_shrinkage(
        support_count=25,
        against_count=2,
        neutral_count=3,
        support_strength=21.0,
        against_strength=1.0,
    )

    assert result["new_weight"] > 0.65
    assert result["alpha_post"] == 23.0
    assert result["beta_post"] == 9.0
    assert result["confidence_lower"] < result["new_weight"] < result["confidence_upper"]


def test_calibration_cli_defaults_to_beta_v2(monkeypatch):
    captured = {}

    def fake_run_calibration(data_track: str, rule_version: str, weight_version: str) -> str:
        captured["data_track"] = data_track
        captured["rule_version"] = rule_version
        captured["weight_version"] = weight_version
        return "calibration-1"

    monkeypatch.setattr(run_kg_calibration, "run_calibration", fake_run_calibration)
    monkeypatch.setattr("sys.argv", ["run_kg_calibration.py"])

    run_kg_calibration.main()

    assert captured == {
        "data_track": "NATURAL",
        "rule_version": "BETA_BINOMIAL_V2",
        "weight_version": "KG_WEIGHT_BETA_V2",
    }


def test_apply_filters_supported_relation_templates():
    snapshots = [
        {
            "calibration_run_id": "11111111-1111-1111-1111-111111111111",
            "relation_key": "HIGH_FEATURE_PSI|INDICATES|FEATURE_DRIFT",
            "weight_version": "KG_WEIGHT_BETA_V2",
        },
        {
            "calibration_run_id": "11111111-1111-1111-1111-111111111111",
            "relation_key": "FEATURE_DRIFT|RECOMMENDS|recent_weighted_retrain",
            "weight_version": "KG_WEIGHT_BETA_V2",
        },
        {
            "calibration_run_id": "11111111-1111-1111-1111-111111111111",
            "relation_key": "unknown|IGNORED|target",
            "weight_version": "KG_WEIGHT_BETA_V2",
        },
    ]

    supported = apply_kg_weights_to_neo4j._supported_snapshots(snapshots)

    assert [item["relation_key"] for item in supported] == [
        "HIGH_FEATURE_PSI|INDICATES|FEATURE_DRIFT",
        "FEATURE_DRIFT|RECOMMENDS|recent_weighted_retrain",
    ]
    assert apply_kg_weights_to_neo4j._job_key(supported[1]) == (
        "11111111-1111-1111-1111-111111111111",
        "RECOMMENDS",
        "KG_WEIGHT_BETA_V2",
    )


def test_rollback_without_w4_completion_writes_no_natural():
    """W4 未完成（OOT 服务不可用/未执行）时，即使 ROLLBACK 也不写 NATURAL。"""
    observations = KnowledgeObservationService.build_observations(
        {
            "lifecycle_run_id": "11111111-1111-1111-1111-111111111111",
            "diagnosis_run_id": "diag-1",
            "primary_root_cause_code": "FEATURE_DRIFT",
            "primary_root_cause_score": 0.85,
            "selected_strategy_code": "recent_weighted_retrain",
            "deployment_id": "deploy-1",
            "deployment_decision": "ROLLBACK",  # Canary 阶段回滚
            "challenger_qualified": True,
            "qualification_run_id": "qual-1",
            # W4 证据缺失：OOT 未完成
            "oot_validation_completed": False,
            "w4_available": False,
            "candidate_frozen_before_oot": False,
            "lifecycle_terminal": False,
            "alert_codes": ["HIGH_FEATURE_PSI"],
        }
    )

    natural = [
        obs for obs in observations
        if obs.data_track == DataTrack.NATURAL
    ]
    assert natural == []
