"""诊断排名治理收窄（2026-08-15）：根因感知验证器 + 平局裁决 + 退化门槛。

背景：自然链路诊断曾出现 4 个候选并列 0.46（全部 0.10 权重 + psi_check
对所有候选 blanket SUPPORT），business_policy_change 靠召回顺序赢并列，
feature_drift 被"反事实修复 AGAINST"压死，自动训练链路永远无法触发。
"""

import pytest

from apps.modelops_api.services.diagnosis.diagnosis_service import (
    DiagnosisService,
    _distance_priority,
)
from apps.modelops_api.services.diagnosis.validators.metric_binding import (
    has_ranking_degradation,
    resolve_metric_from_supporting_alerts,
)
from apps.modelops_api.services.diagnosis.validators.psi_check import psi_check
from apps.modelops_api.services.diagnosis.validators.counterfactual_repair_check import (
    counterfactual_repair_check,
)
from apps.modelops_api.services.diagnosis.validators.permutation_importance_check import (
    permutation_importance_check,
)
from packages.models.common.enums import EvidenceDirection
from packages.models.diagnosis.diagnosis_context import CandidateRootCause


# ═══════════════════════════════════════════════════════
# psi_check 根因感知
# ═══════════════════════════════════════════════════════

def _drift_rows(psi: float = 0.18, feature_type: str = "numeric") -> list[dict]:
    return [{
        "feature_name": "income_level",
        "feature_type": feature_type,
        "psi": psi,
    }]


@pytest.mark.asyncio
async def test_psi_check_supports_drift_root_causes():
    item = await psi_check(_drift_rows(), "HIGH_FEATURE_PSI", root_cause_code="feature_drift")
    assert item.applicable is True
    assert item.direction == EvidenceDirection.SUPPORT


@pytest.mark.asyncio
async def test_psi_check_not_applicable_for_business_policy_change():
    """漂移证据不能 blanket SUPPORT 业务政策变化。"""
    item = await psi_check(
        _drift_rows(), "HIGH_FEATURE_PSI", root_cause_code="business_policy_change"
    )
    assert item.applicable is False
    assert item.direction == EvidenceDirection.NEUTRAL


@pytest.mark.asyncio
async def test_psi_check_numeric_only_drift_not_supporting_population_shift():
    """纯数值漂移与 covariate drift 不可区分，不构成客群迁移的判别证据。"""
    item = await psi_check(
        _drift_rows(), "HIGH_FEATURE_PSI", root_cause_code="population_shift"
    )
    assert item.applicable is False


@pytest.mark.asyncio
async def test_psi_check_categorical_drift_supports_population_shift():
    item = await psi_check(
        _drift_rows(feature_type="categorical"),
        "HIGH_FEATURE_PSI",
        root_cause_code="population_shift",
    )
    assert item.applicable is True
    assert item.direction == EvidenceDirection.SUPPORT


# ═══════════════════════════════════════════════════════
# 退化门槛（counterfactual / permutation）
# ═══════════════════════════════════════════════════════

def _metrics(delta: float | None = None, degraded: bool | None = None) -> list[dict]:
    row = {"metric_code": "AUC", "baseline_value": 0.99, "current_value": 0.97}
    if delta is not None:
        row["delta"] = delta
    if degraded is not None:
        row["degraded"] = degraded
    return [row]


def test_has_ranking_degradation():
    assert has_ranking_degradation(None) is False
    assert has_ranking_degradation([]) is False
    assert has_ranking_degradation(_metrics(degraded=True)) is True
    # monitoring_metrics 的 delta = current - baseline：AUC 退化时为负
    assert has_ranking_degradation(_metrics(delta=-0.03)) is True
    assert has_ranking_degradation(_metrics(delta=-0.01)) is False
    assert has_ranking_degradation([{"metric_code": "ECE", "delta": 0.05}]) is False


@pytest.mark.asyncio
async def test_counterfactual_not_applicable_without_ranking_degradation():
    """没有性能退化时，反事实修复没有评估对象——不构成反证。"""
    item = await counterfactual_repair_check(
        _drift_rows(), "HIGH_FEATURE_PSI",
        root_cause_code="feature_drift",
        feature_importance={"income_level": 0.9, "other": 0.1},
        metrics=_metrics(delta=0.01),
    )
    assert item.applicable is False
    assert item.direction == EvidenceDirection.NEUTRAL


@pytest.mark.asyncio
async def test_permutation_not_applicable_without_ranking_degradation():
    item = await permutation_importance_check(
        _drift_rows(), "HIGH_FEATURE_PSI",
        feature_importance={"income_level": 0.9, "other": 0.1},
        metrics=_metrics(delta=0.01),
    )
    assert item.applicable is False


@pytest.mark.asyncio
async def test_counterfactual_recovery_flips_against_for_concept_drift():
    """修复可恢复性能：对 feature_drift 是 SUPPORT，对 CONCEPT_DRIFT 是 AGAINST。"""
    kwargs = dict(
        feature_importance={"income_level": 1.0},
        metrics=_metrics(delta=-0.03),
    )
    drift_item = await counterfactual_repair_check(
        _drift_rows(psi=0.32), "HIGH_FEATURE_PSI",
        root_cause_code="feature_drift", **kwargs,
    )
    concept_item = await counterfactual_repair_check(
        _drift_rows(psi=0.32), "AUC_DROP",
        root_cause_code="CONCEPT_DRIFT", **kwargs,
    )
    assert drift_item.direction == EvidenceDirection.SUPPORT
    assert concept_item.direction == EvidenceDirection.AGAINST
    # 方向翻转时分数同步翻转（AGAINST 低分=强反证）
    assert abs(concept_item.normalized_score - (1.0 - drift_item.normalized_score)) < 1e-4


# ═══════════════════════════════════════════════════════
# C/T 验证器指标绑定（supporting alerts 兜底）
# ═══════════════════════════════════════════════════════

def test_resolve_metric_from_supporting_alerts():
    assert resolve_metric_from_supporting_alerts("AUC_DROP") == "AUC"
    # HIGH_FEATURE_PSI 自身绑定不到 → 从同候选的 supporting 告警找
    assert resolve_metric_from_supporting_alerts("HIGH_FEATURE_PSI") is None
    assert (
        resolve_metric_from_supporting_alerts(
            "HIGH_FEATURE_PSI", ["HIGH_FEATURE_PSI", "AUC_DROP", "KS_DROP"]
        )
        == "AUC"
    )
    assert (
        resolve_metric_from_supporting_alerts(
            "HIGH_FEATURE_PSI", ["HIGH_FEATURE_PSI", "KS_DROP"]
        )
        == "KS"
    )
    # 没有性能告警同现 → 保持严格，不做 AUC/KS 混用
    assert (
        resolve_metric_from_supporting_alerts(
            "HIGH_FEATURE_PSI", ["HIGH_FEATURE_PSI", "HIGH_SCORE_PSI"]
        )
        is None
    )


# ═══════════════════════════════════════════════════════
# 平局裁决：DIRECT > INDIRECT（不得由召回顺序决定主因）
# ═══════════════════════════════════════════════════════

def _candidate(root_code: str, weight: float, distance: str) -> CandidateRootCause:
    return CandidateRootCause(
        diagnosis_candidate_id=f"cand-{root_code}",
        root_cause_code=root_code,
        alert_code="HIGH_FEATURE_PSI",
        relation_key=f"HIGH_FEATURE_PSI|INDICATES|{root_code}",
        dimension_code="FEATURE",
        effective_weight_snapshot=weight,
        evidence_case_count_snapshot=0,
        confidence_lower_bound_snapshot=0.0,
        causal_distance=distance,
    )


@pytest.mark.asyncio
async def test_rank_tie_break_direct_beats_indirect():
    svc = DiagnosisService(session=None, knowledge=None, repo=None)
    # 反向放入：indirect 在前（模拟召回顺序把 weak 候选排前面）
    candidates = [
        _candidate("population_shift", 0.10, "INDIRECT"),
        _candidate("feature_drift", 0.10, "DIRECT"),
    ]
    evidence_packages = {"population_shift": [], "feature_drift": []}
    ranked = await svc._rank(candidates, evidence_packages)
    assert ranked[0][0].root_cause_code == "feature_drift"
    assert ranked[1][0].root_cause_code == "population_shift"


def test_distance_priority_order():
    assert _distance_priority("DIRECT") > _distance_priority("INDIRECT")
    assert _distance_priority("INDIRECT") > _distance_priority("")
