"""诊断 KG 治理语义专项测试。

覆盖:
- supporting_only 两阶段聚合（顺序无关）
- supporting-only 单独存在时不建独立候选
- required_context 前置条件检查
- gate_only 告警在诊断入口阻断 → INSUFFICIENT_DATA
"""

from __future__ import annotations

import pytest

from packages.models.common.enums import (
    AvailabilityStatus,
    DataTrack,
    DimensionCode,
    ObjectType,
    Severity,
)
from packages.models.diagnosis.diagnosis_context import CandidateRootCause
from packages.models.monitoring.alert_context import AlertContext, AlertDetail
from apps.modelops_api.services.diagnosis.diagnosis_service import (
    DiagnosisService,
)


def _rc(alert_code: str, root_code: str, weight: float = 0.10, **kwargs) -> CandidateRootCause:
    return CandidateRootCause(
        diagnosis_candidate_id=f"cand-{alert_code}-{root_code}",
        alert_code=alert_code,
        relation_key=f"{alert_code}|INDICATES|{root_code}",
        root_cause_code=root_code,
        dimension_code=DimensionCode.FEATURE,
        effective_weight_snapshot=weight,
        evidence_case_count_snapshot=0,
        confidence_lower_bound_snapshot=0.0,
        **kwargs,
    )


class FakeKnowledge:
    """按 alert_code 返回候选的假 KnowledgeService。"""

    def __init__(self, mapping: dict[str, list[CandidateRootCause]]):
        self.mapping = mapping
        self.gate_alerts: list[dict] = []

    async def query_candidate_root_causes(self, alert_code: str):
        return self.mapping.get(alert_code, [])

    async def query_gate_blocking_alerts(self, alert_codes: list[str]):
        return [g for g in self.gate_alerts if g["alert_code"] in alert_codes]


def _service(knowledge) -> DiagnosisService:
    return DiagnosisService(session=None, knowledge=knowledge, repo=None)


def _alert(code: str, metric_detail: dict | None = None) -> object:
    return type("Alert", (), {"alert_code": code, "metric_detail": metric_detail})()


# ── supporting_only 聚合 ──


@pytest.mark.asyncio
async def test_supporting_only_requires_primary_alert():
    """只有 supporting-only 告警时不建独立候选。"""
    knowledge = FakeKnowledge({
        "AUC_DROP": [
            _rc("AUC_DROP", "PRIOR_PROBABILITY_SHIFT", supporting_only=True),
        ],
    })
    candidates = await _service(knowledge)._recall_candidates([_alert("AUC_DROP")])
    assert candidates == []


@pytest.mark.asyncio
async def test_supporting_only_aggregation_is_order_independent():
    """主告警 + supporting-only 告警的聚合结果与告警顺序无关。"""
    def build_mapping():
        return {
            "AUC_DROP": [
                _rc("AUC_DROP", "PRIOR_PROBABILITY_SHIFT", weight=0.03, supporting_only=True),
            ],
            "BAD_RATE_SHIFT": [
                _rc("BAD_RATE_SHIFT", "PRIOR_PROBABILITY_SHIFT", weight=0.30),
            ],
        }

    # 顺序 1: AUC 先
    first = await _service(FakeKnowledge(build_mapping()))._recall_candidates(
        [_alert("AUC_DROP"), _alert("BAD_RATE_SHIFT")]
    )
    # 顺序 2: BAD_RATE 先
    second = await _service(FakeKnowledge(build_mapping()))._recall_candidates(
        [_alert("BAD_RATE_SHIFT"), _alert("AUC_DROP")]
    )

    assert len(first) == len(second) == 1
    for candidates in (first, second):
        cand = candidates[0]
        assert cand.root_cause_code == "PRIOR_PROBABILITY_SHIFT"
        # 主关系是 BAD_RATE_SHIFT（supporting-only 不升主）
        assert cand.alert_code == "BAD_RATE_SHIFT"
        assert cand.effective_weight_snapshot == 0.30
        # supporting-only 告警作为辅助证据保留
        assert set(cand.supporting_alert_codes) == {"AUC_DROP", "BAD_RATE_SHIFT"}


# ── required_context 前置条件 ──


@pytest.mark.asyncio
async def test_required_context_satisfied_keeps_candidate():
    """required_context 能被真实 payload 证据满足 → 保留候选。"""
    knowledge = FakeKnowledge({
        "SCHEMA_MISMATCH": [
            _rc("SCHEMA_MISMATCH", "data_pipeline_issue", weight=0.35,
                required_context=["schema_contract_violation"]),
        ],
    })
    candidates = await _service(knowledge)._recall_candidates(
        [_alert("SCHEMA_MISMATCH", {"missing_columns": ["age"], "extra_columns": ["new_col"]})]
    )
    assert len(candidates) == 1
    assert candidates[0].root_cause_code == "data_pipeline_issue"


@pytest.mark.asyncio
async def test_required_context_schema_without_payload_evidence_dropped():
    """SCHEMA_MISMATCH 告警码本身不算证据：无真实列级差异 → 丢弃候选（防自证）。"""
    knowledge = FakeKnowledge({
        "SCHEMA_MISMATCH": [
            _rc("SCHEMA_MISMATCH", "data_pipeline_issue", weight=0.35,
                required_context=["schema_contract_violation"]),
        ],
    })
    candidates = await _service(knowledge)._recall_candidates([_alert("SCHEMA_MISMATCH")])
    assert candidates == []


@pytest.mark.asyncio
async def test_required_context_own_alert_cannot_self_prove():
    """PREDICTION_MEAN_SHIFT 不能自证 prior_probability_evidence → 单独出现时丢弃。"""
    knowledge = FakeKnowledge({
        "PREDICTION_MEAN_SHIFT": [
            _rc("PREDICTION_MEAN_SHIFT", "PRIOR_PROBABILITY_SHIFT", weight=0.10,
                required_context=["prior_probability_evidence"]),
        ],
    })
    candidates = await _service(knowledge)._recall_candidates(
        [_alert("PREDICTION_MEAN_SHIFT")]
    )
    assert candidates == []


@pytest.mark.asyncio
async def test_required_context_satisfied_by_corroborating_alert():
    """PREDICTION_MEAN_SHIFT + BAD_RATE_SHIFT 同现 → prior_probability_evidence 由
    BAD_RATE_SHIFT 提供，候选保留。"""
    knowledge = FakeKnowledge({
        "PREDICTION_MEAN_SHIFT": [
            _rc("PREDICTION_MEAN_SHIFT", "PRIOR_PROBABILITY_SHIFT", weight=0.10,
                required_context=["prior_probability_evidence"]),
        ],
        "BAD_RATE_SHIFT": [
            _rc("BAD_RATE_SHIFT", "PRIOR_PROBABILITY_SHIFT", weight=0.30),
        ],
    })
    candidates = await _service(knowledge)._recall_candidates(
        [_alert("PREDICTION_MEAN_SHIFT"), _alert("BAD_RATE_SHIFT")]
    )
    assert len(candidates) == 1
    assert candidates[0].root_cause_code == "PRIOR_PROBABILITY_SHIFT"


@pytest.mark.asyncio
async def test_invalid_relation_does_not_veto_valid_relation():
    """同根因下无效关系（required_context 不满足）只丢弃自身，
    不得否决其他告警建立的有效候选。"""
    knowledge = FakeKnowledge({
        "AUC_DROP": [
            _rc("AUC_DROP", "data_pipeline_issue", weight=0.10),
        ],
        "SCHEMA_MISMATCH": [
            _rc("SCHEMA_MISMATCH", "data_pipeline_issue", weight=0.35,
                required_context=["schema_contract_violation"]),
        ],
    })
    # SCHEMA_MISMATCH 无 payload 证据 → 其关系无效；AUC_DROP 关系仍应建候选
    candidates = await _service(knowledge)._recall_candidates(
        [_alert("AUC_DROP"), _alert("SCHEMA_MISMATCH")]
    )
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.root_cause_code == "data_pipeline_issue"
    # 主关系是唯一有效关系 AUC_DROP，而非被丢弃的 SCHEMA_MISMATCH
    assert cand.alert_code == "AUC_DROP"
    assert cand.effective_weight_snapshot == 0.10
    assert cand.required_context == []
    assert cand.supporting_alert_codes == ["AUC_DROP"]


@pytest.mark.asyncio
async def test_required_context_unsatisfied_drops_candidate():
    """required_context 无法被满足 → 丢弃候选。"""
    knowledge = FakeKnowledge({
        "AUC_DROP": [
            _rc("AUC_DROP", "data_pipeline_issue", weight=0.10,
                required_context=["schema_contract_violation"]),
        ],
    })
    candidates = await _service(knowledge)._recall_candidates([_alert("AUC_DROP")])
    assert candidates == []


# ── gate_only 诊断入口阻断 ──


@pytest.mark.asyncio
async def test_gate_blocking_alert_returns_insufficient_data():
    """SAMPLE_SIZE_LOW (DATA_ELIGIBILITY_BLOCK) 阻断性能诊断。"""
    knowledge = FakeKnowledge({})
    knowledge.gate_alerts = [
        {"alert_code": "SAMPLE_SIZE_LOW", "gate_semantics": "DATA_ELIGIBILITY_BLOCK"},
    ]

    alert_context = AlertContext(
        schema_version="1.0",
        trace_id="t1",
        monitoring_run_id="mon-1",
        model_id="m1",
        model_version="v1",
        monitor_window_id="W3",
        baseline_id="W0",
        data_track=DataTrack.NATURAL,
        alert_details=[
            AlertDetail(
                alert_id="a1",
                alert_code="SAMPLE_SIZE_LOW",
                severity=Severity.CRITICAL,
                object_type=ObjectType.MODEL,
                object_code="m1",
                metric_code="SAMPLE_SIZE",
                metric_version="V1",
                availability_status=AvailabilityStatus.AVAILABLE,
            ),
            AlertDetail(
                alert_id="a2",
                alert_code="AUC_DROP",
                severity=Severity.WARNING,
                object_type=ObjectType.MODEL,
                object_code="m1",
                metric_code="AUC",
                metric_version="V1",
                availability_status=AvailabilityStatus.AVAILABLE,
            ),
        ],
    )

    result = await _service(knowledge).diagnose(
        alert_context=alert_context,
        monitoring_run_id="mon-1",
    )

    assert result.diagnosis_status == "INSUFFICIENT_DATA"
    assert result.need_iteration is False
    assert result.diagnosis_run_id is None
    assert result.primary_root_cause_code == "insufficient_data"


@pytest.mark.asyncio
async def test_no_gate_alert_proceeds_normally():
    """无 gate 告警时正常走诊断流程（召回候选并输出 COMPLETED）。"""
    from unittest.mock import AsyncMock, MagicMock, patch

    knowledge = FakeKnowledge({
        "AUC_DROP": [
            _rc("AUC_DROP", "feature_drift", weight=0.10),
        ],
    })
    knowledge.gate_alerts = []

    alert_context = AlertContext(
        schema_version="1.0",
        trace_id="t1",
        monitoring_run_id="mon-1",
        model_id="m1",
        model_version="v1",
        monitor_window_id="W3",
        baseline_id="W0",
        data_track=DataTrack.NATURAL,
        alert_details=[
            AlertDetail(
                alert_id="a2",
                alert_code="AUC_DROP",
                severity=Severity.WARNING,
                object_type=ObjectType.MODEL,
                object_code="m1",
                metric_code="AUC",
                metric_version="V1",
                availability_status=AvailabilityStatus.AVAILABLE,
            ),
        ],
    )

    service = _service(knowledge)
    service.repo = MagicMock()
    service.repo.create_run = AsyncMock(return_value={"diagnosis_run_id": "diag-1"})
    service.repo.batch_insert_candidates = AsyncMock(
        return_value={"feature_drift": "cand-1"}
    )
    service.repo.insert_evidence = AsyncMock()
    service.repo.complete_run = AsyncMock()

    with (
        patch.object(service, "_load_drift_data", AsyncMock(return_value=[])),
        patch.object(service, "_load_multi_window_drift", AsyncMock(return_value={})),
        patch.object(service, "_load_metrics", AsyncMock(return_value=[])),
        patch.object(service, "_load_feature_importance", AsyncMock(return_value=None)),
        patch(
            "apps.modelops_api.services.diagnosis.diagnosis_service.MonitoringRepo",
            return_value=MagicMock(get_run=AsyncMock(return_value={"model_id": "m1"})),
        ),
    ):
        result = await service.diagnose(
            alert_context=alert_context,
            monitoring_run_id="mon-1",
        )

    assert result.diagnosis_status == "COMPLETED"
    assert result.primary_root_cause_code == "feature_drift"


# ── 主关系切换字段同步 ──


@pytest.mark.asyncio
async def test_primary_relation_attributes_are_order_independent():
    """两条普通关系权重不同、输入顺序互换 → 主关系全部属性完全一致。"""
    def build_mapping():
        return {
            "AUC_DROP": [
                _rc("AUC_DROP", "data_pipeline_issue", weight=0.10,
                    causal_distance="INDIRECT", required_context=[]),
            ],
            "SCHEMA_MISMATCH": [
                _rc("SCHEMA_MISMATCH", "data_pipeline_issue", weight=0.35,
                    causal_distance="DIRECT",
                    required_context=["schema_contract_violation"]),
            ],
        }

    first = await _service(FakeKnowledge(build_mapping()))._recall_candidates(
        [_alert("AUC_DROP"),
         _alert("SCHEMA_MISMATCH", {"missing_columns": ["age"], "extra_columns": []})]
    )
    second = await _service(FakeKnowledge(build_mapping()))._recall_candidates(
        [_alert("SCHEMA_MISMATCH", {"missing_columns": ["age"], "extra_columns": []}),
         _alert("AUC_DROP")]
    )

    assert len(first) == len(second) == 1
    for candidates in (first, second):
        cand = candidates[0]
        assert cand.root_cause_code == "data_pipeline_issue"
        # 主关系是 SCHEMA_MISMATCH（权重 0.35 > 0.10）
        assert cand.alert_code == "SCHEMA_MISMATCH"
        assert cand.effective_weight_snapshot == 0.35
        # 治理属性同步完整
        assert cand.required_context == ["schema_contract_violation"]
        assert cand.causal_distance == "DIRECT"
        assert cand.supporting_only is False
        # 支持告警完整
        assert set(cand.supporting_alert_codes) == {"AUC_DROP", "SCHEMA_MISMATCH"}


# ── gate fail-safe ──


@pytest.mark.asyncio
async def test_gate_failsafe_when_neo4j_unavailable():
    """Neo4j 查询异常时，SAMPLE_SIZE_LOW 仍由本地定义阻断。"""
    from apps.modelops_api.services.knowledge_service import (
        KnowledgeService,
        _DEFAULT_GATE_ALERTS,
    )

    class BrokenDriver:
        def session(self, **kwargs):
            raise ConnectionError("neo4j down")

    knowledge = KnowledgeService(BrokenDriver())
    gate_alerts = await knowledge.query_gate_blocking_alerts(
        ["AUC_DROP", "SAMPLE_SIZE_LOW", "KS_DROP"]
    )

    assert gate_alerts == [
        {"alert_code": "SAMPLE_SIZE_LOW",
         "gate_semantics": "DATA_ELIGIBILITY_BLOCK"}
    ]
    # 本地定义与测试保持一致
    assert _DEFAULT_GATE_ALERTS["SAMPLE_SIZE_LOW"] == "DATA_ELIGIBILITY_BLOCK"


@pytest.mark.asyncio
async def test_primary_relation_tie_break_is_order_independent():
    """同权重关系：causal_distance DIRECT > INDIRECT，与输入顺序无关。"""
    def build_mapping():
        return {
            "HIGH_FEATURE_PSI": [
                _rc("HIGH_FEATURE_PSI", "data_pipeline_issue", weight=0.10,
                    causal_distance="INDIRECT", required_context=[]),
            ],
            "MISSING_RATE_SPIKE": [
                _rc("MISSING_RATE_SPIKE", "data_pipeline_issue", weight=0.10,
                    causal_distance="DIRECT", required_context=[]),
            ],
        }

    first = await _service(FakeKnowledge(build_mapping()))._recall_candidates(
        [_alert("HIGH_FEATURE_PSI"), _alert("MISSING_RATE_SPIKE")]
    )
    second = await _service(FakeKnowledge(build_mapping()))._recall_candidates(
        [_alert("MISSING_RATE_SPIKE"), _alert("HIGH_FEATURE_PSI")]
    )

    assert len(first) == len(second) == 1
    for candidates in (first, second):
        cand = candidates[0]
        # 权重相同 → DIRECT 胜出，与顺序无关
        assert cand.alert_code == "MISSING_RATE_SPIKE"
        assert cand.causal_distance == "DIRECT"
        assert cand.effective_weight_snapshot == 0.10
        assert set(cand.supporting_alert_codes) == {
            "HIGH_FEATURE_PSI", "MISSING_RATE_SPIKE",
        }


@pytest.mark.asyncio
async def test_primary_relation_same_weight_and_distance_uses_relation_key():
    """权重和 causal_distance 都相同 → relation_key 字典序兜底。"""
    def build_mapping():
        return {
            "AUC_DROP": [
                _rc("AUC_DROP", "feature_drift", weight=0.10, causal_distance="DIRECT"),
            ],
            "KS_DROP": [
                _rc("KS_DROP", "feature_drift", weight=0.10, causal_distance="DIRECT"),
            ],
        }

    first = await _service(FakeKnowledge(build_mapping()))._recall_candidates(
        [_alert("KS_DROP"), _alert("AUC_DROP")]
    )
    second = await _service(FakeKnowledge(build_mapping()))._recall_candidates(
        [_alert("AUC_DROP"), _alert("KS_DROP")]
    )

    assert len(first) == len(second) == 1
    # 两种顺序下主关系完全一致（relation_key 字典序最大者胜出：KS_DROP|... > AUC_DROP|...）
    assert first[0].alert_code == second[0].alert_code == "KS_DROP"
    assert first[0].relation_key == second[0].relation_key
