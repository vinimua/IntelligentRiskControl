"""
Evidence Package — DiagnosisService 执行 ValidationPlan 后形成的证据集合
"""

from pydantic import Field
from ..common.base import ContractModel

from ..common.enums import (
    AvailabilityStatus,
    ConfidenceLevel,
    DataTrack,
    EvidenceDirection,
    EvidenceType,
)

class EvidenceItem(ContractModel):
    """单条 D/R/C/T/I Evidence"""

    evidence_id: str
    evidence_type: EvidenceType
    method_code: str
    executor_version: str
    raw_value: dict | None = None
    normalized_score: float | None = None
    direction: EvidenceDirection | None = None
    p_value: float | None = None
    q_value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    applicable: bool = True
    availability_status: AvailabilityStatus = AvailabilityStatus.AVAILABLE
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    baseline_id: str | None = None
    data_track: DataTrack = DataTrack.NATURAL
    dataset_snapshot_id: str | None = None
    window_ids: list[str] = Field(default_factory=list)
    evidence_detail_json: dict | None = None

class EvidencePackage(ContractModel):
    """
    对某个候选根因的完整证据集合
    NOT_APPLICABLE 必须显式保存，不能以 normalized_score=0 代替
    """

    diagnosis_run_id: str
    alert_id: str
    diagnosis_candidate_id: str
    root_cause_code: str
    hypothesis_code: str
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    evidence_score: float | None = None
    evidence_coverage: float | None = None
    aggregation_rule_version: str | None = None
