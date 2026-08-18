"""失败归因与案例沉淀合同。"""

from datetime import datetime

from pydantic import Field

from ..common.base import ContractModel
from ..common.enums import DataTrack, FailureCode


class FailureReport(ContractModel):
    failure_report_id: str
    iteration_run_id: str
    experiment_id: str | None = None
    proposal_id: str
    failure_code: FailureCode
    failed_gate_codes: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    adjustment_recommendations: list[str] = Field(default_factory=list)
    retryable: bool = False
    created_at: datetime
    # A7 §5: 经归因确认的不稳定特征（第二轮特征筛选证据的真实来源）。
    # 为空表示未确认任何特征级不稳定，不得授予
    # unstable_feature_subset_confirmed / feature_selection_evidence_available。
    unstable_feature_codes: list[str] = Field(default_factory=list)
    feature_evidence_source: str | None = None


class RepairCaseRecord(ContractModel):
    case_id: str
    data_track: DataTrack
    model_id: str
    diagnosis_run_id: str
    proposal_id: str
    iteration_run_id: str | None = None
    primary_root_cause_code: str
    action: str
    strategy_codes: list[str] = Field(default_factory=list)
    outcome: str
    qualified: bool | None = None
    failure_report_id: str | None = None
    created_at: datetime
