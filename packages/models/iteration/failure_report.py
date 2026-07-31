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
