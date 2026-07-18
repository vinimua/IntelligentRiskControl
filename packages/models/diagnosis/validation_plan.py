"""
ValidationPlan — DiagnosisService 根据 Diagnosis Context 确定性生成
"""

from pydantic import Field
from ..common.base import ContractModel

from ..common.enums import EvidenceType, ValidationStepStatus

class ValidationStep(ContractModel):
    """单条验证步骤"""

    diagnosis_candidate_id: str
    root_cause_code: str
    hypothesis_code: str
    evidence_type: EvidenceType
    method_code: str
    executor_code: str
    required: bool
    input_refs: dict | None = None
    parameter_profile_code: str | None = None
    status: ValidationStepStatus = ValidationStepStatus.PENDING
    skip_reason: str | None = None

class ValidationPlan(ContractModel):
    """
    验证计划 — 由规则引擎确定性生成，不由 LLM 自由规划
    反事实修复是其中产生 R 类 Evidence 的一种验证步骤
    """

    validation_plan_id: str
    diagnosis_run_id: str
    context_pack_id: str
    rule_version: str
    steps: list[ValidationStep] = Field(default_factory=list)
