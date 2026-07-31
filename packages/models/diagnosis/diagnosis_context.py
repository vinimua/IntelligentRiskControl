"""
Diagnosis Context — KnowledgeService 返回给 DiagnosisService 的候选知识召回包
不是最终诊断结论，也不负责选择任务三 Strategy
"""

from pydantic import Field
from ..common.base import ContractModel

from ..common.enums import DimensionCode, EvidenceType

class ValidationMethodRef(ContractModel):
    """候选验证方法引用 — 对应 executor_registry"""

    method_code: str
    executor_code: str
    evidence_type: EvidenceType
    method_weight: float
    required: bool
    parameter_profile_code: str | None = None

class HypothesisRef(ContractModel):
    """待验证因果假设"""

    hypothesis_code: str
    symptom_code: str
    validation_methods: list[ValidationMethodRef] = Field(default_factory=list)

class CandidateRootCause(ContractModel):
    """单个候选根因 — 包含图谱权重快照"""

    diagnosis_candidate_id: str
    alert_code: str
    relation_key: str  # Alert|INDICATES|RootCause
    root_cause_code: str
    dimension_code: DimensionCode
    effective_weight_snapshot: float
    evidence_case_count_snapshot: int
    confidence_lower_bound_snapshot: float
    hypotheses: list[HypothesisRef] = Field(default_factory=list)

class DocumentRef(ContractModel):
    """Qdrant 返回的文档引用"""

    chunk_id: str
    document_id: str
    section_path: str | None = None
    page_number: int | None = None
    score: float

class DiagnosisContext(ContractModel):
    """
    候选知识召回包
    不返回 Strategy 候选，不包含正式根因排名
    """

    context_pack_id: str
    schema_version: str
    query_profile: str
    weight_version: str
    retrieval_degraded: bool = False
    candidate_root_causes: list[CandidateRootCause] = Field(default_factory=list)
    supporting_documents: list[DocumentRef] = Field(default_factory=list)
