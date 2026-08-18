"""
DiagnosisNode 输出
"""

from ..common.base import ContractModel

from ..common.enums import DimensionCode, RecommendedAction

class DiagnosisStateOutput(ContractModel):
    """诊断流程的状态输出。"""

    diagnosis_run_id: str | None = None
    primary_root_cause_code: str
    primary_root_cause_dimension: DimensionCode
    primary_root_cause_score: float
    recommended_action: RecommendedAction
    need_iteration: bool
    requires_manual_review: bool = False
    diagnosis_status: str = "COMPLETED"  # COMPLETED / INSUFFICIENT_DATA / NO_CANDIDATES
    # A7 §4/§5: L1 结构化上下文（CONCEPT_DRIFT 细分推导等）
    impact_scope: str | None = None  # LOCAL / GLOBAL
    change_pattern: str | None = None  # GRADUAL / SUDDEN
    # A7 §4: 冻结合格客群定义（segment_weighted_retrain 的真实权重来源）
    # {"segment_column": str, "affected_segments": list, "segment_boost": float}
    segment_evidence: dict | None = None
