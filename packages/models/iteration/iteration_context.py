"""
Iteration Context — KnowledgeService 返回策略候选
来源：知识图谱接口 V1.1 §5.1
"""

from pydantic import Field
from ..common.base import ContractModel

from ..diagnosis.diagnosis_context import DocumentRef

class StrategyCandidate(ContractModel):
    """策略候选 — 包含图谱权重快照和历史效果"""

    strategy_code: str
    recommends_relation_key: str  # RootCause|RECOMMENDS|Strategy
    mitigates_relation_key: str   # Strategy|MITIGATES|RootCause
    relation_effective_weight_snapshot: float
    historical_effectiveness: float
    support_case_count: int
    total_case_count: int
    natural_case_count: int
    confidence_lower_bound: float
    required_data_codes: list[str] = Field(default_factory=list)
    allowed_training_window_ids: list[str] = Field(default_factory=list)
    training_cost_level: str = "MEDIUM"
    risk_level: str = "LOW"
    executor_code: str

class IterationContext(ContractModel):
    """
    策略候选知识召回包
    不返回 final_strategy_confidence 或 selected_strategy
    """

    context_pack_id: str
    diagnosis_run_id: str
    root_cause_code: str
    weight_version: str
    strategy_candidates: list[StrategyCandidate] = Field(default_factory=list)
    rules: dict | None = None
    retrieved_references: list[DocumentRef] = Field(default_factory=list)
    retrieval_degraded: bool = False
