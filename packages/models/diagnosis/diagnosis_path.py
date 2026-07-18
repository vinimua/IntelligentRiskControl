"""
Top-K 根因路径
来源：技术开发文档 V1.4.2 §7.6
"""

from pydantic import Field
from ..common.base import ContractModel

from ..common.enums import DimensionCode

class DiagnosisPath(ContractModel):
    """单条根因路径 — Top-K 排名中的一条"""

    diagnosis_path_id: str
    rank_no: int
    node_sequence: list[str] = Field(default_factory=list)
    edge_sequence: list[str] = Field(default_factory=list)
    root_cause_code: str
    dimension_code: DimensionCode
    relation_weight_snapshot: float
    evidence_d: float | None = None
    evidence_r: float | None = None
    evidence_c: float | None = None
    evidence_t: float | None = None
    evidence_i: float | None = None
    evidence_coverage: float = 0.0
    temporal_consistency: float = 0.0
    conflict_penalty: float = 0.0
    path_score: float = 0.0
    validation_summary: str | None = None
    ranking_rule_version: str | None = None
