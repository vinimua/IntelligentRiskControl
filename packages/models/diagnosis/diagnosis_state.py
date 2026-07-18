"""
DiagnosisNode 输出
来源：技术开发文档 V1.4.2 §7.7, 接口总汇 V1.1 §6.5
"""

from ..common.base import ContractModel

from ..common.enums import DimensionCode, RecommendedAction

class DiagnosisStateOutput(ContractModel):
    """诊断流程的状态输出。"""

    diagnosis_run_id: str
    primary_root_cause_code: str
    primary_root_cause_dimension: DimensionCode
    primary_root_cause_score: float
    recommended_action: RecommendedAction
    need_iteration: bool
    requires_manual_review: bool = False
