"""
动态查询 Profile — 关系用途由 Profile 派生
"""

from ..common.base import ContractModel

from ..common.enums import QueryProfileCode

class QueryProfile(ContractModel):
    """
    关系成熟度是查询时派生结果，不写成固定状态
    """

    profile_code: QueryProfileCode
    min_effective_weight: float
    min_evidence_case_count: int = 0
    min_natural_case_count: int = 0
    min_confidence_lower_bound: float = 0.0
