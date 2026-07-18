"""
动态权重校准模型
"""

from ..common.base import ContractModel

from datetime import datetime

from ..common.enums import DataTrack, WorkerStatus

class CalibrationRun(ContractModel):
    """KG Calibration Run — 按时间范围和规则版本聚合"""

    calibration_run_id: str
    data_track: DataTrack = DataTrack.NATURAL
    observed_from: datetime
    observed_to: datetime
    calibration_rule_version: str
    target_weight_version: str
    status: WorkerStatus = WorkerStatus.PENDING

class RelationWeightSnapshot(ContractModel):
    """单条关系在新 Weight Version 下的完整计算结果"""

    relation_key: str
    weight_version: str
    old_effective_weight: float
    new_effective_weight: float
    old_confidence_lower_bound: float = 0.0
    new_confidence_lower_bound: float = 0.0
    evidence_count: int = 0
    natural_count: int = 0
    scenario_count: int = 0
    alpha_value: float | None = None
    beta_value: float | None = None
    applied_to_neo4j: bool = False
