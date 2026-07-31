"""任务三数据可用性与缺失率门禁合同。"""

from pydantic import Field

from ..common.base import ContractModel
from ..common.enums import DataTrack, DataUsabilityStatus, MissingRateBand


class FeatureMissingStat(ContractModel):
    feature_code: str
    missing_rate: float = Field(ge=0.0, le=1.0)
    is_critical: bool = False
    band: MissingRateBand | None = None
    training_blocked: bool = False


class DataEligibilityInput(ContractModel):
    window_id: str
    data_track: DataTrack = DataTrack.NATURAL
    data_snapshot_id: str | None = None
    data_checksum: str | None = None
    label_column: str = "is_bad"
    label_missing_rate: float = Field(ge=0.0, le=1.0)
    label_mature: bool = True
    label_imputation_requested: bool = False
    feature_missing_stats: list[FeatureMissingStat] = Field(default_factory=list)
    requested_for_supervised_training: bool = True


class DataEligibilityResult(ContractModel):
    window_id: str
    status: DataUsabilityStatus
    supervised_training_allowed: bool
    unsupervised_analysis_allowed: bool = True
    label_imputation_forbidden: bool = True
    excluded_unlabelled_rows: bool = True
    data_track: DataTrack = DataTrack.NATURAL
    data_snapshot_id: str | None = None
    data_checksum: str | None = None
    feature_results: list[FeatureMissingStat] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rule_version: str
