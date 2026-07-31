"""共享原始数据事件与版本化派生视图合同。"""

from datetime import datetime

from pydantic import Field, model_validator

from ..common.base import ContractModel


class DataIncident(ContractModel):
    data_incident_id: str
    canonical_snapshot_id: str
    incident_code: str
    affected_model_ids: list[str] = Field(default_factory=list)
    affected_feature_codes: list[str] = Field(default_factory=list)
    status: str = "OPEN"
    created_at: datetime
    resolved_at: datetime | None = None


class DerivedDataView(ContractModel):
    derived_view_id: str
    data_incident_id: str
    canonical_snapshot_id: str
    derivation_rule_version: str
    model_id: str | None = None
    affected_feature_codes: list[str] = Field(default_factory=list)
    view_uri: str
    checksum: str
    label_imputation_forbidden: bool = True
    masking_experiment_metrics: dict = Field(default_factory=dict)
    cross_model_replay_metrics: dict = Field(default_factory=dict)
    created_at: datetime

    @model_validator(mode="after")
    def labels_must_remain_observed(self) -> "DerivedDataView":
        if not self.label_imputation_forbidden:
            raise ValueError("derived views may never impute is_bad or other labels")
        return self
