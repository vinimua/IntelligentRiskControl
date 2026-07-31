"""
模型注册、版本与数据窗口
"""

from ..common.base import ContractModel

from datetime import date, datetime

class ModelInfo(ContractModel):
    """模型主目录"""

    model_id: str
    model_name: str
    model_type: str
    current_champion_version: str | None = None
    stable_version: str | None = None
    status: str = "ACTIVE"  # ACTIVE / INACTIVE / RETIRED
    attributes_json: dict | None = None

class ModelVersion(ContractModel):
    """模型业务版本 — 保存 MLflow/MinIO 引用"""

    model_id: str
    version_code: str
    role: str = "CHALLENGER"  # CHAMPION / CHALLENGER / STABLE / ARCHIVED
    status: str = "REGISTERED"  # REGISTERED / VALIDATED / DEPLOYED / REJECTED / ARCHIVED
    base_version_code: str | None = None
    feature_schema_version: str | None = None
    training_snapshot_id: str | None = None
    mlflow_run_id: str | None = None
    artifact_uri: str | None = None
    metrics_json: dict | None = None
    governance_json: dict | None = None
    created_at: datetime | None = None

class DataWindow(ContractModel):
    """W0～W4/OOT 时间窗口访问策略"""

    window_id: str
    window_name: str
    start_date: date
    end_date: date
    allows_training: bool = False
    allows_monitoring_label: bool = False
    allows_diagnosis_label: bool = False
    allows_iteration_label: bool = False
    allows_deployment_label: bool = False
    is_frozen: bool = False


class FeatureSchema(ContractModel):
    """特征结构版本与内容哈希"""

    feature_schema_id: str | None = None
    model_id: str
    schema_version: str
    feature_schema_json: dict
    content_hash: str
    status: str = "ACTIVE"


class DatasetSnapshot(ContractModel):
    """数据快照 — MinIO/Parquet 引用"""

    dataset_snapshot_id: str | None = None
    model_id: str | None = None
    window_id: str | None = None
    data_track: str = "NATURAL"
    scenario_id: str | None = None
    storage_uri: str
    content_hash: str
    row_count: int | None = None
    column_count: int | None = None
    feature_schema_version: str | None = None
    label_maturity_time: datetime | None = None
    metadata_json: dict = {}


class ModelDeploymentState(ContractModel):
    """每环境当前部署状态"""

    deployment_state_id: str | None = None
    model_id: str
    environment: str = "TEST"
    active_version_code: str | None = None
    stable_version_code: str | None = None
    challenger_version_code: str | None = None
    challenger_traffic_ratio: float = 0.0
    state_version: int = 1


class DataAccessViolation(ContractModel):
    """数据访问违规记录"""

    violation_id: str | None = None
    lifecycle_run_id: str | None = None
    model_id: str | None = None
    task_phase: str
    dataset_snapshot_id: str | None = None
    window_id: str | None = None
    violation_code: str
    attempted_operation: str | None = None
    detail_json: dict = {}
