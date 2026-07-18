"""
LangGraph 主图 State
"""

from ..common.base import ContractModel

from ..common.enums import LifecyclePhase, TriggerType

class ModelLifecycleState(ContractModel):
    """
    主图 State — 只保存流程控制字段、结果摘要和 runId
    完整业务数据进入 PostgreSQL、MLflow 或 MinIO
    """

    schema_version: int = 1
    lifecycle_run_id: str
    model_id: str
    champion_version: str
    trigger_type: TriggerType = TriggerType.SCHEDULED_TRIGGER
    current_phase: LifecyclePhase = LifecyclePhase.CREATED
    requires_manual_review: bool = False

    # 任务一摘要
    monitoring_run_id: str | None = None
    has_alerts: bool | None = None
    alert_count: int | None = None
    max_alert_severity: str | None = None

    # 任务二摘要
    diagnosis_run_id: str | None = None
    primary_root_cause_code: str | None = None
    primary_root_cause_dimension: str | None = None
    primary_root_cause_score: float | None = None
    recommended_action: str | None = None
    need_iteration: bool | None = None

    # 任务三摘要
    iteration_run_id: str | None = None
    challenger_version: str | None = None
    challenger_qualified: bool | None = None

    # 任务四摘要
    deployment_id: str | None = None
    deployment_stage: str | None = None
    deployment_decision: str | None = None

    # 异常
    last_error: dict | None = None
