"""T4-GAP-04: 部署健康报告合同。"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


class HealthMetricCheck(BaseModel):
    """单条健康指标检查结果。"""
    metric: str
    value: float | bool | None = None
    threshold: float | bool | None = None
    direction: str = ""
    passed: bool = True
    detail: str = ""


class DeploymentHealthReport(BaseModel):
    """部署阶段健康报告 — DeploymentHealthCheckService 产出。"""

    report_id: str = Field(default_factory=lambda: str(uuid4()))
    deployment_id: str = ""
    stage: str = ""
    lifecycle_run_id: str | None = None
    model_id: str = ""

    traffic_ratio: float = 0.0
    passed: bool = True
    checks: list[HealthMetricCheck] = Field(default_factory=list)
    rollback_recommended: bool = False
    rollback_reasons: list[str] = Field(default_factory=list)

    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
