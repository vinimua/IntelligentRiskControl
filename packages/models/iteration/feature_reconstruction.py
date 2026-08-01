"""Feature reconstruction contracts.

FeatureReconstructionPlan describes one feature engineering plan.
FeatureReconstructionResult is the worker callback payload after execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class FeatureOperation(str, Enum):
    DROP = "DROP"
    LOG_TRANSFORM = "LOG_TRANSFORM"
    INTERACTION = "INTERACTION"
    BINNING = "BINNING"
    STANDARDIZE = "STANDARDIZE"


class FeatureTransformItem(BaseModel):
    """A single feature transformation instruction."""

    operation: FeatureOperation
    source_feature: str
    target_feature: str | None = None
    reason: str = ""
    parameters: dict = Field(default_factory=dict)


class FeatureReconstructionPlan(BaseModel):
    """Plan produced by FeatureReconstructionNode."""

    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    lifecycle_run_id: str | None = None
    diagnosis_run_id: str | None = None
    model_id: str = ""
    current_schema_version: str = ""
    target_schema_version: str = ""
    status: str = "PLANNED"

    transforms: list[FeatureTransformItem] = Field(default_factory=list)

    drift_features: list[str] = Field(default_factory=list)
    high_missing_features: list[str] = Field(default_factory=list)

    expected_feature_count_before: int = 0
    expected_feature_count_after: int = 0

    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class FeatureReconstructionResult(BaseModel):
    """Worker result after feature reconstruction has finished."""

    plan_id: str
    status: str
    lifecycle_run_id: str | None = None
    model_id: str = ""

    transform_artifact_uri: str | None = None
    feature_snapshot_id: str | None = None
    feature_schema_version: str | None = None

    feature_count_before: int = 0
    feature_count_after: int = 0
    dropped_features: list[str] = Field(default_factory=list)
    added_features: list[str] = Field(default_factory=list)

    transform_detail: dict = Field(default_factory=dict)
    error_message: str | None = None

    completed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
