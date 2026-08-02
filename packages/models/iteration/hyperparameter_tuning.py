"""T3-GAP-02: 超参优化合同模型。

HyperparameterTuningPlan — N 组候选超参的描述。
TuningRun — 一次 tuning 执行的总记录。
TuningTrial — 单组超参的训练结果。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class TuningStrategy(str, Enum):
    RANDOM = "RANDOM"       # 随机搜索 N 组
    GRID = "GRID"           # 网格搜索
    # OPTO 暂不引入额外依赖


class TuningTrial(BaseModel):
    """单次 trial 结果。"""
    trial_id: str = Field(default_factory=lambda: str(uuid4()))
    trial_index: int = 0
    status: str = "PENDING"  # PENDING | RUNNING | SUCCEEDED | FAILED
    hyperparameters: dict = Field(default_factory=dict)
    train_auc: float | None = None
    val_auc: float | None = None
    val_ks: float | None = None
    train_time_seconds: float | None = None
    error_message: str | None = None


class HyperparameterTuningPlan(BaseModel):
    """超参搜索计划 — HyperparameterTuningNode 产出。"""

    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    lifecycle_run_id: str | None = None
    training_plan_id: str | None = None
    model_id: str = ""
    algorithm: str = "lightgbm"
    strategy: str = "RANDOM"  # RANDOM | GRID
    num_trials: int = 5
    status: str = "PLANNED"

    # 搜索空间定义
    search_space: dict = Field(default_factory=dict)
    trials: list[TuningTrial] = Field(default_factory=list)
    best_trial_index: int | None = None

    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HyperparameterTuningResult(BaseModel):
    """Worker 回调结果。"""

    plan_id: str
    status: str  # SUCCEEDED | FAILED
    lifecycle_run_id: str | None = None
    algorithm: str = "lightgbm"

    trials: list[TuningTrial] = Field(default_factory=list)
    best_trial_index: int | None = None
    best_hyperparameters: dict = Field(default_factory=dict)
    best_val_auc: float | None = None

    error_message: str | None = None
    completed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
