"""Agent 决策合同 — Pydantic 模型。

LangGraph 开发路线 V1.0 §7。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AgentDecisionInput(BaseModel):
    """Agent 决策输入 — 来自 DiagnosisHandoff + 诊断详情。"""
    lifecycle_run_id: str
    event_id: str
    diagnosis_run_id: str
    model_id: str
    champion_version: str
    primary_root_cause_code: str
    primary_root_cause_score: float | None = None
    recommended_action: str | None = None
    candidates_summary: list[dict] = Field(default_factory=list)
    evidence_summary: list[dict] = Field(default_factory=list)


class AgentDecisionOutput(BaseModel):
    """Agent 决策输出 — RuleAgentAdapter 或真实 Agent 返回值。"""
    agent_decision_id: str
    event_id: str
    diagnosis_run_id: str
    recommended_action: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    selected_root_cause_code: str = ""
    allow_auto_decision: bool = False
    requires_manual_review: bool = True
    forbidden_strategies: list[str] = Field(default_factory=list)
    required_adjustments: list[str] = Field(default_factory=list)
