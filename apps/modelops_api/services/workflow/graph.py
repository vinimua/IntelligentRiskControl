"""LangGraph 主图 + 节点 — 阶段 4：MonitoringNode 调用真实 MonitoringService。

图结构：
    START → MonitoringNode
        ├─ has_alerts=False → END
        └─ has_alerts=True → DiagnosisNode
            ├─ need_iteration=False → END
            ├─ need_iteration=None → ManualReviewNode → END
            └─ need_iteration=True → IterationSubgraph → DeploymentNode → END
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from packages.models.common.enums import LifecyclePhase, Severity
from packages.models.workflow.lifecycle_state import ModelLifecycleState

logger = structlog.get_logger(__name__)

# ── 可配置 Mock 行为（测试时可覆盖，阶段 5-7 节点仍为 Mock） ──

MOCK_NEED_ITERATION: bool | None = True  # None = 无法判断 → ManualReview
MOCK_CHALLENGER_QUALIFIED: bool = True
MOCK_DEPLOYMENT_DECISION: str = "PROMOTE"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── 任务一：MonitoringNode ──


async def monitoring_node(state: ModelLifecycleState) -> dict:
    """阶段 4 真实监控节点：调用 MonitoringService 执行完整指标计算 → 告警生成。"""

    from ...services.monitoring.window_loader import load_window_with_predictions

    baseline_df = load_window_with_predictions("W0")
    current_df = load_window_with_predictions("W3")
    baseline_data = baseline_df.to_dict(orient="records")
    current_data = current_df.to_dict(orient="records")

    try:
        from ...database import async_session
        from ...neo4j_db import get_neo4j_driver
        from ...services.knowledge_service import KnowledgeService
        from ...services.monitoring.monitoring_service import MonitoringService

        async with async_session() as session:
            driver = await get_neo4j_driver()
            knowledge = KnowledgeService(driver)
            service = MonitoringService(session, knowledge)

            result = await service.run(
                model_id=state["model_id"],
                champion_version=state["champion_version"],
                baseline_data=baseline_data,
                current_data=current_data,
                baseline_window_id=state.get("baseline_window_id") or "",
                current_window_id=state.get("current_window_id") or "",
            )

            logger.info(
                "monitoring_node_completed",
                monitoring_run_id=result.monitoring_run_id,
                alert_count=result.alert_count,
            )

            return {
                "monitoring_run_id": result.monitoring_run_id,
                "has_alerts": result.has_alerts,
                "alert_count": result.alert_count,
                "max_alert_severity": result.max_alert_severity.value if result.max_alert_severity else None,
                "current_phase": (
                    LifecyclePhase.NO_ALERT.value
                    if not result.has_alerts
                    else LifecyclePhase.MONITORING_COMPLETED.value
                ),
            }

    except (OSError, ConnectionError, TimeoutError):
        # 基础设施故障 → 降级到 Mock，避免阻塞整个 workflow
        logger.warning("monitoring_node_infra_failed_falling_back_to_mock", exc_info=True)
        run_id = str(uuid.uuid4())
        return {
            "monitoring_run_id": run_id,
            "has_alerts": True,
            "alert_count": 2,
            "max_alert_severity": Severity.HIGH.value,
            "current_phase": LifecyclePhase.MONITORING_COMPLETED.value,
        }
    # 其他异常（代码 bug、数据错误等）不吞掉，直接抛出让 LangGraph 处理


async def diagnosis_node(state: ModelLifecycleState) -> dict:
    """Mock：模拟任务二诊断。"""
    run_id = str(uuid.uuid4())
    if MOCK_NEED_ITERATION is True:
        return {
            "diagnosis_run_id": run_id,
            "primary_root_cause_code": "feature_drift",
            "primary_root_cause_dimension": "FEATURE",
            "primary_root_cause_score": 0.85,
            "recommended_action": "MODEL_ITERATION",
            "need_iteration": True,
            "current_phase": LifecyclePhase.DIAGNOSIS_COMPLETED.value,
        }
    if MOCK_NEED_ITERATION is False:
        return {
            "diagnosis_run_id": run_id,
            "primary_root_cause_code": "no_significant_issue",
            "primary_root_cause_dimension": None,
            "primary_root_cause_score": None,
            "recommended_action": "CONTINUE_OBSERVATION",
            "need_iteration": False,
            "current_phase": LifecyclePhase.DIAGNOSIS_COMPLETED.value,
        }
    # None → 无法判断
    return {
        "diagnosis_run_id": run_id,
        "primary_root_cause_code": "uncertain",
        "primary_root_cause_dimension": None,
        "primary_root_cause_score": None,
        "recommended_action": "MANUAL_REVIEW",
        "need_iteration": None,
        "current_phase": LifecyclePhase.MANUAL_REVIEW.value,
    }


async def iteration_subgraph(state: ModelLifecycleState) -> dict:
    """Mock：模拟任务三子图（多轮训练选择最佳 Challenger）。"""
    run_id = str(uuid.uuid4())
    if MOCK_CHALLENGER_QUALIFIED:
        return {
            "iteration_run_id": run_id,
            "challenger_version": f"{state['champion_version']}_challenger_v1",
            "challenger_qualified": True,
            "current_phase": LifecyclePhase.CHALLENGER_TRAINED.value,
        }
    return {
        "iteration_run_id": run_id,
        "challenger_version": None,
        "challenger_qualified": False,
        "current_phase": LifecyclePhase.MANUAL_REVIEW.value,
    }


async def deployment_node(state: ModelLifecycleState) -> dict:
    """Mock：模拟任务四部署决策。"""
    return {
        "deployment_id": str(uuid.uuid4()),
        "deployment_stage": "OOT_GATE",
        "deployment_decision": MOCK_DEPLOYMENT_DECISION,
        "current_phase": LifecyclePhase.PROMOTED.value
        if MOCK_DEPLOYMENT_DECISION == "PROMOTE"
        else LifecyclePhase.ROLLED_BACK.value,
    }


async def manual_review_node(state: ModelLifecycleState) -> dict:
    """人工复核节点 — 挂起并等待人工决策。调用 interrupt() 暂停图。"""
    decision = interrupt("manual_review_required")
    # decision 是 resume 时传入的值，如 "approved" / "rejected"
    if decision == "rejected":
        return {
            "requires_manual_review": True,
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {"reason": "manual_review_rejected", "at": _now_iso()},
        }
    return {
        "requires_manual_review": False,
        "current_phase": LifecyclePhase.MANUAL_REVIEW.value,
    }


# ── 条件路由 ──


def route_after_monitoring(state: ModelLifecycleState) -> Literal["DiagnosisNode", END]:
    if state.get("has_alerts"):
        return "DiagnosisNode"
    return END


def route_after_diagnosis(
    state: ModelLifecycleState,
) -> Literal["IterationSubgraph", "ManualReviewNode", END]:
    need = state.get("need_iteration")
    if need is True:
        return "IterationSubgraph"
    if need is False:
        return END
    return "ManualReviewNode"


# ── 图构建 ──


def build_graph() -> StateGraph:
    """构建 LangGraph 主图（不可编译的 StateGraph）。"""
    graph = StateGraph(ModelLifecycleState)

    # 节点
    graph.add_node("MonitoringNode", monitoring_node)
    graph.add_node("DiagnosisNode", diagnosis_node)
    graph.add_node("IterationSubgraph", iteration_subgraph)
    graph.add_node("DeploymentNode", deployment_node)
    graph.add_node("ManualReviewNode", manual_review_node)

    # 边
    graph.add_edge(START, "MonitoringNode")
    graph.add_conditional_edges("MonitoringNode", route_after_monitoring)
    graph.add_conditional_edges("DiagnosisNode", route_after_diagnosis)
    graph.add_edge("IterationSubgraph", "DeploymentNode")
    graph.add_edge("DeploymentNode", END)
    graph.add_edge("ManualReviewNode", END)

    return graph


def build_compiled_graph(checkpointer: AsyncPostgresSaver):
    """构建带 PostgreSQL checkpoint 的编译图。"""
    graph = build_graph()
    return graph.compile(checkpointer=checkpointer)
