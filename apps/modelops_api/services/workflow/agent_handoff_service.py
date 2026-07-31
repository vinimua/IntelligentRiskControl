"""DiagnosisHandoffService — Agent handoff 拼装逻辑。

从 Router 下沉为 Service，供 LangGraph 节点和 HTTP Router 复用。

LangGraph 开发路线 V1.0 §6。
"""
from __future__ import annotations

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from apps.modelops_api.core.exceptions import NotFoundError
from apps.modelops_api.repositories.diagnosis_repo import DiagnosisRepo

logger = structlog.get_logger(__name__)


class DiagnosisHandoffService:
    """诊断→Agent 交接服务。

    职责：
    - 读取诊断事件和运行结果
    - 拼装 Agent handoff 合同
    - 验证事件状态
    - 不调用 Agent，不修改 State
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def build_handoff(self, event_id: str) -> dict:
        """构建 Agent handoff 合同数据。

        返回的 dict 可直接用于 HTTP 响应，也可用于 LangGraph 节点读取。
        """
        repo = DiagnosisRepo(self.session)
        event = await repo.get_event(event_id)
        if not event:
            raise NotFoundError(f"诊断事件 {event_id} 不存在")

        run = await repo.get_run_by_event(event_id)

        handoff = {
            "event_id": event_id,
            "event_status": event["status"],
            "model_id": event["model_id"],
            "model_version": event["model_version"],
            "diagnosis_time": (
                event["event_time"].isoformat()
                if hasattr(event["event_time"], "isoformat")
                else str(event["event_time"])
            ),
            "diagnosis_run_id": str(run["diagnosis_run_id"]) if run else None,
            "primary_root_cause_code": (
                run.get("primary_root_cause_code") if run else None
            ),
            "primary_root_cause_score": (
                run.get("primary_root_cause_score") if run else None
            ),
            "recommended_action": run.get("recommended_action") if run else None,
            "next_stage": "AGENT_DECISION",
            "agent_connected": False,
            "handoff_status": "READY_NOT_DISPATCHED",
        }

        logger.info(
            "agent_handoff_built",
            event_id=event_id,
            handoff_status=handoff["handoff_status"],
            recommended_action=handoff["recommended_action"],
        )

        return handoff

    async def validate_handoff(self, handoff: dict) -> None:
        """验证 handoff 合同是否允许推进到 Agent 决策。

        抛出 NotFoundError 如果状态不符合。
        """
        event_status = handoff.get("event_status")
        next_stage = handoff.get("next_stage")

        if event_status != "WAITING_AGENT_DECISION":
            raise NotFoundError(
                f"诊断事件状态 {event_status} 不允许 Agent 决策，"
                f"预期 WAITING_AGENT_DECISION"
            )

        if next_stage != "AGENT_DECISION":
            raise NotFoundError(
                f"handoff next_stage {next_stage} 不是 AGENT_DECISION"
            )

        logger.info("agent_handoff_validated", event_id=handoff.get("event_id"))
