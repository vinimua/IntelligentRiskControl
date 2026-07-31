"""Create the first chronological V2 diagnosis event for every alerted model."""

from __future__ import annotations

import asyncio

from sqlalchemy import text

from apps.modelops_api.database import async_session
from apps.modelops_api.neo4j_db import get_neo4j_driver
from apps.modelops_api.repositories.diagnosis_repo import DiagnosisRepo
from apps.modelops_api.repositories.monitoring_repo import MonitoringRepo
from apps.modelops_api.routers.diagnosis import _build_alert_context, _json_object
from apps.modelops_api.services.diagnosis.diagnosis_service import DiagnosisService
from apps.modelops_api.services.diagnosis.event_timeline import (
    alert_event_time,
    alerts_at_first_event_time,
    four_non_overlapping_windows,
)
from apps.modelops_api.services.knowledge_service import KnowledgeService


async def main() -> int:
    driver = await get_neo4j_driver()
    async with async_session() as session:
        # V1 results are retained for audit but can never become current again.
        await session.execute(
            text("""
                UPDATE diagnosis.diagnosis_runs
                SET status = 'LEGACY_INVALID'
                WHERE logic_version = 'V1_LEGACY'
                  AND status <> 'LEGACY_INVALID'
            """)
        )
        runs = (
            await session.execute(
                text("""
                    SELECT DISTINCT ON (mr.model_id)
                           mr.*
                    FROM monitoring.monitoring_runs mr
                    JOIN monitoring.monitoring_alerts a
                      ON a.monitoring_run_id = mr.monitoring_run_id
                    ORDER BY mr.model_id, mr.started_at DESC
                """)
            )
        ).mappings().all()
        diagnosis_repo = DiagnosisRepo(session)
        monitoring_repo = MonitoringRepo(session)
        service = DiagnosisService(
            session, KnowledgeService(driver), diagnosis_repo
        )

        for index, run in enumerate(runs, start=1):
            model_id = str(run["model_id"])
            version = str(run["champion_version"])
            active = await diagnosis_repo.get_active_event(model_id, version)
            if active:
                print(
                    f"[{index}/{len(runs)}] SKIP {model_id}: "
                    f"event={active['event_id']} status={active['status']}",
                    flush=True,
                )
                continue
            run_id = str(run["monitoring_run_id"])
            alerts = await monitoring_repo.get_unassigned_alerts(run_id)
            event_alerts = alerts_at_first_event_time(alerts)
            if not event_alerts:
                continue
            event_time = alert_event_time(event_alerts[0])
            event = await diagnosis_repo.create_event(
                model_id, version, run_id, event_time,
                [str(alert["alert_id"]) for alert in event_alerts],
            )
            detail = _json_object(event_alerts[0].get("alert_detail"))
            windows = four_non_overlapping_windows(
                event_time, int(detail.get("window_days") or 7)
            )
            result = await service.diagnose(
                alert_context=_build_alert_context(run, event_alerts),
                monitoring_run_id=run_id,
                event_id=str(event["event_id"]),
                evidence_window_ids=[window["window_id"] for window in windows],
            )
            await diagnosis_repo.mark_event_diagnosed(str(event["event_id"]))
            await session.commit()
            print(
                f"[{index}/{len(runs)}] DONE {model_id} "
                f"event_time={event_time.date()} "
                f"root={result.primary_root_cause_code} "
                f"score={result.primary_root_cause_score}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
