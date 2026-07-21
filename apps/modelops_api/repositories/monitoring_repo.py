"""monitoring schema 数据访问 — monitoring_runs / monitoring_metrics / monitoring_alerts"""

from __future__ import annotations

import json
import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class MonitoringRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    # ── monitoring_runs ──

    async def create_run(
        self,
        model_id: str,
        champion_version: str,
        baseline_window_id: str,
        current_window_id: str,
        data_track: str = "NATURAL",
        trace_id: str | None = None,
    ) -> dict:
        new_id = str(uuid.uuid4())
        await self.session.execute(
            text("""
                INSERT INTO monitoring.monitoring_runs
                    (monitoring_run_id, model_id, champion_version,
                     baseline_window_id, current_window_id, data_track, trace_id)
                VALUES (:id, :mid, :ver, :bid, :cid, :track, :trace)
            """),
            {
                "id": new_id, "mid": model_id, "ver": champion_version,
                "bid": baseline_window_id, "cid": current_window_id,
                "track": data_track, "trace": trace_id,
            },
        )
        return {"monitoring_run_id": new_id}

    async def complete_run(
        self,
        monitoring_run_id: str,
        overall_status: str,
        alert_count: int = 0,
        max_alert_severity: str | None = None,
        alert_context_json: dict | None = None,
    ) -> None:
        await self.session.execute(
            text("""
                UPDATE monitoring.monitoring_runs
                SET overall_status = :status,
                    alert_count = :cnt,
                    max_alert_severity = :sev,
                    alert_context_json = :ctx,
                    completed_at = NOW()
                WHERE monitoring_run_id = :id
            """),
            {
                "id": monitoring_run_id, "status": overall_status,
                "cnt": alert_count, "sev": max_alert_severity,
                "ctx": json.dumps(alert_context_json or {}, ensure_ascii=False, default=str),
            },
        )

    async def get_run(self, monitoring_run_id: str) -> dict | None:
        result = await self.session.execute(
            text("SELECT * FROM monitoring.monitoring_runs WHERE monitoring_run_id = :id"),
            {"id": monitoring_run_id},
        )
        row = result.mappings().first()
        return dict(row) if row else None

    async def list_runs(self, model_id: str | None = None, limit: int = 20) -> list[dict]:
        sql = "SELECT * FROM monitoring.monitoring_runs"
        params: dict = {}
        if model_id:
            sql += " WHERE model_id = :mid"
            params["mid"] = model_id
        sql += " ORDER BY started_at DESC LIMIT :lim"
        params["lim"] = limit
        result = await self.session.execute(text(sql), params)
        return [dict(row) for row in result.mappings()]

    # ── monitoring_metrics ──

    async def insert_metric(
        self,
        monitoring_run_id: str,
        metric_code: str,
        metric_version: str = "V1",
        object_type: str = "MODEL",
        object_code: str | None = None,
        baseline_value: float | None = None,
        current_value: float | None = None,
        delta: float | None = None,
        threshold: float | None = None,
        rule_type: str | None = None,
        threshold_rule_id: str | None = None,
        triggered: bool = False,
        availability_status: str = "AVAILABLE",
        metric_detail: dict | None = None,
    ) -> dict:
        new_id = str(uuid.uuid4())
        await self.session.execute(
            text("""
                INSERT INTO monitoring.monitoring_metrics
                    (metric_id, monitoring_run_id, metric_code, metric_version,
                     object_type, object_code, baseline_value, current_value,
                     delta, threshold, rule_type, threshold_rule_id,
                     triggered, availability_status, metric_detail)
                VALUES (:id, :rid, :code, :ver, :otype, :ocode, :base, :cur,
                        :d, :thresh, :rtype, :rid2, :trig, :astat, :det)
            """),
            {
                "id": new_id, "rid": monitoring_run_id, "code": metric_code,
                "ver": metric_version, "otype": object_type, "ocode": object_code,
                "base": baseline_value, "cur": current_value, "d": delta,
                "thresh": threshold, "rtype": rule_type, "rid2": threshold_rule_id,
                "trig": triggered, "astat": availability_status,
                "det": json.dumps(metric_detail or {}, ensure_ascii=False, default=str),
            },
        )
        return {"metric_id": new_id}

    async def get_metrics(self, monitoring_run_id: str) -> list[dict]:
        result = await self.session.execute(
            text("""
                SELECT * FROM monitoring.monitoring_metrics
                WHERE monitoring_run_id = :id ORDER BY created_at
            """),
            {"id": monitoring_run_id},
        )
        return [dict(row) for row in result.mappings()]

    # ── monitoring_alerts ──

    async def insert_alert(
        self,
        monitoring_run_id: str,
        metric_id: str | None,
        alert_code: str,
        severity: str,
        object_type: str = "MODEL",
        object_code: str | None = None,
        metric_code: str = "",
        metric_version: str = "V1",
        baseline_value: float | None = None,
        current_value: float | None = None,
        delta: float | None = None,
        threshold: float | None = None,
        rule_type: str | None = None,
        threshold_rule_id: str | None = None,
        threshold_rule_version: str | None = None,
        availability_status: str = "AVAILABLE",
        alert_detail: dict | None = None,
    ) -> dict:
        new_id = str(uuid.uuid4())
        await self.session.execute(
            text("""
                INSERT INTO monitoring.monitoring_alerts
                    (alert_id, monitoring_run_id, metric_id, alert_code, severity,
                     object_type, object_code, metric_code, metric_version,
                     baseline_value, current_value, delta, threshold,
                     rule_type, threshold_rule_id, threshold_rule_version,
                     availability_status, alert_detail)
                VALUES (:id, :rid, :mid, :acode, :sev, :otype, :ocode, :mcode,
                        :mver, :base, :cur, :d, :thresh, :rtype, :rid2, :rver,
                        :astat, :det)
            """),
            {
                "id": new_id, "rid": monitoring_run_id, "mid": metric_id,
                "acode": alert_code, "sev": severity, "otype": object_type,
                "ocode": object_code, "mcode": metric_code, "mver": metric_version,
                "base": baseline_value, "cur": current_value, "d": delta,
                "thresh": threshold, "rtype": rule_type, "rid2": threshold_rule_id,
                "rver": threshold_rule_version, "astat": availability_status,
                "det": json.dumps(alert_detail or {}, ensure_ascii=False, default=str),
            },
        )
        return {"alert_id": new_id}

    async def get_alerts(self, monitoring_run_id: str) -> list[dict]:
        result = await self.session.execute(
            text("""
                SELECT * FROM monitoring.monitoring_alerts
                WHERE monitoring_run_id = :id ORDER BY created_at
            """),
            {"id": monitoring_run_id},
        )
        return [dict(row) for row in result.mappings()]
