"""monitoring schema 数据访问 — monitoring_runs / monitoring_metrics / monitoring_alerts"""

from __future__ import annotations

import json
import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class MonitoringRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    # ── data_windows lookup ──

    async def get_data_window(self, window_id: str) -> dict | None:
        """查询 model_registry.data_windows 获取窗口真实权限和成熟时间。"""
        result = await self.session.execute(
            text("""
                SELECT window_id, window_name, start_time, end_time,
                       allows_monitoring_label, allows_diagnosis_label,
                       allows_iteration_label, allows_deployment_label,
                       allows_training, is_frozen
                FROM model_registry.data_windows
                WHERE window_id = :wid
            """),
            {"wid": window_id},
        )
        row = result.mappings().first()
        return dict(row) if row else None

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

    async def update_metric_triggered(self, metric_id: str, triggered: bool) -> None:
        """更新指标的 triggered 字段为真实的阈值触发状态。"""
        await self.session.execute(
            text("UPDATE monitoring.monitoring_metrics SET triggered = :trig WHERE metric_id = :id"),
            {"id": metric_id, "trig": triggered},
        )

    async def update_persistence_judgment(
        self, monitoring_run_id: str, judgment_json: dict, diagnosis_status: str,
    ) -> None:
        """写入 B1 持续性判定结果。"""
        await self.session.execute(
            text("""
                UPDATE monitoring.monitoring_runs
                SET persistence_judgment_json = CAST(:judgment AS JSONB),
                    diagnosis_status = :status
                WHERE monitoring_run_id = :id
            """),
            {
                "id": monitoring_run_id,
                "judgment": json.dumps(judgment_json, ensure_ascii=False, default=str),
                "status": diagnosis_status,
            },
        )

    async def update_diagnosis_status(self, monitoring_run_id: str, diagnosis_status: str) -> None:
        """Update diagnosis lifecycle status for a monitoring run."""
        await self.session.execute(
            text("""
                UPDATE monitoring.monitoring_runs
                SET diagnosis_status = :status
                WHERE monitoring_run_id = :id
            """),
            {"id": monitoring_run_id, "status": diagnosis_status},
        )

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
        alerts: list[dict] = []
        for row in result.mappings():
            item = dict(row)
            detail = item.get("alert_detail") or {}
            if isinstance(detail, str):
                try:
                    detail = json.loads(detail)
                except json.JSONDecodeError:
                    detail = {}
            if isinstance(detail, dict):
                item["alert_detail"] = detail
                metric_detail = detail.get("metric_detail") or {}
                source = detail.get("source")
                if not source and isinstance(metric_detail, dict):
                    source = metric_detail.get("source")
                if source:
                    item["source"] = source
            alerts.append(item)
        return alerts

    async def get_unassigned_alerts(self, monitoring_run_id: str) -> list[dict]:
        result = await self.session.execute(
            text("""
                SELECT a.*
                FROM monitoring.monitoring_alerts a
                LEFT JOIN diagnosis.diagnosis_event_alerts ea
                  ON ea.alert_id = a.alert_id
                WHERE a.monitoring_run_id = :id AND ea.alert_id IS NULL
            """),
            {"id": monitoring_run_id},
        )
        return [dict(row) for row in result.mappings()]

    # ── monitoring_feature_drift ──

    async def batch_insert_feature_drift(
        self,
        monitoring_run_id: str,
        drift_rows: list[dict],
        quality_rows: list[dict] | None = None,
    ) -> int:
        """批量持久化 per-feature drift + quality 数据。"""
        if not drift_rows:
            return 0

        merged: dict[tuple[str, str], dict] = {}
        for d in drift_rows:
            key = (d.get("window_id", d.get("monitor_window_id", "?")),
                   d.get("feature_name", "?"))
            merged.setdefault(key, {}).update({
                "feature_type": d.get("feature_type", "continuous"),
                "psi": d.get("psi"),
                "js_divergence": d.get("js_divergence"),
                "wasserstein_distance": d.get("wasserstein_distance"),
                "ks_statistic": d.get("ks_statistic"),
                "ks_p_value": d.get("ks_p_value"),
                "ks_q_value": d.get("ks_q_value"),
                "data_track": d.get("data_track", "NATURAL"),
            })

        if quality_rows:
            for q in quality_rows:
                key = (q.get("window_id", q.get("monitor_window_id", "?")),
                       q.get("feature_name", "?"))
                target = merged.setdefault(key, {})
                target.update({
                    "missing_rate": q.get("missing_rate"),
                    "missing_rate_delta": q.get("missing_rate_delta"),
                    "outlier_rate": q.get("outlier_rate"),
                    "outlier_rate_delta": q.get("outlier_rate_delta"),
                    "default_value_rate": q.get("default_value_rate"),
                    "range_violation_rate": q.get("range_violation_rate"),
                    "unknown_category_rate": q.get("unknown_category_rate"),
                    "dq_score": q.get("dq_score"),
                    "dq_flag": q.get("dq_flag"),
                })

        inserted = 0
        for (window_id, feature_name), fields in merged.items():
            await self.session.execute(
                text("""
                    INSERT INTO monitoring.monitoring_feature_drift
                        (monitoring_run_id, window_id, feature_name, feature_type,
                         psi, js_divergence, wasserstein_distance,
                         ks_statistic, ks_p_value, ks_q_value,
                         missing_rate, missing_rate_delta,
                         outlier_rate, outlier_rate_delta,
                         default_value_rate, range_violation_rate,
                         unknown_category_rate, dq_score, dq_flag,
                         data_track)
                    VALUES (:rid, :wid, :fname, :ftype,
                            :psi, :js, :wd,
                            :ks, :ksp, :ksq,
                            :mr, :mrd,
                            :orr, :ord,
                            :dvr, :rvr,
                            :ucr, :dq, :dqf,
                            :track)
                """),
                {
                    "rid": monitoring_run_id, "wid": window_id,
                    "fname": feature_name, "ftype": fields.get("feature_type", "continuous"),
                    "psi": fields.get("psi"), "js": fields.get("js_divergence"),
                    "wd": fields.get("wasserstein_distance"),
                    "ks": fields.get("ks_statistic"), "ksp": fields.get("ks_p_value"),
                    "ksq": fields.get("ks_q_value"),
                    "mr": fields.get("missing_rate"), "mrd": fields.get("missing_rate_delta"),
                    "orr": fields.get("outlier_rate"), "ord": fields.get("outlier_rate_delta"),
                    "dvr": fields.get("default_value_rate"), "rvr": fields.get("range_violation_rate"),
                    "ucr": fields.get("unknown_category_rate"),
                    "dq": fields.get("dq_score"), "dqf": fields.get("dq_flag"),
                    "track": fields.get("data_track", "NATURAL"),
                },
            )
            inserted += 1
        return inserted

    async def get_feature_drift_by_run(
        self, monitoring_run_id: str, window_id: str | None = None
    ) -> list[dict]:
        """查询一次运行的 per-feature drift 数据。"""
        sql = """
            SELECT * FROM monitoring.monitoring_feature_drift
            WHERE monitoring_run_id = :rid
        """
        params: dict = {"rid": monitoring_run_id}
        if window_id:
            sql += " AND window_id = :wid"
            params["wid"] = window_id
        sql += " ORDER BY window_id, psi DESC NULLS LAST"
        result = await self.session.execute(text(sql), params)
        return [dict(row) for row in result.mappings()]
