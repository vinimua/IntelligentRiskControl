"""KG 校准服务 — 异步封装 Beta-Binomial 校准 + Neo4j 同步逻辑。

从 scripts/run_kg_calibration.py 和 scripts/apply_kg_weights_to_neo4j.py 提取核心逻辑，
适配 FastAPI 异步 Session。
"""

from __future__ import annotations

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..repositories.kg_repo import KgCalibrationRepo

logger = structlog.get_logger(__name__)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _bayesian_shrinkage(
    support_count: int,
    against_count: int,
    neutral_count: int,
    support_strength: float,
    against_strength: float,
    *,
    prior_alpha: float = 2.0,
    prior_beta: float = 8.0,
) -> dict:
    """贝叶斯 Beta-Binomial 收缩。

    先验 Beta(2, 8) → 均值 0.20（弱有效先验）。
    后验 Beta(α + support_strength, β + against_strength)。
    """
    total = support_count + against_count + neutral_count
    if total <= 0:
        return {
            "new_weight": round(prior_alpha / (prior_alpha + prior_beta), 4),
            "alpha_post": prior_alpha,
            "beta_post": prior_beta,
            "confidence_lower": 0.0,
            "confidence_upper": 1.0,
        }

    alpha_post = prior_alpha + support_strength
    beta_post = prior_beta + against_strength
    posterior_mean = alpha_post / (alpha_post + beta_post)

    posterior_var = (alpha_post * beta_post) / (
        (alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1)
    )
    posterior_std = posterior_var ** 0.5
    confidence_lower = max(0.0, posterior_mean - 1.96 * posterior_std)
    confidence_upper = min(1.0, posterior_mean + 1.96 * posterior_std)

    new_weight = round(_clamp(posterior_mean, 0.03, 0.85), 4)

    return {
        "new_weight": new_weight,
        "alpha_post": round(alpha_post, 4),
        "beta_post": round(beta_post, 4),
        "confidence_lower": round(confidence_lower, 4),
        "confidence_upper": round(confidence_upper, 4),
    }


class KgCalibrationService:
    """异步 KG 校准服务。"""

    def __init__(self, session: AsyncSession):
        self.session = session
        self._repo = KgCalibrationRepo(session)

    async def run_calibration(
        self,
        data_track: str,
        rule_version: str,
        weight_version: str,
    ) -> str:
        """执行一次完整校准：聚合观测 → 贝叶斯收缩 → 写入快照。"""
        data_track = data_track.upper()
        if data_track not in {"NATURAL", "SCENARIO"}:
            raise ValueError("data_track must be NATURAL or SCENARIO")

        # 1. 创建校准运行
        run_id = await self._repo.create_run(data_track, rule_version, weight_version)

        # 2. 聚合观测
        result = await self.session.execute(
            text("""
                SELECT
                    relation_key,
                    COUNT(*) AS evidence_case_count,
                    COUNT(*) FILTER (WHERE data_track = 'NATURAL') AS natural_case_count,
                    COUNT(*) FILTER (WHERE data_track = 'SCENARIO') AS scenario_case_count,
                    COUNT(*) FILTER (WHERE direction = 'SUPPORT') AS support_count,
                    COUNT(*) FILTER (WHERE direction = 'AGAINST') AS against_count,
                    COUNT(*) FILTER (WHERE direction = 'NEUTRAL') AS neutral_count,
                    COALESCE(SUM(weighted_strength) FILTER (WHERE direction = 'SUPPORT'), 0.0) AS support_strength,
                    COALESCE(SUM(weighted_strength) FILTER (WHERE direction = 'AGAINST'), 0.0) AS against_strength
                FROM knowledge.kg_relation_observations
                WHERE data_track = :track
                GROUP BY relation_key
                ORDER BY relation_key
            """),
            {"track": data_track},
        )
        rows = result.mappings().all()

        # 3. 贝叶斯收缩 → 快照
        for row in rows:
            bayes = _bayesian_shrinkage(
                support_count=int(row["support_count"]),
                against_count=int(row["against_count"]),
                neutral_count=int(row["neutral_count"]),
                support_strength=float(row["support_strength"]),
                against_strength=float(row["against_strength"]),
            )
            await self._repo.write_snapshot({
                "rid": run_id,
                "rk": row["relation_key"],
                "old_w": None,
                "new_w": bayes["new_weight"],
                "clb": bayes["confidence_lower"],
                "cub": bayes["confidence_upper"],
                "ec": row["evidence_case_count"],
                "nc": row["natural_case_count"],
                "sc": row["scenario_case_count"],
                "sup": row["support_count"],
                "ag": row["against_count"],
                "neu": row["neutral_count"],
                "sup_str": float(row["support_strength"]),
                "ag_str": float(row["against_strength"]),
                "wv": weight_version,
                "det": {
                    "rule": "BETA_BINOMIAL_V2",
                    "prior_alpha": 2.0,
                    "prior_beta": 8.0,
                    "posterior_alpha": bayes["alpha_post"],
                    "posterior_beta": bayes["beta_post"],
                },
            })

        # 4. 标记完成
        observation_count_result = await self.session.execute(
            text("""
                SELECT COUNT(*) AS cnt FROM knowledge.kg_relation_observations
                WHERE data_track = :track
            """),
            {"track": data_track},
        )
        observation_count = observation_count_result.scalar()
        await self._repo.mark_run_completed(run_id, len(rows), observation_count)

        logger.info(
            "kg_calibration_completed",
            calibration_run_id=run_id,
            relation_count=len(rows),
            observation_count=observation_count,
        )
        return run_id

    async def apply_to_neo4j(
        self,
        calibration_run_id: str,
        weight_version: str | None = None,
    ) -> dict:
        """将权重快照同步到 Neo4j 并记录 sync_job。"""
        from ..config import settings
        from neo4j import AsyncGraphDatabase

        # 1. 加载快照
        if calibration_run_id:
            snap_result = await self.session.execute(
                text("""
                    SELECT * FROM knowledge.kg_relation_weight_snapshots
                    WHERE calibration_run_id = :rid
                    ORDER BY relation_key
                """),
                {"rid": calibration_run_id},
            )
        elif weight_version:
            snap_result = await self.session.execute(
                text("""
                    SELECT DISTINCT ON (relation_key) *
                    FROM knowledge.kg_relation_weight_snapshots
                    WHERE weight_version = :wv
                    ORDER BY relation_key, created_at DESC
                """),
                {"wv": weight_version},
            )
        else:
            snap_result = await self.session.execute(
                text("""
                    SELECT DISTINCT ON (relation_key) *
                    FROM knowledge.kg_relation_weight_snapshots
                    ORDER BY relation_key, created_at DESC
                """),
            )

        snapshots = [dict(r) for r in snap_result.mappings()]

        if not snapshots:
            return {"applied": 0, "sync_jobs": []}

        # 2. 按 relation_type 分组
        from collections import defaultdict
        groups: dict[str, list[dict]] = defaultdict(list)
        for snap in snapshots:
            rk = snap["relation_key"]
            parts = rk.split("|")
            if len(parts) != 3:
                continue
            relation_type = parts[1]
            if relation_type in {"INDICATES", "RECOMMENDS", "MITIGATES"}:
                groups[relation_type].append(snap)

        # 3. 连接 Neo4j 并同步
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

        _RELATION_TEMPLATES = {
            "INDICATES": {
                "source_label": "Alert",
                "target_label": "RootCause",
                "source_type": "Alert",
                "target_type": "RootCause",
                "source_ns": "DIAGNOSIS",
                "target_ns": "DIAGNOSIS",
            },
            "RECOMMENDS": {
                "source_label": "RootCause",
                "target_label": "Strategy",
                "source_type": "RootCause",
                "target_type": "Strategy",
                "source_ns": "DIAGNOSIS",
                "target_ns": "ITERATION",
            },
            "MITIGATES": {
                "source_label": "Strategy",
                "target_label": "RootCause",
                "source_type": "Strategy",
                "target_type": "RootCause",
                "source_ns": "ITERATION",
                "target_ns": "DIAGNOSIS",
            },
        }

        sync_jobs = []
        wv = weight_version or snapshots[0].get("weight_version", "unknown")

        async with driver.session(database="neo4j") as neo_session:
            for relation_type, group_snaps in groups.items():
                template = _RELATION_TEMPLATES.get(relation_type)
                if not template:
                    continue

                idempotency_key = f"kg-weight-sync:{calibration_run_id}:{relation_type}:{wv}"

                # 创建 sync_job
                job_result = await self.session.execute(
                    text("""
                        INSERT INTO knowledge.kg_sync_jobs (
                            calibration_run_id, idempotency_key, relation_type,
                            status, snapshot_count, weight_version, started_at
                        ) VALUES (:rid, :ikey, :rtype, 'RUNNING', :cnt, :wv, NOW())
                        ON CONFLICT (idempotency_key) DO UPDATE SET
                            status = 'RUNNING',
                            started_at = NOW(),
                            snapshot_count = :cnt
                        RETURNING sync_job_id
                    """),
                    {
                        "rid": calibration_run_id,
                        "ikey": idempotency_key,
                        "rtype": relation_type,
                        "cnt": len(group_snaps),
                        "wv": wv,
                    },
                )
                job_row = job_result.mappings().first()
                sync_job_id = str(job_row["sync_job_id"]) if job_row else None

                try:
                    applied = 0
                    for snap in group_snaps:
                        rk = snap["relation_key"]
                        parts = rk.split("|")
                        source_code, target_code = parts[0], parts[2]

                        cypher = f"""
                            MERGE (s:{template["source_label"]} {{entity_code: $source}})
                            SET s.entity_type  = '{template["source_type"]}',
                                s.namespace    = '{template["source_ns"]}',
                                s.enabled      = true
                            MERGE (t:{template["target_label"]} {{entity_code: $target}})
                            SET t.entity_type  = '{template["target_type"]}',
                                t.namespace    = '{template["target_ns"]}',
                                t.enabled      = true
                            MERGE (s)-[rel:{relation_type}]->(t)
                            SET rel.relation_key            = $relation_key,
                                rel.relation_type           = '{relation_type}',
                                rel.source_entity_code      = $source,
                                rel.target_entity_code      = $target,
                                rel.effective_weight        = $new_effective_weight,
                                rel.confidence_lower_bound  = $confidence_lower_bound,
                                rel.confidence_upper_bound  = $confidence_upper_bound,
                                rel.evidence_case_count     = $evidence_case_count,
                                rel.natural_case_count      = $natural_case_count,
                                rel.scenario_case_count     = $scenario_case_count,
                                rel.support_count           = $support_count,
                                rel.against_count           = $against_count,
                                rel.neutral_count           = $neutral_count,
                                rel.support_strength        = $support_strength,
                                rel.against_strength        = $against_strength,
                                rel.weight_version          = $weight_version,
                                rel.last_calibrated_at      = datetime(),
                                rel.enabled                 = true
                        """

                        await neo_session.run(
                            cypher,
                            source=source_code,
                            target=target_code,
                            relation_key=rk,
                            new_effective_weight=float(snap["new_effective_weight"]),
                            confidence_lower_bound=float(snap["confidence_lower_bound"]),
                            confidence_upper_bound=float(snap["confidence_upper_bound"]),
                            evidence_case_count=int(snap["evidence_case_count"]),
                            natural_case_count=int(snap["natural_case_count"]),
                            scenario_case_count=int(snap["scenario_case_count"]),
                            support_count=int(snap["support_count"]),
                            against_count=int(snap["against_count"]),
                            neutral_count=int(snap["neutral_count"]),
                            support_strength=float(snap["support_strength"]),
                            against_strength=float(snap["against_strength"]),
                            weight_version=snap["weight_version"],
                        )
                        applied += 1

                    # 标记快照为已同步
                    snap_ids = [s["snapshot_id"] for s in group_snaps]
                    await self.session.execute(
                        text("""
                            UPDATE knowledge.kg_relation_weight_snapshots
                            SET applied_to_neo4j = true
                            WHERE snapshot_id = ANY(:ids)
                        """),
                        {"ids": snap_ids},
                    )

                    # 更新 sync_job 为成功
                    await self.session.execute(
                        text("""
                            UPDATE knowledge.kg_sync_jobs
                            SET status = 'SUCCEEDED',
                                applied_count = :cnt,
                                applied_to_neo4j = true,
                                neo4j_applied_at = NOW(),
                                completed_at = NOW()
                            WHERE sync_job_id = :sid
                        """),
                        {"sid": sync_job_id, "cnt": applied},
                    )

                    sync_jobs.append({
                        "sync_job_id": sync_job_id,
                        "relation_type": relation_type,
                        "status": "SUCCEEDED",
                        "applied_count": applied,
                    })

                except Exception as exc:
                    logger.error(
                        "neo4j_sync_failed",
                        relation_type=relation_type,
                        error=str(exc),
                    )
                    if sync_job_id:
                        await self.session.execute(
                            text("""
                                UPDATE knowledge.kg_sync_jobs
                                SET status = 'FAILED',
                                    error_message = :err,
                                    completed_at = NOW()
                                WHERE sync_job_id = :sid
                            """),
                            {"sid": sync_job_id, "err": str(exc)[:1000]},
                        )
                    sync_jobs.append({
                        "sync_job_id": sync_job_id,
                        "relation_type": relation_type,
                        "status": "FAILED",
                        "error": str(exc),
                    })

        await driver.close()

        total_applied = sum(j.get("applied_count", 0) for j in sync_jobs)
        return {"applied": total_applied, "sync_jobs": sync_jobs}
