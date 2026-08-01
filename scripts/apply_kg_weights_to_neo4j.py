"""Apply calibrated KG relation weights from PostgreSQL snapshots to Neo4j."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from pathlib import Path

import psycopg
from neo4j import AsyncGraphDatabase
from psycopg.rows import dict_row

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apps.modelops_api.config import settings


def _dsn() -> str:
    return settings.database_url_sync.replace("postgresql+psycopg://", "postgresql://")


def _load_snapshots(calibration_run_id: str | None, weight_version: str | None) -> list[dict]:
    with psycopg.connect(_dsn(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            if calibration_run_id:
                cur.execute(
                    """
                    SELECT * FROM knowledge.kg_relation_weight_snapshots
                    WHERE calibration_run_id = %s
                    ORDER BY relation_key
                    """,
                    (calibration_run_id,),
                )
            elif weight_version:
                cur.execute(
                    """
                    SELECT DISTINCT ON (relation_key) *
                    FROM knowledge.kg_relation_weight_snapshots
                    WHERE weight_version = %s
                    ORDER BY relation_key, created_at DESC
                    """,
                    (weight_version,),
                )
            else:
                cur.execute(
                    """
                    SELECT DISTINCT ON (relation_key) *
                    FROM knowledge.kg_relation_weight_snapshots
                    ORDER BY relation_key, created_at DESC
                    """
                )
            return list(cur.fetchall())


def _split_relation_key(relation_key: str) -> tuple[str, str, str]:
    parts = relation_key.split("|")
    if len(parts) != 3:
        raise ValueError(f"Invalid relation_key: {relation_key}")
    return parts[0], parts[1], parts[2]


def _supported_snapshots(snapshots: list[dict]) -> list[dict]:
    supported: list[dict] = []
    for snapshot in snapshots:
        _, relation_type, _ = _split_relation_key(snapshot["relation_key"])
        if relation_type in _RELATION_TEMPLATES:
            supported.append(snapshot)
    return supported


def _job_key(snapshot: dict) -> tuple[str, str, str]:
    _, relation_type, _ = _split_relation_key(snapshot["relation_key"])
    return (
        str(snapshot["calibration_run_id"]),
        relation_type,
        str(snapshot["weight_version"]),
    )


def _create_sync_jobs(snapshots: list[dict]) -> dict[tuple[str, str, str], str]:
    grouped: dict[tuple[str, str, str], int] = defaultdict(int)
    for snapshot in snapshots:
        grouped[_job_key(snapshot)] += 1

    jobs: dict[tuple[str, str, str], str] = {}
    if not grouped:
        return jobs

    with psycopg.connect(_dsn(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            for (calibration_run_id, relation_type, weight_version), count in grouped.items():
                idempotency_key = (
                    f"kg-weight-sync:{calibration_run_id}:{relation_type}:{weight_version}"
                )
                cur.execute(
                    """
                    INSERT INTO knowledge.kg_sync_jobs (
                        calibration_run_id, idempotency_key, relation_type,
                        status, snapshot_count, applied_count, weight_version,
                        applied_to_neo4j, started_at
                    )
                    VALUES (
                        %s, %s, %s,
                        'RUNNING', %s, 0, %s,
                        false, NOW()
                    )
                    ON CONFLICT (idempotency_key)
                    DO UPDATE SET
                        status = 'RUNNING',
                        snapshot_count = EXCLUDED.snapshot_count,
                        applied_count = 0,
                        error_message = NULL,
                        applied_to_neo4j = false,
                        started_at = NOW(),
                        completed_at = NULL
                    RETURNING sync_job_id
                    """,
                    (
                        calibration_run_id,
                        idempotency_key,
                        relation_type,
                        count,
                        weight_version,
                    ),
                )
                jobs[(calibration_run_id, relation_type, weight_version)] = str(
                    cur.fetchone()["sync_job_id"]
                )
        conn.commit()
    return jobs


def _mark_sync_jobs(
    jobs: dict[tuple[str, str, str], str],
    *,
    status: str,
    applied_counts: dict[tuple[str, str, str], int] | None = None,
    error_message: str | None = None,
) -> None:
    if not jobs:
        return

    applied_counts = applied_counts or {}
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            for key, sync_job_id in jobs.items():
                applied_count = applied_counts.get(key, 0)
                cur.execute(
                    """
                    UPDATE knowledge.kg_sync_jobs
                    SET status = %s,
                        applied_count = %s,
                        error_message = %s,
                        applied_to_neo4j = %s,
                        neo4j_applied_at = CASE WHEN %s THEN NOW() ELSE neo4j_applied_at END,
                        completed_at = NOW()
                    WHERE sync_job_id = %s
                    """,
                    (
                        status,
                        applied_count,
                        error_message,
                        status == "SUCCEEDED",
                        status == "SUCCEEDED",
                        sync_job_id,
                    ),
                )
        conn.commit()


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


async def apply(calibration_run_id: str | None, weight_version: str | None) -> int:
    snapshots = _supported_snapshots(_load_snapshots(calibration_run_id, weight_version))
    jobs = _create_sync_jobs(snapshots)
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    applied = 0
    applied_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    try:
        async with driver.session(database="neo4j") as session:
            for snapshot in snapshots:
                source_code, relation_type, target_code = _split_relation_key(
                    snapshot["relation_key"]
                )
                template = _RELATION_TEMPLATES[relation_type]

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

                await session.run(
                    cypher,
                    source=source_code,
                    target=target_code,
                    relation_key=snapshot["relation_key"],
                    new_effective_weight=float(snapshot["new_effective_weight"]),
                    confidence_lower_bound=float(snapshot["confidence_lower_bound"]),
                    confidence_upper_bound=float(snapshot["confidence_upper_bound"]),
                    evidence_case_count=int(snapshot["evidence_case_count"]),
                    natural_case_count=int(snapshot["natural_case_count"]),
                    scenario_case_count=int(snapshot["scenario_case_count"]),
                    support_count=int(snapshot["support_count"]),
                    against_count=int(snapshot["against_count"]),
                    neutral_count=int(snapshot["neutral_count"]),
                    support_strength=float(snapshot["support_strength"]),
                    against_strength=float(snapshot["against_strength"]),
                    weight_version=snapshot["weight_version"],
                )
                applied += 1
                applied_counts[_job_key(snapshot)] += 1
    except Exception as exc:
        _mark_sync_jobs(jobs, status="FAILED", applied_counts=applied_counts, error_message=str(exc))
        raise
    finally:
        await driver.close()

    if snapshots:
        with psycopg.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                ids = [snapshot["snapshot_id"] for snapshot in snapshots]
                cur.execute(
                    """
                    UPDATE knowledge.kg_relation_weight_snapshots
                    SET applied_to_neo4j = true
                    WHERE snapshot_id = ANY(%s)
                    """,
                    (ids,),
                )
            conn.commit()

    _mark_sync_jobs(jobs, status="SUCCEEDED", applied_counts=applied_counts)
    return applied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-run-id")
    parser.add_argument("--weight-version")
    args = parser.parse_args()

    applied = asyncio.run(apply(args.calibration_run_id, args.weight_version))
    print(f"Applied KG weights to Neo4j: relations={applied}")


if __name__ == "__main__":
    main()
