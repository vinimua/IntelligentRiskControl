"""Apply calibrated KG relation weights from PostgreSQL snapshots to Neo4j."""

from __future__ import annotations

import argparse
import asyncio
import sys
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


async def apply(calibration_run_id: str | None, weight_version: str | None) -> int:
    snapshots = _load_snapshots(calibration_run_id, weight_version)
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    applied = 0
    async with driver.session(database="neo4j") as session:
        for snapshot in snapshots:
            alert_code, relation_type, root_cause = _split_relation_key(snapshot["relation_key"])
            if relation_type != "INDICATES":
                continue

            await session.run(
                """
                MERGE (a:Alert {entity_code: $alert_code})
                SET a.entity_type = 'Alert',
                    a.namespace = 'DIAGNOSIS',
                    a.enabled = true
                MERGE (r:RootCause {entity_code: $root_cause})
                SET r.entity_type = 'RootCause',
                    r.namespace = 'DIAGNOSIS',
                    r.enabled = true
                MERGE (a)-[rel:INDICATES]->(r)
                SET rel.relation_key = $relation_key,
                    rel.source_entity_code = $alert_code,
                    rel.relation_type = 'INDICATES',
                    rel.target_entity_code = $root_cause,
                    rel.effective_weight = $new_effective_weight,
                    rel.confidence_lower_bound = $confidence_lower_bound,
                    rel.confidence_upper_bound = $confidence_upper_bound,
                    rel.evidence_case_count = $evidence_case_count,
                    rel.natural_case_count = $natural_case_count,
                    rel.scenario_case_count = $scenario_case_count,
                    rel.support_count = $support_count,
                    rel.against_count = $against_count,
                    rel.neutral_count = $neutral_count,
                    rel.support_strength = $support_strength,
                    rel.against_strength = $against_strength,
                    rel.weight_version = $weight_version,
                    rel.last_calibrated_at = datetime(),
                    rel.enabled = true
                """,
                alert_code=alert_code,
                root_cause=root_cause,
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
