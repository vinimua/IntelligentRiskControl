"""Normalize Neo4j KG naming to the original design.

Original design:
Metric -[:TRIGGERS]-> Alert

This script keeps existing data but adds the canonical Alert label and TRIGGERS
relationships when older AlertType / BREACHES_THRESHOLD artifacts exist.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from neo4j import AsyncGraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apps.modelops_api.config import settings


async def normalize() -> None:
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    async with driver.session(database="neo4j") as session:
        result = await session.run(
            """
            MATCH (a:AlertType)
            SET a:Alert,
                a.entity_type = 'Alert'
            REMOVE a:AlertType
            RETURN count(a) AS count
            """
        )
        alert_count = (await result.single())["count"]

        result = await session.run(
            """
            MATCH (m:Metric)-[old:BREACHES_THRESHOLD]->(a:Alert)
            MERGE (m)-[r:TRIGGERS]->(a)
            SET r.relation_key = coalesce(old.relation_key, m.entity_code + '|TRIGGERS|' + a.entity_code),
                r.relation_type = 'TRIGGERS',
                r.enabled = coalesce(old.enabled, true),
                r.initial_prior_weight = coalesce(old.initial_prior_weight, old.effective_weight, 1.0),
                r.prior_strength = coalesce(old.prior_strength, 1.0),
                r.effective_weight = coalesce(old.effective_weight, 1.0),
                r.confidence_lower_bound = coalesce(old.confidence_lower_bound, 0.0),
                r.confidence_upper_bound = coalesce(old.confidence_upper_bound, 0.0),
                r.evidence_case_count = coalesce(old.evidence_case_count, 0),
                r.weight_version = coalesce(old.weight_version, 'KG_NAMING_NORMALIZED_V1')
            DELETE old
            RETURN count(r) AS count
            """
        )
        rel_count = (await result.single())["count"]

    await driver.close()
    print(f"KG naming normalized: Alert nodes={alert_count}, TRIGGERS relations={rel_count}")


if __name__ == "__main__":
    asyncio.run(normalize())
