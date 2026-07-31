"""Ingest scenario observations JSON into knowledge.kg_relation_observations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apps.modelops_api.config import settings
from apps.modelops_api.services.knowledge_observation_mapper import (
    enrich_scenario_observation,
    validate_mapped_observation,
)


def _source_record_id(observation: dict[str, Any]) -> str:
    return (
        f"scenario:{observation['scenario']}:"
        f"{observation['model_id']}:"
        f"{observation['source_entity']}:"
        f"{observation['target_entity']}:"
        f"{observation['direction']}"
    )


def _prepare(raw: dict[str, Any]) -> dict[str, Any]:
    observation = raw
    if "mapped_relation_key" not in observation:
        observation = enrich_scenario_observation(observation)
    validate_mapped_observation(observation)

    quality_weight = float(observation.get("quality_weight", 1.0))
    evidence_score = observation.get("evidence_score")
    weighted_strength = None
    if evidence_score is not None:
        weighted_strength = float(evidence_score) * quality_weight

    return {
        "relation_key": observation["mapped_relation_key"],
        "source_domain": "SCENARIO_ANALYSIS",
        "source_record_id": _source_record_id(observation),
        "direction": observation["direction"],
        "evidence_score": evidence_score,
        "quality_weight": quality_weight,
        "weighted_strength": weighted_strength,
        "data_track": observation["data_track"],
        "evidence_detail": observation,
    }


def ingest(input_path: Path) -> tuple[int, int]:
    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Scenario observations JSON must contain a list")

    inserted = 0
    updated = 0
    with psycopg.connect(settings.database_url_sync.replace("postgresql+psycopg://", "postgresql://")) as conn:
        with conn.cursor() as cur:
            for raw in records:
                row = _prepare(raw)
                cur.execute(
                    """
                    INSERT INTO knowledge.kg_relation_observations (
                        relation_key, source_domain, source_record_id,
                        direction, evidence_score, quality_weight,
                        weighted_strength, data_track, evidence_detail
                    )
                    VALUES (
                        %(relation_key)s, %(source_domain)s, %(source_record_id)s,
                        %(direction)s, %(evidence_score)s, %(quality_weight)s,
                        %(weighted_strength)s, %(data_track)s, %(evidence_detail)s
                    )
                    ON CONFLICT (relation_key, source_domain, source_record_id)
                    DO UPDATE SET
                        direction = EXCLUDED.direction,
                        evidence_score = EXCLUDED.evidence_score,
                        quality_weight = EXCLUDED.quality_weight,
                        weighted_strength = EXCLUDED.weighted_strength,
                        data_track = EXCLUDED.data_track,
                        evidence_detail = EXCLUDED.evidence_detail
                    RETURNING (xmax = 0) AS inserted
                    """,
                    {**row, "evidence_detail": Jsonb(row["evidence_detail"])},
                )
                was_inserted = bool(cur.fetchone()[0])
                inserted += int(was_inserted)
                updated += int(not was_inserted)
        conn.commit()

    return inserted, updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "tmp" / "scenario_observations.json"),
        help="Path to scenario_observations.json",
    )
    args = parser.parse_args()

    inserted, updated = ingest(Path(args.input))
    print(f"Scenario observations ingested: inserted={inserted}, updated={updated}")


if __name__ == "__main__":
    main()
