"""Build formal W0-W4 windows from the complete competition datasets.

This command never downsamples. It removes ``id_card``, creates deterministic
privacy-safe ``sample_id`` values, writes Parquet atomically, and updates the
manifest with physical checksums and observed label statistics.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_ROOT = PROJECT_ROOT / "assets"
DATA_CONFIG = ASSETS_ROOT / "configs" / "data.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_ids(
    namespace: str, id_cards: pd.Series, apply_times: pd.Series
) -> list[str]:
    return [
        hashlib.sha256(
            f"{namespace}|{id_card}|{apply_time.isoformat()}".encode("utf-8")
        ).hexdigest()
        for id_card, apply_time in zip(id_cards.astype(str), apply_times)
    ]


def main() -> int:
    config = yaml.safe_load(DATA_CONFIG.read_text(encoding="utf-8"))
    manifest_path = ASSETS_ROOT / config["window_manifest_uri"]
    manifest = pd.read_csv(manifest_path, dtype={"window_id": str})
    namespace = str(config["sample_id_namespace"])
    raw_cache: dict[str, tuple[pd.DataFrame, Path]] = {}
    total_written_by_source: dict[str, int] = {}

    for row_index, row in manifest.iterrows():
        window_id = str(row["window_id"])
        source_name = str(config["windows"][window_id]["source"]).upper()
        manifest.loc[row_index, "data_role"] = str(
            config["windows"][window_id]["data_role"]
        )
        raw_key = "raw_train_uri" if source_name == "TRAIN" else "raw_test_uri"
        if raw_key not in raw_cache:
            raw_path = ASSETS_ROOT / config[raw_key]
            raw = pd.read_csv(raw_path)
            raw["apply_time"] = pd.to_datetime(raw["apply_time"], errors="raise")
            if "id_card" not in raw.columns:
                raise ValueError(f"{raw_path} is missing id_card")
            raw_cache[raw_key] = (raw, raw_path)
        raw, raw_path = raw_cache[raw_key]

        start = pd.Timestamp(row["start_date"])
        end = pd.Timestamp(row["end_date"])
        window = raw.loc[
            (raw["apply_time"] >= start) & (raw["apply_time"] < end)
        ].copy()
        if window.empty:
            raise ValueError(f"{window_id} contains no rows")

        window.insert(
            0,
            "sample_id",
            _sample_ids(namespace, window["id_card"], window["apply_time"]),
        )
        window.drop(columns=["id_card"], inplace=True)
        if not window["sample_id"].is_unique:
            raise ValueError(f"{window_id} contains duplicate sample_id values")
        window.sort_values(["apply_time", "sample_id"], inplace=True)
        window.reset_index(drop=True, inplace=True)

        output_path = ASSETS_ROOT / Path(str(row["data_uri"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(".tmp.parquet")
        window.to_parquet(temporary_path, index=False)
        temporary_path.replace(output_path)

        bad_count = int(window["is_bad"].sum())
        manifest.loc[row_index, "row_count"] = len(window)
        manifest.loc[row_index, "source_checksum"] = _sha256(raw_path)
        manifest.loc[row_index, "data_checksum"] = _sha256(output_path)
        manifest.loc[row_index, "bad_count"] = bad_count
        manifest.loc[row_index, "bad_rate"] = float(window["is_bad"].mean())
        manifest.loc[row_index, "population_fraction"] = 1.0
        manifest.loc[row_index, "sampling_mode"] = "FULL_POPULATION"
        manifest.loc[row_index, "created_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        total_written_by_source[raw_key] = (
            total_written_by_source.get(raw_key, 0) + len(window)
        )
        print(
            f"{window_id}: rows={len(window)} bad={bad_count} "
            f"bad_rate={window['is_bad'].mean():.6f}"
        )

    for raw_key, (raw, _) in raw_cache.items():
        written = total_written_by_source.get(raw_key, 0)
        if written != len(raw):
            raise ValueError(
                f"{raw_key} coverage mismatch: raw={len(raw)} windows={written}"
            )

    manifest["row_count"] = pd.to_numeric(
        manifest["row_count"], errors="raise"
    ).astype("int64")
    manifest["bad_count"] = pd.to_numeric(
        manifest["bad_count"], errors="raise"
    ).astype("int64")
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
