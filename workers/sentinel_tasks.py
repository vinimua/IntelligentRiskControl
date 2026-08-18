"""Sentinel 离线训练 Celery 任务。

只接收 dataset_snapshot_id，不接收文件路径。
支持 MinIO s3:// 快照；训练产物写入版本化目录 sentinel/{training_run_id}/。
"""

from __future__ import annotations

import re
import uuid
from io import BytesIO
from pathlib import Path

import pandas as pd
from celery.utils.log import get_task_logger

from workers.app import app
from apps.modelops_api.services.monitoring.sentinel.feature_schema import (
    SENTINEL_FEATURES,
    compute_schema_hash,
)

logger = get_task_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _read_snapshot(storage_uri: str) -> pd.DataFrame:
    """从 storage_uri 加载 Parquet。支持 s3:// MinIO 和本地路径。"""
    from apps.modelops_api.config import settings

    bucket = settings.minio_bucket
    s3_prefix = f"s3://{bucket}/"

    if storage_uri.startswith(s3_prefix):
        from minio import Minio

        object_name = storage_uri.removeprefix(s3_prefix)
        client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        response = client.get_object(bucket, object_name)
        try:
            return pd.read_parquet(BytesIO(response.read()))
        finally:
            response.close()
            response.release_conn()

    # 本地文件（仅开发环境）
    local = Path(storage_uri)
    if not local.is_absolute():
        local = PROJECT_ROOT / storage_uri
    if not local.is_file():
        raise FileNotFoundError(f"Sentinel training dataset not found: {local}")
    return pd.read_parquet(local)


def _validate_snapshot(snapshot: dict, model_id: str) -> list[str]:
    """校验快照身份和内容。返回问题列表，空列表 = 通过。"""
    issues = []
    if snapshot.get("model_id") != model_id:
        issues.append(f"snapshot model_id mismatch: {snapshot.get('model_id')} != {model_id}")
    if snapshot.get("data_track") not in ("SCENARIO", "NATURAL"):
        issues.append(f"unexpected data_track: {snapshot.get('data_track')}")
    return issues


def _inspect_training_readiness(dataset: pd.DataFrame) -> tuple[bool, list[str]]:
    """预检数据集是否可用于训练。返回 (ready, reasons)。"""
    reasons = []

    # 契约字段
    missing = sorted(set(SENTINEL_FEATURES) - set(dataset.columns))
    if missing:
        reasons.append(f"missing contract features: {missing}")

    # 标签检查（只接受精确 0/1）
    if "anomaly_label" not in dataset.columns:
        reasons.append("missing anomaly_label column")
    else:
        raw_labels = set(dataset["anomaly_label"].dropna().unique())
        if raw_labels != {0, 1} and raw_labels != {0.0, 1.0} and raw_labels != {0, 1, 0.0, 1.0}:
            reasons.append(f"labels must be exactly {{0,1}}; got {sorted(raw_labels)}")

    # 分组字段
    if "scenario_instance_id" not in dataset.columns:
        reasons.append("missing scenario_instance_id column")

    # 最小数据量
    if len(dataset) < 30:
        reasons.append(f"dataset too small ({len(dataset)} rows)")

    # 特征不能全空
    present_features = [f for f in SENTINEL_FEATURES if f in dataset.columns]
    if present_features and dataset[present_features].isna().all().all():
        reasons.append("all contract features are NaN")

    # 无穷值检查
    if present_features:
        inf_count = dataset[present_features].isin([float("inf"), float("-inf")]).sum().sum()
        if inf_count > 0:
            reasons.append(f"{int(inf_count)} infinite values in features")

    # UNCERTAIN 拒绝
    if "scenario_acceptance_status" in dataset.columns:
        uncertain = dataset[dataset["scenario_acceptance_status"] == "UNCERTAIN"]
        if len(uncertain) > 0:
            reasons.append(f"{len(uncertain)} UNCERTAIN rows must be dropped before training")

    # group_split 预检
    if not reasons:
        try:
            from apps.modelops_api.services.monitoring.sentinel.train import group_split
            splits = group_split(dataset)
            for name, df in splits.items():
                if df.empty:
                    reasons.append(f"split '{name}' is empty")
                else:
                    split_labels = set(df["anomaly_label"].astype(int).unique())
                    if split_labels != {0, 1}:
                        reasons.append(f"split '{name}' labels {sorted(split_labels)}, need {{0,1}}")
        except ValueError as e:
            reasons.append(f"group_split: {e}")

    # schema hash
    expected_hash = compute_schema_hash()
    snapshot_hash = dataset.attrs.get("feature_schema_hash") or ""
    if snapshot_hash and snapshot_hash != expected_hash:
        reasons.append(f"schema hash mismatch: snapshot={snapshot_hash[:16]} expected={expected_hash[:16]}")

    return len(reasons) == 0, reasons


@app.task(
    bind=True,
    name="workers.sentinel_tasks.train_sentinel_model",
    max_retries=0,
)
def train_sentinel_model(
    self,
    *,
    model_id: str,
    champion_version: str,
    dataset_snapshot_id: str,
    training_run_id: str | None = None,
) -> dict:
    """Celery 任务：从 dataset_snapshot_id 加载 → 训练 → 门禁 → 发布。

    Args:
        model_id: 模型编码。
        champion_version: Champion 版本。
        dataset_snapshot_id: model_registry.dataset_snapshots 中的快照 ID。
        training_run_id: 训练运行 ID，不传则 UUID4。
    """
    # 路径安全
    if not SAFE_NAME.match(model_id):
        raise ValueError(f"Invalid model_id: {model_id}")
    if not SAFE_NAME.match(champion_version):
        raise ValueError(f"Invalid champion_version: {champion_version}")
    run_id = str(uuid.UUID(training_run_id)) if training_run_id else str(uuid.uuid4())

    # ① 从 SnapshotRepo 加载快照
    import asyncio
    from apps.modelops_api.database import async_session
    from apps.modelops_api.repositories.snapshot_repo import SnapshotRepo

    async def _load():
        async with async_session() as s:
            repo = SnapshotRepo(s)
            snapshot = await repo.get_by_id(dataset_snapshot_id)
            if snapshot is None:
                raise ValueError(f"Snapshot {dataset_snapshot_id} not found")
            return dict(snapshot)

    snapshot = asyncio.run(_load())

    # ② 身份校验
    issues = _validate_snapshot(snapshot, model_id)
    if issues:
        logger.error("sentinel_snapshot_validation_failed", issues=issues)
        return {"training_run_id": run_id, "status": "REJECTED", "reasons": issues}

    # ③ 加载 Parquet（支持 s3:// MinIO）
    storage_uri = snapshot.get("storage_uri") or snapshot.get("artifact_uri")
    if not storage_uri:
        return {"training_run_id": run_id, "status": "REJECTED", "reasons": ["no storage_uri"]}
    dataset = _read_snapshot(storage_uri)

    # ④ 就绪检查
    ready, reasons = _inspect_training_readiness(dataset)
    if not ready:
        logger.error("sentinel_training_not_ready", reasons=reasons)
        return {"training_run_id": run_id, "status": "SKIPPED", "reasons": reasons}

    # ⑤ 训练
    from apps.modelops_api.services.monitoring.sentinel.train import train_sentinel

    artifact_dir = (
        PROJECT_ROOT / "assets" / "champion_models" / model_id / champion_version
    )
    artifact_dir = artifact_dir.resolve()
    training_root = (PROJECT_ROOT / "assets" / "champion_models").resolve()
    if training_root not in artifact_dir.parents and artifact_dir != training_root:
        raise ValueError(f"artifact_dir {artifact_dir} outside training root")

    _, metrics, _ = train_sentinel(
        dataset=dataset,
        artifact_dir=artifact_dir,
        training_run_id=run_id,
        sentinel_version="sentinel_lgbm_v2",
    )

    logger.info(
        "sentinel_training_completed model=%s version=%s published=%s run=%s",
        model_id, champion_version, metrics.get("published"), run_id,
    )

    return {
        "training_run_id": run_id,
        "status": "SUCCEEDED",
        "published": metrics.get("published"),
        "qualified": metrics.get("qualified"),
        "active_model_changed": metrics.get("active_model_changed"),
        "test_recall": (metrics.get("test") or {}).get("recall"),
        "test_fpr": (metrics.get("test") or {}).get("fpr"),
        "run_dir": str(artifact_dir / "sentinel" / run_id),
    }
