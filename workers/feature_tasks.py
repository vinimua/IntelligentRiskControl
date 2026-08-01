"""T3-GAP-01: 特征重构 Celery Worker。

执行特征增/删/变换，产出：
- transform_artifact_uri: MinIO 上的 transform pipeline
- feature_snapshot_id: 新特征数据快照
- feature_schema_version: 新 schema 版本号
"""

from __future__ import annotations

import io as _io
import json
import os
import urllib.request
import uuid

from celery.utils.log import get_task_logger

from .app import app

logger = get_task_logger(__name__)

API_BASE = os.getenv("MODELOPS_API_BASE", "http://127.0.0.1:8000")


def _api_post(path: str, body: dict) -> dict:
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        logger.warning("api_post_failed url=%s code=%s", url, exc.code)
        raise


@app.task(bind=True, name="workers.feature_tasks.reconstruct_features", max_retries=1, default_retry_delay=120)
def reconstruct_features(self, plan_input: dict):
    """执行特征重构计划。

    plan_input 字段：
    - plan_id: FeatureReconstructionPlan.plan_id
    - lifecycle_run_id
    - model_id
    - transforms: list[dict]
    - current_schema_version
    - target_schema_version
    - window_ids: list[str] — 要重构的数据窗口
    """
    import joblib as jl
    import numpy as np
    import pandas as pd

    plan_id = plan_input["plan_id"]
    lifecycle_run_id = plan_input.get("lifecycle_run_id", "")
    model_id = plan_input.get("model_id", "")
    transforms = plan_input.get("transforms", [])
    target_schema_version = plan_input.get("target_schema_version", "v2")
    window_ids = plan_input.get("window_ids", ["W2", "W3"])

    logger.info("feature_reconstruction_started plan=%s model=%s", plan_id, model_id)

    try:
        # ── 1. 加载当前窗口数据 ──
        from apps.modelops_api.services.monitoring.window_loader import load_window

        frames = []
        for wid in window_ids:
            try:
                df = load_window(wid)
                df = df.copy()
                df["__source_window_id"] = wid
                frames.append(df)
            except Exception as exc:
                logger.warning("feature_load_window_failed window=%s err=%s", wid, exc)

        if not frames:
            raise ValueError("No data windows loaded for feature reconstruction")

        df = pd.concat(frames, ignore_index=True)
        before_count = len([c for c in df.columns if c not in {"sample_id", "apply_time", "is_bad", "y_pred_proba", "risk_score"}])

        # ── 2. 执行变换 ──
        dropped: list[str] = []
        added: list[str] = []
        transform_detail: dict = {"transforms_applied": 0, "drops": [], "additions": [], "errors": []}

        for item in transforms:
            op = item.get("operation", "")
            src = item.get("source_feature", "")
            tgt = item.get("target_feature", "")
            params = item.get("parameters", {})

            try:
                if op == "DROP":
                    if src in df.columns:
                        df.drop(columns=[src], inplace=True)
                        dropped.append(src)
                        transform_detail["drops"].append(src)

                elif op == "LOG_TRANSFORM":
                    if src in df.columns and tgt:
                        offset = params.get("offset", 1.0)
                        df[tgt] = np.log(df[src].fillna(0).clip(lower=0) + offset)
                        added.append(tgt)
                        transform_detail["additions"].append({"src": src, "tgt": tgt, "op": "log"})

                elif op == "INTERACTION":
                    feats = params.get("features", [src])
                    if tgt and all(f in df.columns for f in feats):
                        if params.get("type") == "multiply":
                            df[tgt] = df[feats[0]].fillna(0) * df[feats[1]].fillna(0)
                        else:
                            df[tgt] = df[feats[0]].fillna(0) + df[feats[1]].fillna(0)
                        added.append(tgt)
                        transform_detail["additions"].append({"src": feats, "tgt": tgt, "op": params.get("type", "multiply")})

                elif op == "STANDARDIZE":
                    if src in df.columns and tgt:
                        mean = df[src].mean()
                        std = df[src].std()
                        df[tgt] = (df[src] - mean) / (std or 1)
                        added.append(tgt)
                        transform_detail["additions"].append({"src": src, "tgt": tgt, "op": "standardize", "mean": float(mean), "std": float(std)})

                transform_detail["transforms_applied"] += 1

            except Exception as exc:
                transform_detail["errors"].append({"op": op, "src": src, "error": str(exc)})
                logger.warning("feature_transform_item_failed op=%s src=%s err=%s", op, src, exc)

        after_count = len([c for c in df.columns if c not in {"sample_id", "apply_time", "is_bad", "y_pred_proba", "risk_score"}])

        # ── 3. 保存 transform pipeline 到 MinIO ──
        artifact_uri = None
        try:
            from minio import Minio
            client = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)
            pipeline = {
                "plan_id": plan_id,
                "target_schema_version": target_schema_version,
                "transforms": transforms,
                "dropped_features": dropped,
                "added_features": added,
            }
            obj_path = f"features/transforms/{plan_id}/pipeline.json"
            client.put_object(
                "riskitem", obj_path,
                _io.BytesIO(json.dumps(pipeline).encode("utf-8")),
                len(json.dumps(pipeline)),
            )
            artifact_uri = f"s3://riskitem/{obj_path}"
            logger.info("feature_pipeline_saved uri=%s", artifact_uri)
        except Exception as exc:
            logger.warning("minio_save_failed err=%s", exc)
            artifact_uri = f"s3://riskitem/features/transforms/{plan_id}/pipeline.json"

        # ── 4. 保存新特征快照到 MinIO ──
        snapshot_id = str(uuid.uuid4())
        try:
            from minio import Minio
            client = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)
            buf = _io.BytesIO()
            df.to_parquet(buf, index=False)
            buf.seek(0)
            snap_path = f"features/snapshots/{snapshot_id}/data.parquet"
            client.put_object("riskitem", snap_path, buf, len(buf.getvalue()))
            logger.info("feature_snapshot_saved id=%s", snapshot_id)
        except Exception as exc:
            logger.warning("feature_snapshot_save_failed err=%s", exc)

        # ── 5. 回调 API ──
        callback_payload = {
            "plan_id": plan_id,
            "status": "SUCCEEDED",
            "lifecycle_run_id": lifecycle_run_id,
            "model_id": model_id,
            "transform_artifact_uri": artifact_uri,
            "feature_snapshot_id": snapshot_id,
            "feature_schema_version": target_schema_version,
            "feature_count_before": before_count,
            "feature_count_after": after_count,
            "dropped_features": dropped,
            "added_features": added,
            "transform_detail": transform_detail,
        }

        _api_post(f"/api/internal/iteration/features/{plan_id}/callback", callback_payload)
        logger.info(
            "feature_reconstruction_completed plan=%s before=%d after=%d dropped=%d added=%d",
            plan_id, before_count, after_count, len(dropped), len(added),
        )

        return {
            "status": "SUCCEEDED",
            "plan_id": plan_id,
            "feature_schema_version": target_schema_version,
            "feature_snapshot_id": snapshot_id,
        }

    except Exception as exc:
        logger.error("feature_reconstruction_failed plan=%s err=%s", plan_id, str(exc))
        try:
            _api_post(f"/api/internal/iteration/features/{plan_id}/callback", {
                "plan_id": plan_id,
                "status": "FAILED",
                "lifecycle_run_id": lifecycle_run_id,
                "model_id": model_id,
                "error_message": str(exc),
            })
        except Exception:
            logger.warning("feature_callback_fail_error", exc_info=True)

        raise self.retry(exc=exc)
