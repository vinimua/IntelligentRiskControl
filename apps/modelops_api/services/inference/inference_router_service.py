"""Inference routing and real model prediction service."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ...config import settings
from ...core.exceptions import NotFoundError, ServiceUnavailableError, ValidationAppError

logger = structlog.get_logger(__name__)

_MODEL_CACHE: dict[str, "LoadedModel"] = {}


@dataclass
class ModelArtifactRef:
    version_code: str
    artifact_uri: str
    source: str
    metadata: dict


@dataclass
class LoadedModel:
    model: Any
    artifact_uri: str
    loader: str


class InferenceRouterService:
    """Read routing config, choose a version, then run real predict_proba."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_routing_state(self, model_id: str, environment: str = "PROD") -> dict:
        result = await self.session.execute(
            text("""
                SELECT * FROM model_registry.model_deployment_state
                WHERE model_id = :mid AND environment = :env
                ORDER BY updated_at DESC LIMIT 1
            """),
            {"mid": model_id, "env": environment},
        )
        row = result.mappings().first()
        if not row:
            return {
                "model_id": model_id,
                "environment": environment,
                "active_version_code": None,
                "stable_version_code": None,
                "challenger_version_code": None,
                "challenger_traffic_ratio": 0.0,
                "state_version": 0,
                "message": "no routing config - model has not been deployed",
            }
        return dict(row)

    async def route(self, model_id: str, request_id: str) -> dict:
        state = await self.get_routing_state(model_id)
        champion = state.get("active_version_code") or state.get("stable_version_code") or "champion_default"
        challenger = state.get("challenger_version_code")
        ratio = float(state.get("challenger_traffic_ratio", 0))

        if not challenger or ratio <= 0:
            return {
                "model_id": model_id,
                "request_id": request_id,
                "chosen_version": champion,
                "chosen_role": "CHAMPION",
                "hash_value": 1.0,
                "challenger_traffic_ratio": 0.0,
                "routing_reason": "no_challenger_or_zero_traffic",
            }

        if ratio >= 1.0:
            return {
                "model_id": model_id,
                "request_id": request_id,
                "chosen_version": challenger,
                "chosen_role": "CHAMPION",
                "hash_value": 0.0,
                "challenger_traffic_ratio": 1.0,
                "routing_reason": "challenger_promoted_to_full_traffic",
            }

        hash_val = _stable_hash(request_id, model_id)
        if hash_val < ratio:
            chosen = challenger
            role = "CHALLENGER"
            reason = f"hash={hash_val:.4f} < ratio={ratio}"
        else:
            chosen = champion
            role = "CHAMPION"
            reason = f"hash={hash_val:.4f} >= ratio={ratio}"

        logger.info(
            "inference_routed",
            model_id=model_id,
            request_id=request_id[:12],
            role=role,
            version=chosen,
            ratio=ratio,
            hash_val=round(hash_val, 4),
        )
        return {
            "model_id": model_id,
            "request_id": request_id,
            "chosen_version": chosen,
            "chosen_role": role,
            "hash_value": round(hash_val, 4),
            "challenger_traffic_ratio": ratio,
            "routing_reason": reason,
            "champion_version": champion,
            "challenger_version": challenger,
        }

    async def predict(self, model_id: str, request_id: str, features: dict | None = None) -> dict:
        """Route the request and score it with the selected persisted model artifact."""
        features = features or {}
        route_result = await self.route(model_id, request_id)
        version = route_result["chosen_version"]
        artifact_ref = await self.get_model_artifact_ref(model_id, version)
        loaded = self._load_model_artifact(artifact_ref.artifact_uri)
        frame, feature_names = _build_feature_frame(loaded.model, features)
        score = _predict_score(loaded.model, frame)
        threshold = 0.5

        return {
            **route_result,
            "prediction": {
                "score": round(score, 6),
                "threshold": threshold,
                "decision": "REJECT" if score >= threshold else "APPROVE",
                "score_source": "real_model_predict_proba",
            },
            "artifact": {
                "artifact_uri": artifact_ref.artifact_uri,
                "artifact_source": artifact_ref.source,
                "loader": loaded.loader,
            },
            "feature_schema": {
                "feature_count": len(feature_names),
                "features_used": feature_names,
                "missing_features_filled_with_zero": [
                    name for name in feature_names if name not in features
                ],
                "extra_features_ignored": [
                    name for name in features.keys() if name not in feature_names
                ],
            },
            "features_received": list(features.keys()),
        }

    async def simulate_predict(self, model_id: str, request_id: str, features: dict | None = None) -> dict:
        """Backward compatible alias. The implementation is now real inference."""
        return await self.predict(model_id, request_id, features)

    async def get_model_artifact_ref(self, model_id: str, version_code: str) -> ModelArtifactRef:
        registry = await self.session.execute(
            text("""
                SELECT artifact_uri, metrics_json
                FROM model_registry.model_versions
                WHERE model_id = :mid
                  AND version_code = :ver
                  AND artifact_uri IS NOT NULL
                ORDER BY updated_at DESC
                LIMIT 1
            """),
            {"mid": model_id, "ver": version_code},
        )
        row = registry.mappings().first()
        if row and row.get("artifact_uri"):
            return ModelArtifactRef(
                version_code=version_code,
                artifact_uri=row["artifact_uri"],
                source="model_registry.model_versions",
                metadata=dict(row.get("metrics_json") or {}),
            )

        training = await self.session.execute(
            text("""
                SELECT result_json, request_json
                FROM iteration.training_jobs
                WHERE result_json IS NOT NULL
                  AND result_json ->> 'status' = 'SUCCEEDED'
                  AND result_json ->> 'candidate_version' = :ver
                  AND (
                    request_json ->> 'model_id' = :mid
                    OR request_json ->> 'model_id' IS NULL
                  )
                  AND result_json ->> 'model_artifact_uri' IS NOT NULL
                ORDER BY completed_at DESC NULLS LAST, updated_at DESC
                LIMIT 1
            """),
            {"mid": model_id, "ver": version_code},
        )
        training_row = training.mappings().first()
        if training_row:
            result_json = dict(training_row.get("result_json") or {})
            artifact_uri = result_json.get("model_artifact_uri")
            if artifact_uri:
                return ModelArtifactRef(
                    version_code=version_code,
                    artifact_uri=artifact_uri,
                    source="iteration.training_jobs.result_json",
                    metadata=result_json,
                )

        raise NotFoundError(
            f"model version {version_code} has no persisted artifact; real inference cannot run"
        )

    def _load_model_artifact(self, artifact_uri: str) -> LoadedModel:
        cached = _MODEL_CACHE.get(artifact_uri)
        if cached:
            return cached

        try:
            if _is_joblib_uri(artifact_uri):
                import joblib

                payload = _read_artifact_bytes(artifact_uri)
                model = joblib.load(io.BytesIO(payload))
                loaded = LoadedModel(model=model, artifact_uri=artifact_uri, loader="joblib")
            else:
                loaded = _load_mlflow_model(artifact_uri)
        except ServiceUnavailableError:
            raise
        except Exception as exc:
            logger.exception("model_artifact_load_failed", artifact_uri=artifact_uri)
            raise ServiceUnavailableError(
                f"failed to load model artifact {artifact_uri}: {exc}"
            ) from exc

        _MODEL_CACHE[artifact_uri] = loaded
        return loaded


def _stable_hash(request_id: str, model_id: str) -> float:
    seed = f"{model_id}:{request_id}"
    digest = hashlib.md5(seed.encode()).digest()[:8]
    num = int.from_bytes(digest, "big")
    return num / (2 ** 64)


def _is_joblib_uri(uri: str) -> bool:
    parsed = urlparse(uri)
    path = parsed.path or uri
    return path.endswith((".joblib", ".pkl"))


def _read_artifact_bytes(uri: str) -> bytes:
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        try:
            from minio import Minio

            bucket = parsed.netloc
            object_name = parsed.path.lstrip("/")
            client = Minio(
                settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=settings.minio_secure,
            )
            response = client.get_object(bucket, object_name)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except Exception as exc:
            raise ServiceUnavailableError(
                f"cannot read model artifact from MinIO: {uri}; {exc}"
            ) from exc

    if parsed.scheme == "file":
        path = parsed.path
        if os.name == "nt" and path.startswith("/") and len(path) > 2 and path[2] == ":":
            path = path[1:]
        return Path(path).read_bytes()

    if parsed.scheme == "" and uri:
        return Path(uri).read_bytes()

    raise ValidationAppError("UNSUPPORTED_ARTIFACT_URI", f"unsupported artifact uri: {uri}")


def _load_mlflow_model(uri: str) -> LoadedModel:
    try:
        import mlflow.pyfunc

        if settings.mlflow_s3_endpoint_url:
            os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", settings.mlflow_s3_endpoint_url)
        model = mlflow.pyfunc.load_model(uri)
        return LoadedModel(model=model, artifact_uri=uri, loader="mlflow.pyfunc")
    except Exception as exc:
        raise ServiceUnavailableError(
            f"cannot load MLflow model artifact {uri}: {exc}"
        ) from exc


def _build_feature_frame(model: Any, features: dict) -> tuple[pd.DataFrame, list[str]]:
    feature_names = _feature_names(model)
    if not feature_names:
        if not features:
            raise ValidationAppError(
                "MISSING_FEATURES",
                "real inference requires features when artifact has no feature schema",
            )
        feature_names = list(features.keys())

    row = {name: features.get(name, 0) for name in feature_names}
    frame = pd.DataFrame([row], columns=feature_names)
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    return frame, feature_names


def _feature_names(model: Any) -> list[str]:
    for attr in ("feature_name_", "feature_names_in_"):
        value = getattr(model, attr, None)
        if callable(value):
            value = value()
        if value is not None:
            return [str(v) for v in list(value)]
    return []


def _predict_score(model: Any, frame: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(frame)
        return _extract_positive_probability(proba)

    if hasattr(model, "predict"):
        pred = model.predict(frame)
        return _extract_positive_probability(pred)

    raise ValidationAppError(
        "UNSUPPORTED_MODEL_OBJECT",
        "loaded model does not expose predict_proba or predict",
    )


def _extract_positive_probability(values: Any) -> float:
    import numpy as np

    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return float(arr[0, 1])
    if arr.ndim >= 1:
        return float(arr.reshape(-1)[0])
    return float(arr)
