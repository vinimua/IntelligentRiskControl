"""初始 Champion 训练 + MLflow 记录 + Baseline 基准

运行方式（需要 Docker 基础设施就绪）：
    python -m apps.modelops_api.scripts.train_initial_champion

等价于路线图 §5.3 第 8–10 条：
- 训练初始 Champion
- 模型记录到 MLflow 和 model_versions
- 建立健康基准 baseline
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import ModelSignature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score

from apps.modelops_api.config import settings
from apps.modelops_api.database import async_session
from apps.modelops_api.repositories.model_repo import ModelRepo
from apps.modelops_api.repositories.snapshot_repo import SnapshotRepo
from apps.modelops_api.services.snapshot_service import compute_dataframe_hash

RANDOM_SEED = 42
MODEL_ID = "credit_score_v1"
MODEL_NAME = "信用评分模型 (初始 Champion)"
MODEL_TYPE = "CREDIT_RISK"
FIRST_VERSION = "credit_score_v1"
N_SAMPLES = 5_000
N_FEATURES = 20
N_INFORMATIVE = 8


def _make_deterministic_data(n_samples: int, n_features: int, n_informative: int):
    """固定的合成双分类数据集（二进制标签），保证 seed 确定可复现。

    即使没有 sklearn，也可手工生成（后备路径）。
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # 生成特征
    X = rng.normal(0, 1, size=(n_samples, n_features))

    # 仅 n_informative 个特征与标签相关
    coef = rng.normal(0, 0.6, size=n_features)
    coef[n_informative:] = 0
    linear = X @ coef

    # sigmoid → probability → binary label
    prob = 1.0 / (1.0 + np.exp(-linear))
    y = (rng.random(n_samples) < prob).astype(int)

    columns = [f"feature_{i:03d}" for i in range(n_features)]

    records = []
    for i in range(n_samples):
        row = {col: float(X[i, j]) for j, col in enumerate(columns)}
        row["label"] = int(y[i])
        records.append(row)

    return records, columns


def ks_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """KS (Kolmogorov–Smirnov) 统计量。"""
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


async def train(db_only: bool = False):
    """执行 Champion 训练全流程。

    db_only=True 时不连接 MLflow/MinIO，仅写数据库记录（CI 环境）。
    """
    print(f"[{datetime.now(timezone.utc).isoformat()}] 开始初始 Champion 训练")

    # ── 1. 生成训练数据 ──
    records, feature_cols = _make_deterministic_data(
        N_SAMPLES, N_FEATURES, N_INFORMATIVE
    )
    X = np.array([[r[c] for c in feature_cols] for r in records], dtype=np.float64)
    y = np.array([r["label"] for r in records], dtype=np.int64)
    print(f"  训练集: {len(records)} 行 × {len(feature_cols)} 列, 正样本率={y.mean():.4f}")

    # ── 2. 训练模型 ──
    lr = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=500, random_state=RANDOM_SEED
    )
    model = CalibratedClassifierCV(lr, method="isotonic", cv=3)
    model.fit(X, y)

    y_prob = model.predict_proba(X)[:, 1]
    train_auc = float(roc_auc_score(y, y_prob))
    train_ks = float(ks_score(y, y_prob))
    print(f"  训练集 AUC={train_auc:.6f}  KS={train_ks:.6f}")

    content_hash = compute_dataframe_hash(records)
    artifact_uri: str | None = None
    mlflow_run_id: str | None = None

    if not db_only:
        # ── 3. MLflow 记录 ──
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        if not mlflow.get_experiment_by_name("RiskItem"):
            mlflow.create_experiment("RiskItem")
        mlflow.set_experiment("RiskItem")

        with mlflow.start_run(run_name=f"{MODEL_ID}_champion_{FIRST_VERSION}") as run:
            mlflow_run_id = run.info.run_id
            mlflow.log_param("model_id", MODEL_ID)
            mlflow.log_param("version_code", FIRST_VERSION)
            mlflow.log_param("random_seed", RANDOM_SEED)
            mlflow.log_param("n_samples", N_SAMPLES)
            mlflow.log_param("n_features", N_FEATURES)

            mlflow.log_metric("train_auc", train_auc)
            mlflow.log_metric("train_ks", train_ks)

            signature = ModelSignature.from_dict({
                "inputs": json.dumps(
                    [{"name": c, "type": "double"} for c in feature_cols]
                ),
                "outputs": json.dumps(
                    [{"name": "probability", "type": "double"}]
                ),
            })
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="champion_model",
                signature=signature,
                registered_model_name=MODEL_ID,
            )
            artifact_uri = f"{settings.mlflow_artifact_root}{mlflow_run_id}/artifacts/champion_model"
            print(f"  MLflow run_id={mlflow_run_id}")

    # ── 4. 落库 ──
    async with async_session() as session:
        model_repo = ModelRepo(session)
        snapshot_repo = SnapshotRepo(session)

        # 注册模型（幂等）
        existing = await model_repo.get_model(MODEL_ID)
        if not existing:
            await model_repo.insert_model(MODEL_ID, MODEL_NAME, MODEL_TYPE)

        # 训练快照元数据
        snapshot = await snapshot_repo.insert_snapshot(
            storage_uri=(
                f"s3://{settings.minio_bucket}/snapshots/{MODEL_ID}/{content_hash}.parquet"
                if not db_only
                else f"parquet://local/{content_hash}.parquet"
            ),
            content_hash=content_hash,
            model_id=MODEL_ID,
            window_id="W0_20250101_20250331",
            data_track="NATURAL",
            row_count=len(records),
            column_count=len(feature_cols),
            label_maturity_time=datetime(2025, 6, 1, tzinfo=timezone.utc),
        )

        # 注册版本
        existing_ver = await model_repo.get_version(MODEL_ID, FIRST_VERSION)
        if not existing_ver:
            await model_repo.insert_version(
                model_id=MODEL_ID,
                version_code=FIRST_VERSION,
                role="CHAMPION",
                mlflow_run_id=mlflow_run_id,
                artifact_uri=artifact_uri,
                metrics_json={
                    "train_auc": train_auc,
                    "train_ks": train_ks,
                    "n_samples": N_SAMPLES,
                    "n_features": N_FEATURES,
                    "random_seed": RANDOM_SEED,
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                },
            )

        # 设置当前 Champion
        await model_repo.set_champion(MODEL_ID, FIRST_VERSION)
        await session.commit()

    # ── 5. 输出 baseline ──
    baseline = {
        "model_id": MODEL_ID,
        "version_code": FIRST_VERSION,
        "train_auc": train_auc,
        "train_ks": train_ks,
        "random_seed": RANDOM_SEED,
        "content_hash": content_hash,
        "mlflow_run_id": mlflow_run_id,
        "artifact_uri": artifact_uri,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    print(f"  Baseline: {json.dumps(baseline, indent=2, ensure_ascii=False, default=str)}")
    return baseline


if __name__ == "__main__":
    db_only_flag = "--db-only" in sys.argv
    result = asyncio.run(train(db_only=db_only_flag))
    print(f"\n训练完成。Champion={FIRST_VERSION}")
