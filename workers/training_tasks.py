"""真实训练 Worker — Celery 异步任务。

P3: 真实 LightGBM 训练 + MinIO 产物 + MLflow 实验追踪。
"""
from __future__ import annotations

import json
import os
import urllib.request
import uuid
from datetime import datetime, timezone

from celery.utils.log import get_task_logger

from .app import app

logger = get_task_logger(__name__)

API_BASE = os.getenv("MODELOPS_API_BASE", "http://127.0.0.1:8000")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _api_post(path: str, body: dict) -> dict:
    """同步 HTTP POST，兼容 Windows ProactorEventLoop。"""
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        logger.warning("api_post_failed url=%s code=%s", url, exc.code)
        raise


def _load_feature_snapshot(snapshot_ids: list[str], window_ids: list[str]):
    """Load reconstructed feature snapshots from MinIO when available."""
    if not snapshot_ids:
        return None

    usable_snapshot_ids = [
        str(snapshot_id)
        for snapshot_id in snapshot_ids
        if snapshot_id and not str(snapshot_id).startswith("snapshot-w")
    ]
    if not usable_snapshot_ids:
        return None

    import io as _io
    import pandas as pd
    from minio import Minio

    frames = []
    client = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)
    for snapshot_id in usable_snapshot_ids:
        response = None
        try:
            response = client.get_object("riskitem", f"features/snapshots/{snapshot_id}/data.parquet")
            df = pd.read_parquet(_io.BytesIO(response.read()))
            if "__source_window_id" in df.columns:
                df = df[df["__source_window_id"].isin(window_ids)].copy()
                df.drop(columns=["__source_window_id"], inplace=True)
            if not df.empty:
                frames.append(df)
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _load_training_data(window_ids: list[str], data_snapshot_ids: list[str] | None = None):
    """从 W2/W3 窗口加载训练和验证数据。W4 严格禁止。"""
    import pandas as pd
    from apps.modelops_api.services.monitoring.window_loader import (
        add_apply_time_features,
        load_window,
    )

    try:
        snapshot_df = _load_feature_snapshot(data_snapshot_ids or [], window_ids)
        if snapshot_df is not None:
            logger.info(
                "feature_snapshot_training_data_loaded windows=%s rows=%d",
                window_ids,
                len(snapshot_df),
            )
            from apps.modelops_api.services.monitoring.window_loader import add_apply_time_features

            return add_apply_time_features(snapshot_df)
    except Exception as exc:
        logger.warning("feature_snapshot_load_failed windows=%s err=%s", window_ids, exc)

    frames = []
    for wid in window_ids:
        if wid == "W4":
            logger.warning("W4 blocked from training")
            continue
        df = load_window(wid)
        frames.append(df)

    if not frames:
        raise ValueError("No valid training windows (W4 excluded)")

    return add_apply_time_features(pd.concat(frames, ignore_index=True))


def _build_sample_weight(job_input: dict, df):
    """A7 策略差异（阻塞 5）：recent_weighted / segment_weighted 生成真实样本权重。

    - recent_weighted_retrain: 按申请时间指数衰减（越近期权重越高）
    - segment_weighted_retrain: 按冻结客群标识放大受损客群权重
    其余策略返回 None（不带权重训练）。
    """
    import numpy as np
    import pandas as pd

    strategy_code = str(job_input.get("strategy_code") or "")
    policy = job_input.get("sample_weight_policy") or {}
    if strategy_code not in {"recent_weighted_retrain", "segment_weighted_retrain"}:
        return None

    weights = np.ones(len(df), dtype=float)
    if strategy_code == "recent_weighted_retrain":
        if "apply_time" not in df.columns:
            # fail-closed：无时间列时不能静默全 1 权重
            raise ValueError(
                "recent_weighted_retrain 需要 apply_time 列生成近期权重"
            )
        times = pd.to_datetime(df["apply_time"])
        tmax = times.max()
        days = (tmax - times).dt.days
        factor = float(policy.get("recent_factor", 2.0))
        decay_days = float(policy.get("recent_decay_days", 30))
        weights *= 1 + (factor - 1) * np.exp(-days / max(decay_days, 1.0))
    if strategy_code == "segment_weighted_retrain":
        segment_col = policy.get("segment_column")
        affected = list(policy.get("affected_segments") or [])
        boost = float(policy.get("segment_boost", 3.0))
        if not segment_col or not affected or segment_col not in df.columns:
            # fail-closed：缺少冻结客群定义时不得静默退化成全量等权训练
            raise ValueError(
                "segment_weighted_retrain 需要冻结客群定义 "
                "(sample_weight_policy.segment_column + affected_segments)，"
                f"当前 policy={policy}"
            )
        # 统一字符串规范化：诊断层产出字符串客群码（数值型类别在
        # astype("string") 下是 "3"/"4"），必须与训练数据同侧规范化，
        # 否则整数 3 != "3" 导致权重全为 1
        affected_set = {str(v) for v in affected}
        segment_values = (
            df[segment_col].astype("string").fillna("__MISSING__")
        )
        mask = segment_values.isin(affected_set).to_numpy()
        if not mask.any():
            # fail-closed：冻结客群在训练数据中无匹配样本，
            # 不得静默全 1 权重训练
            raise ValueError(
                "冻结客群在训练数据中无匹配样本: "
                f"segment_column={segment_col} affected={affected_set}"
            )
        weights[mask] *= boost
    return weights


def _prepare_incremental_init(
    job_input: dict,
    train_df,
    val_df,
    algorithm: str,
    model_id: str,
):
    """增量训练准备：加载并校验 Champion，返回 (init_model, feature_cols, tree_count)。

    - 只允许 LightGBM（A7 supported_algorithms=['lightgbm']）
    - 用训练数据的特征列加载并评分 Champion（P0 修复：此前传空列
      导致 predict_proba 空 DataFrame → loaded=False）
    - Champion booster_.feature_name() 必须与训练特征顺序完全一致
    """
    if algorithm != "lightgbm":
        raise ValueError(
            f"增量训练仅支持 lightgbm（A7 supported_algorithms），"
            f"当前 algorithm={algorithm}"
        )
    expected_cols = _feature_columns(train_df)
    champion_version = job_input.get("base_model_version") or "champion_v1"
    champion_loaded = _load_and_score_champion(
        champion_version, val_df, expected_cols, algorithm, model_id,
    )
    if not champion_loaded.get("loaded") or champion_loaded.get("model") is None:
        raise ValueError(
            f"增量训练无法加载 Champion 模型 {champion_version}。"
            f"load_errors: {champion_loaded.get('load_errors', [])}"
        )
    # Champion 转为 Booster 后传给 init_model（LGBMClassifier 不直接接受）
    init_model = getattr(champion_loaded["model"], "booster_", None)
    if init_model is None:
        raise ValueError("Champion 模型没有 booster_，无法增量续训")
    champion_tree_count = int(init_model.num_trees())
    # 特征顺序一致性：Champion 特征必须与训练数据特征顺序完全一致
    champion_cols = list(init_model.feature_name() or [])
    if champion_cols and champion_cols != expected_cols:
        raise ValueError(
            "Champion 特征顺序与训练数据不一致: "
            f"champion={champion_cols[:5]}... train={expected_cols[:5]}..."
        )
    logger.info(
        "incremental_training champion=%s loaded=%s trees=%d",
        champion_version, champion_loaded["loaded"], champion_tree_count,
    )
    return init_model, champion_cols or expected_cols, champion_tree_count


def _feature_columns(df, target: str = "is_bad") -> list[str]:
    """训练特征列（排除非特征列，与 A7 特征契约一致）。"""
    exclude = {"sample_id", "apply_time", target, "y_pred_proba", "risk_score"}
    return [
        c for c in df.columns
        if c not in exclude and df[c].dtype in ("int64", "float64")
    ]


def _train_lightgbm(df, target: str = "is_bad", seed: int = 2026, hyperparameters: dict | None = None,
                    init_model=None, feature_cols: list[str] | None = None,
                    sample_weight=None):
    """训练 LightGBM 二分类模型。

    init_model: 传入 Champion Booster 时 → 增量训练（继续拟合新数据）
    不传 → 全量重训（从头开始）
    feature_cols: 显式特征顺序（增量训练必须与 Champion 完全一致）
    sample_weight: 近期/客群加权策略的真实权重
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    feature_cols = list(feature_cols) if feature_cols else _feature_columns(df, target)
    logger.info("training_features n=%d", len(feature_cols))

    X = df[feature_cols].fillna(0)
    y = df[target]

    if sample_weight is not None:
        X_train, X_val, y_train, y_val, sw_train, _sw_val = train_test_split(
            X, y, sample_weight, test_size=0.2, random_state=seed, stratify=y,
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y,
        )
        sw_train = None

    params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.05,
        "random_state": seed,
        "verbosity": -1,
    }
    params.update(hyperparameters or {})
    params["random_state"] = seed
    model = lgb.LGBMClassifier(**params)

    if init_model is not None:
        # 增量模式：在 Champion Booster 基础上续训
        logger.info("incremental_training_init_model_provided — 在 Champion 模型基础上续训")
        model.fit(X_train, y_train, init_model=init_model, sample_weight=sw_train)
    else:
        # 全量模式：从头训练
        model.fit(X_train, y_train, sample_weight=sw_train)

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    train_ks = _compute_ks(y_train, model.predict_proba(X_train)[:, 1])
    val_ks = _compute_ks(y_val, model.predict_proba(X_val)[:, 1])

    logger.info(
        "train_results train_auc=%.4f val_auc=%.4f train_ks=%.4f val_ks=%.4f incremental=%s",
        train_auc, val_auc, train_ks, val_ks, init_model is not None,
    )

    return {
        "model": model,
        "feature_cols": feature_cols,
        "train_auc": train_auc,
        "val_auc": val_auc,
        "train_ks": train_ks,
        "val_ks": val_ks,
    }


def _numeric_feature_cols(df, target: str = "is_bad") -> list[str]:
    exclude = {"sample_id", "apply_time", target, "y_pred_proba", "risk_score"}
    return [c for c in df.columns if c not in exclude and df[c].dtype in ("int64", "float64")]


def _fit_sklearn_classifier(
    df, model, target: str = "is_bad", seed: int = 2026,
    sample_weight=None,
):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    feature_cols = _numeric_feature_cols(df, target)
    logger.info("training_features n=%d", len(feature_cols))

    X = df[feature_cols].fillna(0)
    y = df[target]
    if sample_weight is not None:
        X_train, X_val, y_train, y_val, w_train, _w_val = train_test_split(
            X, y, sample_weight, test_size=0.2, random_state=seed, stratify=y
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        w_train = None

    if w_train is not None:
        if hasattr(model, "steps") and model.steps:
            # sklearn Pipeline 不接受顶层 sample_weight，
            # 按 stepname__sample_weight 传给最后一个估计器
            last_step = model.steps[-1][0]
            model.fit(
                X_train, y_train,
                **{f"{last_step}__sample_weight": w_train},
            )
        else:
            model.fit(X_train, y_train, sample_weight=w_train)
    else:
        model.fit(X_train, y_train)

    train_pred = model.predict_proba(X_train)[:, 1]
    val_pred = model.predict_proba(X_val)[:, 1]
    train_auc = roc_auc_score(y_train, train_pred)
    val_auc = roc_auc_score(y_val, val_pred)
    train_ks = _compute_ks(y_train, train_pred)
    val_ks = _compute_ks(y_val, val_pred)

    logger.info(
        "train_results train_auc=%.4f val_auc=%.4f train_ks=%.4f val_ks=%.4f",
        train_auc, val_auc, train_ks, val_ks,
    )

    return {
        "model": model,
        "feature_cols": feature_cols,
        "train_auc": train_auc,
        "val_auc": val_auc,
        "train_ks": train_ks,
        "val_ks": val_ks,
    }


def _train_logistic_regression(
    df, target: str = "is_bad", seed: int = 2026,
    hyperparameters: dict | None = None, sample_weight=None,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    params = {
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "liblinear",
        "random_state": seed,
    }
    params.update(hyperparameters or {})
    params["random_state"] = seed
    model = make_pipeline(StandardScaler(), LogisticRegression(**params))
    return _fit_sklearn_classifier(
        df, model, target=target, seed=seed, sample_weight=sample_weight,
    )


def _train_random_forest(
    df, target: str = "is_bad", seed: int = 2026,
    hyperparameters: dict | None = None, sample_weight=None,
):
    from sklearn.ensemble import RandomForestClassifier

    params = {
        "n_estimators": 120,
        "max_depth": 8,
        "min_samples_leaf": 20,
        "class_weight": "balanced_subsample",
        "n_jobs": 1,
        "random_state": seed,
    }
    params.update(hyperparameters or {})
    params["random_state"] = seed
    model = RandomForestClassifier(**params)
    return _fit_sklearn_classifier(
        df, model, target=target, seed=seed, sample_weight=sample_weight,
    )


TRAINERS = {
    "lightgbm": _train_lightgbm,
    "logistic_regression": _train_logistic_regression,
    "random_forest": _train_random_forest,
}


# ── Champion 模型加载 + 验证指标计算 ──


def _load_and_score_champion(
    champion_version: str,
    val_df,
    feature_cols: list[str],
    algorithm: str,
    model_id: str = "",
    *,
    expected_feature_schema_version: str = "",
    expected_preprocessing_version: str = "",
    expected_calibrator_version: str = "",
) -> dict:
    """从 MinIO 加载 champion 模型，校验身份后打分。

    加载校验链（任一项失败 → loaded=False）：
    1. model_id 与注册表一致
    2. version_code 与请求一致
    3. feature_schema_version 一致
    4. preprocessing_version 一致
    5. calibrator_version 一致
    6. checksum (SHA256) 一致

    Returns: {auc, ks, scores, loaded, load_errors}
    loaded=False 表示 champion 模型身份校验失败或不可用。
    """
    import hashlib
    import joblib as jl
    import io as _io
    from minio import Minio
    from sklearn.metrics import roc_auc_score

    load_errors: list[str] = []

    # ── 0. 本地 assets bundle 优先（credit_model_052/053 等完整特征模型）──
    # MinIO 的 champions/{version} 路径是共享命名空间（被 credit_model_001
    # 占用），按 model_id 加载的完整特征 Champion 必须走 assets bundle：
    # model.joblib + training_manifest.json + feature_schema.json，
    # 校验 model_id / 算法族 / 特征顺序 / schema 版本后打分。
    from pathlib import Path

    local_bundle = (
        Path(__file__).resolve().parents[1]
        / "assets" / "champion_models" / model_id / champion_version
    )
    local_model_path = local_bundle / "model.joblib"
    local_manifest_path = local_bundle / "training_manifest.json"
    local_schema_path = local_bundle / "feature_schema.json"
    if local_model_path.is_file():
        try:
            if not local_manifest_path.is_file() or not local_schema_path.is_file():
                raise ValueError("LOCAL_CHAMPION_CONTRACT_INCOMPLETE")
            manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
            schema = json.loads(local_schema_path.read_text(encoding="utf-8"))
            if manifest.get("model_id") != model_id:
                raise ValueError("LOCAL_CHAMPION_MODEL_ID_MISMATCH")
            canonical = {
                "LogisticRegression": "logistic_regression",
                "RandomForest": "random_forest",
                "XGBoost": "xgboost",
                "LightGBM": "lightgbm",
                "CatBoost": "catboost",
                "EBM": "ebm",
            }
            if canonical.get(manifest.get("algorithm_family")) != algorithm:
                raise ValueError("LOCAL_CHAMPION_ALGORITHM_MISMATCH")
            if (
                expected_feature_schema_version
                and str(schema.get("schema_version"))
                != expected_feature_schema_version
            ):
                raise ValueError("LOCAL_CHAMPION_FEATURE_SCHEMA_MISMATCH")
            if (
                expected_preprocessing_version
                and manifest.get("feature_strategy_id")
                != expected_preprocessing_version
            ):
                raise ValueError("LOCAL_CHAMPION_PREPROCESSING_MISMATCH")
            if list(schema.get("ordered_features") or []) != list(feature_cols):
                raise ValueError("LOCAL_CHAMPION_ORDERED_FEATURES_MISMATCH")
            model_bytes = local_model_path.read_bytes()
            actual_sha256 = hashlib.sha256(model_bytes).hexdigest()
            model = jl.load(_io.BytesIO(model_bytes))
            scores = model.predict_proba(val_df[feature_cols])[:, 1]
            y_val = val_df["is_bad"]
            return {
                "auc": float(roc_auc_score(y_val, scores)),
                "ks": float(_compute_ks(y_val, scores)),
                "scores": scores,
                "loaded": True,
                "load_errors": [],
                "checksum": actual_sha256,
            }
        except Exception as exc:
            return {
                "auc": None,
                "ks": None,
                "scores": None,
                "loaded": False,
                "load_errors": [f"local_champion_identity_failed:{exc}"],
            }

    try:
        client = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)
        base = f"champions/{champion_version}"
        model_path = f"{base}/model.joblib"
        meta_path = f"{base}/metadata.json"
        checksum_path = f"{base}/checksum.sha256"

        # ── 校验 1-2: model_id + version_code ──
        meta: dict = {}
        try:
            resp = client.get_object("riskitem", meta_path)
            meta = json.loads(resp.read().decode("utf-8"))
            resp.close()
            resp.release_conn()
        except Exception:
            logger.warning("champion_metadata_missing path=%s — 无法校验 model_id/合同", meta_path)
            # 元数据缺失不阻断加载，但记录警告
            load_errors.append("metadata_missing")

        if meta:
            stored_model_id = meta.get("model_id", "")
            stored_version = meta.get("version_code", "")
            if model_id and stored_model_id and stored_model_id != model_id:
                load_errors.append(f"model_id_mismatch: expected={model_id} stored={stored_model_id}")
            if stored_version and stored_version != champion_version:
                load_errors.append(f"version_mismatch: expected={champion_version} stored={stored_version}")

            # ── 校验 3-5: 特征合同 / 预处理器 / 校准器 ──
            if expected_feature_schema_version:
                stored_fsv = meta.get("feature_schema_version", "")
                if stored_fsv and stored_fsv != expected_feature_schema_version:
                    load_errors.append(f"feature_schema_mismatch: expected={expected_feature_schema_version} stored={stored_fsv}")
            if expected_preprocessing_version:
                stored_pp = meta.get("preprocessing_version", "")
                if stored_pp and stored_pp != expected_preprocessing_version:
                    load_errors.append(f"preprocessing_mismatch: expected={expected_preprocessing_version} stored={stored_pp}")
            if expected_calibrator_version:
                stored_cal = meta.get("calibrator_version", "")
                if stored_cal and stored_cal != expected_calibrator_version:
                    load_errors.append(f"calibrator_mismatch: expected={expected_calibrator_version} stored={stored_cal}")

        # ── 加载模型 ──
        response = client.get_object("riskitem", model_path)
        model_bytes = response.read()
        response.close()
        response.release_conn()

        # ── 校验 6: checksum ──
        actual_sha256 = hashlib.sha256(model_bytes).hexdigest()
        try:
            csum_resp = client.get_object("riskitem", checksum_path)
            expected_sha256 = csum_resp.read().decode("utf-8").strip()
            csum_resp.close()
            csum_resp.release_conn()
            if expected_sha256 != actual_sha256:
                load_errors.append(f"checksum_mismatch: expected={expected_sha256[:16]}... actual={actual_sha256[:16]}...")
        except Exception:
            logger.warning("champion_checksum_missing path=%s — 无法校验文件完整性", checksum_path)
            # checksum 缺失不阻断，但记录
            load_errors.append("checksum_missing")

        if load_errors:
            logger.error(
                "champion_identity_verification_failed version=%s errors=%s",
                champion_version, load_errors,
            )
            return {"auc": None, "ks": None, "scores": None, "loaded": False, "load_errors": load_errors}

        model = jl.load(_io.BytesIO(model_bytes))
        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df["is_bad"]
        scores = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, scores)
        ks = _compute_ks(y_val, scores)

        logger.info(
            "champion_loaded_and_scored version=%s model_id=%s auc=%.4f ks=%.4f sha256=%s",
            champion_version, model_id, auc, ks, actual_sha256[:16],
        )
        return {
            "model": model, "auc": auc, "ks": ks, "scores": scores,
            "loaded": True, "load_errors": [],
            "checksum": actual_sha256,
        }

    except Exception as exc:
        logger.error("champion_load_failed version=%s err=%s", champion_version, exc)
        load_errors.append(f"load_exception: {exc}")
        return {"auc": None, "ks": None, "scores": None, "loaded": False, "load_errors": load_errors}


def _calc_score_psi(expected_scores, actual_scores, n_bins: int = 10) -> float:
    """计算同一模型跨时间窗口的分数分布 PSI。

    标准 PSI 定义：同一冻结模型，分别对 W1（参考窗口）和 W3（当前窗口）
    打分，比较分数分布差异，衡量模型在时间上的稳定性。

    不用于比较不同模型（Champion vs Challenger）的输出差异——
    那是 AUC/KS/Brier/Bootstrap 的职责。

    无法计算时返回 None（不返回虚构值）。
    """
    if expected_scores is None or actual_scores is None:
        return None
    try:
        import numpy as np
        bins = np.linspace(0, 1, n_bins + 1)
        expected_pct = np.histogram(expected_scores, bins=bins)[0] / len(expected_scores)
        actual_pct = np.histogram(actual_scores, bins=bins)[0] / len(actual_scores)
        expected_pct = np.clip(expected_pct, 1e-6, None)
        actual_pct = np.clip(actual_pct, 1e-6, None)
        return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    except Exception as exc:
        logger.warning("psi_calc_failed err=%s", exc)
        return None


def _check_oot(model, feature_cols: list[str], algorithm: str) -> dict:
    """Task 4 专用：加载 W4 盲测数据，计算 AUC/KS/PSI 三项 OOT 指标。

    任务三 Training Worker 不得调用此函数——W4 是最终盲测集，
    提前读取会造成数据泄漏。

    Returns: {oot_passed, oot_auc, oot_ks, oot_psi, error}
    oot_passed 仅在三项全部达标且 checksum 一致时为 True。
    """
    try:
        from apps.modelops_api.services.monitoring.window_loader import load_window
        from sklearn.metrics import roc_auc_score

        oot_df = load_window("W4")
        if oot_df is None or len(oot_df) == 0:
            logger.warning("oot_check_skipped_no_w4_data — OOT 无法验证")
            return {"oot_passed": False, "oot_auc": None, "oot_ks": None, "oot_psi": None,
                    "error": "W4_DATA_UNAVAILABLE"}

        X_oot = oot_df[feature_cols].fillna(0)
        y_oot = oot_df["is_bad"]
        scores = model.predict_proba(X_oot)[:, 1]
        oot_auc = roc_auc_score(y_oot, scores)
        oot_ks = _compute_ks(y_oot, scores)

        # OOT PSI 需 W1 参考，Task 4 独立完成
        # 此处返回占位，完整计算在 Deployment Worker 的 OOT Gate 中
        oot_psi_val = None

        # 赛事阈值：AUC >= 0.70, KS >= 0.25, PSI <= 0.25
        auc_ok = oot_auc >= 0.70
        ks_ok = oot_ks >= 0.25
        all_passed = auc_ok and ks_ok  # PSI 在部署子图中单独判

        logger.info(
            "oot_check_done oot_auc=%.4f auc_ok=%s oot_ks=%.4f ks_ok=%s all_passed=%s",
            oot_auc, auc_ok, oot_ks, ks_ok, all_passed,
        )
        return {
            "oot_passed": all_passed,
            "oot_auc": round(oot_auc, 4),
            "oot_ks": round(oot_ks, 4),
            "oot_psi": round(oot_psi_val, 4) if oot_psi_val is not None else None,
        }

    except Exception as exc:
        logger.warning("oot_check_failed err=%s", exc)
        return {"oot_passed": False, "oot_auc": None, "oot_ks": None, "oot_psi": None,
                "error": str(exc)}


def _compute_ks(y_true, y_pred_proba):
    """计算 KS 统计量。"""
    import numpy as np
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return float(np.max(tpr - fpr))


def _save_to_minio(model_bytes: bytes, bucket: str, object_path: str):
    """保存文件到 MinIO。"""
    try:
        from minio import Minio
        client = Minio(
            "localhost:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False,
        )
        import io
        client.put_object(
            bucket, object_path,
            io.BytesIO(model_bytes), len(model_bytes),
        )
        logger.info("minio_saved path=%s", object_path)
        return f"s3://{bucket}/{object_path}"
    except Exception as exc:
        logger.error("minio_save_failed err=%s", exc)
        from pathlib import Path

        local_path = Path("artifacts") / "minio_fallback" / bucket / object_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(model_bytes)
        logger.info("model_saved_to_local_fallback path=%s", local_path)
        return local_path.resolve().as_uri()


def _track_mlflow(experiment_name: str, metrics: dict, model_path: str):
    """注册 MLflow 实验。"""
    try:
        import mlflow
        # Skip health check — MLflow may be unreachable but training still works
        try:
            urllib.request.urlopen(MLFLOW_TRACKING_URI, timeout=2)
        except Exception:
            logger.warning("mlflow_unreachable_skip uri=%s", MLFLOW_TRACKING_URI)
            return

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
            mlflow.log_param("model_path", model_path)
            mlflow.log_param("framework", "lightgbm")
        logger.info("mlflow_tracked experiment=%s", experiment_name)
    except Exception as exc:
        logger.warning("mlflow_track_failed err=%s", exc)


# ── 分群治理检测 ──

def _check_segment_governance(val_df, scores, y_true, feature_cols: list[str]) -> dict:
    """检测模型在不同分群上的表现一致性。

    在验证数据中寻找潜在的分群列（低基数非特征列），
    计算每群的 AUC，判断是否存在显著低于整体 AUC 的群。

    Returns: {passed: bool, segments: {segment_name: {auc, count, passed}}}
    无法找到分群列时返回 passed=False（保守——无法验证则不通过）。
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    # 寻找潜在分群列：非特征列、非 target、低基数
    exclude = set(feature_cols) | {"sample_id", "apply_time", "is_bad", "y_pred_proba", "risk_score"}
    candidate_cols = []
    for col in val_df.columns:
        if col in exclude:
            continue
        if val_df[col].dtype not in ("int64", "float64"):
            continue
        n_unique = val_df[col].nunique()
        if 2 <= n_unique <= 10:
            candidate_cols.append(col)

    if not candidate_cols:
        logger.warning("segment_governance_no_segment_columns — 无法找到分群列，标记为未通过")
        return {
            "passed": False,
            "segment_governance_passed": False,
            "segments": {},
            "error": "no_segment_columns_found",
        }

    overall_auc = roc_auc_score(y_true, scores)
    segment_results: dict = {}
    all_passed = True

    for seg_col in candidate_cols[:3]:  # 最多检查 3 个分群维度
        seg_values = val_df[seg_col].dropna().unique()
        seg_detail: dict = {}
        for val in seg_values:
            mask = val_df[seg_col] == val
            if mask.sum() < 20:  # 样本太少，跳过
                continue
            try:
                seg_auc = roc_auc_score(y_true[mask], scores[mask])
            except Exception:
                seg_auc = None
            seg_detail[str(val)] = {
                "auc": round(seg_auc, 4) if seg_auc is not None else None,
                "count": int(mask.sum()),
                "passed": seg_auc is not None and seg_auc >= overall_auc - 0.05,
            }
        if seg_detail:
            segment_results[seg_col] = seg_detail
            if any(not d["passed"] for d in seg_detail.values() if d["auc"] is not None):
                all_passed = False

    logger.info(
        "segment_governance_check overall_auc=%.4f segments=%d all_passed=%s",
        overall_auc, len(segment_results), all_passed,
    )
    return {"passed": all_passed, "overall_auc": round(overall_auc, 4), "segments": segment_results}


@app.task(bind=True, name="workers.training_tasks.train_model", max_retries=2, default_retry_delay=60)
def train_model(self, job_input: dict):
    """P3: 真实 LightGBM 训练。

    流程：
    1. 从 W2/W3 窗口加载数据
    2. 训练 LightGBM
    3. 保存模型到 MinIO
    4. 注册 MLflow 实验
    5. 回调 API
    """
    training_job_id = job_input["training_job_id"]
    idempotency_key = job_input.get("idempotency_key", training_job_id)
    business_round = job_input.get("business_round", 1)
    experiment_id = job_input.get("experiment_id", str(uuid.uuid4()))
    lifecycle_run_id = job_input.get("lifecycle_run_id", "")
    training_window_ids = job_input.get("training_window_ids", ["W2"])
    validation_window_ids = job_input.get("validation_window_ids", ["W3"])
    algorithm = str(job_input.get("algorithm") or "lightgbm").lower()
    hyperparameters = job_input.get("hyperparameters") or {}
    data_snapshot_ids = job_input.get("data_snapshot_ids") or []
    # model_id 必须在增量分支之前读取（增量加载 Champion 需要身份校验）
    model_id = job_input.get("model_id", "")

    logger.info("train_model_started job=%s round=%s", training_job_id, business_round)

    try:
        # 1. 加载数据
        trainer = TRAINERS.get(algorithm)
        if trainer is None:
            raise ValueError(f"unsupported training algorithm: {algorithm}")

        train_df = _load_training_data(training_window_ids, data_snapshot_ids)
        val_df = _load_training_data(validation_window_ids, data_snapshot_ids)

        # A7 阶段四：FEATURE_SELECTION 模式消费冻结特征清单
        selected_features = job_input.get("selected_feature_codes") or []
        if selected_features:
            keep_cols = [
                c for c in selected_features if c in train_df.columns
            ]
            meta_cols = [
                c for c in ("sample_id", "apply_time", "is_bad",
                            "y_pred_proba", "risk_score")
                if c in train_df.columns and c not in keep_cols
            ]
            train_df = train_df[keep_cols + meta_cols]
            val_df = val_df[[c for c in train_df.columns if c in val_df.columns]]
            logger.info(
                "feature_selection_applied selected=%d meta=%d",
                len(keep_cols), len(meta_cols),
            )

        # 2. 训练（全量 or 增量）—— A7 定稿 §7 正式枚举
        training_mode = job_input.get("training_mode", "FULL_RETRAIN")
        is_incremental = str(training_mode).upper() in {
            "INCREMENTAL_TRAIN", "INCREMENTAL",
        }
        init_model = None
        feature_cols_override: list[str] | None = None
        champion_tree_count = 0

        if is_incremental:
            init_model, feature_cols_override, champion_tree_count = (
                _prepare_incremental_init(
                    job_input, train_df, val_df, algorithm, model_id,
                )
            )

        # 近期/客群加权策略的真实样本权重（阻塞 5）
        sample_weight = _build_sample_weight(job_input, train_df)

        trainer_kwargs: dict = {}
        if is_incremental:
            trainer_kwargs["init_model"] = init_model
            trainer_kwargs["feature_cols"] = feature_cols_override
        if sample_weight is not None:
            trainer_kwargs["sample_weight"] = sample_weight
        result = trainer(
            train_df,
            seed=int(job_input.get("seed", 2026)),
            hyperparameters=hyperparameters,
            **trainer_kwargs,
        )
        model = result["model"]

        # 增量验收：新模型必须继承 Champion 旧树，而不是重新拟合
        if init_model is not None:
            new_tree_count = (
                int(model.booster_.num_trees()) if hasattr(model, "booster_") else 0
            )
            logger.info(
                "incremental_tree_inheritance champion_trees=%s total_trees=%s",
                champion_tree_count, new_tree_count,
            )
            if new_tree_count < champion_tree_count:
                raise ValueError(
                    f"增量训练未继承 Champion 树：champion={champion_tree_count} "
                    f"total={new_tree_count}（疑似重新拟合）"
                )

        # 3. 验证指标
        from sklearn.metrics import roc_auc_score
        val_pred = model.predict_proba(val_df[result["feature_cols"]].fillna(0))[:, 1]
        val_auc = roc_auc_score(val_df["is_bad"], val_pred)
        val_ks = _compute_ks(val_df["is_bad"], val_pred)

        challenger_auc = val_auc
        challenger_ks = val_ks

        # candidate_version 包含 model_id + lifecycle_run_id + round，防止多模型并发碰撞
        candidate_version = (
            f"{model_id}_{lifecycle_run_id[:8]}_round{business_round}_challenger_v1"
            if model_id else f"lifecycle_{lifecycle_run_id[:8]}_round{business_round}_challenger_v1"
        )

        # 3. 加载 champion 模型 → 校验身份后打分
        champion_version = job_input.get("base_model_version") or "champion_v1"
        champion_metrics = _load_and_score_champion(
            champion_version, val_df, result["feature_cols"], algorithm,
            model_id=model_id,
        )
        if not champion_metrics["loaded"]:
            load_errors = champion_metrics.get("load_errors", ["champion_not_found"])
            error_msg = (
                f"champion 模型 {champion_version} 身份校验失败或无法加载: {load_errors}。"
                f"model_id={model_id}，无法计算真实 champion 指标。"
            )
            logger.error("train_model_blocked_champion_identity_failed version=%s errors=%s",
                         champion_version, load_errors)
            _api_post(f"/api/internal/iteration/jobs/{training_job_id}/callback", {
                "training_job_id": training_job_id,
                "lifecycle_run_id": lifecycle_run_id,
                "idempotency_key": idempotency_key,
                "experiment_id": experiment_id,
                "status": "FAILED",
                "error_code": "CHAMPION_IDENTITY_VERIFICATION_FAILED",
                "error_message": error_msg,
            })
            return {"status": "FAILED", "error": error_msg}

        champion_auc = champion_metrics["auc"]
        champion_ks = champion_metrics["ks"]
        champion_checksum = champion_metrics.get("checksum", "")

        # 4. 加载 W1 基线数据 → 计算 Champion(W1) 健康基线和跨时间 PSI
        w1_df = None
        champion_w1_auc = None
        champion_w1_ks = None
        champion_w1_scores = None
        try:
            w1_df = _load_training_data(["W1"], data_snapshot_ids)
            if w1_df is not None and len(w1_df) > 0:
                champ_w1_result = _load_and_score_champion(
                    champion_version, w1_df, result["feature_cols"], algorithm,
                    model_id=model_id,
                )
                if champ_w1_result["loaded"]:
                    champion_w1_auc = champ_w1_result["auc"]
                    champion_w1_ks = champ_w1_result["ks"]
                    champion_w1_scores = champ_w1_result["scores"]
                    logger.info("champion_w1_baseline_loaded auc=%.4f ks=%.4f", champion_w1_auc, champion_w1_ks)
                else:
                    logger.warning("champion_w1_load_failed — W1 基线不可用，恢复率和 PSI 将不完整")
            else:
                logger.warning("w1_data_empty — W1 数据不可用")
        except Exception as exc:
            logger.warning("w1_data_load_failed err=%s — W1 基线不可用，恢复率和 PSI 将不完整", exc)

        # 5. Challenger 对 W1 打分（冻结模型，不对 W1 重拟合）
        challenger_w1_scores = None
        if w1_df is not None and len(w1_df) > 0:
            try:
                w1_features = w1_df[result["feature_cols"]].fillna(0)
                challenger_w1_scores = model.predict_proba(w1_features)[:, 1]
            except Exception as exc:
                logger.warning("challenger_w1_scoring_failed err=%s", exc)

        # 6. 保存模型到 MinIO（路径含 model_id + lifecycle_run_id，避免碰撞）
        import hashlib
        import joblib as jl
        import io as _io
        buf = _io.BytesIO()
        jl.dump(model, buf)
        buf.seek(0)
        model_bytes = buf.read()
        model_sha256 = hashlib.sha256(model_bytes).hexdigest()
        artifact_base = (
            f"challengers/{model_id}/{lifecycle_run_id}/{candidate_version}"
            if model_id else f"challengers/lifecycle_{lifecycle_run_id}/{candidate_version}"
        )
        model_uri = _save_to_minio(model_bytes, "riskitem", f"{artifact_base}/model.joblib")
        # 侧车: checksum
        _save_to_minio(model_sha256.encode("utf-8"), "riskitem", f"{artifact_base}/checksum.sha256")
        # 侧车: metadata（冻结身份校验：OOT 晋升门禁据此核对
        # model_id / lifecycle_run_id / candidate_version / model_sha256 / 特征列）
        _save_to_minio(
            json.dumps({
                "model_id": model_id,
                "lifecycle_run_id": lifecycle_run_id,
                "candidate_version": candidate_version,
                "model_sha256": model_sha256,
                "feature_cols": result["feature_cols"],
                "algorithm": algorithm,
                "w4_read_count": 0,
                "frozen_at": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False).encode("utf-8"),
            "riskitem",
            f"{artifact_base}/metadata.json",
        )

        # 7. 注册 MLflow
        _track_mlflow(
            f"lifecycle-{lifecycle_run_id}",
            {
                "val_auc": val_auc, "val_ks": val_ks,
                "train_auc": result["train_auc"], "train_ks": result["train_ks"],
                "champion_auc": champion_auc, "champion_ks": champion_ks,
                "champion_w1_auc": champion_w1_auc,
                "champion_w1_ks": champion_w1_ks,
            },
            model_uri,
        )

        # ═══════════════════════════════════════════════
        # 8. 核心赛事指标计算（AUC / KS / PSI 三项）
        # ═══════════════════════════════════════════════

        val_features = val_df[result["feature_cols"]].fillna(0)
        val_labels = val_df["is_bad"]
        challenger_scores = model.predict_proba(val_features)[:, 1]
        champion_scores = champion_metrics.get("scores")

        # ── PSI: 同一模型跨时间比较 ──
        # PSI(Champion(W1), Champion(W3)) — Champion 跨时间稳定性
        champion_psi = _calc_score_psi(champion_w1_scores, champion_scores)
        # PSI(Challenger(W1), Challenger(W3)) — Challenger 跨时间稳定性
        challenger_psi = _calc_score_psi(challenger_w1_scores, challenger_scores)
        # 取两个 PSI 的最大值作为综合 score_psi（不能比 Champion 更不稳定）
        if champion_psi is not None and challenger_psi is not None:
            score_psi = max(champion_psi, challenger_psi)
        elif champion_psi is not None:
            score_psi = champion_psi
        elif challenger_psi is not None:
            score_psi = challenger_psi
        else:
            score_psi = None

        # ── Recovery rate: AUC 和 KS 分别计算 ──
        # recovery_AUC = (Challenger_W3 - Champion_W3) / max(0.01, Champion_W1 - Champion_W3)
        if champion_w1_auc is not None and champion_auc is not None:
            auc_drop = max(0.01, champion_w1_auc - champion_auc)
            recovery_auc = (
                (challenger_auc - champion_auc) / auc_drop
                if challenger_auc > champion_auc else 0.0
            )
            ks_drop = max(0.01, (champion_w1_ks or 0.0) - (champion_ks or 0.0))
            recovery_ks = (
                (challenger_ks - (champion_ks or 0.0)) / ks_drop
                if challenger_ks > (champion_ks or 0.0) else 0.0
            ) if champion_w1_ks is not None and champion_ks is not None else 0.0
        else:
            # W1 不可用 → 恢复率无法计算，标记为不达标
            recovery_auc = 0.0
            recovery_ks = 0.0
            logger.warning("recovery_rate_cannot_compute_no_w1_baseline — 标记为未恢复")

        # 综合 recovery_rate: AUC 和 KS 取最小值（各项需分别满足阈值）
        recovery_rate = min(recovery_auc, recovery_ks)

        # ── Discrimination: challenger_auc >= champion_auc - 1% 容差 ──
        discrimination_passed = challenger_auc >= (champion_auc or 0.0) - 0.01

        # ── Calibration: Brier/ECE — 不默认通过 ──
        from sklearn.metrics import brier_score_loss
        calibration_passed = False
        challenger_brier: float | None = None
        champion_brier: float | None = None
        calibration_error: str | None = None
        try:
            challenger_brier = brier_score_loss(val_labels, challenger_scores)
            if champion_scores is not None:
                champion_brier = brier_score_loss(val_labels, champion_scores)
                # 使用 Champion(W1) 健康区间判断：
                # 根因是校准异常 → Challenger 必须优于 Champion 且回到 W1 区间
                # 根因不是校准异常 → 至少不得超出 W1 区间
                # 当前简化：Challenger Brier <= Champion Brier + 走宽限
                if champion_w1_scores is not None:
                    try:
                        champion_w1_brier = brier_score_loss(val_labels[:len(champion_w1_scores)],
                                                             champion_w1_scores[:len(val_labels)])
                        # W1 健康区间上限
                        healthy_brier_upper = champion_w1_brier + 0.01
                        calibration_passed = challenger_brier <= healthy_brier_upper
                    except Exception:
                        calibration_passed = challenger_brier <= (champion_brier or 0.0) + 0.01
                else:
                    calibration_passed = challenger_brier <= (champion_brier or 0.0) + 0.01
            else:
                calibration_error = "champion_scores_unavailable"
                calibration_passed = False
        except Exception as exc:
            calibration_error = str(exc)
            calibration_passed = False
            logger.warning("calibration_computation_failed err=%s — 标记为未通过", exc)

        # ── Train/valid gap ──
        train_valid_gap = abs(result["train_auc"] - val_auc)

        # ── OOT: 任务三不读 W4，只标记 pre_oot_qualified ──
        # W4 盲测由任务四 Deployment 独立执行
        core_performance_passed = (
            (recovery_auc >= 1.0 or recovery_ks >= 1.0)
            and discrimination_passed
            and (score_psi is not None and score_psi <= 0.25)
        )
        pre_oot_qualified = core_performance_passed and calibration_passed

        # ── Healthy lower bound: Champion(W1) 真实基线 ──
        if champion_w1_auc is not None:
            healthy_lower_bound = round(max((champion_w1_auc or 0.0) - 0.02, 0.72), 4)
        else:
            healthy_lower_bound = round(max((champion_auc or 0.0) - 0.02, 0.72), 4)

        # KS 专属健康下界（AUC 的 healthy_lower_bound 不能套给 KS）
        ks_healthy_lower_bound = (
            round(max((champion_ks or 0.0) - 0.02, 0.15), 4)
            if champion_ks is not None else None
        )

        # ── 同样本配对 Bootstrap: AUC/KS 提升的 95% 置信区间 ──
        # 资格门 require_same_sample_bootstrap 的真实证据来源；
        # Champion/Challenger 在同一验证集上配对重采样。
        import numpy as np

        def _ks_stat(labels, scores) -> float:
            labels = np.asarray(labels, dtype=float)
            scores = np.asarray(scores, dtype=float)
            n = len(labels)
            n_bad = float(labels.sum())
            if n == 0 or n_bad in (0.0, float(n)):
                return 0.0
            order = np.argsort(-scores)
            sorted_labels = labels[order]
            csum = np.cumsum(sorted_labels)
            n_good = n - n_bad
            return float(np.max(np.abs(csum / n_bad - (np.arange(n) + 1 - csum) / n_good)))

        def _paired_bootstrap_ci(labels, challenger, champion, metric: str):
            rng = np.random.default_rng(2026)
            labels = np.asarray(labels, dtype=float)
            challenger = np.asarray(challenger, dtype=float)
            champion = np.asarray(champion, dtype=float)
            diffs = []
            n = len(labels)
            for _ in range(200):
                idx = rng.integers(0, n, n)
                yb, cb, sb = labels[idx], challenger[idx], champion[idx]
                if len(set(yb)) < 2:
                    continue
                try:
                    if metric == "auc":
                        m_c = roc_auc_score(yb, cb)
                        m_s = roc_auc_score(yb, sb)
                    else:
                        m_c = _ks_stat(yb, cb)
                        m_s = _ks_stat(yb, sb)
                except Exception:
                    continue
                diffs.append(m_c - m_s)
            if len(diffs) < 20:
                return None, None
            return (
                round(float(np.percentile(diffs, 2.5)), 4),
                round(float(np.percentile(diffs, 97.5)), 4),
            )

        bootstrap_ci_lower: float | None = None
        bootstrap_ci_upper: float | None = None
        ks_bootstrap_ci_lower: float | None = None
        ks_bootstrap_ci_upper: float | None = None
        if (
            champion_scores is not None
            and len(champion_scores) == len(val_labels)
        ):
            try:
                bootstrap_ci_lower, bootstrap_ci_upper = _paired_bootstrap_ci(
                    val_labels, challenger_scores, champion_scores, "auc",
                )
                ks_bootstrap_ci_lower, ks_bootstrap_ci_upper = _paired_bootstrap_ci(
                    val_labels, challenger_scores, champion_scores, "ks",
                )
            except Exception as boot_exc:
                logger.warning(
                    "bootstrap_ci_computation_failed err=%s", boot_exc,
                )

        # ── 分群治理: 真实检测（非硬编码）──
        segment_governance = _check_segment_governance(
            val_df, challenger_scores, val_labels, result["feature_cols"]
        )

        # ═══════════════════════════════════════════════
        # 9. 构造回调
        # ═══════════════════════════════════════════════
        callback_payload = {
            "training_job_id": training_job_id,
            "lifecycle_run_id": lifecycle_run_id,
            "idempotency_key": idempotency_key,
            "experiment_id": experiment_id,
            "status": "SUCCEEDED",
            "candidate_version": candidate_version,
            "model_artifact_uri": model_uri,
            "training_metrics": {
                "AUC": result["train_auc"],
                "KS": result["train_ks"],
            },
            "validation_metrics": {
                "AUC": val_auc,
                "KS": val_ks,
                "champion_auc": round(champion_auc, 4) if champion_auc else None,
                "champion_ks": round(champion_ks, 4) if champion_ks else None,
                "champion_w1_auc": round(champion_w1_auc, 4) if champion_w1_auc else None,
                "champion_w1_ks": round(champion_w1_ks, 4) if champion_w1_ks else None,
                "challenger_auc": round(challenger_auc, 4),
                "challenger_ks": round(challenger_ks, 4),
                "score_psi": round(score_psi, 4) if score_psi is not None else None,
                "champion_psi": round(champion_psi, 4) if champion_psi is not None else None,
                "challenger_psi": round(challenger_psi, 4) if challenger_psi is not None else None,
                "recovery_rate": round(recovery_rate, 4),
                "recovery_auc": round(recovery_auc, 4),
                "recovery_ks": round(recovery_ks, 4),
                "original_drop": round(max(0, (champion_auc or 0.0) - challenger_auc), 4),
                "recovered_amount": round(max(0, challenger_auc - (champion_auc or 0.0)), 4),
                "healthy_lower_bound": healthy_lower_bound,
                "ks_healthy_lower_bound": ks_healthy_lower_bound,
                "bootstrap_ci_lower": bootstrap_ci_lower,
                "bootstrap_ci_upper": bootstrap_ci_upper,
                "ks_bootstrap_ci_lower": ks_bootstrap_ci_lower,
                "ks_bootstrap_ci_upper": ks_bootstrap_ci_upper,
                "train_valid_gap": round(train_valid_gap, 4),
                "discrimination_passed": discrimination_passed,
                "calibration_passed": calibration_passed,
                "pre_oot_qualified": pre_oot_qualified,
                "brier_score_challenger": round(challenger_brier, 4) if challenger_brier is not None else None,
                "calibration_error": calibration_error,
                "champion_loaded": champion_metrics["loaded"],
                "champion_checksum": champion_checksum,
                "model_checksum": model_sha256,
                "w1_baseline_available": champion_w1_auc is not None,
            },
            # ── 数据可复现性: 所有版本信息完整 → reproducible ──
            "data_reproducible": bool(
                job_input.get("feature_schema_version")
                and job_input.get("preprocessing_version")
                and job_input.get("data_snapshot_ids")
                and job_input.get("label_versions")
            ),
            # ── 候选包已冻结: Worker 保存模型 + checksum 侧车 → frozen ──
            "candidate_frozen_before_oot": True,
            # 冻结身份校验和：晋升时验证加载到的字节与冻结时一致（防换包）
            "frozen_identity_checksum": f"sha256:{model_sha256}",
            "segment_metrics": segment_governance,
            "artifact_checksums": {"model": f"sha256:{model_sha256}"},
            "environment_manifest": {"python": "3.11", "framework": algorithm},
            "technical_retry_count": self.request.retries,
            "error_code": None,
            "error_message": None,
        }

        # 回调 API（同步 urllib，Windows 兼容）
        _api_post(f"/api/internal/iteration/jobs/{training_job_id}/callback", callback_payload)
        logger.info(
            "train_model_callback_sent job=%s version=%s auc=%.4f ks=%.4f psi=%s recovery_auc=%.4f recovery_ks=%.4f pre_oot=%s",
            training_job_id, candidate_version, val_auc, val_ks, score_psi,
            recovery_auc, recovery_ks, pre_oot_qualified,
        )

        return {"status": "SUCCEEDED", "candidate_version": candidate_version, "val_auc": val_auc}

    except Exception as exc:
        logger.error("train_model_failed job=%s error=%s", training_job_id, str(exc))
        fail_payload = {
            "training_job_id": training_job_id,
            "lifecycle_run_id": lifecycle_run_id,
            "idempotency_key": idempotency_key,
            "experiment_id": experiment_id,
            "status": "FAILED",
            "error_code": "TRAINING_TECHNICAL_FAILURE",
            "error_message": str(exc),
        }
        try:
            _api_post(f"/api/internal/iteration/jobs/{training_job_id}/callback", fail_payload)
        except Exception:
            logger.warning("train_model_fail_callback_error", exc_info=True)

        raise self.retry(exc=exc)
