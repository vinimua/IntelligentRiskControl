"""T3-GAP-02: 超参优化 Celery Worker。

执行 N 组 trial 训练 → 比较 val_auc → 选出 best → 回调 API。
"""

from __future__ import annotations

import json
import os
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        logger.warning("api_post_failed url=%s code=%s", url, exc.code)
        raise


@app.task(bind=True, name="workers.tuning_tasks.run_tuning", max_retries=1, default_retry_delay=120)
def run_tuning(self, plan_input: dict):
    """执行超参搜索。

    plan_input 字段：
    - plan_id, lifecycle_run_id, algorithm, trials, training_plan_id
    """
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    plan_id = plan_input["plan_id"]
    lifecycle_run_id = plan_input.get("lifecycle_run_id", "")
    algorithm = plan_input.get("algorithm", "lightgbm")
    trials_data = plan_input.get("trials", [])
    seed = plan_input.get("seed", 2026)
    training_window_ids = list(plan_input.get("training_window_ids") or ["W2"])
    validation_window_ids = list(plan_input.get("validation_window_ids") or ["W3"])
    window_ids = list(dict.fromkeys(training_window_ids + validation_window_ids))
    if "W4" in window_ids:
        raise ValueError("W4/OOT window must never be used for tuning")

    logger.info("tuning_started plan=%s algorithm=%s trials=%d", plan_id, algorithm, len(trials_data))

    try:
        # ── 1. Load data（时间隔离：W2 只训练、W3 只评分，禁止合并随机切分）──
        from apps.modelops_api.services.monitoring.window_loader import load_window

        train_frames: list = []
        val_frames: list = []
        for wid in training_window_ids:
            try:
                train_frames.append(load_window(wid))
            except Exception as exc:
                logger.warning("tuning_load_window_failed window=%s err=%s", wid, exc)
        for wid in validation_window_ids:
            try:
                val_frames.append(load_window(wid))
            except Exception as exc:
                logger.warning("tuning_load_window_failed window=%s err=%s", wid, exc)
        if not train_frames or not val_frames:
            raise ValueError("No data windows loaded for tuning")
        train_df = pd.concat(train_frames, ignore_index=True)
        val_df = pd.concat(val_frames, ignore_index=True)

        exclude = {"sample_id", "apply_time", "is_bad", "y_pred_proba", "risk_score"}
        feature_cols = [c for c in train_df.columns if c not in exclude and train_df[c].dtype in ("int64", "float64")]
        target = "is_bad"
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target]
        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df[target]

        # 训练集可随机降采样提速；验证集保持完整、时间隔离
        if len(X_train) > 10000:
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, train_size=10000,
                random_state=seed, stratify=y_train,
            )
        logger.info("tuning_data_ready train=%d val=%d features=%d", len(X_train), len(X_val), len(feature_cols))

        # ── 2. Run trials ──
        results: list[dict] = []
        best_auc = -1.0
        best_idx = -1
        best_params = {}

        def run_trial(i: int, trial: dict) -> dict:
            params = trial.get("hyperparameters", {})
            trial_id = trial.get("trial_id", str(uuid.uuid4()))
            logger.info("tuning_trial_start idx=%d params=%s", i, params)

            try:
                if algorithm == "lightgbm":
                    model_params = {
                        "objective": "binary", "metric": "auc",
                        "n_estimators": int(params.get("n_estimators", 100)),
                        "max_depth": int(params.get("max_depth", 6)),
                        "learning_rate": float(params.get("learning_rate", 0.05)),
                        "num_leaves": int(params.get("num_leaves", 31)),
                        "min_child_samples": int(params.get("min_child_samples", 20)),
                        "subsample": float(params.get("subsample", 0.8)),
                        "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
                        "random_state": seed, "verbosity": -1, "n_jobs": 1,
                    }
                    model = lgb.LGBMClassifier(**model_params)
                else:
                    # Other algorithms: use default trainer from training_tasks
                    from workers.training_tasks import TRAINERS
                    trainer = TRAINERS.get(algorithm)
                    if trainer is None:
                        return {
                            "trial_id": trial_id,
                            "trial_index": i,
                            "status": "FAILED",
                            "hyperparameters": params,
                            "error_message": f"Unsupported algorithm: {algorithm}",
                        }
                    result = trainer(
                        pd.concat([X_train, y_train], axis=1) if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train, columns=feature_cols).assign(**{target: y_train.values}),
                        target=target, seed=seed, hyperparameters=params,
                    )
                    model = result["model"]

                # Train and evaluate
                if algorithm == "lightgbm":
                    model.fit(X_train, y_train)
                    train_pred = model.predict_proba(X_train)[:, 1]
                    val_pred = model.predict_proba(X_val)[:, 1]
                else:
                    val_pred = model.predict_proba(X_val)[:, 1]
                    train_pred = model.predict_proba(X_train)[:, 1]

                train_auc = roc_auc_score(y_train, train_pred)
                val_auc = roc_auc_score(y_val, val_pred)
                val_ks = _compute_ks(y_val, val_pred)

                logger.info("tuning_trial_done idx=%d val_auc=%.4f", i, val_auc)
                return {
                    "trial_id": trial_id, "trial_index": i, "status": "SUCCEEDED",
                    "hyperparameters": params,
                    "train_auc": round(train_auc, 4), "val_auc": round(val_auc, 4),
                    "val_ks": round(val_ks, 4),
                }

            except Exception as exc:
                logger.error("tuning_trial_failed idx=%d err=%s", i, exc)
                return {
                    "trial_id": trial_id, "trial_index": i, "status": "FAILED",
                    "hyperparameters": params, "error_message": str(exc),
                }

        max_workers = min(5, max(1, len(trials_data)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(run_trial, i, trial)
                for i, trial in enumerate(trials_data)
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        results.sort(key=lambda item: int(item.get("trial_index", 0)))
        for result in results:
            if result.get("status") != "SUCCEEDED":
                continue
            val_auc = float(result.get("val_auc", -1.0))
            if val_auc > best_auc:
                best_auc = val_auc
                best_idx = int(result.get("trial_index", -1))
                best_params = dict(result.get("hyperparameters") or {})

        # ── 3. Callback ──
        callback_payload = {
            "plan_id": plan_id,
            "status": "SUCCEEDED" if best_idx >= 0 else "FAILED",
            "lifecycle_run_id": lifecycle_run_id,
            "algorithm": algorithm,
            "trials": results,
            "best_trial_index": best_idx,
            "best_hyperparameters": best_params,
            "best_val_auc": round(best_auc, 4),
        }
        _api_post(f"/api/internal/iteration/tuning-runs/{plan_id}/callback", callback_payload)
        logger.info("tuning_completed plan=%s best_idx=%d best_auc=%.4f", plan_id, best_idx, best_auc)

        return {"status": "SUCCEEDED", "plan_id": plan_id, "best_val_auc": best_auc}

    except Exception as exc:
        logger.error("tuning_failed plan=%s err=%s", plan_id, str(exc))
        try:
            _api_post(f"/api/internal/iteration/tuning-runs/{plan_id}/callback", {
                "plan_id": plan_id, "status": "FAILED",
                "lifecycle_run_id": lifecycle_run_id, "error_message": str(exc),
            })
        except Exception:
            pass
        raise self.retry(exc=exc)


def _compute_ks(y_true, y_pred_proba):
    import numpy as np
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return float(np.max(tpr - fpr))
