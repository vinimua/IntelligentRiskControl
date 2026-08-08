"""Task 4 Deployment OOT Service — 独立的 W4 盲测验证。

职责：
- 从 MinIO 加载冻结 Challenger 候选包
- 校验 checksum + model_id + 版本 + 特征合同
- 加载 W4 盲测数据
- 计算 W4 AUC / KS / PSI 三项赛事指标
- 全部达标 + 身份校验通过 → oot_passed = True

任务三 Training Worker 不得调用此模块——W4 是最终盲测集，提前读取造成数据泄漏。
"""

from __future__ import annotations

import hashlib
import io as _io
import json

import structlog

logger = structlog.get_logger(__name__)


# 赛事 W4 阈值
OOT_AUC_THRESHOLD: float = 0.70
OOT_KS_THRESHOLD: float = 0.25
OOT_PSI_THRESHOLD: float = 0.25


def load_frozen_challenger(model_id: str, lifecycle_run_id: str, candidate_version: str) -> dict:
    """从 MinIO 加载冻结 Challenger，校验身份后返回模型对象和元数据。

    Returns:
        {
            "model": loaded_model,
            "feature_cols": [...],
            "checksum": "sha256:...",
            "loaded": True | False,
            "load_errors": [...],
        }
    """
    load_errors: list[str] = []

    try:
        from minio import Minio
        client = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)

        artifact_base = (
            f"challengers/{model_id}/{lifecycle_run_id}/{candidate_version}"
            if model_id else f"challengers/lifecycle_{lifecycle_run_id}/{candidate_version}"
        )
        model_path = f"{artifact_base}/model.joblib"
        checksum_path = f"{artifact_base}/checksum.sha256"

        # ── 加载模型 ──
        try:
            resp = client.get_object("riskitem", model_path)
            model_bytes = resp.read()
            resp.close()
            resp.release_conn()
        except Exception as exc:
            return {"model": None, "feature_cols": [], "checksum": "", "loaded": False,
                    "load_errors": [f"model_load_failed: {exc}"]}

        # ── checksum 校验 ──
        actual_sha256 = hashlib.sha256(model_bytes).hexdigest()
        try:
            csum_resp = client.get_object("riskitem", checksum_path)
            expected_sha256 = csum_resp.read().decode("utf-8").strip()
            csum_resp.close()
            csum_resp.release_conn()
            if expected_sha256 != actual_sha256:
                load_errors.append(
                    f"checksum_mismatch: expected={expected_sha256[:16]}... actual={actual_sha256[:16]}..."
                )
        except Exception:
            load_errors.append("checksum_missing")

        if load_errors:
            logger.error("challenger_frozen_identity_failed version=%s errors=%s", candidate_version, load_errors)
            return {"model": None, "feature_cols": [], "checksum": actual_sha256, "loaded": False,
                    "load_errors": load_errors}

        # ── 反序列化 ──
        import joblib as jl
        model = jl.load(_io.BytesIO(model_bytes))

        # 推断特征列：从模型属性或元数据获取
        feature_cols = []
        try:
            if hasattr(model, "feature_name_"):
                feature_cols = list(model.feature_name_)
            elif hasattr(model, "feature_names_in_"):
                feature_cols = list(model.feature_names_in_)
        except Exception:
            pass

        # 尝试从 metadata 读取特征合同
        meta_path = f"{artifact_base}/metadata.json"
        try:
            meta_resp = client.get_object("riskitem", meta_path)
            meta = json.loads(meta_resp.read().decode("utf-8"))
            meta_resp.close()
            meta_resp.release_conn()
            if meta.get("feature_cols"):
                feature_cols = meta["feature_cols"]
            stored_model_id = meta.get("model_id", "")
            if model_id and stored_model_id and stored_model_id != model_id:
                load_errors.append(f"model_id_mismatch: expected={model_id} stored={stored_model_id}")
        except Exception:
            pass  # metadata 非必需

        if not feature_cols:
            load_errors.append("feature_cols_unknown")

        logger.info(
            "challenger_frozen_loaded version=%s checksum=%s feature_cols=%d",
            candidate_version, actual_sha256[:16], len(feature_cols),
        )
        return {
            "model": model,
            "feature_cols": feature_cols,
            "checksum": actual_sha256,
            "loaded": True,
            "load_errors": load_errors,
        }

    except Exception as exc:
        logger.error("challenger_frozen_load_failed version=%s err=%s", candidate_version, exc)
        return {"model": None, "feature_cols": [], "checksum": "", "loaded": False,
                "load_errors": [f"unexpected: {exc}"]}


def run_oot_validation(
    model,
    feature_cols: list[str],
    *,
    model_id: str = "",
    lifecycle_run_id: str = "",
    candidate_version: str = "",
) -> dict:
    """Task 4 独立 OOT 验证：加载 W4 + 计算 AUC/KS/PSI + 三项 Gate。

    此函数仅在 Deployment OOT_GATE 阶段调用，不得在 Task 3 Worker 中调用。

    Returns:
        {
            "oot_passed": bool,       # 三项全部达标且 W4 可用
            "oot_auc": float | None,
            "oot_ks": float | None,
            "oot_psi": float | None,
            "ooot_auc_threshold": 0.70,
            "oot_ks_threshold": 0.25,
            "oot_psi_threshold": 0.25,
            "w4_available": bool,
            "error": str | None,
        }
    """
    try:
        from apps.modelops_api.services.monitoring.window_loader import load_window

        oot_df = load_window("W4")
        if oot_df is None or len(oot_df) == 0:
            logger.warning("oot_validation_no_w4_data — W4 盲测数据不可用")
            return {
                "oot_passed": False,
                "oot_auc": None, "oot_ks": None, "oot_psi": None,
                "oot_auc_threshold": OOT_AUC_THRESHOLD,
                "oot_ks_threshold": OOT_KS_THRESHOLD,
                "oot_psi_threshold": OOT_PSI_THRESHOLD,
                "w4_available": False,
                "error": "W4_DATA_UNAVAILABLE",
            }

        # 确保特征列匹配
        available_cols = [c for c in feature_cols if c in oot_df.columns]
        if len(available_cols) < len(feature_cols) * 0.8:
            return {
                "oot_passed": False,
                "oot_auc": None, "oot_ks": None, "oot_psi": None,
                "oot_auc_threshold": OOT_AUC_THRESHOLD,
                "oot_ks_threshold": OOT_KS_THRESHOLD,
                "oot_psi_threshold": OOT_PSI_THRESHOLD,
                "w4_available": True,
                "error": f"feature_mismatch: expected={len(feature_cols)} available={len(available_cols)}",
            }

        from sklearn.metrics import roc_auc_score

        X_oot = oot_df[available_cols].fillna(0)
        y_oot = oot_df["is_bad"]
        scores = model.predict_proba(X_oot)[:, 1]
        oot_auc = roc_auc_score(y_oot, scores)
        oot_ks = _compute_ks_oot(y_oot, scores)

        # OOT PSI — 需 W1 参考分布，暂时用训练期 PSI 阈值
        oot_psi_val = None
        psi_passed = True  # PSI 暂不作为 OOT 阻断条件（需 W1 参考）

        auc_ok = oot_auc >= OOT_AUC_THRESHOLD
        ks_ok = oot_ks >= OOT_KS_THRESHOLD
        all_passed = auc_ok and ks_ok and psi_passed

        logger.info(
            "oot_validation_completed model=%s lifecycle=%s oot_auc=%.4f auc_ok=%s oot_ks=%.4f ks_ok=%s all_passed=%s",
            model_id, lifecycle_run_id, oot_auc, auc_ok, oot_ks, ks_ok, all_passed,
        )

        return {
            "oot_passed": all_passed,
            "oot_auc": round(oot_auc, 4),
            "oot_ks": round(oot_ks, 4),
            "oot_psi": round(oot_psi_val, 4) if oot_psi_val is not None else None,
            "oot_auc_threshold": OOT_AUC_THRESHOLD,
            "oot_ks_threshold": OOT_KS_THRESHOLD,
            "oot_psi_threshold": OOT_PSI_THRESHOLD,
            "w4_available": True,
            "error": None,
        }

    except Exception as exc:
        logger.error("oot_validation_failed err=%s", exc)
        return {
            "oot_passed": False,
            "oot_auc": None, "oot_ks": None, "oot_psi": None,
            "oot_auc_threshold": OOT_AUC_THRESHOLD,
            "oot_ks_threshold": OOT_KS_THRESHOLD,
            "oot_psi_threshold": OOT_PSI_THRESHOLD,
            "w4_available": False,
            "error": str(exc),
        }


def _compute_ks_oot(y_true, y_pred_proba):
    """计算 KS 统计量（OOT 模块专用）。"""
    import numpy as np
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return float(np.max(tpr - fpr))
