"""W4 盲测晋升门禁服务（Task 4 Deployment OOT Service）。

【系统角色】「模型性能衰减自动修复」系统（A1-A7 七个修复动作：A1关闭、A2观察、
  A3数据修复、A4管道修复、A5校准、A6阈值、A7模型迭代）的最后一道路障。
  完整流程：监控检测 → 诊断定位根因 → Agent决策 → 执行修复 → 资格验证(PRE-OOT)
  → OOT盲测 → 部署晋升。本模块实现其中的「OOT 盲测」环节，是 Challenger（挑战者
  模型）晋升为 Champion（现任冠军模型）之前的最终样本外（Out-Of-Time）验证门禁。

【职责】
- 从 MinIO（或本地 fallback）加载冻结 Challenger 候选包
- 校验 checksum + model_id + lifecycle_run_id + 候选版本 + 特征合同（防身份篡改）
- 加载 W4 盲测数据（最终盲测集）
- 计算 W4 上的 AUC / KS / PSI 三项赛事指标，并与 Champion 做点对点对比
- 依据治理口径做晋升判定（见下方 PRIMARY_METRIC_NON_INFERIOR 等常量）
- 全部达标 + 身份校验通过 → oot_passed = True

【重要红线】任务三（Training Worker）不得调用此模块——W4 是最终盲测集，
  训练阶段提前读取会造成数据泄漏。
"""

from __future__ import annotations

import hashlib
import io as _io
import json
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


# ── OOT 晋升门禁治理口径（人工定稿，见 doc/review 治理规则）──
# 主指标（AUC / KS）：采用「点估计不劣」——修复后不得低于 Champion 点估计。
#   bootstrap 改善是否显著仅作为审计证据记录，不再作为放行硬门。
# 次要指标（Bad Recall）：允许相对下降不超过 2%——
#   由主指标修复导致的次要指标小幅退化可接受，但不能跌破该护栏。
PRIMARY_METRIC_NON_INFERIOR: bool = True
SECONDARY_METRIC_TOLERANCE: float = 0.02


def _prepare_features(frame, feature_cols: list[str]):
    """从原始窗口数据构造模型输入特征矩阵（OOT 模块专用）。

    将 apply_time（申请时间）展开为周期特征（小时/星期 sin-cos 编码、是否周末、
    是否深夜），随后校验特征合同完整性，并对缺失值做 0 填充、对无穷值做 NaN 处理。

    参数:
        frame: 含 apply_time 列的原始窗口 DataFrame。
        feature_cols: 模型要求的特征列清单（特征合同）。

    返回:
        仅含 feature_cols 列、已清洗的特征 DataFrame。

    异常:
        ValueError: 当数据缺少 feature_cols 中某些列时抛出（feature_schema_mismatch）。
    """
    import numpy as np
    import pandas as pd

    data = frame.copy()
    ts = pd.to_datetime(data["apply_time"], errors="raise")
    hour = ts.dt.hour + ts.dt.minute / 60.0
    weekday = ts.dt.weekday
    # 小时与星期的周期编码（sin/cos），避免 0 点/23 点、周日/周一的边界跳变
    data["apply_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["apply_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["apply_weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    data["apply_weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    data["apply_is_weekend"] = (weekday >= 5).astype(float)
    data["apply_is_night"] = ((ts.dt.hour < 6) | (ts.dt.hour >= 22)).astype(float)
    missing = [name for name in feature_cols if name not in data.columns]
    if missing:
        raise ValueError(f"feature_schema_mismatch: missing={missing}")
    return data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)


def _score_psi(reference_scores, current_scores, bins: int = 10) -> float:
    """计算 PSI（Population Stability Index，群体稳定性指标）。

    将分数在 [0,1] 区间等分为 bins 个桶，比较 reference（W1 基准）与 current（W4 当前）
    两套分数分布之间的偏移程度。PSI 越大说明分数分布漂移越严重。
    分母做 1e-6 下限截断，避免空桶导致 log(0) 溢出。

    参数:
        reference_scores: 基准窗口（W1）的预测分数。
        current_scores: 当前窗口（W4）的预测分数。
        bins: 分桶数量，默认 10。

    返回:
        PSI 值（float）。
    """
    import numpy as np

    edges = np.linspace(0, 1, bins + 1)
    reference = np.histogram(reference_scores, bins=edges)[0] / len(reference_scores)
    current = np.histogram(current_scores, bins=edges)[0] / len(current_scores)
    reference = np.clip(reference, 1e-6, None)
    current = np.clip(current, 1e-6, None)
    return float(np.sum((current - reference) * np.log(current / reference)))


def _bad_recall(y_true, scores) -> float:
    """计算 Bad Recall（坏样本召回率，OOT 模块专用）。

    以预测分数 80 分位数作为 cutoff（约等于命中前 20% 高风险样本），
    统计真实坏样本中被该高分段命中的比例。Bad Recall 是晋升门禁中的「次要指标」，
    允许因主指标修复带来不超过 2% 的相对下降（见 SECONDARY_METRIC_TOLERANCE）。

    参数:
        y_true: 真实标签（1 表示坏样本）。
        scores: 模型预测的坏样本概率。

    返回:
        Bad Recall（float，0~1）。

    异常:
        ValueError: 当数据中没有任何坏样本时抛出（W4_BAD_SAMPLE_MISSING）。
    """
    import numpy as np

    y = np.asarray(y_true)
    cutoff = float(np.quantile(scores, 0.8))
    bad = int((y == 1).sum())
    if bad == 0:
        raise ValueError("W4_BAD_SAMPLE_MISSING")
    return float(((y == 1) & (scores >= cutoff)).sum() / bad)


def _paired_bootstrap_ci(y_true, champion_scores, challenger_scores, metric: str, *, seed: int = 2026, rounds: int = 300):
    """计算 Challenger 相对 Champion 指标增量（delta）的配对 bootstrap 置信区间。

    在每一轮 bootstrap 中对同一样本同时重采样 Champion 与 Challenger 的预测，
    计算「新-旧」指标差值，最终取差值的 2.5% / 97.5% 分位数作为 95% 置信区间。
    注意：该区间仅作为审计证据记录，不再作为 OOT 放行的硬门（硬门是点估计不劣）。

    参数:
        y_true: 真实标签。
        champion_scores: Champion 模型分数。
        challenger_scores: Challenger 模型分数。
        metric: 指标名，取值 AUC / KS / BAD_RECALL。
        seed: 随机种子（默认 2026，保证可复现）。
        rounds: bootstrap 重采样轮数（默认 300）。

    返回:
        (ci_low, ci_high) 元组，即 delta 的 95% 置信区间下/上界。

    异常:
        ValueError: 不支持的 metric，或有效重采样轮数不足（BOOTSTRAP_UNAVAILABLE）。
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true)
    champion = np.asarray(champion_scores)
    challenger = np.asarray(challenger_scores)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(rounds):
        idx = rng.integers(0, len(y), len(y))
        sampled_y = y[idx]
        if np.unique(sampled_y).size < 2:
            continue
        if metric == "AUC":
            old = roc_auc_score(sampled_y, champion[idx])
            new = roc_auc_score(sampled_y, challenger[idx])
        elif metric == "KS":
            old = _compute_ks_oot(sampled_y, champion[idx])
            new = _compute_ks_oot(sampled_y, challenger[idx])
        elif metric == "BAD_RECALL":
            old = _bad_recall(sampled_y, champion[idx])
            new = _bad_recall(sampled_y, challenger[idx])
        else:
            raise ValueError(f"unsupported bootstrap metric: {metric}")
        deltas.append(float(new - old))
    if len(deltas) < max(30, rounds // 2):
        raise ValueError(f"BOOTSTRAP_UNAVAILABLE:{metric}")
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def load_frozen_challenger(model_id: str, lifecycle_run_id: str, candidate_version: str) -> dict:
    """从 MinIO 加载冻结 Challenger，校验身份后返回模型对象和元数据。

    冻结身份校验是晋升门禁的第一道防线：必须保证加载到的模型字节、模型 ID、
    生命周期 run_id、候选版本、特征列与预期完全一致，任何不一致都会导致 loaded=False。

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

    # 候选包在对象存储中的逻辑路径，以及本地 fallback 目录（供离线/无 MinIO 环境使用）
    artifact_base = f"challengers/{model_id}/{lifecycle_run_id}/{candidate_version}"
    local_base = (
        Path(__file__).resolve().parents[4]
        / "artifacts"
        / "minio_fallback"
        / "riskitem"
        / artifact_base
    )
    local_model = local_base / "model.joblib"
    local_checksum = local_base / "checksum.sha256"
    local_metadata = local_base / "metadata.json"
    # ── 优先尝试本地 fallback 加载 ──
    if local_model.is_file():
        try:
            model_bytes = local_model.read_bytes()
            actual_sha256 = hashlib.sha256(model_bytes).hexdigest()
            if not local_checksum.is_file() or local_checksum.read_text(encoding="utf-8").strip() != actual_sha256:
                raise ValueError("local_checksum_missing_or_mismatch")
            if not local_metadata.is_file():
                raise ValueError("local_metadata_missing")
            meta = json.loads(local_metadata.read_text(encoding="utf-8"))
            if meta.get("model_id") != model_id:
                raise ValueError("local_model_id_mismatch")
            if meta.get("lifecycle_run_id") != lifecycle_run_id:
                raise ValueError("local_lifecycle_run_id_mismatch")
            if meta.get("candidate_version") != candidate_version:
                raise ValueError("local_candidate_version_mismatch")
            if meta.get("model_sha256") != actual_sha256:
                raise ValueError("local_metadata_checksum_mismatch")
            feature_cols = list(meta.get("feature_cols") or [])
            if not feature_cols:
                raise ValueError("local_feature_cols_unknown")
            import joblib as jl

            return {
                "model": jl.load(_io.BytesIO(model_bytes)),
                "feature_cols": feature_cols,
                "checksum": actual_sha256,
                "loaded": True,
                "load_errors": [],
            }
        except Exception as exc:
            return {
                "model": None,
                "feature_cols": [],
                "checksum": "",
                "loaded": False,
                "load_errors": [f"local_frozen_identity_failed:{exc}"],
            }

    # ── 走 MinIO 对象存储加载 ──
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
        metadata_loaded = False
        try:
            meta_resp = client.get_object("riskitem", meta_path)
            meta = json.loads(meta_resp.read().decode("utf-8"))
            meta_resp.close()
            meta_resp.release_conn()
            if meta.get("feature_cols"):
                feature_cols = meta["feature_cols"]
            metadata_loaded = True
            stored_model_id = meta.get("model_id", "")
            if stored_model_id != model_id:
                load_errors.append(f"model_id_mismatch: expected={model_id} stored={stored_model_id}")
            if meta.get("lifecycle_run_id") != lifecycle_run_id:
                load_errors.append("lifecycle_run_id_mismatch")
            if meta.get("candidate_version") != candidate_version:
                load_errors.append("candidate_version_mismatch")
            if meta.get("model_sha256") != actual_sha256:
                load_errors.append("metadata_model_checksum_mismatch")
        except Exception:
            pass  # metadata 非必需

        if not metadata_loaded:
            load_errors.append("metadata_missing_or_invalid")
        if not feature_cols:
            load_errors.append("feature_cols_unknown")

        if load_errors:
            return {
                "model": None,
                "feature_cols": [],
                "checksum": actual_sha256,
                "loaded": False,
                "load_errors": load_errors,
            }

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
    champion_version: str = "champion_v1",
) -> dict:
    """Task 4 独立 OOT 验证：加载 W4 + 计算 AUC/KS/PSI + 三项 Gate。

    此函数仅在 Deployment OOT_GATE 阶段调用，不得在 Task 3 Worker 中调用。

    【核心判断口径】
    - 主指标（AUC / KS）：点估计不劣，即 challenger >= champion（修复后允许持平，
      不允许更差）；bootstrap CI 仅作为审计证据写入返回结果。
    - 次要指标（Bad Recall）：允许相对下降不超过 SECONDARY_METRIC_TOLERANCE(0.02)。
    - PSI：oot_psi_val <= 0.20。
    三项同时满足才 oot_passed=True。

    Returns:
        {
            "oot_passed": bool,       # 三项全部达标且 W4 可用
            "oot_auc": float | None,
            "oot_ks": float | None,
            "oot_psi": float | None,
            "oot_auc_threshold": 0.70,
            "oot_ks_threshold": 0.25,
            "oot_psi_threshold": 0.25,
            "w4_available": bool,
            "error": str | None,
        }
    """
    try:
        from apps.modelops_api.services.monitoring.window_loader import (
            load_champion_model,
            load_window,
        )

        # 读取最终盲测集 W4——训练阶段禁止读取此窗口（防数据泄漏）
        oot_df = load_window("W4")
        if oot_df is None or len(oot_df) == 0:
            logger.warning("oot_validation_no_w4_data — W4 盲测数据不可用")
            return {
                "oot_passed": False,
                "oot_auc": None, "oot_ks": None, "oot_psi": None,
                "w4_available": False,
                "oot_metrics_available": False,
                "error": "W4_DATA_UNAVAILABLE", "w4_read_count": 1,
            }

        from sklearn.metrics import roc_auc_score

        # ── Challenger 在 W4 盲测集上的指标 ──
        X_oot = _prepare_features(oot_df, feature_cols)
        y_oot = oot_df["is_bad"]
        scores = model.predict_proba(X_oot)[:, 1]
        oot_auc = roc_auc_score(y_oot, scores)
        oot_ks = _compute_ks_oot(y_oot, scores)

        # ── PSI：以 W1 基准分布对比 W4 当前分布 ──
        w1_df = load_window("W1")
        X_w1 = _prepare_features(w1_df, feature_cols)
        w1_scores = model.predict_proba(X_w1)[:, 1]
        oot_psi_val = _score_psi(w1_scores, scores)

        # ── Champion 基线：在同一 W4 盲测集上做点对点对比 ──
        champion_model, _, champion_features = load_champion_model(model_id)
        if champion_features != feature_cols:
            raise ValueError("CHAMPION_CHALLENGER_FEATURE_SCHEMA_MISMATCH")
        champion_scores = champion_model.predict_proba(X_oot)[:, 1]
        champion_auc = roc_auc_score(y_oot, champion_scores)
        champion_ks = _compute_ks_oot(y_oot, champion_scores)
        challenger_bad_recall = _bad_recall(y_oot, scores)
        champion_bad_recall = _bad_recall(y_oot, champion_scores)
        # bootstrap 差值置信区间——仅作审计证据，不参与硬门判定
        auc_ci = _paired_bootstrap_ci(y_oot, champion_scores, scores, "AUC")
        ks_ci = _paired_bootstrap_ci(y_oot, champion_scores, scores, "KS")
        bad_recall_ci = _paired_bootstrap_ci(
            y_oot, champion_scores, scores, "BAD_RECALL"
        )

        # ── 治理口径（人工定稿 2026-08-15）──
        # 主指标 AUC / KS：点估计不劣，即 challenger >= champion（可接受修复后持平）。
        #   bootstrap CI 只作为审计证据记录，不再作为放行硬门。
        # 次要指标 Bad Recall：允许相对下降不超过 SECONDARY_METRIC_TOLERANCE。
        # 主指标硬门：点估计不劣（challenger >= champion）
        auc_ok = (oot_auc is not None) and (oot_auc >= champion_auc)
        ks_ok = (oot_ks is not None) and (oot_ks >= champion_ks)
        # 次要指标护栏：challenger 的 Bad Recall 不得低于 champion * (1 - 2%)
        bad_recall_floor = (
            champion_bad_recall * (1 - SECONDARY_METRIC_TOLERANCE)
            if champion_bad_recall is not None
            else None
        )
        bad_recall_ok = (
            challenger_bad_recall is not None
            and bad_recall_floor is not None
            and challenger_bad_recall >= bad_recall_floor
        )
        # PSI 门槛：分数分布漂移不得超过 0.20
        psi_passed = oot_psi_val <= 0.20
        # 四项条件（AUC 不劣、KS 不劣、Bad Recall 容忍、PSI 达标）同时成立才放行
        all_passed = auc_ok and ks_ok and bad_recall_ok and psi_passed

        logger.info(
            "oot_validation_completed model=%s lifecycle=%s oot_auc=%.4f auc_ok=%s oot_ks=%.4f ks_ok=%s bad_recall_ok=%s all_passed=%s",
            model_id, lifecycle_run_id, oot_auc, auc_ok, oot_ks, ks_ok,
            bad_recall_ok, all_passed,
        )

        return {
            "oot_passed": all_passed,
            "oot_auc": round(oot_auc, 4),
            "oot_ks": round(oot_ks, 4),
            "oot_psi": round(oot_psi_val, 4) if oot_psi_val is not None else None,
            "champion_oot_auc": round(champion_auc, 4),
            "champion_oot_ks": round(champion_ks, 4),
            "challenger_bad_recall": round(challenger_bad_recall, 4),
            "champion_bad_recall": round(champion_bad_recall, 4),
            "bad_recall_floor": round(bad_recall_floor, 6) if bad_recall_floor is not None else None,
            "bad_recall_relative_drop": (
                round(1 - challenger_bad_recall / champion_bad_recall, 6)
                if challenger_bad_recall is not None and champion_bad_recall
                else None
            ),
            # 以下三项 bootstrap 区间仅供审计留痕，非放行依据
            "auc_delta_bootstrap_ci": [round(auc_ci[0], 6), round(auc_ci[1], 6)],
            "ks_delta_bootstrap_ci": [round(ks_ci[0], 6), round(ks_ci[1], 6)],
            "bad_recall_delta_bootstrap_ci": [
                round(bad_recall_ci[0], 6), round(bad_recall_ci[1], 6)
            ],
            "governance": {
                "primary_metric_rule": "POINT_ESTIMATE_NON_INFERIOR",
                "secondary_metric_tolerance": SECONDARY_METRIC_TOLERANCE,
                "auc_ok": auc_ok,
                "ks_ok": ks_ok,
                "bad_recall_ok": bad_recall_ok,
                "psi_ok": psi_passed,
            },
            "w4_available": True,
            "oot_metrics_available": True,
            "w4_read_count": 1,
            "error": None,
        }

    except Exception as exc:
        logger.error("oot_validation_failed err=%s", exc)
        return {
            "oot_passed": False,
            "oot_auc": None, "oot_ks": None, "oot_psi": None,
            "w4_available": False,
            "oot_metrics_available": False,
            "w4_read_count": 1,
            "error": str(exc),
        }


def _compute_ks_oot(y_true, y_pred_proba):
    """计算 KS 统计量（OOT 模块专用）。

    KS = max(TPR - FPR)，即 ROC 曲线上真正率与假正率差值的最大值，
    用于衡量模型区分好坏的判别能力。
    """
    import numpy as np
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return float(np.max(tpr - fpr))
