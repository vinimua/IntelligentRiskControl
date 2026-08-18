"""模型性能衰减自动修复系统 —— 执行层：A3-A6 修复动作的实际执行与资格判定。

本模块处于「监控检测异常 → 诊断定位根因 → Agent决策选动作 → 执行修复 → 资格验证
→ OOT盲测 → 部署晋升」全流程中的「执行修复」与「执行后资格判定」环节，是真正
落地修复动作的代码。它覆盖七个动作中的 A3-A6：

- A3 数据修复（DATA_REPAIR）：基于 W2 参考统计构造派生视图，修复缺失/异常字段，
  绝不改动或删除原始行；
- A4 管道修复（PIPELINE_REPAIR）：用单独校验和的「可信快照」按 sample_id 回填受
  影响字段；
- A5 校准调整（CALIBRATION_ADJUSTMENT）：在充足样本上用 Isotonic/Platt 重拟合
  校准器，并校验校准改善与排序护栏；
- A6 阈值调整（THRESHOLD_ADJUSTMENT）：业务目标变更后，在拟合集上搜索最优决策
  阈值。

本模块刻意不依赖 Celery 或 HTTP，保证 worker 进程与确定性集成测试共享同一份实现。
每个成功结果都携带不可变输入身份（checksum）、输出校验和与真实重放指标；
任何输入缺失都「fail closed」（立即抛错拒绝执行），宁可不执行也不带病上线。

执行产物随后交给 qualify_repair（A3/A4）与 qualify_adjustment（A5/A6）做资格判定，
判断这次修复是否「真的修好了」，作为进入下一道资格门/晋升流程的依据。
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# 项目根目录（向上四级定位到仓库根），用于解析相对路径的工件 URI。
PROJECT_ROOT = Path(__file__).resolve().parents[4]
# A3-A7 的 pre-OOT 阶段禁止读取 W4 盲测窗口数据（防止用未来数据「作弊」）。
FORBIDDEN_W4_TOKEN = "W4"


def _sha256(payload: bytes) -> str:
    """计算字节流的 SHA-256 摘要，返回带 'sha256:' 前缀的字符串。

    用于校验输入快照未被篡改，以及为输出工件生成身份校验和。
    """
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _local_path(uri: str) -> Path:
    """把工件 URI 解析为本地文件路径。

    支持三种形态：
    - file:// 协议（含 netloc 的 file://host/path 形式）；
    - 绝对路径；
    - 相对路径（相对 PROJECT_ROOT 解析）。
    空 URI 直接抛错（fail closed）。
    """
    if not uri:
        raise ValueError("ARTIFACT_URI_REQUIRED")
    if uri.startswith("file://"):
        parsed = urlparse(uri)
        path = Path(parsed.path.lstrip("/") if parsed.netloc else parsed.path)
        if parsed.netloc:
            path = Path(f"{parsed.netloc}/{parsed.path.lstrip('/')}")
        return path
    path = Path(uri)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _minio_client():
    """懒加载构造 MinIO 客户端（只在真正访问 s3:// 时才导入依赖与配置）。

    延迟导入避免在无对象存储环境下启动时就报错。
    """
    from minio import Minio

    from ...config import settings

    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )


def _read_bytes(uri: str) -> bytes:
    """按 URI 读取工件字节：s3:// 走 MinIO，否则走本地文件。

    s3 URI 缺少 bucket 或 key 时抛错；本地文件不存在时抛 FileNotFoundError。
    """
    if uri.startswith("s3://"):
        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            raise ValueError("S3_URI_INVALID")
        response = _minio_client().get_object(bucket, key)
        try:
            return response.read()
        finally:
            # 确保连接被释放，避免泄漏。
            response.close()
            response.release_conn()
    path = _local_path(uri)
    if not path.is_file():
        raise FileNotFoundError(f"ARTIFACT_NOT_FOUND:{path}")
    return path.read_bytes()


def _write_bytes(uri: str, payload: bytes, content_type: str) -> None:
    """把字节写入目标 URI：s3:// 走 MinIO（自动建桶），否则写本地文件（自动建目录）。"""
    if uri.startswith("s3://"):
        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        client = _minio_client()
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
        client.put_object(
            bucket,
            key,
            io.BytesIO(payload),
            length=len(payload),
            content_type=content_type,
        )
        return
    path = _local_path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _read_json(uri: str) -> dict:
    """读取 JSON 工件并解析为字典。"""
    return json.loads(_read_bytes(uri).decode("utf-8"))


def _read_frame(uri: str, expected_checksum: str | None = None) -> tuple[pd.DataFrame, str]:
    """读取并校验一个 Parquet 数据快照，返回 (DataFrame, 实际校验和)。

    多道防御校验（全部 fail closed）：
    1. 路径含 W4 令牌 → 拒绝访问（pre-OOT 阶段禁止看盲测数据）；
    2. 校验和比对：期望校验和与实算不一致 → 拒绝（快照被篡改或版本错配）；
    3. 必需列 sample_id / apply_time / is_bad 缺失 → 拒绝；
    4. sample_id 重复 → 拒绝（样本身份不唯一）；
    5. is_bad 标签缺失 → 拒绝；
    6. 标签域必须是 {0,1} 且至少两类 → 拒绝（否则无法计算 AUC 等指标）。
    """
    # 防泄漏：A3-A7 的 pre-OOT 阶段任何数据都不得来自 W4 盲测窗口。
    if FORBIDDEN_W4_TOKEN in Path(urlparse(uri).path).parts:
        raise ValueError("W4_ACCESS_FORBIDDEN_FOR_A3_A7_PRE_OOT")
    payload = _read_bytes(uri)
    checksum = _sha256(payload)
    if expected_checksum and checksum != expected_checksum:
        raise ValueError(
            f"SNAPSHOT_CHECKSUM_MISMATCH:expected={expected_checksum}:actual={checksum}"
        )
    frame = pd.read_parquet(io.BytesIO(payload))
    required = {"sample_id", "apply_time", "is_bad"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"SNAPSHOT_REQUIRED_COLUMNS_MISSING:{missing}")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("SNAPSHOT_DUPLICATE_SAMPLE_ID")
    if frame["is_bad"].isna().any():
        raise ValueError("SNAPSHOT_LABEL_MISSING")
    labels = set(pd.to_numeric(frame["is_bad"], errors="raise").astype(int).unique())
    if not labels.issubset({0, 1}) or len(labels) < 2:
        raise ValueError(f"SNAPSHOT_LABEL_DOMAIN_INVALID:{sorted(labels)}")
    return frame, checksum


def _bundle(plan: dict) -> dict:
    """加载 Champion 模型捆绑包（模型、校准器、特征 schema、训练 manifest）。

    校验项：
    - model_id 与 champion_version 必须存在；
    - 四个必需文件齐全；
    - manifest/schema 中的 model_id 与版本必须一致；
    - 若提供了期望的模型校验和，必须匹配。

    返回包含根路径、已加载模型/校准器、schema、manifest 及校验和的字典。
    """
    model_id = str(plan.get("model_id") or "")
    version = str(plan.get("champion_version") or "")
    if not model_id or not version:
        raise ValueError("MODEL_ID_AND_CHAMPION_VERSION_REQUIRED")
    root = (
        _local_path(str(plan["champion_bundle_uri"]))
        if plan.get("champion_bundle_uri")
        else PROJECT_ROOT / "assets" / "champion_models" / model_id / version
    )
    required = [
        root / "model.joblib",
        root / "calibrator.joblib",
        root / "feature_schema.json",
        root / "training_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CHAMPION_BUNDLE_INCOMPLETE:{missing}")
    model_payload = required[0].read_bytes()
    calibrator_payload = required[1].read_bytes()
    schema = json.loads(required[2].read_text(encoding="utf-8"))
    manifest = json.loads(required[3].read_text(encoding="utf-8"))
    if manifest.get("model_id") != model_id or schema.get("model_id") != model_id:
        raise ValueError("CHAMPION_MODEL_ID_MISMATCH")
    if schema.get("model_version") != version:
        raise ValueError("CHAMPION_VERSION_MISMATCH")
    model_checksum = _sha256(model_payload)
    expected = plan.get("champion_model_checksum")
    if expected and expected != model_checksum:
        raise ValueError("CHAMPION_MODEL_CHECKSUM_MISMATCH")
    return {
        "root": root,
        "model": joblib.load(io.BytesIO(model_payload)),
        "calibrator": joblib.load(io.BytesIO(calibrator_payload)),
        "schema": schema,
        "manifest": manifest,
        "model_checksum": model_checksum,
        "calibrator_checksum": _sha256(calibrator_payload),
    }


def _feature_frame(frame: pd.DataFrame, ordered_features: list[str]) -> pd.DataFrame:
    """从原始快照构造特征矩阵（与 Champion 线上特征工程对齐）。

    关键点：
    - 由 apply_time 派生时间周期特征（小时/星期的 sin/cos、周末、深夜标志）；
    - 只保留 schema 声明且存在的派生特征，缺失必需特征即报错；
    - 缺失值交由「已冻结的 Champion 管道」负责填充，这里不做静默填补，
      只把 inf/-inf 替换为 NaN，避免静默篡改数据语义。
    """
    result = frame.copy()
    ts = pd.to_datetime(result["apply_time"], errors="raise")
    hour = ts.dt.hour + ts.dt.minute / 60.0
    weekday = ts.dt.weekday
    derived = {
        "apply_hour_sin": np.sin(2 * np.pi * hour / 24),
        "apply_hour_cos": np.cos(2 * np.pi * hour / 24),
        "apply_weekday_sin": np.sin(2 * np.pi * weekday / 7),
        "apply_weekday_cos": np.cos(2 * np.pi * weekday / 7),
        "apply_is_weekend": (weekday >= 5).astype(float),
        "apply_is_night": ((ts.dt.hour < 6) | (ts.dt.hour >= 22)).astype(float),
    }
    for name, values in derived.items():
        if name in ordered_features:
            result[name] = values
    missing = sorted(set(ordered_features).difference(result.columns))
    if missing:
        raise ValueError(f"FEATURE_SCHEMA_NOT_CONSUMED:{missing}")
    # The frozen Champion Pipeline owns imputation.  Do not silently fill here.
    return result[ordered_features].replace([np.inf, -np.inf], np.nan)


def _raw_scores(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """用 Champion 模型对特征矩阵打分，返回正类（坏样本）的原始概率分。"""
    features = list(bundle["schema"].get("ordered_features") or [])
    if not features:
        raise ValueError("ORDERED_FEATURES_EMPTY")
    return np.asarray(
        bundle["model"].predict_proba(_feature_frame(frame, features))[:, 1],
        dtype=float,
    )


def _apply_calibrator(calibrator, scores: np.ndarray) -> np.ndarray:
    """把原始分送入校准器，得到校准后的概率，并裁剪到 [0,1]。

    校准器接口兼容两种：predict_proba（概率型）与 predict（分值型）。
    两种都缺则抛错。
    """
    if hasattr(calibrator, "predict_proba"):
        values = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
    elif hasattr(calibrator, "predict"):
        values = calibrator.predict(scores)
    else:
        raise ValueError("CALIBRATOR_INTERFACE_UNSUPPORTED")
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def _ks(y_true, scores) -> float:
    """计算 KS 统计量：ROC 曲线上 TPR-FPR 的最大差值（判别能力）。"""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.max(tpr - fpr))


def _ece(y_true, scores, bins: int = 10) -> float:
    """计算期望校准误差（Expected Calibration Error, ECE）。

    把预测概率分成 bins 个等宽区间，逐桶计算「预测均值与真实均值的绝对差」，
    按桶占比加权求和。ECE 越低说明概率估计越准。
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(scores, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assigned = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for index in range(bins):
        mask = assigned == index
        if mask.any():
            total += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def _score_psi(expected_scores, actual_scores) -> float:
    """计算两批分数的 PSI（Population Stability Index，人群稳定指数）。

    以 expected 的十分位数为分箱边界，比较 expected 与 actual 的分布差异，
    分箱占比做 1e-6 下限裁剪避免 log(0)。PSI 越大说明分布漂移越严重。
    """
    expected = np.asarray(expected_scores, dtype=float)
    actual = np.asarray(actual_scores, dtype=float)
    quantiles = np.unique(np.quantile(expected, np.linspace(0.0, 1.0, 11)))
    if len(quantiles) < 3:
        return 0.0
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    expected_pct = np.histogram(expected, bins=quantiles)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=quantiles)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _ranking(y_true, scores) -> dict:
    """一次算齐四个核心指标：AUC、KS、Brier、ECE。

    返回字典供修复前后对比与资格判定使用。
    """
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "ks": _ks(y_true, scores),
        "brier": float(brier_score_loss(y_true, scores)),
        "ece": _ece(y_true, scores),
    }


def _fit_calibrator(raw_scores: np.ndarray, labels, method: str):
    """按指定方法拟合校准器：isotonic（保序回归）或 platt（逻辑回归）。

    不支持的 method 抛错。
    """
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(raw_scores, labels)
    if method == "platt":
        return LogisticRegression(solver="lbfgs", max_iter=1000).fit(
            raw_scores.reshape(-1, 1), labels
        )
    raise ValueError(f"CALIBRATOR_TYPE_UNSUPPORTED:{method}")


def _segment_brier(frame: pd.DataFrame, old_scores, new_scores) -> dict:
    """按 city_tier 分段统计校准器替换前后的 Brier 分数变化。

    仅统计样本量 >=200 且坏样本 >=20 的分段（小分段统计不可靠，跳过）。
    返回每个分段的样本量、坏样本量、前后 Brier 与 delta（改善为负值）。
    """
    if "city_tier" not in frame.columns:
        return {}
    result = {}
    for value, group in frame.groupby("city_tier", dropna=False):
        if len(group) < 200 or int(group["is_bad"].sum()) < 20:
            continue
        indices = group.index.to_numpy()
        # Frames are reset before this function, so positional and label index agree.
        before = float(brier_score_loss(group["is_bad"], np.asarray(old_scores)[indices]))
        after = float(brier_score_loss(group["is_bad"], np.asarray(new_scores)[indices]))
        result[str(value)] = {
            "sample_count": int(len(group)),
            "bad_count": int(group["is_bad"].sum()),
            "before_brier": before,
            "after_brier": after,
            "delta": after - before,
        }
    return result


def execute_calibration(plan: dict) -> dict:
    """执行 A5 校准调整：在充足样本上重拟合校准器，输出新校准器工件与指标。

    关键参数：
        plan: 校准计划，含三个快照（fit/validation/healthy）的 URI 与校验和、
              calibrator_type、artifact_output_path 等。

    关键逻辑：
    1. 读取并校验三个快照；fit 与 validation 样本不允许重叠（防评估污染）；
    2. 依据 fit 样本量/坏样本量选择方法：>=2000/50 用 isotonic，>=1000/20 用 platt，
       否则样本不足直接拒绝；
    3. 用 Champion 模型打分 → 用旧校准器打分作基线 → 拟合新校准器 → 重新打分；
    4. 计算前后 Brier/ECE 改善、AUC/KS 变化、分数 PSI 变化与分段护栏；
    5. 序列化新校准器写入输出工件，返回指标与消费回执。

    返回：包含 status、artifact_uri、artifact_checksum、metrics、receipt 的字典。
    """
    plan_id = str(plan.get("calibration_plan_id") or "")
    if not plan_id:
        raise ValueError("CALIBRATION_PLAN_ID_REQUIRED")
    bundle = _bundle(plan)
    fit, fit_checksum = _read_frame(
        str(plan.get("fit_snapshot_uri") or ""), plan.get("fit_snapshot_checksum")
    )
    validation, validation_checksum = _read_frame(
        str(plan.get("validation_snapshot_uri") or ""),
        plan.get("validation_snapshot_checksum"),
    )
    healthy, healthy_checksum = _read_frame(
        str(plan.get("healthy_snapshot_uri") or ""),
        plan.get("healthy_snapshot_checksum"),
    )
    # 拟合集与验证集样本不允许重叠，否则「验证」会泄漏拟合信息。
    overlap = set(fit["sample_id"].astype(str)) & set(validation["sample_id"].astype(str))
    if overlap:
        raise ValueError(f"CALIBRATION_VALIDATION_SAMPLE_OVERLAP:{len(overlap)}")
    fit_rows, fit_bads = len(fit), int(fit["is_bad"].sum())
    if fit_rows >= 2000 and fit_bads >= 50:
        method = "isotonic"
    elif fit_rows >= 1000 and fit_bads >= 20:
        method = "platt"
    else:
        raise ValueError(
            f"CALIBRATION_SAMPLE_INSUFFICIENT:rows={fit_rows}:bads={fit_bads}"
        )
    requested = str(plan.get("calibrator_type") or "isotonic").lower()
    if requested not in {"isotonic", "platt"}:
        raise ValueError(f"CALIBRATOR_TYPE_UNSUPPORTED:{requested}")
    # Formal A5 contract: sufficient data always uses Isotonic.  Platt is only
    # the mandatory downgrade for the 1000/20..1999/49 sample band; callers may
    # not force a downgrade when Isotonic is eligible.
    fit_raw = _raw_scores(bundle, fit)
    valid_raw = _raw_scores(bundle, validation)
    healthy_raw = _raw_scores(bundle, healthy)
    old_valid = _apply_calibrator(bundle["calibrator"], valid_raw)
    old_healthy = _apply_calibrator(bundle["calibrator"], healthy_raw)
    candidate = _fit_calibrator(fit_raw, fit["is_bad"], method)
    new_valid = _apply_calibrator(candidate, valid_raw)
    new_healthy = _apply_calibrator(candidate, healthy_raw)
    # 验证集上比较新旧校准器；健康集上记录 W1 基线（供后续资格门对比）。
    before = _ranking(validation["is_bad"], old_valid)
    after = _ranking(validation["is_bad"], new_valid)
    healthy_metrics = _ranking(healthy["is_bad"], old_healthy)
    old_score_psi = _score_psi(old_healthy, old_valid)
    new_score_psi = _score_psi(new_healthy, new_valid)
    validation_reset = validation.reset_index(drop=True)
    segment_metrics = _segment_brier(validation_reset, old_valid, new_valid)
    # 分段护栏：每个分段的新 Brier 不得比旧 Brier 差超过 0.005，
    # 防止「整体改善但某个分段恶化」。
    segment_guardrail_passed = all(
        item["after_brier"] <= item["before_brier"] + 0.005
        for item in segment_metrics.values()
    )
    metrics = {
        "method": method,
        "before": before,
        "after": after,
        "healthy_w1": healthy_metrics,
        "brier_improvement": before["brier"] - after["brier"],
        "ece_improvement": before["ece"] - after["ece"],
        "auc_delta": after["auc"] - before["auc"],
        "ks_delta": after["ks"] - before["ks"],
        "old_score_psi": old_score_psi,
        "new_score_psi": new_score_psi,
        # 健康区间上界 = 健康基线 * 1.10 + 0.005，留 10% 裕度。
        "healthy_brier_upper_bound": healthy_metrics["brier"] * 1.10 + 0.005,
        "healthy_ece_upper_bound": healthy_metrics["ece"] * 1.10 + 0.005,
        "segment_metrics": segment_metrics,
        "segment_guardrail_passed": segment_guardrail_passed,
    }
    artifact_buffer = io.BytesIO()
    joblib.dump(candidate, artifact_buffer)
    artifact_payload = artifact_buffer.getvalue()
    artifact_uri = str(plan.get("artifact_output_path") or "")
    if not artifact_uri:
        raise ValueError("CALIBRATION_ARTIFACT_OUTPUT_REQUIRED")
    _write_bytes(artifact_uri, artifact_payload, "application/octet-stream")
    # 消费回执：记录全部输入身份与校验和，供资格判定与审计追溯。
    receipt = {
        "fit_snapshot_id": plan.get("fit_snapshot_id"),
        "fit_snapshot_uri": plan.get("fit_snapshot_uri"),
        "fit_snapshot_checksum": fit_checksum,
        "validation_snapshot_id": plan.get("validation_snapshot_id"),
        "validation_snapshot_uri": plan.get("validation_snapshot_uri"),
        "validation_snapshot_checksum": validation_checksum,
        "healthy_snapshot_id": plan.get("healthy_snapshot_id"),
        "healthy_snapshot_uri": plan.get("healthy_snapshot_uri"),
        "healthy_snapshot_checksum": healthy_checksum,
        "fit_sample_count": fit_rows,
        "fit_bad_count": fit_bads,
        "validation_sample_count": int(len(validation)),
        "validation_bad_count": int(validation["is_bad"].sum()),
        "sample_overlap_count": 0,
        "champion_model_checksum": bundle["model_checksum"],
        "champion_calibrator_checksum": bundle["calibrator_checksum"],
        "w4_read_count": 0,
    }
    return {
        "status": "SUCCEEDED",
        "plan_id": plan_id,
        "artifact_uri": artifact_uri,
        "artifact_checksum": _sha256(artifact_payload),
        "metrics": metrics,
        "consumption_receipt": receipt,
    }


def _threshold_metrics(labels, scores, threshold: float) -> dict:
    """在给定阈值下计算业务指标：F1、precision、recall、高风险率。

    预测规则：score >= threshold 判为正类（高风险）。
    """
    predictions = np.asarray(scores) >= threshold
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "high_risk_rate": float(predictions.mean()),
    }


def execute_threshold_search(plan: dict) -> dict:
    """执行 A6 阈值调整：在拟合集上搜索最优决策阈值，输出阈值工件与指标。

    关键参数：
        plan: 阈值计划，含 fit/validation 快照、search_range、search_metric、
              current_threshold、authorization_id、artifact_output_path 等。

    前置强制条件：
    - 必须声明 business_objective_changed（业务目标确实变了，才允许动阈值）；
    - 必须携带 authorization_id（阈值调整需人工授权，防自动化乱调）。

    关键逻辑：
    1. 读取并校验快照，fit 与 validation 不允许重叠；
    2. 在 [min, max] 内按 step 网格搜索，拟合集上选 F1 最大（并列时依次比
       recall、precision、再取更小阈值）的阈值；
    3. 在验证集上对比新旧阈值的业务指标，并保留排序指标（ranking_unchanged）
       证明阈值调整没有改变模型排序能力；
    4. 输出阈值工件（JSON）与消费回执。

    返回：包含 status、artifact_uri、artifact_checksum、metrics、receipt 的字典。
    """
    plan_id = str(plan.get("threshold_plan_id") or "")
    if not plan_id:
        raise ValueError("THRESHOLD_PLAN_ID_REQUIRED")
    if not bool(plan.get("business_objective_changed")):
        raise ValueError("A6_REAL_BUSINESS_OBJECTIVE_CHANGE_REQUIRED")
    if not plan.get("authorization_id"):
        raise ValueError("A6_MANUAL_AUTHORIZATION_REQUIRED")
    bundle = _bundle(plan)
    fit, fit_checksum = _read_frame(
        str(plan.get("fit_snapshot_uri") or ""), plan.get("fit_snapshot_checksum")
    )
    validation, validation_checksum = _read_frame(
        str(plan.get("validation_snapshot_uri") or ""),
        plan.get("validation_snapshot_checksum"),
    )
    overlap = set(fit["sample_id"].astype(str)) & set(validation["sample_id"].astype(str))
    if overlap:
        raise ValueError(f"THRESHOLD_VALIDATION_SAMPLE_OVERLAP:{len(overlap)}")
    fit_scores = _apply_calibrator(bundle["calibrator"], _raw_scores(bundle, fit))
    validation_scores = _apply_calibrator(
        bundle["calibrator"], _raw_scores(bundle, validation)
    )
    search = dict(plan.get("search_range") or {})
    minimum = float(search.get("min", 0.01))
    maximum = float(search.get("max", 0.99))
    step = float(search.get("step", 0.01))
    if not (0 <= minimum < maximum <= 1 and 0 < step <= maximum - minimum):
        raise ValueError("THRESHOLD_SEARCH_RANGE_INVALID")
    metric = str(plan.get("search_metric") or "F1").upper()
    if metric != "F1":
        raise ValueError(f"THRESHOLD_SEARCH_METRIC_NOT_IMPLEMENTED:{metric}")
    candidates = np.arange(minimum, maximum + step / 2, step)
    fit_results = [
        _threshold_metrics(fit["is_bad"], fit_scores, float(value))
        for value in candidates
    ]
    # 在拟合集上按 (F1, recall, precision, -threshold) 字典序取最大，即
    # F1 优先、并列时 recall 高者、再并列 precision 高者、最后取更小阈值。
    selected = max(
        fit_results,
        key=lambda item: (item["f1"], item["recall"], item["precision"], -item["threshold"]),
    )
    current_threshold = plan.get("current_threshold")
    if current_threshold is None:
        # 未显式给当前阈值时，从 Champion 包的 decision_threshold.json 读取。
        threshold_path = bundle["root"] / "decision_threshold.json"
        if not threshold_path.is_file():
            raise FileNotFoundError("CURRENT_THRESHOLD_ARTIFACT_MISSING")
        current_threshold = json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"]
    before = _threshold_metrics(validation["is_bad"], validation_scores, float(current_threshold))
    after = _threshold_metrics(validation["is_bad"], validation_scores, selected["threshold"])
    # 阈值只改变正负类划分，不改变模型排序，故排序指标应保持不变（用于护栏校验）。
    raw_ranking = _ranking(validation["is_bad"], validation_scores)
    artifact = {
        "threshold_id": f"{plan['model_id']}_{plan_id}",
        "model_id": plan["model_id"],
        "model_version": plan["champion_version"],
        "score_field": "calibrated_pd",
        "comparison": ">=",
        "threshold": selected["threshold"],
        "search_metric": metric,
        "selected_on_fit": selected,
        "validated_before": before,
        "validated_after": after,
        "authorization_id": plan["authorization_id"],
    }
    payload = json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    artifact_uri = str(plan.get("artifact_output_path") or "")
    if not artifact_uri:
        raise ValueError("THRESHOLD_ARTIFACT_OUTPUT_REQUIRED")
    _write_bytes(artifact_uri, payload, "application/json")
    receipt = {
        "fit_snapshot_id": plan.get("fit_snapshot_id"),
        "fit_snapshot_uri": plan.get("fit_snapshot_uri"),
        "fit_snapshot_checksum": fit_checksum,
        "validation_snapshot_id": plan.get("validation_snapshot_id"),
        "validation_snapshot_uri": plan.get("validation_snapshot_uri"),
        "validation_snapshot_checksum": validation_checksum,
        "fit_sample_count": int(len(fit)),
        "fit_bad_count": int(fit["is_bad"].sum()),
        "validation_sample_count": int(len(validation)),
        "validation_bad_count": int(validation["is_bad"].sum()),
        "sample_overlap_count": 0,
        "champion_model_checksum": bundle["model_checksum"],
        "champion_calibrator_checksum": bundle["calibrator_checksum"],
        "w4_read_count": 0,
    }
    return {
        "status": "SUCCEEDED",
        "plan_id": plan_id,
        "artifact_uri": artifact_uri,
        "artifact_checksum": _sha256(payload),
        "metrics": {
            "search_metric": metric,
            "selected_on_fit": selected,
            "before": before,
            "after": after,
            "f1_improvement": after["f1"] - before["f1"],
            "ranking_unchanged": raw_ranking,
        },
        "consumption_receipt": receipt,
    }


def _frame_payload(frame: pd.DataFrame) -> bytes:
    """把 DataFrame 序列化为 Parquet 字节（不含索引）。"""
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _recovery_rate(healthy: float, degraded: float, repaired: float) -> float:
    """计算单一数值指标的恢复率。

    恢复率 = (repaired - degraded) / (healthy - degraded)，即「修复挽回的退化量
    占原始退化量的比例」。
    - 若原始退化量 drop <= 0（本来就没退化），则修复后只要不更差（repaired >=
      degraded - 1e-12）就算 1.0（无退化可恢复，视为满恢复），否则 0.0。
    """
    drop = healthy - degraded
    if drop <= 0:
        return 1.0 if repaired >= degraded - 1e-12 else 0.0
    return float((repaired - degraded) / drop)


def execute_repair_and_replay(plan: dict) -> dict:
    """执行 A3/A4 修复，并在同一批带标签样本上重放 Champion 模型。

    两个动作的差异：
    - DATA_REPAIR（A3）：用 W2 参考统计构造「派生视图」修复缺失/异常值，
      从不动也不删原始行；
    - PIPELINE_REPAIR（A4）：用单独校验和的「可信快照」按 sample_id 回填受影响字段。

    两者都在完全相同的带标签样本上重放冻结的 Champion 模型，产出修复前后指标，
    用于后续恢复率判定。

    关键校验（全部 fail closed）：修复后行数不变、样本 ID 与顺序不变、标签不变。
    """

    plan_id = str(plan.get("repair_plan_id") or "")
    if not plan_id:
        raise ValueError("REPAIR_PLAN_ID_REQUIRED")
    action = str(plan.get("action") or "").upper()
    if action not in {"DATA_REPAIR", "PIPELINE_REPAIR"}:
        raise ValueError(f"REPAIR_ACTION_UNSUPPORTED:{action}")
    bundle = _bundle(plan)
    source, source_checksum = _read_frame(
        str(plan.get("source_snapshot_uri") or ""), plan.get("source_snapshot_checksum")
    )
    reference, reference_checksum = _read_frame(
        str(plan.get("reference_snapshot_uri") or ""),
        plan.get("reference_snapshot_checksum"),
    )
    healthy, healthy_checksum = _read_frame(
        str(plan.get("healthy_snapshot_uri") or ""), plan.get("healthy_snapshot_checksum")
    )
    affected = [str(name) for name in plan.get("affected_features") or []]
    if not affected:
        raise ValueError("REPAIR_AFFECTED_FEATURES_REQUIRED")
    repaired = source.copy()
    repair_audit: list[dict] = []

    if action == "DATA_REPAIR":
        # 从 Champion schema 里取字段的 kind，决定用中位数还是众数填充。
        schema_fields = {
            str(item.get("name")): item
            for item in bundle["schema"].get("fields") or []
            if item.get("name")
        }
        for feature in affected:
            if feature not in repaired.columns or feature not in reference.columns:
                raise ValueError(f"DATA_REPAIR_FEATURE_MISSING:{feature}")
            missing_before = int(repaired[feature].isna().sum())
            field = schema_fields.get(feature) or {}
            kind = str(field.get("kind") or "numeric").lower()
            if kind in {"numeric", "continuous", "integer", "float"}:
                # 数值型用参考集（W2）中位数填充——对异常值/偏态更稳健。
                fill_value = pd.to_numeric(reference[feature], errors="coerce").median()
                if pd.isna(fill_value):
                    raise ValueError(f"REFERENCE_MEDIAN_UNAVAILABLE:{feature}")
            else:
                # 类别型用众数，无众数时回退为 "UNKNOWN"。
                modes = reference[feature].dropna().mode()
                fill_value = modes.iloc[0] if len(modes) else "UNKNOWN"
            # 只填充缺失值，绝不改动非缺失行（不改动也不删除原始行）。
            repaired.loc[repaired[feature].isna(), feature] = fill_value
            repair_audit.append(
                {
                    "feature": feature,
                    "method": "REFERENCE_MEDIAN" if kind in {"numeric", "continuous", "integer", "float"} else "REFERENCE_MODE_OR_UNKNOWN",
                    "missing_before": missing_before,
                    "missing_after": int(repaired[feature].isna().sum()),
                    "reference_snapshot_id": plan.get("reference_snapshot_id"),
                }
            )
    else:
        # PIPELINE_REPAIR：必须样本集合一致、标签一致，才能按 sample_id 回填。
        if set(source["sample_id"].astype(str)) != set(reference["sample_id"].astype(str)):
            raise ValueError("PIPELINE_REFERENCE_SAMPLE_SET_MISMATCH")
        source_labels = source.set_index(source["sample_id"].astype(str))["is_bad"].sort_index()
        reference_labels = reference.set_index(reference["sample_id"].astype(str))["is_bad"].sort_index()
        if not source_labels.equals(reference_labels):
            raise ValueError("PIPELINE_REFERENCE_LABEL_MISMATCH")
        trusted = reference.copy()
        trusted.index = trusted["sample_id"].astype(str)
        source_ids = repaired["sample_id"].astype(str)
        for feature in affected:
            if feature not in trusted.columns:
                raise ValueError(f"PIPELINE_REFERENCE_FEATURE_MISSING:{feature}")
            before_missing = int(repaired[feature].isna().sum()) if feature in repaired else len(repaired)
            # 用可信快照按 sample_id 对齐回填受污染字段。
            repaired[feature] = source_ids.map(trusted[feature])
            # 回填后若仍有缺失（而可信快照没有缺失）说明 join 失败，拒绝。
            if repaired[feature].isna().any() and not trusted[feature].isna().any():
                raise ValueError(f"PIPELINE_RESTORE_JOIN_FAILED:{feature}")
            repair_audit.append(
                {
                    "feature": feature,
                    "method": "TRUSTED_SNAPSHOT_RESTORE_BY_SAMPLE_ID",
                    "missing_before": before_missing,
                    "missing_after": int(repaired[feature].isna().sum()),
                    "reference_snapshot_id": plan.get("reference_snapshot_id"),
                }
            )

    # 数据身份三重校验：修复绝不能改变行数、样本 ID/顺序、标签。
    if len(repaired) != len(source):
        raise ValueError("REPAIR_ROW_COUNT_CHANGED")
    if not repaired["sample_id"].astype(str).equals(source["sample_id"].astype(str)):
        raise ValueError("REPAIR_SAMPLE_ORDER_OR_ID_CHANGED")
    if not repaired["is_bad"].equals(source["is_bad"]):
        raise ValueError("REPAIR_LABEL_CHANGED")

    # 在同一批带标签样本上重放 Champion：修复前 vs 修复后 vs 健康基线。
    source_scores = _apply_calibrator(bundle["calibrator"], _raw_scores(bundle, source))
    repaired_scores = _apply_calibrator(bundle["calibrator"], _raw_scores(bundle, repaired))
    healthy_scores = _apply_calibrator(bundle["calibrator"], _raw_scores(bundle, healthy))
    degraded_metrics = _ranking(source["is_bad"], source_scores)
    repaired_metrics = _ranking(repaired["is_bad"], repaired_scores)
    healthy_metrics = _ranking(healthy["is_bad"], healthy_scores)
    missing_before = float(source.reindex(columns=affected).isna().to_numpy().mean())
    missing_after = float(repaired.reindex(columns=affected).isna().to_numpy().mean())
    metrics = {
        "action": action,
        "healthy_w1": healthy_metrics,
        "degraded": degraded_metrics,
        "repaired": repaired_metrics,
        "auc_recovery_rate": _recovery_rate(
            healthy_metrics["auc"], degraded_metrics["auc"], repaired_metrics["auc"]
        ),
        "ks_recovery_rate": _recovery_rate(
            healthy_metrics["ks"], degraded_metrics["ks"], repaired_metrics["ks"]
        ),
        "missing_rate_before": missing_before,
        "missing_rate_after": missing_after,
        "score_psi_after": _score_psi(healthy_scores, repaired_scores),
        "row_count_unchanged": True,
        "sample_ids_unchanged": True,
        "labels_unchanged": True,
        "repair_audit": repair_audit,
    }
    payload = _frame_payload(repaired)
    artifact_uri = str(plan.get("artifact_output_path") or "")
    if not artifact_uri:
        raise ValueError("REPAIR_ARTIFACT_OUTPUT_REQUIRED")
    _write_bytes(artifact_uri, payload, "application/octet-stream")
    receipt = {
        "source_snapshot_id": plan.get("source_snapshot_id"),
        "source_snapshot_uri": plan.get("source_snapshot_uri"),
        "source_snapshot_checksum": source_checksum,
        "reference_snapshot_id": plan.get("reference_snapshot_id"),
        "reference_snapshot_uri": plan.get("reference_snapshot_uri"),
        "reference_snapshot_checksum": reference_checksum,
        "healthy_snapshot_id": plan.get("healthy_snapshot_id"),
        "healthy_snapshot_uri": plan.get("healthy_snapshot_uri"),
        "healthy_snapshot_checksum": healthy_checksum,
        "source_sample_count": int(len(source)),
        "output_sample_count": int(len(repaired)),
        "champion_model_checksum": bundle["model_checksum"],
        "champion_calibrator_checksum": bundle["calibrator_checksum"],
        "w4_read_count": 0,
    }
    return {
        "status": "SUCCEEDED",
        "plan_id": plan_id,
        "artifact_uri": artifact_uri,
        "artifact_checksum": _sha256(payload),
        "metrics": metrics,
        "consumption_receipt": receipt,
    }


def qualify_repair(result: dict) -> tuple[bool, list[str]]:
    """A3/A4 修复后的资格判定：判断这次数据/管道修复是否达标。

    校验点：
    - 执行状态必须 SUCCEEDED；
    - 输出工件身份（URI + checksum）完整；
    - 消费回执有效且 w4_read_count == 0（pre-OOT 阶段未读盲测数据）；
    - 行数、样本 ID、标签均未变（数据身份不变）；
    - 修复后无缺失值残留；
    - 重放指标齐备，且 AUC/KS 恢复率达标（核心门槛 0.90）或未退化场景下不显著变差；
    - 修复后分数 PSI 不超 0.10（分布不能漂得太远）。

    返回：(是否合格, 失败原因列表)。
    """
    reasons: list[str] = []
    if result.get("status") != "SUCCEEDED":
        reasons.append("WORKER_STATUS_NOT_SUCCEEDED")
    if not result.get("artifact_uri") or not result.get("artifact_checksum"):
        reasons.append("REPAIR_ARTIFACT_IDENTITY_MISSING")
    receipt = result.get("consumption_receipt") or {}
    if not receipt or receipt.get("w4_read_count") != 0:
        reasons.append("REPAIR_CONSUMPTION_RECEIPT_INVALID")
    if receipt.get("source_sample_count") != receipt.get("output_sample_count"):
        reasons.append("REPAIR_ROW_COUNT_CHANGED")
    metrics = result.get("metrics") or {}
    if not all(
        metrics.get(name) is True
        for name in ("row_count_unchanged", "sample_ids_unchanged", "labels_unchanged")
    ):
        reasons.append("REPAIR_DATA_IDENTITY_FAILED")
    if metrics.get("missing_rate_after", 1.0) > 0:
        reasons.append("REPAIR_MISSING_VALUES_REMAIN")
    degraded = metrics.get("degraded") or {}
    repaired = metrics.get("repaired") or {}
    healthy = metrics.get("healthy_w1") or {}
    if not degraded or not repaired or not healthy:
        reasons.append("REPAIR_REPLAY_METRICS_MISSING")
    else:
        auc_drop = healthy["auc"] - degraded["auc"]
        ks_drop = healthy["ks"] - degraded["ks"]
        # 恢复率门槛（0.90）：退化显著（drop>0.02）时必须恢复至少 90%；
        # 退化不显著时则要求修复后不能比退化态更差（非退化护栏）。
        if auc_drop > 0.02 and metrics.get("auc_recovery_rate", 0) < 0.90:
            reasons.append("AUC_RECOVERY_BELOW_90_PERCENT")
        elif auc_drop <= 0.02 and repaired["auc"] < degraded["auc"] - 0.01:
            reasons.append("AUC_NON_DEGRADATION_FAILED")
        if ks_drop > 0.02 and metrics.get("ks_recovery_rate", 0) < 0.90:
            reasons.append("KS_RECOVERY_BELOW_90_PERCENT")
        elif ks_drop <= 0.02 and repaired["ks"] < degraded["ks"] - 0.02:
            reasons.append("KS_NON_DEGRADATION_FAILED")
    if metrics.get("score_psi_after", float("inf")) > 0.10:
        reasons.append("REPAIR_SCORE_PSI_FAILED")
    return not reasons, reasons


def qualify_adjustment(action: str, result: dict) -> tuple[bool, list[str]]:
    """A5/A6 调整后的资格判定：判断校准/阈值调整是否达标。

    公共校验（两者都做）：
    - 执行状态 SUCCEEDED、工件身份完整、消费回执有效、w4_read_count==0、
      fit 与 validation 无样本重叠。

    CALIBRATION_ADJUSTMENT（A5）专属：
    - Brier 与 ECE 都必须改善；
    - 改善后 Brier/ECE 不得超出健康区间上界；
    - AUC/KS 排序护栏：调整后不能显著变差（AUC 掉 >0.01、KS 掉 >0.02）；
    - 分数 PSI 硬上限 0.10，且不得比旧 PSI 放大；
    - 分段校准护栏必须通过；
    - 健康基线必须存在。

    THRESHOLD_ADJUSTMENT（A6）专属：
    - 业务目标指标（F1）必须改善。

    返回：(是否合格, 失败原因列表)。
    """
    reasons: list[str] = []
    if result.get("status") != "SUCCEEDED":
        reasons.append("WORKER_STATUS_NOT_SUCCEEDED")
    if not result.get("artifact_uri") or not result.get("artifact_checksum"):
        reasons.append("ARTIFACT_IDENTITY_MISSING")
    receipt = result.get("consumption_receipt") or {}
    if not receipt or receipt.get("w4_read_count") != 0:
        reasons.append("CONSUMPTION_RECEIPT_INVALID")
    if receipt.get("sample_overlap_count") != 0:
        reasons.append("FIT_VALIDATION_SAMPLE_OVERLAP")
    metrics = result.get("metrics") or {}
    if action == "CALIBRATION_ADJUSTMENT":
        before = metrics.get("before") or {}
        after = metrics.get("after") or {}
        healthy = metrics.get("healthy_w1") or {}
        if metrics.get("brier_improvement", -1) <= 0:
            reasons.append("BRIER_NOT_IMPROVED")
        if metrics.get("ece_improvement", -1) <= 0:
            reasons.append("ECE_NOT_IMPROVED")
        if after.get("brier", float("inf")) > metrics.get("healthy_brier_upper_bound", -1):
            reasons.append("BRIER_HEALTHY_RANGE_FAILED")
        if after.get("ece", float("inf")) > metrics.get("healthy_ece_upper_bound", -1):
            reasons.append("ECE_HEALTHY_RANGE_FAILED")
        # 排序护栏：校准只能改善概率，绝不能牺牲判别（排序）能力。
        if after.get("auc", -1) < before.get("auc", 0) - 0.01:
            reasons.append("AUC_RANKING_GUARDRAIL_FAILED")
        if after.get("ks", -1) < before.get("ks", 0) - 0.02:
            reasons.append("KS_RANKING_GUARDRAIL_FAILED")
        if metrics.get("new_score_psi", float("inf")) > 0.10:
            reasons.append("SCORE_PSI_HARD_LIMIT_FAILED")
        # 校准不应让分数分布漂移得更厉害。
        if metrics.get("new_score_psi", float("inf")) > metrics.get("old_score_psi", -1) + 1e-12:
            reasons.append("SCORE_PSI_AMPLIFIED")
        if not metrics.get("segment_guardrail_passed"):
            reasons.append("SEGMENT_CALIBRATION_GUARDRAIL_FAILED")
        if not healthy:
            reasons.append("HEALTHY_BASELINE_MISSING")
    elif action == "THRESHOLD_ADJUSTMENT":
        if metrics.get("f1_improvement", 0) <= 0:
            reasons.append("BUSINESS_OBJECTIVE_NOT_IMPROVED")
    else:
        reasons.append(f"ADJUSTMENT_ACTION_UNSUPPORTED:{action}")
    return not reasons, reasons
