"""T4-GAP-01: 模型比对服务 — champion vs challenger 完整指标矩阵。

输入: y_true, champion_scores, challenger_scores
输出: ModelComparisonReport (12+ 指标)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    precision_score,
    roc_curve,
)


class ModelComparisonService:
    """计算 champion vs challenger 的完整指标对比。"""

    # 指标定义: (code, direction, threshold)
    METRIC_DEFS = [
        ("AUC", "higher_is_better", 0.005),
        ("KS", "higher_is_better", 0.02),
        ("PR_AUC", "higher_is_better", 0.005),
        ("F1", "higher_is_better", 0.01),
        ("Recall", "higher_is_better", 0.01),
        ("Precision", "higher_is_better", 0.01),
        ("Bad_Recall", "higher_is_better", 0.01),
        ("Brier_Score", "lower_is_better", 0.005),
        ("ECE", "lower_is_better", 0.02),
    ]

    def compare(
        self,
        y_true: np.ndarray,
        champion_scores: np.ndarray,
        challenger_scores: np.ndarray,
        *,
        model_id: str = "",
        champion_version: str = "",
        challenger_version: str = "",
        lifecycle_run_id: str | None = None,
        qualification_run_id: str | None = None,
        bad_rate_threshold: float = 0.5,
    ) -> "ModelComparisonReport":
        """计算完整对比报告。"""
        from packages.models.iteration.model_comparison import (
            MetricPair, ModelComparisonReport,
        )

        metrics: list[MetricPair] = []

        # ── Discrimination metrics ──
        try:
            champ_auc = roc_auc_score(y_true, champion_scores)
            chall_auc = roc_auc_score(y_true, challenger_scores)
            metrics.append(_make_pair("AUC", champ_auc, chall_auc, "higher_is_better"))
        except Exception:
            metrics.append(_make_pair("AUC", None, None, "higher_is_better"))

        try:
            champ_ks = _compute_ks(y_true, champion_scores)
            chall_ks = _compute_ks(y_true, challenger_scores)
            metrics.append(_make_pair("KS", champ_ks, chall_ks, "higher_is_better"))
        except Exception:
            metrics.append(_make_pair("KS", None, None, "higher_is_better"))

        try:
            champ_pr = average_precision_score(y_true, champion_scores)
            chall_pr = average_precision_score(y_true, challenger_scores)
            metrics.append(_make_pair("PR_AUC", champ_pr, chall_pr, "higher_is_better"))
        except Exception:
            metrics.append(_make_pair("PR_AUC", None, None, "higher_is_better"))

        # ── Threshold-dependent metrics (at bad_rate_threshold) ──
        try:
            champ_pred = (champion_scores >= bad_rate_threshold).astype(int)
            chall_pred = (challenger_scores >= bad_rate_threshold).astype(int)
        except Exception:
            champ_pred = np.zeros_like(y_true)
            chall_pred = np.zeros_like(y_true)

        for code, fn in [("F1", f1_score), ("Recall", recall_score), ("Precision", precision_score)]:
            try:
                cv = fn(y_true, champ_pred)
                clv = fn(y_true, chall_pred)
                metrics.append(_make_pair(code, cv, clv, "higher_is_better"))
            except Exception:
                metrics.append(_make_pair(code, None, None, "higher_is_better"))

        # Bad recall: recall on the bad class (y_true=1)
        try:
            champ_br = recall_score(y_true, champ_pred, pos_label=1)
            chall_br = recall_score(y_true, chall_pred, pos_label=1)
            metrics.append(_make_pair("Bad_Recall", champ_br, chall_br, "higher_is_better"))
        except Exception:
            metrics.append(_make_pair("Bad_Recall", None, None, "higher_is_better"))

        # ── Calibration metrics ──
        try:
            champ_brier = brier_score_loss(y_true, champion_scores)
            chall_brier = brier_score_loss(y_true, challenger_scores)
            metrics.append(_make_pair("Brier_Score", champ_brier, chall_brier, "lower_is_better"))
        except Exception:
            metrics.append(_make_pair("Brier_Score", None, None, "lower_is_better"))

        try:
            champ_ece = _expected_calibration_error(y_true, champion_scores, n_bins=10)
            chall_ece = _expected_calibration_error(y_true, challenger_scores, n_bins=10)
            metrics.append(_make_pair("ECE", champ_ece, chall_ece, "lower_is_better"))
        except Exception:
            metrics.append(_make_pair("ECE", None, None, "lower_is_better"))

        # ── PSI (champion vs challenger score distribution drift) ──
        try:
            psi_val = _calc_psi(champion_scores, challenger_scores, n_bins=10)
            metrics.append(MetricPair(
                metric_code="Score_PSI", champion_value=None, challenger_value=None,
                delta=psi_val, delta_pct=None, direction="lower_is_better",
                passed=psi_val < 0.25,
            ))
        except Exception:
            metrics.append(_make_pair("Score_PSI", None, None, "lower_is_better"))

        # ── Overall pass/fail ──
        expected_metric_count = 10
        all_passed = len(metrics) == expected_metric_count and all(
            m.passed is True for m in metrics
        )
        summary = (
            f"Champion {champion_version} vs Challenger {challenger_version}: "
            f"{'PASSED' if all_passed else 'FAILED'} "
            f"({sum(1 for m in metrics if m.passed is True)}/{len(metrics)} metrics passed)"
        )

        return ModelComparisonReport(
            model_id=model_id,
            champion_version=champion_version,
            challenger_version=challenger_version,
            lifecycle_run_id=lifecycle_run_id,
            qualification_run_id=qualification_run_id,
            metrics=metrics,
            passed=all_passed,
            summary=summary,
        )


def _make_pair(code: str, cv: float | None, clv: float | None, direction: str) -> "MetricPair":
    """构建 MetricPair 并判断是否通过。"""
    from packages.models.iteration.model_comparison import MetricPair

    delta = None
    delta_pct = None
    if cv is not None and clv is not None:
        delta = round(clv - cv, 6)
        if cv != 0:
            delta_pct = round(delta / abs(cv) * 100, 2)

    # 判断是否退化
    passed = False
    if delta is not None:
        if direction == "higher_is_better":
            passed = delta >= -0.005  # 允许 0.005 的微小退化
        elif direction == "lower_is_better":
            passed = delta <= 0.005

    return MetricPair(
        metric_code=code,
        champion_value=round(cv, 6) if cv is not None else None,
        challenger_value=round(clv, 6) if clv is not None else None,
        delta=delta, delta_pct=delta_pct,
        direction=direction, passed=passed,
    )


def _compute_ks(y_true, scores):
    """KS 统计量。"""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.max(tpr - fpr))


def _expected_calibration_error(y_true, scores, n_bins=10):
    """Expected Calibration Error。"""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (scores >= bins[i]) & (scores < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = scores[mask].mean()
        ece += abs(bin_acc - bin_conf) * mask.sum() / len(y_true)
    return float(ece)


def _calc_psi(expected_scores, actual_scores, n_bins=10):
    """Population Stability Index between two score distributions。"""
    bins = np.linspace(0, 1, n_bins + 1)
    expected_pct = np.histogram(expected_scores, bins=bins)[0] / len(expected_scores)
    actual_pct = np.histogram(actual_scores, bins=bins)[0] / len(actual_scores)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
