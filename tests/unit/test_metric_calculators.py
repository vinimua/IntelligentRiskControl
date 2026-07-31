"""指标计算器单元测试 — 7 个指标用固定数据集验证计算正确性。"""

from __future__ import annotations

import math

import pytest

# 导入 metric_calculators 触发 @register 装饰器，填充 METRIC_CALCULATORS
import apps.modelops_api.services.monitoring.metric_calculators  # noqa: F401
from apps.modelops_api.services.monitoring.metrics_registry import METRIC_CALCULATORS
from packages.models.common.enums import AvailabilityStatus


def _make_data(n: int, seed: int = 42, drift: bool = False) -> list[dict]:
    """生成固定测试数据集。"""
    import random as _random
    rng = _random.Random(seed)
    data = []
    for _ in range(n):
        f1 = rng.gauss(0, 1) + (0.5 if drift else 0)
        f2 = rng.gauss(0, 1) + (0.2 if drift else 0)
        z = 0.7 * f1 + 0.4 * f2 + rng.gauss(0, 0.1)
        proba = 1.0 / (1.0 + math.exp(-z))
        y_true = 1 if rng.random() < proba else 0
        data.append({
            "y_true": y_true,
            "y_pred_proba": round(proba, 6),
            "score": int(300 + 200 * proba + rng.gauss(0, 5)),
            "feature_f1": round(f1, 4),
            "feature_f2": round(f2, 4),
        })
    return data


class TestAUC:
    def test_auc_calculated(self):
        baseline = _make_data(500, seed=1)
        current = _make_data(500, seed=2)
        calc = METRIC_CALCULATORS["AUC"]
        result = calc(baseline, current)
        assert result.metric_code == "AUC"
        assert result.current_value is not None
        assert 0.0 <= result.current_value <= 1.0

    def test_auc_too_small(self):
        calc = METRIC_CALCULATORS["AUC"]
        result = calc([], [{"y_true": 1, "y_pred_proba": 0.5}])
        assert result.availability_status == AvailabilityStatus.SAMPLE_TOO_SMALL


class TestKS:
    def test_ks_calculated(self):
        data = _make_data(500, seed=3)
        calc = METRIC_CALCULATORS["KS"]
        result = calc(data, data)
        assert result.metric_code == "KS"
        assert result.current_value is not None
        assert 0.0 <= result.current_value <= 1.0

    def test_ks_single_class(self):
        data = [{"y_true": 0, "y_pred_proba": 0.5}] * 50
        calc = METRIC_CALCULATORS["KS"]
        result = calc(data, data)
        assert result.availability_status == AvailabilityStatus.CALCULATION_FAILED


class TestFeaturePSI:
    def test_psi_no_drift(self):
        baseline = _make_data(500, seed=10, drift=False)
        current = _make_data(500, seed=10, drift=False)
        calc = METRIC_CALCULATORS["FEATURE_PSI"]
        result = calc(baseline, current)
        assert result.metric_code == "FEATURE_PSI"
        assert result.current_value is not None
        # 同分布 → PSI 接近 0
        assert result.current_value < 0.1

    def test_psi_with_drift(self):
        baseline = _make_data(500, seed=10, drift=False)
        current = _make_data(500, seed=10, drift=True)
        calc = METRIC_CALCULATORS["FEATURE_PSI"]
        result = calc(baseline, current)
        assert result.metric_code == "FEATURE_PSI"
        assert result.current_value is not None
        # 有漂移 → PSI > 0
        assert result.current_value > 0.0


class TestScorePSI:
    def test_score_psi(self):
        baseline = _make_data(500, seed=20, drift=False)
        current = _make_data(500, seed=20, drift=False)
        calc = METRIC_CALCULATORS["SCORE_PSI"]
        result = calc(baseline, current)
        assert result.metric_code == "SCORE_PSI"
        assert result.current_value is not None
        assert result.current_value < 0.1  # same distribution


class TestMissingRate:
    def test_missing_rate_zero(self):
        data = [{"a": 1.0, "b": 2.0}] * 100
        calc = METRIC_CALCULATORS["MISSING_RATE"]
        result = calc(data, data)
        assert result.metric_code == "MISSING_RATE"
        assert result.current_value == 0.0

    def test_missing_rate_change(self):
        baseline = [{"a": 1.0, "b": 2.0} for _ in range(100)]
        current = [{"a": 1.0, "b": 2.0} for _ in range(80)] + [{"a": None, "b": 2.0} for _ in range(20)]
        calc = METRIC_CALCULATORS["MISSING_RATE"]
        result = calc(baseline, current)
        assert result.metric_code == "MISSING_RATE"
        assert result.current_value > 0.0


class TestSchemaConsistency:
    def test_schema_match(self):
        data = [{"a": 1, "b": 2.0}]
        calc = METRIC_CALCULATORS["SCHEMA_CONSISTENCY"]
        result = calc(data, data)
        assert result.metric_code == "SCHEMA_CONSISTENCY"
        assert result.current_value == 0

    def test_schema_column_added(self):
        baseline = [{"a": 1}]
        current = [{"a": 1, "b": 2}]
        calc = METRIC_CALCULATORS["SCHEMA_CONSISTENCY"]
        result = calc(baseline, current)
        assert result.current_value is not None
        assert result.current_value >= 1  # 1 added column


class TestSampleSize:
    def test_sample_size(self):
        data = [{"x": 1}] * 500
        calc = METRIC_CALCULATORS["SAMPLE_SIZE"]
        result = calc([], data)
        assert result.metric_code == "SAMPLE_SIZE"
        assert result.current_value == 500
        assert result.baseline_value == 0


class TestRegistry:
    def test_all_metrics_registered(self):
        expected = {
            "AUC", "KS", "FEATURE_PSI", "SCORE_PSI", "MISSING_RATE",
            "SCHEMA_CONSISTENCY", "SAMPLE_SIZE",
            # V2 新增
            "PREDICTION_MEAN", "MAX_FEATURE_PSI_7D", "MAX_FEATURE_PSI_30D",
            "BAD_RATE", "OUTLIER_RATE", "DATA_QUALITY_SCORE",
            # 交接包完整性能指标
            "PR_AUC", "BRIER", "ECE", "BAD_RECALL",
        }
        assert set(METRIC_CALCULATORS.keys()) == expected
