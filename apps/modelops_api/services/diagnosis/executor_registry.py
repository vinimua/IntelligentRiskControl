"""验证器执行注册表 — 按 method_code 分发到具体实现。

D = Distribution/Data      — 分布、数据质量和直接异常事实
R = Repair/Recovery        — 反事实修复后性能是否恢复
C = Conditional/Causal     — 控制变量后根因与性能的关联
T = Temporal               — 原因是否先于症状出现
I = Importance/Dependency  — 模型是否依赖该特征或机制
"""

from __future__ import annotations

from collections.abc import Callable

from packages.models.diagnosis.evidence import EvidenceItem

# 执行器签名：接受 candidate + drift_data + alert_context → EvidenceItem
ValidatorFunc = Callable[..., EvidenceItem]

EXECUTOR_REGISTRY: dict[str, ValidatorFunc] = {}


def register(method_code: str):
    """装饰器：将验证器注册到全局注册表。"""
    def decorator(fn: ValidatorFunc) -> ValidatorFunc:
        EXECUTOR_REGISTRY[method_code] = fn
        return fn
    return decorator


# ── 延迟导入以避免循环依赖 ──
def _lazy_register_all():
    """在首次使用前注册所有验证器（避免模块加载顺序问题）。"""
    if EXECUTOR_REGISTRY:
        return  # 已注册

    from .validators.psi_check import psi_check
    from .validators.data_quality_check import data_quality_check
    from .validators.counterfactual_repair_check import counterfactual_repair_check
    from .validators.drift_group_regression import drift_group_regression
    from .validators.temporal_precedence_check import temporal_precedence_check
    from .validators.permutation_importance_check import permutation_importance_check

    # ── D 类型：数据/分布 ──
    EXECUTOR_REGISTRY["psi_check"] = psi_check
    EXECUTOR_REGISTRY["missing_outlier_range_check"] = data_quality_check

    # ── R 类型：反事实修复 ──
    EXECUTOR_REGISTRY["counterfactual_repair_check"] = counterfactual_repair_check

    # ── C 类型：关联/回归 ──
    EXECUTOR_REGISTRY["drift_group_regression"] = drift_group_regression

    # ── T 类型：时序优先 ──
    EXECUTOR_REGISTRY["temporal_precedence_check"] = temporal_precedence_check

    # ── I 类型：重要性依赖 ──
    EXECUTOR_REGISTRY["permutation_importance_check"] = permutation_importance_check
