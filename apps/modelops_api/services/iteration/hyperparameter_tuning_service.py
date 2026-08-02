"""T3-GAP-02: 超参优化服务。

根据算法类型生成 N 组候选超参（random search）。
"""

from __future__ import annotations

import random

from packages.models.iteration.hyperparameter_tuning import (
    HyperparameterTuningPlan,
    TuningStrategy,
    TuningTrial,
)

# 预定义搜索空间
_SEARCH_SPACES: dict[str, dict] = {
    "lightgbm": {
        "n_estimators": [50, 100, 150, 200, 300],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.10, 0.15],
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [10, 20, 50, 100],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    },
    "logistic_regression": {
        "C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        "max_iter": [500, 1000, 2000],
        "solver": ["liblinear", "lbfgs", "saga"],
        "penalty": ["l1", "l2"],
    },
    "random_forest": {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [4, 6, 8, 10, 12],
        "min_samples_leaf": [5, 10, 20, 50],
        "max_features": ["sqrt", "log2", None],
    },
}

# 默认最佳值（当无需 tuning 或作为基准时使用）
_DEFAULT_PARAMS: dict[str, dict] = {
    "lightgbm": {
        "n_estimators": 100, "max_depth": 6, "learning_rate": 0.05,
        "num_leaves": 31, "min_child_samples": 20,
        "subsample": 0.8, "colsample_bytree": 0.8,
    },
    "logistic_regression": {
        "C": 1.0, "max_iter": 1000, "solver": "liblinear", "penalty": "l2",
    },
    "random_forest": {
        "n_estimators": 120, "max_depth": 8, "min_samples_leaf": 20,
        "max_features": "sqrt",
    },
}


class HyperparameterTuningService:
    """生成候选超参组合。"""

    def build_plan(
        self,
        *,
        model_id: str = "",
        lifecycle_run_id: str | None = None,
        training_plan_id: str | None = None,
        algorithm: str = "lightgbm",
        num_trials: int = 5,
        seed: int = 2026,
        base_params: dict | None = None,
    ) -> HyperparameterTuningPlan:
        """生成超参搜索计划。"""
        algorithm = algorithm.lower()
        if algorithm not in _SEARCH_SPACES:
            algorithm = "lightgbm"

        space = _SEARCH_SPACES[algorithm]
        rng = random.Random(seed)
        base = base_params or {}

        trials: list[TuningTrial] = []
        for i in range(num_trials):
            params: dict = {}
            for param_name, candidates in space.items():
                if param_name in base:
                    params[param_name] = base[param_name]  # 保留 base 值
                elif isinstance(candidates, list) and candidates:
                    params[param_name] = rng.choice(candidates)

            trials.append(TuningTrial(
                trial_index=i,
                hyperparameters=params,
            ))

        # 第一组使用默认最佳值作为基准
        if trials and algorithm in _DEFAULT_PARAMS:
            default = dict(_DEFAULT_PARAMS[algorithm])
            default.update(base)
            trials[0] = TuningTrial(
                trial_index=0,
                hyperparameters=default,
            )

        return HyperparameterTuningPlan(
            lifecycle_run_id=lifecycle_run_id,
            training_plan_id=training_plan_id,
            model_id=model_id,
            algorithm=algorithm,
            strategy=TuningStrategy.RANDOM.value,
            num_trials=len(trials),
            search_space={k: list(v) for k, v in space.items()},
            trials=trials,
        )
