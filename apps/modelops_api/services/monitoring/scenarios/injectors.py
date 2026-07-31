"""受控异常场景注入引擎 — 基于交接包 scenarios/injectors.py。

Sentinel 训练的正样本来源。在 W1/W2 数据的隔离副本上注入
已知类型的异常，生成带 anomaly_label 的训练数据。

12 个场景族覆盖业务漂移和运维异常两大类。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScenarioResult:
    """场景注入结果。"""

    dataframe: pd.DataFrame   # 注入后的数据副本（含 data_track/injected_flag/anomaly_label）
    metadata: dict[str, Any]  # 注入元数据（seed/checksum/受影响样本数等）


def _choose(indices: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """从索引中随机选择 fraction 比例的样本。"""
    if len(indices) == 0:
        return np.array([], dtype=indices.dtype)
    count = max(1, min(len(indices), int(round(len(indices) * fraction))))
    return rng.choice(indices, size=count, replace=False)


class ScenarioFactory:
    """按场景名分发的注入工厂。

    每个场景在数据的隔离副本上操作，不修改原始数据。
    所有注入逻辑是确定性的（给定 random_seed 可复现）。
    """

    @staticmethod
    def inject(
        dataframe: pd.DataFrame,
        scenario_config: dict[str, Any],
        random_seed: int,
    ) -> ScenarioResult:
        """在数据隔离副本上注入受控异常。

        Args:
            dataframe: 要注入的基础数据（W1/W2 的副本）。
            scenario_config: 场景配置，必须包含 scenario_name/intensity/affected_features/event_start_date/event_end_date。
            random_seed: 随机种子（保证确定性复现）。

        Returns:
            ScenarioResult（含修改后的 DataFrame 和元数据）。

        Raises:
            ValueError: 场景名为空、未知场景、或 W4 锁定数据。
        """
        # W4 保护
        if dataframe.attrs.get("window_id") == "W4" or scenario_config.get("base_window_id") == "W4":
            raise ValueError("Scenarios cannot use locked W4")

        frame = dataframe.copy(deep=True)
        rng = np.random.default_rng(random_seed)
        name = str(scenario_config["scenario_name"])
        intensity = float(scenario_config.get("intensity", 0.3))
        affected = list(scenario_config.get("affected_features", []))

        apply_time = pd.to_datetime(frame["apply_time"])
        event_start = pd.Timestamp(scenario_config.get("event_start_date", apply_time.min())).normalize()
        event_end = pd.Timestamp(scenario_config.get("event_end_date", apply_time.max() + pd.Timedelta(days=1))).normalize()

        eligible = frame.index[(apply_time >= event_start) & (apply_time < event_end)].to_numpy()
        if len(eligible) == 0:
            raise ValueError(f"Scenario event range contains no rows: [{event_start}, {event_end})")

        changed = np.array([], dtype=eligible.dtype)

        # ═══ 12 个场景实现 ═══

        if name == "missing_rate_anomaly":
            changed = _choose(eligible, intensity, rng)
            frame.loc[changed, affected] = np.nan

        elif name == "numeric_scaling_anomaly":
            changed = _choose(eligible, min(1.0, intensity * 1.5), rng)
            frame.loc[changed, affected] = frame.loc[changed, affected] * (1.0 + 4.0 * intensity)

        elif name == "covariate_drift":
            changed = _choose(eligible, min(1.0, intensity * 1.5), rng)
            for feature in affected:
                scale = max(float(pd.to_numeric(frame[feature], errors="coerce").std()), 1e-6)
                frame.loc[changed, feature] = frame.loc[changed, feature] + intensity * 2.5 * scale

        elif name == "feature_staleness":
            rows = frame.loc[eligible].sort_values("apply_time").index
            changed = rows.to_numpy()
            for feature in affected:
                lag = max(1, int(len(rows) * intensity))
                frame.loc[rows, feature] = frame.loc[rows, feature].shift(lag).bfill().to_numpy()

        elif name == "customer_mix_shift":
            event = frame.loc[eligible]
            high_risk = event[(event["city_tier"] >= 3) | (event["age"] <= 25)]
            changed = _choose(eligible, intensity, rng)
            if high_risk.empty:
                raise ValueError("customer_mix_shift requires at least one high-risk donor row")
            donors = high_risk.sample(n=len(changed), replace=True, random_state=random_seed).reset_index(drop=True)
            replace_columns = [c for c in frame.columns if c not in {"sample_id", "apply_time"}]
            frame.loc[changed, replace_columns] = donors[replace_columns].to_numpy()

        elif name == "concept_drift":
            changed = _choose(eligible, intensity, rng)
            # 保持坏样本率不变，但打破 X→Y 的原有映射关系
            labels = frame.loc[changed, "is_bad"].astype(int).to_numpy(copy=True)
            frame.loc[changed, "is_bad"] = rng.permutation(labels)

        elif name == "bad_rate_shift":
            good = frame.index[frame.index.isin(eligible) & frame["is_bad"].astype(int).eq(0)].to_numpy()
            changed = _choose(good, intensity, rng)
            frame.loc[changed, "is_bad"] = 1

        elif name == "policy_selection_shift":
            event = frame.loc[eligible]
            debt_cut = pd.to_numeric(event["debt_income_ratio"], errors="coerce").quantile(0.70)
            donors_pool = event[
                (pd.to_numeric(event["debt_income_ratio"], errors="coerce") >= debt_cut)
                | (pd.to_numeric(event["city_tier"], errors="coerce") >= 3)
                | (pd.to_numeric(event["age"], errors="coerce") <= 25)
            ]
            if donors_pool.empty:
                raise ValueError("policy_selection_shift requires eligible policy-shift donor rows")
            changed = _choose(eligible, intensity, rng)
            donors = donors_pool.sample(n=len(changed), replace=True, random_state=random_seed).reset_index(drop=True)
            replace_columns = [c for c in frame.columns if c not in {"sample_id", "apply_time"}]
            frame.loc[changed, replace_columns] = donors[replace_columns].to_numpy()

        elif name == "fraud_pattern_shift":
            event = frame.loc[eligible]
            stealth = event[
                (pd.to_numeric(event["login_fail_count"], errors="coerce") <= 1)
                & (pd.to_numeric(event["max_overdue_days"], errors="coerce") <= 1)
                & frame.loc[eligible, "is_bad"].astype(int).eq(0)
            ]
            pool = (
                stealth.index.to_numpy()
                if not stealth.empty
                else event.index[event["is_bad"].astype(int).eq(0)].to_numpy()
            )
            changed = _choose(pool, intensity, rng)
            frame.loc[changed, "is_bad"] = 1

        elif name == "preprocessing_version_mismatch":
            changed = _choose(eligible, min(1.0, intensity * 1.5), rng)
            for feature in affected:
                frame[feature] = pd.to_numeric(frame[feature], errors="coerce").astype(float)
                numeric = pd.to_numeric(frame.loc[eligible, feature], errors="coerce")
                scale = max(float(numeric.std()), 1e-6)
                frame.loc[changed, feature] = (
                    pd.to_numeric(frame.loc[changed, feature], errors="coerce") - float(numeric.mean())
                ) / scale

        elif name == "key_feature_failure":
            changed = _choose(eligible, min(1.0, intensity * 2.0), rng)
            frame.loc[changed, affected] = 0.0

        elif name == "multi_root_cause":
            changed = _choose(eligible, intensity, rng)
            if affected:
                frame.loc[changed, affected[0]] = np.nan
            if len(affected) > 1:
                frame.loc[changed, affected[1]] = frame.loc[changed, affected[1]] * (1.0 + 5.0 * intensity)

        elif name == "clean_control":
            frame = frame.copy(deep=True)

        else:
            raise ValueError(f"Unknown scenario implementation: {name}")

        # 标记
        anomaly = 0 if name == "clean_control" else None
        frame["data_track"] = "SCENARIO"
        frame["injected_flag"] = anomaly
        frame["anomaly_label"] = anomaly

        # 元数据（含确定性 checksum）
        stable = {key: scenario_config.get(key) for key in sorted(scenario_config)} | {
            "seed": random_seed,
            "rows": len(frame),
        }
        checksum = hashlib.sha256(
            json.dumps(stable, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        metadata = dict(scenario_config)
        metadata.update({
            "random_seed": random_seed,
            "checksum": checksum,
            "anomaly_label": anomaly,
            "data_track": "SCENARIO",
            "event_start_date": event_start.isoformat(),
            "event_end_date": event_end.isoformat(),
            "affected_sample_count": int(len(changed)),
            "affected_sample_ratio": float(len(changed) / len(eligible)),
        })

        return ScenarioResult(frame, metadata)
