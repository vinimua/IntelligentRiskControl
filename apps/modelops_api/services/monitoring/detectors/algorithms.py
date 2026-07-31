"""四个流式漂移检测器 — 基于交接包 WP07 实现。

本模块从 risk_inquiry_agent/src/monitoring/detectors/core.py 移植。

四个检测器均为有状态（stateful）算法，按时间顺序逐个值更新：
- ADWIN: 自适应窗口均值变化检测
- PageHinkley: 累积偏差渐变检测（CUSUM 变体）
- KSWIN: 基于 KS 检验的分布变化检测
- RobustZ: 抗异常值的离群点检测（中位数 + MAD）

所有检测器只依赖 numpy + scipy，无需标签。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.stats import ks_2samp


# ═══════════════════════════════════════════════════════════════
# 通用结构
# ═══════════════════════════════════════════════════════════════


@dataclass
class DetectorResult:
    """单个检测器在单个时间点的输出。"""

    alarm_flag: bool
    detector_score: float  # 0~1，归一化异常分数
    warmup_status: str     # "WARMUP" | "READY" | "MISSING"
    value: float | None
    timestamp: datetime | None


class DetectorBase:
    """检测器基类。子类必须实现 update() 方法。"""

    name: str = "BASE"

    def __init__(self) -> None:
        self._result = DetectorResult(False, 0.0, "WARMUP", None, None)

    def reset(self) -> None:
        """重置检测器内部状态。"""
        self._result = DetectorResult(False, 0.0, "WARMUP", None, None)

    def update(self, value: float, timestamp: datetime) -> None:
        """输入一个新值，更新内部状态。"""
        raise RuntimeError("DetectorBase cannot update directly; use a concrete detector")

    def get_result(self) -> DetectorResult:
        """获取最近一次 update 后的检测结果。"""
        return self._result


# ═══════════════════════════════════════════════════════════════
# ADWIN — 自适应滑动窗口均值变化检测
# ═══════════════════════════════════════════════════════════════


class ADWINDetector(DetectorBase):
    """自适应窗口检测器。

    核心问题："最近的平均值，跟之前比，是不是不一样了？"

    将历史值队列从正中间切成两半（左半=以前，右半=最近），
    比较两半的均值差是否显著超过理论边界。

    擅长：检测突变（均值突然跳变）。
    盲区：缓慢渐变。

    Args:
        delta: 置信参数（默认 0.002，对应约 99.8% 置信度）。
        grace_period: 最少需要积累多少个值才启动检测。
        max_window: 历史队列最大长度。
    """

    name = "ADWIN"

    def __init__(
        self,
        delta: float = 0.002,
        grace_period: int = 5,
        max_window: int = 64,
    ) -> None:
        super().__init__()
        self.delta = delta
        self.grace_period = grace_period
        self.max_window = max_window
        self.values: deque[float] = deque(maxlen=max_window)

    def reset(self) -> None:
        super().reset()
        self.values.clear()

    def update(self, value: float, timestamp: datetime) -> None:
        self.values.append(float(value))
        min_len = max(self.grace_period, 4)
        if len(self.values) < min_len:
            self._result = DetectorResult(False, 0.0, "WARMUP", value, timestamp)
            return

        data = np.asarray(self.values)
        split = len(data) // 2
        left, right = data[:split], data[split:]

        scale = max(float(np.std(data)), 1e-9)
        se = math.sqrt(1.0 / len(left) + 1.0 / len(right))
        z = abs(float(right.mean() - left.mean())) / (scale * se)

        boundary = math.sqrt(2.0 * math.log(2.0 / self.delta))
        score = float(np.clip(z / boundary, 0.0, 1.0))
        self._result = DetectorResult(z > boundary, score, "READY", value, timestamp)


# ═══════════════════════════════════════════════════════════════
# Page-Hinkley — 累积偏差渐变检测
# ═══════════════════════════════════════════════════════════════


class PageHinkleyDetector(DetectorBase):
    """累积和（CUSUM）变体检测器。

    核心问题："这个信号是不是一直在往一个方向偏，而且偏了很久了？"

    每次 update 时累积 (value - running_mean - delta)，
    用指数衰减（alpha）保留旧累积值。当累积值偏离历史最低点
    超过阈值时触发告警。

    擅长：检测渐变趋势（微小的持续偏离累积到报警）。
    盲区：孤立异常点（偏离后迅速恢复，累积值被衰减拉回）。

    Args:
        min_instances: 最少观察数。
        delta: 可容忍的微小波动（低于此值不累积）。
        threshold: "怒气值"触发阈值。
        alpha: 指数衰减因子（0.999 ≈ 缓慢衰减，适合长趋势检测）。
    """

    name = "PAGE_HINKLEY"

    def __init__(
        self,
        min_instances: int = 5,
        delta: float = 0.005,
        threshold: float = 25.0,
        alpha: float = 0.999,
    ) -> None:
        super().__init__()
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.count = 0
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0

    def reset(self) -> None:
        super().reset()
        self.count = 0
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0

    def update(self, value: float, timestamp: datetime) -> None:
        self.count += 1
        # Welford 增量均值更新
        self.mean += (value - self.mean) / self.count
        # 累积偏差（带衰减）
        self.cumulative = self.alpha * self.cumulative + value - self.mean - self.delta
        self.minimum = min(self.minimum, self.cumulative)
        statistic = self.cumulative - self.minimum

        ready = self.count >= self.min_instances
        score = (
            float(np.clip(statistic / max(self.threshold, 1e-9), 0.0, 1.0))
            if ready
            else 0.0
        )
        self._result = DetectorResult(
            bool(ready and statistic > self.threshold),
            score,
            "READY" if ready else "WARMUP",
            value,
            timestamp,
        )


# ═══════════════════════════════════════════════════════════════
# KSWIN — 基于 KS 检验的分布变化检测
# ═══════════════════════════════════════════════════════════════


class KSWINDetector(DetectorBase):
    """KSWIN 窗口分布变化检测器。

    核心问题："最近的值所属的分布，跟之前比，还是同一个分布吗？"

    维护 20 个历史值。每次 update 时比较：
    - 参考窗口（前 12 个值）
    - 近期窗口（后 8 个值）
    做两样本 KS 检验，p 值 < alpha 时触发告警。

    擅长：检测分布形态变化（即使均值没变，分布形状变了也能发现）。
    盲区：样本量小时统计效力不足（需 20 个值才启动）。

    Args:
        alpha: KS 检验显著水平（默认 0.005，极严格）。
        window_size: 历史队列大小。
        stat_size: 近期窗口大小。
        seed: 保留（未使用，保持接口兼容）。
    """

    name = "KSWIN"

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 20,
        stat_size: int = 8,
        seed: int = 2026,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.seed = seed
        self.values: deque[float] = deque(maxlen=window_size)

    def reset(self) -> None:
        super().reset()
        self.values.clear()

    def update(self, value: float, timestamp: datetime) -> None:
        self.values.append(float(value))
        if len(self.values) < self.window_size:
            self._result = DetectorResult(False, 0.0, "WARMUP", value, timestamp)
            return

        data = np.asarray(self.values)
        recent = data[-self.stat_size :]
        reference = data[: -self.stat_size]
        test = ks_2samp(reference, recent)

        score = float(np.clip(1.0 - test.pvalue, 0.0, 1.0))
        self._result = DetectorResult(
            bool(test.pvalue < self.alpha), score, "READY", value, timestamp
        )


# ═══════════════════════════════════════════════════════════════
# Robust Z-Score — 抗异常值的离群点检测
# ═══════════════════════════════════════════════════════════════


class RobustZDetector(DetectorBase):
    """鲁棒 Z-Score 离群点检测器。

    核心问题："最新这个值，在历史的正常范围里，是不是太离谱了？"

    使用中位数 + MAD（中位数绝对偏差）代替均值 + 标准差。
    即使历史数据中混入了极端异常值，中位数和 MAD 也不受影响
    （避免掩蔽效应 masking effect）。

    0.6745 是正态分布下 MAD 对标标准差的比例常数。

    Args:
        window_size: 滑动窗口大小。
        warmup: 热身期（积累多少个值后才开始判定）。
        z_threshold: Z 值触发阈值（默认 3.5，对应约 0.05% 假阳性率）。
    """

    name = "ROBUST_Z"

    def __init__(
        self,
        window_size: int = 10,
        warmup: int = 5,
        z_threshold: float = 3.5,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.warmup = warmup
        self.z_threshold = z_threshold
        self.values: deque[float] = deque(maxlen=window_size)

    def reset(self) -> None:
        super().reset()
        self.values.clear()

    def update(self, value: float, timestamp: datetime) -> None:
        if len(self.values) < self.warmup:
            self.values.append(float(value))
            self._result = DetectorResult(False, 0.0, "WARMUP", value, timestamp)
            return

        data = np.asarray(self.values)
        median = float(np.median(data))
        mad = float(np.median(np.abs(data - median)))

        z = abs(0.67448975 * (float(value) - median) / max(mad, 1e-9))
        self.values.append(float(value))

        score = float(np.clip(z / self.z_threshold, 0.0, 1.0))
        self._result = DetectorResult(
            z > self.z_threshold, score, "READY", value, timestamp
        )
