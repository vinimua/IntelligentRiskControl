"""Challenger 七道资格门合同。"""

from pydantic import Field

from ..common.base import ContractModel
from ..common.enums import QualificationGateCode, QualificationStatus


class MetricComparison(ContractModel):
    metric_code: str
    direction: str
    original_drop: float | None = None
    recovered_amount: float | None = None
    recovery_rate: float | None = None
    champion_value: float | None = None
    challenger_value: float | None = None
    healthy_lower_bound: float | None = None
    healthy_upper_bound: float | None = None
    bootstrap_ci_lower: float | None = None
    bootstrap_ci_upper: float | None = None


class QualificationGateResult(ContractModel):
    gate_code: QualificationGateCode
    gate_order: int = Field(ge=0, le=6)
    status: QualificationStatus
    required: bool = True
    metric_code: str | None = None
    expected: dict = Field(default_factory=dict)
    actual: dict = Field(default_factory=dict)
    bootstrap_interval: tuple[float, float] | None = None
    affected_segments: list[str] = Field(default_factory=list)
    failure_code: str | None = None
    reasons: list[str] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    # A7 §5: 结构化不稳定特征（从特征级 PSI 结果计算，不依赖原因文本正则）
    unstable_feature_codes: list[str] = Field(default_factory=list)


class QualificationInput(ContractModel):
    qualification_run_id: str
    iteration_run_id: str
    experiment_id: str
    candidate_version: str
    target_metrics: list[MetricComparison] = Field(default_factory=list)
    data_reproducible: bool
    # 冻结身份三要素：候选在 OOT 前冻结 + 冻结产物校验和（防"换包晋升"）
    candidate_frozen_before_oot: bool
    frozen_identity_checksum: str | None = None
    discrimination_passed: bool
    # Bad Recall 护栏（W3 判别门的次要指标，不允许主指标修复换来坏样本召回崩坏）
    bad_recall_passed: bool | None = None
    calibration_passed: bool
    score_psi: float
    train_valid_gap: float
    segment_governance_passed: bool
    # SEGMENT_GOVERNANCE 是否为必需门（无分段治理需求时可不阻断）
    segment_governance_required: bool = True
    oot_window_id: str
    oot_usage: str = "FINAL_QUALIFICATION"
    oot_passed: bool
    # OOT 门治理证据：W4 只读一次 + 冻结身份与晋升包一致 + 指标可用性
    w4_read_count: int | None = None
    frozen_identity_matches: bool | None = None
    oot_metrics_available: bool | None = None
    # 特征级 PSI（feature → psi），用于 STABILITY 门生成结构化不稳定特征。
    # 内部合同字段：由服务端根据受信任的 monitoring_run_id 加载，
    # 外部 API 请求不得提交（QualificationRequest 独立建模）。
    # 阈值不在此模型：统一由 qualification.yaml 配置，调用方无法篡改。
    feature_psi: dict[str, float] = Field(default_factory=dict)


class QualificationReport(ContractModel):
    qualification_run_id: str
    iteration_run_id: str
    experiment_id: str
    candidate_version: str
    status: QualificationStatus
    qualified: bool
    gate_results: list[QualificationGateResult]
    failed_gate_codes: list[QualificationGateCode] = Field(default_factory=list)
    rule_version: str
    # 资格阶段：PRE_OOT（W3 预资格，不得含 W4 结果）/ FINAL_OOT（最终七门汇总）
    qualification_stage: str = "FINAL_OOT"
    # PRE_OOT 通过后允许读取 W4 的显式授权（防未授权 OOT 读取）
    allow_w4: bool = False
