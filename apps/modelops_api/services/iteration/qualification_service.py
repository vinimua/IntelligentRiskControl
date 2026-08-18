"""Challenger 七道资格门的确定性判定。"""

from packages.models.common.enums import (
    QualificationGateCode,
    QualificationStatus,
)
from packages.models.iteration.qualification import (
    MetricComparison,
    QualificationGateResult,
    QualificationInput,
    QualificationReport,
)

from .config_loader import IterationConfigBundle, load_iteration_config


class QualificationEvidenceIncompleteError(ValueError):
    """资格评估证据不完整：缺失必填实验字段时拒绝评估，禁止 fail-open。"""

    def __init__(self, missing_fields: list[str]):
        self.missing_fields = missing_fields
        super().__init__(
            "QUALIFICATION_EVIDENCE_INCOMPLETE: " + ",".join(missing_fields)
        )


def _segment_governance_passed(segment_metrics: dict) -> bool | None:
    """兼容 Worker 两种键名：segment_governance_passed（新）与 passed（旧）。

    返回 None 表示证据缺失（QUALIFICATION_EVIDENCE_INCOMPLETE），
    与 False（模型未达标）语义区分。
    """
    if segment_metrics is None:
        return None
    value = segment_metrics.get("segment_governance_passed")
    if value is None:
        value = segment_metrics.get("passed")
    return bool(value) if value is not None else None


# 资格评估必填证据（W3 验证结果 + OOT 写回），缺失即拒绝评估
_REQUIRED_EXPERIMENT_FIELDS = (
    "challenger_auc", "challenger_ks", "score_psi", "train_valid_gap",
    "discrimination_passed", "calibration_passed",
)


def build_qualification_input(
    *,
    qualification_run_id: str,
    iteration_run_id: str,
    experiment_id: str,
    candidate_version: str,
    experiment_json: dict,
    feature_psi: dict[str, float] | None = None,
    config: IterationConfigBundle | None = None,
    include_oot: bool = True,
) -> QualificationInput:
    """从受信任的 experiment_json 构建内部 QualificationInput。

    Graph 资格节点与外部资格端点共用的唯一构建入口：
    - 目标恢复字段完整读取（original_drop / recovered_amount /
      healthy bounds / bootstrap CIs），杜绝固定失败
    - 必填实验字段集中完整性检查，缺失抛
      QualificationEvidenceIncompleteError（禁止 fail-open）
    - oot_window_id 来自 qualification.yaml 配置，不硬编码
    - include_oot=False → W3 预资格：不要求 OOT 证据（W4 尚不可用）
    """
    rule = (config or load_iteration_config()).qualification
    validation_metrics = experiment_json.get("validation_metrics") or {}
    segment_metrics = experiment_json.get("segment_metrics") or {}

    # 动态完整性检查：根据启用的规则决定所需证据。
    # 证据缺失（QUALIFICATION_EVIDENCE_INCOMPLETE）与模型未达标
    # （门禁 FAILED）必须保持不同语义。
    missing = [
        field
        for field in _REQUIRED_EXPERIMENT_FIELDS
        if validation_metrics.get(field) is None
    ]
    missing += [
        field
        for field in ("recovery_auc", "recovery_ks")
        if validation_metrics.get(field) is None
    ]
    if _segment_governance_passed(segment_metrics) is None:
        missing.append("segment_governance_passed")
    if rule.require_healthy_range:
        for field in ("healthy_lower_bound", "ks_healthy_lower_bound"):
            if validation_metrics.get(field) is None:
                missing.append(field)
    if rule.require_same_sample_bootstrap:
        for field in (
            "bootstrap_ci_lower", "bootstrap_ci_upper",
            "ks_bootstrap_ci_lower", "ks_bootstrap_ci_upper",
        ):
            if validation_metrics.get(field) is None:
                missing.append(field)
    if experiment_json.get("data_reproducible") is None:
        missing.append("data_reproducible")
    if include_oot:
        # OOT 证据只在最终资格要求（W3 预资格时 W4 尚不可用）
        for field in ("candidate_frozen_before_oot", "oot_passed"):
            if experiment_json.get(field) is None:
                missing.append(field)
    if missing:
        raise QualificationEvidenceIncompleteError(missing)

    def _f(key: str, default=None):
        value = validation_metrics.get(key)
        return value if value is not None else default

    target_metrics: list[MetricComparison] = []
    if _f("challenger_auc") is not None:
        target_metrics.append(MetricComparison(
            metric_code="AUC",
            direction="HIGHER_BETTER",
            original_drop=_f("original_drop"),
            recovered_amount=_f("recovered_amount"),
            recovery_rate=_f("recovery_auc"),
            champion_value=_f("champion_auc"),
            challenger_value=_f("challenger_auc"),
            healthy_lower_bound=_f("healthy_lower_bound"),
            healthy_upper_bound=_f("healthy_upper_bound"),
            bootstrap_ci_lower=_f("bootstrap_ci_lower"),
            bootstrap_ci_upper=_f("bootstrap_ci_upper"),
        ))
    if _f("challenger_ks") is not None:
        target_metrics.append(MetricComparison(
            metric_code="KS",
            direction="HIGHER_BETTER",
            original_drop=None,
            recovered_amount=None,
            recovery_rate=_f("recovery_ks"),
            champion_value=_f("champion_ks"),
            challenger_value=_f("challenger_ks"),
            # KS 专属健康区间与 Bootstrap（Worker 真实计算，
            # AUC 的 healthy_lower_bound 不能套给 KS）
            healthy_lower_bound=_f("ks_healthy_lower_bound"),
            healthy_upper_bound=None,
            bootstrap_ci_lower=_f("ks_bootstrap_ci_lower"),
            bootstrap_ci_upper=_f("ks_bootstrap_ci_upper"),
        ))

    return QualificationInput(
        qualification_run_id=qualification_run_id,
        iteration_run_id=iteration_run_id,
        experiment_id=experiment_id,
        candidate_version=candidate_version,
        target_metrics=target_metrics,
        data_reproducible=bool(experiment_json.get("data_reproducible")),
        discrimination_passed=bool(_f("discrimination_passed")),
        calibration_passed=bool(_f("calibration_passed")),
        score_psi=float(_f("score_psi", 1.0)),
        train_valid_gap=float(_f("train_valid_gap")),
        segment_governance_passed=bool(
            _segment_governance_passed(segment_metrics)
        ),
        oot_window_id=rule.required_oot_window_id,
        candidate_frozen_before_oot=(
            bool(experiment_json.get("candidate_frozen_before_oot"))
            if include_oot else False
        ),
        # 冻结身份校验和：晋升防换包（Worker 冻结时写回）
        frozen_identity_checksum=(
            str(experiment_json.get("frozen_identity_checksum") or "")
            if include_oot else ""
        ),
        oot_usage="FINAL_QUALIFICATION",
        oot_passed=(
            bool(experiment_json.get("oot_passed")) if include_oot else False
        ),
        w4_read_count=(
            experiment_json.get("w4_read_count") if include_oot else None
        ),
        frozen_identity_matches=(
            experiment_json.get("frozen_identity_matches") if include_oot else None
        ),
        oot_metrics_available=(
            experiment_json.get("oot_metrics_available") if include_oot else None
        ),
        feature_psi=feature_psi or {},
    )


class QualificationService:
    def __init__(self, config: IterationConfigBundle | None = None) -> None:
        self.config = config or load_iteration_config()

    @staticmethod
    def _metric_recovery_rate(metric: MetricComparison) -> float | None:
        if metric.recovery_rate is not None:
            return metric.recovery_rate
        if metric.original_drop in (None, 0) or metric.recovered_amount is None:
            return None
        return abs(metric.recovered_amount) / abs(metric.original_drop)

    @staticmethod
    def _in_healthy_range(metric: MetricComparison) -> bool:
        value = metric.challenger_value
        if value is None:
            return False
        if (
            metric.healthy_lower_bound is not None
            and value < metric.healthy_lower_bound
        ):
            return False
        if (
            metric.healthy_upper_bound is not None
            and value > metric.healthy_upper_bound
        ):
            return False
        return (
            metric.healthy_lower_bound is not None
            or metric.healthy_upper_bound is not None
        )

    @staticmethod
    def _bootstrap_supports_improvement(metric: MetricComparison) -> bool:
        if metric.bootstrap_ci_lower is None or metric.bootstrap_ci_upper is None:
            return False
        direction = metric.direction.upper()
        if direction == "HIGHER_BETTER":
            return metric.bootstrap_ci_lower > 0
        if direction == "LOWER_BETTER":
            return metric.bootstrap_ci_upper < 0
        return not (
            metric.bootstrap_ci_lower <= 0 <= metric.bootstrap_ci_upper
        )

    def _target_gate(
        self, metrics: list[MetricComparison]
    ) -> QualificationGateResult:
        rule = self.config.qualification
        reasons: list[str] = []
        details: dict[str, dict] = {}
        if not metrics:
            reasons.append("NO_TARGET_METRICS")
        # 严格 A7：AUC + KS 双指标必须齐备，缺一即失败（不许用单指标蒙混）
        present = {str(m.metric_code or "").upper() for m in metrics}
        missing_targets = {"AUC", "KS"} - present
        if missing_targets:
            reasons.append(
                "REQUIRED_TARGET_METRICS_MISSING:" + ",".join(sorted(missing_targets))
            )
        for metric in metrics:
            recovery_rate = self._metric_recovery_rate(metric)
            healthy = self._in_healthy_range(metric)
            bootstrap = self._bootstrap_supports_improvement(metric)
            details[metric.metric_code] = {
                "recovery_rate": recovery_rate,
                "healthy_range_reached": healthy,
                "bootstrap_improvement_supported": bootstrap,
            }
            if recovery_rate is None or recovery_rate < rule.min_recovery_rate:
                reasons.append(f"RECOVERY_RATE_FAILED:{metric.metric_code}")
            if rule.require_healthy_range and not healthy:
                reasons.append(f"HEALTHY_RANGE_FAILED:{metric.metric_code}")
            if rule.require_same_sample_bootstrap and not bootstrap:
                reasons.append(f"BOOTSTRAP_FAILED:{metric.metric_code}")
        return self._gate(
            QualificationGateCode.TARGET_RECOVERY,
            passed=not reasons,
            reasons=reasons,
            metrics=details,
        )

    @staticmethod
    def _gate(
        code: QualificationGateCode,
        *,
        passed: bool,
        reasons: list[str] | None = None,
        metrics: dict | None = None,
        unstable_feature_codes: list[str] | None = None,
        status: QualificationStatus | None = None,
    ) -> QualificationGateResult:
        return QualificationGateResult(
            gate_code=code,
            gate_order=list(QualificationGateCode).index(code),
            status=status or (
                QualificationStatus.PASSED
                if passed
                else QualificationStatus.FAILED
            ),
            reasons=reasons or [],
            metrics=metrics or {},
            unstable_feature_codes=unstable_feature_codes or [],
        )

    def evaluate(
        self,
        request: QualificationInput,
        *,
        include_oot: bool = True,
    ) -> QualificationReport:
        """七道资格门判定。

        include_oot=False → W3 预资格（Gate 0-5，不含 OOT）：
        OOT 门需要 W4 结果，而 W4 由 OOT_GATE 阶段执行并回写实验后
        才可用 —— 最终资格（Gate 6 + 汇总前六道）在 OOT 完成后重跑。
        """
        rule = self.config.qualification
        # 阈值来自服务端配置（qualification.yaml），API 请求无法篡改
        unstable_features = [
            fname
            for fname, psi in request.feature_psi.items()
            if psi > rule.feature_psi_threshold
        ]
        # ── 冻结身份三要素：数据可复现 + 候选 OOT 前冻结 + 冻结产物校验和 ──
        # 冻结身份是晋升语境（最终资格）的硬要求；W3 预资格（include_oot=False）
        # 只检查数据可复现（候选尚未进入 OOT 晋升流程）。
        data_repro_conditions = [
            (
                not request.data_reproducible,
                "DATA_NOT_REPRODUCIBLE",
            ),
            (
                include_oot and not request.candidate_frozen_before_oot,
                "CANDIDATE_NOT_FROZEN_BEFORE_OOT",
            ),
            (
                include_oot and not request.frozen_identity_checksum,
                "FROZEN_IDENTITY_CHECKSUM_MISSING",
            ),
        ]
        gates = [
            self._gate(
                QualificationGateCode.DATA_REPRODUCIBILITY,
                passed=not any(c for c, _ in data_repro_conditions),
                reasons=[reason for c, reason in data_repro_conditions if c],
            ),
            self._target_gate(request.target_metrics),
            self._gate(
                QualificationGateCode.DISCRIMINATION,
                # Bad Recall 护栏：主指标修复不得换来坏样本召回崩坏
                passed=(
                    request.discrimination_passed
                    and request.bad_recall_passed is not False
                ),
                reasons=[
                    reason
                    for condition, reason in (
                        (
                            not request.discrimination_passed,
                            "DISCRIMINATION_REGRESSION",
                        ),
                        (
                            request.bad_recall_passed is False,
                            "BAD_RECALL_GUARDRAIL_FAILED",
                        ),
                    )
                    if condition
                ],
            ),
            self._gate(
                QualificationGateCode.CALIBRATION,
                passed=request.calibration_passed,
                reasons=[] if request.calibration_passed else ["CALIBRATION_REGRESSION"],
            ),
            self._gate(
                QualificationGateCode.STABILITY,
                passed=(
                    request.score_psi <= rule.max_score_psi
                    and request.train_valid_gap <= rule.max_train_valid_gap
                    and not unstable_features
                ),
                reasons=[
                    reason
                    for condition, reason in (
                        (
                            request.score_psi > rule.max_score_psi,
                            "SCORE_PSI_EXCEEDED",
                        ),
                        (
                            request.train_valid_gap > rule.max_train_valid_gap,
                            "TRAIN_VALID_GAP_EXCEEDED",
                        ),
                        (
                            bool(unstable_features),
                            f"FEATURE_PSI_EXCEEDED:{','.join(unstable_features)}",
                        ),
                    )
                    if condition
                ],
                metrics={
                    "score_psi": request.score_psi,
                    "train_valid_gap": request.train_valid_gap,
                },
                # A7 §5: 结构化不稳定特征，来自特征级 PSI 结果，
                # 不依赖原因文本正则；参与门禁判定
                unstable_feature_codes=unstable_features,
            ),
            self._gate(
                QualificationGateCode.SEGMENT_GOVERNANCE,
                passed=request.segment_governance_passed,
                reasons=(
                    []
                    if request.segment_governance_passed
                    else ["SEGMENT_OR_GOVERNANCE_FAILED"]
                ),
            ),
            self._gate(
                QualificationGateCode.OOT,
                # OOT 指标不可用 → INCONCLUSIVE 悬置（不淘汰候选，等补测）
                passed=(
                    request.oot_window_id == rule.required_oot_window_id
                    and request.candidate_frozen_before_oot
                    and request.oot_usage == "FINAL_QUALIFICATION"
                    and request.oot_passed
                ),
                status=(
                    QualificationStatus.INCONCLUSIVE
                    if request.oot_metrics_available is False
                    else None
                ),
                reasons=[
                    reason
                    for condition, reason in (
                        (
                            request.oot_metrics_available is False,
                            "OOT_METRIC_UNAVAILABLE",
                        ),
                        (
                            request.oot_window_id != rule.required_oot_window_id,
                            "INVALID_OOT_WINDOW",
                        ),
                        (
                            not request.candidate_frozen_before_oot,
                            "CANDIDATE_NOT_FROZEN_BEFORE_OOT",
                        ),
                        (
                            request.oot_usage != "FINAL_QUALIFICATION",
                            "OOT_USAGE_FORBIDDEN",
                        ),
                        (
                            request.w4_read_count is not None
                            and request.w4_read_count != 1,
                            "W4_READ_COUNT_INVALID",
                        ),
                        (
                            request.frozen_identity_matches is False,
                            "FROZEN_IDENTITY_MISMATCH",
                        ),
                        (not request.oot_passed, "OOT_PERFORMANCE_FAILED"),
                    )
                    if condition
                ],
                metrics={"oot_window_id": request.oot_window_id},
            ),
        ]
        if not include_oot:
            # W3 预资格：移除 OOT 门（Gate 6），W4 结果尚不可用
            gates = [
                gate for gate in gates
                if gate.gate_code != QualificationGateCode.OOT
            ]
        failed = [
            gate.gate_code
            for gate in gates
            if gate.required and gate.status == QualificationStatus.FAILED
        ]
        qualified = not failed
        return QualificationReport(
            qualification_run_id=request.qualification_run_id,
            iteration_run_id=request.iteration_run_id,
            experiment_id=request.experiment_id,
            candidate_version=request.candidate_version,
            status=(
                QualificationStatus.PASSED
                if qualified
                else QualificationStatus.FAILED
            ),
            qualified=qualified,
            gate_results=gates,
            failed_gate_codes=failed,
            rule_version=rule.rule_version,
        )
