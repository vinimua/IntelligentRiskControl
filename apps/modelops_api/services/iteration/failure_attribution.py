"""将资格门失败转换为可反馈给 Agent 的结构化失败报告。"""

from datetime import UTC, datetime
from uuid import uuid4

from packages.models.common.enums import FailureCode, QualificationGateCode
from packages.models.iteration.failure_report import FailureReport
from packages.models.iteration.qualification import QualificationReport


GATE_FAILURE_CODES = {
    QualificationGateCode.DATA_REPRODUCIBILITY: FailureCode.DATA_GATE_BLOCKED,
    QualificationGateCode.TARGET_RECOVERY: FailureCode.TARGET_RECOVERY_FAILED,
    QualificationGateCode.DISCRIMINATION: FailureCode.DISCRIMINATION_FAILED,
    QualificationGateCode.CALIBRATION: FailureCode.CALIBRATION_FAILED,
    QualificationGateCode.STABILITY: FailureCode.STABILITY_FAILED,
    QualificationGateCode.SEGMENT_GOVERNANCE: FailureCode.SEGMENT_GOVERNANCE_FAILED,
    QualificationGateCode.OOT: FailureCode.OOT_FAILED,
}


class FailureAttributionService:
    def from_qualification(
        self, proposal_id: str, report: QualificationReport
    ) -> FailureReport | None:
        if report.qualified:
            return None

        failed_gate = report.failed_gate_codes[0]
        reasons = [
            reason
            for gate in report.gate_results
            if gate.gate_code in report.failed_gate_codes
            for reason in gate.reasons
        ]
        recommendations = [
            f"针对 {gate.value} 调整策略并生成新的 experiment_id"
            for gate in report.failed_gate_codes
        ]
        unstable_features = _extract_unstable_features(report)
        return FailureReport(
            failure_report_id=str(uuid4()),
            iteration_run_id=report.iteration_run_id,
            experiment_id=report.experiment_id,
            proposal_id=proposal_id,
            failure_code=GATE_FAILURE_CODES[failed_gate],
            failed_gate_codes=[code.value for code in report.failed_gate_codes],
            reasons=reasons,
            adjustment_recommendations=recommendations,
            retryable=True,
            created_at=datetime.now(UTC),
            unstable_feature_codes=unstable_features,
            feature_evidence_source=(
                "QUALIFICATION_STABILITY_GATE_REASONS"
                if unstable_features else None
            ),
        )


def _extract_unstable_features(report: QualificationReport) -> list[str]:
    """从 STABILITY 门失败原因中提取经归因确认的不稳定特征码。

    保守策略：只在原因文本里出现显式 feature 标注时提取；
    提取不到就返回空列表 —— 空列表不得授予
    unstable_feature_subset_confirmed / feature_selection_evidence_available。
    """
    import re

    from packages.models.common.enums import QualificationStatus

    features: list[str] = []
    for gate in report.gate_results:
        if (
            gate.gate_code.value != "STABILITY"
            or gate.status == QualificationStatus.PASSED
        ):
            continue
        # 结构化来源优先（QualificationService 从特征级 PSI 计算）
        if gate.unstable_feature_codes:
            features.extend(gate.unstable_feature_codes)
            continue
        # 兜底：原因文本中的显式 feature 标注
        for reason in gate.reasons:
            for match in re.findall(
                r"feature[=\s:：]+([A-Za-z0-9_]+)", str(reason), re.IGNORECASE
            ):
                features.append(match)
    return sorted(set(features))
