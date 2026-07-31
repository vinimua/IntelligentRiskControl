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
        )
