"""任务三共享合同。"""

from .a7_contracts import (
    A7Authorization,
    A7DecisionEnvelope,
    A7PrimaryRootCause,
    AffectedSegment,
    L1StrategyDecision,
)
from .data_incident import DataIncident, DerivedDataView
from .decision_proposal import (
    DecisionInput,
    DecisionProposal,
    MetricDegradation,
    RootCauseCandidate,
    StrategySelection,
)
from .failure_report import FailureReport, RepairCaseRecord
from .manual_review import ManualReviewReport, ManualReviewSubmission
from .qualification import (
    MetricComparison,
    QualificationGateResult,
    QualificationInput,
    QualificationReport,
)
from .risk_assessment import RiskAssessment
from .round_control import RetryIdentity, RoundTransition
from .training_plan import TrainingPlan, TrainingWindowSpec

__all__ = [
    "A7Authorization",
    "A7DecisionEnvelope",
    "A7PrimaryRootCause",
    "AffectedSegment",
    "DataIncident",
    "DecisionInput",
    "DecisionProposal",
    "DerivedDataView",
    "FailureReport",
    "L1StrategyDecision",
    "ManualReviewReport",
    "ManualReviewSubmission",
    "MetricComparison",
    "MetricDegradation",
    "QualificationGateResult",
    "QualificationInput",
    "QualificationReport",
    "RepairCaseRecord",
    "RiskAssessment",
    "RetryIdentity",
    "RootCauseCandidate",
    "RoundTransition",
    "StrategySelection",
    "TrainingPlan",
    "TrainingWindowSpec",
]
