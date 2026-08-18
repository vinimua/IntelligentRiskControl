"""Run the real credit_model_007 A7 P0 vertical fixture without DB/Celery.

The runner uses the production A7 contracts, L1 selector, plan builder and
Worker computation functions. W4 is loaded only after PRE_OOT passes.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
from sklearn.metrics import brier_score_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.modelops_api.services.deployment.deployment_oot_service import (  # noqa: E402
    load_frozen_challenger,
    run_oot_validation,
)
from apps.modelops_api.services.iteration import (  # noqa: E402
    QualificationService,
    RepairDecisionService,
    RiskAssessmentService,
    TrainingPlanBuilder,
)
from apps.modelops_api.services.monitoring.window_loader import load_window  # noqa: E402
from packages.models.iteration import (  # noqa: E402
    A7DecisionEnvelope,
    MetricComparison,
    QualificationInput,
)
from packages.models.iteration.training_job import (  # noqa: E402
    TrainingConsumptionReceipt,
    TrainingJobInput,
)
from workers.training_tasks import (  # noqa: E402
    _bad_recall_at_top20,
    _build_sample_weights,
    _calc_score_psi,
    _check_segment_governance,
    _compute_ks,
    _load_and_score_champion,
    _load_training_data,
    _paired_bootstrap_delta_ci,
    _prepare_features,
    _stable_hash,
    _train_logistic_regression,
    _verify_window_snapshots,
)


MODEL_ID = "credit_model_007"
CHAMPION_VERSION = "champion_v1"
LIFECYCLE_ID = "a7-p0-007-fixture"
CANDIDATE_VERSION = "credit_model_007_a7_p0_challenger_v1"


def _champion_checksum() -> str:
    model_path = (
        PROJECT_ROOT
        / "assets"
        / "champion_models"
        / MODEL_ID
        / CHAMPION_VERSION
        / "model.joblib"
    )
    return "sha256:" + hashlib.sha256(model_path.read_bytes()).hexdigest()


def _envelope() -> A7DecisionEnvelope:
    return A7DecisionEnvelope(
        decision_source="SIMULATED",
        lifecycle_run_id=LIFECYCLE_ID,
        event_id="event-a7-p0-007",
        monitoring_run_id="monitoring-a7-p0-007",
        diagnosis_run_id="diagnosis-a7-p0-007",
        agent_decision_id="agent-a7-p0-007",
        model_id=MODEL_ID,
        champion_version=CHAMPION_VERSION,
        champion_artifact_checksum=_champion_checksum(),
        model_task_type="BINARY_CLASSIFICATION",
        algorithm_family="LogisticRegression",
        recommended_action="MODEL_ITERATION",
        primary_root_cause={
            "root_cause_code": "FEATURE_DRIFT",
            "candidate_status": "CONFIRMED",
            "confidence": 0.92,
            "evidence_refs": ["fixture:confirmed-feature-drift"],
        },
        decay_degree="SHORT_TERM_7D",
        trigger_context={
            "trigger_metric_codes": ["AUC", "KS"],
            "max_alert_severity": "CRITICAL",
        },
        authorization={
            "authorization_type": "AUTO_RULE",
            "authorization_id": "auto-rule-a7-p0-007",
            "approved": True,
        },
        rule_versions={
            "agent_rule_version": "agent-v1",
            "l1_matrix_version": "l1-v1",
            "window_rule_version": "W3_LAST_7D_HOLDOUT_V1",
            "qualification_rule_version": "qualification-gates-v2",
        },
    )


def _job(plan) -> TrainingJobInput:
    return TrainingJobInput(
        training_job_id="job-a7-p0-007",
        idempotency_key="a7-p0-007:round-1",
        model_id=plan.model_id,
        lifecycle_run_id=plan.lifecycle_run_id,
        iteration_run_id=plan.iteration_run_id,
        training_plan_id=plan.training_plan_id,
        experiment_id=plan.experiment_id,
        business_round=plan.business_round,
        strategy_code=plan.strategy_code,
        execution_mode=plan.execution_mode,
        training_data_mode=plan.training_data_mode,
        training_window_ids=plan.windows.training_window_ids,
        validation_window_ids=plan.windows.validation_window_ids,
        train_time_ranges=[item.model_dump() for item in plan.windows.training_time_ranges],
        validation_time_ranges=[item.model_dump() for item in plan.windows.validation_time_ranges],
        oot_window_id=plan.windows.oot_window_id,
        data_snapshot_ids=plan.data_snapshot_ids,
        data_snapshot_checksums=plan.data_snapshot_checksums,
        label_versions=plan.label_versions,
        sample_weight_policy=plan.sample_weight_policy,
        sample_weight_required=plan.sample_weight_required,
        affected_segments=plan.affected_segments,
        change_point=plan.change_point,
        feature_schema_version=plan.feature_schema_version,
        ordered_features=plan.ordered_features,
        ordered_features_hash=plan.ordered_features_hash,
        preprocessing_version=plan.preprocessing_version,
        preprocessing_hash=plan.preprocessing_hash,
        algorithm=plan.algorithm,
        algorithm_family=plan.algorithm_family,
        champion_artifact_checksum=plan.champion_artifact_checksum,
        hyperparameters=plan.hyperparameter_space,
        target_metrics=plan.target_metric_codes,
        qualification_rule_version=plan.qualification_rule_version,
        base_model_version=plan.frozen_champion_version,
        seed=plan.random_seed,
        artifact_output_uri=(
            f"s3://riskitem/challengers/{MODEL_ID}/{LIFECYCLE_ID}/{CANDIDATE_VERSION}"
        ),
        training_mode="full",
    )


def _freeze(model, job: TrainingJobInput, feature_cols: list[str]) -> tuple[str, Path]:
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    model_bytes = buffer.getvalue()
    checksum = hashlib.sha256(model_bytes).hexdigest()
    base = (
        PROJECT_ROOT
        / "artifacts"
        / "minio_fallback"
        / "riskitem"
        / "challengers"
        / MODEL_ID
        / LIFECYCLE_ID
        / CANDIDATE_VERSION
    )
    base.mkdir(parents=True, exist_ok=True)
    (base / "model.joblib").write_bytes(model_bytes)
    (base / "checksum.sha256").write_text(checksum, encoding="utf-8")
    metadata = {
        "model_id": MODEL_ID,
        "lifecycle_run_id": LIFECYCLE_ID,
        "candidate_version": CANDIDATE_VERSION,
        "feature_cols": feature_cols,
        "ordered_features_hash": job.ordered_features_hash,
        "feature_schema_version": job.feature_schema_version,
        "preprocessing_version": job.preprocessing_version,
        "preprocessing_hash": job.preprocessing_hash,
        "algorithm": job.algorithm,
        "algorithm_family": job.algorithm_family,
        "model_sha256": checksum,
        "training_job_id": job.training_job_id,
        "experiment_id": job.experiment_id,
        "w4_read_count": 0,
    }
    (base / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return f"sha256:{checksum}", base


def run() -> tuple[dict, Path]:
    envelope = _envelope()
    proposal, l1 = RepairDecisionService().propose_a7(envelope)
    risk = RiskAssessmentService().assess(proposal)
    plan = TrainingPlanBuilder().build(
        proposal,
        risk,
        approval_id=envelope.authorization.authorization_id,
        iteration_run_id="iteration-a7-p0-007",
    )
    job = _job(plan)

    train = _load_training_data(
        job.training_window_ids,
        job.data_snapshot_ids,
        [item.model_dump(mode="json") for item in job.train_time_ranges],
    )
    valid = _load_training_data(
        job.validation_window_ids,
        job.data_snapshot_ids,
        [item.model_dump(mode="json") for item in job.validation_time_ranges],
    )
    _verify_window_snapshots(train, job.training_window_ids, job.data_snapshot_checksums)
    _verify_window_snapshots(valid, job.validation_window_ids, job.data_snapshot_checksums)
    overlap_count = len(set(train["sample_id"]).intersection(valid["sample_id"]))
    if overlap_count:
        raise RuntimeError(f"SAMPLE_OVERLAP_DETECTED:{overlap_count}")

    weights, affected_ids = _build_sample_weights(
        train, job.sample_weight_required, job.sample_weight_policy
    )
    trained = _train_logistic_regression(
        train,
        seed=job.seed,
        hyperparameters=job.hyperparameters,
        sample_weight=weights,
        ordered_features=job.ordered_features,
    )
    model = trained["model"]
    unweighted_control = _train_logistic_regression(
        train,
        seed=job.seed,
        hyperparameters=job.hyperparameters,
        sample_weight=None,
        ordered_features=job.ordered_features,
    )
    control_dir = PROJECT_ROOT / "artifacts" / "a7_p0_007" / "unweighted_control"
    control_dir.mkdir(parents=True, exist_ok=True)
    control_model_path = control_dir / "model.joblib"
    joblib.dump(unweighted_control["model"], control_model_path)
    control_checksum = hashlib.sha256(control_model_path.read_bytes()).hexdigest()

    weight_failure_code = None
    try:
        # A required WINDOW_WEIGHT policy on a single equally weighted role
        # must fail instead of claiming that weights were consumed.
        _build_sample_weights(
            train.loc[train["__window_role_id"] == "W1"].copy(),
            True,
            job.sample_weight_policy,
        )
    except ValueError as exc:
        weight_failure_code = str(exc)
    if weight_failure_code != "SAMPLE_WEIGHT_NOT_CONSUMED":
        raise RuntimeError(
            f"WEIGHT_NOT_CONSUMED_NEGATIVE_CASE_FAILED:{weight_failure_code}"
        )
    val_scores = model.predict_proba(_prepare_features(valid, job.ordered_features))[:, 1]
    val_y = valid["is_bad"]
    challenger_auc = roc_auc_score(val_y, val_scores)
    challenger_ks = _compute_ks(val_y, val_scores)
    champion = _load_and_score_champion(
        CHAMPION_VERSION,
        valid,
        job.ordered_features,
        job.algorithm,
        model_id=MODEL_ID,
        expected_feature_schema_version=job.feature_schema_version,
        expected_preprocessing_version=job.preprocessing_version,
    )
    if not champion["loaded"]:
        raise RuntimeError(f"CHAMPION_LOAD_FAILED:{champion['load_errors']}")
    if "sha256:" + champion["checksum"] != job.champion_artifact_checksum:
        raise RuntimeError("CHAMPION_ARTIFACT_CHECKSUM_MISMATCH")

    w1 = load_window("W1")
    champion_w1 = _load_and_score_champion(
        CHAMPION_VERSION,
        w1,
        job.ordered_features,
        job.algorithm,
        model_id=MODEL_ID,
    )
    challenger_w1_scores = model.predict_proba(
        _prepare_features(w1, job.ordered_features)
    )[:, 1]
    score_psi = max(
        _calc_score_psi(champion_w1["scores"], champion["scores"]),
        _calc_score_psi(challenger_w1_scores, val_scores),
    )
    auc_drop = max(0.01, champion_w1["auc"] - champion["auc"])
    ks_drop = max(0.01, champion_w1["ks"] - champion["ks"])
    recovery_auc = max(0.0, (challenger_auc - champion["auc"]) / auc_drop)
    recovery_ks = max(0.0, (challenger_ks - champion["ks"]) / ks_drop)
    auc_ci = _paired_bootstrap_delta_ci(
        val_y, champion["scores"], val_scores, "AUC", seed=job.seed
    )
    ks_ci = _paired_bootstrap_delta_ci(
        val_y, champion["scores"], val_scores, "KS", seed=job.seed
    )
    bad_recall_champion = _bad_recall_at_top20(val_y, champion["scores"])
    bad_recall_challenger = _bad_recall_at_top20(val_y, val_scores)
    bad_recall_passed = bad_recall_challenger >= bad_recall_champion
    challenger_brier = brier_score_loss(val_y, val_scores)
    champion_brier = brier_score_loss(val_y, champion["scores"])
    calibration_passed = challenger_brier <= champion_brier + 0.01
    segment = {"passed": True, "segments": {}, "status": "NOT_APPLICABLE"}
    train_valid_gap = abs(trained["train_auc"] - challenger_auc)
    healthy_auc = max(champion_w1["auc"] - 0.02, 0.72)
    healthy_ks = max(champion_w1["ks"] - 0.02, 0.20)

    artifact_checksum, artifact_dir = _freeze(model, job, job.ordered_features)
    receipt = TrainingConsumptionReceipt(
        consumed_training_snapshot_ids=[f"window:{item}" for item in job.training_window_ids],
        consumed_validation_snapshot_ids=[f"window:{item}" for item in job.validation_window_ids],
        observed_train_sample_count=len(train),
        observed_validation_sample_count=len(valid),
        observed_train_bad_count=int(train["is_bad"].sum()),
        observed_validation_bad_count=int(valid["is_bad"].sum()),
        sample_overlap_count=overlap_count,
        actual_algorithm_family=job.algorithm_family,
        actual_execution_mode=job.execution_mode,
        actual_base_model_checksum=None,
        sample_weight_consumed=True,
        sample_weight_min=float(weights.min()),
        sample_weight_max=float(weights.max()),
        sample_weight_mean=float(weights.mean()),
        non_unit_weight_sample_count=int((abs(weights - 1.0) > 1e-12).sum()),
        affected_segment_ids_consumed=affected_ids,
        actual_ordered_features_hash=_stable_hash(job.ordered_features),
        actual_preprocessing_hash=job.preprocessing_hash,
        actual_hyperparameters_hash=_stable_hash(job.hyperparameters),
        w4_read_count=0,
    )

    targets = [
        MetricComparison(
            metric_code="AUC",
            direction="HIGHER_BETTER",
            original_drop=auc_drop,
            recovered_amount=max(0.0, challenger_auc - champion["auc"]),
            recovery_rate=recovery_auc,
            champion_value=champion["auc"],
            challenger_value=challenger_auc,
            healthy_lower_bound=healthy_auc,
            bootstrap_ci_lower=auc_ci[0] if auc_ci else None,
            bootstrap_ci_upper=auc_ci[1] if auc_ci else None,
        ),
        MetricComparison(
            metric_code="KS",
            direction="HIGHER_BETTER",
            original_drop=ks_drop,
            recovered_amount=max(0.0, challenger_ks - champion["ks"]),
            recovery_rate=recovery_ks,
            champion_value=champion["ks"],
            challenger_value=challenger_ks,
            healthy_lower_bound=healthy_ks,
            bootstrap_ci_lower=ks_ci[0] if ks_ci else None,
            bootstrap_ci_upper=ks_ci[1] if ks_ci else None,
        ),
    ]
    pre = QualificationService().evaluate(
        QualificationInput(
            qualification_run_id="pre-oot-a7-p0-007",
            iteration_run_id=job.iteration_run_id,
            experiment_id=job.experiment_id,
            candidate_version=CANDIDATE_VERSION,
            qualification_stage="PRE_OOT",
            target_metrics=targets,
            data_reproducible=True,
            discrimination_passed=(challenger_auc >= champion["auc"] - 0.01),
            calibration_passed=calibration_passed,
            score_psi=score_psi,
            train_valid_gap=train_valid_gap,
            segment_governance_passed=bool(segment["passed"]),
            segment_governance_required=False,
            bad_recall_passed=bad_recall_passed,
            candidate_frozen_before_oot=True,
            oot_usage="PRE_OOT",
            w4_read_count=0,
            frozen_identity_checksum=artifact_checksum,
        )
    )

    final = None
    oot_metrics = None
    if pre.allow_w4:
        frozen = load_frozen_challenger(MODEL_ID, LIFECYCLE_ID, CANDIDATE_VERSION)
        if not frozen["loaded"]:
            raise RuntimeError(f"FROZEN_CHALLENGER_LOAD_FAILED:{frozen['load_errors']}")
        oot_metrics = run_oot_validation(
            frozen["model"],
            frozen["feature_cols"],
            model_id=MODEL_ID,
            lifecycle_run_id=LIFECYCLE_ID,
            candidate_version=CANDIDATE_VERSION,
            champion_version=CHAMPION_VERSION,
        )
        final = QualificationService().evaluate(
            QualificationInput(
                qualification_run_id="final-oot-a7-p0-007",
                iteration_run_id=job.iteration_run_id,
                experiment_id=job.experiment_id,
                candidate_version=CANDIDATE_VERSION,
                qualification_stage="FINAL_OOT",
                target_metrics=[],
                data_reproducible=True,
                discrimination_passed=True,
                calibration_passed=True,
                score_psi=0.0,
                train_valid_gap=0.0,
                segment_governance_passed=True,
                bad_recall_passed=True,
                oot_window_id="W4",
                candidate_frozen_before_oot=True,
                oot_usage="FINAL_QUALIFICATION",
                oot_passed=oot_metrics["oot_passed"],
                oot_metrics_available=oot_metrics["oot_metrics_available"],
                frozen_identity_matches=(
                    "sha256:" + frozen["checksum"] == artifact_checksum
                ),
                w4_read_count=oot_metrics["w4_read_count"],
                frozen_identity_checksum=artifact_checksum,
            )
        )

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "A7_P0_CREDIT_MODEL_007_ONLY",
        "envelope": envelope.model_dump(mode="json"),
        "l1": l1.model_dump(mode="json"),
        "proposal": proposal.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "job": job.model_dump(mode="json"),
        "consumption_receipt": receipt.model_dump(mode="json"),
        "worker_acceptance_cases": {
            "weighted_refit": {
                "status": "SUCCEEDED",
                "artifact_checksum": artifact_checksum,
                "non_unit_weight_sample_count": receipt.non_unit_weight_sample_count,
            },
            "unweighted_refit": {
                "status": "SUCCEEDED",
                "artifact_path": str(control_model_path),
                "artifact_checksum": f"sha256:{control_checksum}",
                "train_auc": unweighted_control["train_auc"],
            },
            "weight_not_consumed": {
                "status": "BLOCKED",
                "failure_code": weight_failure_code,
                "artifact_created": False,
            },
        },
        "validation_metrics": {
            "champion_auc": champion["auc"],
            "challenger_auc": challenger_auc,
            "recovery_auc": recovery_auc,
            "champion_ks": champion["ks"],
            "challenger_ks": challenger_ks,
            "recovery_ks": recovery_ks,
            "score_psi": score_psi,
            "train_valid_gap": train_valid_gap,
            "bad_recall_passed": bad_recall_passed,
            "calibration_passed": calibration_passed,
            "segment_governance_passed": segment["passed"],
        },
        "pre_oot": pre.model_dump(mode="json"),
        "w4_accessed": pre.allow_w4,
        "oot_metrics": oot_metrics,
        "final_oot": final.model_dump(mode="json") if final else None,
        "artifact_directory": str(artifact_dir),
        "claims": {
            "p0_007_vertical_complete": bool(pre.qualified and final and final.qualified),
            "all_algorithms_complete": False,
            "all_50_models_complete": False,
        },
    }
    report_dir = PROJECT_ROOT / "artifacts" / "a7_p0_007"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "latest_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report, report_path


if __name__ == "__main__":
    fixture_report, path = run()
    print(
        json.dumps(
            {
                "report": str(path),
                "pre_oot_status": fixture_report["pre_oot"]["status"],
                "w4_accessed": fixture_report["w4_accessed"],
                "final_oot_status": (
                    fixture_report["final_oot"]["status"]
                    if fixture_report["final_oot"]
                    else None
                ),
                "claims": fixture_report["claims"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
