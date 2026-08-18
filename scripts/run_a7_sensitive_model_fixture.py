"""Run an isolated, reproducible sensitive-model A7 fixture.

This fixture never replaces credit_model_007 and never modifies W0-W4.  It
creates a test-only champion, a controlled W3 scenario snapshot, an A7 plan,
a same-family challenger, and a PRE_OOT qualification report.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.modelops_api.services.iteration import (  # noqa: E402
    QualificationService,
    RepairDecisionService,
    RiskAssessmentService,
    TrainingPlanBuilder,
)
from apps.modelops_api.services.monitoring.metric_calculators import (  # noqa: E402
    _compute_psi_frozen,
)
from apps.modelops_api.services.monitoring.scenarios.injectors import (  # noqa: E402
    ScenarioFactory,
)
from apps.modelops_api.services.monitoring.threshold_rules import (  # noqa: E402
    DEFAULT_THRESHOLD_RULES,
)
from apps.modelops_api.services.monitoring.window_loader import load_window  # noqa: E402
from packages.models.iteration import (  # noqa: E402
    A7DecisionEnvelope,
    MetricComparison,
    QualificationInput,
)
from packages.models.iteration.training_job import TrainingConsumptionReceipt  # noqa: E402
from workers.training_tasks import (  # noqa: E402
    _bad_recall_at_top20,
    _calc_score_psi,
    _compute_ks,
    _paired_bootstrap_delta_ci,
    _prepare_features,
    _stable_hash,
    _train_random_forest,
)


MODEL_ID = "fixture_sensitive_007"
CHAMPION_VERSION = "champion_v1"
CANDIDATE_VERSION = "fixture_sensitive_007_challenger_v1"
LIFECYCLE_ID = "a7-sensitive-fixture"
SEED = 2026007
FEATURES = ["reg_to_apply_days"]
HYPERPARAMETERS = {
    "n_estimators": 80,
    "max_depth": 4,
    "min_samples_leaf": 5,
    "max_features": None,
    # Keep raw probabilities meaningful for the mandatory calibration gate.
    # The production prevalence is already represented by the full-population
    # windows, so balancing here would distort probabilities unnecessarily.
    "class_weight": None,
    "n_jobs": 1,
}
SCENARIO = {
    "scenario_name": "covariate_drift",
    # Produces a controlled critical feature drift below the fixture stress
    # ceiling (PSI < 0.50) while crossing both AUC/KS warning thresholds.
    "intensity": 0.35,
    "affected_features": FEATURES,
    "base_window_id": "W3",
}


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str),
        encoding="utf-8",
    )


def _freeze_joblib(path: Path, value: object) -> str:
    buffer = io.BytesIO()
    joblib.dump(value, buffer)
    payload = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256_bytes(payload)


def _build_champion() -> tuple[object, pd.DataFrame, pd.DataFrame, str, Path]:
    w0 = load_window("W0").sort_values("apply_time").reset_index(drop=True)
    split = int(len(w0) * 0.70)
    fit, healthy = w0.iloc[:split].copy(), w0.iloc[split:].copy()
    trained = _train_random_forest(
        fit,
        seed=SEED,
        hyperparameters=HYPERPARAMETERS,
        sample_weight=None,
        ordered_features=FEATURES,
    )
    bundle = (
        PROJECT_ROOT
        / "assets"
        / "champion_models"
        / MODEL_ID
        / CHAMPION_VERSION
    )
    checksum = _freeze_joblib(bundle / "model.joblib", trained["model"])
    _write_json(
        bundle / "training_manifest.json",
        {
            "model_id": MODEL_ID,
            "algorithm_family": "RandomForest",
            "feature_strategy_id": "SENSITIVE_FIXTURE_F01",
            "random_seed": SEED,
            "selected_parameters": HYPERPARAMETERS,
            "training_scope": "TEST_ONLY_W0_FIRST_70_PERCENT",
            "healthy_reference_scope": "TEST_ONLY_W0_LAST_30_PERCENT",
            "production_replacement_allowed": False,
        },
    )
    _write_json(
        bundle / "feature_schema.json",
        {
            "schema_version": "sensitive_fixture_feature_schema/1.0",
            "model_id": MODEL_ID,
            "model_version": CHAMPION_VERSION,
            "ordered_features": FEATURES,
            "fields": [
                {"name": FEATURES[0], "kind": "numeric", "nullable": True}
            ],
            "forbidden_model_inputs": ["sample_id", "apply_time", "is_bad"],
        },
    )
    return trained["model"], fit, healthy, checksum, bundle


def _envelope(champion_checksum: str) -> A7DecisionEnvelope:
    return A7DecisionEnvelope(
        decision_source="SIMULATED",
        lifecycle_run_id=LIFECYCLE_ID,
        event_id="event-a7-sensitive-fixture",
        monitoring_run_id="monitoring-a7-sensitive-fixture",
        diagnosis_run_id="diagnosis-a7-sensitive-fixture",
        agent_decision_id="agent-a7-sensitive-fixture",
        model_id=MODEL_ID,
        champion_version=CHAMPION_VERSION,
        champion_artifact_checksum=champion_checksum,
        model_task_type="BINARY_CLASSIFICATION",
        algorithm_family="RandomForest",
        recommended_action="MODEL_ITERATION",
        primary_root_cause={
            "root_cause_code": "FEATURE_DRIFT",
            "candidate_status": "CONFIRMED",
            "confidence": 0.92,
            "evidence_refs": ["scenario:covariate_drift:reg_to_apply_days"],
        },
        decay_degree="SUSTAINED_30D",
        trigger_context={
            "trigger_metric_codes": ["AUC", "KS"],
            "max_alert_severity": "WARNING",
            "data_track": "SCENARIO",
        },
        kg_strategy_candidates=[{"strategy_code": "sliding_window_retrain"}],
        authorization={
            "authorization_type": "AUTO_RULE",
            "authorization_id": "auto-rule-a7-sensitive-fixture",
            "approved": True,
        },
        rule_versions={
            "agent_rule_version": "agent-v1",
            "l1_matrix_version": "l1-v1",
            "window_rule_version": "W3_LAST_7D_HOLDOUT_V1",
            "qualification_rule_version": "qualification-rules-v2",
        },
    )


def _scenario_snapshot(plan) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    w2 = load_window("W2").copy()
    w3 = load_window("W3").copy()
    w3[FEATURES[0]] = w3[FEATURES[0]].astype(float)
    times = pd.to_datetime(w3["apply_time"], errors="raise")
    config = dict(SCENARIO)
    config.update(
        {
            "event_start_date": str(times.min().date()),
            "event_end_date": str((times.max() + pd.Timedelta(days=1)).date()),
        }
    )
    injected = ScenarioFactory.inject(w3, config, SEED)
    ranges = {
        item.window_id: item for item in plan.windows.training_time_ranges
    } | {
        item.window_id: item for item in plan.windows.validation_time_ranges
    }
    w2_range = ranges["W2"]
    w2_times = pd.to_datetime(w2["apply_time"], errors="raise")
    w2_train = w2.loc[
        (w2_times >= w2_range.start_at) & (w2_times < w2_range.end_at)
    ].copy()
    w2_train["__window_role_id"] = "W2"
    scenario_times = pd.to_datetime(injected.dataframe["apply_time"], errors="raise")
    w3_train_range = ranges["W3_TRAIN_SPLIT"]
    validation_range = ranges["W3_VALIDATION_SPLIT"]
    w3_train = injected.dataframe.loc[
        (scenario_times >= w3_train_range.start_at)
        & (scenario_times < w3_train_range.end_at)
    ].copy()
    validation = injected.dataframe.loc[
        (scenario_times >= validation_range.start_at)
        & (scenario_times < validation_range.end_at)
    ].copy()
    w3_train["__window_role_id"] = "W3_TRAIN_SPLIT"
    validation["__window_role_id"] = "W3_VALIDATION_SPLIT"
    train = pd.concat([w2_train, w3_train], ignore_index=True)
    overlap = set(train["sample_id"].astype(str)) & set(
        validation["sample_id"].astype(str)
    )
    if overlap:
        raise RuntimeError(f"SAMPLE_OVERLAP_DETECTED:{len(overlap)}")
    if train["is_bad"].isna().any() or validation["is_bad"].isna().any():
        raise RuntimeError("LABEL_MISSING")

    snapshot_dir = PROJECT_ROOT / "artifacts" / "a7_sensitive_fixture" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    train_path = snapshot_dir / "training_snapshot.parquet"
    validation_path = snapshot_dir / "validation_snapshot.parquet"
    train.to_parquet(train_path, index=False)
    validation.to_parquet(validation_path, index=False)
    snapshot_identity = {
        "training_snapshot_id": "fixture:a7-sensitive:training:v1",
        "training_snapshot_checksum": _sha256_bytes(train_path.read_bytes()),
        "validation_snapshot_id": "fixture:a7-sensitive:validation:v1",
        "validation_snapshot_checksum": _sha256_bytes(validation_path.read_bytes()),
        "training_snapshot_path": str(train_path),
        "validation_snapshot_path": str(validation_path),
    }
    metadata = dict(injected.metadata)
    metadata.update(snapshot_identity)
    _write_json(snapshot_dir / "scenario_metadata.json", metadata)
    return train, validation, metadata, snapshot_identity


def run() -> tuple[dict, Path]:
    champion_model, _, healthy, champion_checksum, champion_bundle = _build_champion()
    envelope = _envelope(champion_checksum)
    proposal, l1 = RepairDecisionService().propose_a7(envelope)
    risk = RiskAssessmentService().assess(proposal)
    plan = TrainingPlanBuilder().build(
        proposal,
        risk,
        approval_id=envelope.authorization.authorization_id,
        iteration_run_id="iteration-a7-sensitive-fixture",
    )
    train, validation, scenario_metadata, snapshot_identity = _scenario_snapshot(plan)

    trained = _train_random_forest(
        train,
        seed=plan.random_seed,
        hyperparameters=plan.hyperparameter_space,
        sample_weight=None,
        ordered_features=plan.ordered_features,
    )
    challenger_model = trained["model"]
    healthy_scores = champion_model.predict_proba(
        _prepare_features(healthy, FEATURES)
    )[:, 1]
    degraded_scores = champion_model.predict_proba(
        _prepare_features(validation, FEATURES)
    )[:, 1]
    challenger_scores = challenger_model.predict_proba(
        _prepare_features(validation, FEATURES)
    )[:, 1]
    healthy_y = healthy["is_bad"]
    validation_y = validation["is_bad"]

    healthy_auc = roc_auc_score(healthy_y, healthy_scores)
    degraded_auc = roc_auc_score(validation_y, degraded_scores)
    challenger_auc = roc_auc_score(validation_y, challenger_scores)
    healthy_ks = _compute_ks(healthy_y, healthy_scores)
    degraded_ks = _compute_ks(validation_y, degraded_scores)
    challenger_ks = _compute_ks(validation_y, challenger_scores)
    auc_drop = healthy_auc - degraded_auc
    ks_drop = healthy_ks - degraded_ks
    auc_gain = challenger_auc - degraded_auc
    ks_gain = challenger_ks - degraded_ks
    auc_recovery = auc_gain / auc_drop if auc_drop > 0 else 0.0
    ks_recovery = ks_gain / ks_drop if ks_drop > 0 else 0.0
    auc_alert, auc_severity = DEFAULT_THRESHOLD_RULES["AUC"].evaluate(
        degraded_auc - healthy_auc, degraded_auc
    )
    ks_alert, ks_severity = DEFAULT_THRESHOLD_RULES["KS"].evaluate(
        degraded_ks - healthy_ks, degraded_ks
    )
    feature_psi = _compute_psi_frozen(
        healthy[FEATURES[0]].tolist(), validation[FEATURES[0]].tolist()
    )
    auc_ci = _paired_bootstrap_delta_ci(
        validation_y, degraded_scores, challenger_scores, "AUC", seed=SEED
    )
    ks_ci = _paired_bootstrap_delta_ci(
        validation_y, degraded_scores, challenger_scores, "KS", seed=SEED
    )
    challenger_checksum = _freeze_joblib(
        PROJECT_ROOT
        / "artifacts"
        / "a7_sensitive_fixture"
        / "challenger"
        / "model.joblib",
        challenger_model,
    )
    degraded_brier = brier_score_loss(validation_y, degraded_scores)
    challenger_brier = brier_score_loss(validation_y, challenger_scores)
    degraded_recall = _bad_recall_at_top20(validation_y, degraded_scores)
    challenger_recall = _bad_recall_at_top20(validation_y, challenger_scores)
    score_psi = _calc_score_psi(degraded_scores, challenger_scores)
    train_valid_gap = abs(trained["train_auc"] - challenger_auc)

    targets = [
        MetricComparison(
            metric_code="AUC",
            direction="HIGHER_BETTER",
            original_drop=auc_drop,
            recovered_amount=auc_gain,
            recovery_rate=auc_recovery,
            champion_value=degraded_auc,
            challenger_value=challenger_auc,
            healthy_lower_bound=healthy_auc - 0.02,
            bootstrap_ci_lower=auc_ci[0] if auc_ci else None,
            bootstrap_ci_upper=auc_ci[1] if auc_ci else None,
        ),
        MetricComparison(
            metric_code="KS",
            direction="HIGHER_BETTER",
            original_drop=ks_drop,
            recovered_amount=ks_gain,
            recovery_rate=ks_recovery,
            champion_value=degraded_ks,
            challenger_value=challenger_ks,
            healthy_lower_bound=healthy_ks - 0.02,
            bootstrap_ci_lower=ks_ci[0] if ks_ci else None,
            bootstrap_ci_upper=ks_ci[1] if ks_ci else None,
        ),
    ]
    qualification = QualificationService().evaluate(
        QualificationInput(
            qualification_run_id="pre-oot-a7-sensitive-fixture",
            iteration_run_id=plan.iteration_run_id,
            experiment_id=plan.experiment_id,
            candidate_version=CANDIDATE_VERSION,
            qualification_stage="PRE_OOT",
            target_metrics=targets,
            data_reproducible=True,
            discrimination_passed=challenger_auc >= degraded_auc - 0.01,
            calibration_passed=challenger_brier <= degraded_brier + 0.01,
            score_psi=score_psi,
            train_valid_gap=train_valid_gap,
            segment_governance_passed=True,
            segment_governance_required=False,
            bad_recall_passed=challenger_recall >= degraded_recall,
            candidate_frozen_before_oot=True,
            oot_usage="PRE_OOT",
            w4_read_count=0,
            frozen_identity_checksum=challenger_checksum,
        )
    )
    receipt = TrainingConsumptionReceipt(
        consumed_training_snapshot_ids=[snapshot_identity["training_snapshot_id"]],
        consumed_validation_snapshot_ids=[snapshot_identity["validation_snapshot_id"]],
        observed_train_sample_count=len(train),
        observed_validation_sample_count=len(validation),
        observed_train_bad_count=int(train["is_bad"].sum()),
        observed_validation_bad_count=int(validation["is_bad"].sum()),
        sample_overlap_count=0,
        actual_algorithm_family="RandomForest",
        actual_execution_mode=plan.execution_mode,
        actual_base_model_checksum=None,
        sample_weight_consumed=False,
        non_unit_weight_sample_count=0,
        actual_ordered_features_hash=_stable_hash(FEATURES),
        actual_preprocessing_hash=plan.preprocessing_hash,
        actual_hyperparameters_hash=_stable_hash(plan.hyperparameter_space),
        w4_read_count=0,
    )

    plan_snapshot_ids = set(plan.data_snapshot_ids)
    consumed_snapshot_ids = {
        snapshot_identity["training_snapshot_id"],
        snapshot_identity["validation_snapshot_id"],
    }
    snapshot_contract_match = plan_snapshot_ids == consumed_snapshot_ids
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "TEST_ONLY_SENSITIVE_MODEL_FIXTURE",
        "production_model_007_modified": False,
        "production_windows_modified": False,
        "champion_bundle": str(champion_bundle),
        "scenario": scenario_metadata,
        "monitoring": {
            "feature_psi": feature_psi,
            "auc": {
                "healthy": healthy_auc,
                "degraded": degraded_auc,
                "drop": auc_drop,
                "triggered": auc_alert,
                "severity": auc_severity.value if auc_severity else None,
            },
            "ks": {
                "healthy": healthy_ks,
                "degraded": degraded_ks,
                "drop": ks_drop,
                "triggered": ks_alert,
                "severity": ks_severity.value if ks_severity else None,
            },
        },
        "l1": l1.model_dump(mode="json"),
        "risk": risk.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "consumption_receipt": receipt.model_dump(mode="json"),
        "repair": {
            "champion_auc": degraded_auc,
            "challenger_auc": challenger_auc,
            "auc_gain": auc_gain,
            "auc_recovery_rate": auc_recovery,
            "auc_bootstrap_ci": auc_ci,
            "champion_ks": degraded_ks,
            "challenger_ks": challenger_ks,
            "ks_gain": ks_gain,
            "ks_recovery_rate": ks_recovery,
            "ks_bootstrap_ci": ks_ci,
            "score_psi": score_psi,
            "train_valid_gap": train_valid_gap,
            "bad_recall_champion": degraded_recall,
            "bad_recall_challenger": challenger_recall,
            "brier_champion": degraded_brier,
            "brier_challenger": challenger_brier,
            "challenger_checksum": challenger_checksum,
        },
        "pre_oot": qualification.model_dump(mode="json"),
        "w4_accessed": False,
        "strict_flow_audit": {
            "scenario_stress_ceiling": 0.50,
            "scenario_stress_valid": bool(feature_psi is not None and feature_psi < 0.50),
            "plan_snapshot_contract_match": snapshot_contract_match,
            "plan_snapshot_ids": sorted(plan_snapshot_ids),
            "consumed_snapshot_ids": sorted(consumed_snapshot_ids),
            "blocking_issue": (
                None
                if snapshot_contract_match
                else "SCENARIO_SNAPSHOT_NOT_ACCEPTED_BY_TRAINING_PLAN_BUILDER"
            ),
            "effect_proven": bool(
                auc_alert
                and ks_alert
                and auc_recovery >= 0.90
                and ks_recovery >= 0.90
                and auc_ci
                and auc_ci[0] > 0
                and ks_ci
                and ks_ci[0] > 0
            ),
            "full_production_flow_proven": bool(
                qualification.qualified and snapshot_contract_match
            ),
        },
        "claims": {
            "sensitive_fixture_trained": True,
            "controlled_repair_effect_proven": bool(
                auc_alert and ks_alert and auc_recovery >= 0.90 and ks_recovery >= 0.90
            ),
            "production_model_ready": False,
            "all_50_models_complete": False,
        },
    }
    report_path = (
        PROJECT_ROOT / "artifacts" / "a7_sensitive_fixture" / "latest_report.json"
    )
    _write_json(report_path, report)
    return report, report_path


if __name__ == "__main__":
    result, output = run()
    print(
        json.dumps(
            {
                "report": str(output),
                "auc_drop": result["monitoring"]["auc"]["drop"],
                "ks_drop": result["monitoring"]["ks"]["drop"],
                "auc_recovery_rate": result["repair"]["auc_recovery_rate"],
                "ks_recovery_rate": result["repair"]["ks_recovery_rate"],
                "pre_oot_status": result["pre_oot"]["status"],
                "strict_flow_audit": result["strict_flow_audit"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
