"""Prove the A7 repair path on a healthy production champion.

This is an isolated fixture.  It reuses the existing scenario injector without
changing it, simulates an already-confirmed upstream diagnosis, exercises the
real A7 decision/plan services and the matching training adapter, and never
replaces the production champion or reads W4.
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
    TRAINERS,
    _bad_recall_at_top20,
    _calc_score_psi,
    _compute_ks,
    _paired_bootstrap_delta_ci,
    _prepare_features,
    _stable_hash,
)


MODEL_ID = "credit_model_012"
CHAMPION_VERSION = "champion_v1"
MODEL_SUFFIX = "012"
CANDIDATE_VERSION = f"{MODEL_ID}_a7_repair_fixture_v1"
LIFECYCLE_ID = f"a7-repair-model-{MODEL_SUFFIX}-fixture"
SEED = 2026007
SCENARIO = {
    "scenario_name": "covariate_drift",
    "intensity": 0.40,
    "affected_features": ["login_fail_count"],
    "base_window_id": "W3",
}
SCENARIO_STRESS_CEILING = 0.50


def _configure_model(model_id: str) -> None:
    """Configure one existing Champion without weakening its identity checks."""
    global MODEL_ID, MODEL_SUFFIX, CANDIDATE_VERSION, LIFECYCLE_ID, SEED
    MODEL_ID = model_id
    MODEL_SUFFIX = model_id.removeprefix("credit_model_")
    CANDIDATE_VERSION = f"{MODEL_ID}_a7_repair_fixture_v1"
    LIFECYCLE_ID = f"a7-repair-model-{MODEL_SUFFIX}-fixture"
    manifest_path = (
        PROJECT_ROOT / "assets" / "champion_models" / MODEL_ID
        / CHAMPION_VERSION / "training_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    SEED = int(manifest["random_seed"])
    # A deeper boosted tree is materially more robust to one shifted feature
    # than the RandomForest fixture. Use a deterministic joint drift while
    # retaining the same per-feature stress ceiling and untouched labels.
    SCENARIO["affected_features"] = (
        ["login_fail_count", "max_overdue_days", "judicial_risk_score"]
        if manifest["algorithm_family"] == "LightGBM"
        else ["login_fail_count"]
    )
    SCENARIO["intensity"] = 0.30 if manifest["algorithm_family"] == "LightGBM" else 0.40


def _champion_manifest() -> dict:
    return json.loads(
        (_champion_bundle() / "training_manifest.json").read_text(encoding="utf-8")
    )


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _champion_bundle() -> Path:
    return (
        PROJECT_ROOT
        / "assets"
        / "champion_models"
        / MODEL_ID
        / CHAMPION_VERSION
    )


def _champion_checksum() -> str:
    return _sha256_bytes((_champion_bundle() / "model.joblib").read_bytes())


def _envelope() -> A7DecisionEnvelope:
    return A7DecisionEnvelope(
        decision_source="SIMULATED",
        lifecycle_run_id=LIFECYCLE_ID,
        event_id=f"event-a7-repair-model-{MODEL_SUFFIX}",
        monitoring_run_id=f"monitoring-a7-repair-model-{MODEL_SUFFIX}",
        diagnosis_run_id=f"diagnosis-a7-repair-model-{MODEL_SUFFIX}",
        agent_decision_id=f"agent-a7-repair-model-{MODEL_SUFFIX}",
        model_id=MODEL_ID,
        champion_version=CHAMPION_VERSION,
        champion_artifact_checksum=_champion_checksum(),
        model_task_type="BINARY_CLASSIFICATION",
        algorithm_family=_champion_manifest()["algorithm_family"],
        recommended_action="MODEL_ITERATION",
        primary_root_cause={
            "root_cause_code": "FEATURE_DRIFT",
            "candidate_status": "CONFIRMED",
            "confidence": 0.92,
            "evidence_refs": [
                "scenario:covariate_drift:login_fail_count",
                "metric:MAX_FEATURE_PSI_30D:CRITICAL",
                "metric:KS:WARNING",
            ],
        },
        decay_degree="SUSTAINED_30D",
        trigger_context={
            "trigger_metric_codes": ["MAX_FEATURE_PSI_30D", "KS"],
            "max_alert_severity": "CRITICAL",
            "data_track": "SCENARIO",
        },
        kg_strategy_candidates=[{"strategy_code": "sliding_window_retrain"}],
        authorization={
            "authorization_type": "AUTO_RULE",
            "authorization_id": f"auto-a7-repair-model-{MODEL_SUFFIX}",
            "approved": True,
        },
        rule_versions={
            "agent_rule_version": "agent-v1",
            "l1_matrix_version": "l1-v1",
            "window_rule_version": "W3_LAST_7D_HOLDOUT_V1",
            "qualification_rule_version": "qualification-rules-v2",
        },
    )


def _freeze_challenger(model: object) -> tuple[str, Path]:
    path = (
        PROJECT_ROOT
        / "artifacts"
        / f"a7_repair_model_{MODEL_SUFFIX}"
        / "challenger"
        / "model.joblib"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    payload = buffer.getvalue()
    path.write_bytes(payload)
    return _sha256_bytes(payload), path


def _scenario_data(plan):
    w1 = load_window("W1").copy()
    w2 = load_window("W2").copy()
    w3 = load_window("W3").copy()
    times = pd.to_datetime(w3["apply_time"], errors="raise")
    config = {
        **SCENARIO,
        "event_start_date": str(times.min().date()),
        "event_end_date": str((times.max() + pd.Timedelta(days=1)).date()),
    }
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

    injected_times = pd.to_datetime(injected.dataframe["apply_time"], errors="raise")
    train_range = ranges["W3_TRAIN_SPLIT"]
    valid_range = ranges["W3_VALIDATION_SPLIT"]
    w3_train = injected.dataframe.loc[
        (injected_times >= train_range.start_at)
        & (injected_times < train_range.end_at)
    ].copy()
    validation = injected.dataframe.loc[
        (injected_times >= valid_range.start_at)
        & (injected_times < valid_range.end_at)
    ].copy()
    control_validation = w3.loc[
        (times >= valid_range.start_at) & (times < valid_range.end_at)
    ].copy()
    w3_train["__window_role_id"] = "W3_TRAIN_SPLIT"
    validation["__window_role_id"] = "W3_VALIDATION_SPLIT"
    train = pd.concat([w2_train, w3_train], ignore_index=True)

    overlap = set(train["sample_id"].astype(str)) & set(
        validation["sample_id"].astype(str)
    )
    if overlap:
        raise RuntimeError(f"SAMPLE_OVERLAP_DETECTED:{len(overlap)}")
    if any(frame["is_bad"].isna().any() for frame in (train, validation)):
        raise RuntimeError("LABEL_MISSING")

    output = PROJECT_ROOT / "artifacts" / f"a7_repair_model_{MODEL_SUFFIX}" / "snapshots"
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "training_snapshot.parquet"
    valid_path = output / "validation_snapshot.parquet"
    train.to_parquet(train_path, index=False)
    validation.to_parquet(valid_path, index=False)
    identities = {
        "training_snapshot_id": f"scenario:a7-repair-model-{MODEL_SUFFIX}:training:v1",
        "training_snapshot_checksum": _sha256_bytes(train_path.read_bytes()),
        "validation_snapshot_id": f"scenario:a7-repair-model-{MODEL_SUFFIX}:validation:v1",
        "validation_snapshot_checksum": _sha256_bytes(valid_path.read_bytes()),
    }
    metadata = {**injected.metadata, **identities}
    _write_json(output / "scenario_metadata.json", metadata)
    return w1, train, validation, control_validation, metadata, identities


def run(model_id: str = MODEL_ID) -> tuple[dict, Path]:
    _configure_model(model_id)
    envelope = _envelope()
    proposal, l1 = RepairDecisionService().propose_a7(envelope)
    risk = RiskAssessmentService().assess(proposal)
    plan = TrainingPlanBuilder().build(
        proposal,
        risk,
        approval_id=envelope.authorization.authorization_id,
        iteration_run_id=f"iteration-a7-repair-model-{MODEL_SUFFIX}",
    )
    if plan.strategy_code != "sliding_window_retrain":
        raise RuntimeError(f"UNEXPECTED_STRATEGY:{plan.strategy_code}")

    w1, train, validation, control, scenario, identities = _scenario_data(plan)
    champion = joblib.load(_champion_bundle() / "model.joblib")
    features = plan.ordered_features
    trained = TRAINERS[plan.algorithm](
        train,
        seed=plan.random_seed,
        hyperparameters=plan.hyperparameter_space,
        sample_weight=None,
        ordered_features=features,
    )
    challenger = trained["model"]

    control_scores = champion.predict_proba(_prepare_features(control, features))[:, 1]
    healthy_scores = champion.predict_proba(_prepare_features(w1, features))[:, 1]
    degraded_scores = champion.predict_proba(
        _prepare_features(validation, features)
    )[:, 1]
    challenger_scores = challenger.predict_proba(
        _prepare_features(validation, features)
    )[:, 1]
    challenger_w1_scores = challenger.predict_proba(
        _prepare_features(w1, features)
    )[:, 1]
    y = validation["is_bad"]

    healthy_auc = float(roc_auc_score(w1["is_bad"], healthy_scores))
    healthy_ks = _compute_ks(w1["is_bad"], healthy_scores)
    control_auc = float(roc_auc_score(control["is_bad"], control_scores))
    degraded_auc = float(roc_auc_score(y, degraded_scores))
    challenger_auc = float(roc_auc_score(y, challenger_scores))
    control_ks = _compute_ks(control["is_bad"], control_scores)
    degraded_ks = _compute_ks(y, degraded_scores)
    challenger_ks = _compute_ks(y, challenger_scores)
    # A7 recovery is measured against the frozen W1 monitoring baseline.
    # The paired W3 clean control is retained separately as causal evidence.
    auc_drop = healthy_auc - degraded_auc
    ks_drop = healthy_ks - degraded_ks
    paired_control_auc_drop = control_auc - degraded_auc
    paired_control_ks_drop = control_ks - degraded_ks
    auc_gain = challenger_auc - degraded_auc
    ks_gain = challenger_ks - degraded_ks
    auc_recovery = auc_gain / auc_drop if auc_drop > 0 else 0.0
    ks_recovery = ks_gain / ks_drop if ks_drop > 0 else 0.0
    auc_ci = _paired_bootstrap_delta_ci(
        y, degraded_scores, challenger_scores, "AUC", seed=SEED, rounds=500
    )
    ks_ci = _paired_bootstrap_delta_ci(
        y, degraded_scores, challenger_scores, "KS", seed=SEED, rounds=500
    )

    feature_psi_by_feature = {
        feature: _compute_psi_frozen(
            w1[feature].tolist(), validation[feature].tolist()
        )
        for feature in SCENARIO["affected_features"]
    }
    feature_psi = max(feature_psi_by_feature.values())
    score_psi = _calc_score_psi(challenger_w1_scores, challenger_scores)
    train_valid_gap = abs(float(trained["train_auc"]) - challenger_auc)
    degraded_recall = _bad_recall_at_top20(y, degraded_scores)
    challenger_recall = _bad_recall_at_top20(y, challenger_scores)
    degraded_brier = float(brier_score_loss(y, degraded_scores))
    challenger_brier = float(brier_score_loss(y, challenger_scores))
    challenger_checksum, challenger_path = _freeze_challenger(challenger)

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
            qualification_run_id=f"pre-oot-a7-repair-model-{MODEL_SUFFIX}",
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
        consumed_training_snapshot_ids=[identities["training_snapshot_id"]],
        consumed_validation_snapshot_ids=[identities["validation_snapshot_id"]],
        observed_train_sample_count=len(train),
        observed_validation_sample_count=len(validation),
        observed_train_bad_count=int(train["is_bad"].sum()),
        observed_validation_bad_count=int(validation["is_bad"].sum()),
        sample_overlap_count=0,
        actual_algorithm_family=plan.algorithm_family,
        actual_execution_mode=plan.execution_mode,
        actual_base_model_checksum=None,
        sample_weight_consumed=False,
        non_unit_weight_sample_count=0,
        actual_ordered_features_hash=_stable_hash(features),
        actual_preprocessing_hash=plan.preprocessing_hash,
        actual_hyperparameters_hash=_stable_hash(plan.hyperparameter_space),
        w4_read_count=0,
    )

    auc_alert, auc_severity = DEFAULT_THRESHOLD_RULES["AUC"].evaluate(
        degraded_auc - healthy_auc, degraded_auc
    )
    ks_alert, ks_severity = DEFAULT_THRESHOLD_RULES["KS"].evaluate(
        degraded_ks - healthy_ks, degraded_ks
    )
    psi_alert, psi_severity = DEFAULT_THRESHOLD_RULES["FEATURE_PSI"].evaluate(
        feature_psi, feature_psi
    )
    snapshot_contract_match = set(plan.data_snapshot_ids) == {
        identities["training_snapshot_id"],
        identities["validation_snapshot_id"],
    }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": f"TEST_ONLY_A7_REPAIR_{MODEL_ID.upper()}",
        "production_champion_modified": False,
        "production_windows_modified": False,
        "w4_accessed": False,
        "scenario_injector_modified": False,
        "scenario": scenario,
        "diagnosis_input": envelope.model_dump(mode="json"),
        "l1": l1.model_dump(mode="json"),
        "risk": risk.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "consumption_receipt": receipt.model_dump(mode="json"),
        "monitoring": {
            "feature_psi": {
                "value": feature_psi,
                "by_feature": feature_psi_by_feature,
                "triggered": psi_alert,
                "severity": psi_severity.value if psi_severity else None,
            },
            "auc": {
                "healthy_w1": healthy_auc,
                "paired_w3_control": control_auc,
                "degraded": degraded_auc,
                "drop": auc_drop,
                "paired_causal_effect_drop": paired_control_auc_drop,
                "triggered": auc_alert,
                "severity": auc_severity.value if auc_severity else None,
            },
            "ks": {
                "healthy_w1": healthy_ks,
                "paired_w3_control": control_ks,
                "degraded": degraded_ks,
                "drop": ks_drop,
                "paired_causal_effect_drop": paired_control_ks_drop,
                "triggered": ks_alert,
                "severity": ks_severity.value if ks_severity else None,
            },
        },
        "repair": {
            "challenger_path": str(challenger_path),
            "challenger_checksum": challenger_checksum,
            "challenger_auc": challenger_auc,
            "challenger_ks": challenger_ks,
            "auc_gain": auc_gain,
            "ks_gain": ks_gain,
            "auc_recovery_rate": auc_recovery,
            "ks_recovery_rate": ks_recovery,
            "auc_bootstrap_ci": auc_ci,
            "ks_bootstrap_ci": ks_ci,
            "score_psi": score_psi,
            "train_valid_gap": train_valid_gap,
            "degraded_bad_recall": degraded_recall,
            "challenger_bad_recall": challenger_recall,
            "degraded_brier": degraded_brier,
            "challenger_brier": challenger_brier,
        },
        "pre_oot": qualification.model_dump(mode="json"),
        "strict_flow_audit": {
            "scenario_stress_ceiling": SCENARIO_STRESS_CEILING,
            "scenario_stress_valid": feature_psi < SCENARIO_STRESS_CEILING,
            "plan_snapshot_contract_match": snapshot_contract_match,
            "blocking_issue": (
                None
                if snapshot_contract_match
                else "SCENARIO_SNAPSHOT_NOT_ACCEPTED_BY_TRAINING_PLAN_BUILDER"
            ),
            "repair_effect_proven": qualification.qualified,
            "full_production_flow_proven": False,
        },
        "claims": {
            "healthy_production_model_used": True,
            "a7_contract_and_l1_exercised": True,
            "real_training_adapter_exercised": True,
            "controlled_repair_effect_proven": qualification.qualified,
            "production_ready": False,
            "all_50_models_complete": False,
        },
    }
    output = PROJECT_ROOT / "artifacts" / f"a7_repair_model_{MODEL_SUFFIX}" / "latest_report.json"
    _write_json(output, report)
    return report, output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="credit_model_012")
    args = parser.parse_args()
    result, report_path = run(args.model_id)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "model_id": MODEL_ID,
                "strategy": result["plan"]["strategy_code"],
                "pre_oot_status": result["pre_oot"]["status"],
                "repair_effect_proven": result["claims"][
                    "controlled_repair_effect_proven"
                ],
                "production_ready": result["claims"]["production_ready"],
                "w4_accessed": result["w4_accessed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
