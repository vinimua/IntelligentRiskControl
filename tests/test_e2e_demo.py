"""End-to-end demo smoke test for the ModelOps lifecycle workflow.

Usage:
    python tests/test_e2e_demo.py

This script expects the backend API to be running on http://127.0.0.1:8000.
It drives the same path that the frontend demo uses:
start lifecycle -> manual review -> training callback -> qualification -> deploy.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

API = "http://127.0.0.1:8000"
passed = 0
failed = 0


def call(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    global failed
    url = f"{API}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"message": raw}
        print(f"  FAIL HTTP {exc.code}: {payload.get('message', payload)}")
        failed += 1
        return {}
    except Exception as exc:
        print(f"  FAIL Error: {exc}")
        failed += 1
        return {}

    if payload.get("success", True) is False:
        print(f"  FAIL API error: {payload.get('code')} - {payload.get('message')}")
        failed += 1
        return {}
    return payload.get("data", payload)


def assert_ok(label: str, condition: bool, detail: Any = "") -> None:
    global passed, failed
    if condition:
        print(f"  PASS {label}: {detail}")
        passed += 1
    else:
        print(f"  FAIL {label}: {detail}")
        failed += 1


def state_of(result: dict[str, Any]) -> dict[str, Any]:
    state = result.get("state")
    if isinstance(state, dict):
        return state
    return result


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def submit_manual_review(run_id: str, proposal_id: str) -> dict[str, Any]:
    review = call(
        "POST",
        f"/api/iteration/decisions/{proposal_id}/reviews",
        {
            "proposal_id": proposal_id,
            "reviewer_id": "e2e-demo",
            "decision": "APPROVE",
            "reason": "Approve demo MODEL_ITERATION path",
            "rejection_reason_codes": [],
            "adjustment_instructions": [],
            "forbidden_adjustments": [],
            "expected_evidence": [],
            "reviewed_at": iso_now(),
        },
    )
    review_id = review.get("review_id")
    assert_ok("manual review id", bool(review_id), review_id)

    resumed = call(
        "POST",
        f"/api/lifecycle-runs/{run_id}/resume",
        {
            "decision": "approved",
            "manual_review_id": review_id,
            "review_id": review_id,
        },
    )
    phase = resumed.get("current_phase")
    assert_ok("manual review resume accepted", bool(phase), phase)
    return resumed


def submit_training_callback(run_id: str, state: dict[str, Any]) -> dict[str, Any]:
    training_job_id = state.get("training_job_id")
    iteration_run_id = state.get("iteration_run_id")
    experiment_id = state.get("experiment_id")
    business_round = state.get("business_round") or 1
    assert_ok("training job id", bool(training_job_id), training_job_id)
    assert_ok("iteration run id", bool(iteration_run_id), iteration_run_id)
    assert_ok("experiment id", bool(experiment_id), experiment_id)
    if not training_job_id or not iteration_run_id or not experiment_id:
        return {}

    callback = call(
        "POST",
        f"/api/internal/iteration/jobs/{training_job_id}/callback",
        {
            "training_job_id": training_job_id,
            "lifecycle_run_id": run_id,
            "idempotency_key": (
                f"{iteration_run_id}:round-{business_round}:exp-{experiment_id}"
            ),
            "experiment_id": experiment_id,
            "status": "SUCCEEDED",
            "candidate_version": "v1_challenger_e2e",
            "model_artifact_uri": "s3://riskitem/demo/models/v1_challenger_e2e",
            "training_metrics": {"auc": 0.81, "ks": 0.43},
            "validation_metrics": {
                "original_drop": 0.04,
                "recovered_amount": 0.035,
                "recovery_rate": 0.875,
                "champion_auc": 0.74,
                "challenger_auc": 0.775,
                "healthy_lower_bound": 0.76,
                "bootstrap_ci_lower": 0.01,
                "bootstrap_ci_upper": 0.06,
                "discrimination_passed": True,
                "calibration_passed": True,
                "score_psi": 0.08,
                "train_valid_gap": 0.015,
                "oot_passed": True,
            },
            "segment_metrics": {"segment_governance_passed": True},
            "artifact_checksums": {},
            "environment_manifest": {"runtime": "e2e-demo"},
            "technical_retry_count": 0,
        },
    )
    assert_ok("callback applied", callback.get("callback_applied") is True, callback)
    assert_ok("callback auto resumed", callback.get("lifecycle_resumed") is True, callback)
    return callback


def run() -> None:
    print("=" * 60)
    print("P0 end-to-end demo test - RiskItem ModelOps")
    print(f"API: {API}")
    print("=" * 60)

    print("\n[1/9] Health check")
    health = call("GET", "/health/live")
    assert_ok("API alive", health.get("status") == "ok", health.get("status"))

    print("\n[2/9] Start lifecycle")
    start = call(
        "POST",
        "/api/lifecycle-runs",
        {
            "model_id": "credit_model_001",
            "champion_version": "champion_v1",
            "trigger_type": "SCHEDULED_TRIGGER",
        },
    )
    run_id = start.get("lifecycle_run_id")
    start_state = state_of(start)
    assert_ok("lifecycle run id", bool(run_id), run_id)
    assert_ok("initial interrupt", "__interrupt__" in start_state, start_state.get("current_phase"))
    if not run_id:
        return

    print("\n[3/9] Query lifecycle")
    queried = call("GET", f"/api/lifecycle-runs/{run_id}")
    current = state_of(queried)
    assert_ok("phase is decision proposed", current.get("current_phase") == "DECISION_PROPOSED", current.get("current_phase"))
    proposal_id = current.get("decision_proposal_id")
    assert_ok("decision proposal id", bool(proposal_id), proposal_id)
    if not proposal_id:
        return

    print("\n[4/9] Manual review approve")
    after_review = submit_manual_review(run_id, proposal_id)
    review_state = state_of(after_review)

    print("\n[5/9] Reached training callback interrupt")
    phase_after_review = review_state.get("current_phase")
    assert_ok(
        "waiting training callback",
        phase_after_review == "WAITING_TRAINING_CALLBACK",
        phase_after_review,
    )

    print("\n[6/9] Training callback")
    submit_training_callback(run_id, review_state)

    print("\n[7/9] Idempotency check")
    dup = call("POST", f"/api/lifecycle-runs/{run_id}/resume", {"decision": "approved"})
    assert_ok("terminal resume is safe", bool(dup.get("current_phase")), dup.get("current_phase"))

    print("\n[8/9] Final state verification")
    final = call("GET", f"/api/lifecycle-runs/{run_id}")
    final_state = state_of(final)
    checks = [
        ("phase", final_state.get("current_phase") == "EVENT_CLOSED", final_state.get("current_phase")),
        ("action", final_state.get("recommended_action") == "MODEL_ITERATION", final_state.get("recommended_action")),
        ("monitoring_run_id", bool(final_state.get("monitoring_run_id")), final_state.get("monitoring_run_id")),
        ("diagnosis_run_id", bool(final_state.get("diagnosis_run_id")), final_state.get("diagnosis_run_id")),
        ("agent_decision_id", bool(final_state.get("agent_decision_id")), final_state.get("agent_decision_id")),
        ("manual_review_id", bool(final_state.get("manual_review_id")), final_state.get("manual_review_id")),
        ("training_job_id", bool(final_state.get("training_job_id")), final_state.get("training_job_id")),
        ("qualification_run_id", bool(final_state.get("qualification_run_id")), final_state.get("qualification_run_id")),
        ("deployment_id", bool(final_state.get("deployment_id")), final_state.get("deployment_id")),
        ("challenger qualified", final_state.get("challenger_qualified") is True, final_state.get("challenger_qualified")),
    ]
    for label, condition, detail in checks:
        assert_ok(label, condition, detail)

    print("\n[9/9] Cleanup")
    assert_ok("cleanup not needed", True, "terminal run")

    print("\n" + "=" * 60)
    print(f"Result: {passed} passed, {failed} failed")
    print("PASS all checks" if failed == 0 else f"FAIL {failed} checks failed")
    print("=" * 60)


if __name__ == "__main__":
    run()
    sys.exit(0 if failed == 0 else 1)
