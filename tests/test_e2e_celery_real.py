"""Real Celery worker end-to-end smoke test.

Prerequisites:
    1. Redis, PostgreSQL, MinIO and MLflow are running.
    2. Backend API is running with WORKFLOW_USE_CELERY=true.
    3. Celery worker is running:
       celery -A workers.app worker --loglevel=info --pool=solo

Usage:
    python tests/test_e2e_celery_real.py

Unlike tests/test_e2e_demo.py, this script does not submit a manual training
callback. It approves the manual review and waits for the real Worker to train
on W2/W3, call back, qualify and deploy.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

API = os.getenv("MODELOPS_API_BASE", "http://127.0.0.1:8000")
TIMEOUT_SECONDS = 240
POLL_SECONDS = 5
passed = 0
failed = 0


def call(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    global failed
    url = f"{API}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
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
    return state if isinstance(state, dict) else result


def submit_manual_review(run_id: str, proposal_id: str) -> dict[str, Any]:
    review = call(
        "POST",
        f"/api/iteration/decisions/{proposal_id}/reviews",
        {
            "proposal_id": proposal_id,
            "reviewer_id": "celery-e2e",
            "decision": "APPROVE",
            "reason": "Approve real Celery worker training path",
            "rejection_reason_codes": [],
            "adjustment_instructions": [],
            "forbidden_adjustments": [],
            "expected_evidence": [],
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
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
    assert_ok("manual review resumed", bool(resumed.get("current_phase")), resumed.get("current_phase"))
    return resumed


def wait_for_terminal(run_id: str) -> dict[str, Any]:
    deadline = time.time() + TIMEOUT_SECONDS
    last_state: dict[str, Any] = {}
    while time.time() < deadline:
        result = call("GET", f"/api/lifecycle-runs/{run_id}")
        last_state = state_of(result)
        phase = last_state.get("current_phase")
        mode = last_state.get("training_dispatch_mode")
        callback_status = last_state.get("training_callback_status")
        print(f"  poll phase={phase} mode={mode} callback={callback_status}")
        if phase in {"EVENT_CLOSED", "ROLLED_BACK", "FAILED", "NO_ALERT", "COMPLETED"}:
            return last_state
        time.sleep(POLL_SECONDS)
    return last_state


def run() -> None:
    print("=" * 68)
    print("Real Celery E2E test - RiskItem ModelOps")
    print(f"API: {API}")
    print("=" * 68)

    print("\n[1/7] Health check")
    health = call("GET", "/health/live")
    assert_ok("API alive", health.get("status") == "ok", health.get("status"))

    print("\n[2/7] Start lifecycle")
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
    assert_ok("lifecycle run id", bool(run_id), run_id)
    if not run_id:
        return

    print("\n[3/7] Query decision proposal")
    queried = call("GET", f"/api/lifecycle-runs/{run_id}")
    current = state_of(queried)
    proposal_id = current.get("decision_proposal_id")
    assert_ok("decision proposal id", bool(proposal_id), proposal_id)
    if not proposal_id:
        return

    print("\n[4/7] Manual review approve")
    after_review = state_of(submit_manual_review(run_id, proposal_id))
    assert_ok(
        "celery dispatch mode",
        after_review.get("training_dispatch_mode") == "celery",
        after_review.get("training_dispatch_mode"),
    )
    assert_ok("training dispatched", after_review.get("training_dispatched") is True, after_review.get("training_dispatched"))
    assert_ok("training job id", bool(after_review.get("training_job_id")), after_review.get("training_job_id"))

    print("\n[5/7] Wait Worker callback")
    final_state = wait_for_terminal(run_id)

    print("\n[6/7] Final state verification")
    checks = [
        ("phase", final_state.get("current_phase") == "EVENT_CLOSED", final_state.get("current_phase")),
        ("callback status", final_state.get("training_callback_status") == "SUCCEEDED", final_state.get("training_callback_status")),
        ("action", final_state.get("recommended_action") == "MODEL_ITERATION", final_state.get("recommended_action")),
        ("qualification", bool(final_state.get("qualification_run_id")), final_state.get("qualification_run_id")),
        ("deployment", bool(final_state.get("deployment_id")), final_state.get("deployment_id")),
        ("deployment stage", final_state.get("deployment_stage") == "PRODUCTION", final_state.get("deployment_stage")),
        ("deployment decision", final_state.get("deployment_decision") == "PROMOTE", final_state.get("deployment_decision")),
        ("challenger qualified", final_state.get("challenger_qualified") is True, final_state.get("challenger_qualified")),
    ]
    for label, condition, detail in checks:
        assert_ok(label, condition, detail)

    print("\n[7/7] Summary")
    print("=" * 68)
    print(f"Result: {passed} passed, {failed} failed")
    print("PASS real Celery worker closed the lifecycle" if failed == 0 else "FAIL real Celery worker path failed")
    print("=" * 68)


if __name__ == "__main__":
    run()
    sys.exit(0 if failed == 0 else 1)
