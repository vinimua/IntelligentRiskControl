from apps.modelops_api.services.workflow.executors import (
    create_calibration_plan,
    create_repair_plan,
    create_threshold_plan,
    dispatch_training_job,
    execute_deployment_stage,
)


class FakeCeleryApp:
    def __init__(self):
        self.sent = []

    def send_task(self, name, args):
        self.sent.append((name, args))

        class Result:
            id = "celery-task-001"

        return Result()


async def test_dispatch_training_job_sends_celery_task_when_app_provided():
    app = FakeCeleryApp()
    job_input = {
        "training_job_id": "job-001",
        "idempotency_key": "iter:round-1:exp-exp",
    }

    result = await dispatch_training_job(job_input, celery_app=app)

    assert result["training_job_id"] == "job-001"
    assert result["dispatched"] is True
    assert result["error"] is None
    assert app.sent == [("workers.training_tasks.train_model", [job_input])]


async def test_dispatch_training_job_mock_path_without_app():
    result = await dispatch_training_job({"training_job_id": "job-002"})

    assert result["training_job_id"] == "job-002"
    assert result["dispatched"] is False
    assert result["error"] is None


def test_plan_executors_accept_plain_state_dict():
    state = {
        "recommended_action": "DATA_REPAIR",
        "diagnosis_run_id": "diag-001",
        "champion_version": "champion_v1",
        "business_round": 2,
    }

    repair = create_repair_plan(state)
    calibration = create_calibration_plan(state)
    threshold = create_threshold_plan(state)

    assert repair["status"] == "PENDING_EXTERNAL_REPAIR"
    assert repair["callback_endpoint"].startswith("/api/internal/iteration/repair/")
    assert calibration["calibrator_type"] == "isotonic"
    assert threshold["search_range"]["step"] == 0.01


def test_deployment_executor_advances_qualified_challenger():
    result = execute_deployment_stage(
        {
            "challenger_qualified": True,
            "champion_version": "champion_v1",
            "challenger_version": "challenger_v1",
        }
    )

    assert result["deployment_stage"] == "OOT_GATE"
    assert result["deployment_decision"] == "ADVANCE_STAGE"


def test_deployment_executor_promotes_after_final_canary():
    result = execute_deployment_stage(
        {
            "challenger_qualified": True,
            "champion_version": "champion_v1",
            "challenger_version": "challenger_v1",
            "deployment_stage": "CANARY_50",
        }
    )

    assert result["deployment_stage"] == "PRODUCTION"
    assert result["deployment_decision"] == "PROMOTE"


def test_deployment_executor_holds_when_health_check_fails():
    result = execute_deployment_stage(
        {
            "challenger_qualified": True,
            "deployment_stage": "CANARY_20",
            "deployment_health_passed": False,
        }
    )

    assert result["deployment_stage"] == "CANARY_20"
    assert result["deployment_decision"] == "HOLD"
    assert result["hold_reason"] == "DEPLOYMENT_HEALTH_CHECK_FAILED"


def test_deployment_executor_rolls_back_when_forced():
    result = execute_deployment_stage(
        {
            "challenger_qualified": True,
            "champion_version": "champion_v1",
            "deployment_stage": "CANARY_50",
            "deployment_force_rollback": True,
        }
    )

    assert result["deployment_stage"] == "CANARY_50"
    assert result["deployment_decision"] == "ROLLBACK"
    assert result["rollback_target"] == "champion_v1"
