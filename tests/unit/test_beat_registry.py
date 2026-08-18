"""Beat 配置与 Celery 注册一致性测试（A7 §8）。"""


def _celery_app():
    from workers.app import app as celery_app

    # 显式加载 app.conf.imports 声明的任务模块，触发注册
    for module_name in celery_app.conf.imports:
        __import__(module_name)
    celery_app.finalize()
    return celery_app


def test_beat_schedule_task_names_registered_in_celery():
    """Beat 配置中的每个任务名必须存在于 Celery 注册表，
    否则 Beat 会发送未注册任务，定时触发不会执行。"""
    celery_app = _celery_app()

    beat_schedule = celery_app.conf.beat_schedule or {}
    assert beat_schedule, "beat_schedule 为空"

    registered = set(celery_app.tasks.keys())
    missing = [
        name
        for entry in beat_schedule.values()
        for name in [entry["task"]]
        if name not in registered
    ]
    assert missing == [], f"Beat 引用了未注册任务: {missing}"


def test_scheduled_trigger_task_name_matches_module():
    """模块名与注册任务名一致（workers.lifecycle_trigger_tasks.*）。"""
    celery_app = _celery_app()

    name = "workers.lifecycle_trigger_tasks.scheduled_lifecycle_trigger"
    assert name in celery_app.tasks
