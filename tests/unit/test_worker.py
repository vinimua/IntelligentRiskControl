from apps.modelops_api.config import settings
from workers.app import app


def test_worker_uses_shared_settings():
    assert app.conf.broker_url == settings.celery_broker_url
    assert app.conf.result_backend == settings.celery_result_backend
