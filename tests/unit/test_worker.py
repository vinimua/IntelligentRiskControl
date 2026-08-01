from apps.modelops_api.config import settings
from workers import training_tasks
from workers.app import app
from workers.training_tasks import TRAINERS


def test_worker_uses_shared_settings():
    assert app.conf.broker_url == settings.celery_broker_url
    assert app.conf.result_backend == settings.celery_result_backend


def _training_frame():
    import pandas as pd

    rows = []
    for idx in range(120):
        label = idx % 2
        rows.append(
            {
                "sample_id": f"s-{idx}",
                "feature_a": float(idx % 11),
                "feature_b": float((idx * 3) % 17),
                "feature_c": float(label * 2 + (idx % 5)),
                "is_bad": label,
            }
        )
    return pd.DataFrame(rows)


def test_training_registry_exposes_supported_algorithms():
    assert set(TRAINERS) >= {"lightgbm", "logistic_regression", "random_forest"}


def test_logistic_regression_trainer_returns_common_result_contract():
    result = TRAINERS["logistic_regression"](
        _training_frame(),
        seed=2026,
        hyperparameters={"C": 0.5},
    )

    assert result["model"] is not None
    assert result["feature_cols"] == ["feature_a", "feature_b", "feature_c"]
    assert 0 <= result["train_auc"] <= 1
    assert 0 <= result["val_auc"] <= 1
    assert 0 <= result["train_ks"] <= 1
    assert 0 <= result["val_ks"] <= 1


def test_random_forest_trainer_returns_common_result_contract():
    result = TRAINERS["random_forest"](
        _training_frame(),
        seed=2026,
        hyperparameters={"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 2},
    )

    assert result["model"] is not None
    assert result["feature_cols"] == ["feature_a", "feature_b", "feature_c"]
    assert 0 <= result["train_auc"] <= 1
    assert 0 <= result["val_auc"] <= 1


def test_training_data_prefers_feature_snapshot(monkeypatch):
    snapshot_df = _training_frame()

    monkeypatch.setattr(
        training_tasks,
        "_load_feature_snapshot",
        lambda snapshot_ids, window_ids: snapshot_df,
    )

    def fail_load_window(_window_id):
        raise AssertionError("raw window loader should not be called when snapshot is available")

    monkeypatch.setattr(
        "apps.modelops_api.services.monitoring.window_loader.load_window",
        fail_load_window,
    )

    loaded = training_tasks._load_training_data(["W2"], ["feature-snapshot-001"])

    assert loaded is snapshot_df
