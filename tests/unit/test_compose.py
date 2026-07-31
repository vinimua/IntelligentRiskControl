from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_compose_declares_seven_services_and_health_management():
    payload = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = payload["services"]

    assert set(services) == {
        "postgres",
        "redis",
        "minio",
        "minio-init",
        "mlflow",
        "neo4j",
        "qdrant",
    }
    assert all(
        "healthcheck" in config
        for name, config in services.items()
        if name != "minio-init"
    )
    assert services["mlflow"]["depends_on"]["minio-init"]["condition"] == (
        "service_completed_successfully"
    )


def test_mlflow_runtime_dependencies_are_pinned():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "mlflow==2.17.2" in compose
    assert "boto3==1.35.54" in compose
