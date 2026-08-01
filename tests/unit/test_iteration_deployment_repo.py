from pathlib import Path


def test_deployment_queries_use_model_registry_champion_column():
    source = Path("apps/modelops_api/repositories/iteration_repo.py").read_text(
        encoding="utf-8"
    )

    assert "m.current_champion_version as current_champion" in source
    assert "m.champion_version as current_champion" not in source
