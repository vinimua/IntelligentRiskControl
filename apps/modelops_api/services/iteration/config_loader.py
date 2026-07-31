"""加载任务三版本化 YAML 配置。"""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from packages.models.config.gatekeeper import (
    IterationRuleConfig,
    QualificationRuleConfig,
    RiskRuleConfig,
    StrategyCatalog,
)


class IterationConfigBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    iteration: IterationRuleConfig
    strategies: StrategyCatalog
    qualification: QualificationRuleConfig
    risk: RiskRuleConfig


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return payload


@lru_cache(maxsize=1)
def load_iteration_config() -> IterationConfigBundle:
    config_dir = _project_root() / "assets" / "configs"
    return IterationConfigBundle(
        iteration=IterationRuleConfig.model_validate(
            _read_yaml(config_dir / "iteration.yaml")
        ),
        strategies=StrategyCatalog.model_validate(
            _read_yaml(config_dir / "repair_strategies.yaml")
        ),
        qualification=QualificationRuleConfig.model_validate(
            _read_yaml(config_dir / "qualification.yaml")
        ),
        risk=RiskRuleConfig.model_validate(
            _read_yaml(config_dir / "risk_review.yaml")
        ),
    )
