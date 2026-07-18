"""
Gatekeeper 规则与迭代策略配置
来源：技术开发文档 V1.4.2 §8.6 (训练侧 Gatekeeper), §9.5 (部署 Gatekeeper), §8.4
"""

from pydantic import Field
from ..common.base import ContractModel

class GatekeeperRule(ContractModel):
    """单条 Gatekeeper 规则"""

    rule_code: str
    rule_version: str
    metric_code: str
    min_improvement: float | None = None
    max_score_psi: float | None = None
    max_train_valid_gap: float | None = None
    data_leakage_check: bool = True
    interpretability_required: bool = False

class IterationRuleConfig(ContractModel):
    """
    任务三策略约束 — 来自版本化 YAML
    max_iteration_rounds, allowed_strategies, training_window_policy, Risk Guard
    """

    max_iteration_rounds: int = 3
    allowed_strategy_codes: list[str] = Field(default_factory=list)
    training_window_policy: str = "W1_W3_ONLY"
    rule_version: str
    oscillation_threshold: int = 3
