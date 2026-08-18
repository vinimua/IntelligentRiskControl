"""
Iteration Context — KnowledgeService 返回策略候选
"""

from pydantic import Field

from ..common.enums import TrainingMode
from ..common.base import ContractModel

from ..diagnosis.diagnosis_context import DocumentRef

class StrategyCandidate(ContractModel):
    """策略候选 — 包含图谱权重快照和历史效果"""

    strategy_code: str
    recommends_relation_key: str  # RootCause|RECOMMENDS|Strategy
    mitigates_relation_key: str   # Strategy|MITIGATES|RootCause
    relation_effective_weight_snapshot: float
    # 历史有效率与先验分离：无真实历史案例时为 None，不得把初始专家权重
    # 伪装成历史有效率；排序使用 strategy_rank_score。
    historical_effectiveness: float | None = None
    # 候选排序分：有真实历史 = 历史有效率；无历史 = 初始先验权重
    #（与 doc/接口约束总汇_V1.0 §26 一致：禁止与 final_strategy_confidence 混用）
    strategy_rank_score: float = 0.0
    rank_score_source: str = "INITIAL_PRIOR"  # INITIAL_PRIOR / CALIBRATED_HISTORY
    support_case_count: int
    total_case_count: int
    natural_case_count: int
    confidence_lower_bound: float
    required_data_codes: list[str] = Field(default_factory=list)
    allowed_training_window_ids: list[str] = Field(default_factory=list)
    validation_window_ids: list[str] = Field(default_factory=list)
    algorithm: str | None = None
    feature_schema_version: str | None = None
    preprocessing_version: str | None = None
    label_versions: list[str] = Field(default_factory=list)
    hyperparameters: dict = Field(default_factory=dict)
    sample_weight_policy: dict = Field(default_factory=dict)
    training_cost_level: str = "MEDIUM"
    risk_level: str = "LOW"
    executor_code: str
    strategy_tier: str = "full"  # "full" / "light" / "minimal" — KG 边上的策略等级
    # A7 §6.1: 边级证据门控（sustained_30d / champion_artifact_available /
    # schema_compatible / incremental_algorithm_supported /
    # unstable_feature_subset_confirmed / manual_approval 等），
    # 查询时按 available_context_codes 逐边校验
    required_context: list[str] = Field(default_factory=list)
    # 主训练模式：TrainingMode 正式枚举 —— 来自 Strategy 节点
    # primary_training_mode 属性，禁止下游从 strategy_tier 猜测训练模式
    primary_training_mode: TrainingMode = TrainingMode.FULL_RETRAIN

class IterationContext(ContractModel):
    """
    策略候选知识召回包
    不返回 final_strategy_confidence 或 selected_strategy
    """

    context_pack_id: str
    diagnosis_run_id: str
    root_cause_code: str
    weight_version: str
    strategy_candidates: list[StrategyCandidate] = Field(default_factory=list)
    rules: dict | None = None
    retrieved_references: list[DocumentRef] = Field(default_factory=list)
    retrieval_degraded: bool = False
