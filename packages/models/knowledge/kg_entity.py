"""
统一知识实体、关系与 Observation
"""

from ..common.base import ContractModel

from datetime import datetime

from ..common.enums import DataTrack, EvidenceDirection

class KgEntity(ContractModel):
    """统一知识实体"""

    entity_code: str
    entity_type: str
    name: str
    namespace: str = "CORE"
    is_core: bool = False
    enabled: bool = True
    schema_version: str | None = None
    attributes_json: dict | None = None

class KgRelation(ContractModel):
    """
    统一知识关系 — 保存 initial/effective weight、案例数、置信区间
    不保存 CANDIDATE/PUBLISHED 成熟度
    """

    relation_key: str  # source_code|relation_type|target_code
    source_entity_code: str
    relation_type: str
    target_entity_code: str
    initial_prior_weight: float
    prior_strength: float = 1.0
    effective_weight: float
    confidence_lower_bound: float = 0.0
    confidence_upper_bound: float = 0.0
    evidence_case_count: int = 0
    natural_case_count: int = 0
    scenario_case_count: int = 0
    support_count: int = 0
    against_count: int = 0
    neutral_count: int = 0
    support_strength: float = 0.0
    against_strength: float = 0.0
    alpha_value: float | None = None
    beta_value: float | None = None
    weight_version: str
    enabled: bool = True
    last_calibrated_at: datetime | None = None

class KgRelationObservation(ContractModel):
    """
    动态权重统一事实表 — 最关键的校准输入
    唯一约束：relation_key + source_domain + source_record_id
    """

    relation_key: str
    source_domain: str  # DIAGNOSIS / ITERATION / DEPLOYMENT
    source_record_id: str
    direction: EvidenceDirection
    evidence_score: float | None = None
    quality_weight: float = 1.0
    weighted_strength: float | None = None
    data_track: DataTrack = DataTrack.NATURAL
    observed_at: datetime | None = None
