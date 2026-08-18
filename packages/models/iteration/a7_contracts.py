"""A7 V1 严格决策合同与 L1 策略选择合同（Pydantic 契约模型）。

【系统角色】「模型性能衰减自动修复」系统（A1-A7 七个修复动作：A1关闭、A2观察、
  A3数据修复、A4管道修复、A5校准、A6阈值、A7模型迭代）中 A7 模型迭代的「严格
  决策合同」。上游 Agent 产生的决策必须满足本文件的契约才能被下游消费，防止
  带病决策流入训练环节。

【核心模型】
- AffectedSegment：受影响客群片段（漂移客群）的契约；
- A7Authorization：A7 授权（自动规则 / 人工复核）；
- A7PrimaryRootCause：A7 首要根因（必须 CONFIRMED，仅自然漂移类根因）；
- A7DecisionEnvelope：A7 决策信封——A7 L1 选择器唯一接受的输入，
  decision_source 区分 SIMULATED（模拟）与 NATURAL（自然产生）；
- L1StrategyDecision：L1 策略选择结果，约束 SELECTED/BLOCKED 两种终态的字段完备性，
  并禁止 W4 进入训练/验证。

所有字段约束与跨字段校验均由 Pydantic 的 Field / model_validator 强制，
  不合规输入在反序列化阶段即被拒绝。
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ..common.base import ContractModel


# 自然根因编码：仅三类「自然漂移」根因可作为 A7 首要根因（与 A3/A4/A5/A6 区分）
NaturalRootCauseCode = Literal[
    "FEATURE_DRIFT",
    "SEGMENT_DRIFT",
    "CONCEPT_DRIFT",
]
# 受支持的算法家族（A7 迭代必须与 Champion 保持同一家族）
AlgorithmFamily = Literal[
    "LogisticRegression",
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "EBM",
]


class AffectedSegment(ContractModel):
    """受影响客群片段（漂移客群）契约。

    字段:
        segment_id: 客群片段唯一标识。
        segment_rule: 客群圈选规则（字典）。
        rule_version: 圈选规则版本。
        evidence_window_id: 证据窗口 ID。
        reference_share: 基准窗口中的客群占比（0~1）。
        current_share: 当前窗口中的客群占比（0~1）。
        current_sample_count: 当前窗口样本数（非负）。
        current_bad_count: 当前窗口坏样本数（非负，且不能超过样本总数）。
        performance_affected: 该片段性能是否受影响。
        primary: 是否为首要受影响片段。
    """
    segment_id: str
    segment_rule: dict
    rule_version: str
    evidence_window_id: str
    reference_share: float = Field(ge=0.0, le=1.0)
    current_share: float = Field(ge=0.0, le=1.0)
    current_sample_count: int = Field(ge=0)
    current_bad_count: int = Field(ge=0)
    performance_affected: bool
    primary: bool = False

    @model_validator(mode="after")
    def validate_counts(self) -> "AffectedSegment":
        # 坏样本数不得大于样本总数（基本逻辑一致性）
        if self.current_bad_count > self.current_sample_count:
            raise ValueError("current_bad_count cannot exceed current_sample_count")
        return self


class A7Authorization(ContractModel):
    """A7 授权契约。

    字段:
        authorization_type: 授权类型——AUTO_RULE（自动规则）或 MANUAL_REVIEW（人工复核）。
        authorization_id: 授权 ID（非空）。
        approved: 是否已批准。
    """
    authorization_type: Literal["AUTO_RULE", "MANUAL_REVIEW"]
    authorization_id: str = Field(min_length=1)
    approved: bool


class A7PrimaryRootCause(ContractModel):
    """A7 首要根因契约。

    字段:
        root_cause_code: 根因编码（仅自然漂移类：FEATURE_DRIFT/SEGMENT_DRIFT/CONCEPT_DRIFT）。
        candidate_status: 候选状态——必须为 CONFIRMED（已确认）。
        confidence: 置信度（0~1）。
        evidence_refs: 证据引用（至少一条）。
    """
    root_cause_code: NaturalRootCauseCode
    candidate_status: Literal["CONFIRMED"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(min_length=1)


class A7DecisionEnvelope(ContractModel):
    """A7 决策信封——A7 L1 选择器唯一接受的输入契约。

    决策信封把 Agent 的一次 A7 决策「封」成结构化契约，任何字段缺失、越界或
    跨字段矛盾都会在反序列化时被 model_validator 拒绝。其中：
    - decision_source 标记该决策是 SIMULATED（模拟回放）还是 NATURAL（自然运行产生）；
    - contract_version 固定为 "a7-decision-envelope-v1"，保证契约版本可追溯。

    字段:
        contract_version: 契约版本（固定 v1）。
        decision_source: 决策来源——SIMULATED / NATURAL。
        lifecycle_run_id: 生命周期运行 ID。
        event_id: 事件 ID。
        monitoring_run_id: 监控运行 ID。
        diagnosis_run_id: 诊断运行 ID。
        agent_decision_id: Agent 决策 ID。
        model_id: 目标模型 ID。
        champion_version: Champion 版本。
        champion_artifact_checksum: Champion 制品 checksum（sha256 格式）。
        model_task_type: 任务类型（固定二分类）。
        algorithm_family: 算法家族。
        recommended_action: 推荐动作（固定 MODEL_ITERATION，即 A7）。
        primary_root_cause: 首要根因。
        decay_degree: 衰减程度——短期 7 天 / 持续 30 天 / 严重。
        trigger_context: 触发上下文（字典）。
        kg_strategy_candidates: 知识图谱策略候选（列表）。
        affected_segments: 受影响客群片段（列表）。
        impact_scope: 影响范围——LOCAL / GLOBAL（概念漂移时必填）。
        change_pattern: 变化模式——GRADUAL / SUDDEN（概念漂移时必填）。
        change_point: 变更点（突变型概念漂移时必填）。
        ordered_evidence_window_ids: 有序证据窗口 ID（渐变型概念漂移时必填）。
        authorization: 授权信息。
        rule_versions: 规则版本表（含 agent/l1/window/qualification 四个必需版本）。
    """
    contract_version: Literal["a7-decision-envelope-v1"] = (
        "a7-decision-envelope-v1"
    )
    decision_source: Literal["SIMULATED", "NATURAL"]
    lifecycle_run_id: str
    event_id: str
    monitoring_run_id: str
    diagnosis_run_id: str
    agent_decision_id: str
    model_id: str
    champion_version: str
    # Champion 制品 checksum：必须是 sha256: 前缀 + 16~64 位十六进制
    champion_artifact_checksum: str = Field(pattern=r"^sha256:[0-9a-fA-F]{16,64}$")
    model_task_type: Literal["BINARY_CLASSIFICATION"]
    algorithm_family: AlgorithmFamily
    recommended_action: Literal["MODEL_ITERATION"]
    primary_root_cause: A7PrimaryRootCause
    decay_degree: Literal["SHORT_TERM_7D", "SUSTAINED_30D", "SEVERE"]
    trigger_context: dict
    kg_strategy_candidates: list[dict] = Field(default_factory=list)
    affected_segments: list[AffectedSegment] = Field(default_factory=list)
    impact_scope: Literal["LOCAL", "GLOBAL"] | None = None
    change_pattern: Literal["GRADUAL", "SUDDEN"] | None = None
    change_point: str | None = None
    ordered_evidence_window_ids: list[str] = Field(default_factory=list)
    authorization: A7Authorization
    rule_versions: dict[str, str]

    @model_validator(mode="after")
    def validate_conditional_context(self) -> "A7DecisionEnvelope":
        """跨字段条件校验：按根因与衰减程度施加一致性约束。

        约束要点：
        - 授权必须已批准；
        - SEVERE（严重衰减）必须走 MANUAL_REVIEW（人工复核）授权；
        - CONCEPT_DRIFT 需根据影响范围/变化模式补充或禁止相应上下文字段；
        - 非 CONCEPT_DRIFT 根因不得携带概念漂移上下文；
        - SEGMENT_DRIFT 必须给出受影响客群；
        - rule_versions 必须包含 agent/l1/window/qualification 四个必需版本。
        """
        cause = self.primary_root_cause.root_cause_code
        if not self.authorization.approved:
            raise ValueError("authorization must be approved before A7")
        # 严重衰减是高危场景，必须人工复核，不允许自动规则放行
        if self.decay_degree == "SEVERE" and self.authorization.authorization_type != "MANUAL_REVIEW":
            raise ValueError("SEVERE requires MANUAL_REVIEW authorization")
        if cause == "CONCEPT_DRIFT":
            # 概念漂移必须声明影响范围与变化模式
            if self.impact_scope is None or self.change_pattern is None:
                raise ValueError("CONCEPT_DRIFT requires impact_scope and change_pattern")
            # 局部概念漂移必须有受影响客群；全局概念漂移禁止携带受影响客群
            if self.impact_scope == "LOCAL" and not self.affected_segments:
                raise ValueError("LOCAL CONCEPT_DRIFT requires affected_segments")
            if self.impact_scope == "GLOBAL" and self.affected_segments:
                raise ValueError("GLOBAL CONCEPT_DRIFT forbids affected_segments")
            # 突变型必须有变更点；渐变型禁止变更点但必须有有序证据窗口
            if self.change_pattern == "SUDDEN" and not self.change_point:
                raise ValueError("SUDDEN CONCEPT_DRIFT requires change_point")
            if self.change_pattern == "GRADUAL":
                if self.change_point is not None:
                    raise ValueError("GRADUAL CONCEPT_DRIFT forbids change_point")
                if not self.ordered_evidence_window_ids:
                    raise ValueError(
                        "GRADUAL CONCEPT_DRIFT requires ordered evidence windows"
                    )
        elif any((self.impact_scope, self.change_pattern, self.change_point)):
            # 概念漂移上下文只对 CONCEPT_DRIFT 根因有效，其它根因不得携带
            raise ValueError("concept drift context is only valid for CONCEPT_DRIFT")
        if cause == "SEGMENT_DRIFT" and not self.affected_segments:
            # 客群漂移必须给出受影响客群
            raise ValueError("SEGMENT_DRIFT requires affected_segments")
        # 必须包含四个核心规则版本，保证决策可追溯、可复现
        required_versions = {
            "agent_rule_version",
            "l1_matrix_version",
            "window_rule_version",
            "qualification_rule_version",
        }
        missing = sorted(required_versions.difference(self.rule_versions))
        if missing:
            raise ValueError(f"missing rule versions: {missing}")
        return self


class L1StrategyDecision(ContractModel):
    """L1 策略选择结果契约。

    表示 L1 策略选择器对 A7 决策信封给出的最终策略结论，终态只有两种：
    - SELECTED：选中某条策略，必须携带完整可执行字段；
    - BLOCKED：策略被阻断，必须携带阻断原因，且不得携带可执行字段。

    字段:
        selection_status: 选择状态——SELECTED / BLOCKED。
        primary_strategy: 主策略编码（SELECTED 时必填）。
        execution_mode: 执行模式（SELECTED 时必填）。
        training_data_mode: 训练数据模式（SELECTED 时必填）。
        training_window_ids: 训练窗口 ID 列表（SELECTED 时必填）。
        validation_window_ids: 验证窗口 ID 列表（SELECTED 时必填）。
        oot_window_ids: OOT 窗口 ID（默认 ["W4"]）。
        same_algorithm_family: 是否与 Champion 同算法家族（必须为 True）。
        algorithm_family: 算法家族。
        sample_weight_required: 是否需要样本加权。
        sample_weight_policy: 样本加权策略（需要加权时必填）。
        feature_reconstruction_required: 是否需要特征重建。
        feature_schema_change_allowed: 是否允许特征 schema 变更。
        kg_strategy_candidates: 知识图谱策略候选。
        strategy_source: 策略来源。
        kg_consistency_status: KG 一致性状态。
        kg_repair_required: 是否需要 KG 修复。
        selection_reason_codes: 选择理由编码列表。
        selection_rule_version: 选择规则版本。
        blocking_reasons: 阻断原因列表（BLOCKED 时必填）。
    """
    selection_status: Literal["SELECTED", "BLOCKED"]
    primary_strategy: str | None = None
    execution_mode: str | None = None
    training_data_mode: str | None = None
    training_window_ids: list[str] = Field(default_factory=list)
    validation_window_ids: list[str] = Field(default_factory=list)
    oot_window_ids: list[str] = Field(default_factory=lambda: ["W4"])
    same_algorithm_family: bool
    algorithm_family: AlgorithmFamily
    sample_weight_required: bool = False
    sample_weight_policy: dict = Field(default_factory=dict)
    feature_reconstruction_required: bool = False
    feature_schema_change_allowed: bool = False
    kg_strategy_candidates: list[dict] = Field(default_factory=list)
    strategy_source: Literal[
        "KG_WITH_L1_GUARDRAILS",
        "KG_AND_L1",
        "L1_OVERRIDE",
        "L1_FALLBACK",
    ]
    kg_consistency_status: Literal[
        "KG_SELECTED_L1_VALIDATED",
        "KG_TOP_BLOCKED_L1_SELECTED_NEXT",
        "KG_CANDIDATES_BLOCKED_BY_L1",
        "CONSISTENT",
        "KG_STRATEGY_MISMATCH",
        "KG_STRATEGY_MISSING",
    ]
    kg_repair_required: bool = False
    selection_reason_codes: list[str] = Field(default_factory=list)
    selection_rule_version: str
    blocking_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_selected_or_blocked(self) -> "L1StrategyDecision":
        """强制 SELECTED / BLOCKED 两种终态各自的字段完备性。

        约束要点：
        - 算法家族不一致（跨家族变更）必须显式阻断（ALGORITHM_FAMILY_CHANGE_FORBIDDEN）；
        - BLOCKED 状态不得携带任何可执行字段，且必须有阻断原因；
        - SELECTED 状态必须携带全部可执行字段，且保持 Champion 算法家族；
        - W4（最终盲测集）绝不能进入训练或验证窗口（防数据泄漏）；
        - 需要样本加权时必须给出加权策略。
        """
        if not self.same_algorithm_family:
            if "ALGORITHM_FAMILY_CHANGE_FORBIDDEN" not in self.blocking_reasons:
                raise ValueError("algorithm family mismatch must be blocked explicitly")
        if self.selection_status == "BLOCKED":
            # 被阻断的决策不得残留任何可执行字段，避免被误执行
            if any(
                (
                    self.primary_strategy,
                    self.execution_mode,
                    self.training_data_mode,
                    self.training_window_ids,
                    self.validation_window_ids,
                )
            ):
                raise ValueError("BLOCKED selection cannot contain executable fields")
            if not self.blocking_reasons:
                raise ValueError("BLOCKED selection requires blocking_reasons")
            return self
        if not self.same_algorithm_family:
            # 选中策略必须保持 Champion 算法家族（跨家族变更被禁止）
            raise ValueError("SELECTED strategy must keep the Champion algorithm family")
        if not all(
            (
                self.primary_strategy,
                self.execution_mode,
                self.training_data_mode,
                self.training_window_ids,
                self.validation_window_ids,
            )
        ):
            # 选中策略必须字段齐全，否则无法被下游训练 Worker 执行
            raise ValueError("SELECTED strategy requires executable fields")
        # 训练/验证窗口绝不能包含 W4——W4 是最终盲测集，防止数据泄漏
        if "W4" in self.training_window_ids or "W4" in self.validation_window_ids:
            raise ValueError("W4 must never enter training or validation")
        if self.sample_weight_required and not self.sample_weight_policy:
            # 需要加权时必须提供加权策略，否则训练无法落地
            raise ValueError("weighted strategy requires sample_weight_policy")
        return self
