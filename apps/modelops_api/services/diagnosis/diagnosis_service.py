"""DiagnosisService — 任务二四维根因诊断核心。

给定 AlertContext + 监控数据，通过知识图谱 + 全量 D/R/C/T/I 验证器输出根因排序。

六步管线:
  1. 候选召回 (KG)
  2. 加载证据 (PG: drift + metrics + feature importance)
  3. 执行全部验证器 (D/R/C/T/I — 每个候选跑 3~5 个独立验证器)
  4. PathRanker 融合 (KG weight × 0.6 + avg_evidence × 0.4)
  5. 持久化 (diagnosis schema)
  6. 输出 DiagnosisStateOutput
"""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from packages.models.diagnosis.diagnosis_context import CandidateRootCause
from packages.models.diagnosis.diagnosis_path import DiagnosisPath
from packages.models.diagnosis.diagnosis_state import DiagnosisStateOutput
from packages.models.diagnosis.evidence import EvidenceItem
from packages.models.common.enums import (
    DimensionCode,
    RecommendedAction,
)
from packages.models.monitoring.alert_context import AlertContext

from ..knowledge_service import KnowledgeService
from ...repositories.diagnosis_repo import DiagnosisRepo
from ...repositories.monitoring_repo import MonitoringRepo
from .executor_registry import EXECUTOR_REGISTRY, _lazy_register_all

# ── 项目根目录（用于读取模型产物）──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# ── 每个 RootCause 应执行的验证器（D/R/C/T/I 全覆盖）──
ROOT_CAUSE_VALIDATORS: dict[str, list[str]] = {
    "feature_drift": [
        "psi_check",                       # D: PSI 分布漂移
        "counterfactual_repair_check",     # R: 修复漂移特征后性能可恢复？
        "drift_group_regression",          # C: 集体漂移与退化的相关性
        "temporal_precedence_check",       # T: 漂移是否先于退化？
        "permutation_importance_check",    # I: 漂移特征是否重要？
    ],
    "model_aging": [
        "psi_check",
        "counterfactual_repair_check",
        "drift_group_regression",
        "temporal_precedence_check",
        "permutation_importance_check",
    ],
    "data_quality_issue": [
        "missing_outlier_range_check",     # D: 缺失/异常检测
        "drift_group_regression",          # C: 质量劣化与退化的相关性
        "temporal_precedence_check",       # T: 质量劣化是否先于退化？
    ],
    "data_pipeline_issue": [
        "missing_outlier_range_check",     # D: Schema/数据异常
        "drift_group_regression",
        "temporal_precedence_check",
    ],
    "feature_failure": [
        "psi_check",
        "counterfactual_repair_check",
        "permutation_importance_check",
        "drift_group_regression",
    ],
    "population_shift": [
        "psi_check",
        "drift_group_regression",
        "temporal_precedence_check",
    ],
    "PRIOR_PROBABILITY_SHIFT": [
        "psi_check",                       # D: 标签分布漂移
        "drift_group_regression",          # C: 标签率变化与退化的相关性
        "temporal_precedence_check",       # T: 标签变化先于退化？
    ],
    # business_policy_change 无适用验证器：本系统没有"业务政策变更"的直接
    # 证据生产者（政策事件源不在监控范围内）。给空清单，候选只靠 KG 先验
    # 参与排名（证据中性 0.5），不再吃默认验证器的 blanket SUPPORT。
    # 这与"诚实初始化"一致：证据拿不到就不编造。
    "business_policy_change": [],
    "CONCEPT_DRIFT": [
        "psi_check",                       # D: 性能指标漂移
        "counterfactual_repair_check",     # R: 假设修复后性能能否恢复
        "drift_group_regression",          # C: 性能退化集体相关性
        "temporal_precedence_check",       # T: 时间先后
        "permutation_importance_check",    # I: 重要特征是否受影响
    ],
    "FRAUD_PATTERN_SHIFT": [
        "psi_check",
        "counterfactual_repair_check",
        "drift_group_regression",
        "temporal_precedence_check",
    ],
}

# ── 默认验证器集合（兜底：未知根因类型时跑 D 类型的两个基础验证器）──
_DEFAULT_VALIDATORS = ["psi_check", "missing_outlier_range_check"]

# ── required_context 前置条件映射：告警码 → 可满足的上下文类型 ──
# 静态映射只用于"其他告警佐证自身关系的 required_context"；自身告警不能自证
#（见 _recall_candidates 的检查逻辑）。
# schema_contract_violation 不在此静态映射中：必须来自 SCHEMA_MISMATCH 告警
# payload 里的真实列级差异证据（missing_columns/extra_columns），凭告警码不算。
_ALERT_CONTEXT_MAP: dict[str, set[str]] = {
    "MISSING_RATE_SPIKE": {"data_quality_evidence"},
    "OUTLIER_RATE_SPIKE": {"data_quality_evidence"},
    "PREDICTION_MEAN_SHIFT": {"prior_probability_evidence"},
    "BAD_RATE_SHIFT": {"prior_probability_evidence"},
}


def _payload_alert_contexts(alert) -> set[str]:
    """从告警 payload 收集真实内容证据（非告警码映射）。

    schema_contract_violation：仅当 SCHEMA_MISMATCH 告警详情携带真实的
    列级差异（missing_columns/extra_columns）时授予。
    """
    contexts: set[str] = set()
    if getattr(alert, "alert_code", "") == "SCHEMA_MISMATCH":
        detail = getattr(alert, "metric_detail", None) or {}
        if detail.get("missing_columns") or detail.get("extra_columns"):
            contexts.add("schema_contract_violation")
    return contexts


@dataclass
class DiagnosisService:
    session: AsyncSession
    knowledge: KnowledgeService
    repo: DiagnosisRepo

    async def diagnose(
        self,
        alert_context: AlertContext,
        monitoring_run_id: str,
        lifecycle_run_id: str | None = None,
        event_id: str | None = None,
        evidence_window_ids: list[str] | None = None,
    ) -> DiagnosisStateOutput:
        """执行完整诊断流程 — 六步管线。"""

        # 确保验证器已注册
        _lazy_register_all()

        alert_details = alert_context.alert_details or []
        if not alert_details:
            return DiagnosisStateOutput(
                diagnosis_run_id=str(uuid.uuid4()),
                primary_root_cause_code="no_alerts",
                primary_root_cause_dimension=DimensionCode.DATA,
                primary_root_cause_score=0.0,
                recommended_action=RecommendedAction.CONTINUE_OBSERVATION,
                need_iteration=False,
            )

        # ── 0a. 资格门禁检查（gate_only 告警阻断性能类根因诊断）──
        alert_codes = [
            getattr(a, "alert_code", "")
            for a in alert_details
            if getattr(a, "alert_code", None)
        ]
        gate_alerts = await self.knowledge.query_gate_blocking_alerts(alert_codes)
        if gate_alerts:
            blocking = [g["gate_semantics"] for g in gate_alerts
                        if g.get("gate_semantics") == "DATA_ELIGIBILITY_BLOCK"]
            if blocking:
                import structlog
                structlog.get_logger(__name__).warning(
                    "diagnosis_blocked_by_gate_alert",
                    gate_alerts=[g["alert_code"] for g in gate_alerts],
                )
                return DiagnosisStateOutput(
                    diagnosis_run_id=None,  # 未创建诊断记录，不造假 run_id
                    primary_root_cause_code="insufficient_data",
                    primary_root_cause_dimension=DimensionCode.DATA,
                    primary_root_cause_score=0.0,
                    recommended_action=RecommendedAction.CONTINUE_OBSERVATION,
                    need_iteration=False,
                    diagnosis_status="INSUFFICIENT_DATA",
                )

        # ── 0. 加载上下文数据（model_id, feature importance, multi-window drift, metrics）──
        mon_repo = MonitoringRepo(self.session)
        run = await mon_repo.get_run(monitoring_run_id)
        model_id = run["model_id"] if run else None

        # ── 1. 候选召回 ──
        candidates = await self._recall_candidates(alert_details)

        # ── 2. 加载证据（三步并行加载）──
        drift_data = await self._load_drift_data(monitoring_run_id, evidence_window_ids)
        multi_window_drift = await self._load_multi_window_drift(
            monitoring_run_id, evidence_window_ids
        )
        metrics = await self._load_metrics(monitoring_run_id, evidence_window_ids)
        feature_importance = await self._load_feature_importance(model_id)

        # ── 3. 执行验证器（D/R/C/T/I 全覆盖）──
        evidence_packages = await self._execute_validation(
            candidates=candidates,
            drift_data=drift_data,
            alert_details=alert_details,
            multi_window_drift=multi_window_drift,
            metrics=metrics,
            feature_importance=feature_importance,
        )

        # ── 4. PathRanker ──
        ranked = await self._rank(candidates, evidence_packages)

        # ── 5. 持久化 ──
        run_result = await self.repo.create_run(
            monitoring_run_id, lifecycle_run_id, len(alert_details), event_id,
        )
        diag_id = run_result["diagnosis_run_id"]

        candidate_records = []
        for i, (rc, path) in enumerate(ranked[:10]):
            candidate_records.append({
                "alert_code": rc.alert_code,
                "root_cause_code": rc.root_cause_code,
                "dimension_code": rc.dimension_code,
                "relation_key": rc.relation_key,
                "effective_weight": rc.effective_weight_snapshot,
                "evidence_case_count": rc.evidence_case_count_snapshot,
                "confidence_lower_bound": rc.confidence_lower_bound_snapshot,
                "ranked_score": path.path_score,
                "rank_no": path.rank_no,
                "is_primary": i == 0,
            })
        candidate_ids = await self.repo.batch_insert_candidates(
            diag_id, candidate_records
        )

        # ── 5b. 持久化证据 ──
        for rc_root_cause, ev_items in evidence_packages.items():
            candidate_id = candidate_ids.get(rc_root_cause)
            if not candidate_id:
                continue
            for ev_item in ev_items:
                await self.repo.insert_evidence({
                    "diagnosis_run_id": diag_id,
                    "candidate_id": candidate_id,
                    "hypothesis_code": rc_root_cause,
                    "evidence_type": ev_item.evidence_type.value
                    if hasattr(ev_item.evidence_type, 'value') else str(ev_item.evidence_type),
                    "method_code": ev_item.method_code,
                    "normalized_score": ev_item.normalized_score,
                    "direction": ev_item.direction.value
                    if hasattr(ev_item.direction, 'value') else str(ev_item.direction),
                    "applicable": ev_item.applicable,
                    "evidence_detail_json": json.dumps(
                        ev_item.evidence_detail_json, ensure_ascii=False, default=str
                    ) if ev_item.evidence_detail_json else "{}",
                })

        # ── 6. 输出 ──
        primary = ranked[0][1] if ranked else None
        if primary:
            primary_rc = ranked[0][0]
            recommended_action = _dimension_to_action(primary_rc.dimension_code)
            # A7 §4/§5: 从真实漂移证据推导 L1 结构化上下文
            #（impact_scope / change_pattern 生产者，不依赖人工构造输入）
            impact_scope, change_pattern = _infer_drift_context(
                drift_data=drift_data,
                multi_window_drift=multi_window_drift,
                root_cause_code=primary.root_cause_code,
            )
            # A7 §4: 客群漂移时生成冻结合格客群定义（segment_weighted 证据）
            segment_evidence = None
            if (primary.root_cause_code or "").upper() in {
                "POPULATION_SHIFT", "SEGMENT_DRIFT",
            }:
                segment_evidence = _infer_segment_evidence(
                    model_id, drift_data,
                )
            await self.repo.complete_run(
                diag_id,
                primary_root_cause_code=primary.root_cause_code,
                primary_root_cause_dimension=primary.dimension_code.value
                if hasattr(primary.dimension_code, 'value') else str(primary.dimension_code),
                primary_root_cause_score=primary.path_score,
                recommended_action=recommended_action.value,
                need_iteration=recommended_action == RecommendedAction.MODEL_ITERATION,
            )
            return DiagnosisStateOutput(
                diagnosis_run_id=diag_id,
                primary_root_cause_code=primary.root_cause_code,
                primary_root_cause_dimension=primary.dimension_code,
                primary_root_cause_score=primary.path_score,
                recommended_action=recommended_action,
                need_iteration=recommended_action == RecommendedAction.MODEL_ITERATION,
                impact_scope=impact_scope,
                change_pattern=change_pattern,
                segment_evidence=segment_evidence,
            )
        else:
            await self.repo.complete_run(
                diag_id,
                primary_root_cause_code="uncertain",
                primary_root_cause_dimension=DimensionCode.FEATURE.value,
                primary_root_cause_score=0.0,
                recommended_action=RecommendedAction.MANUAL_REVIEW.value,
                need_iteration=False,
                status="NO_CANDIDATES",
            )
            return DiagnosisStateOutput(
                diagnosis_run_id=diag_id,
                primary_root_cause_code="uncertain",
                primary_root_cause_dimension=DimensionCode.FEATURE,
                primary_root_cause_score=0.0,
                recommended_action=RecommendedAction.MANUAL_REVIEW,
                need_iteration=False,
                requires_manual_review=True,
            )

    # ═══════════════════════════════════════════════════════════════
    #  内部方法
    # ═══════════════════════════════════════════════════════════════

    async def _recall_candidates(
        self, alert_details: list
    ) -> list[CandidateRootCause]:
        """Step 1: 从 Neo4j 召回候选根因。

        流程：
        1. 每条关系在聚合前独立校验 required_context —— 不满足的关系单独
           丢弃，不得否决同根因的其他有效关系；
        2. 两阶段聚合（顺序无关）：
           - 第一遍：普通关系 → 建候选，权重取 MAX
           - 第二遍：supporting_only 关系 → 只附加到已存在的候选，绝不新建
        """
        raw_relations: list[tuple[str, CandidateRootCause]] = []
        for alert in alert_details:
            code = getattr(alert, "alert_code", None)
            if not code:
                continue
            for rc in await self.knowledge.query_candidate_root_causes(code):
                raw_relations.append((code, rc))

        # ── 关系级 required_context 前置校验（聚合前，防自证）──
        # 每条关系的 required_context 必须能被"其他告警"满足：
        # - 静态映射上下文（prior_probability_evidence 等）：排除该关系自身的
        #   告警码，避免同一告警自证（如 PREDICTION_MEAN_SHIFT 不能单独证明
        #   标签先验变化，需 BAD_RATE_SHIFT 同现）。
        # - payload 上下文（schema_contract_violation）：来自告警详情的真实
        #   列级差异证据，自身告警的 payload 也有效。
        alert_by_code = {
            getattr(a, "alert_code", ""): a
            for a in alert_details
            if getattr(a, "alert_code", None)
        }
        valid_relations: list[tuple[str, CandidateRootCause]] = []
        for code, rc in raw_relations:
            if not rc.required_context:
                valid_relations.append((code, rc))
                continue
            available: set[str] = set()
            for ac, detail in alert_by_code.items():
                if ac != code:
                    available |= _ALERT_CONTEXT_MAP.get(ac, set())
                available |= _payload_alert_contexts(detail)
            if set(rc.required_context).issubset(available):
                valid_relations.append((code, rc))
            else:
                import structlog
                structlog.get_logger(__name__).warning(
                    "relation_dropped_required_context_unsatisfied",
                    relation_key=rc.relation_key,
                    relation_alert_code=code,
                    required_context=rc.required_context,
                    alert_codes=list(alert_by_code.keys()),
                )

        aggregated: dict[str, CandidateRootCause] = {}
        ordered: list[str] = []
        pending_supporting: list[tuple[str, CandidateRootCause]] = []

        # ── 第一遍：普通关系建候选 ──
        for code, rc in valid_relations:
            root_code = rc.root_cause_code
            if rc.supporting_only:
                pending_supporting.append((code, rc))
                continue

            if root_code not in aggregated:
                aggregated[root_code] = rc.model_copy(update={
                    "supporting_alert_codes": [code],
                    "supporting_relation_keys": [rc.relation_key],
                })
                ordered.append(root_code)
                continue

            existing = aggregated[root_code]
            if code not in existing.supporting_alert_codes:
                existing.supporting_alert_codes.append(code)
            if rc.relation_key not in existing.supporting_relation_keys:
                existing.supporting_relation_keys.append(rc.relation_key)
            if _primary_relation_key(rc) > _primary_relation_key(existing):
                _replace_primary_relation(existing, rc)

        # ── 第二遍：supporting_only 关系附加到已有候选 ──
        for code, rc in pending_supporting:
            existing = aggregated.get(rc.root_cause_code)
            if existing is None:
                continue  # 没有主告警支持 → 不建独立候选
            if code not in existing.supporting_alert_codes:
                existing.supporting_alert_codes.append(code)
            if rc.relation_key not in existing.supporting_relation_keys:
                existing.supporting_relation_keys.append(rc.relation_key)

        return [aggregated[root_code] for root_code in ordered]

    async def _load_drift_data(
        self, monitoring_run_id: str, evidence_window_ids: list[str] | None = None
    ) -> list[dict]:
        """Load event-aligned feature drift; shared aggregate fallback is forbidden."""
        mon_repo = MonitoringRepo(self.session)
        rows = await mon_repo.get_feature_drift_by_run(monitoring_run_id)
        allowed = set(evidence_window_ids or [])
        return [
            row for row in rows
            if not allowed or str(row.get("window_id")) in allowed
        ]

    async def _load_multi_window_drift(
        self, monitoring_run_id: str,
        evidence_window_ids: list[str] | None = None,
    ) -> dict[str, list[dict]]:
        """Step 2b: 加载所有窗口的 drift 数据，按 window_id 分组。

        Returns:
            {"W1": [34 rows], "W3": [34 rows], "W6": [34 rows]}
        """
        all_rows = await self._load_drift_data(monitoring_run_id, evidence_window_ids)
        grouped: dict[str, list[dict]] = {}
        for row in all_rows:
            wid = row.get("window_id", "?")
            grouped.setdefault(wid, []).append(row)
        return grouped

    async def _load_metrics(
        self, monitoring_run_id: str, evidence_window_ids: list[str] | None = None
    ) -> list[dict]:
        """Load metrics and expose metric_detail fields to legacy validators."""
        mon_repo = MonitoringRepo(self.session)
        rows = await mon_repo.get_metrics(monitoring_run_id)
        normalized: list[dict] = []
        allowed = set(evidence_window_ids or [])
        for row in rows:
            detail = row.get("metric_detail") or {}
            item = {**row, **detail}
            window_id = str(
                item.get("window_id") or item.get("monitor_window_id") or ""
            )
            if allowed and window_id not in allowed:
                continue
            normalized.append(item)
        return normalized

    async def _load_feature_importance(
        self, model_id: str | None
    ) -> dict[str, float] | None:
        """Step 2d: 从模型产物中加载特征重要性。

        读取 assets/champion_models/{model_id}/champion_v1/results/feature_importance.csv
        返回 {feature_name: importance_score} 映射。
        文件不存在或 model_id 为空时返回 None。
        """
        if not model_id:
            return None

        csv_path = (
            _PROJECT_ROOT / "assets" / "champion_models"
            / model_id / "champion_v1" / "results" / "feature_importance.csv"
        )

        if not csv_path.is_file():
            return None

        try:
            importance: dict[str, float] = {}
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fn = row.get("feature", "").strip()
                    imp = float(row.get("importance", 0))
                    if fn:
                        # feature_importance.csv 中的特征名带前缀（如 numeric__, categorical__）
                        # 同时保存原始名和去前缀名，提高匹配率
                        importance[fn] = imp
                        if "__" in fn:
                            short_name = fn.split("__", 1)[1]
                            # 不去覆盖已有值，保留完整名的优先级
                            if short_name not in importance:
                                importance[short_name] = imp
            return importance
        except Exception:
            return None

    async def _execute_validation(
        self,
        candidates: list[CandidateRootCause],
        drift_data: list[dict],
        alert_details: list,
        multi_window_drift: dict[str, list[dict]] | None = None,
        metrics: list[dict] | None = None,
        feature_importance: dict[str, float] | None = None,
    ) -> dict[str, list[EvidenceItem]]:
        """Step 3: 对每个候选根因，运行全部适用的 D/R/C/T/I 验证器。

        分派逻辑：
          1. 查 ROOT_CAUSE_VALIDATORS 获取该根因的验证器列表
          2. 逐个从 EXECUTOR_REGISTRY 取出并执行
          3. 每个验证器独立返回 EvidenceItem（SUPPORT/AGAINST/NEUTRAL）
          4. 收集到 {root_cause_code: [EvidenceItem, ...]} 中

        单个验证器失败不阻塞其他验证器。
        """
        packages: dict[str, list[EvidenceItem]] = {}

        for rc in candidates:
            evidence_items: list[EvidenceItem] = []

            # 获取该根因的验证器列表
            validator_codes = ROOT_CAUSE_VALIDATORS.get(
                rc.root_cause_code, _DEFAULT_VALIDATORS
            )

            for method_code in validator_codes:
                validator_fn = EXECUTOR_REGISTRY.get(method_code)
                if validator_fn is None:
                    continue

                try:
                    # 所有验证器统一接收 drift_data + alert_code
                    # 额外数据通过 kwargs 传递（各验证器按需取用）。
                    # root_cause_code 供根因感知验证器（psi_check 等）使用：
                    # 漂移证据只能支持漂移类假设，不能 blanket SUPPORT 所有候选。
                    item = await validator_fn(
                        drift_data,
                        rc.alert_code,
                        root_cause_code=rc.root_cause_code,
                        multi_window_drift=multi_window_drift,
                        metrics=metrics,
                        feature_importance=feature_importance,
                        supporting_alert_codes=rc.supporting_alert_codes,
                        supporting_relation_keys=rc.supporting_relation_keys,
                        required_context=rc.required_context,
                        causal_distance=rc.causal_distance,
                    )
                    evidence_items.append(item)
                except Exception:
                    # 单个验证器失败不阻塞其他验证器
                    import structlog
                    logger = structlog.get_logger(__name__)
                    logger.warning(
                        "validator_failed",
                        method_code=method_code,
                        root_cause_code=rc.root_cause_code,
                        exc_info=True,
                    )

            packages[rc.root_cause_code] = evidence_items

        return packages

    async def _rank(
        self,
        candidates: list[CandidateRootCause],
        evidence_packages: dict[str, list[EvidenceItem]],
    ) -> list[tuple[CandidateRootCause, DiagnosisPath]]:
        """Step 4: PathRanker — 融合 KG 权重 + D/R/C/T/I 证据得分。

        rank_score = effective_weight × 0.6 + avg_evidence_score × 0.4

        avg_evidence_score 取所有 applicable=True 的验证器的 normalized_score 均值。
        现在每个候选有 3~5 条独立证据（D/R/C/T/I），PathRanker 综合所有维度。
        """
        ranked = []
        for rc in candidates:
            ev_items = evidence_packages.get(rc.root_cause_code, [])
            applicable = [e for e in ev_items if e.applicable]
            if applicable:
                avg_evidence = sum(
                    (e.normalized_score or 0.5) for e in applicable
                ) / len(applicable)
            else:
                avg_evidence = 0.5  # 无适用证据时给中性分

            rank_score = rc.effective_weight_snapshot * 0.6 + avg_evidence * 0.4

            ranked.append((rc, DiagnosisPath(
                diagnosis_path_id=str(uuid.uuid4()),
                rank_no=0,  # 稍后排序更新
                root_cause_code=rc.root_cause_code,
                dimension_code=DimensionCode(rc.dimension_code)
                if rc.dimension_code else DimensionCode.FEATURE,
                relation_weight_snapshot=rc.effective_weight_snapshot,
                path_score=round(rank_score, 4),
            )))

        # 平局裁决：同分时 DIRECT 因果边优先于 INDIRECT（告警本身就是该
        # 根因的直接证据，比"汇总告警间接暗示"更可信），再按权重、
        # 证据案例数递减。杜绝"召回顺序决定主因"的非确定性。
        ranked.sort(
            key=lambda x: (
                x[1].path_score,
                _distance_priority(x[0].causal_distance),
                x[0].effective_weight_snapshot,
                x[0].evidence_case_count_snapshot or 0,
            ),
            reverse=True,
        )

        # 更新 rank_no
        for i, (rc, path) in enumerate(ranked):
            path.rank_no = i + 1

        return ranked


def _distance_priority(causal_distance: str | None) -> int:
    """因果距离优先级：DIRECT=2 > INDIRECT=1 > 其他=0。"""
    return {
        "DIRECT": 2,
        "INDIRECT": 1,
        "": 0,
    }.get(causal_distance or "", 0)


def _primary_relation_key(rc: CandidateRootCause) -> tuple:
    """主关系确定性排序键：(weight, causal_distance_priority, relation_key)。

    权重相同（图谱中大量 0.10 边）时用 causal_distance 定序
    （DIRECT > INDIRECT > 其他），最后以 relation_key 兜底，
    保证主关系选择与告警遍历顺序无关。
    """
    return (
        rc.effective_weight_snapshot,
        _distance_priority(rc.causal_distance),
        rc.relation_key,
    )


def _replace_primary_relation(existing: CandidateRootCause, rc: CandidateRootCause) -> None:
    """将更高权重的关系设为主关系，同步全部主关系字段。

    注意：新增 CandidateRootCause 关系字段时必须同步更新此函数，
    否则主关系切换时会漏同步治理属性（如 required_context/causal_distance）。
    """
    existing.alert_code = rc.alert_code
    existing.relation_key = rc.relation_key
    existing.effective_weight_snapshot = rc.effective_weight_snapshot
    existing.evidence_case_count_snapshot = rc.evidence_case_count_snapshot
    existing.confidence_lower_bound_snapshot = rc.confidence_lower_bound_snapshot
    existing.dimension_code = rc.dimension_code
    existing.required_context = list(rc.required_context)
    existing.causal_distance = rc.causal_distance
    existing.supporting_only = rc.supporting_only


def _dimension_to_action(dimension: str) -> RecommendedAction:
    mapping = {
        "FEATURE": RecommendedAction.MODEL_ITERATION,
        "MODEL": RecommendedAction.MODEL_ITERATION,
        "DATA": RecommendedAction.DATA_REPAIR,
        "BUSINESS": RecommendedAction.CONTINUE_OBSERVATION,
    }
    return mapping.get(dimension, RecommendedAction.MANUAL_REVIEW)


def _infer_drift_context(
    drift_data: list[dict] | None,
    multi_window_drift: dict | None,
    root_cause_code: str,
) -> tuple[str | None, str | None]:
    """A7 §4/§5: 从真实漂移证据推导 L1 结构化上下文。

    - impact_scope: 显著漂移特征占比 ≥40% → GLOBAL，否则 LOCAL
    - change_pattern: 最近窗口平均 PSI ≥ 前序窗口均值 ×1.5 → SUDDEN，否则 GRADUAL
    仅对漂移类根因（FEATURE_DRIFT / CONCEPT_DRIFT）有意义，其余返回 (None, None)。
    无真实漂移数据时诚实返回 None，不伪造上下文。
    """
    rc_code = (root_cause_code or "").upper().replace("-", "_")
    if rc_code not in {"FEATURE_DRIFT", "CONCEPT_DRIFT"}:
        return None, None

    rows = [r for r in (drift_data or []) if isinstance(r, dict)]
    if not rows:
        return None, None

    def _psi(row: dict) -> float:
        try:
            return float(row.get("psi") or 0)
        except (TypeError, ValueError):
            return 0.0

    # ── impact_scope：漂移特征占比 ──
    feature_psi: dict[str, float] = {}
    for row in rows:
        fname = row.get("feature_name")
        if not fname:
            continue
        feature_psi[str(fname)] = max(feature_psi.get(str(fname), 0.0), _psi(row))
    if feature_psi:
        drifted_count = sum(1 for v in feature_psi.values() if v >= 0.1)
        scope = (
            "GLOBAL" if drifted_count / len(feature_psi) >= 0.4 else "LOCAL"
        )
    else:
        scope = None

    # ── change_pattern：最近窗口 vs 前序窗口均值 ──
    if multi_window_drift and len(multi_window_drift) >= 2:
        window_means: dict[str, float] = {}
        for wid, wrows in multi_window_drift.items():
            values = [_psi(r) for r in (wrows or []) if isinstance(r, dict)]
            if values:
                window_means[str(wid)] = sum(values) / len(values)
        if len(window_means) >= 2:
            ordered = sorted(window_means.keys())
            last_mean = window_means[ordered[-1]]
            prev_means = [
                mean for wid, mean in window_means.items() if wid != ordered[-1]
            ]
            prev_mean = sum(prev_means) / len(prev_means)
            if prev_mean > 0 and last_mean >= prev_mean * 1.5:
                pattern = "SUDDEN"
            else:
                pattern = "GRADUAL"
        else:
            pattern = None
    else:
        pattern = None

    return scope, pattern


def _infer_segment_evidence(
    model_id: str | None,
    drift_data: list[dict] | None,
) -> dict | None:
    """A7 §4: 从真实窗口数据生成"受损且合格的冻结客群"定义。

    业务语义（不只"结构变化客群"）：
    1. W3 客群占比显著增加（delta >= 0.05，只选占比增加的类别——
       纯 40/60→50/50 的整体迁移不能把两个类别都选出来统一加权）
    2. 该客群在 W3 相对 W0 存在真实退化（AUC 下降或坏样本率上升）
    3. 样本数与正负类别数量满足最低门槛

    无满足全部条件的客群时返回 None（下游 fail-closed，不伪造客群）。
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from apps.modelops_api.services.monitoring.window_loader import (
        load_window_with_predictions,
    )

    if not model_id:
        return None
    try:
        w0_df = load_window_with_predictions("W0", model_id=model_id)
        w3_df = load_window_with_predictions("W3", model_id=model_id)
    except Exception:
        return None

    categorical_features = {
        str(row.get("feature_name"))
        for row in (drift_data or [])
        if isinstance(row, dict)
        and str(row.get("feature_type") or "") in {"categorical", "CATEGORICAL"}
    }
    if not categorical_features:
        return None

    def _segment_metrics(df: pd.DataFrame, feature: str, value: str) -> dict:
        mask = (
            df[feature].astype("string").fillna("__MISSING__") == value
        ).to_numpy()
        sub = df[mask]
        metrics = {
            "count": int(mask.sum()),
            "bad": int(sub["is_bad"].sum()) if "is_bad" in sub else 0,
        }
        if (
            metrics["count"] >= 50
            and metrics["bad"] >= 1
            and metrics["count"] - metrics["bad"] >= 1
            and "y_pred_proba" in sub and "is_bad" in sub
        ):
            try:
                metrics["auc"] = float(
                    roc_auc_score(sub["is_bad"], sub["y_pred_proba"])
                )
            except Exception:
                metrics["auc"] = None
        else:
            metrics["auc"] = None
        metrics["bad_rate"] = (
            metrics["bad"] / metrics["count"] if metrics["count"] else None
        )
        return metrics

    best_column: str | None = None
    best_affected: list[str] = []
    best_score = 0.0
    for feature in categorical_features:
        if feature not in w0_df.columns or feature not in w3_df.columns:
            continue
        ref = w0_df[feature].astype("string").fillna("__MISSING__")
        cur = w3_df[feature].astype("string").fillna("__MISSING__")
        universe = sorted(set(ref.unique()) | set(cur.unique()))
        if len(universe) < 2 or len(universe) > 50:
            continue
        p = ref.value_counts(normalize=True)
        q = cur.value_counts(normalize=True)
        deltas = {v: float(q.get(v, 0.0) - p.get(v, 0.0)) for v in universe}
        affected: list[str] = []
        for value in universe:
            if deltas[value] < 0.05:
                continue  # 只选占比增加的客群
            w0_metrics = _segment_metrics(w0_df, feature, value)
            w3_metrics = _segment_metrics(w3_df, feature, value)
            # 最低样本/正负样本门槛
            if (
                w3_metrics["count"] < 50
                or w3_metrics["bad"] < 1
                or w3_metrics["count"] - w3_metrics["bad"] < 1
            ):
                continue
            # 退化证据：AUC 下降或坏样本率上升
            degraded = False
            if w0_metrics["auc"] is not None and w3_metrics["auc"] is not None:
                degraded = w3_metrics["auc"] <= w0_metrics["auc"] - 0.02
            if (
                not degraded
                and w0_metrics["bad_rate"] is not None
                and w3_metrics["bad_rate"] is not None
            ):
                degraded = w3_metrics["bad_rate"] >= w0_metrics["bad_rate"] + 0.02
            if not degraded:
                continue  # 占比增加但未退化 → 不是"受损客群"
            affected.append(value)
        score = sum(abs(d) for d in deltas.values())
        if affected and score > best_score:
            best_column, best_affected, best_score = feature, affected, score

    if not best_column or not best_affected:
        return None
    return {
        "segment_column": best_column,
        "affected_segments": best_affected,
        "segment_boost": 3.0,
        "evidence_source": "W0_VS_W3_CATEGORICAL_SHARE_DELTA_AND_DEGRADATION",
    }
