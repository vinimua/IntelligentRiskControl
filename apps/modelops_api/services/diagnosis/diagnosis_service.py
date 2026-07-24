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
    "label_distribution_shift": [
        "psi_check",
        "drift_group_regression",
        "temporal_precedence_check",
        "permutation_importance_check",
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
}

# ── 默认验证器集合（兜底：未知根因类型时跑 D 类型的两个基础验证器）──
_DEFAULT_VALIDATORS = ["psi_check", "missing_outlier_range_check"]


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

        # ── 0. 加载上下文数据（model_id, feature importance, multi-window drift, metrics）──
        mon_repo = MonitoringRepo(self.session)
        run = await mon_repo.get_run(monitoring_run_id)
        model_id = run["model_id"] if run else None

        # ── 1. 候选召回 ──
        candidates = await self._recall_candidates(alert_details)

        # ── 2. 加载证据（三步并行加载）──
        drift_data = await self._load_drift_data(monitoring_run_id)
        multi_window_drift = await self._load_multi_window_drift(monitoring_run_id)
        metrics = await self._load_metrics(monitoring_run_id)
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
            monitoring_run_id, lifecycle_run_id, len(alert_details),
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
        await self.repo.batch_insert_candidates(diag_id, candidate_records)

        # ── 5b. 持久化证据 ──
        for rc_root_cause, ev_items in evidence_packages.items():
            for ev_item in ev_items:
                await self.repo.insert_evidence({
                    "diagnosis_run_id": diag_id,
                    "candidate_id": None,  # 证据先不绑具体候选，后续可按 method_code 关联
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
            )
        else:
            await self.repo.complete_run(diag_id, status="NO_CANDIDATES")
            return DiagnosisStateOutput(
                diagnosis_run_id=diag_id,
                primary_root_cause_code="uncertain",
                primary_root_cause_dimension=DimensionCode.FEATURE,
                primary_root_cause_score=0.0,
                recommended_action=RecommendedAction.MANUAL_REVIEW,
                need_iteration=None,
            )

    # ═══════════════════════════════════════════════════════════════
    #  内部方法
    # ═══════════════════════════════════════════════════════════════

    async def _recall_candidates(
        self, alert_details: list
    ) -> list[CandidateRootCause]:
        """Step 1: 从 Neo4j 召回候选根因。"""
        seen = set()
        candidates: list[CandidateRootCause] = []
        for alert in alert_details:
            code = getattr(alert, "alert_code", None)
            if not code:
                continue
            for rc in await self.knowledge.query_candidate_root_causes(code):
                if rc.root_cause_code not in seen:
                    seen.add(rc.root_cause_code)
                    candidates.append(rc)
        return candidates

    async def _load_drift_data(self, monitoring_run_id: str) -> list[dict]:
        """Step 2a: 从 PostgreSQL 加载 per-feature drift 数据（默认 W3 窗口）。"""
        mon_repo = MonitoringRepo(self.session)
        return await mon_repo.get_feature_drift_by_run(monitoring_run_id, window_id="W3")

    async def _load_multi_window_drift(
        self, monitoring_run_id: str
    ) -> dict[str, list[dict]]:
        """Step 2b: 加载所有窗口的 drift 数据，按 window_id 分组。

        Returns:
            {"W1": [34 rows], "W3": [34 rows], "W6": [34 rows]}
        """
        mon_repo = MonitoringRepo(self.session)
        all_rows = await mon_repo.get_feature_drift_by_run(monitoring_run_id)
        grouped: dict[str, list[dict]] = {}
        for row in all_rows:
            wid = row.get("window_id", "?")
            grouped.setdefault(wid, []).append(row)
        return grouped

    async def _load_metrics(self, monitoring_run_id: str) -> list[dict]:
        """Step 2c: 加载该次运行的指标数据。"""
        mon_repo = MonitoringRepo(self.session)
        return await mon_repo.get_metrics(monitoring_run_id)

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
                    # 额外数据通过 kwargs 传递（各验证器按需取用）
                    item = await validator_fn(
                        drift_data,
                        rc.alert_code,
                        multi_window_drift=multi_window_drift,
                        metrics=metrics,
                        feature_importance=feature_importance,
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

        ranked.sort(key=lambda x: x[1].path_score, reverse=True)

        # 更新 rank_no
        for i, (rc, path) in enumerate(ranked):
            path.rank_no = i + 1

        return ranked


def _dimension_to_action(dimension: str) -> RecommendedAction:
    mapping = {
        "FEATURE": RecommendedAction.MODEL_ITERATION,
        "MODEL": RecommendedAction.MODEL_ITERATION,
        "DATA": RecommendedAction.DATA_REPAIR,
        "BUSINESS": RecommendedAction.CONTINUE_OBSERVATION,
    }
    return mapping.get(dimension, RecommendedAction.MANUAL_REVIEW)
