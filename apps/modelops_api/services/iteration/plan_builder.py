"""审核通过后构造不可越界的训练计划。

双路径：
- 严格 A7 路径（proposal.training_window_ids 非空，来自 select_a7_strategy/propose_a7）：
  执行 Champion 身份真实性校验（模型/manifest/schema 文件 + checksum + 算法族）、
  W3 确定性时间切分、W4 防泄漏与快照校验和（包版严格合同）。
- 自然链路路径（decide_with_kg 产出，无严格窗口字段）：
  保持本地既有构建逻辑（配置窗口 + training_mode 正式传递 + 特征筛选合同），
  并尽力从 Champion bundle 富集 ordered_features/algorithm_family 等身份字段
  （文件缺失不阻断——自然链路身份由 Worker 的 Champion 加载校验兜底）。
"""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd

from packages.models.common.enums import (
    ProposalStatus,
    RecommendedAction,
    TrainingPlanStatus,
)
from packages.models.iteration.decision_proposal import DecisionProposal
from packages.models.iteration.risk_assessment import RiskAssessment
from packages.models.iteration.training_plan import (
    TrainingPlan,
    TrainingWindowSpec,
    WindowTimeRange,
)

from .config_loader import IterationConfigBundle, load_iteration_config


# Champion 算法家族 → 训练引擎 slug 映射
_ALGORITHM_SLUG = {
    "LogisticRegression": "logistic_regression",
    "RandomForest": "random_forest",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "CatBoost": "catboost",
    "EBM": "ebm",
}


def _sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256（1MB 分块）。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _json_hash(payload: object) -> str:
    """规范化 JSON 序列化后取 SHA-256（稳定排序，同构对象哈希一致）。"""
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


class TrainingPlanBuilder:
    def __init__(self, config: IterationConfigBundle | None = None) -> None:
        self.config = config or load_iteration_config()
        self.project_root = Path(__file__).resolve().parents[4]

    # ═══════════════════════════════════════════════════════════════
    #  Champion 身份与窗口真实性校验（严格 A7 路径）
    # ═══════════════════════════════════════════════════════════════

    def _champion_identity(
        self,
        proposal: DecisionProposal,
        *,
        strict: bool,
    ) -> dict:
        """校验 Champion 身份包（model/manifest/schema）。

        strict=True（严格 A7 路径）：文件缺失、算法族不支持、model_id 不一致、
        checksum 不匹配、特征 schema 非法均抛错。
        strict=False（自然链路）：尽力富集身份字段；文件缺失返回空字典不阻断。
        """
        bundle = (
            self.project_root / "assets" / "champion_models"
            / proposal.model_id / proposal.champion_version
        )
        required = {
            "model": bundle / "model.joblib",
            "manifest": bundle / "training_manifest.json",
            "schema": bundle / "feature_schema.json",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            if strict:
                raise ValueError(f"CHAMPION_IDENTITY_INCOMPLETE:{','.join(missing)}")
            return {}

        manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
        schema = json.loads(required["schema"].read_text(encoding="utf-8"))
        family = manifest.get("algorithm_family")
        if family not in _ALGORITHM_SLUG:
            if strict:
                raise ValueError(f"UNSUPPORTED_CHAMPION_ALGORITHM:{family}")
            return {}
        if manifest.get("model_id") != proposal.model_id:
            raise ValueError("CHAMPION_MANIFEST_MODEL_ID_MISMATCH")
        ordered = schema.get("ordered_features") or []
        if not ordered or len(ordered) != len(set(ordered)):
            raise ValueError("CHAMPION_FEATURE_SCHEMA_INVALID")
        checksum = _sha256_file(required["model"])
        if strict and proposal.champion_artifact_checksum:
            if proposal.champion_artifact_checksum != checksum:
                raise ValueError("CHAMPION_ARTIFACT_CHECKSUM_MISMATCH")
        if strict and proposal.algorithm_family and proposal.algorithm_family != family:
            raise ValueError("ALGORITHM_FAMILY_MISMATCH")
        return {
            "algorithm_family": family,
            "algorithm": _ALGORITHM_SLUG[family],
            "champion_artifact_checksum": checksum,
            "feature_schema_version": str(schema.get("schema_version") or ""),
            "ordered_features": ordered,
            "ordered_features_hash": _json_hash(ordered),
            "preprocessing_version": str(manifest.get("feature_strategy_id") or ""),
            "preprocessing_hash": _json_hash(
                {
                    "feature_strategy_id": manifest.get("feature_strategy_id"),
                    "algorithm_family": family,
                }
            ),
            "hyperparameters": manifest.get("selected_parameters") or {},
            "random_seed": int(manifest.get("random_seed")),
        }

    def _window_manifest(self) -> dict[str, dict]:
        """加载窗口清单（必须含 W0-W4，window_id 无重复）。"""
        path = (
            self.project_root / "assets" / "data" / "contracts" / "window_manifest.csv"
        )
        if not path.is_file():
            raise ValueError("WINDOW_MANIFEST_MISSING")
        frame = pd.read_csv(path)
        if frame["window_id"].duplicated().any():
            raise ValueError("WINDOW_MANIFEST_DUPLICATE_ID")
        rows = {str(row["window_id"]): row.to_dict() for _, row in frame.iterrows()}
        for required in ("W0", "W1", "W2", "W3", "W4"):
            if required not in rows:
                raise ValueError(f"WINDOW_MANIFEST_ROW_MISSING:{required}")
        return rows

    @staticmethod
    def _base_window_id(window_id: str) -> str:
        if window_id in {"W3_TRAIN_SPLIT", "W3_VALIDATION_SPLIT"}:
            return "W3"
        return window_id

    def _build_windows(
        self, proposal: DecisionProposal
    ) -> tuple[TrainingWindowSpec, dict[str, str]]:
        """严格 A7 路径：W3 确定性时间切分 + W4 防泄漏 + 快照校验和。"""
        if not proposal.training_window_ids or not proposal.validation_window_ids:
            raise ValueError("L1_WINDOWS_MISSING")
        if proposal.validation_window_ids != ["W3_VALIDATION_SPLIT"]:
            raise ValueError("VALIDATION_MUST_USE_W3_VALIDATION_SPLIT")
        if "W3_TRAIN_SPLIT" not in proposal.training_window_ids:
            raise ValueError("TRAINING_MUST_USE_W3_TRAIN_SPLIT")
        manifest = self._window_manifest()
        w3 = manifest["W3"]
        w3_start = pd.Timestamp(w3["start_date"]).to_pydatetime()
        w3_end = pd.Timestamp(w3["end_date"]).to_pydatetime()
        split = w3_end - timedelta(days=self.config.iteration.w3_validation_days)
        if not w3_start < split < w3_end:
            raise ValueError("W3_SPLIT_BOUNDARY_INVALID")

        train_ranges: list[WindowTimeRange] = []
        for window_id in proposal.training_window_ids:
            base = self._base_window_id(window_id)
            if base not in manifest or base == "W4":
                raise ValueError(f"INVALID_TRAINING_WINDOW:{window_id}")
            row = manifest[base]
            start = pd.Timestamp(row["start_date"]).to_pydatetime()
            end = pd.Timestamp(row["end_date"]).to_pydatetime()
            if window_id == "W3_TRAIN_SPLIT":
                end = split
                if proposal.training_data_mode == "POST_CHANGE_ONLY":
                    if not proposal.change_point:
                        raise ValueError("POST_CHANGE_TRAINING_REQUIRES_CHANGE_POINT")
                    start = max(
                        start, pd.Timestamp(proposal.change_point).to_pydatetime()
                    )
                    if start >= end:
                        raise ValueError("POST_CHANGE_TRAINING_SAMPLE_RANGE_EMPTY")
            train_ranges.append(
                WindowTimeRange(window_id=window_id, start_at=start, end_at=end)
            )
        validation_ranges = [
            WindowTimeRange(
                window_id="W3_VALIDATION_SPLIT",
                start_at=split,
                end_at=w3_end,
            )
        ]

        w3_path = self.project_root / "assets" / str(w3["data_uri"])
        if not w3_path.is_file():
            raise ValueError("W3_DATA_MISSING")
        w3_frame = pd.read_parquet(
            w3_path, columns=["sample_id", "apply_time", "is_bad"]
        )
        if w3_frame["sample_id"].isna().any() or not w3_frame["sample_id"].is_unique:
            raise ValueError("W3_SAMPLE_ID_INVALID")
        if (
            w3_frame["is_bad"].isna().any()
            or not w3_frame["is_bad"].isin([0, 1]).all()
        ):
            raise ValueError("W3_LABEL_CONTRACT_FAILED")
        times = pd.to_datetime(w3_frame["apply_time"], errors="raise")
        w3_training_range = next(
            item for item in train_ranges if item.window_id == "W3_TRAIN_SPLIT"
        )
        train_ids = sorted(
            w3_frame.loc[
                (times >= w3_training_range.start_at)
                & (times < w3_training_range.end_at),
                "sample_id",
            ].astype(str)
        )
        valid_ids = sorted(
            w3_frame.loc[(times >= split) & (times < w3_end), "sample_id"].astype(str)
        )
        if not train_ids or not valid_ids:
            raise ValueError("W3_SPLIT_EMPTY")
        if set(train_ids).intersection(valid_ids):
            raise ValueError("W3_SPLIT_SAMPLE_OVERLAP")

        train_checksum = _json_hash(
            {"source": w3["data_checksum"], "sample_ids": train_ids}
        )
        valid_checksum = _json_hash(
            {"source": w3["data_checksum"], "sample_ids": valid_ids}
        )
        snapshot_checksums: dict[str, str] = {}
        for window_id in proposal.training_window_ids + proposal.validation_window_ids:
            key = f"window:{window_id}"
            base = self._base_window_id(window_id)
            if window_id == "W3_TRAIN_SPLIT":
                snapshot_checksums[key] = train_checksum
            elif window_id == "W3_VALIDATION_SPLIT":
                snapshot_checksums[key] = valid_checksum
            else:
                snapshot_checksums[key] = f"sha256:{manifest[base]['data_checksum']}"

        windows = TrainingWindowSpec(
            baseline_window_id="W1",
            training_window_ids=proposal.training_window_ids,
            validation_window_ids=proposal.validation_window_ids,
            training_time_ranges=train_ranges,
            validation_time_ranges=validation_ranges,
            oot_window_id="W4",
            oot_locked=True,
            w3_split_method=self.config.iteration.w3_split_rule_version,
            w3_split_boundary=split,
            w3_train_snapshot_id="window:W3_TRAIN_SPLIT",
            w3_validation_snapshot_id="window:W3_VALIDATION_SPLIT",
            w3_train_checksum=train_checksum,
            w3_validation_checksum=valid_checksum,
        )
        return windows, snapshot_checksums

    # ═══════════════════════════════════════════════════════════════
    #  build()：双路径
    # ═══════════════════════════════════════════════════════════════

    def build(
        self,
        proposal: DecisionProposal,
        risk: RiskAssessment,
        *,
        approval_id: str,
        iteration_run_id: str,
        model_algorithm: str | None = None,
        feature_schema_version: str | None = None,
        preprocessing_version: str | None = None,
        business_round: int = 1,
        data_snapshot_ids: list[str] | None = None,
        label_versions: list[str] | None = None,
        unstable_feature_codes: list[str] | None = None,
        selected_feature_codes: list[str] | None = None,
        feature_selection_artifact_uri: str | None = None,
    ) -> TrainingPlan:
        if proposal.action != RecommendedAction.MODEL_ITERATION:
            raise ValueError("only MODEL_ITERATION can produce a TrainingPlan")
        if proposal.status != ProposalStatus.APPROVED:
            raise ValueError("DecisionProposal must be approved before plan generation")
        if not approval_id:
            raise ValueError("approval_id is required")
        if not proposal.strategies:
            raise ValueError("approved model iteration has no selected strategy")
        strategy = proposal.strategies[0]
        strategy_params = strategy.parameters or {}

        # ── 路径判定：严格 A7（信封/L1 选择）vs 自然链路 ──
        strict_a7 = bool(proposal.training_window_ids)

        if strict_a7:
            # 严格 A7 前置校验（无默认值、无隐式替换）
            if proposal.selection_status != "SELECTED":
                raise ValueError("L1_SELECTION_NOT_SELECTED")
            if proposal.root_cause_status != "CONFIRMED":
                raise ValueError("ROOT_CAUSE_NOT_CONFIRMED")
            if proposal.primary_strategy is None or proposal.execution_mode is None:
                raise ValueError("L1_EXECUTION_FIELDS_MISSING")
            if len(proposal.strategies) != 1:
                raise ValueError("A7_REQUIRES_EXACTLY_ONE_STRATEGY")
            if proposal.strategies[0].strategy_code != proposal.primary_strategy:
                raise ValueError("PROPOSAL_STRATEGY_MISMATCH")
            if not proposal.authorization_id or proposal.authorization_id != approval_id:
                raise ValueError("AUTHORIZATION_ID_MISMATCH")
            if business_round not in {1, 2}:
                raise ValueError("BUSINESS_ROUND_OUT_OF_RANGE")
            identity = self._champion_identity(proposal, strict=True)
            if (
                model_algorithm is not None
                and model_algorithm.lower() != identity["algorithm"]
            ):
                raise ValueError("CALLER_ALGORITHM_MISMATCH")
            if (
                feature_schema_version is not None
                and feature_schema_version != identity["feature_schema_version"]
            ):
                raise ValueError("CALLER_FEATURE_SCHEMA_MISMATCH")
            if (
                preprocessing_version is not None
                and preprocessing_version != identity["preprocessing_version"]
            ):
                raise ValueError("CALLER_PREPROCESSING_MISMATCH")
            if data_snapshot_ids is not None:
                raise ValueError("CALLER_DATA_SNAPSHOT_OVERRIDE_FORBIDDEN")
            if label_versions is not None and label_versions != [
                "is_bad@window_manifest:1.0"
            ]:
                raise ValueError("CALLER_LABEL_VERSION_MISMATCH")

            windows, snapshot_checksums = self._build_windows(proposal)
            resolved_snapshot_ids = list(snapshot_checksums)
            resolved_label_versions = ["is_bad@window_manifest:1.0"]
            resolved_feature_schema = identity["feature_schema_version"]
            resolved_preprocessing = identity["preprocessing_version"]
            resolved_algorithm = identity["algorithm"]
            resolved_hyperparameters = identity["hyperparameters"]
            resolved_random_seed = identity["random_seed"]
        else:
            # 自然链路：本地既有逻辑 + 身份字段尽力富集
            identity = self._champion_identity(proposal, strict=False)
            resolved_label_versions = (
                label_versions
                or strategy_params.get("label_versions")
                or ["label-v1"]
            )
            if not resolved_label_versions:
                raise ValueError("observed label versions are required")
            windows = TrainingWindowSpec(
                baseline_window_id=self.config.iteration.baseline_window_id,
                training_window_ids=(
                    strategy_params.get("training_window_ids")
                    or strategy_params.get("allowed_training_window_ids")
                    or self.config.iteration.default_training_window_ids
                ),
                validation_window_ids=(
                    strategy_params.get("validation_window_ids")
                    or self.config.iteration.default_validation_window_ids
                ),
                oot_window_id=self.config.iteration.oot_window_id,
                oot_locked=True,
            )
            # 数据快照由窗口清单派生（自然链路无特征重构快照时，
            # Worker 按窗口 ID 直接加载窗口数据训练）
            snapshot_ids = data_snapshot_ids or [
                f"window:{wid}"
                for wid in windows.training_window_ids + windows.validation_window_ids
            ]
            snapshot_checksums = {}
            resolved_snapshot_ids = snapshot_ids
            resolved_feature_schema = (
                feature_schema_version
                or identity.get("feature_schema_version")
                or strategy_params.get("feature_schema_version")
                or "feature-schema-v1"
            )
            resolved_preprocessing = (
                preprocessing_version
                or identity.get("preprocessing_version")
                or strategy_params.get("preprocessing_version")
                or "preprocess-v1"
            )
            resolved_algorithm = (
                model_algorithm
                or identity.get("algorithm")
                or strategy_params.get("algorithm")
                or "lightgbm"
            )
            resolved_hyperparameters = strategy_params.get("hyperparameters", {})
            resolved_random_seed = 2026

        return TrainingPlan(
            training_plan_id=str(uuid4()),
            proposal_id=proposal.proposal_id,
            approval_id=approval_id,
            iteration_run_id=iteration_run_id,
            experiment_id=str(uuid4()),
            business_round=business_round,
            diagnosis_run_id=proposal.diagnosis_run_id,
            model_id=proposal.model_id,
            frozen_champion_version=proposal.champion_version,
            root_cause_code=proposal.primary_root_cause_code,
            strategy_code=strategy.strategy_code,
            strategy_parameters=strategy.parameters,
            # A7 §7: 训练模式从正式合同字段传递，不从 strategy_tier 猜测
            training_mode=strategy.primary_training_mode,
            # A7 阶段四：特征筛选合同（FEATURE_SELECTION 模式）
            unstable_feature_codes=unstable_feature_codes or [],
            selected_feature_codes=selected_feature_codes or [],
            feature_selection_artifact_uri=feature_selection_artifact_uri,
            target_metric_codes=proposal.target_metric_codes,
            windows=windows,
            data_snapshot_ids=resolved_snapshot_ids,
            data_snapshot_checksums=snapshot_checksums,
            label_versions=resolved_label_versions,
            sample_weight_policy=(
                proposal.sample_weight_policy
                or strategy_params.get("sample_weight_policy", {})
            ),
            sample_weight_required=proposal.sample_weight_required,
            feature_schema_version=resolved_feature_schema,
            ordered_features=identity.get("ordered_features", []),
            ordered_features_hash=identity.get("ordered_features_hash"),
            preprocessing_version=resolved_preprocessing,
            preprocessing_hash=identity.get("preprocessing_hash"),
            algorithm=resolved_algorithm,
            algorithm_family=identity.get("algorithm_family"),
            same_algorithm_family=bool(identity.get("algorithm_family")),
            champion_artifact_checksum=identity.get("champion_artifact_checksum"),
            hyperparameter_space=resolved_hyperparameters,
            random_seed=resolved_random_seed,
            risk_level=risk.risk_level.value,
            max_business_rounds=self.config.iteration.max_iteration_rounds,
            rollback_target=proposal.champion_version,
            status=TrainingPlanStatus.READY,
            blocking_reasons=[],
            rule_version=self.config.iteration.rule_version,
            # ── 严格 A7 传递/审计字段 ──
            lifecycle_run_id=proposal.lifecycle_run_id,
            event_id=proposal.event_id,
            monitoring_run_id=proposal.monitoring_run_id,
            agent_decision_id=proposal.agent_decision_id,
            decision_source=proposal.decision_source,
            root_cause_status=proposal.root_cause_status,
            decay_degree=proposal.decay_degree,
            impact_scope=proposal.impact_scope,
            change_pattern=proposal.change_pattern,
            change_point=proposal.change_point,
            affected_segments=proposal.affected_segments,
            primary_strategy=proposal.primary_strategy or strategy.strategy_code,
            execution_mode=proposal.execution_mode,
            training_data_mode=proposal.training_data_mode,
            strategy_source=proposal.strategy_source,
            kg_consistency_status=proposal.kg_consistency_status,
            kg_repair_required=proposal.kg_repair_required,
            selection_reason_codes=proposal.selection_reason_codes,
            authorization_type=proposal.authorization_type,
            authorization_id=proposal.authorization_id,
            l1_matrix_version=proposal.rule_versions.get("l1_matrix_version"),
            window_rule_version=self.config.iteration.w3_split_rule_version,
            threshold_status=self.config.qualification.threshold_status,
        )
