"""A7 阶段四：特征筛选服务。

基于三路真实证据生成冻结特征清单：
1. 稳定性：W3 失败归因确认的 unstable_feature_codes（不简单删除所有漂移特征，
   只剔除经归因确认的）
2. 重要性：特征重要性过低的特征
3. 共线性：皮尔逊相关系数 > 阈值的高共线特征（保留更重要的一个）

产出：selected_feature_codes（新 schema 快照使用的特征顺序）
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class FeatureSelectionResult:
    selection_report_id: str
    input_feature_codes: list[str]
    selected_feature_codes: list[str]
    dropped_feature_codes: list[str] = field(default_factory=list)
    drop_reasons: dict[str, str] = field(default_factory=dict)
    feature_selection_artifact_uri: str | None = None
    report_json: dict = field(default_factory=dict)


def select_features(
    feature_codes: list[str],
    *,
    unstable_feature_codes: list[str] | None = None,
    feature_importance: dict[str, float] | None = None,
    correlation_matrix: dict[str, dict[str, float]] | None = None,
    min_importance: float = 0.005,
    correlation_threshold: float = 0.90,
) -> FeatureSelectionResult:
    """生成冻结特征清单（纯函数，便于单测）。

    - 不稳定特征只剔除"经归因确认"的（unstable_feature_codes 显式列表）
    - 低重要性特征剔除（feature_importance 低于 min_importance）
    - 高共线性特征对中保留重要性更高者
    """
    unstable = {str(c) for c in (unstable_feature_codes or [])}
    importance = {str(k): float(v) for k, v in (feature_importance or {}).items()}
    corr = correlation_matrix or {}

    dropped: dict[str, str] = {}
    selected: list[str] = []

    for code in feature_codes:
        code = str(code)
        if code in unstable:
            dropped[code] = "UNSTABLE_ATTRIBUTED"
            continue
        imp = importance.get(code)
        if imp is not None and imp < min_importance:
            dropped[code] = f"LOW_IMPORTANCE:{imp:.4f}"
            continue
        selected.append(code)

    # 共线性剪枝：按特征顺序，与已保留特征 |相关| > 阈值时保留重要性更高者
    #（负相关同样代表冗余，-0.95 必须被识别）
    pruned: list[str] = []
    for code in selected:
        for kept in pruned:
            pair = (corr.get(code) or {}).get(kept) or (corr.get(kept) or {}).get(code)
            if pair is None:
                continue
            if abs(float(pair)) >= correlation_threshold:
                drop, keep = (
                    (code, kept)
                    if (importance.get(code) or 0) < (importance.get(kept) or 0)
                    else (kept, code)
                )
                if drop == code:
                    dropped[code] = f"HIGH_COLLINEARITY_WITH:{keep}"
                else:
                    # 已保留的特征被替换为更重要者
                    pruned.remove(keep)
                    dropped[keep] = f"HIGH_COLLINEARITY_WITH:{code}"
                    pruned.append(code)
                break
        else:
            pruned.append(code)
    selected = pruned

    if not selected:
        # 兜底：全部被剔除时不产出空特征集（阻止训练无法执行）
        raise ValueError("feature selection would drop all features")

    report = {
        "input_feature_codes": list(feature_codes),
        "selected_feature_codes": selected,
        "dropped_feature_codes": [c for c in feature_codes if c not in selected],
        "drop_reasons": dropped,
        "unstable_feature_codes": sorted(unstable),
    }
    return FeatureSelectionResult(
        selection_report_id=str(uuid.uuid4()),
        input_feature_codes=list(feature_codes),
        selected_feature_codes=selected,
        dropped_feature_codes=[c for c in feature_codes if c not in selected],
        drop_reasons=dropped,
        report_json=report,
    )


def serialize_selection(result: FeatureSelectionResult) -> str:
    """序列化筛选报告为 JSON 字符串（Neo4j/MinIO 存储用）。"""
    return json.dumps(result.report_json, ensure_ascii=False)
