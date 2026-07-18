"""
稳定枚举定义
来源：技术开发文档 V1.4.2 附录 A + 全文各章节
"""

from enum import Enum


# ── 数据轨道 ──
class DataTrack(str, Enum):
    NATURAL = "NATURAL"
    SCENARIO = "SCENARIO"


# ── 严重度 ──
class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ── 规则类型 ──
class RuleType(str, Enum):
    DROP_THRESHOLD = "DROP_THRESHOLD"
    UPPER_THRESHOLD = "UPPER_THRESHOLD"
    LOWER_THRESHOLD = "LOWER_THRESHOLD"
    SHIFT_THRESHOLD = "SHIFT_THRESHOLD"


# ── 指标方向 ──
class MetricDirection(str, Enum):
    HIGHER_BETTER = "HIGHER_BETTER"
    LOWER_BETTER = "LOWER_BETTER"
    DEVIATION_BAD = "DEVIATION_BAD"


# ── 证据方向 ──
class EvidenceDirection(str, Enum):
    SUPPORT = "SUPPORT"
    AGAINST = "AGAINST"
    NEUTRAL = "NEUTRAL"


# ── 置信度 ──
class ConfidenceLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ── 可用性状态 ──
class AvailabilityStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    LABEL_NOT_MATURE = "LABEL_NOT_MATURE"
    DATA_NOT_AVAILABLE = "DATA_NOT_AVAILABLE"
    SAMPLE_TOO_SMALL = "SAMPLE_TOO_SMALL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    CALCULATION_FAILED = "CALCULATION_FAILED"


# ── 告警对象类型 ──
class ObjectType(str, Enum):
    FEATURE = "FEATURE"
    MODEL = "MODEL"
    SEGMENT = "SEGMENT"
    BUSINESS = "BUSINESS"


# ── 四维维度 ──
class DimensionCode(str, Enum):
    DATA = "DATA"
    FEATURE = "FEATURE"
    MODEL = "MODEL"
    BUSINESS = "BUSINESS"


# ── D/R/C/T/I 证据类型 ──
class EvidenceType(str, Enum):
    D = "D"  # Distribution / Data      — 分布、数据质量和直接异常事实
    R = "R"  # Repair / Recovery        — 反事实修复后性能是否恢复
    C = "C"  # Conditional / Causal     — 控制变量后根因与性能的关联
    T = "T"  # Temporal                 — 原因是否先于症状出现
    I = "I"  # Importance / Dependency  — 模型是否依赖该特征或机制


# ── 任务二推荐动作 ──
class RecommendedAction(str, Enum):
    MODEL_ITERATION = "MODEL_ITERATION"
    DATA_REPAIR = "DATA_REPAIR"
    PIPELINE_REPAIR = "PIPELINE_REPAIR"
    THRESHOLD_ADJUSTMENT = "THRESHOLD_ADJUSTMENT"
    CONTINUE_OBSERVATION = "CONTINUE_OBSERVATION"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    NO_ACTION = "NO_ACTION"


# ── 部署决策 ──
class DeploymentDecision(str, Enum):
    PROMOTE = "PROMOTE"
    ADVANCE_STAGE = "ADVANCE_STAGE"
    HOLD = "HOLD"
    PAUSE_CANARY = "PAUSE_CANARY"
    REDUCE_TRAFFIC = "REDUCE_TRAFFIC"
    ROLLBACK = "ROLLBACK"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    ABORT_DEPLOYMENT = "ABORT_DEPLOYMENT"


# ── 部署阶段 ──
class DeploymentStage(str, Enum):
    OFFLINE_VALIDATION = "OFFLINE_VALIDATION"
    OOT_GATE = "OOT_GATE"
    SHADOW = "SHADOW"
    CANARY_5 = "CANARY_5"
    CANARY_20 = "CANARY_20"
    CANARY_50 = "CANARY_50"
    PRODUCTION = "PRODUCTION"


# ── 触发类型 ──
class TriggerType(str, Enum):
    SCHEDULED_TRIGGER = "SCHEDULED_TRIGGER"
    THRESHOLD_TRIGGER = "THRESHOLD_TRIGGER"
    ABNORMAL_TRIGGER = "ABNORMAL_TRIGGER"
    MANUAL_TRIGGER = "MANUAL_TRIGGER"
    DEPLOYMENT_FAILURE_ANALYSIS = "DEPLOYMENT_FAILURE_ANALYSIS"


# ── 生命周期阶段 ──
class LifecyclePhase(str, Enum):
    CREATED = "CREATED"
    MONITORING = "MONITORING"
    MONITORING_COMPLETED = "MONITORING_COMPLETED"
    NO_ALERT = "NO_ALERT"
    DIAGNOSING = "DIAGNOSING"
    DIAGNOSIS_COMPLETED = "DIAGNOSIS_COMPLETED"
    ITERATING = "ITERATING"
    CHALLENGER_TRAINED = "CHALLENGER_TRAINED"
    OFFLINE_VALIDATING = "OFFLINE_VALIDATING"
    OOT_VALIDATING = "OOT_VALIDATING"
    SHADOW_RUNNING = "SHADOW_RUNNING"
    CANARY_RUNNING = "CANARY_RUNNING"
    PROMOTED = "PROMOTED"
    ROLLED_BACK = "ROLLED_BACK"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


# ── Worker/任务状态 ──
class WorkerStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    RETRYING = "RETRYING"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    DEAD_LETTER = "DEAD_LETTER"
    LOST = "LOST"


# ── Qdrant 同步状态 ──
class SyncStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    RETRYING = "RETRYING"
    SUCCEEDED = "SUCCEEDED"
    DEAD_LETTER = "DEAD_LETTER"
    CANCELLED = "CANCELLED"


# ── 验证步骤状态 ──
class ValidationStepStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


# ── Query Profile ──
class QueryProfileCode(str, Enum):
    EXPLORATION = "exploration"
    PRODUCTION_MONITORING = "production_monitoring"
    PRODUCTION_DIAGNOSIS = "production_diagnosis"
    AUTOMATIC_ITERATION_STRATEGY = "automatic_iteration_strategy"
    PRODUCTION_DEPLOYMENT_ADVICE = "production_deployment_advice"


# ── 向量同步事件类型 ──
class VectorSyncEventType(str, Enum):
    UPSERT_CHUNK = "UPSERT_CHUNK"
    DELETE_CHUNK = "DELETE_CHUNK"
    DELETE_DOCUMENT_VERSION = "DELETE_DOCUMENT_VERSION"
    REINDEX_DOCUMENT_VERSION = "REINDEX_DOCUMENT_VERSION"
    REBUILD_COLLECTION = "REBUILD_COLLECTION"
