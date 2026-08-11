"use client";

/* ── 监控判定台共享类型 ── */

export type MonitoringRun = {
  monitoring_run_id: string;
  model_id?: string;
  champion_version?: string;
  overall_status?: string;
  alert_count?: number;
  max_alert_severity?: string | null;
  diagnosis_status?: string | null;
  started_at?: string;
};

export type EnrichedMetric = {
  metric_code: string;
  display_name: string;
  category: "performance" | "drift" | "quality" | "stability";
  baseline_value: number | null;
  current_value: number | null;
  delta: number | null;
  direction: "higher_better" | "lower_better" | "deviation_bad" | null;
  availability_status: string;
  rule_enabled: boolean;
  warning_threshold: number | null;
  critical_threshold: number | null;
  triggered: boolean;
  severity: string | null;
  threshold_usage_ratio: number | null;
  status_reason: string;
  metric_detail: Record<string, unknown> | null;
};

export type CategoryBreakdown = {
  total: number;
  normal: number;
  warning: number;
  critical: number;
  unmonitored: number;
  unavailable: number;
};

export type UnmonitoredMetric = {
  metric_code: string;
  display_name: string;
};

export type CoverageSummary = {
  total_metrics: number;
  calculated: number;
  available: number;
  rules_enabled: number;
  triggered: number;
  category_breakdown: Record<string, CategoryBreakdown>;
  closest_thresholds: Array<{
    metric_code: string;
    display_name: string;
    usage_ratio: number | null;
  }>;
  label_maturity: Record<string, unknown>;
  unmonitored_metrics: UnmonitoredMetric[];
};

export type FeatureDriftItem = {
  feature_name: string;
  psi_7d: number | null;
  psi_30d: number | null;
  max_psi: number;
  threshold: number;
  status: "normal" | "warning" | "critical";
  model_importance: string | null;
  trend: "up" | "down" | "stable";
  js_divergence?: number | null;
  wasserstein_distance?: number | null;
  ks_statistic?: number | null;
  missing_rate?: number | null;
  missing_rate_delta?: number | null;
  outlier_rate?: number | null;
  dq_score?: number | null;
  dq_flag?: string;
};

export type DataQualityField = {
  field_name: string;
  baseline_missing_rate: number | null;
  current_missing_rate: number | null;
  missing_delta: number | null;
  outlier_rate: number | null;
  outlier_delta: number | null;
  dq_flag: string;
};

export type SchemaChange = {
  change_type: "added" | "removed" | "type_changed";
  column_name: string;
  detail: string;
};

export type DataQualityData = {
  overall_missing_rate: number | null;
  overall_outlier_rate: number | null;
  dq_score: number | null;
  fields: DataQualityField[];
  schema_changes: SchemaChange[];
};

export type MonitoringStatus = "healthy" | "at_risk" | "critical" | "incomplete";

export type AlertItem = {
  alert_id?: string;
  alert_code?: string;
  severity?: string;
  source?: string;
  metric_code?: string;
  metric_version?: string;
  baseline_value?: number | null;
  current_value?: number | null;
  delta?: number | null;
  threshold?: number | null;
  rule_type?: string;
  alert_detail?: Record<string, unknown> | null;
  metric_detail?: Record<string, unknown> | null;
  created_at?: string;
};

/* ── 类别映射 ── */

export type EnrichedMetricsResponse = {
  metrics: EnrichedMetric[];
  summary: CoverageSummary | null;
  persistence: PersistenceJudgment | null;
  diagnosis_status: string | null;
};

/* ── B1 持续性判定 ── */

export type PersistenceJudgment = {
  trigger_diagnosis: boolean;
  decay_degree: "SHORT_TERM_7D" | "SUSTAINED_30D" | "SEVERE" | "NONE";
  requires_manual_review: boolean;
  status_7d: "NORMAL" | "OBSERVING" | "TRIGGERED";
  status_30d: "NORMAL" | "OBSERVING" | "TRIGGERED";
  persistence_evidence: Array<{
    metric_code: string;
    window_count_7d: number;
    window_count_30d: number;
    max_severity: string | null;
    consecutive_count: number;
    count_7d?: Record<string, number>;
    count_30d?: Record<string, number>;
  }>;
  dimension_alert_summary: Record<string, {
    total?: number;
    warning?: number;
    critical?: number;
    alert_count?: number;
    max_severity?: number;
    triggered_metrics?: string[];
  }>;
  recovery_status: string;
};

export const DECAY_LABELS: Record<string, string> = {
  NONE: "无衰减",
  SHORT_TERM_7D: "短期波动 (7D)",
  SUSTAINED_30D: "持续衰减 (30D)",
  SEVERE: "严重衰减",
};

export const WINDOW_STATUS_LABELS: Record<string, string> = {
  NORMAL: "正常",
  OBSERVING: "观察中",
  TRIGGERED: "已触发",
};

export const DIAGNOSIS_STATUS_LABELS: Record<string, string> = {
  PENDING: "待诊断",
  SKIPPED: "已跳过",
  COMPLETED: "已完成",
  NONE: "未执行",
};

export const CATEGORY_LABELS: Record<string, string> = {
  performance: "模型性能",
  drift: "分布漂移",
  quality: "数据质量",
  stability: "数据稳定性",
};

export const CATEGORY_ORDER = ["performance", "drift", "quality", "stability"];

/* ── 状态判定逻辑 ── */

export function computeMonitoringStatus(
  summary: CoverageSummary | null,
  alerts: AlertItem[]
): MonitoringStatus {
  if (!summary) return "incomplete";

  const hasCritical = alerts.some((a) => a.severity === "CRITICAL" || a.severity === "HIGH");
  const hasWarning = alerts.some((a) => a.severity === "WARNING");

  if (hasCritical) return "critical";
  if (hasWarning) return "at_risk";

  const allCalculated = summary.calculated === summary.total_metrics;
  const allRules = summary.rules_enabled === summary.total_metrics;
  const allAvailable = summary.available === summary.total_metrics;

  if (!allCalculated || !allAvailable || !allRules) return "incomplete";

  return "healthy";
}

export const STATUS_META: Record<
  MonitoringStatus,
  { label: string; color: "green" | "amber" | "red" | "slate"; dotColor: string; description: string }
> = {
  healthy: {
    label: "健康",
    color: "green",
    dotColor: "bg-emerald-500",
    description: "所有受监控指标均未达到告警阈值",
  },
  at_risk: {
    label: "有风险",
    color: "amber",
    dotColor: "bg-amber-500",
    description: "至少一个指标达到预警阈值，需要关注",
  },
  critical: {
    label: "严重",
    color: "red",
    dotColor: "bg-red-500",
    description: "至少一个指标达到严重告警阈值，需要立即处理",
  },
  incomplete: {
    label: "监控不完整",
    color: "slate",
    dotColor: "bg-slate-400",
    description: "部分指标计算失败、标签未成熟或规则未配置",
  },
};
