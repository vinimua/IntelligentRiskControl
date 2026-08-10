"use client";

import type { EnrichedMetric } from "./monitoring-types";
import { CATEGORY_LABELS, CATEGORY_ORDER } from "./monitoring-types";
import { formatValue } from "../shared";

type Props = {
  metrics: EnrichedMetric[];
  activeCategory: string;
  onSelectCategory: (cat: string) => void;
};

export default function MetricDetailSection({ metrics, activeCategory, onSelectCategory }: Props) {
  const filtered = metrics.filter((m) => m.category === activeCategory);
  const categoryLabel = CATEGORY_LABELS[activeCategory] || activeCategory;

  return (
    <div>
      {/* Tab 栏 */}
      <div className="flex items-center gap-1 border-b border-slate-200 pb-0 mb-4">
        {CATEGORY_ORDER.map((cat) => {
          const catMetrics = metrics.filter((m) => m.category === cat);
          const label = CATEGORY_LABELS[cat] || cat;
          const isActive = cat === activeCategory;
          return (
            <button
              key={cat}
              onClick={() => onSelectCategory(cat)}
              className={`rounded-t-lg px-4 py-2 text-sm font-semibold transition ${
                isActive
                  ? "border-b-2 border-indigo-600 text-indigo-700 bg-indigo-50/50"
                  : "text-slate-500 hover:text-slate-700 hover:bg-slate-50"
              }`}
            >
              {label}
              <span className="ml-1.5 text-xs text-slate-400">({catMetrics.length})</span>
            </button>
          );
        })}
      </div>

      {/* 指标卡片网格 */}
      {filtered.length === 0 ? (
        <div className="rounded-lg border border-dashed border-slate-300 p-6 text-center text-sm text-slate-400">
          该类别下暂无指标数据
        </div>
      ) : (
        <div className="grid gap-3 grid-cols-1 sm:grid-cols-2 xl:grid-cols-3">
          {filtered.map((m) => (
            <MetricCard key={m.metric_code} metric={m} />
          ))}
        </div>
      )}
    </div>
  );
}

/* ── 单张指标卡片 ── */

function MetricCard({ metric: m }: { metric: EnrichedMetric }) {
  // 确定卡片状态
  const isUnavailable = m.availability_status !== "AVAILABLE";
  const isUnmonitored = !m.rule_enabled && !isUnavailable;
  const isTriggered = m.triggered;
  const severity = m.severity;

  let borderColor = "border-emerald-200";
  let bgColor = "bg-white";
  let statusBadge: { label: string; className: string } | null = null;

  if (isUnavailable) {
    borderColor = "border-slate-200";
    bgColor = "bg-slate-50";
    const labels: Record<string, string> = {
      LABEL_NOT_MATURE: "标签未成熟",
      DATA_NOT_AVAILABLE: "数据不可用",
      SAMPLE_TOO_SMALL: "样本过小",
      CALCULATION_FAILED: "计算失败",
      NOT_APPLICABLE: "不适用",
    };
    statusBadge = {
      label: labels[m.availability_status] || m.availability_status,
      className: "border-slate-200 bg-slate-100 text-slate-500",
    };
  } else if (isTriggered && severity) {
    if (severity === "CRITICAL" || severity === "HIGH") {
      borderColor = "border-red-300";
      bgColor = "bg-red-50/50";
      statusBadge = { label: "严重", className: "border-red-200 bg-red-50 text-red-700" };
    } else if (severity === "WARNING") {
      borderColor = "border-amber-300";
      bgColor = "bg-amber-50/50";
      statusBadge = { label: "预警", className: "border-amber-200 bg-amber-50 text-amber-700" };
    }
  } else if (isUnmonitored) {
    borderColor = "border-slate-200";
    bgColor = "bg-slate-50/50";
    statusBadge = { label: "未监控", className: "border-slate-200 bg-slate-100 text-slate-500" };
  } else {
    statusBadge = { label: "正常", className: "border-emerald-200 bg-emerald-50 text-emerald-700" };
  }

  // 方向箭头
  const directionArrow = (() => {
    if (m.delta == null) return "";
    if (m.direction === "higher_better") return m.delta < 0 ? " ↓" : " ↑";
    if (m.direction === "lower_better") return m.delta > 0 ? " ↓" : " ↑";
    return m.delta !== 0 ? (Math.abs(m.delta) > 0 ? " ⚡" : "") : "";
  })();

  return (
    <div className={`rounded-xl border ${borderColor} ${bgColor} px-4 py-3.5 transition hover:shadow-sm`}>
      {/* 标题行 */}
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-slate-800 truncate" title={m.display_name}>
            {m.display_name}
          </p>
          <p className="text-[10px] font-mono text-slate-400 uppercase">{m.metric_code}</p>
        </div>
        {statusBadge && (
          <span className={`shrink-0 inline-flex items-center rounded-md border px-2 py-0.5 text-[10px] font-semibold ${statusBadge.className}`}>
            {statusBadge.label}
          </span>
        )}
      </div>

      {/* 值区域 */}
      <div className="mt-3 grid grid-cols-2 gap-x-3 gap-y-1.5 text-sm">
        <div>
          <span className="text-xs text-slate-400">当前值</span>
          <p className="font-mono font-bold text-slate-900">
            {formatValue(m.current_value)}
            {directionArrow && <span className="text-xs font-normal">{directionArrow}</span>}
          </p>
        </div>
        <div>
          <span className="text-xs text-slate-400">基线值</span>
          <p className="font-mono text-slate-600">{formatValue(m.baseline_value)}</p>
        </div>
        <div>
          <span className="text-xs text-slate-400">变化</span>
          <p className={`font-mono text-xs font-semibold ${m.delta != null && m.delta < 0 ? (m.direction === "higher_better" ? "text-red-600" : "text-emerald-600") : m.delta != null && m.delta > 0 ? (m.direction === "higher_better" ? "text-emerald-600" : "text-red-600") : "text-slate-600"}`}>
            {formatValue(m.delta)}
          </p>
        </div>
        <div>
          <span className="text-xs text-slate-400">方向</span>
          <p className="text-xs text-slate-500">
            {m.direction === "higher_better" ? "越高越好" : m.direction === "lower_better" ? "越低越好" : m.direction === "deviation_bad" ? "偏离越少越好" : "-"}
          </p>
        </div>
      </div>

      {/* 阈值条 */}
      {m.rule_enabled && m.warning_threshold != null && (
        <div className="mt-3">
          <div className="flex justify-between text-[10px] text-slate-400 mb-1">
            <span>预警 ≥ {formatValue(m.warning_threshold)}</span>
            <span>严重 ≥ {formatValue(m.critical_threshold)}</span>
          </div>
          <div className="h-2 rounded-full bg-slate-100 overflow-hidden">
            {m.threshold_usage_ratio != null ? (
              <div
                className={`h-full rounded-full transition-all ${
                  (m.threshold_usage_ratio ?? 0) >= 1
                    ? "bg-red-500"
                    : (m.threshold_usage_ratio ?? 0) >= 0.5
                    ? "bg-amber-500"
                    : "bg-emerald-500"
                }`}
                style={{ width: `${Math.min(100, (m.threshold_usage_ratio ?? 0) * 100)}%` }}
              />
            ) : (
              <div className="h-full w-0" />
            )}
          </div>
        </div>
      )}

      {/* 状态说明 */}
      <p className="mt-2.5 text-xs leading-relaxed text-slate-500">{m.status_reason}</p>

      {/* 未接入规则标识 */}
      {isUnmonitored && (
        <div className="mt-2 flex items-center gap-1.5 text-[10px] text-slate-400">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-slate-300" />
          已计算，但未参与告警判定
        </div>
      )}
    </div>
  );
}
