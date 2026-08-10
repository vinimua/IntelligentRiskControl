"use client";

import type { AlertItem, CoverageSummary, MonitoringStatus } from "./monitoring-types";
import { computeMonitoringStatus, STATUS_META } from "./monitoring-types";

type Props = {
  summary: CoverageSummary | null;
  alerts: AlertItem[];
  runInfo: { model_id?: string; champion_version?: string } | null;
};

export default function StatusBar({ summary, alerts, runInfo }: Props) {
  const status: MonitoringStatus = computeMonitoringStatus(summary, alerts);
  const meta = STATUS_META[status];

  const isIncomplete = status === "incomplete";
  const incompleteBecauseRules =
    isIncomplete && summary && summary.rules_enabled < summary.total_metrics && summary.calculated === summary.total_metrics;

  return (
    <div className="space-y-3">
      {/* 顶部模型信息 */}
      {runInfo && (
        <div className="flex items-center gap-3 text-sm text-slate-600">
          <span className="font-semibold text-slate-800">{runInfo.model_id ?? "-"}</span>
          <span className="text-slate-300">|</span>
          <span className="font-mono text-xs">{runInfo.champion_version ?? "-"}</span>
          <span className="text-slate-300">|</span>
          <span className="text-xs text-slate-400">W0 → W3</span>
        </div>
      )}

      {/* 状态横幅 */}
      <div
        className={`rounded-xl border px-5 py-4 ${
          status === "healthy"
            ? "border-emerald-200 bg-emerald-50"
            : status === "at_risk"
            ? "border-amber-200 bg-amber-50"
            : status === "critical"
            ? "border-red-200 bg-red-50"
            : "border-slate-200 bg-slate-50"
        }`}
      >
        <div className="flex items-center gap-3">
          <span className={`inline-block h-3 w-3 rounded-full ${meta.dotColor} ${status === "incomplete" ? "" : "animate-pulse"}`} />
          <span
            className={`text-lg font-bold ${
              status === "healthy"
                ? "text-emerald-700"
                : status === "at_risk"
                ? "text-amber-700"
                : status === "critical"
                ? "text-red-700"
                : "text-slate-600"
            }`}
          >
            运行状态：{meta.label}
          </span>
        </div>

        {/* 统计条 */}
        {summary && (
          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
            <span className="text-slate-700">
              <span className="font-semibold">{summary.calculated}</span>/{summary.total_metrics} 已计算
            </span>
            <span className="text-slate-300">·</span>
            <span className="text-slate-700">
              <span className="font-semibold">{summary.available}</span>/{summary.total_metrics} 可用
            </span>
            <span className="text-slate-300">·</span>
            <span className={summary.rules_enabled < summary.total_metrics ? "text-amber-600 font-semibold" : "text-slate-700"}>
              <span className="font-semibold">{summary.rules_enabled}</span>/{summary.total_metrics} 已接入规则
            </span>
            <span className="text-slate-300">·</span>
            <span className={summary.triggered > 0 ? "text-red-600 font-semibold" : "text-slate-700"}>
              <span className="font-semibold">{summary.triggered}</span> 告警
            </span>
            <span className="text-slate-300">·</span>
            <span className="text-xs text-slate-500">数据标签已成熟</span>
          </div>
        )}

        {/* 描述 */}
        <p className="mt-2 text-sm text-slate-600">{meta.description}</p>

        {/* 不完整监控警告 */}
        {incompleteBecauseRules && summary && (
          <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-2.5">
            <p className="text-sm font-semibold text-amber-800">
              ⚠ 监控不完整：{summary.total_metrics} 个指标已计算，但只有 {summary.rules_enabled} 个参与告警判定
            </p>
            {summary.unmonitored_metrics.length > 0 && (
              <p className="mt-1 text-xs text-amber-700">
                未监控指标：{summary.unmonitored_metrics.map((m) => m.display_name).join("、")}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
