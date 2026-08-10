"use client";

import type { AlertItem, CoverageSummary } from "./monitoring-types";
import { formatValue } from "../shared";

type Props = {
  summary: CoverageSummary | null;
  alerts: AlertItem[];
};

export default function AlertExplanationPanel({ summary, alerts }: Props) {
  const hasAlerts = alerts.length > 0;
  const isIncomplete =
    summary && (summary.calculated < summary.total_metrics || summary.rules_enabled < summary.total_metrics);

  return (
    <div className="dash-card">
      <div className="flex items-center justify-between gap-3 px-5 pt-4 pb-0">
        <h2 className="text-sm font-semibold text-slate-700">告警与判定说明</h2>
      </div>
      <div className="dash-card-body space-y-4">
        {hasAlerts ? (
          /* ── 有告警：展示告警列表 ── */
          <div className="space-y-2">
            <p className="text-xs text-slate-500">
              本周期共触发 <span className="font-semibold text-red-600">{alerts.length}</span> 个告警
            </p>
            {alerts.map((a, i) => (
              <div
                key={a.alert_id || i}
                className={`rounded-lg border px-3 py-2.5 ${
                  a.severity === "CRITICAL" || a.severity === "HIGH"
                    ? "border-red-200 bg-red-50"
                    : a.severity === "WARNING"
                    ? "border-amber-200 bg-amber-50"
                    : "border-slate-200 bg-slate-50"
                }`}
              >
                <div className="flex items-center gap-2">
                  <span
                    className={`inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-bold ${
                      a.severity === "CRITICAL" || a.severity === "HIGH"
                        ? "bg-red-100 text-red-700"
                        : a.severity === "WARNING"
                        ? "bg-amber-100 text-amber-700"
                        : "bg-slate-100 text-slate-600"
                    }`}
                  >
                    {a.severity === "CRITICAL" ? "严重" : a.severity === "HIGH" ? "高风险" : a.severity === "WARNING" ? "预警" : a.severity || "INFO"}
                  </span>
                  <span className="text-sm font-semibold text-slate-800">{a.metric_code || a.alert_code}</span>
                </div>
                <div className="mt-1.5 grid grid-cols-3 gap-x-3 text-xs">
                  <div>
                    <span className="text-slate-400">当前值</span>
                    <p className="font-mono font-semibold text-slate-700">{formatValue(a.current_value)}</p>
                  </div>
                  <div>
                    <span className="text-slate-400">阈值</span>
                    <p className="font-mono text-slate-600">{formatValue(a.threshold)}</p>
                  </div>
                  <div>
                    <span className="text-slate-400">变化</span>
                    <p className="font-mono text-slate-600">{formatValue(a.delta)}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          /* ── 零告警：展示判定说明 ── */
          <div className="space-y-3">
            <p className="text-sm font-semibold text-emerald-700">✓ 本周期未触发告警</p>

            <div className="space-y-1.5 text-xs text-slate-600">
              <div className="flex items-center gap-2">
                <span className="text-emerald-500">✓</span>
                <span>{summary?.calculated ?? 0} 个指标计算成功</span>
              </div>
              <div className="flex items-center gap-2">
                {summary && summary.rules_enabled >= summary.total_metrics ? (
                  <>
                    <span className="text-emerald-500">✓</span>
                    <span>{summary.rules_enabled} 个指标已接入规则</span>
                  </>
                ) : (
                  <>
                    <span className="text-amber-500">⚠</span>
                    <span className="text-amber-700">
                      {summary?.rules_enabled ?? 0}/{summary?.total_metrics ?? 0} 个指标已接入规则
                    </span>
                  </>
                )}
              </div>
              <div className="flex items-center gap-2">
                <span className="text-emerald-500">✓</span>
                <span>无指标达到 Warning 阈值</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-emerald-500">✓</span>
                <span>标签与窗口数据完整</span>
              </div>
            </div>

            {/* 最接近阈值 */}
            {summary && summary.closest_thresholds.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-slate-500 mb-2">最接近预警阈值：</p>
                <div className="space-y-1.5">
                  {summary.closest_thresholds.map((ct, i) => (
                    <div key={ct.metric_code} className="flex items-center gap-2 text-xs">
                      <span className="text-slate-400 w-4">{i + 1}.</span>
                      <span className="font-medium text-slate-700 flex-1">{ct.display_name}</span>
                      <span className="font-mono text-slate-500">
                        已使用预警阈值的 {ct.usage_ratio != null ? (ct.usage_ratio * 100).toFixed(1) : "-"}%
                      </span>
                      <div className="w-16 h-1.5 rounded-full bg-slate-100 overflow-hidden">
                        <div
                          className="h-full rounded-full bg-sky-400"
                          style={{ width: `${Math.min(100, (ct.usage_ratio ?? 0) * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 不完整监控警告 */}
            {isIncomplete && (
              <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2.5">
                <p className="text-xs font-semibold text-amber-800">本周期没有产生告警，但不能判定为健康</p>
                {summary && summary.unmonitored_metrics.length > 0 && (
                  <p className="mt-1 text-xs text-amber-700">
                    {summary.unmonitored_metrics.length} 个指标尚未配置告警规则：{summary.unmonitored_metrics.map((m) => m.display_name).join("、")}
                  </p>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
