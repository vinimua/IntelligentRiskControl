"use client";

import type { EnrichedMetric, CoverageSummary } from "./monitoring-types";
import { formatValue } from "../shared";

type Props = {
  metrics: EnrichedMetric[];
  summary: CoverageSummary | null;
};

export default function RuleCoverageTable({ metrics, summary }: Props) {
  if (!summary) return null;

  const coveragePct =
    summary.total_metrics > 0 ? Math.round((summary.rules_enabled / summary.total_metrics) * 100) : 0;
  const isFullCoverage = coveragePct >= 100;

  // 按类别排序
  const categoryOrder = ["performance", "drift", "quality", "stability"];
  const sorted = [...metrics].sort((a, b) => {
    const ca = categoryOrder.indexOf(a.category);
    const cb = categoryOrder.indexOf(b.category);
    if (ca !== cb) return ca - cb;
    return a.metric_code.localeCompare(b.metric_code);
  });

  return (
    <div className="space-y-4">
      {/* 覆盖率横幅 */}
      <div
        className={`rounded-xl border px-5 py-3.5 ${
          isFullCoverage ? "border-emerald-200 bg-emerald-50" : "border-amber-200 bg-amber-50"
        }`}
      >
        <div className="flex items-center justify-between">
          <span className={`text-lg font-bold ${isFullCoverage ? "text-emerald-700" : "text-amber-700"}`}>
            规则覆盖率：{coveragePct}%
          </span>
          <span className={`text-sm ${isFullCoverage ? "text-emerald-600" : "text-amber-600"}`}>
            {summary.rules_enabled} / {summary.total_metrics} 个指标已配置告警规则
          </span>
        </div>
        {!isFullCoverage && (
          <p className="mt-1 text-xs text-amber-600">
            覆盖率低于 100%，整体状态不应显示为绿色健康
          </p>
        )}
      </div>

      {/* 规则明细表 */}
      <div className="overflow-x-auto rounded-lg border border-slate-200">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-xs text-slate-500">
              <th className="px-3 py-2 text-left font-semibold">指标</th>
              <th className="px-3 py-2 text-left font-semibold">类别</th>
              <th className="px-3 py-2 text-center font-semibold">已计算</th>
              <th className="px-3 py-2 text-center font-semibold">数据可用</th>
              <th className="px-3 py-2 text-center font-semibold">规则启用</th>
              <th className="px-3 py-2 text-right font-semibold">Warning</th>
              <th className="px-3 py-2 text-right font-semibold">Critical</th>
              <th className="px-3 py-2 text-center font-semibold">最近判定</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((m) => {
              const isAvailable = m.availability_status === "AVAILABLE";
              const isCalculated = m.availability_status !== "CALCULATION_FAILED";
              const categoryLabels: Record<string, string> = {
                performance: "性能",
                drift: "漂移",
                quality: "质量",
                stability: "稳定性",
              };

              let verdictLabel = "未监控";
              let verdictColor = "bg-slate-50 text-slate-500 border-slate-200";
              if (!isAvailable) {
                verdictLabel = "不可用";
                verdictColor = "bg-slate-50 text-slate-400 border-slate-200";
              } else if (m.triggered && m.severity) {
                verdictLabel = m.severity === "CRITICAL" || m.severity === "HIGH" ? "严重" : "预警";
                verdictColor =
                  m.severity === "CRITICAL" || m.severity === "HIGH"
                    ? "bg-red-50 text-red-700 border-red-200"
                    : "bg-amber-50 text-amber-700 border-amber-200";
              } else if (m.rule_enabled) {
                verdictLabel = "正常";
                verdictColor = "bg-emerald-50 text-emerald-700 border-emerald-200";
              }

              return (
                <tr key={m.metric_code} className={`border-b border-slate-100 hover:bg-slate-50 ${!isAvailable ? "opacity-60" : ""}`}>
                  <td className="px-3 py-2.5">
                    <span className="font-medium text-slate-700">{m.display_name}</span>
                    <span className="ml-1.5 text-[10px] font-mono text-slate-400 uppercase">{m.metric_code}</span>
                  </td>
                  <td className="px-3 py-2.5 text-xs text-slate-500">{categoryLabels[m.category] || m.category}</td>
                  <td className="px-3 py-2.5 text-center">
                    <span className={isCalculated ? "text-emerald-600" : "text-red-500"}>
                      {isCalculated ? "是" : "否"}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 text-center">
                    <span className={isAvailable ? "text-emerald-600" : "text-slate-400"}>
                      {isAvailable ? "是" : "否"}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 text-center">
                    <span className={m.rule_enabled ? "text-emerald-600" : "text-amber-600 font-semibold"}>
                      {m.rule_enabled ? "是" : "否"}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono text-xs">
                    {m.warning_threshold != null ? formatValue(m.warning_threshold) : "—"}
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono text-xs">
                    {m.critical_threshold != null ? formatValue(m.critical_threshold) : "—"}
                  </td>
                  <td className="px-3 py-2.5 text-center">
                    <span className={`inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-semibold border ${verdictColor}`}>
                      {verdictLabel}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
