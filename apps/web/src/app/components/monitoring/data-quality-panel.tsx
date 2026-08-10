"use client";

import type { DataQualityData } from "./monitoring-types";
import { formatValue } from "../shared";

type Props = {
  data: DataQualityData | null;
  loading: boolean;
};

export default function DataQualityPanel({ data, loading }: Props) {
  if (loading) {
    return (
      <div className="rounded-lg border border-dashed border-slate-300 p-6 text-center text-sm text-slate-400">
        加载数据质量数据中...
      </div>
    );
  }

  if (!data) {
    return (
      <div className="rounded-lg border border-dashed border-slate-300 p-6 text-center text-sm text-slate-400">
        暂无数据质量数据
      </div>
    );
  }

  const { overall_missing_rate, overall_outlier_rate, dq_score, fields, schema_changes } = data;

  const dqFlagColor = (flag: string) => {
    if (flag === "ALERT") return "bg-red-50 text-red-700 border-red-200";
    if (flag === "WARN") return "bg-amber-50 text-amber-700 border-amber-200";
    return "bg-emerald-50 text-emerald-700 border-emerald-200";
  };

  const hasAnomalyFields = fields.some((f) => f.dq_flag !== "OK");

  return (
    <div className="space-y-5">
      {/* 整体概览 */}
      <div className="grid grid-cols-3 gap-3">
        <div className="rounded-xl border border-slate-200 bg-white px-4 py-3.5">
          <p className="text-xs text-slate-400">缺失率最大变化</p>
          <p className={`mt-1 text-xl font-bold ${(overall_missing_rate ?? 0) > 0.1 ? "text-red-600" : (overall_missing_rate ?? 0) > 0.05 ? "text-amber-600" : "text-slate-800"}`}>
            {overall_missing_rate != null ? `${(overall_missing_rate * 100).toFixed(1)}%` : "-"}
          </p>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white px-4 py-3.5">
          <p className="text-xs text-slate-400">异常值率最大变化</p>
          <p className={`mt-1 text-xl font-bold ${(overall_outlier_rate ?? 0) > 0.15 ? "text-red-600" : (overall_outlier_rate ?? 0) > 0.05 ? "text-amber-600" : "text-slate-800"}`}>
            {overall_outlier_rate != null ? `${(overall_outlier_rate * 100).toFixed(1)}%` : "-"}
          </p>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white px-4 py-3.5">
          <p className="text-xs text-slate-400">数据质量综合分</p>
          <p className={`mt-1 text-xl font-bold ${(dq_score ?? 1) < 0.7 ? "text-red-600" : (dq_score ?? 1) < 0.9 ? "text-amber-600" : "text-emerald-600"}`}>
            {dq_score != null ? dq_score.toFixed(2) : "-"}
          </p>
        </div>
      </div>

      {hasAnomalyFields && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-2.5 text-sm text-amber-700">
          异常字段：{fields.filter((f) => f.dq_flag !== "OK").length} / {fields.length}
        </div>
      )}

      {/* 字段明细表 */}
      {fields.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-slate-200">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-xs text-slate-500">
                <th className="px-3 py-2 text-left font-semibold">字段</th>
                <th className="px-3 py-2 text-right font-semibold">基线缺失率</th>
                <th className="px-3 py-2 text-right font-semibold">当前缺失率</th>
                <th className="px-3 py-2 text-right font-semibold">变化</th>
                <th className="px-3 py-2 text-right font-semibold">异常值率</th>
                <th className="px-3 py-2 text-right font-semibold">异常值变化</th>
                <th className="px-3 py-2 text-center font-semibold">DQ标记</th>
              </tr>
            </thead>
            <tbody>
              {fields.map((f) => (
                <tr key={f.field_name} className="border-b border-slate-100 hover:bg-slate-50">
                  <td className="px-3 py-2 font-medium text-slate-700">{f.field_name}</td>
                  <td className="px-3 py-2 font-mono text-xs text-right">{f.baseline_missing_rate != null ? (f.baseline_missing_rate * 100).toFixed(2) + "%" : "-"}</td>
                  <td className="px-3 py-2 font-mono text-xs text-right">{f.current_missing_rate != null ? (f.current_missing_rate * 100).toFixed(2) + "%" : "-"}</td>
                  <td className={`px-3 py-2 font-mono text-xs text-right font-semibold ${(f.missing_delta ?? 0) > 0.05 ? "text-red-600" : (f.missing_delta ?? 0) > 0.02 ? "text-amber-600" : "text-slate-600"}`}>
                    {f.missing_delta != null ? (f.missing_delta > 0 ? "+" : "") + (f.missing_delta * 100).toFixed(2) + "%" : "-"}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs text-right">{f.outlier_rate != null ? (f.outlier_rate * 100).toFixed(2) + "%" : "-"}</td>
                  <td className="px-3 py-2 font-mono text-xs text-right">{f.outlier_delta != null ? (f.outlier_delta > 0 ? "+" : "") + (f.outlier_delta * 100).toFixed(2) + "%" : "-"}</td>
                  <td className="px-3 py-2 text-center">
                    <span className={`inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-semibold border ${dqFlagColor(f.dq_flag)}`}>
                      {f.dq_flag}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Schema 变更 */}
      {schema_changes.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-slate-700 mb-2">Schema 变更</h4>
          <div className="space-y-1">
            {schema_changes.map((sc, i) => (
              <div key={`${sc.change_type}-${sc.column_name}-${i}`} className="flex items-center gap-2 rounded-lg border border-slate-200 px-3 py-1.5 text-xs">
                <span
                  className={`inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                    sc.change_type === "removed"
                      ? "bg-red-50 text-red-700"
                      : sc.change_type === "added"
                      ? "bg-emerald-50 text-emerald-700"
                      : "bg-amber-50 text-amber-700"
                  }`}
                >
                  {sc.change_type === "removed" ? "删除" : sc.change_type === "added" ? "新增" : "类型变化"}
                </span>
                <span className="font-mono text-slate-700">{sc.column_name}</span>
                {sc.detail && <span className="text-slate-400">{sc.detail}</span>}
              </div>
            ))}
          </div>
        </div>
      )}

      {fields.length === 0 && schema_changes.length === 0 && (
        <div className="rounded-lg border border-dashed border-slate-300 p-6 text-center text-sm text-slate-400">
          暂无字段级数据质量数据
        </div>
      )}
    </div>
  );
}
