"use client";

import type { FeatureDriftItem } from "./monitoring-types";
import { formatValue } from "../shared";

type Props = {
  item: FeatureDriftItem;
  onClose: () => void;
};

export default function FeatureDriftDrawer({ item, onClose }: Props) {
  return (
    <>
      {/* 遮罩 */}
      <div className="fixed inset-0 z-40 bg-slate-900/30 transition-opacity" onClick={onClose} />

      {/* 抽屉 */}
      <div className="fixed inset-y-0 right-0 z-50 w-full max-w-md bg-white shadow-2xl border-l border-slate-200 overflow-y-auto">
        {/* 头部 */}
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-200 bg-white px-5 py-4">
          <div>
            <h3 className="text-lg font-bold text-slate-800">{item.feature_name}</h3>
            <p className="text-xs text-slate-400">特征漂移详情</p>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-600 transition"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* 内容 */}
        <div className="space-y-5 p-5">
          {/* PSI 概览 */}
          <section>
            <h4 className="text-xs font-semibold uppercase tracking-[.12em] text-slate-400 mb-3">PSI 漂移</h4>
            <div className="grid grid-cols-3 gap-3">
              <StatBox label="PSI 7D" value={item.psi_7d} />
              <StatBox label="PSI 30D" value={item.psi_30d} />
              <StatBox label="最大 PSI" value={item.max_psi} highlight />
            </div>
            <div className="mt-3 flex items-center gap-2 text-xs text-slate-500">
              <span>阈值: {formatValue(item.threshold)}</span>
              <span>·</span>
              <span>状态: </span>
              <span
                className={`inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                  item.status === "critical"
                    ? "bg-red-50 text-red-700"
                    : item.status === "warning"
                    ? "bg-amber-50 text-amber-700"
                    : "bg-emerald-50 text-emerald-700"
                }`}
              >
                {item.status === "critical" ? "严重" : item.status === "warning" ? "预警" : "正常"}
              </span>
            </div>
          </section>

          {/* 分布对比 */}
          <section>
            <h4 className="text-xs font-semibold uppercase tracking-[.12em] text-slate-400 mb-3">统计量</h4>
            <div className="grid grid-cols-2 gap-3">
              <StatBox label="JS 散度" value={item.js_divergence} />
              <StatBox label="Wasserstein 距离" value={item.wasserstein_distance} />
              <StatBox label="KS 统计量" value={item.ks_statistic} />
              <StatBox label="模型重要性" value={item.model_importance} isString />
            </div>
          </section>

          {/* 数据质量 */}
          <section>
            <h4 className="text-xs font-semibold uppercase tracking-[.12em] text-slate-400 mb-3">数据质量</h4>
            <div className="grid grid-cols-2 gap-3">
              <StatBox label="缺失率" value={item.missing_rate} />
              <StatBox label="缺失率变化" value={item.missing_rate_delta} />
              <StatBox label="异常值率" value={item.outlier_rate} />
              <StatBox label="DQ 分" value={item.dq_score} />
            </div>
            {item.dq_flag && item.dq_flag !== "OK" && (
              <div className="mt-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs text-amber-700">
                DQ 标记: {item.dq_flag}
              </div>
            )}
          </section>
        </div>
      </div>
    </>
  );
}

function StatBox({
  label,
  value,
  highlight,
  isString,
}: {
  label: string;
  value: number | string | null | undefined;
  highlight?: boolean;
  isString?: boolean;
}) {
  return (
    <div className={`rounded-lg border px-3 py-2.5 ${highlight ? "border-indigo-200 bg-indigo-50/50" : "border-slate-200 bg-white"}`}>
      <p className="text-[10px] text-slate-400">{label}</p>
      <p className={`mt-0.5 text-sm font-bold ${highlight ? "text-indigo-700" : "text-slate-800"}`}>
        {isString ? (value || "—") : formatValue(value)}
      </p>
    </div>
  );
}
