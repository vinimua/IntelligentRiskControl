"use client";

import type { PersistenceJudgment } from "./monitoring-types";
import {
  DECAY_LABELS,
  WINDOW_STATUS_LABELS,
  DIAGNOSIS_STATUS_LABELS,
} from "./monitoring-types";

type Props = {
  persistence: PersistenceJudgment | null;
  diagnosisStatus: string | null;
};

export default function PersistenceCard({ persistence, diagnosisStatus }: Props) {
  const statusLabel = DIAGNOSIS_STATUS_LABELS[diagnosisStatus || ""] || diagnosisStatus || "未执行";

  if (!persistence) {
    return (
      <div className="rounded-xl border border-slate-200 bg-slate-50 px-5 py-4">
        <div className="flex items-center gap-3">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-slate-300" />
          <span className="text-sm font-semibold text-slate-500">持续性判定</span>
          <span className="text-xs text-slate-400">— 暂无数据（运行未完成或 pipeline 未产出）</span>
        </div>
      </div>
    );
  }

  const {
    trigger_diagnosis,
    decay_degree,
    requires_manual_review,
    status_7d,
    status_30d,
    persistence_evidence,
    dimension_alert_summary,
  } = persistence;

  const decayLabel = DECAY_LABELS[decay_degree] || decay_degree;
  const status7Label = WINDOW_STATUS_LABELS[status_7d] || status_7d;
  const status30Label = WINDOW_STATUS_LABELS[status_30d] || status_30d;

  // 卡片整体色调
  let borderColor = "border-slate-200";
  let bgColor = "bg-white";
  if (decay_degree === "SEVERE") {
    borderColor = "border-red-300";
    bgColor = "bg-red-50/50";
  } else if (decay_degree === "SUSTAINED_30D") {
    borderColor = "border-amber-300";
    bgColor = "bg-amber-50/50";
  } else if (decay_degree === "SHORT_TERM_7D") {
    borderColor = "border-sky-300";
    bgColor = "bg-sky-50/30";
  }

  return (
    <div className={`rounded-xl border ${borderColor} ${bgColor} px-5 py-4 space-y-4`}>
      {/* 标题行 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className={`inline-block h-2.5 w-2.5 rounded-full ${
              decay_degree === "SEVERE"
                ? "bg-red-500 animate-pulse"
                : decay_degree === "SUSTAINED_30D"
                ? "bg-amber-500"
                : decay_degree === "SHORT_TERM_7D"
                ? "bg-sky-500"
                : "bg-emerald-500"
            }`}
          />
          <span className="text-sm font-bold text-slate-700">持续性判定</span>
        </div>
        <span
          className={`inline-flex items-center rounded-md border px-2 py-0.5 text-[10px] font-semibold ${
            diagnosisStatus === "COMPLETED"
              ? "border-emerald-200 bg-emerald-50 text-emerald-700"
              : diagnosisStatus === "PENDING"
              ? "border-amber-200 bg-amber-50 text-amber-700"
              : "border-slate-200 bg-slate-100 text-slate-500"
          }`}
        >
          诊断: {statusLabel}
        </span>
      </div>

      {/* 核心判定字段 */}
      <div className="grid grid-cols-2 gap-x-6 gap-y-2.5 sm:grid-cols-4">
        <Field
          label="诊断触发"
          value={trigger_diagnosis ? "是" : "否"}
          highlight={trigger_diagnosis}
          highlightColor="amber"
        />
        <Field
          label="衰减类型"
          value={decayLabel}
          highlight={decay_degree !== "NONE"}
          highlightColor={decay_degree === "SEVERE" ? "red" : "amber"}
        />
        <Field
          label="人工复核"
          value={requires_manual_review ? "是" : "否"}
          highlight={requires_manual_review}
          highlightColor="red"
        />
        <Field label="诊断状态" value={statusLabel} />
      </div>

      {/* 7D / 30D 状态 */}
      <div className="grid grid-cols-2 gap-x-6 gap-y-2.5">
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">7D 状态</span>
          <span
            className={`inline-flex items-center rounded px-2 py-0.5 text-xs font-semibold ${
              status_7d === "TRIGGERED"
                ? "bg-red-50 text-red-700 border border-red-200"
                : status_7d === "OBSERVING"
                ? "bg-amber-50 text-amber-700 border border-amber-200"
                : "bg-emerald-50 text-emerald-700 border border-emerald-200"
            }`}
          >
            {status7Label}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">30D 状态</span>
          <span
            className={`inline-flex items-center rounded px-2 py-0.5 text-xs font-semibold ${
              status_30d === "TRIGGERED"
                ? "bg-red-50 text-red-700 border border-red-200"
                : status_30d === "OBSERVING"
                ? "bg-amber-50 text-amber-700 border border-amber-200"
                : "bg-emerald-50 text-emerald-700 border border-emerald-200"
            }`}
          >
            {status30Label}
          </span>
        </div>
      </div>

      {/* 维度告警汇总 */}
      {dimension_alert_summary && Object.keys(dimension_alert_summary).length > 0 && (
        <div>
          <p className="text-xs font-semibold text-slate-500 mb-1.5">维度告警汇总</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(dimension_alert_summary).map(([dim, counts]) => (
              <span
                key={dim}
                className={`inline-flex items-center gap-1 rounded border px-2 py-0.5 text-[10px] font-semibold ${
                  counts.critical > 0
                    ? "border-red-200 bg-red-50 text-red-700"
                    : counts.warning > 0
                    ? "border-amber-200 bg-amber-50 text-amber-700"
                    : "border-slate-200 bg-slate-100 text-slate-500"
                }`}
              >
                {dim}: {counts.total} 告警
                {counts.critical > 0 && <span className="text-red-500">({counts.critical}C)</span>}
                {counts.warning > 0 && <span className="text-amber-500">({counts.warning}W)</span>}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* 持续性证据（折叠，只显示有告警的指标） */}
      {persistence_evidence && persistence_evidence.length > 0 && (
        <details className="text-xs">
          <summary className="cursor-pointer font-semibold text-slate-500 hover:text-slate-700">
            持续性证据（{persistence_evidence.length} 项）
          </summary>
          <div className="mt-2 max-h-40 overflow-y-auto space-y-1">
            {persistence_evidence
              .filter((e) => (e.window_count_7d || 0) > 0 || (e.window_count_30d || 0) > 0)
              .map((e) => (
                <div
                  key={e.metric_code}
                  className="flex items-center justify-between rounded bg-white/60 px-2 py-1"
                >
                  <span className="font-mono text-slate-600">{e.metric_code}</span>
                  <div className="flex items-center gap-3">
                    <span className="text-slate-400">
                      7D: <span className="font-semibold text-slate-600">{e.window_count_7d}</span>
                    </span>
                    <span className="text-slate-400">
                      30D: <span className="font-semibold text-slate-600">{e.window_count_30d}</span>
                    </span>
                    <span className="text-slate-400">
                      连续: <span className="font-semibold text-slate-600">{e.consecutive_count}</span>
                    </span>
                    {e.max_severity && (
                      <span
                        className={`inline-flex items-center rounded px-1 py-0 text-[10px] font-semibold ${
                          e.max_severity === "CRITICAL" || e.max_severity === "HIGH"
                            ? "text-red-600"
                            : "text-amber-600"
                        }`}
                      >
                        {e.max_severity}
                      </span>
                    )}
                  </div>
                </div>
              ))}
          </div>
        </details>
      )}
    </div>
  );
}

function Field({
  label,
  value,
  highlight,
  highlightColor,
}: {
  label: string;
  value: string;
  highlight?: boolean;
  highlightColor?: string;
}) {
  return (
    <div>
      <p className="text-[10px] text-slate-400">{label}</p>
      <p
        className={`text-sm font-semibold ${
          highlight
            ? highlightColor === "red"
              ? "text-red-600"
              : "text-amber-600"
            : "text-slate-700"
        }`}
      >
        {value}
      </p>
    </div>
  );
}
