"use client";

import type { PersistenceJudgment } from "./monitoring-types";
import {
  DECAY_LABELS,
  DIAGNOSIS_STATUS_LABELS,
  WINDOW_STATUS_LABELS,
} from "./monitoring-types";

type Props = {
  persistence: PersistenceJudgment | null;
  diagnosisStatus: string | null;
  visibleAlertCount?: number;
};

const LOCAL_DIAGNOSIS_LABELS: Record<string, string> = {
  PENDING: "待诊断",
  SKIPPED: "已跳过",
  COMPLETED: "已完成",
  MANUAL_REVIEW: "待人工复核",
  FAILED: "诊断失败",
};

const LOCAL_DECAY_LABELS: Record<string, string> = {
  NONE: "无衰减",
  SHORT_TERM_7D: "短期波动",
  SUSTAINED_30D: "持续衰减",
  SEVERE: "严重衰减",
};

const LOCAL_WINDOW_STATUS_LABELS: Record<string, string> = {
  NORMAL: "正常",
  OBSERVING: "观察中",
  TRIGGERED: "已触发",
};

function diagnosisLabel(status: string | null): string {
  if (!status) return "未执行";
  return LOCAL_DIAGNOSIS_LABELS[status] || DIAGNOSIS_STATUS_LABELS[status] || status;
}

function yesNo(value: boolean): string {
  return value ? "是" : "否";
}

export default function PersistenceCard({
  persistence,
  diagnosisStatus,
  visibleAlertCount = 0,
}: Props) {
  const statusLabel = diagnosisLabel(diagnosisStatus);

  if (!persistence) {
    return (
      <div className="rounded-xl border border-slate-200 bg-slate-50 px-5 py-4">
        <div className="flex items-center gap-3">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-slate-300" />
          <span className="text-sm font-semibold text-slate-500">持续性判定</span>
          <span className="text-xs text-slate-400">
            暂无数据（运行未完成或 pipeline 未产出）
          </span>
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

  const decayLabel = LOCAL_DECAY_LABELS[decay_degree] || DECAY_LABELS[decay_degree] || decay_degree;
  const status7Label =
    LOCAL_WINDOW_STATUS_LABELS[status_7d] || WINDOW_STATUS_LABELS[status_7d] || status_7d;
  const status30Label =
    LOCAL_WINDOW_STATUS_LABELS[status_30d] || WINDOW_STATUS_LABELS[status_30d] || status_30d;
  const triggerSource = trigger_diagnosis
    ? visibleAlertCount > 0
      ? "汇总告警 + B1持续性"
      : "B1持续性判定"
    : visibleAlertCount > 0
      ? "汇总告警"
      : "无";

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
              : diagnosisStatus === "PENDING" || diagnosisStatus === "MANUAL_REVIEW"
                ? "border-amber-200 bg-amber-50 text-amber-700"
                : diagnosisStatus === "FAILED"
                  ? "border-red-200 bg-red-50 text-red-700"
                  : "border-slate-200 bg-slate-100 text-slate-500"
          }`}
        >
          诊断: {statusLabel}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-2.5 sm:grid-cols-5">
        <Field
          label="诊断触发"
          value={yesNo(trigger_diagnosis)}
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
          value={yesNo(requires_manual_review)}
          highlight={requires_manual_review}
          highlightColor="red"
        />
        <Field label="诊断状态" value={statusLabel} />
        <Field
          label="触发来源"
          value={triggerSource}
          highlight={trigger_diagnosis}
          highlightColor={visibleAlertCount > 0 ? "amber" : "red"}
        />
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-2.5">
        <WindowStatus label="7D状态" status={status_7d} value={status7Label} />
        <WindowStatus label="30D状态" status={status_30d} value={status30Label} />
      </div>

      {dimension_alert_summary && Object.keys(dimension_alert_summary).length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-semibold text-slate-500">维度告警汇总</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(dimension_alert_summary).map(([dim, counts]) => {
              const total = counts.total ?? counts.alert_count ?? 0;
              const critical = counts.critical ?? 0;
              const warning = counts.warning ?? 0;
              return (
                <span
                  key={dim}
                  className={`inline-flex items-center gap-1 rounded border px-2 py-0.5 text-[10px] font-semibold ${
                    critical > 0
                      ? "border-red-200 bg-red-50 text-red-700"
                      : warning > 0
                        ? "border-amber-200 bg-amber-50 text-amber-700"
                        : "border-slate-200 bg-slate-100 text-slate-500"
                  }`}
                >
                  {dim}: {total} 告警
                  {critical > 0 && <span className="text-red-500">({critical}C)</span>}
                  {warning > 0 && <span className="text-amber-500">({warning}W)</span>}
                </span>
              );
            })}
          </div>
        </div>
      )}

      {persistence_evidence && persistence_evidence.length > 0 && (
        <details className="text-xs">
          <summary className="cursor-pointer font-semibold text-slate-500 hover:text-slate-700">
            持续性证据（{persistence_evidence.length} 项）
          </summary>
          <div className="mt-2 max-h-40 space-y-1 overflow-y-auto">
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
                      7D滚窗:{" "}
                      <span className="font-semibold text-slate-600">{e.window_count_7d}</span>
                    </span>
                    <span className="text-slate-400">
                      30D滚窗:{" "}
                      <span className="font-semibold text-slate-600">{e.window_count_30d}</span>
                    </span>
                    <span className="text-slate-400">
                      连续:{" "}
                      <span className="font-semibold text-slate-600">
                        {e.consecutive_count ?? "-"}
                      </span>
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

function WindowStatus({
  label,
  status,
  value,
}: {
  label: string;
  status: string;
  value: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-slate-400">{label}</span>
      <span
        className={`inline-flex items-center rounded border px-2 py-0.5 text-xs font-semibold ${
          status === "TRIGGERED"
            ? "border-red-200 bg-red-50 text-red-700"
            : status === "OBSERVING"
              ? "border-amber-200 bg-amber-50 text-amber-700"
              : "border-emerald-200 bg-emerald-50 text-emerald-700"
        }`}
      >
        {value}
      </span>
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
