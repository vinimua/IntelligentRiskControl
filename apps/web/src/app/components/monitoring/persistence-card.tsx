"use client";

import type { PersistenceJudgment, SentinelEvidence } from "./monitoring-types";
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
  NONE: "无持续衰减",
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
  visibleAlertCount: _visibleAlertCount = 0,
}: Props) {
  const statusLabel = diagnosisLabel(diagnosisStatus);

  if (!persistence) {
    return (
      <div className="rounded-xl border border-slate-200 bg-slate-50 px-5 py-4">
        <div className="flex items-center gap-3">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-slate-300" />
          <span className="text-sm font-semibold text-slate-500">诊断触发判定</span>
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
    trigger_sources,
    sentinel_evidence,
  } = persistence;

  const sources = trigger_sources ?? [];
  const b1Triggered = sources.includes("B1_PERSISTENCE");
  const sentinelTriggered = sources.includes("WP08_SENTINEL");
  const triggerSourceLabel =
    b1Triggered && sentinelTriggered
      ? "B1 持续性 + Sentinel"
      : b1Triggered
        ? "B1 持续性"
        : sentinelTriggered
          ? "Sentinel 异常"
          : "未触发";

  const decayLabel = LOCAL_DECAY_LABELS[decay_degree] || DECAY_LABELS[decay_degree] || decay_degree;
  const status7Label =
    LOCAL_WINDOW_STATUS_LABELS[status_7d] || WINDOW_STATUS_LABELS[status_7d] || status_7d;
  const status30Label =
    LOCAL_WINDOW_STATUS_LABELS[status_30d] || WINDOW_STATUS_LABELS[status_30d] || status_30d;

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
  } else if (sentinelTriggered) {
    borderColor = "border-violet-300";
    bgColor = "bg-violet-50/30";
  }

  // 分开 B1 持续性证据（非 Sentinel）和 Sentinel 证据
  const b1Evidence = (persistence_evidence ?? []).filter(
    (e) => !e.source || e.source !== "WP08_SENTINEL"
  );

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
                    : sentinelTriggered
                      ? "bg-violet-500"
                      : "bg-emerald-500"
            }`}
          />
          <span className="text-sm font-bold text-slate-700">诊断触发判定</span>
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
          label="最终结论"
          value={trigger_diagnosis ? "进入诊断" : "无需诊断"}
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
          label="触发机制"
          value={triggerSourceLabel}
          highlight={trigger_diagnosis}
          highlightColor={sentinelTriggered && !b1Triggered ? "red" : "amber"}
        />
      </div>

      {/* Sentinel 证据区 */}
      <SentinelSection evidence={sentinel_evidence ?? null} />

      <div className="grid grid-cols-2 gap-x-6 gap-y-2.5">
        <div>
          <p className="text-[10px] text-slate-400">B1 持续性</p>
          <span
            className={`text-sm font-semibold ${b1Triggered ? "text-amber-600" : "text-slate-500"}`}
          >
            {b1Triggered ? "已触发" : "未触发"}
          </span>
        </div>
        <div>
          <p className="text-[10px] text-slate-400">Sentinel</p>
          <span
            className={`text-sm font-semibold ${sentinelTriggered ? "text-violet-600" : "text-slate-500"}`}
          >
            {sentinelTriggered ? "已触发" : "未触发"}
          </span>
        </div>
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

      {/* B1 持续性证据（排除 Sentinel） */}
      {b1Evidence.length > 0 && (
        <details className="text-xs">
          <summary className="cursor-pointer font-semibold text-slate-500 hover:text-slate-700">
            B1 持续性证据（{b1Evidence.length} 项）
          </summary>
          <div className="mt-2 max-h-40 space-y-1 overflow-y-auto">
            {b1Evidence
              .filter((e) => (e.window_count_7d || 0) > 0 || (e.window_count_30d || 0) > 0)
              .map((e) => (
                <div
                  key={e.metric_code}
                  className="flex items-center justify-between rounded bg-white/60 px-2 py-1"
                >
                  <span className="font-mono text-slate-600">{e.metric_code}</span>
                  <div className="flex items-center gap-3">
                    <span className="text-slate-400">
                      7D:{" "}
                      <span className="font-semibold text-slate-600">{e.window_count_7d}</span>
                    </span>
                    <span className="text-slate-400">
                      30D:{" "}
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

const SENTINEL_STATUS_LABELS: Record<string, string> = {
  NOT_PUBLISHED: "Sentinel 未发布",
  SCHEMA_MISMATCH: "特征契约不一致",
  ACTIVE: "运行正常",
  INFERENCE_FAILED: "推理失败",
  NO_FEATURE_ROWS: "无特征行",
};

function SentinelSection({ evidence }: { evidence: SentinelEvidence | null }) {
  if (!evidence) return null;

  const status = evidence.sentinel_status ?? "NOT_PUBLISHED";
  const statusLabel = SENTINEL_STATUS_LABELS[status] || status;
  const isHealthy = status === "ACTIVE";
  const prob = evidence.anomaly_probability ?? 0;
  const pct = Math.min(100, prob * 100);
  const topSignals = evidence.top_signals ?? [];

  // 只有 ACTIVE + triggered=false 才显示绿色"未触发"
  // 其他 status 都显示对应的故障/降级信息
  const triggeredLabel = evidence.triggered ? "已触发" : (isHealthy ? "未触发" : statusLabel);
  const triggeredColor = evidence.triggered
    ? "text-red-600"
    : isHealthy ? "text-emerald-600" : "text-amber-600";

  return (
    <div className={`rounded-lg border px-3 py-3 ${
      isHealthy ? "border-violet-200 bg-violet-50" : "border-amber-200 bg-amber-50"
    }`}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-violet-700">Sentinel 异常检测</span>
        <span className={`text-xs font-semibold ${triggeredColor}`}>
          {triggeredLabel}
        </span>
      </div>

      {!isHealthy && (
        <p className="mt-1 text-[10px] text-amber-700">
          {status === "NOT_PUBLISHED" && "尚无已发布的 Sentinel 模型，仅使用 B1 阈值判定"}
          {status === "SCHEMA_MISMATCH" && "特征契约与当前模型不匹配，需重新训练"}
          {status === "INFERENCE_FAILED" && "推理过程发生异常，详见服务日志"}
          {status === "NO_FEATURE_ROWS" && "特征 DataFrame 为空，无法执行推理"}
        </p>
      )}

      {isHealthy && (
        <>
          <div className="mt-2">
            <div className="flex items-center justify-between text-[10px] text-slate-500">
              <span>异常概率</span>
              <span>{evidence.anomaly_probability != null ? `${pct.toFixed(1)}%` : "-"}</span>
            </div>
            <div className="mt-0.5 h-2 overflow-hidden rounded-full bg-slate-200">
              <div
                className={evidence.triggered ? "h-full bg-red-500" : "h-full bg-violet-500"}
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>

          <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
            <Field
              label="触发阈值"
              value={evidence.alert_threshold != null ? `${(evidence.alert_threshold * 100).toFixed(1)}%` : "-"}
            />
            <Field label="模型版本" value={evidence.sentinel_version || "-"} />
            <Field label="异常窗口" value={evidence.monitor_window_id?.slice(0, 20) || "-"} />
          </div>

          {topSignals.length > 0 && (
            <div className="mt-2">
              <p className="text-[10px] text-slate-500">主要异常信号</p>
              <div className="mt-1 flex flex-wrap gap-1">
                {topSignals.map((signal: string) => (
                  <span key={signal} className="rounded bg-violet-100 px-2 py-0.5 font-mono text-[10px] text-violet-700">
                    {signal}
                  </span>
                ))}
              </div>
            </div>
          )}
        </>
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
