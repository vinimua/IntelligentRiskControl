"use client";

import type { FormEvent, ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";

type Envelope<T> = {
  success?: boolean;
  code?: string;
  message?: string;
  data?: T;
  trace_id?: string;
  details?: unknown;
};

type Items<T> = { items: T[] };

type LifecycleState = {
  lifecycle_run_id?: string;
  model_id?: string;
  champion_version?: string;
  current_phase?: string;
  monitoring_run_id?: string | null;
  diagnosis_run_id?: string | null;
  event_id?: string | null;
  agent_decision_id?: string | null;
  decision_proposal_id?: string | null;
  manual_review_id?: string | null;
  training_plan_id?: string | null;
  training_job_id?: string | null;
  experiment_id?: string | null;
  qualification_run_id?: string | null;
  deployment_id?: string | null;
  recommended_action?: string | null;
  need_iteration?: boolean | null;
  requires_manual_review?: boolean;
  agent_confidence?: number | null;
  primary_root_cause_code?: string | null;
  primary_root_cause_score?: number | null;
  challenger_version?: string | null;
  challenger_qualified?: boolean | null;
  business_round?: number | null;
  iteration_exit_reason?: string | null;
  training_callback_status?: string | null;
  training_dispatched?: boolean | null;
  training_dispatch_mode?: string | null;
  deployment_stage?: string | null;
  deployment_decision?: string | null;
  last_error?: Record<string, unknown> | null;
  [key: string]: unknown;
};

type LifecycleRun = {
  lifecycle_run_id: string;
  current_phase?: string;
  state?: LifecycleState;
};

type ManualReviewReport = {
  review_id: string;
  proposal_id: string;
  reviewer_id: string;
  decision: "APPROVE" | "REJECT";
  reason: string;
  reviewed_at: string;
};

type MonitoringRun = {
  monitoring_run_id: string;
  model_id?: string;
  champion_version?: string;
  overall_status?: string;
  alert_count?: number;
  max_alert_severity?: string | null;
  started_at?: string;
};

type Metric = {
  metric_code?: string;
  current_value?: number | string | null;
  baseline_value?: number | string | null;
  delta?: number | string | null;
  metric_detail?: Record<string, unknown> | null;
};

type TrainingPlanDetail = {
  training_plan_id?: string;
  model_id?: string;
  algorithm?: string;
  status?: string;
  risk_level?: string;
  root_cause_code?: string;
  champion_version?: string;
  frozen_champion_version?: string;
  rollback_target?: string;
  strategy_code?: string;
  strategy_parameters?: Record<string, unknown>;
  random_seed?: number;
  business_round?: number;
  max_business_rounds?: number;
  experiment_id?: string;
  iteration_run_id?: string;
  proposal_id?: string;
  approval_id?: string;
  diagnosis_run_id?: string;
  preprocessing_version?: string;
  feature_schema_version?: string;
  label_versions?: string[];
  data_snapshot_ids?: string[];
  data_eligibility_assessment_ids?: string[];
  target_metric_codes?: string[];
  qualification_rule_version?: string;
  windows?: {
    baseline_window_id?: string;
    training_window_ids?: string[];
    validation_window_ids?: string[];
    oot_window_id?: string;
    oot_locked?: boolean;
  };
  blocking_reasons?: string[];
};

const DEFAULT_API_BASE =
  process.env.NEXT_PUBLIC_MODEL_OPS_API_BASE ?? "http://localhost:8000";

const lifecycleSteps = [
  ["monitoring_run_id", "监控"],
  ["diagnosis_run_id", "诊断"],
  ["event_id", "事件"],
  ["agent_decision_id", "Agent"],
  ["decision_proposal_id", "决策"],
  ["manual_review_id", "复核"],
  ["training_plan_id", "训练计划"],
  ["training_job_id", "训练任务"],
  ["qualification_run_id", "资格验证"],
  ["deployment_id", "部署"],
] as const;

const deploymentStages = [
  "OFFLINE_VALIDATION",
  "OOT_GATE",
  "SHADOW",
  "CANARY_5",
  "CANARY_20",
  "CANARY_50",
  "PRODUCTION",
];

const deploymentStageMeta: Record<string, { label: string; desc: string }> = {
  OFFLINE_VALIDATION: {
    label: "离线验证",
    desc: "先用离线数据检查候选模型是否安全。",
  },
  OOT_GATE: {
    label: "跨期验证",
    desc: "用未来窗口数据验证模型是否稳健。",
  },
  SHADOW: {
    label: "影子部署",
    desc: "新模型只旁路打分，不影响真实业务。",
  },
  CANARY_5: {
    label: "灰度 5%",
    desc: "只让 5% 流量试用新模型。",
  },
  CANARY_20: {
    label: "灰度 20%",
    desc: "扩大到 20% 流量，继续观察指标。",
  },
  CANARY_50: {
    label: "灰度 50%",
    desc: "半量流量验证，确认风险可控。",
  },
  PRODUCTION: {
    label: "全量生产",
    desc: "候选模型通过检查，成为生产版本。",
  },
};

const terminalPhases = new Set([
  "EVENT_CLOSED",
  "NO_ALERT",
  "FAILED",
  "COMPLETED",
  "PROMOTED",
  "ROLLED_BACK",
]);

const metricCodes = new Set([
  "AUC",
  "KS",
  "BAD_RATE",
  "PREDICTION_MEAN",
  "SCORE_PSI",
  "FEATURE_PSI",
  "SAMPLE_SIZE",
]);

const phaseMeta: Record<string, { label: string; desc: string }> = {
  NO_RUN: { label: "尚未启动", desc: "还没有创建生命周期流程。" },
  CREATED: { label: "已创建", desc: "后端已创建 lifecycle_run，准备进入监控。" },
  MONITORING: { label: "监控中", desc: "正在检查模型指标和漂移告警。" },
  MONITORING_COMPLETED: { label: "监控完成", desc: "监控已完成，准备根据告警决定是否诊断。" },
  NO_ALERT: { label: "无告警关闭", desc: "没有发现异常，本次流程结束。" },
  DIAGNOSING: { label: "诊断中", desc: "正在分析异常根因。" },
  DIAGNOSIS_COMPLETED: { label: "诊断完成", desc: "已经得到根因和建议动作。" },
  WAITING_AGENT_DECISION: { label: "等待 Agent 决策", desc: "诊断结果已交给 Agent/规则决策器。" },
  AGENT_DECIDING: { label: "Agent 决策中", desc: "正在判断下一步处理方向。" },
  DECISION_PROPOSED: { label: "等待人工复核", desc: "已生成决策建议，需要人工确认后继续。" },
  MANUAL_REVIEW: { label: "人工复核中", desc: "流程暂停，等待前端提交复核意见。" },
  ITERATING: { label: "迭代处理中", desc: "正在生成修复、训练或外部执行计划。" },
  WAITING_TRAINING_CALLBACK: { label: "等待训练回调", desc: "训练任务已派发，等待 Worker 自动回调。" },
  CHALLENGER_TRAINED: { label: "候选模型已训练", desc: "challenger 模型训练完成，准备验证。" },
  OFFLINE_VALIDATING: { label: "离线验证中", desc: "正在执行资格验证和质量门禁。" },
  QUALIFICATION_COMPLETED: { label: "资格验证完成", desc: "候选模型已经完成上线前检查。" },
  CANARY_RUNNING: { label: "灰度部署中", desc: "部署阶段正在逐步推进。" },
  PROMOTED: { label: "已提升生产", desc: "候选模型已被提升为生产版本。" },
  ROLLED_BACK: { label: "已回滚", desc: "部署失败或健康检查不通过，已回滚。" },
  EVENT_CLOSED: { label: "事件已关闭", desc: "本次异常处理闭环已经完成。" },
  FAILED: { label: "流程失败", desc: "流程遇到无法自动处理的问题。" },
};

const actionMeta: Record<string, { label: string; desc: string }> = {
  NO_ACTION: { label: "无需处理", desc: "监控或诊断认为不用采取修复动作。" },
  CONTINUE_OBSERVATION: { label: "继续观察", desc: "暂不修复，后续继续监控指标变化。" },
  DATA_REPAIR: { label: "数据修复", desc: "数据质量或特征数据异常，需要修数据。" },
  PIPELINE_REPAIR: { label: "管道修复", desc: "数据处理链路或 ETL 逻辑异常，需要修管道。" },
  CALIBRATION_ADJUSTMENT: { label: "校准调整", desc: "模型排序可用，但概率分数需要重新校准。" },
  THRESHOLD_ADJUSTMENT: { label: "阈值调整", desc: "模型可用，但业务截断阈值需要重新搜索。" },
  MODEL_ITERATION: { label: "模型迭代", desc: "需要训练 challenger 模型并通过资格验证。" },
  MANUAL_REVIEW: { label: "人工判断", desc: "系统无法自动定夺，需要人工复核处理方向。" },
};

function describePhase(value?: string | null) {
  const key = value || "NO_RUN";
  return phaseMeta[key] ?? { label: key, desc: "当前阶段暂未配置中文说明。" };
}

function describeAction(value?: string | null) {
  const key = value || "";
  return actionMeta[key] ?? {
    label: key || "暂无修复方向",
    desc: key ? "当前修复方向暂未配置中文说明。" : "流程尚未生成 recommended_action。",
  };
}

function formatDeploymentDecision(value?: unknown): string {
  const key = String(value || "");
  const labels: Record<string, string> = {
    PROMOTE: "提升为生产",
    ADVANCE_STAGE: "进入下一阶段",
    HOLD: "暂停观察",
    PAUSE_CANARY: "暂停灰度",
    REDUCE_TRAFFIC: "降低流量",
    ROLLBACK: "回滚",
    MANUAL_REVIEW: "需要人工确认",
    ABORT_DEPLOYMENT: "终止部署",
  };
  return labels[key] || formatValue(value);
}

function formatValue(value: unknown): string {
  if (value === undefined || value === null || value === "") return "-";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  if (typeof value === "boolean") return value ? "是" : "否";
  return String(value);
}

function joinValues(values?: Array<string | number | boolean> | null): string {
  return values && values.length > 0 ? values.map((value) => formatValue(value)).join("、") : "-";
}

function formatTrainingStrategy(value?: string): string {
  const labels: Record<string, string> = {
    recent_weighted_retrain: "近期加权重训",
  };
  return value ? `${labels[value] || value}（${value}）` : "-";
}

function formatRootCause(value?: string): string {
  const labels: Record<string, string> = {
    FEATURE_DRIFT: "特征漂移",
    feature_drift: "特征漂移",
    LABEL_DRIFT: "标签漂移",
    DATA_QUALITY: "数据质量异常",
    PERFORMANCE_DROP: "模型效果下降",
  };
  return value ? `${labels[value] || value}（${value}）` : "-";
}

function formatRiskLevel(value?: string): string {
  const labels: Record<string, string> = {
    LOW: "低风险",
    MEDIUM: "中风险",
    HIGH: "高风险",
  };
  return value ? `${labels[value] || value}（${value}）` : "-";
}

function phaseClass(value?: string | null): string {
  const phase = (value ?? "").toUpperCase();
  if (/FAILED|ERROR|REJECT|ROLLBACK/.test(phase)) return "border-red-200 bg-red-50 text-red-700";
  if (/WAITING|MANUAL|PENDING/.test(phase)) return "border-amber-200 bg-amber-50 text-amber-700";
  if (/CLOSED|COMPLETED|PROMOTED|SUCCEEDED/.test(phase)) {
    return "border-emerald-200 bg-emerald-50 text-emerald-700";
  }
  if (/AGENT|ITERAT|DIAGNOS|MONITOR|VALIDAT|CANARY/.test(phase)) {
    return "border-sky-200 bg-sky-50 text-sky-700";
  }
  return "border-slate-200 bg-white text-slate-700";
}

async function requestJson<T>(apiBase: string, path: string, init?: RequestInit): Promise<T> {
  const base = apiBase.trim().replace(/\/+$/, "");
  const targetUrl = `${base}${path}`;
  const url = `/api/modelops${path}`;

  let response: Response;
  try {
    response = await fetch(url, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        "X-ModelOps-Api-Base": base,
        ...(init?.headers ?? {}),
      },
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : "network error";
    throw new Error(`无法连接后端接口：${targetUrl}。请确认后端已启动，API 地址正确。原始错误：${detail}`);
  }

  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    payload = { message: await response.text() };
  }

  if (!response.ok) {
    const envelope = payload as Envelope<unknown>;
    throw new Error(
      envelope.message ||
        (typeof envelope.details === "string" ? envelope.details : JSON.stringify(payload)),
    );
  }

  const envelope = payload as Envelope<T>;
  return envelope.data === undefined ? (payload as T) : envelope.data;
}

export default function Page() {
  const [apiBase, setApiBase] = useState(DEFAULT_API_BASE);
  const [activeTab, setActiveTab] = useState<"workflow" | "monitoring" | "state">("workflow");
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<{ type: "ok" | "error"; text: string } | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [showManualCallback, setShowManualCallback] = useState(false);

  const [monitoringRuns, setMonitoringRuns] = useState<MonitoringRun[]>([]);
  const [selectedMonitoringRunId, setSelectedMonitoringRunId] = useState("");
  const [metrics, setMetrics] = useState<Metric[]>([]);

  const [modelId, setModelId] = useState("credit_model_001");
  const [championVersion, setChampionVersion] = useState("champion_v1");
  const [triggerType, setTriggerType] = useState("SCHEDULED_TRIGGER");
  const [runId, setRunId] = useState("");
  const [reviewerId, setReviewerId] = useState("admin");
  const [reviewReason, setReviewReason] = useState("确认本次修复方向，允许进入真实训练链路。");
  const [callbackStatus, setCallbackStatus] = useState("SUCCEEDED");
  const [candidateVersion, setCandidateVersion] = useState("v1_challenger_manual");
  const [experimentId, setExperimentId] = useState("");
  const [lifecycleRun, setLifecycleRun] = useState<LifecycleRun | null>(null);
  const [trainingPlan, setTrainingPlan] = useState<TrainingPlanDetail | null>(null);

  const state = useMemo(() => lifecycleRun?.state ?? {}, [lifecycleRun]);
  const currentRunId = runId || lifecycleRun?.lifecycle_run_id || "";
  const currentPhase = lifecycleRun?.current_phase || state.current_phase || "NO_RUN";
  const currentPhaseMeta = describePhase(currentPhase);
  const currentAction = String(state.recommended_action || "");
  const currentActionMeta = describeAction(currentAction);
  const currentTrainingPlanId = String(state.training_plan_id || "");
  const trainingJobId = String(state.training_job_id || "");
  const decisionProposalId = String(state.decision_proposal_id || "");
  const currentExperimentId = experimentId || String(state.experiment_id || "");
  const iterationRunId = String(state.iteration_run_id || "");
  const businessRound = Number(state.business_round || 1);
  const isTerminal = terminalPhases.has(String(currentPhase));
  const completedSteps = lifecycleSteps.filter(([key]) => Boolean(state[key])).length;
  const progress = Math.round((completedSteps / lifecycleSteps.length) * 100);
  const deploymentIndex = Math.max(
    deploymentStages.findIndex((stage) => stage === state.deployment_stage),
    state.deployment_id ? 0 : -1,
  );

  const coreMetrics = useMemo(
    () => metrics.filter((metric) => metricCodes.has(String(metric.metric_code))),
    [metrics],
  );

  const driftRows = useMemo(
    () =>
      metrics
        .filter((metric) => metric.metric_detail?.category === "drift")
        .map((metric) => ({
          name: String(metric.metric_detail?.feature_name ?? metric.metric_code ?? "-"),
          value: Number(metric.current_value ?? 0),
        }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 10),
    [metrics],
  );

  useEffect(() => {
    if (!autoRefresh || !currentRunId || isTerminal) return;
    const timer = window.setInterval(() => {
      void loadLifecycleRun(currentRunId, false);
    }, 3000);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRefresh, currentRunId, isTerminal]);

  useEffect(() => {
    if (!currentTrainingPlanId) {
      setTrainingPlan(null);
      return;
    }

    let cancelled = false;
    async function loadTrainingPlanDetail() {
      try {
        const detail = await requestJson<TrainingPlanDetail>(
          apiBase,
          `/api/iteration/plans/${currentTrainingPlanId}`,
        );
        if (!cancelled) setTrainingPlan(detail);
      } catch {
        if (!cancelled) setTrainingPlan(null);
      }
    }

    void loadTrainingPlanDetail();
    return () => {
      cancelled = true;
    };
  }, [apiBase, currentTrainingPlanId]);

  async function runAction<T>(
    key: string,
    action: () => Promise<T>,
    successText: string,
    showSuccess = true,
  ) {
    setBusy(key);
    if (showSuccess) setMessage(null);
    try {
      const result = await action();
      if (showSuccess) setMessage({ type: "ok", text: successText });
      return result;
    } catch (error) {
      setMessage({
        type: "error",
        text: error instanceof Error ? error.message : "请求失败，请稍后重试。",
      });
      return null;
    } finally {
      setBusy(null);
    }
  }

  async function testApi() {
    await runAction("health", () => requestJson(apiBase, "/health/live"), "后端连接正常。");
  }

  async function loadMonitoringRuns() {
    const data = await runAction(
      "monitoring",
      () => requestJson<Items<MonitoringRun>>(apiBase, "/api/monitoring/runs?limit=20"),
      "监控运行已加载。",
    );
    if (!data) return;
    setMonitoringRuns(data.items);
    const firstRunId = data.items[0]?.monitoring_run_id;
    if (firstRunId) {
      setSelectedMonitoringRunId(firstRunId);
      await loadMetrics(firstRunId);
    }
  }

  async function loadMetrics(monitoringRunId = selectedMonitoringRunId) {
    if (!monitoringRunId) {
      setMessage({ type: "error", text: "请先选择一个监控运行。" });
      return;
    }
    const data = await runAction(
      "metrics",
      () => requestJson<Items<Metric>>(apiBase, `/api/monitoring/runs/${monitoringRunId}/metrics`),
      "指标已加载。",
    );
    if (data) setMetrics(data.items);
  }

  async function loadLifecycleRun(id = currentRunId, showSuccess = true) {
    if (!id) {
      setMessage({ type: "error", text: "请先输入 lifecycle_run_id。" });
      return null;
    }
    const data = await runAction(
      "load-run",
      () => requestJson<LifecycleRun>(apiBase, `/api/lifecycle-runs/${id}`),
      "生命周期状态已刷新。",
      showSuccess,
    );
    if (data) {
      setLifecycleRun(data);
      setRunId(data.lifecycle_run_id);
    }
    return data;
  }

  async function startLifecycle(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    const data = await runAction(
      "start",
      () =>
        requestJson<LifecycleRun>(apiBase, "/api/lifecycle-runs", {
          method: "POST",
          body: JSON.stringify({
            model_id: modelId,
            champion_version: championVersion,
            trigger_type: triggerType,
          }),
        }),
      "生命周期已启动，正在等待人工复核。",
    );
    if (data) {
      setLifecycleRun(data);
      setRunId(data.lifecycle_run_id);
      setTrainingPlan(null);
      setActiveTab("workflow");
    }
  }

  async function resumeLifecycle(payload: Record<string, unknown>, actionKey: string) {
    if (!currentRunId) {
      setMessage({ type: "error", text: "请先启动或加载一个生命周期运行。" });
      return null;
    }
    const data = await runAction(
      actionKey,
      () =>
        requestJson<LifecycleRun>(apiBase, `/api/lifecycle-runs/${currentRunId}/resume`, {
          method: "POST",
          body: JSON.stringify(payload),
        }),
      "生命周期已恢复。",
    );
    if (data) {
      setLifecycleRun(data);
      setRunId(data.lifecycle_run_id);
    }
    return data;
  }

  async function submitManualReview(decision: "APPROVE" | "REJECT") {
    if (!currentRunId || !decisionProposalId) {
      setMessage({ type: "error", text: "当前流程还没有可复核的决策建议。" });
      return;
    }
    const isApproved = decision === "APPROVE";
    const reason = reviewReason.trim() || (isApproved ? "人工确认通过。" : "人工确认拒绝。");
    const report = await runAction(
      isApproved ? "approve-review" : "reject-review",
      () =>
        requestJson<ManualReviewReport>(
          apiBase,
          `/api/iteration/decisions/${decisionProposalId}/reviews`,
          {
            method: "POST",
            body: JSON.stringify({
              proposal_id: decisionProposalId,
              reviewer_id: reviewerId.trim() || "admin",
              decision,
              reason,
              rejection_reason_codes: isApproved ? [] : ["MANUAL_REJECTED"],
              adjustment_instructions: isApproved ? [] : ["请重新生成修复建议并补充风险证据。"],
              forbidden_adjustments: [],
              expected_evidence: [],
              reviewed_at: new Date().toISOString(),
            }),
          },
        ),
      isApproved ? "人工复核已通过，真实 Worker 将自动接管训练。" : "人工复核已拒绝。",
    );
    if (!report) return;

    await resumeLifecycle(
      {
        decision: isApproved ? "approved" : "rejected",
        manual_review_id: report.review_id,
        review_id: report.review_id,
      },
      isApproved ? "approve" : "reject",
    );
  }

  async function submitTrainingCallback() {
    if (!currentRunId || !trainingJobId || !currentExperimentId || !iterationRunId) {
      setMessage({ type: "error", text: "当前流程缺少训练任务、实验或迭代 ID，不能手动回调。" });
      return;
    }

    const normalizedStatus = callbackStatus.trim() || "SUCCEEDED";
    const normalizedCandidate =
      candidateVersion.trim() ||
      String(state.challenger_version || `${state.champion_version || "champion"}_challenger_v1`);

    const callback = await runAction(
      "callback-worker",
      () =>
        requestJson<{ training_job_id: string; callback_applied: boolean; lifecycle_resumed?: boolean }>(
          apiBase,
          `/api/internal/iteration/jobs/${trainingJobId}/callback`,
          {
            method: "POST",
            body: JSON.stringify({
              training_job_id: trainingJobId,
              lifecycle_run_id: currentRunId,
              idempotency_key: `${iterationRunId}:round-${businessRound}:exp-${currentExperimentId}`,
              experiment_id: currentExperimentId,
              status: normalizedStatus,
              candidate_version: normalizedCandidate,
              model_artifact_uri:
                normalizedStatus === "SUCCEEDED"
                  ? `s3://riskitem/demo/models/${normalizedCandidate}`
                  : undefined,
              training_metrics: { auc: 0.81, ks: 0.43 },
              validation_metrics: {
                original_drop: 0.04,
                recovered_amount: 0.035,
                recovery_rate: 0.875,
                champion_auc: 0.74,
                challenger_auc: 0.775,
                healthy_lower_bound: 0.76,
                bootstrap_ci_lower: 0.01,
                bootstrap_ci_upper: 0.06,
                discrimination_passed: true,
                calibration_passed: true,
                score_psi: 0.08,
                train_valid_gap: 0.015,
                oot_passed: true,
              },
              segment_metrics: { segment_governance_passed: true },
              artifact_checksums: {},
              environment_manifest: { runtime: "frontend-manual-fallback" },
              technical_retry_count: 0,
            }),
          },
        ),
      "手动训练回调已提交。",
    );
    if (callback) await loadLifecycleRun(currentRunId, false);
  }

  const primaryAction = getPrimaryAction(currentPhase, Boolean(decisionProposalId));
  const workerStatus =
    state.training_dispatch_mode === "celery"
      ? state.training_callback_status
        ? `真实 Worker 已回调：${state.training_callback_status}`
        : "真实 Worker 已派发，正在训练 W2/W3 数据"
      : state.training_job_id
        ? "当前为手动回调兜底模式"
        : "尚未创建训练任务";

  return (
    <main className="min-h-screen bg-slate-100 text-slate-950">
      <div className="mx-auto flex max-w-7xl flex-col gap-5 px-5 py-5">
        <header className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-sky-700">
                RiskItem ModelOps
              </p>
              <h1 className="mt-1 text-2xl font-semibold tracking-normal">
                智能风控模型生命周期控制台
              </h1>
              <p className="mt-2 text-sm text-slate-500">
                默认走真实 Celery Worker 训练链路；页面会自动刷新训练、验证和部署状态。
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <StatusBadge value={currentPhase} />
              <button
                className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-semibold hover:bg-slate-50"
                onClick={testApi}
                disabled={busy === "health"}
              >
                测试后端
              </button>
              <a
                className="rounded-md bg-slate-950 px-3 py-2 text-sm font-semibold text-white hover:bg-slate-800"
                href={`${apiBase}/docs`}
                target="_blank"
              >
                API 文档
              </a>
            </div>
          </div>

          <div className="mt-4 grid gap-3 lg:grid-cols-[1fr_auto_auto]">
            <input
              className="rounded-md border border-slate-300 bg-slate-50 px-3 py-2 font-mono text-sm"
              value={apiBase}
              onChange={(event) => setApiBase(event.target.value)}
            />
            <label className="flex items-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-sm text-slate-600">
              <input
                checked={autoRefresh}
                onChange={(event) => setAutoRefresh(event.target.checked)}
                type="checkbox"
              />
              自动刷新
            </label>
            <button
              className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-semibold hover:bg-slate-50"
              onClick={() => loadLifecycleRun()}
              disabled={!currentRunId || busy === "load-run"}
            >
              刷新当前流程
            </button>
          </div>
        </header>

        {message ? <Message type={message.type}>{message.text}</Message> : null}

        <nav className="flex flex-wrap gap-2">
          <Tab active={activeTab === "workflow"} onClick={() => setActiveTab("workflow")}>
            流程控制
          </Tab>
          <Tab active={activeTab === "monitoring"} onClick={() => setActiveTab("monitoring")}>
            监控看板
          </Tab>
          <Tab active={activeTab === "state"} onClick={() => setActiveTab("state")}>
            状态详情
          </Tab>
        </nav>

        {activeTab === "workflow" ? (
          <section className="grid gap-5 lg:grid-cols-[390px_1fr]">
            <div className="flex flex-col gap-5">
              <Panel title="一键流程">
                <form className="grid gap-3" onSubmit={startLifecycle}>
                  <Input label="模型 ID" value={modelId} onChange={setModelId} />
                  <Input label="Champion 版本" value={championVersion} onChange={setChampionVersion} />
                  <label className="grid gap-1 text-sm">
                    <span className="font-medium text-slate-600">触发类型</span>
                    <select
                      className="rounded-md border border-slate-300 px-3 py-2"
                      value={triggerType}
                      onChange={(event) => setTriggerType(event.target.value)}
                    >
                      <option>SCHEDULED_TRIGGER</option>
                      <option>THRESHOLD_TRIGGER</option>
                      <option>ABNORMAL_TRIGGER</option>
                      <option>MANUAL_TRIGGER</option>
                    </select>
                  </label>
                  <PrimaryButton disabled={busy === "start"}>
                    {currentRunId ? "启动新的生命周期" : "启动生命周期"}
                  </PrimaryButton>
                </form>
                <div className="mt-4 rounded-md border border-sky-100 bg-sky-50 p-3 text-sm text-sky-800">
                  推荐动作：{primaryAction}
                </div>
              </Panel>

              <Panel title="智能操作">
                <div className="grid gap-3">
                  <PrimaryButton
                    disabled={!decisionProposalId || busy === "approve-review" || busy === "approve"}
                    onClick={() => submitManualReview("APPROVE")}
                  >
                    通过人工复核并启动真实训练
                  </PrimaryButton>
                  <SecondaryButton
                    disabled={!decisionProposalId || busy === "reject-review" || busy === "reject"}
                    onClick={() => submitManualReview("REJECT")}
                  >
                    拒绝本次建议
                  </SecondaryButton>
                  <Input label="复核人" value={reviewerId} onChange={setReviewerId} />
                  <Input label="复核意见" value={reviewReason} onChange={setReviewReason} />
                  <ReadOnly label="人工复核 ID" value={String(state.manual_review_id || "")} />
                  <ReadOnly label="决策建议 ID" value={decisionProposalId} />
                </div>
              </Panel>

              <Panel title="真实 Worker">
                <div className="grid gap-3">
                  <StatusLine label="派发模式" value={formatValue(state.training_dispatch_mode)} />
                  <StatusLine label="派发状态" value={workerStatus} />
                  <StatusLine label="训练任务" value={trainingJobId} mono />
                  <StatusLine label="实验 ID" value={String(state.experiment_id || "")} mono />
                  <StatusLine label="回调状态" value={formatValue(state.training_callback_status)} />
                  <button
                    className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-semibold hover:bg-slate-50"
                    onClick={() => setShowManualCallback((value) => !value)}
                  >
                    {showManualCallback ? "收起手动兜底回调" : "显示手动兜底回调"}
                  </button>
                  {showManualCallback ? (
                    <div className="grid gap-3 rounded-md border border-amber-200 bg-amber-50 p-3">
                      <label className="grid gap-1 text-sm">
                        <span className="font-medium text-slate-600">回调状态</span>
                        <select
                          className="rounded-md border border-slate-300 px-3 py-2"
                          value={callbackStatus}
                          onChange={(event) => setCallbackStatus(event.target.value)}
                        >
                          <option>SUCCEEDED</option>
                          <option>FAILED</option>
                          <option>CANCELLED</option>
                        </select>
                      </label>
                      <Input label="候选版本" value={candidateVersion} onChange={setCandidateVersion} />
                      <Input label="实验 ID" value={currentExperimentId} onChange={setExperimentId} />
                      <SecondaryButton
                        disabled={!trainingJobId || busy === "callback-worker"}
                        onClick={submitTrainingCallback}
                      >
                        提交手动回调
                      </SecondaryButton>
                    </div>
                  ) : null}
                </div>
              </Panel>
            </div>

            <div className="flex flex-col gap-5">
              <Panel title="当前生命周期">
                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                  <MetricCard
                    label="阶段"
                    value={currentPhaseMeta.label}
                    sub={`${currentPhase}：${currentPhaseMeta.desc}`}
                  />
                  <MetricCard
                    label="修复方向"
                    value={currentActionMeta.label}
                    sub={`${formatValue(state.recommended_action)}：${currentActionMeta.desc}`}
                  />
                  <MetricCard
                    label="业务轮次"
                    value={formatValue(state.business_round)}
                    sub="第几轮自动迭代；模型训练最多尝试 3 轮。"
                  />
                  <MetricCard
                    label="Agent 置信度"
                    value={formatValue(state.agent_confidence)}
                    sub="Agent/规则决策对当前修复方向的把握程度。"
                  />
                </div>
                <div className="mt-4 break-all rounded-md bg-slate-50 p-3 font-mono text-xs text-slate-600">
                  lifecycle_run_id: {formatValue(currentRunId)}
                </div>
                <div className="mt-4 grid gap-3 lg:grid-cols-2">
                  <ReferenceBox
                    title="常见阶段"
                    items={[
                      ["DECISION_PROPOSED", "已生成建议，等待人工复核"],
                      ["WAITING_TRAINING_CALLBACK", "Worker 正在训练，等待自动回调"],
                      ["OFFLINE_VALIDATING", "正在做资格验证"],
                      ["CANARY_RUNNING", "部署阶段推进中"],
                      ["EVENT_CLOSED", "本次生命周期已完成"],
                      ["FAILED", "流程失败，需要查看错误详情"],
                    ]}
                  />
                  <ReferenceBox
                    title="修复方向"
                    items={[
                      ["MODEL_ITERATION", "重新训练 challenger 模型"],
                      ["DATA_REPAIR", "修复数据质量或特征数据"],
                      ["PIPELINE_REPAIR", "修复数据处理管道"],
                      ["CALIBRATION_ADJUSTMENT", "重新校准模型概率"],
                      ["THRESHOLD_ADJUSTMENT", "重新搜索业务阈值"],
                      ["CONTINUE_OBSERVATION", "暂不处理，继续观察"],
                    ]}
                  />
                </div>
              </Panel>

              <Panel title="流程进度">
                <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-5">
                  {lifecycleSteps.map(([key, label]) => (
                    <StepTile active={Boolean(state[key])} key={key} label={label} value={state[key]} />
                  ))}
                </div>
                <ProgressBar value={Math.max(progress, currentRunId ? 5 : 0)} />
              </Panel>

              <Panel title="部署进度">
                <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-7">
                  {deploymentStages.map((stage, index) => {
                    const meta = deploymentStageMeta[stage];
                    const active = index <= deploymentIndex;
                    return (
                    <div
                      className={`rounded-md border px-3 py-3 ${
                        active
                          ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                          : "border-slate-200 bg-white text-slate-400"
                      }`}
                      key={stage}
                    >
                      <p className="text-center text-sm font-semibold">{meta.label}</p>
                      <p className="mt-1 break-all text-center font-mono text-[10px] font-semibold">{stage}</p>
                      <p className={`mt-2 text-center text-[11px] leading-4 ${active ? "text-emerald-700" : "text-slate-400"}`}>
                        {meta.desc}
                      </p>
                    </div>
                    );
                  })}
                </div>
                <div className="mt-4 grid gap-3 sm:grid-cols-3">
                  <MetricCard label="部署 ID" value={formatValue(state.deployment_id)} />
                  <MetricCard
                    label="部署阶段"
                    value={
                      state.deployment_stage
                        ? deploymentStageMeta[String(state.deployment_stage)]?.label || formatValue(state.deployment_stage)
                        : "-"
                    }
                    sub={state.deployment_stage ? `${formatValue(state.deployment_stage)}：${deploymentStageMeta[String(state.deployment_stage)]?.desc || ""}` : "尚未进入部署。"}
                  />
                  <MetricCard
                    label="部署决策"
                    value={formatDeploymentDecision(state.deployment_decision)}
                    sub={state.deployment_decision ? `原始值：${formatValue(state.deployment_decision)}` : "尚未产生部署决策。"}
                  />
                </div>
              </Panel>

              <Panel title="关键结果">
                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                  <MetricCard label="训练计划" value={formatValue(state.training_plan_id)} />
                  <MetricCard label="候选版本" value={formatValue(state.challenger_version)} />
                  <MetricCard label="资格验证" value={formatValue(state.qualification_run_id)} />
                  <MetricCard label="是否合格" value={formatValue(state.challenger_qualified)} />
                </div>
                <TrainingPlanDetailPanel plan={trainingPlan} trainingPlanId={currentTrainingPlanId} />
              </Panel>
            </div>
          </section>
        ) : null}

        {activeTab === "monitoring" ? (
          <section className="grid gap-5 lg:grid-cols-[360px_1fr]">
            <Panel
              title="监控运行"
              action={
                <SecondaryButton disabled={busy === "monitoring"} onClick={loadMonitoringRuns}>
                  加载监控
                </SecondaryButton>
              }
            >
              <div className="max-h-[640px] space-y-2 overflow-auto pr-1">
                {monitoringRuns.length === 0 ? (
                  <Empty text="尚未加载监控运行。" />
                ) : (
                  monitoringRuns.map((run) => (
                    <button
                      className={`w-full rounded-md border px-3 py-3 text-left transition ${
                        selectedMonitoringRunId === run.monitoring_run_id
                          ? "border-sky-300 bg-sky-50"
                          : "border-slate-200 bg-white hover:bg-slate-50"
                      }`}
                      key={run.monitoring_run_id}
                      onClick={() => {
                        setSelectedMonitoringRunId(run.monitoring_run_id);
                        void loadMetrics(run.monitoring_run_id);
                      }}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate text-sm font-semibold">{formatValue(run.model_id)}</span>
                        <span className="text-xs font-semibold text-amber-600">
                          {formatValue(run.alert_count)} 个告警
                        </span>
                      </div>
                      <div className="mt-1 truncate font-mono text-xs text-slate-500">
                        {formatValue(run.champion_version)} / {formatValue(run.started_at)}
                      </div>
                    </button>
                  ))
                )}
              </div>
            </Panel>

            <div className="flex flex-col gap-5">
              <Panel title="核心指标">
                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                  {coreMetrics.length === 0 ? (
                    <Empty text="请选择一个监控运行以显示指标。" />
                  ) : (
                    coreMetrics.slice(0, 8).map((metric, index) => (
                      <MetricCard
                        key={`${metric.metric_code}-${index}`}
                        label={formatValue(metric.metric_code)}
                        value={formatValue(metric.current_value)}
                        sub={`基线 ${formatValue(metric.baseline_value)} / 变化 ${formatValue(metric.delta)}`}
                      />
                    ))
                  )}
                </div>
              </Panel>
              <Panel title="特征漂移 Top 10">
                <BarList rows={driftRows} emptyText="未找到漂移指标。" />
              </Panel>
            </div>
          </section>
        ) : null}

        {activeTab === "state" ? (
          <Panel title="状态详情">
            <pre className="max-h-[680px] overflow-auto rounded-md bg-slate-950 p-4 text-xs leading-relaxed text-slate-100">
              {JSON.stringify(state, null, 2)}
            </pre>
          </Panel>
        ) : null}
      </div>
    </main>
  );
}

function getPrimaryAction(phase: string, hasProposal: boolean): string {
  if (!phase || phase === "NO_RUN") return "启动生命周期";
  if (phase === "DECISION_PROPOSED" && hasProposal) return "通过人工复核";
  if (phase === "WAITING_TRAINING_CALLBACK") return "等待真实 Worker 回调";
  if (phase === "EVENT_CLOSED") return "流程已闭环";
  return "刷新状态";
}

function Panel({
  title,
  action,
  children,
}: {
  title: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold">{title}</h2>
        {action}
      </div>
      {children}
    </section>
  );
}

function Tab({ active, children, onClick }: { active: boolean; children: ReactNode; onClick: () => void }) {
  return (
    <button
      className={`rounded-md px-4 py-2 text-sm font-semibold ${
        active
          ? "bg-slate-950 text-white"
          : "border border-slate-300 bg-white text-slate-700 hover:bg-slate-50"
      }`}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

function Input({
  label,
  value,
  placeholder,
  onChange,
}: {
  label: string;
  value: string;
  placeholder?: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="grid gap-1 text-sm">
      <span className="font-medium text-slate-600">{label}</span>
      <input
        className="rounded-md border border-slate-300 px-3 py-2"
        placeholder={placeholder}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}

function ReadOnly({ label, value }: { label: string; value: string }) {
  return (
    <label className="grid gap-1 text-sm">
      <span className="font-medium text-slate-600">{label}</span>
      <input
        className="rounded-md border border-slate-300 bg-slate-50 px-3 py-2 font-mono text-xs"
        value={value}
        readOnly
        placeholder="由流程生成"
      />
    </label>
  );
}

function PrimaryButton({
  children,
  disabled,
  onClick,
}: {
  children: ReactNode;
  disabled?: boolean;
  onClick?: () => void;
}) {
  return (
    <button
      className="rounded-md bg-slate-950 px-3 py-2 text-sm font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
      disabled={disabled}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

function SecondaryButton({
  children,
  disabled,
  onClick,
}: {
  children: ReactNode;
  disabled?: boolean;
  onClick?: () => void;
}) {
  return (
    <button
      className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-semibold hover:bg-slate-50 disabled:cursor-not-allowed disabled:text-slate-400"
      disabled={disabled}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

function MetricCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-3">
      <p className="text-xs font-medium text-slate-500">{label}</p>
      <p className="mt-1 break-words text-lg font-semibold text-slate-950">{value}</p>
      {sub ? <p className="mt-1 text-xs text-slate-500">{sub}</p> : null}
    </div>
  );
}

function TrainingPlanDetailPanel({
  plan,
  trainingPlanId,
}: {
  plan: TrainingPlanDetail | null;
  trainingPlanId: string;
}) {
  if (!trainingPlanId) {
    return (
      <div className="mt-4">
        <Empty text="还没有生成训练计划。流程通过人工复核后，后端才会生成具体训练计划。" />
      </div>
    );
  }

  if (!plan) {
    return (
      <div className="mt-4">
        <Empty text="正在读取训练计划详情，或后端暂时没有返回该计划。" />
      </div>
    );
  }

  const windows = plan.windows ?? {};
  const roundText =
    plan.business_round || plan.max_business_rounds
      ? `第 ${formatValue(plan.business_round)} / ${formatValue(plan.max_business_rounds)} 轮`
      : "-";

  return (
    <div className="mt-5 space-y-4">
      <div className="rounded-md border border-sky-200 bg-sky-50 px-4 py-3">
        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-sm font-semibold text-sky-900">训练计划详情</p>
            <p className="mt-1 text-xs leading-5 text-sky-700">
              这不是单纯的 ID。它是后端为本次模型迭代生成的执行说明，Worker 会按这里的数据窗口、算法和验收规则去训练候选模型。
            </p>
          </div>
          <span className="break-all rounded-md bg-white px-3 py-2 font-mono text-[11px] font-semibold text-sky-800">
            {trainingPlanId}
          </span>
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <DetailGroup
          title="1. 训练对象"
          rows={[
            ["模型", formatValue(plan.model_id), "这次要被修复或迭代的模型。"],
            ["算法", formatValue(plan.algorithm), "当前演示链路使用 LightGBM 训练 challenger。"],
            ["当前生产版本", formatValue(plan.frozen_champion_version || plan.champion_version), "训练开始时冻结的 champion。"],
            ["候选模型回滚目标", formatValue(plan.rollback_target), "如果部署失败，回到这个版本。"],
            ["根因", formatRootCause(plan.root_cause_code), "诊断阶段认为最主要的问题来源。"],
            ["风险等级", formatRiskLevel(plan.risk_level), "用于决定是否需要更严格复核和门禁。"],
          ]}
        />

        <DetailGroup
          title="2. 数据窗口"
          rows={[
            ["基线窗口", formatValue(windows.baseline_window_id), "用于对比正常时期指标。"],
            ["训练窗口", joinValues(windows.training_window_ids), "Worker 实际拿来训练模型的数据。"],
            ["验证窗口", joinValues(windows.validation_window_ids), "训练后先在这个窗口验证效果。"],
            [
              "OOT 窗口",
              `${formatValue(windows.oot_window_id)} / ${windows.oot_locked ? "已锁定" : "未锁定"}`,
              "跨期验证窗口，只做最终稳定性检查，不参与训练。",
            ],
            ["数据快照", joinValues(plan.data_snapshot_ids), "保证训练和验证使用可复现的数据版本。"],
            ["标签版本", joinValues(plan.label_versions), "训练使用的 is_bad 标签版本。"],
          ]}
        />

        <DetailGroup
          title="3. 训练配置"
          rows={[
            ["策略", formatTrainingStrategy(plan.strategy_code), "当前问题适合采用的训练方式。"],
            ["业务轮次", roundText, "失败后最多自动进入下一轮，达到上限会停止自动迭代。"],
            ["随机种子", formatValue(plan.random_seed), "保证训练结果尽量可复现。"],
            ["特征版本", formatValue(plan.feature_schema_version), "训练时使用的特征字段定义。"],
            ["预处理版本", formatValue(plan.preprocessing_version), "缺失值、编码、标准化等预处理规则版本。"],
            ["策略参数", JSON.stringify(plan.strategy_parameters ?? {}), "更细的训练策略参数。"],
          ]}
        />

        <DetailGroup
          title="4. 验收规则"
          rows={[
            ["目标指标", joinValues(plan.target_metric_codes), "资格验证重点看的指标。"],
            ["资格规则版本", formatValue(plan.qualification_rule_version), "决定 challenger 是否合格的规则集。"],
            ["计划状态", formatValue(plan.status), "READY 表示计划已经可以派发训练。"],
            ["阻塞原因", joinValues(plan.blocking_reasons), "如果这里有值，说明训练计划不能继续执行。"],
          ]}
        />
      </div>

      <DetailGroup
        title="5. 关联记录"
        rows={[
          ["决策建议 ID", formatValue(plan.proposal_id), "任务三生成的 decision_proposal。"],
          ["人工复核 ID", formatValue(plan.approval_id), "前端点击通过后保存的 review。"],
          ["诊断运行 ID", formatValue(plan.diagnosis_run_id), "任务二诊断结果。"],
          ["迭代运行 ID", formatValue(plan.iteration_run_id), "本次模型迭代主记录。"],
          ["实验 ID", formatValue(plan.experiment_id), "Worker 训练和 MLflow 追踪对应的实验。"],
          [
            "数据资格检查 ID",
            joinValues(plan.data_eligibility_assessment_ids),
            "训练前的数据门禁检查结果。",
          ],
        ]}
      />
    </div>
  );
}

function DetailGroup({
  title,
  rows,
}: {
  title: string;
  rows: Array<[string, string, string]>;
}) {
  return (
    <div className="rounded-md border border-slate-200 bg-white p-3">
      <p className="text-sm font-semibold text-slate-800">{title}</p>
      <div className="mt-3 grid gap-2">
        {rows.map(([label, value, desc]) => (
          <div className="grid gap-1 rounded-md bg-slate-50 px-3 py-2 sm:grid-cols-[120px_1fr]" key={label}>
            <span className="text-xs font-medium text-slate-500">{label}</span>
            <div className="min-w-0">
              <p className="break-words text-sm font-semibold text-slate-800">{value}</p>
              <p className="mt-1 text-xs leading-5 text-slate-500">{desc}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ReferenceBox({ title, items }: { title: string; items: Array<[string, string]> }) {
  return (
    <div className="rounded-md border border-slate-200 bg-white p-3">
      <p className="text-sm font-semibold text-slate-800">{title}</p>
      <div className="mt-2 grid gap-2">
        {items.map(([key, desc]) => (
          <div className="grid gap-1 rounded-md bg-slate-50 px-3 py-2 sm:grid-cols-[190px_1fr]" key={key}>
            <span className="font-mono text-xs font-semibold text-slate-700">{key}</span>
            <span className="text-xs text-slate-500">{desc}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function StatusLine({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="grid grid-cols-[96px_1fr] gap-3 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-sm">
      <span className="text-slate-500">{label}</span>
      <span className={mono ? "break-all font-mono text-xs text-slate-700" : "text-slate-800"}>
        {value || "-"}
      </span>
    </div>
  );
}

function StatusBadge({ value }: { value: string }) {
  return (
    <span className={`rounded-md border px-3 py-2 text-xs font-semibold ${phaseClass(value)}`}>
      {value}
    </span>
  );
}

function Message({ type, children }: { type: "ok" | "error"; children: ReactNode }) {
  return (
    <div
      className={`rounded-md border px-4 py-3 text-sm font-medium ${
        type === "ok"
          ? "border-emerald-200 bg-emerald-50 text-emerald-700"
          : "border-red-200 bg-red-50 text-red-700"
      }`}
    >
      {children}
    </div>
  );
}

function StepTile({ active, label, value }: { active: boolean; label: string; value: unknown }) {
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-slate-700">{label}</span>
        <span className={`h-2.5 w-2.5 rounded-full ${active ? "bg-emerald-500" : "bg-slate-300"}`} />
      </div>
      <p className="mt-2 truncate font-mono text-[11px] text-slate-500">{formatValue(value)}</p>
    </div>
  );
}

function ProgressBar({ value }: { value: number }) {
  return (
    <div className="mt-4">
      <div className="mb-1 flex justify-between text-xs text-slate-500">
        <span>完成度</span>
        <span>{value}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-100">
        <div className="h-full rounded-full bg-sky-500" style={{ width: `${Math.min(value, 100)}%` }} />
      </div>
    </div>
  );
}

function Empty({ text }: { text: string }) {
  return (
    <div className="rounded-md border border-dashed border-slate-300 p-4 text-sm text-slate-500">
      {text}
    </div>
  );
}

function BarList({ rows, emptyText }: { rows: Array<{ name: string; value: number }>; emptyText: string }) {
  if (rows.length === 0) return <Empty text={emptyText} />;
  const max = Math.max(...rows.map((row) => row.value), 0.001);
  return (
    <div className="space-y-2">
      {rows.map((row, index) => (
        <div className="grid grid-cols-[28px_1fr_64px] items-center gap-3" key={`${row.name}-${index}`}>
          <span className="text-xs font-semibold text-slate-400">{index + 1}</span>
          <div>
            <div className="mb-1 flex justify-between gap-2">
              <span className="truncate text-sm font-medium text-slate-700">{row.name}</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-slate-100">
              <div className="h-full rounded-full bg-sky-500" style={{ width: `${(row.value / max) * 100}%` }} />
            </div>
          </div>
          <span className="text-right font-mono text-xs font-semibold text-slate-700">
            {formatValue(row.value)}
          </span>
        </div>
      ))}
    </div>
  );
}
