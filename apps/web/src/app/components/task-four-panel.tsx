"use client";

import { useEffect, useMemo, useState } from "react";
import { requestJson, Panel, StatTile, StatusDot, Badge, formatValue, fmtTs, Btn, Empty, Spinner } from "./shared";

type Items<T> = { items: T[]; total?: number };

type DeploymentItem = {
  deployment_id: string;
  lifecycle_run_id?: string;
  model_id?: string;
  champion_version?: string;
  candidate_version?: string;
  current_stage?: string;
  decision?: string;
  status?: string;
  created_at?: string;
  updated_at?: string;
};

type DeploymentDetail = {
  deployment: DeploymentItem;
  stages: DeploymentStageRecord[];
};

type DeploymentStageRecord = {
  stage_record_id?: string;
  deployment_id?: string;
  stage?: string;
  decision?: string;
  status?: string;
  health_json?: Record<string, unknown> | string | null;
  result_json?: Record<string, unknown> | string | null;
  created_at?: string;
};

type RoutingState = {
  model_id?: string;
  environment?: string;
  active_version_code?: string | null;
  stable_version_code?: string | null;
  challenger_version_code?: string | null;
  challenger_traffic_ratio?: number | string | null;
  state_version?: number | string;
  updated_at?: string;
  message?: string;
};

type HealthReport = {
  report_id?: string;
  deployment_id?: string;
  stage?: string;
  traffic_ratio?: number;
  passed?: boolean;
  rollback_recommended?: boolean;
  rollback_reasons?: string[];
  checks?: Array<{
    metric?: string;
    value?: number | boolean | null;
    threshold?: number | boolean | null;
    direction?: string;
    passed?: boolean;
    detail?: string;
  }>;
};

type ComparisonReport = {
  comparison_id?: string;
  model_id?: string;
  champion_version?: string;
  challenger_version?: string;
  passed?: boolean;
  summary?: string;
  metrics?: Array<{
    metric_code?: string;
    champion_value?: number | null;
    challenger_value?: number | null;
    delta?: number | null;
    delta_pct?: number | null;
    direction?: string;
    passed?: boolean | null;
  }>;
};

type PredictResult = {
  chosen_version?: string;
  chosen_role?: string;
  routing_reason?: string;
  hash_value?: number;
  challenger_traffic_ratio?: number;
  prediction?: {
    score?: number;
    threshold?: number;
    decision?: string;
    score_source?: string;
  };
  artifact?: {
    artifact_uri?: string;
    artifact_source?: string;
    loader?: string;
  };
  feature_schema?: {
    feature_count?: number;
    features_used?: string[];
    missing_features_filled_with_zero?: string[];
    extra_features_ignored?: string[];
  };
};

type BatchPredictResult = {
  total?: number;
  champion_count?: number;
  challenger_count?: number;
  actual_challenger_ratio?: number;
  results?: Array<{
    request_id?: string;
    chosen_version?: string;
    chosen_role?: string;
    score?: number;
    decision?: string;
  }>;
};

type ParallelControlItem = DeploymentItem & {
  active_version_code?: string | null;
  stable_version_code?: string | null;
  challenger_version_code?: string | null;
  challenger_traffic_ratio?: number | string | null;
  rollback_count?: number | string;
  last_patrol_at?: string | null;
  last_patrol_status?: string | null;
  last_patrol_decision?: string | null;
};

type ParallelControl = {
  summary?: {
    target_parallel_models?: number;
    listed_models?: number;
    total_registered_models?: number;
    total_deployed_models?: number;
    coverage_passed?: boolean;
    stage_distribution?: Record<string, number>;
    status_distribution?: Record<string, number>;
    canary_models?: number;
    production_models?: number;
    active_challenger_models?: number;
    rollback_models?: number;
    rollback_ready?: boolean;
    batch_action_limit?: number;
  };
  items?: ParallelControlItem[];
};

type RollbackDrillResult = {
  source_deployment_id?: string;
  drill_deployment_id?: string;
  persisted?: boolean;
  transaction?: string;
  stage?: string;
  simulated_canary_traffic_ratio?: number;
  health_result?: {
    passed?: boolean;
    failures?: string[];
    warnings?: string[];
    rollback_recommended?: boolean;
    rollback_reasons?: string[];
  };
  gatekeeper_decision?: {
    decision?: string;
    decision_reasons?: string[];
    gatekeeper_rule_refs?: string[];
    selected_strategy_code?: string | null;
  };
  rollback_result?: {
    status?: string;
    rollback_target?: string;
    reason?: string;
    rolled_back_at?: string;
  } | null;
  post_rollback_record?: DeploymentItem | null;
  post_rollback_routing?: RoutingState | null;
};

type ProactiveReleaseResult = {
  deployment_id?: string;
  release_type?: string;
  initial_stage?: string;
  challenger_traffic_ratio?: number;
  predeploy_health?: {
    passed?: boolean;
    health_status?: string;
    failures?: string[];
  };
  routing?: RoutingState | null;
};

type PatrolRunResult = {
  scheduler?: {
    mode?: string;
    interval_seconds?: number;
    persisted?: boolean;
    checked_at?: string;
  };
  summary?: {
    checked?: number;
    healthy?: number;
    held?: number;
    rolled_back?: number;
    skipped?: number;
  };
  results?: Array<{
    deployment_id?: string;
    model_id?: string;
    stage?: string;
    patrol_status?: string;
    action?: string;
    health_result?: {
      passed?: boolean;
      rollback_recommended?: boolean;
      rollback_reasons?: string[];
    };
  }>;
};

const stageLabels: Record<string, { label: string; desc: string }> = {
  OFFLINE_VALIDATION: { label: "离线验证", desc: "先用离线验证集检查候选模型是否达标，不接业务流量。" },
  OOT_GATE: { label: "跨期验证", desc: "用未参与训练的未来窗口做稳定性检查。" },
  SHADOW: { label: "影子部署", desc: "只旁路打分，不影响真实业务决策。" },
  CANARY_5: { label: "灰度 5%", desc: "5% 请求进入候选版本，观察风险。" },
  CANARY_20: { label: "灰度 20%", desc: "扩大到 20% 流量，继续观察。" },
  CANARY_50: { label: "灰度 50%", desc: "一半流量进入候选版本。" },
  PRODUCTION: { label: "全量生产", desc: "候选版本晋升为新的生产版本。" },
};

const defaultFeatures = {
  credit_query_times: 2,
  multi_loan_count: 1,
  overdue_history: 0,
  credit_utilization: 0.35,
  credit_length_months: 36,
  max_overdue_days: 0,
  social_score: 0.72,
  telecom_score: 0.68,
  ecomm_risk_score: 0.22,
  judicial_risk_score: 0.03,
  blacklist_hit: 0,
  app_duration: 180,
  click_frequency: 8,
  page_depth: 5,
  session_count: 3,
  night_activity_ratio: 0.12,
  login_fail_count: 0,
  reg_to_apply_days: 120,
  device_risk_score: 0.18,
  ip_change_freq: 1,
  gps_anomaly: 0,
  device_type: 1,
  emulator_flag: 0,
  age: 35,
  income_level: 4,
  consumption_level: 3,
  education_level: 3,
  job_stability: 4,
  marital_status: 1,
  gender: 1,
  city_tier: 2,
  debt_income_ratio: 0.28,
  loan_amount_request: 60000,
  repayment_period: 12,
};

const defaultComparison = {
  labels: [0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
  champion_scores: [0.08, 0.18, 0.61, 0.22, 0.68, 0.31, 0.74, 0.2, 0.12, 0.59],
  challenger_scores: [0.06, 0.14, 0.73, 0.18, 0.8, 0.25, 0.82, 0.16, 0.09, 0.69],
};

const defaultHealthMetrics = {
  challenger_auc: 0.81,
  challenger_ks: 0.42,
  score_psi: 0.08,
  bad_rate_drift: 0.02,
  recovery_rate: 0.72,
  train_valid_gap: 0.018,
  discrimination_passed: true,
  calibration_passed: true,
  oot_passed: true,
  segment_governance_passed: true,
  online_metrics: { rejection_rate: 0.08 },
};

export default function TaskFourPanel({
  apiBase,
  initialModelId,
  view = "full",
}: {
  apiBase: string;
  initialModelId?: string;
  view?: "patrol" | "full";
}) {
  const [modelId, setModelId] = useState(initialModelId || "credit_model_001");
  const [deployments, setDeployments] = useState<DeploymentItem[]>([]);
  const [selectedDeploymentId, setSelectedDeploymentId] = useState("");
  const [detail, setDetail] = useState<DeploymentDetail | null>(null);
  const [routing, setRouting] = useState<RoutingState | null>(null);
  const [inferenceRouting, setInferenceRouting] = useState<RoutingState | null>(null);
  const [healthReport, setHealthReport] = useState<HealthReport | null>(null);
  const [healthHistory, setHealthHistory] = useState<DeploymentStageRecord[]>([]);
  const [rollbackEvents, setRollbackEvents] = useState<Record<string, unknown>[]>([]);
  const [comparison, setComparison] = useState<ComparisonReport | null>(null);
  const [comparisonId, setComparisonId] = useState("");
  const [predictResult, setPredictResult] = useState<PredictResult | null>(null);
  const [batchResult, setBatchResult] = useState<BatchPredictResult | null>(null);
  const [featuresText, setFeaturesText] = useState(JSON.stringify(defaultFeatures, null, 2));
  const [comparisonText, setComparisonText] = useState(JSON.stringify(defaultComparison, null, 2));
  const [healthText, setHealthText] = useState(JSON.stringify(defaultHealthMetrics, null, 2));
  const [requestId, setRequestId] = useState("demo-user-001");
  const [proactiveModelId, setProactiveModelId] = useState(initialModelId || "credit_model_053");
  const [proactiveVersion, setProactiveVersion] = useState("challenger_v1");
  const [proactiveRollbackTarget, setProactiveRollbackTarget] = useState("champion_v1");
  const [proactiveStage, setProactiveStage] = useState("SHADOW");
  const [proactiveResult, setProactiveResult] = useState<ProactiveReleaseResult | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<{ type: "ok" | "error"; text: string } | null>(null);
  const [parallelControl, setParallelControl] = useState<ParallelControl | null>(null);
  const [selectedBatchIds, setSelectedBatchIds] = useState<string[]>([]);
  const [rollbackDrill, setRollbackDrill] = useState<RollbackDrillResult | null>(null);
  const [patrolEnabled, setPatrolEnabled] = useState(false);
  const [patrolIntervalSec, setPatrolIntervalSec] = useState(10);
  const [failurePatrolModelId, setFailurePatrolModelId] = useState("");
  const [patrolResult, setPatrolResult] = useState<PatrolRunResult | null>(null);
  const [patrolLastRunAt, setPatrolLastRunAt] = useState("");
  const [patrolNextRunAt, setPatrolNextRunAt] = useState("");
  const [nowMs, setNowMs] = useState(() => Date.now());

  const selectedDeployment = detail?.deployment || deployments.find((item) => item.deployment_id === selectedDeploymentId);
  const patrolHomeOnly = view === "patrol";
  const stage = selectedDeployment?.current_stage || detail?.stages?.at(-1)?.stage || "OFFLINE_VALIDATION";
  const stageMeta = stageLabels[stage] || { label: stage, desc: "" };

  const trafficPercent = useMemo(() => {
    const raw = inferenceRouting?.challenger_traffic_ratio ?? routing?.challenger_traffic_ratio ?? 0;
    const ratio = Number(raw || 0);
    return `${Math.round(ratio * 100)}%`;
  }, [routing, inferenceRouting]);

  const patrolCountdownSec = useMemo(() => {
    if (!patrolEnabled || !patrolNextRunAt) return 0;
    return Math.max(0, Math.ceil((Date.parse(patrolNextRunAt) - nowMs) / 1000));
  }, [nowMs, patrolEnabled, patrolNextRunAt]);

  const patrolProgress = useMemo(() => {
    if (!patrolEnabled || !patrolNextRunAt) return 0;
    const total = Math.max(10, patrolIntervalSec);
    return Math.max(0, Math.min(100, Math.round(((total - patrolCountdownSec) / total) * 100)));
  }, [patrolEnabled, patrolIntervalSec, patrolNextRunAt, patrolCountdownSec]);

  useEffect(() => {
    void refreshAll(false);
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!patrolEnabled) return;
    const intervalMs = Math.max(10, patrolIntervalSec) * 1000;
    setPatrolNextRunAt(new Date(Date.now() + intervalMs).toISOString());
    const timer = window.setInterval(() => {
      void runPatrolOnce(false);
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [patrolEnabled, patrolIntervalSec, failurePatrolModelId]);

  async function withBusy<T>(key: string, fn: () => Promise<T>, ok?: string) {
    setBusy(key);
    setMessage(null);
    try {
      const data = await fn();
      if (ok) setMessage({ type: "ok", text: ok });
      return data;
    } catch (error) {
      setMessage({ type: "error", text: error instanceof Error ? error.message : "请求失败" });
      return null;
    } finally {
      setBusy(null);
    }
  }

  async function refreshAll(showMessage = true) {
    return withBusy("refresh", async () => {
      const [list, control] = await Promise.all([
        requestJson<Items<DeploymentItem>>(apiBase, `/api/iteration/deployments?model_id=${encodeURIComponent(modelId)}&limit=50`),
        requestJson<ParallelControl>(apiBase, "/api/iteration/task4/parallel-control?limit=50"),
      ]);
      setDeployments(list.items || []);
      setParallelControl(control);
      setSelectedBatchIds((current) =>
        current.filter((id) => (control.items || []).some((item) => item.deployment_id === id)).slice(0, 50),
      );
      const nextSelected = selectedDeploymentId || list.items?.[0]?.deployment_id || "";
      if (nextSelected) {
        await loadDeployment(nextSelected, false);
      }
      await loadRouting(false);
      return list;
    }, showMessage ? "任务四数据已刷新" : undefined);
  }

  async function loadRouting(showMessage = true) {
    return withBusy("routing", async () => {
      const [routeState, inferenceState] = await Promise.all([
        requestJson<RoutingState>(apiBase, `/api/iteration/routing-configs/${encodeURIComponent(modelId)}`),
        requestJson<RoutingState>(apiBase, `/api/inference/${encodeURIComponent(modelId)}/routing-state`),
      ]);
      setRouting(routeState);
      setInferenceRouting(inferenceState);
      return routeState;
    }, showMessage ? "路由状态已刷新" : undefined);
  }

  async function loadDeployment(deploymentId: string, showMessage = true) {
    return withBusy("deployment", async () => {
      setSelectedDeploymentId(deploymentId);
      const data = await requestJson<DeploymentDetail>(apiBase, `/api/iteration/deployments/${deploymentId}`);
      setDetail(data);
      await Promise.all([loadHealthHistory(deploymentId), loadRollbackEvents(deploymentId)]);
      return data;
    }, showMessage ? "部署详情已加载" : undefined);
  }

  async function loadHealthHistory(deploymentId = selectedDeploymentId) {
    if (!deploymentId) return null;
    const data = await requestJson<{ deployment_id: string; health_checks: DeploymentStageRecord[] }>(
      apiBase,
      `/api/iteration/deployments/${deploymentId}/health-checks`,
    );
    setHealthHistory(data.health_checks || []);
    return data;
  }

  async function loadRollbackEvents(deploymentId = selectedDeploymentId) {
    if (!deploymentId) return null;
    const data = await requestJson<{ events: Record<string, unknown>[] }>(
      apiBase,
      `/api/iteration/deployments/${deploymentId}/rollback-events`,
    );
    setRollbackEvents(data.events || []);
    return data;
  }

  async function runHealthCheck() {
    if (!selectedDeploymentId) {
      setMessage({ type: "error", text: "请先选择一条部署记录" });
      return;
    }
    const payload = parseJson<Record<string, unknown>>(healthText, "健康检查指标 JSON");
    if (!payload) return;
    const report = await withBusy("health", () => requestJson<HealthReport>(
      apiBase,
      `/api/iteration/deployments/${selectedDeploymentId}/health-checks`,
      {
        method: "POST",
        body: JSON.stringify({
          deployment_id: selectedDeploymentId,
          stage,
          model_id: modelId,
          health_metrics: payload,
        }),
      },
    ), "健康检查已完成");
    if (report) {
      setHealthReport(report);
      await loadHealthHistory(selectedDeploymentId);
    }
  }

  async function runComparison() {
    const payload = parseJson<typeof defaultComparison>(comparisonText, "模型比对 JSON");
    if (!payload) return;
    const report = await withBusy("comparison", () => requestJson<ComparisonReport>(
      apiBase,
      "/api/iteration/comparisons",
      {
        method: "POST",
        body: JSON.stringify({
          model_id: modelId,
          champion_version: routing?.active_version_code || selectedDeployment?.champion_version || "champion_v1",
          challenger_version: routing?.challenger_version_code || selectedDeployment?.candidate_version || "challenger_v1",
          labels: payload.labels,
          champion_scores: payload.champion_scores,
          challenger_scores: payload.challenger_scores,
        }),
      },
    ), "模型比对已完成");
    if (report) {
      setComparison(report);
      setComparisonId(report.comparison_id || "");
    }
  }

  async function loadComparison() {
    if (!comparisonId.trim()) {
      setMessage({ type: "error", text: "请先输入 comparison_id" });
      return;
    }
    const report = await withBusy("load-comparison", () => requestJson<ComparisonReport>(
      apiBase,
      `/api/iteration/comparisons/${comparisonId.trim()}`,
    ), "模型比对报告已加载");
    if (report) setComparison(report);
  }

  async function runPredict() {
    const features = parseJson<Record<string, unknown>>(featuresText, "推理特征 JSON");
    if (!features) return;
    const result = await withBusy("predict", () => requestJson<PredictResult>(
      apiBase,
      `/api/inference/${encodeURIComponent(modelId)}/predict`,
      {
        method: "POST",
        body: JSON.stringify({
          request_id: requestId.trim(),
          features,
        }),
      },
    ), "真实推理已完成");
    if (result) setPredictResult(result);
  }

  async function runBatchPredict() {
    const features = parseJson<Record<string, unknown>>(featuresText, "推理特征 JSON");
    if (!features) return;
    const items = Array.from({ length: 30 }, (_, index) => ({
      request_id: `batch-user-${String(index + 1).padStart(3, "0")}`,
      features,
    }));
    const result = await withBusy("batch-predict", () => requestJson<BatchPredictResult>(
      apiBase,
      `/api/inference/${encodeURIComponent(modelId)}/batch-predict`,
      {
        method: "POST",
        body: JSON.stringify({ items }),
      },
    ), "批量推理已完成");
    if (result) setBatchResult(result);
  }

  async function createHealthyRelease() {
    const releaseModelId = proactiveModelId.trim();
    if (!releaseModelId || !proactiveVersion.trim() || !proactiveRollbackTarget.trim()) {
      setMessage({ type: "error", text: "model_id, challenger_version, and rollback_target are required" });
      return;
    }
    const result = await withBusy("proactive-release", () => requestJson<ProactiveReleaseResult>(
      apiBase,
      "/api/iteration/deployments/proactive-release",
      {
        method: "POST",
        body: JSON.stringify({
          model_id: releaseModelId,
          challenger_version: proactiveVersion.trim(),
          champion_version: proactiveRollbackTarget.trim(),
          rollback_target: proactiveRollbackTarget.trim(),
          release_type: "NEW_SCENARIO",
          initial_stage: proactiveStage,
          health_status: "PASSED",
          health_metrics: {
            artifact_loadable: true,
            schema_consistency: true,
            inference_smoke_passed: true,
          },
          updated_by: "frontend_proactive_release",
        }),
      },
    ), "Healthy new-scenario release created");
    if (result) {
      setProactiveResult(result);
      setModelId(releaseModelId);
      await refreshAll(false);
      if (result.deployment_id) {
        await loadDeployment(result.deployment_id, false);
      }
      await loadRouting(false);
    }
  }

  function toggleBatchId(deploymentId: string) {
    setSelectedBatchIds((current) => {
      if (current.includes(deploymentId)) return current.filter((id) => id !== deploymentId);
      return [...current, deploymentId].slice(0, 50);
    });
  }

  function selectVisibleParallelItems() {
    const ids = (parallelControl?.items || []).map((item) => item.deployment_id).filter(Boolean).slice(0, 50);
    setSelectedBatchIds(ids);
  }

  async function batchAdvanceSelected() {
    if (selectedBatchIds.length === 0) {
      setMessage({ type: "error", text: "请先选择要批量推进的部署记录" });
      return;
    }
    const result = await withBusy("batch-advance", () => requestJson(
      apiBase,
      "/api/iteration/deployments/batch/advance",
      {
        method: "POST",
        body: JSON.stringify({ deployment_ids: selectedBatchIds, updated_by: "frontend", resume_lifecycle: true }),
      },
    ), `已提交 ${selectedBatchIds.length} 条部署的批量推进`);
    if (result) await refreshAll(false);
  }

  async function batchRollbackSelected() {
    if (selectedBatchIds.length === 0) {
      setMessage({ type: "error", text: "请先选择要批量回滚的部署记录" });
      return;
    }
    const result = await withBusy("batch-rollback", () => requestJson(
      apiBase,
      "/api/iteration/deployments/batch/rollback",
      {
        method: "POST",
        body: JSON.stringify({ deployment_ids: selectedBatchIds, updated_by: "frontend" }),
      },
    ), `已提交 ${selectedBatchIds.length} 条部署的批量回滚`);
    if (result) await refreshAll(false);
  }

  async function rollbackSelected() {
    if (!selectedDeploymentId) {
      setMessage({ type: "error", text: "请先选择一条部署记录" });
      return;
    }
    const result = await withBusy("rollback", () => requestJson(
      apiBase,
      `/api/iteration/deployments/${selectedDeploymentId}/rollback`,
      { method: "POST", body: JSON.stringify({ reason: "FRONTEND_MANUAL_ROLLBACK", updated_by: "frontend" }) },
    ), "回滚已提交");
    if (result) {
      await loadDeployment(selectedDeploymentId, false);
      await loadRouting(false);
    }
  }

  async function advanceSelectedStage() {
    if (!selectedDeploymentId) {
      setMessage({ type: "error", text: "Please select a deployment first" });
      return;
    }
    const result = await withBusy("advance-stage", () => requestJson<{
      deployment?: DeploymentItem;
      routing?: RoutingState;
      action_result?: Record<string, unknown>;
    }>(
      apiBase,
      `/api/iteration/deployments/${selectedDeploymentId}/advance-stage`,
      {
        method: "POST",
        body: JSON.stringify({
          updated_by: "frontend_direct_advance",
          health_metrics: {},
        }),
      },
    ), "Deployment advanced to next stage");
    if (result) {
      await loadDeployment(selectedDeploymentId, false);
      await loadRouting(false);
      await refreshAll(false);
    }
  }

  async function runRollbackDrill() {
    if (!selectedDeploymentId) {
      setMessage({ type: "error", text: "Please select a deployment first" });
      return;
    }
    const result = await withBusy("rollback-drill", () => requestJson<RollbackDrillResult>(
      apiBase,
      `/api/iteration/deployments/${selectedDeploymentId}/rollback-drill`,
      {
        method: "POST",
        body: JSON.stringify({
          stage: "CANARY_20",
          persist: false,
          updated_by: "frontend_rollback_drill",
        }),
      },
    ), "Automatic rollback drill completed; no data was persisted");
    if (result) {
      setRollbackDrill(result);
      await loadRouting(false);
    }
  }

  async function runPatrolOnce(showMessage = true) {
    const result = await withBusy("patrol", () => requestJson<PatrolRunResult>(
      apiBase,
      "/api/iteration/task4/patrol/run-once",
      {
        method: "POST",
        body: JSON.stringify({
          interval_seconds: patrolIntervalSec,
          failure_model_id: failurePatrolModelId.trim() || null,
          persist: true,
          updated_by: "frontend_scheduled_patrol",
        }),
      },
    ), showMessage ? "定时巡检已完成" : undefined);
    if (result) {
      const checkedAt = result.scheduler?.checked_at || new Date().toISOString();
      setPatrolResult(result);
      setPatrolLastRunAt(checkedAt);
      setPatrolNextRunAt(new Date(Date.parse(checkedAt) + Math.max(10, patrolIntervalSec) * 1000).toISOString());
      await refreshAll(false);
    }
  }

  function parseJson<T>(text: string, label: string): T | null {
    try {
      return JSON.parse(text) as T;
    } catch (error) {
      setMessage({ type: "error", text: `${label} 格式不正确：${error instanceof Error ? error.message : "JSON 解析失败"}` });
      return null;
    }
  }

  return (
    <div className="space-y-5 p-5">
      {!patrolHomeOnly ? <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[.16em] text-indigo-600">任务四操作台</p>
          <h1 className="mt-1 text-2xl font-bold tracking-tight text-slate-900">模型验证与灰度发布</h1>
          <p className="mt-1 text-sm text-slate-500">集中查看模型验证、部署灰度、推理分流、回滚和再迭代闭环。</p>
        </div>
        <div className="flex items-center gap-2">
          <input
            className="w-56 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-mono focus:border-indigo-400 focus:outline-none"
            value={modelId}
            onChange={(event) => setModelId(event.target.value)}
          />
          <Btn onClick={() => refreshAll()} disabled={busy === "refresh"}>{busy === "refresh" ? <Spinner /> : "刷新"}</Btn>
        </div>
      </div> : null}

      {message ? (
        <div className={`rounded-lg border px-4 py-3 text-sm ${message.type === "ok" ? "border-emerald-200 bg-emerald-50 text-emerald-700" : "border-red-200 bg-red-50 text-red-700"}`}>
          {message.text}
        </div>
      ) : null}

      {!patrolHomeOnly ? (
        <div className="grid gap-4 xl:grid-cols-4">
          <StatTile label="生产版本" value={formatValue(inferenceRouting?.active_version_code || routing?.active_version_code)} sub="当前承接主流量" />
          <StatTile label="稳定回滚版本" value={formatValue(inferenceRouting?.stable_version_code || routing?.stable_version_code)} sub="异常时恢复目标" />
          <StatTile label="候选灰度版本" value={formatValue(inferenceRouting?.challenger_version_code || routing?.challenger_version_code)} sub={`当前流量 ${trafficPercent}`} />
          <StatTile label="当前部署阶段" value={stageMeta.label} sub={stageMeta.desc} />
        </div>
      ) : null}

      {patrolHomeOnly ? (
      <section className="min-h-[calc(100vh-120px)] space-y-4">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[.16em] text-emerald-600">实时巡检</p>
            <h2 className="mt-1 text-xl font-bold tracking-tight text-slate-900">模型定时巡检与异常回滚</h2>
            <p className="mt-1 text-sm text-slate-500">首次触发后，页面按周期巡检部署模型；健康异常时由 Gatekeeper 自动切回稳定版本。</p>
          </div>
          <div className="flex flex-wrap gap-2 text-xs text-slate-500">
            <Badge label={`部署模型 ${parallelControl?.summary?.total_deployed_models ?? 0}/50`} color={(parallelControl?.summary?.total_deployed_models ?? 0) >= 50 ? "green" : "amber"} />
            <Badge label={`灰度中 ${parallelControl?.summary?.canary_models ?? 0}`} color="blue" />
            <Badge label={`回滚模型 ${parallelControl?.summary?.rollback_models ?? 0}`} color={(parallelControl?.summary?.rollback_models ?? 0) > 0 ? "red" : "green"} />
          </div>
        </div>

        <Panel
          title="定时巡检"
          className="min-h-[520px]"
        action={
          <div className="flex flex-wrap gap-2">
            <Btn onClick={() => setPatrolEnabled((enabled) => !enabled)}>
              {patrolEnabled ? "暂停巡检" : "开启巡检"}
            </Btn>
            <Btn onClick={() => runPatrolOnce()} disabled={busy === "patrol"}>
              {busy === "patrol" ? <Spinner /> : "执行一次巡检"}
            </Btn>
          </div>
        }
      >
        <div className="grid gap-3 md:grid-cols-5">
          <Info label="巡检状态" value={patrolEnabled ? "运行中" : "未开启"} />
          <Info label="巡检周期" value={`${patrolIntervalSec} 秒`} />
          <Info label="最近巡检" value={patrolLastRunAt ? fmtTs(patrolLastRunAt) : "-"} />
          <Info label="下一次巡检" value={patrolEnabled && patrolNextRunAt ? fmtTs(patrolNextRunAt) : "-"} />
          <Info label="自动处理" value="异常回滚" />
        </div>
        <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-3">
          <div className="mb-2 flex items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2 font-semibold text-slate-700">
              <span className={`status-dot ${busy === "patrol" ? "blue pulse" : patrolEnabled ? "green pulse" : "amber"}`} />
              {busy === "patrol" ? "正在执行巡检" : patrolEnabled ? "自动巡检计时中" : "巡检未开启"}
            </div>
            <div className="font-mono text-slate-500">
              {patrolEnabled ? `${patrolCountdownSec}s 后再次巡检` : "手动执行或开启巡检"}
            </div>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-white">
            <div
              className={`h-full transition-all duration-700 ${busy === "patrol" ? "bg-sky-500" : patrolEnabled ? "bg-emerald-500" : "bg-amber-400"}`}
              style={{ width: `${busy === "patrol" ? 100 : patrolProgress}%` }}
            />
          </div>
        </div>
        <div className="mt-3 grid gap-3 md:grid-cols-[180px_1fr]">
          <label className="text-xs font-semibold text-slate-500">
            周期秒数
            <input
              className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 font-mono text-xs"
              type="number"
              min={10}
              value={patrolIntervalSec}
              onChange={(event) => setPatrolIntervalSec(Math.max(10, Number(event.target.value) || 10))}
            />
          </label>
          <label className="text-xs font-semibold text-slate-500">
            异常模型 ID
            <input
              className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 font-mono text-xs"
              placeholder="例如 credit_model_053"
              value={failurePatrolModelId}
              onChange={(event) => setFailurePatrolModelId(event.target.value)}
            />
          </label>
        </div>
        {patrolResult ? (
          <div className="mt-3 grid gap-2 md:grid-cols-5">
            <Info label="本次检查" value={patrolResult.summary?.checked ?? 0} />
            <Info label="健康" value={patrolResult.summary?.healthy ?? 0} />
            <Info label="暂停" value={patrolResult.summary?.held ?? 0} />
            <Info label="已回滚" value={patrolResult.summary?.rolled_back ?? 0} />
            <Info label="跳过" value={patrolResult.summary?.skipped ?? 0} />
          </div>
        ) : null}
        {patrolResult?.results?.length ? (
          <div className="mt-3 max-h-40 overflow-auto rounded-lg border border-slate-200">
            <table className="w-full min-w-[720px] text-left text-xs">
              <thead className="bg-slate-50 text-slate-500">
                <tr>
                  <th className="px-3 py-2">模型</th>
                  <th>阶段</th>
                  <th>巡检结果</th>
                  <th>系统动作</th>
                  <th>回滚建议</th>
                </tr>
              </thead>
              <tbody>
                {patrolResult.results.slice(0, 12).map((item) => (
                  <tr key={`${item.deployment_id}-${item.model_id}`} className="border-t border-slate-100">
                    <td className="px-3 py-2 font-mono text-slate-700">{formatValue(item.model_id)}</td>
                    <td>{stageLabels[item.stage || ""]?.label || formatValue(item.stage)}</td>
                    <td>
                      <Badge
                        label={formatValue(item.patrol_status)}
                        color={item.patrol_status === "ROLLED_BACK" || item.patrol_status === "HELD" ? "red" : item.patrol_status === "PASSED" ? "green" : "amber"}
                      />
                    </td>
                    <td>{formatValue(item.action)}</td>
                    <td>{item.health_result?.rollback_recommended ? "是" : "否"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
        </Panel>
      </section>
      ) : null}

      {!patrolHomeOnly ? (
      <>
      <Panel
        title="50 模型并行管控"
        action={
          <div className="flex flex-wrap gap-2">
            <Btn onClick={selectVisibleParallelItems}>选择当前 50</Btn>
            <Btn onClick={batchAdvanceSelected} disabled={busy === "batch-advance" || selectedBatchIds.length === 0}>
              批量推进
            </Btn>
            <Btn danger onClick={batchRollbackSelected} disabled={busy === "batch-rollback" || selectedBatchIds.length === 0}>
              批量回滚
            </Btn>
          </div>
        }
      >
        <div className="mb-4 rounded-lg border border-slate-200 bg-slate-50 p-3">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <div className="text-xs font-semibold text-slate-700">Healthy new-scenario release</div>
              <div className="text-[11px] text-slate-400">release_type=NEW_SCENARIO, comparison is not required</div>
            </div>
            <Btn primary onClick={createHealthyRelease} disabled={busy === "proactive-release"}>
              {busy === "proactive-release" ? <Spinner /> : "Create release"}
            </Btn>
          </div>
          <div className="grid gap-3 lg:grid-cols-4">
            <label className="text-xs font-semibold text-slate-500">
              model_id
              <input className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 font-mono text-xs" value={proactiveModelId} onChange={(event) => setProactiveModelId(event.target.value)} />
            </label>
            <label className="text-xs font-semibold text-slate-500">
              challenger_version
              <input className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 font-mono text-xs" value={proactiveVersion} onChange={(event) => setProactiveVersion(event.target.value)} />
            </label>
            <label className="text-xs font-semibold text-slate-500">
              rollback_target
              <input className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 font-mono text-xs" value={proactiveRollbackTarget} onChange={(event) => setProactiveRollbackTarget(event.target.value)} />
            </label>
            <label className="text-xs font-semibold text-slate-500">
              initial_stage
              <select className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-xs" value={proactiveStage} onChange={(event) => setProactiveStage(event.target.value)}>
                <option value="OFFLINE_VALIDATION">OFFLINE_VALIDATION</option>
                <option value="OOT_GATE">OOT_GATE</option>
                <option value="SHADOW">SHADOW</option>
                <option value="CANARY_5">CANARY_5</option>
              </select>
            </label>
          </div>
          {proactiveResult ? (
            <div className="mt-3 grid gap-2 md:grid-cols-4">
              <Info label="deployment_id" value={proactiveResult.deployment_id} mono />
              <Info label="health" value={proactiveResult.predeploy_health?.passed ? "PASSED" : "FAILED"} />
              <Info label="stage" value={proactiveResult.initial_stage} />
              <Info label="traffic" value={`${Math.round(Number(proactiveResult.challenger_traffic_ratio || 0) * 100)}%`} />
            </div>
          ) : null}
        </div>
        <ParallelControlView
          control={parallelControl}
          selectedIds={selectedBatchIds}
          onToggle={toggleBatchId}
        />
      </Panel>

      <div className="grid gap-5 xl:grid-cols-[360px_1fr]">
        <Panel title={`部署记录 (${deployments.length})`} action={<Btn onClick={() => refreshAll()} disabled={busy === "refresh"}>重新加载</Btn>}>
          <div className="max-h-[620px] space-y-1 overflow-auto">
            {deployments.length === 0 ? (
              <Empty text="暂无部署记录。先跑一次生命周期闭环后，这里会出现部署记录。" />
            ) : deployments.map((item) => (
              <button
                key={item.deployment_id}
                className={`w-full rounded-lg px-3 py-2.5 text-left transition hover:bg-slate-50 ${selectedDeploymentId === item.deployment_id ? "border border-indigo-100 bg-indigo-50" : ""}`}
                onClick={() => loadDeployment(item.deployment_id)}
              >
                <div className="flex items-center gap-3">
                  <StatusDot status={item.status} />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-semibold text-slate-800">{item.model_id}</div>
                    <div className="truncate font-mono text-xs text-slate-400">{item.deployment_id}</div>
                  </div>
                  <Badge label={stageLabels[item.current_stage || ""]?.label || item.current_stage || "未知"} color={item.current_stage === "PRODUCTION" ? "green" : item.decision === "ROLLBACK" ? "red" : item.decision === "HOLD" ? "amber" : "blue"} />
                </div>
                <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-slate-500">
                  <span>候选：{formatValue(item.candidate_version)}</span>
                  <span>决策：{formatValue(item.decision)}</span>
                </div>
              </button>
            ))}
          </div>
        </Panel>

        <div className="space-y-5">
          <Panel title="部署总览与灰度路由" action={<Btn onClick={() => loadRouting()} disabled={busy === "routing"}>刷新路由</Btn>}>
            <div className="grid gap-3 lg:grid-cols-3">
              <Info label="部署 ID" value={selectedDeployment?.deployment_id} mono />
              <Info label="生命周期 ID" value={selectedDeployment?.lifecycle_run_id} mono />
              <Info label="部署状态" value={selectedDeployment?.status} />
              <Info label="当前阶段" value={`${stageMeta.label} (${stage})`} />
              <Info label="当前决策" value={selectedDeployment?.decision} />
              <Info label="路由版本号" value={routing?.state_version} />
            </div>
            <div className="mt-4 grid gap-2 md:grid-cols-7">
              {Object.entries(stageLabels).map(([key, meta]) => {
                const active = key === stage;
                const passed = stageOrder(key) <= stageOrder(stage);
                return (
                  <div key={key} className={`rounded-lg border px-2 py-2 text-center ${active ? "border-indigo-300 bg-indigo-50" : passed ? "border-emerald-200 bg-emerald-50" : "border-slate-200 bg-slate-50"}`}>
                    <div className="text-xs font-semibold text-slate-700">{meta.label}</div>
                    <div className="mt-1 text-[10px] text-slate-400">{stageTraffic(key)}</div>
                  </div>
                );
              })}
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              <Btn onClick={advanceSelectedStage} disabled={busy === "advance-stage" || !selectedDeploymentId}>
                {busy === "advance-stage" ? <Spinner /> : "Advance stage"}
              </Btn>
              <Btn onClick={runRollbackDrill} disabled={busy === "rollback-drill" || !selectedDeploymentId}>
                {busy === "rollback-drill" ? <Spinner /> : "Auto rollback drill"}
              </Btn>
              <Btn danger onClick={rollbackSelected} disabled={busy === "rollback" || !selectedDeploymentId}>手动回滚</Btn>
              <Btn onClick={() => loadRollbackEvents()} disabled={!selectedDeploymentId}>查询回滚事件</Btn>
            </div>
            {rollbackDrill ? <RollbackDrillView result={rollbackDrill} /> : null}
          </Panel>

          <div className="grid gap-5 xl:grid-cols-2">
            <Panel title="部署健康检查" action={<Btn onClick={runHealthCheck} disabled={busy === "health"}>{busy === "health" ? <Spinner /> : "执行检查"}</Btn>}>
              <textarea className="h-40 w-full rounded-lg border border-slate-200 bg-slate-50 p-3 font-mono text-xs focus:border-indigo-400 focus:outline-none" value={healthText} onChange={(event) => setHealthText(event.target.value)} />
              {healthReport ? <HealthReportView report={healthReport} /> : <Empty text="执行健康检查后，这里显示 AUC、KS、PSI、恢复率和回滚建议。" />}
              <HistoryList title="健康检查历史" items={healthHistory} empty="暂无健康检查历史" />
            </Panel>

            <Panel title="新老模型比对" action={<Btn onClick={runComparison} disabled={busy === "comparison"}>{busy === "comparison" ? <Spinner /> : "运行比对"}</Btn>}>
              <textarea className="h-40 w-full rounded-lg border border-slate-200 bg-slate-50 p-3 font-mono text-xs focus:border-indigo-400 focus:outline-none" value={comparisonText} onChange={(event) => setComparisonText(event.target.value)} />
              <div className="mt-3 flex gap-2">
                <input className="min-w-0 flex-1 rounded-lg border border-slate-200 px-3 py-2 font-mono text-xs" placeholder="comparison_id" value={comparisonId} onChange={(event) => setComparisonId(event.target.value)} />
                <Btn onClick={loadComparison} disabled={busy === "load-comparison"}>读取报告</Btn>
              </div>
              {comparison ? <ComparisonReportView report={comparison} /> : <Empty text="运行比对后，这里显示 10 个指标是否支持上线。" />}
            </Panel>
          </div>

          <Panel title="真实推理测试" action={<div className="flex gap-2"><Btn onClick={runPredict} disabled={busy === "predict"}>{busy === "predict" ? <Spinner /> : "单笔推理"}</Btn><Btn onClick={runBatchPredict} disabled={busy === "batch-predict"}>批量分流</Btn></div>}>
            <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
              <div>
                <label className="mb-2 block text-xs font-semibold text-slate-500">request_id</label>
                <input className="mb-3 w-full rounded-lg border border-slate-200 px-3 py-2 font-mono text-xs" value={requestId} onChange={(event) => setRequestId(event.target.value)} />
                <textarea className="h-72 w-full rounded-lg border border-slate-200 bg-slate-50 p-3 font-mono text-xs focus:border-indigo-400 focus:outline-none" value={featuresText} onChange={(event) => setFeaturesText(event.target.value)} />
              </div>
              <div className="space-y-3">
                {predictResult ? <PredictResultView result={predictResult} /> : <Empty text="单笔推理后显示选中的版本、真实分数、模型产物和特征对齐情况。" />}
                {batchResult ? <BatchPredictView result={batchResult} /> : null}
              </div>
            </div>
          </Panel>

          <Panel title="回滚事件">
            {rollbackEvents.length === 0 ? <Empty text="暂无回滚事件" /> : rollbackEvents.map((event, index) => (
              <div key={index} className="mb-2 rounded-lg border border-red-100 bg-red-50 px-3 py-2 text-sm text-red-700">
                <div className="font-semibold">{formatValue(event.stage)} / {formatValue(event.decision)}</div>
                <div className="mt-1 font-mono text-xs">{formatValue(event.created_at)}</div>
              </div>
            ))}
          </Panel>
        </div>
      </div>
      </>
      ) : null}
    </div>
  );
}

function ParallelControlView({
  control,
  selectedIds,
  onToggle,
}: {
  control: ParallelControl | null;
  selectedIds: string[];
  onToggle: (deploymentId: string) => void;
}) {
  const summary = control?.summary || {};
  const items = control?.items || [];
  const stageDistribution = summary.stage_distribution || {};
  const statusDistribution = summary.status_distribution || {};
  const target = Number(summary.target_parallel_models || 50);
  const deployed = Number(summary.total_deployed_models || 0);
  const listed = Number(summary.listed_models || items.length);
  const coveragePassed = Boolean(summary.coverage_passed);

  return (
    <div className="space-y-4">
      <div className="grid gap-3 md:grid-cols-4">
        <Info label="并行部署模型" value={`${deployed}/${target}`} />
        <Info label="当前列表" value={`${listed} 条`} />
        <Info label="灰度中" value={summary.canary_models ?? 0} />
        <Info label="可回滚状态" value={summary.rollback_ready ? "是" : "否"} />
      </div>
      <div className={`rounded-lg border px-3 py-2 text-sm ${coveragePassed ? "border-emerald-200 bg-emerald-50 text-emerald-700" : "border-amber-200 bg-amber-50 text-amber-700"}`}>
        {coveragePassed
          ? "已满足不少于 50 个模型并行管控覆盖。"
          : `当前只有 ${deployed} 个模型存在部署记录，尚未达到 50 模型并行管控验收口径。`}
      </div>
      <div className="grid gap-3 lg:grid-cols-2">
        <Distribution title="阶段分布" data={stageDistribution} />
        <Distribution title="状态分布" data={statusDistribution} />
      </div>
      <div className="overflow-auto rounded-lg border border-slate-200">
        <table className="w-full min-w-[1120px] text-left text-xs">
          <thead className="bg-slate-50 text-slate-500">
            <tr>
              <th className="px-3 py-2">选择</th>
              <th>模型</th>
              <th>巡检状态</th>
              <th>当前动作</th>
              <th>最近巡检</th>
              <th>阶段</th>
              <th>状态</th>
              <th>生产版本</th>
              <th>候选版本</th>
              <th>候选流量</th>
              <th>回滚次数</th>
              <th>部署 ID</th>
            </tr>
          </thead>
          <tbody>
            {items.length === 0 ? (
              <tr>
                <td colSpan={12} className="px-3 py-6 text-center text-slate-400">
                  暂无部署记录。先让生命周期跑到资格验证和部署阶段，再做任务4并行管控。
                </td>
              </tr>
            ) : items.map((item) => {
              const promoted = item.status === "PROMOTED" || item.current_stage === "PRODUCTION";
              const rolledBack = item.status === "ROLLED_BACK";
              const patrolStatus = item.last_patrol_status || (rolledBack ? "ROLLED_BACK" : "WAITING");
              const patrolAction = item.last_patrol_decision === "ROLLBACK"
                ? "自动回滚"
                : item.last_patrol_decision === "HOLD"
                  ? "暂停观察"
                  : item.last_patrol_decision === "HEALTH_CHECK"
                    ? "继续巡检"
                    : rolledBack
                      ? "已回滚"
                      : "等待巡检";
              const candidateVersion = promoted || rolledBack
                ? null
                : item.challenger_version_code || item.candidate_version;
              const candidateTraffic = promoted
                ? "已转生产"
                : rolledBack
                  ? "已回滚"
                  : `${Math.round(Number(item.challenger_traffic_ratio || 0) * 100)}%`;

              return (
                <tr key={item.deployment_id} className="border-t border-slate-100">
                  <td className="px-3 py-2">
                    <input
                      type="checkbox"
                      checked={selectedIds.includes(item.deployment_id)}
                      onChange={() => onToggle(item.deployment_id)}
                    />
                  </td>
                  <td className="font-semibold text-slate-700">{formatValue(item.model_id)}</td>
                  <td>
                    <Badge
                      label={formatValue(patrolStatus)}
                      color={patrolStatus === "ROLLED_BACK" || patrolStatus === "HELD" ? "red" : patrolStatus === "PASSED" ? "green" : "amber"}
                    />
                  </td>
                  <td>{patrolAction}</td>
                  <td className="font-mono text-slate-400">{item.last_patrol_at ? fmtTs(item.last_patrol_at) : "-"}</td>
                  <td>{stageLabels[item.current_stage || ""]?.label || formatValue(item.current_stage)}</td>
                  <td><Badge label={formatValue(item.status)} color={item.status === "ROLLED_BACK" || item.status === "FAILED" ? "red" : item.status === "PROMOTED" ? "green" : "blue"} /></td>
                  <td className="font-mono">{formatValue(item.active_version_code || item.champion_version)}</td>
                  <td className="font-mono">{formatValue(candidateVersion)}</td>
                  <td>{candidateTraffic}</td>
                  <td>{formatValue(item.rollback_count)}</td>
                  <td className="font-mono text-slate-400">{item.deployment_id}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Distribution({ title, data }: { title: string; data: Record<string, number> }) {
  const entries = Object.entries(data);
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-3 py-2">
      <div className="mb-2 text-xs font-semibold text-slate-500">{title}</div>
      {entries.length === 0 ? (
        <div className="text-xs text-slate-400">暂无数据</div>
      ) : (
        <div className="flex flex-wrap gap-2">
          {entries.map(([key, value]) => (
            <span key={key} className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1 text-xs text-slate-600">
              {stageLabels[key]?.label || key}: <b>{value}</b>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function Info({ label, value, mono }: { label: string; value: unknown; mono?: boolean }) {
  return (
    <div className="rounded-lg bg-slate-50 px-3 py-2">
      <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-400">{label}</div>
      <div className={`mt-0.5 break-all text-sm font-semibold text-slate-800 ${mono ? "font-mono text-xs" : ""}`}>{formatValue(value)}</div>
    </div>
  );
}

function RollbackDrillView({ result }: { result: RollbackDrillResult }) {
  const decision = result.gatekeeper_decision?.decision || "-";
  const traffic = Number(result.post_rollback_routing?.challenger_traffic_ratio || 0);
  return (
    <div className="mt-4 rounded-lg border border-red-100 bg-red-50 p-3">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Badge label={`Gatekeeper ${decision}`} color={decision === "ROLLBACK" ? "red" : "amber"} />
        <Badge label={result.transaction === "rolled_back" ? "No DB changes persisted" : "Committed"} color={result.transaction === "rolled_back" ? "green" : "amber"} />
        <span className="font-mono text-xs text-red-500">{result.drill_deployment_id}</span>
      </div>
      <div className="grid gap-2 md:grid-cols-4">
        <Info label="Health passed" value={result.health_result?.passed} />
        <Info label="Rollback recommended" value={result.health_result?.rollback_recommended} />
        <Info label="Rollback target" value={result.rollback_result?.rollback_target} mono />
        <Info label="Challenger traffic" value={`${Math.round(traffic * 100)}%`} />
      </div>
      <div className="mt-3 rounded-lg bg-white/80 px-3 py-2 text-xs text-red-700">
        {(result.gatekeeper_decision?.decision_reasons || []).slice(0, 5).join(", ") || "No decision reasons"}
      </div>
    </div>
  );
}

function HealthReportView({ report }: { report: HealthReport }) {
  return (
    <div className="mt-3 space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Badge label={report.passed ? "健康通过" : "健康失败"} color={report.passed ? "green" : "red"} />
        <Badge label={report.rollback_recommended ? "建议回滚" : "无需回滚"} color={report.rollback_recommended ? "red" : "green"} />
        <span className="text-xs text-slate-400">流量比例 {formatValue(report.traffic_ratio)}</span>
      </div>
      <div className="overflow-auto rounded-lg border border-slate-200">
        <table className="w-full min-w-[620px] text-left text-xs">
          <thead className="bg-slate-50 text-slate-500">
            <tr><th className="px-3 py-2">指标</th><th>当前值</th><th>阈值</th><th>方向</th><th>结果</th><th>说明</th></tr>
          </thead>
          <tbody>
            {(report.checks || []).map((check) => (
              <tr key={check.metric} className="border-t border-slate-100">
                <td className="px-3 py-2 font-semibold text-slate-700">{check.metric}</td>
                <td>{formatValue(check.value)}</td>
                <td>{formatValue(check.threshold)}</td>
                <td>{formatValue(check.direction)}</td>
                <td><Badge label={check.passed ? "通过" : "失败"} color={check.passed ? "green" : "red"} /></td>
                <td className="text-slate-500">{formatValue(check.detail)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ComparisonReportView({ report }: { report: ComparisonReport }) {
  return (
    <div className="mt-3 space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Badge label={report.passed ? "允许上线" : "不建议上线"} color={report.passed ? "green" : "red"} />
        <span className="font-mono text-xs text-slate-400">{report.comparison_id}</span>
      </div>
      <div className="text-xs text-slate-500">{report.summary}</div>
      <div className="overflow-auto rounded-lg border border-slate-200">
        <table className="w-full min-w-[660px] text-left text-xs">
          <thead className="bg-slate-50 text-slate-500">
            <tr><th className="px-3 py-2">指标</th><th>Champion</th><th>Challenger</th><th>差值</th><th>方向</th><th>结果</th></tr>
          </thead>
          <tbody>
            {(report.metrics || []).map((metric) => (
              <tr key={metric.metric_code} className="border-t border-slate-100">
                <td className="px-3 py-2 font-semibold text-slate-700">{metric.metric_code}</td>
                <td>{formatValue(metric.champion_value)}</td>
                <td>{formatValue(metric.challenger_value)}</td>
                <td>{formatValue(metric.delta)}</td>
                <td>{metric.direction === "lower_is_better" ? "越低越好" : "越高越好"}</td>
                <td><Badge label={metric.passed ? "通过" : "失败"} color={metric.passed ? "green" : "red"} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PredictResultView({ result }: { result: PredictResult }) {
  const missing = result.feature_schema?.missing_features_filled_with_zero || [];
  const extra = result.feature_schema?.extra_features_ignored || [];
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="grid gap-2 md:grid-cols-2">
        <Info label="路由版本" value={`${result.chosen_role || "-"} / ${result.chosen_version || "-"}`} />
        <Info label="真实分数" value={result.prediction?.score} />
        <Info label="模型产物" value={result.artifact?.artifact_uri} mono />
        <Info label="加载方式" value={result.artifact?.loader} />
      </div>
      <div className="mt-3 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-600">
        {result.routing_reason}；分数来源：{result.prediction?.score_source}
      </div>
      <details className="mt-3">
        <summary className="cursor-pointer text-xs font-semibold text-slate-500">特征对齐详情</summary>
        <div className="mt-2 grid gap-2 md:grid-cols-2">
          <Info label="训练特征数" value={result.feature_schema?.feature_count} />
          <Info label="缺失填 0" value={`${missing.length} 个`} />
          <Info label="被忽略输入" value={`${extra.length} 个`} />
          <div className="rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-500">
            {extra.slice(0, 8).join(", ") || "无额外字段"}
          </div>
        </div>
      </details>
    </div>
  );
}

function BatchPredictView({ result }: { result: BatchPredictResult }) {
  return (
    <div className="rounded-lg border border-sky-100 bg-sky-50 p-3">
      <div className="grid gap-2 md:grid-cols-3">
        <Info label="总请求" value={result.total} />
        <Info label="Champion" value={result.champion_count} />
        <Info label="Challenger" value={`${result.challenger_count} / ${formatValue(result.actual_challenger_ratio)}`} />
      </div>
      <div className="mt-2 max-h-28 overflow-auto text-xs text-sky-700">
        {(result.results || []).slice(0, 12).map((item) => (
          <div key={item.request_id} className="font-mono">{item.request_id}: {item.chosen_role} {formatValue(item.score)}</div>
        ))}
      </div>
    </div>
  );
}

function HistoryList({ title, items, empty }: { title: string; items: DeploymentStageRecord[]; empty: string }) {
  return (
    <div className="mt-3">
      <div className="mb-2 text-xs font-semibold text-slate-500">{title}</div>
      {items.length === 0 ? <Empty text={empty} /> : items.slice(-5).reverse().map((item, index) => (
        <div key={item.stage_record_id || index} className="mb-1 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-600">
          <span className="font-semibold">{formatValue(item.stage)}</span>
          <span className="mx-2">{formatValue(item.decision)}</span>
          <span className="text-slate-400">{fmtTs(item.created_at)}</span>
        </div>
      ))}
    </div>
  );
}

function stageOrder(stage: string) {
  return Object.keys(stageLabels).indexOf(stage);
}

function stageTraffic(stage: string) {
  const map: Record<string, string> = {
    OFFLINE_VALIDATION: "0%",
    OOT_GATE: "0%",
    SHADOW: "0%",
    CANARY_5: "5%",
    CANARY_20: "20%",
    CANARY_50: "50%",
    PRODUCTION: "100%",
  };
  return map[stage] || "-";
}
