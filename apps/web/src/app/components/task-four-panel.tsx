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

export default function TaskFourPanel({ apiBase, initialModelId }: { apiBase: string; initialModelId?: string }) {
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
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<{ type: "ok" | "error"; text: string } | null>(null);

  const selectedDeployment = detail?.deployment || deployments.find((item) => item.deployment_id === selectedDeploymentId);
  const stage = selectedDeployment?.current_stage || detail?.stages?.at(-1)?.stage || "OFFLINE_VALIDATION";
  const stageMeta = stageLabels[stage] || { label: stage, desc: "" };

  const trafficPercent = useMemo(() => {
    const raw = inferenceRouting?.challenger_traffic_ratio ?? routing?.challenger_traffic_ratio ?? 0;
    const ratio = Number(raw || 0);
    return `${Math.round(ratio * 100)}%`;
  }, [routing, inferenceRouting]);

  useEffect(() => {
    void refreshAll(false);
  }, []);

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
      const list = await requestJson<Items<DeploymentItem>>(apiBase, `/api/iteration/deployments?model_id=${encodeURIComponent(modelId)}&limit=50`);
      setDeployments(list.items || []);
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
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[.16em] text-indigo-600">任务四操作台</p>
          <h1 className="mt-1 text-2xl font-bold tracking-tight text-slate-900">验证、灰度、真实推理与回滚</h1>
          <p className="mt-1 text-sm text-slate-500">从模型验证到部署灰度，再到推理流量分流，集中查看任务四闭环。</p>
        </div>
        <div className="flex items-center gap-2">
          <input
            className="w-56 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-mono focus:border-indigo-400 focus:outline-none"
            value={modelId}
            onChange={(event) => setModelId(event.target.value)}
          />
          <Btn onClick={() => refreshAll()} disabled={busy === "refresh"}>{busy === "refresh" ? <Spinner /> : "刷新"}</Btn>
        </div>
      </div>

      {message ? (
        <div className={`rounded-lg border px-4 py-3 text-sm ${message.type === "ok" ? "border-emerald-200 bg-emerald-50 text-emerald-700" : "border-red-200 bg-red-50 text-red-700"}`}>
          {message.text}
        </div>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-4">
        <StatTile label="生产版本" value={formatValue(inferenceRouting?.active_version_code || routing?.active_version_code)} sub="当前承接主流量" />
        <StatTile label="稳定回滚版本" value={formatValue(inferenceRouting?.stable_version_code || routing?.stable_version_code)} sub="异常时恢复目标" />
        <StatTile label="候选灰度版本" value={formatValue(inferenceRouting?.challenger_version_code || routing?.challenger_version_code)} sub={`当前流量 ${trafficPercent}`} />
        <StatTile label="当前部署阶段" value={stageMeta.label} sub={stageMeta.desc} />
      </div>

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
              <Btn danger onClick={rollbackSelected} disabled={busy === "rollback" || !selectedDeploymentId}>手动回滚</Btn>
              <Btn onClick={() => loadRollbackEvents()} disabled={!selectedDeploymentId}>查询回滚事件</Btn>
            </div>
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
