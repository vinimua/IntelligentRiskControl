"use client";

import { useEffect, useState } from "react";
import { requestJson, Panel, StatusDot, Badge, formatValue, Btn, Spinner, Empty } from "./shared";
import type {
  MonitoringRun,
  EnrichedMetric,
  CoverageSummary,
  FeatureDriftItem,
  DataQualityData,
  AlertItem,
  EnrichedMetricsResponse,
  PersistenceJudgment,
} from "./monitoring/monitoring-types";
import { CATEGORY_ORDER } from "./monitoring/monitoring-types";
import StatusBar from "./monitoring/status-bar";
import PersistenceCard from "./monitoring/persistence-card";
import CategoryCards from "./monitoring/category-cards";
import MetricDetailSection from "./monitoring/metric-detail-section";
import AlertExplanationPanel from "./monitoring/alert-explanation-panel";
import FeatureDriftTable from "./monitoring/feature-drift-table";
import DataQualityPanel from "./monitoring/data-quality-panel";
import RuleCoverageTable from "./monitoring/rule-coverage-table";

type Props = { apiBase: string };

type Items<T> = { items: T[] };

export default function MonitoringPanel({ apiBase }: Props) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 运行列表
  const [runs, setRuns] = useState<MonitoringRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState("");

  // 富化指标
  const [metrics, setMetrics] = useState<EnrichedMetric[]>([]);
  const [summary, setSummary] = useState<CoverageSummary | null>(null);

  // 告警
  const [alerts, setAlerts] = useState<AlertItem[]>([]);

  // 特征漂移
  const [driftItems, setDriftItems] = useState<FeatureDriftItem[]>([]);
  const [driftLoading, setDriftLoading] = useState(false);

  // 数据质量
  const [qualityData, setQualityData] = useState<DataQualityData | null>(null);
  const [qualityLoading, setQualityLoading] = useState(false);

  // B1 持续性判定
  const [persistence, setPersistence] = useState<PersistenceJudgment | null>(null);
  const [diagnosisStatus, setDiagnosisStatus] = useState<string | null>(null);

  // 底部 Tab
  const [bottomTab, setBottomTab] = useState<string>("drift");

  // 指标类别 Tab
  const [activeCategory, setActiveCategory] = useState<string>(CATEGORY_ORDER[0]);

  // 选中的 run 信息
  const selectedRun = runs.find((r) => r.monitoring_run_id === selectedRunId) || null;

  // ── 加载运行列表 ──
  async function loadRuns() {
    setBusy(true);
    setError(null);
    try {
      const d = await requestJson<Items<MonitoringRun>>(apiBase, "/api/monitoring/runs?limit=20");
      setRuns(d.items || []);
      const firstId = d.items?.[0]?.monitoring_run_id;
      if (firstId) {
        setSelectedRunId(firstId);
        await loadRunData(firstId);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载运行列表失败");
    } finally {
      setBusy(false);
    }
  }

  // ── 加载运行数据 ──
  async function loadRunData(runId: string) {
    setError(null);

    // 并行加载 enriched-metrics 和 alerts
    try {
      const [metricsRes, alertsRes] = await Promise.all([
        requestJson<EnrichedMetricsResponse>(apiBase, `/api/monitoring/runs/${runId}/enriched-metrics`).catch(() => null),
        requestJson<Items<AlertItem>>(apiBase, `/api/monitoring/runs/${runId}/alerts`).catch(() => null),
      ]);

      if (metricsRes) {
        setMetrics(metricsRes.metrics || []);
        setSummary(metricsRes.summary || null);
        setPersistence(metricsRes.persistence || null);
        setDiagnosisStatus(metricsRes.diagnosis_status || null);
      }
      if (alertsRes) {
        setAlerts(alertsRes.items || []);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载指标数据失败");
    }
  }

  // ── 懒加载特征漂移 ──
  async function loadDrift(runId: string) {
    setDriftLoading(true);
    try {
      const d = await requestJson<Items<FeatureDriftItem>>(apiBase, `/api/monitoring/runs/${runId}/feature-drift`);
      setDriftItems(d.items || []);
    } catch {
      // 静默失败
    } finally {
      setDriftLoading(false);
    }
  }

  // ── 懒加载数据质量 ──
  async function loadQuality(runId: string) {
    setQualityLoading(true);
    try {
      const d = await requestJson<DataQualityData>(apiBase, `/api/monitoring/runs/${runId}/data-quality`);
      setQualityData(d);
    } catch {
      // 静默失败
    } finally {
      setQualityLoading(false);
    }
  }

  // 切换 Run
  function selectRun(runId: string) {
    setSelectedRunId(runId);
    loadRunData(runId);
    // 重置懒加载状态
    setDriftItems([]);
    setQualityData(null);
  }

  // 切换底部 Tab 时懒加载
  useEffect(() => {
    if (!selectedRunId) return;
    if (bottomTab === "drift" && driftItems.length === 0) loadDrift(selectedRunId);
    if (bottomTab === "quality" && !qualityData) loadQuality(selectedRunId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bottomTab, selectedRunId]);

  // ── 初始加载 ──
  useEffect(() => {
    loadRuns();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const bottomTabs = [
    { key: "drift", label: "特征漂移" },
    { key: "quality", label: "数据质量" },
    { key: "coverage", label: "规则覆盖" },
  ];

  return (
    <div className="space-y-5 p-5">
      {/* 标题 */}
      <div>
        <p className="text-[11px] font-semibold uppercase tracking-[.16em] text-indigo-600">监控判定台</p>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 mt-1">模型监控与漂移检测</h1>
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 flex items-center justify-between">
          <span>{error}</span>
          <button onClick={() => { setError(null); loadRuns(); }} className="text-xs font-semibold underline">重试</button>
        </div>
      )}

      <div className="grid gap-5 lg:grid-cols-[300px_1fr]">
        {/* ── 左侧：运行列表 ── */}
        <Panel
          title="监控运行"
          action={
            <Btn onClick={loadRuns} disabled={busy}>
              {busy ? <Spinner /> : "刷新"}
            </Btn>
          }
        >
          <div className="max-h-[500px] space-y-1 overflow-auto">
            {runs.length === 0 ? (
              <Empty text="暂无监控运行" />
            ) : (
              runs.map((r) => {
                const isActive = selectedRunId === r.monitoring_run_id;
                return (
                  <div
                    key={r.monitoring_run_id}
                    className={`flex items-start gap-3 rounded-lg px-3 py-2.5 cursor-pointer transition hover:bg-slate-50 ${
                      isActive ? "bg-indigo-50 border border-indigo-100" : ""
                    }`}
                    onClick={() => selectRun(r.monitoring_run_id)}
                  >
                      <StatusDot status={r.overall_status} />
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-semibold text-slate-800 truncate">{formatValue(r.model_id)}</div>
                        <div className="text-xs text-slate-400 font-mono">{formatValue(r.champion_version)}</div>
                        <div className="mt-0.5 text-[11px] text-slate-500">
                          生命周期{" "}
                          <span className="font-mono text-slate-600 break-all">
                            {r.lifecycle_run_id ?? "-"}
                          </span>
                        </div>
                      </div>
                    <Badge
                      label={`${r.alert_count ?? 0} 告警`}
                      color={(r.alert_count ?? 0) > 0 ? "red" : "green"}
                    />
                  </div>
                );
              })
            )}
          </div>
        </Panel>

        {/* ── 右侧：判定台主体 ── */}
        <div className="space-y-5">
          {!selectedRun ? (
            <Empty text="请选择一次监控运行查看判定结果" />
          ) : (
            <>
              {/* 状态栏 */}
              <StatusBar summary={summary} alerts={alerts} runInfo={selectedRun} />

              {/* B1 持续性判定 */}
              <PersistenceCard
                persistence={persistence}
                diagnosisStatus={diagnosisStatus}
                visibleAlertCount={alerts.length}
              />

              {/* 类别概览卡片 */}
              <CategoryCards
                summary={summary}
                activeCategory={activeCategory}
                onSelectCategory={setActiveCategory}
              />

              {/* 指标详情 + 告警面板 双栏 */}
              <div className="grid gap-5 lg:grid-cols-[1fr_320px]">
                <Panel title="指标详情与趋势">
                  <MetricDetailSection
                    metrics={metrics}
                    activeCategory={activeCategory}
                    onSelectCategory={setActiveCategory}
                  />
                </Panel>

                <AlertExplanationPanel summary={summary} alerts={alerts} />
              </div>

              {/* 底部 Tabs */}
              <div>
                <div className="flex items-center gap-1 border-b border-slate-200 mb-4">
                  {bottomTabs.map((tab) => (
                    <button
                      key={tab.key}
                      onClick={() => setBottomTab(tab.key)}
                      className={`rounded-t-lg px-4 py-2 text-sm font-semibold transition ${
                        bottomTab === tab.key
                          ? "border-b-2 border-indigo-600 text-indigo-700 bg-indigo-50/50"
                          : "text-slate-500 hover:text-slate-700 hover:bg-slate-50"
                      }`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>

                {bottomTab === "drift" && (
                  <Panel title="特征漂移明细">
                    <FeatureDriftTable items={driftItems} loading={driftLoading} />
                  </Panel>
                )}

                {bottomTab === "quality" && (
                  <Panel title="数据质量下钻">
                    <DataQualityPanel data={qualityData} loading={qualityLoading} />
                  </Panel>
                )}

                {bottomTab === "coverage" && (
                  <Panel title="规则覆盖">
                    <RuleCoverageTable metrics={metrics} summary={summary} />
                  </Panel>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
