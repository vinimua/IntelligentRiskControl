"use client";

import type { ReactNode } from "react";
import { useEffect, useState } from "react";

/* ── types ── */

type Envelope<T> = {
  success?: boolean;
  code?: string;
  message?: string;
  data?: T;
  trace_id?: string;
};

type Items<T> = { items: T[]; total?: number };

type CalibrationRun = {
  calibration_run_id: string;
  data_track: string;
  calibration_rule_version: string;
  target_weight_version?: string;
  status: string;
  relation_count?: number;
  observation_count?: number;
  started_at?: string;
  completed_at?: string;
  created_at?: string;
};

type WeightSnapshot = {
  snapshot_id: string;
  calibration_run_id: string;
  relation_key: string;
  old_effective_weight: number | null;
  new_effective_weight: number;
  confidence_lower_bound: number;
  confidence_upper_bound: number;
  evidence_case_count: number;
  support_count: number;
  against_count: number;
  neutral_count: number;
  support_strength: number;
  against_strength: number;
  weight_version: string;
  applied_to_neo4j?: boolean;
  snapshot_detail?: Record<string, unknown>;
  created_at?: string;
};

type SyncJob = {
  sync_job_id: string;
  calibration_run_id: string;
  idempotency_key: string;
  relation_type: string;
  status: string;
  snapshot_count: number;
  applied_count: number;
  error_message?: string | null;
  weight_version: string;
  applied_to_neo4j: boolean;
  neo4j_applied_at?: string | null;
  started_at?: string;
  completed_at?: string;
  created_at?: string;
};

type RelationKeyInfo = {
  relation_key: string;
  new_effective_weight: number;
  confidence_lower_bound: number;
  confidence_upper_bound: number;
  evidence_case_count: number;
  support_count: number;
  against_count: number;
  neutral_count: number;
  weight_version: string;
  created_at?: string;
};

type TrendPoint = {
  snapshot_id: string;
  calibration_run_id: string;
  new_effective_weight: number;
  old_effective_weight: number | null;
  confidence_lower_bound: number;
  confidence_upper_bound: number;
  evidence_case_count: number;
  support_count: number;
  against_count: number;
  neutral_count: number;
  weight_version: string;
  created_at?: string;
  calibration_run?: CalibrationRun;
};

type KgStats = {
  total_observations: number;
  observations_by_direction: Record<string, number>;
  calibration_runs_by_status: Record<string, number>;
  sync_jobs_by_status: Record<string, number>;
  unique_relation_keys: number;
  latest_calibration: CalibrationRun | null;
};

/* ── helpers ── */

const DEFAULT_API_BASE =
  process.env.NEXT_PUBLIC_MODEL_OPS_API_BASE ?? "http://localhost:8001";

function formatValue(value: unknown): string {
  if (value === undefined || value === null || value === "") return "-";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  if (typeof value === "boolean") return value ? "是" : "否";
  return String(value);
}

function requestJson<T>(apiBase: string, path: string, init?: RequestInit): Promise<T> {
  const base = apiBase.trim().replace(/\/+$/, "");
  const url = `/api/modelops${path}`;

  return fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      "X-ModelOps-Api-Base": base,
      ...(init?.headers ?? {}),
    },
  }).then(async (response) => {
    const payload = await response.json().catch(() => ({
      message: "invalid json",
    }));
    if (!response.ok) {
      throw new Error(
        (payload as Envelope<unknown>).message || JSON.stringify(payload)
      );
    }
    const envelope = payload as Envelope<T>;
    return envelope.data === undefined ? (payload as T) : envelope.data;
  });
}

/* ── sub-components ── */

function Panel({ title, children, action }: { title: string; children: ReactNode; action?: ReactNode }) {
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

function Badge({ label, color }: { label: string; color: "green" | "amber" | "red" | "slate" | "sky" }) {
  const colors: Record<string, string> = {
    green: "border-emerald-200 bg-emerald-50 text-emerald-700",
    amber: "border-amber-200 bg-amber-50 text-amber-700",
    red: "border-red-200 bg-red-50 text-red-700",
    slate: "border-slate-200 bg-slate-50 text-slate-600",
    sky: "border-sky-200 bg-sky-50 text-sky-700",
  };
  return (
    <span className={`rounded-md border px-2 py-0.5 text-xs font-semibold ${colors[color]}`}>
      {label}
    </span>
  );
}

function statusBadgeColor(status: string): "green" | "amber" | "red" | "slate" | "sky" {
  const s = (status ?? "").toUpperCase();
  if (s === "SUCCEEDED" || s === "COMPLETED") return "green";
  if (s === "RUNNING" || s === "PENDING") return "sky";
  if (s === "FAILED") return "red";
  return "slate";
}

function Spinner() {
  return <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-slate-700" />;
}

/* ── Trend Chart (SVG) ── */

function TrendChart({ points, height = 180 }: { points: TrendPoint[]; height?: number }) {
  if (points.length === 0) {
    return <p className="py-8 text-center text-xs text-slate-400">暂无趋势数据</p>;
  }

  const WIDTH = 640;
  const H = height;
  const PAD = { top: 20, right: 20, bottom: 40, left: 50 };
  const W = WIDTH - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const weights = points.map((p) => p.new_effective_weight);
  const lowers = points.map((p) => p.confidence_lower_bound);
  const uppers = points.map((p) => p.confidence_upper_bound);
  const allVals = [...weights, ...lowers, ...uppers];
  const yMin = Math.max(0, Math.min(...allVals) - 0.05);
  const yMax = Math.min(1, Math.max(...allVals) + 0.05);

  const xScale = (i: number) => PAD.left + (i / Math.max(1, points.length - 1)) * W;
  const yScale = (v: number) => PAD.top + plotH * (1 - (v - yMin) / (yMax - yMin || 0.01));

  // Build line paths
  const weightPath = points
    .map((p, i) => `${i === 0 ? "M" : "L"}${xScale(i)},${yScale(p.new_effective_weight)}`)
    .join(" ");
  const upperPath = points
    .map((p, i) => `${i === 0 ? "M" : "L"}${xScale(i)},${yScale(p.confidence_upper_bound)}`)
    .join(" ");
  const lowerPath = points
    .map((p, i) => `${i === 0 ? "M" : "L"}${xScale(i)},${yScale(p.confidence_lower_bound)}`)
    .join(" ");

  // Y-axis ticks
  const yTicks = [yMin, (yMin + yMax) / 2, yMax].map((v) => Math.round(v * 100) / 100);

  return (
    <svg viewBox={`0 0 ${WIDTH} ${H}`} className="w-full" style={{ maxHeight: height + 40 }}>
      {/* Grid lines */}
      {yTicks.map((t) => (
        <line
          key={`grid-${t}`}
          x1={PAD.left} y1={yScale(t)} x2={PAD.left + W} y2={yScale(t)}
          stroke="#e2e8f0" strokeWidth={1} strokeDasharray="4 3"
        />
      ))}

      {/* Confidence band */}
      {points.length >= 2 ? (
        <path
          d={`${upperPath} L${xScale(points.length - 1)},${yScale(points[points.length - 1].confidence_lower_bound)} ${lowerPath.split(" ").slice(1).join(" ")} Z`}
          fill="rgba(14,165,233,0.12)" stroke="none"
        />
      ) : null}

      {/* Weight line */}
      <path d={weightPath} fill="none" stroke="#0284c7" strokeWidth={2.5} strokeLinejoin="round" />

      {/* Data points */}
      {points.map((p, i) => (
        <circle
          key={p.snapshot_id}
          cx={xScale(i)} cy={yScale(p.new_effective_weight)} r={4}
          fill="white" stroke="#0284c7" strokeWidth={2}
        />
      ))}

      {/* Y-axis labels */}
      {yTicks.map((t) => (
        <text key={`ylbl-${t}`} x={PAD.left - 8} y={yScale(t) + 4} textAnchor="end" className="fill-slate-400 text-[9px]">
          {t.toFixed(2)}
        </text>
      ))}

      {/* X-axis labels (show first, middle, last) */}
      {points.length > 0 ? (
        [0, Math.floor((points.length - 1) / 2), points.length - 1]
          .filter((i, idx, arr) => arr.indexOf(i) === idx)
          .map((i) => {
            const label = points[i].weight_version?.replace("KG_WEIGHT_", "").replace("BETA_", "") || `#${i + 1}`;
            return (
              <text key={`xlbl-${i}`} x={xScale(i)} y={H - 8} textAnchor="middle" className="fill-slate-400 text-[8px]">
                {label.length > 12 ? label.slice(0, 12) + "…" : label}
              </text>
            );
          })
      ) : null}
    </svg>
  );
}

/* ── Relation selector ── */

function RelationSelector({
  apiBase,
  selected,
  onSelect,
}: {
  apiBase: string;
  selected: string;
  onSelect: (key: string) => void;
}) {
  const [keys, setKeys] = useState<RelationKeyInfo[]>([]);

  useEffect(() => {
    requestJson<Items<RelationKeyInfo>>(apiBase, "/api/kg/relation-keys")
      .then((d) => setKeys(d.items))
      .catch(() => setKeys([]));
  }, [apiBase]);

  return (
    <select
      className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm"
      value={selected}
      onChange={(e) => onSelect(e.target.value)}
    >
      <option value="">选择关系 (relation_key)</option>
      {keys.map((k) => (
        <option key={k.relation_key} value={k.relation_key}>
          {k.relation_key} (w={k.new_effective_weight?.toFixed(3)})
        </option>
      ))}
    </select>
  );
}

/* ── Main Panel ── */

export default function KgCalibrationPanel({ apiBase: propsApiBase }: { apiBase?: string }) {
  const apiBase = propsApiBase ?? DEFAULT_API_BASE;

  // State
  const [stats, setStats] = useState<KgStats | null>(null);
  const [runs, setRuns] = useState<CalibrationRun[]>([]);
  const [syncJobs, setSyncJobs] = useState<SyncJob[]>([]);
  const [snapshots, setSnapshots] = useState<WeightSnapshot[]>([]);
  const [trend, setTrend] = useState<TrendPoint[]>([]);
  const [selectedRelation, setSelectedRelation] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<{ type: "ok" | "error"; text: string } | null>(null);
  const [activeSubTab, setActiveSubTab] = useState<"overview" | "runs" | "trend" | "sync">("overview");

  // Load stats on mount
  useEffect(() => {
    requestJson<KgStats>(apiBase, "/api/kg/stats")
      .then(setStats)
      .catch(() => {});
  }, [apiBase]);

  // Load runs
  async function loadRuns() {
    setBusy("runs");
    try {
      const d = await requestJson<Items<CalibrationRun>>(apiBase, "/api/kg/calibration-runs?limit=30");
      setRuns(d.items);
    } catch (e) {
      setMessage({ type: "error", text: String(e) });
    } finally {
      setBusy(null);
    }
  }

  // Load sync jobs
  async function loadSyncJobs() {
    setBusy("sync");
    try {
      const d = await requestJson<Items<SyncJob>>(apiBase, "/api/kg/sync-jobs?limit=30");
      setSyncJobs(d.items);
    } catch (e) {
      setMessage({ type: "error", text: String(e) });
    } finally {
      setBusy(null);
    }
  }

  // Load snapshots for latest run
  async function loadSnapshots(runId: string) {
    setBusy("snapshots");
    try {
      const d = await requestJson<{ run: CalibrationRun; snapshots: WeightSnapshot[] }>(
        apiBase,
        `/api/kg/calibration-runs/${runId}`
      );
      setSnapshots(d.snapshots);
    } catch (e) {
      setMessage({ type: "error", text: String(e) });
    } finally {
      setBusy(null);
    }
  }

  // Load trend for selected relation
  useEffect(() => {
    if (!selectedRelation) {
      setTrend([]);
      return;
    }
    setBusy("trend");
    requestJson<{ relation_key: string; trend: TrendPoint[] }>(
      apiBase,
      `/api/kg/weight-trend/${encodeURIComponent(selectedRelation)}?limit=30`
    )
      .then((d) => setTrend(d.trend))
      .catch((e) => setMessage({ type: "error", text: String(e) }))
      .finally(() => setBusy(null));
  }, [apiBase, selectedRelation]);

  // Trigger calibration
  async function triggerCalibration() {
    setBusy("calibrate");
    setMessage(null);
    try {
      const d = await requestJson<{ calibration_run_id: string }>(apiBase, "/api/kg/calibration-runs", {
        method: "POST",
        body: JSON.stringify({ data_track: "NATURAL", rule_version: "BETA_BINOMIAL_V2", weight_version: "KG_WEIGHT_MANUAL_V1" }),
      });
      setMessage({ type: "ok", text: `校准完成：${d.calibration_run_id}` });
      loadRuns();
    } catch (e) {
      setMessage({ type: "error", text: String(e) });
    } finally {
      setBusy(null);
    }
  }

  // Apply to Neo4j
  async function applyToNeo4j(runId: string) {
    setBusy("apply");
    setMessage(null);
    try {
      const d = await requestJson<{ applied: number }>(apiBase, `/api/kg/calibration-runs/${runId}/apply-to-neo4j`, {
        method: "POST",
        body: JSON.stringify({}),
      });
      setMessage({ type: "ok", text: `已同步 ${d.applied} 条关系到 Neo4j` });
      loadSyncJobs();
    } catch (e) {
      setMessage({ type: "error", text: String(e) });
    } finally {
      setBusy(null);
    }
  }

  return (
    <section className="flex flex-col gap-5">
      {/* Stats overview */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <StatCard
          label="总观测数"
          value={formatValue(stats?.total_observations)}
          sub={`${stats?.unique_relation_keys ?? "-"} 个关系`}
        />
        <StatCard
          label="SUPPORT 观测"
          value={formatValue(stats?.observations_by_direction?.SUPPORT)}
          sub="支持 KG 关系的证据"
        />
        <StatCard
          label="AGAINST 观测"
          value={formatValue(stats?.observations_by_direction?.AGAINST)}
          sub="反对 KG 关系的证据"
        />
        <StatCard
          label="NEUTRAL 观测"
          value={formatValue(stats?.observations_by_direction?.NEUTRAL)}
          sub="中性观测"
        />
        <StatCard
          label="最新校准"
          value={stats?.latest_calibration?.status ?? "无"}
          sub={stats?.latest_calibration?.completed_at?.slice(0, 16) ?? "尚未执行"}
        />
      </div>

      {message ? (
        <div
          className={`rounded-md border px-4 py-3 text-sm font-medium ${
            message.type === "ok"
              ? "border-emerald-200 bg-emerald-50 text-emerald-700"
              : "border-red-200 bg-red-50 text-red-700"
          }`}
        >
          {message.text}
        </div>
      ) : null}

      {/* Sub-tabs */}
      <nav className="flex flex-wrap gap-2">
        {(["overview", "runs", "trend", "sync"] as const).map((tab) => (
          <button
            key={tab}
            className={`rounded-md px-4 py-2 text-sm font-semibold ${
              activeSubTab === tab
                ? "bg-slate-950 text-white"
                : "border border-slate-300 bg-white text-slate-700 hover:bg-slate-50"
            }`}
            onClick={() => {
              setActiveSubTab(tab);
              if (tab === "runs") loadRuns();
              if (tab === "sync") loadSyncJobs();
            }}
          >
            {{ overview: "概览", runs: "校准运行", trend: "权重趋势", sync: "Neo4j 同步" }[tab]}
          </button>
        ))}
      </nav>

      {/* Overview */}
      {activeSubTab === "overview" ? (
        <div className="grid gap-5 lg:grid-cols-2">
          <Panel
            title="快速操作"
            action={
              <div className="flex gap-2">
                <button
                  className="rounded-md bg-slate-950 px-3 py-2 text-sm font-semibold text-white hover:bg-slate-800 disabled:bg-slate-400"
                  disabled={busy === "calibrate"}
                  onClick={triggerCalibration}
                >
                  {busy === "calibrate" ? <Spinner /> : "执行校准"}
                </button>
                <button
                  className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-semibold hover:bg-slate-50"
                  onClick={() => {
                    loadRuns();
                    setActiveSubTab("runs");
                  }}
                >
                  查看运行
                </button>
              </div>
            }
          >
            <div className="space-y-2 text-sm text-slate-600">
              <p>
                <span className="font-semibold">校准算法：</span>
                Beta-Binomial V2（先验 Beta(2, 8)，后验收缩到 [0.03, 0.85]）
              </p>
              <p>
                <span className="font-semibold">数据轨道：</span>
                NATURAL（真实生命周期产出）
              </p>
              <p>
                <span className="font-semibold">关系类型：</span>
                INDICATES / RECOMMENDS / MITIGATES
              </p>
              <p>
                <span className="font-semibold">Celery 定时：</span>
                每 6 小时自动执行校准 → Neo4j 同步
              </p>
            </div>
          </Panel>

          <Panel title="运行状态分布">
            <div className="grid gap-2">
              {Object.entries(stats?.calibration_runs_by_status ?? {}).map(([status, count]) => (
                <div key={status} className="flex items-center justify-between rounded-md bg-slate-50 px-3 py-2">
                  <Badge label={status} color={statusBadgeColor(status)} />
                  <span className="text-sm font-semibold text-slate-700">{count}</span>
                </div>
              ))}
              {Object.keys(stats?.calibration_runs_by_status ?? {}).length === 0 ? (
                <p className="py-4 text-center text-sm text-slate-400">暂无校准运行记录</p>
              ) : null}
            </div>
          </Panel>
        </div>
      ) : null}

      {/* Runs */}
      {activeSubTab === "runs" ? (
        <Panel
          title="校准运行记录"
          action={
            <button
              className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-semibold hover:bg-slate-50"
              onClick={loadRuns}
              disabled={busy === "runs"}
            >
              {busy === "runs" ? <Spinner /> : "刷新"}
            </button>
          }
        >
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-xs text-slate-500">
                  <th className="px-3 py-2">Run ID</th>
                  <th className="px-3 py-2">数据轨道</th>
                  <th className="px-3 py-2">状态</th>
                  <th className="px-3 py-2">关系数</th>
                  <th className="px-3 py-2">观测数</th>
                  <th className="px-3 py-2">权重版本</th>
                  <th className="px-3 py-2">时间</th>
                  <th className="px-3 py-2">操作</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr key={run.calibration_run_id} className="border-b border-slate-100 hover:bg-slate-50">
                    <td className="px-3 py-2 font-mono text-xs">{run.calibration_run_id.slice(0, 12)}…</td>
                    <td className="px-3 py-2">{run.data_track}</td>
                    <td className="px-3 py-2">
                      <Badge label={run.status} color={statusBadgeColor(run.status)} />
                    </td>
                    <td className="px-3 py-2">{formatValue(run.relation_count)}</td>
                    <td className="px-3 py-2">{formatValue(run.observation_count)}</td>
                    <td className="px-3 py-2 font-mono text-xs">{run.target_weight_version ?? "-"}</td>
                    <td className="px-3 py-2 text-xs">{run.completed_at?.slice(0, 16) ?? run.started_at?.slice(0, 16) ?? "-"}</td>
                    <td className="px-3 py-2">
                      <div className="flex gap-1">
                        <button
                          className="rounded border border-sky-200 bg-sky-50 px-2 py-1 text-xs text-sky-700 hover:bg-sky-100"
                          onClick={() => loadSnapshots(run.calibration_run_id)}
                        >
                          快照
                        </button>
                        {run.status === "SUCCEEDED" ? (
                          <button
                            className="rounded border border-emerald-200 bg-emerald-50 px-2 py-1 text-xs text-emerald-700 hover:bg-emerald-100"
                            onClick={() => applyToNeo4j(run.calibration_run_id)}
                            disabled={busy === "apply"}
                          >
                            同步
                          </button>
                        ) : null}
                      </div>
                    </td>
                  </tr>
                ))}
                {runs.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="px-3 py-8 text-center text-slate-400">暂无记录</td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>

          {/* Snapshots panel */}
          {snapshots.length > 0 ? (
            <div className="mt-5 border-t border-slate-200 pt-4">
              <h3 className="mb-3 text-sm font-semibold text-slate-800">权重快照</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 text-left text-xs text-slate-500">
                      <th className="px-3 py-2">Relation Key</th>
                      <th className="px-3 py-2">新权重</th>
                      <th className="px-3 py-2">置信下界</th>
                      <th className="px-3 py-2">置信上界</th>
                      <th className="px-3 py-2">SUPPORT</th>
                      <th className="px-3 py-2">AGAINST</th>
                      <th className="px-3 py-2">NEUTRAL</th>
                      <th className="px-3 py-2">已同步</th>
                    </tr>
                  </thead>
                  <tbody>
                    {snapshots.map((snap) => (
                      <tr key={snap.snapshot_id} className="border-b border-slate-100 hover:bg-slate-50">
                        <td className="px-3 py-2 font-mono text-xs">{snap.relation_key}</td>
                        <td className="px-3 py-2 font-semibold">{snap.new_effective_weight?.toFixed(4)}</td>
                        <td className="px-3 py-2">{snap.confidence_lower_bound?.toFixed(4)}</td>
                        <td className="px-3 py-2">{snap.confidence_upper_bound?.toFixed(4)}</td>
                        <td className="px-3 py-2 text-emerald-600">{snap.support_count}</td>
                        <td className="px-3 py-2 text-red-500">{snap.against_count}</td>
                        <td className="px-3 py-2 text-slate-400">{snap.neutral_count}</td>
                        <td className="px-3 py-2">
                          {snap.applied_to_neo4j ? (
                            <Badge label="已同步" color="green" />
                          ) : (
                            <Badge label="未同步" color="amber" />
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}
        </Panel>
      ) : null}

      {/* Trend */}
      {activeSubTab === "trend" ? (
        <Panel
          title="权重变化趋势"
          action={
            <RelationSelector apiBase={apiBase} selected={selectedRelation} onSelect={setSelectedRelation} />
          }
        >
          {!selectedRelation ? (
            <p className="py-8 text-center text-sm text-slate-400">
              请先选择一个 relation_key 查看权重变化趋势。
            </p>
          ) : busy === "trend" ? (
            <div className="flex items-center justify-center py-12 gap-2 text-sm text-slate-400">
              <Spinner /> 加载中…
            </div>
          ) : (
            <div className="space-y-4">
              <TrendChart points={trend} height={200} />
              {trend.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-200 text-left text-xs text-slate-500">
                        <th className="px-3 py-2">版本</th>
                        <th className="px-3 py-2">权重</th>
                        <th className="px-3 py-2">CI 下界</th>
                        <th className="px-3 py-2">CI 上界</th>
                        <th className="px-3 py-2">SUPPORT</th>
                        <th className="px-3 py-2">AGAINST</th>
                        <th className="px-3 py-2">时间</th>
                      </tr>
                    </thead>
                    <tbody>
                      {trend.map((p) => (
                        <tr key={p.snapshot_id} className="border-b border-slate-100 hover:bg-slate-50">
                          <td className="px-3 py-2 font-mono text-xs">{p.weight_version?.replace("KG_WEIGHT_", "")}</td>
                          <td className="px-3 py-2 font-semibold text-sky-700">{p.new_effective_weight?.toFixed(4)}</td>
                          <td className="px-3 py-2 text-xs">{p.confidence_lower_bound?.toFixed(4)}</td>
                          <td className="px-3 py-2 text-xs">{p.confidence_upper_bound?.toFixed(4)}</td>
                          <td className="px-3 py-2 text-emerald-600">{p.support_count}</td>
                          <td className="px-3 py-2 text-red-500">{p.against_count}</td>
                          <td className="px-3 py-2 text-xs">{p.created_at?.slice(0, 10) ?? "-"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null}
            </div>
          )}
        </Panel>
      ) : null}

      {/* Sync jobs */}
      {activeSubTab === "sync" ? (
        <Panel
          title="Neo4j 同步任务"
          action={
            <button
              className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-semibold hover:bg-slate-50"
              onClick={loadSyncJobs}
              disabled={busy === "sync"}
            >
              {busy === "sync" ? <Spinner /> : "刷新"}
            </button>
          }
        >
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-xs text-slate-500">
                  <th className="px-3 py-2">Job ID</th>
                  <th className="px-3 py-2">关系类型</th>
                  <th className="px-3 py-2">状态</th>
                  <th className="px-3 py-2">快照数</th>
                  <th className="px-3 py-2">已同步</th>
                  <th className="px-3 py-2">权重版本</th>
                  <th className="px-3 py-2">错误</th>
                  <th className="px-3 py-2">时间</th>
                </tr>
              </thead>
              <tbody>
                {syncJobs.map((job) => (
                  <tr key={job.sync_job_id} className="border-b border-slate-100 hover:bg-slate-50">
                    <td className="px-3 py-2 font-mono text-xs">{job.sync_job_id.slice(0, 12)}…</td>
                    <td className="px-3 py-2">
                      <Badge
                        label={job.relation_type}
                        color={
                          job.relation_type === "INDICATES" ? "sky"
                          : job.relation_type === "RECOMMENDS" ? "green"
                          : job.relation_type === "MITIGATES" ? "amber"
                          : "slate"
                        }
                      />
                    </td>
                    <td className="px-3 py-2">
                      <Badge label={job.status} color={statusBadgeColor(job.status)} />
                    </td>
                    <td className="px-3 py-2">{job.snapshot_count}</td>
                    <td className="px-3 py-2">{job.applied_count}</td>
                    <td className="px-3 py-2 font-mono text-xs">{job.weight_version?.replace("KG_WEIGHT_", "") ?? "-"}</td>
                    <td className="px-3 py-2 text-xs text-red-500 max-w-[200px] truncate">
                      {job.error_message ?? "-"}
                    </td>
                    <td className="px-3 py-2 text-xs">{job.completed_at?.slice(0, 16) ?? job.started_at?.slice(0, 16) ?? "-"}</td>
                  </tr>
                ))}
                {syncJobs.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="px-3 py-8 text-center text-slate-400">暂无同步记录</td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </Panel>
      ) : null}
    </section>
  );
}

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-3">
      <p className="text-xs font-medium text-slate-500">{label}</p>
      <p className="mt-1 break-words text-lg font-semibold text-slate-950">{value}</p>
      {sub ? <p className="mt-1 text-xs text-slate-500">{sub}</p> : null}
    </div>
  );
}
