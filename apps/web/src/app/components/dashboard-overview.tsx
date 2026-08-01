"use client";

import { useEffect, useState } from "react";
import { requestJson, StatTile, Panel, StatusDot, Badge, formatValue, fmtTs, getApiBase, Items, Empty } from "./shared";

type LifecycleRunItem = {
  lifecycle_run_id: string; model_id: string; champion_version?: string;
  current_phase?: string; created_at?: string; updated_at?: string; state?: Record<string,unknown>;
};
type DeploymentItem = {
  deployment_id: string; model_id: string; candidate_version?: string;
  current_stage?: string; decision?: string; status?: string; created_at?: string;
};

export default function DashboardOverview({ apiBase, onNav }: { apiBase: string; onNav: (k: string) => void }) {
  const [runs, setRuns] = useState<LifecycleRunItem[]>([]);
  const [deployments, setDeployments] = useState<DeploymentItem[]>([]);
  const [kgStats, setKgStats] = useState<Record<string,unknown> | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (loaded) return;
    setLoaded(true);
    Promise.all([
      requestJson<Items<LifecycleRunItem>>(apiBase, "/api/lifecycle-runs?limit=10").then(d => d.items).catch(() => []),
      requestJson<Items<DeploymentItem>>(apiBase, "/api/iteration/deployments?limit=20").then(d => d.items).catch(() => []),
      requestJson<Record<string,unknown>>(apiBase, "/api/kg/stats").catch(() => null),
    ]).then(([r, d, k]) => { setRuns(r); setDeployments(d); setKgStats(k); });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const running = runs.filter(r => !/CLOSED|FAILED|PROMOTED|ROLLED_BACK/.test(String(r.current_phase ?? ""))).length;
  const closed = runs.length - running;
  const activeDeployments = deployments.filter(d => d.status === "RUNNING").length;
  const totalObs = kgStats ? (kgStats.total_observations as number) ?? 0 : 0;

  // Stage distribution
  const stageCounts: Record<string,number> = {};
  deployments.forEach(d => { const s = d.current_stage ?? "unknown"; stageCounts[s] = (stageCounts[s] || 0) + 1; });
  const maxStage = Math.max(1, ...Object.values(stageCounts));

  return (
    <div className="space-y-5 p-5 animate-fade-up">
      {/* Hero */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[.16em] text-indigo-600">RiskItem ModelOps</p>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 mt-1">智能风控模型控制台</h1>
          <p className="text-sm text-slate-500 mt-1">实时监控 · 自动诊断 · 灰度部署 · KG 决策</p>
        </div>
        <button className="rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-indigo-700 transition shadow-sm" onClick={() => onNav("workflow")}>
          启动生命周期 →
        </button>
      </div>

      {/* Stat tiles */}
      <div className="grid gap-4 grid-cols-2 lg:grid-cols-4">
        <StatTile label="运行中" value={String(running)} sub={`${closed} 已关闭`} color="#0ea5e9" />
        <StatTile label="部署中" value={String(activeDeployments)} sub={`${deployments.length} 条部署记录`} color="#8b5cf6" />
        <StatTile label="KG 观测" value={String(totalObs)} sub={kgStats ? `${kgStats.unique_relation_keys ?? 0} 个关系` : "加载中…"} color="#f59e0b" />
        <StatTile label="后端状态" value="在线" sub={getApiBase()} color="#10b981" />
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        {/* Recent lifecycle runs */}
        <Panel title="最近生命周期">
          {runs.length === 0 ? <Empty text="暂无生命周期记录，点击上方按钮启动" /> : (
            <div className="space-y-1">
              {runs.slice(0, 6).map(r => (
                <div key={r.lifecycle_run_id} className="flex items-center gap-3 rounded-lg px-3 py-2.5 hover:bg-slate-50 transition cursor-pointer" onClick={() => onNav("workflow")}>
                  <StatusDot status={r.current_phase} />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-semibold text-slate-800 truncate">{r.model_id}</div>
                    <div className="text-xs text-slate-400 font-mono">{r.lifecycle_run_id?.slice(0, 12)}…</div>
                  </div>
                  <Badge label={(r.current_phase ?? "UNKNOWN").replace(/_/g, " ")} color={
                    /CLOSED|PROMOTED/.test(String(r.current_phase ?? "")) ? "green" :
                    /FAILED|ROLLBACK/.test(String(r.current_phase ?? "")) ? "red" :
                    /RUNNING|CANARY|DEPLOY/.test(String(r.current_phase ?? "")) ? "blue" : "amber"
                  } />
                  <span className="text-xs text-slate-400 w-16 text-right">{fmtTs(r.created_at)?.slice(5)}</span>
                </div>
              ))}
            </div>
          )}
        </Panel>

        {/* Deployment stage distribution */}
        <Panel title="部署阶段分布">
          {deployments.length === 0 ? <Empty text="暂无部署记录" /> : (
            <div className="bar-chart">
              {Object.entries(stageCounts).sort(([a],[b]) => a.localeCompare(b)).map(([stage, count]) => (
                <div key={stage} className="bar-item">
                  <div className="text-xs font-semibold text-slate-700">{count}</div>
                  <div className="bar" style={{
                    height: `${(count/maxStage)*80}px`,
                    background: stage === "PRODUCTION" ? "#10b981" : stage.includes("CANARY") ? "#0ea5e9" : stage === "SHADOW" ? "#8b5cf6" : "#94a3b8"
                  }} />
                  <div className="bar-label">{stage.replace("_", " ").slice(0, 8)}</div>
                </div>
              ))}
            </div>
          )}
        </Panel>
      </div>

      {/* Quick links */}
      <div className="grid gap-3 grid-cols-3">
        {[
          { k: "workflow", label: "流程控制", desc: "启动/管理生命周期", color: "indigo" },
          { k: "deployment", label: "部署监控", desc: "灰度阶段与回滚", color: "violet" },
          { k: "kg", label: "KG 校准", desc: "知识图谱权重管理", color: "amber" },
        ].map(item => (
          <div key={item.k} className="dash-card px-5 py-4 cursor-pointer hover:shadow-md transition" onClick={() => onNav(item.k)}>
            <div className={`text-xs font-semibold uppercase tracking-wider text-${item.color}-600`}>{item.label}</div>
            <div className="text-sm text-slate-500 mt-1">{item.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export { getApiBase };
