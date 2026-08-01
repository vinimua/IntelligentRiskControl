"use client";

import { useEffect, useState } from "react";
import { requestJson, Panel, StatusDot, Badge, formatValue, fmtTs, Btn, Items, Empty, Spinner } from "./shared";

type DeploymentItem = {
  deployment_id: string; model_id: string; champion_version?: string; candidate_version?: string;
  current_stage?: string; decision?: string; status?: string; created_at?: string; updated_at?: string;
};

type DeploymentDetail = {
  deployment: DeploymentItem;
  stages: Array<{
    stage_record_id: string; stage: string; decision: string; status: string;
    health_json?: Record<string,unknown>; created_at?: string;
  }>;
};

export default function DeploymentPanel({ apiBase }: { apiBase: string }) {
  const [deployments, setDeployments] = useState<DeploymentItem[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [detail, setDetail] = useState<DeploymentDetail | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    requestJson<Items<DeploymentItem>>(apiBase, "/api/iteration/deployments?limit=30")
      .then(d => setDeployments(d.items)).catch(() => {});
  }, [apiBase]);

  async function loadDetail(id: string) {
    setSelected(id); setBusy(true);
    try {
      const d = await requestJson<DeploymentDetail>(apiBase, `/api/iteration/deployments/${id}`);
      setDetail(d);
    } catch { setDetail(null); }
    finally { setBusy(false); }
  }

  async function rollback(id: string) {
    if (!confirm("确认回滚此部署？challenger 流量将归零，恢复 stable 版本。")) return;
    try {
      await requestJson(apiBase, `/api/iteration/deployments/${id}/rollback`, { method: "POST", body: JSON.stringify({ reason: "MANUAL_ROLLBACK" }) });
      loadDetail(id);
      // Refresh list
      requestJson<Items<DeploymentItem>>(apiBase, "/api/iteration/deployments?limit=30").then(d => setDeployments(d.items));
    } catch (e) { alert(String(e)); }
  }

  return (
    <div className="space-y-5 p-5">
      <div>
        <p className="text-[11px] font-semibold uppercase tracking-[.16em] text-indigo-600">部署监控</p>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 mt-1">部署状态与阶段管控</h1>
      </div>

      <div className="grid gap-5 lg:grid-cols-[1fr_2fr]">
        {/* Deployment list */}
        <Panel title={`部署记录 (${deployments.length})`}>
          <div className="space-y-1 max-h-[500px] overflow-auto">
            {deployments.length === 0 ? <Empty text="暂无部署记录" /> : deployments.map(d => (
              <div key={d.deployment_id}
                className={`flex items-center gap-3 rounded-lg px-3 py-2.5 cursor-pointer transition hover:bg-slate-50 ${selected === d.deployment_id ? "bg-indigo-50 border border-indigo-100" : ""}`}
                onClick={() => loadDetail(d.deployment_id)}>
                <StatusDot status={d.status} />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-semibold text-slate-800 truncate">{d.model_id}</div>
                  <div className="text-xs text-slate-400">{d.candidate_version ?? "-"} / {fmtTs(d.updated_at ?? d.created_at)}</div>
                </div>
                <Badge label={d.current_stage ?? "?"} color={
                  d.current_stage === "PRODUCTION" ? "green" : d.current_stage?.includes("CANARY") ? "blue" : "slate"
                } />
              </div>
            ))}
          </div>
        </Panel>

        {/* Detail */}
        <Panel title={selected ? `部署详情` : "选择一条部署记录"} action={
          selected && detail?.deployment?.status === "RUNNING" ? <Btn danger onClick={() => rollback(selected)}>回滚</Btn> : undefined
        }>
          {busy ? <div className="flex justify-center py-12"><Spinner /></div> :
           !detail ? <Empty text="← 从左侧列表选择部署记录查看详情" /> : (
            <div className="space-y-4">
              {/* Key info */}
              <div className="grid grid-cols-3 gap-3">
                {[
                  ["模型", detail.deployment.model_id],
                  ["Champion", detail.deployment.champion_version ?? "-"],
                  ["Challenger", detail.deployment.candidate_version ?? "-"],
                  ["当前阶段", detail.deployment.current_stage ?? "-"],
                  ["决策", detail.deployment.decision ?? "-"],
                  ["状态", detail.deployment.status ?? "-"],
                ].map(([label, value]) => (
                  <div key={label} className="rounded-lg bg-slate-50 px-3 py-2">
                    <div className="text-[10px] text-slate-400 uppercase">{label}</div>
                    <div className="text-sm font-semibold text-slate-800 mt-0.5">{formatValue(value)}</div>
                  </div>
                ))}
              </div>

              {/* Stage timeline */}
              <div>
                <div className="text-xs font-semibold text-slate-500 mb-2">阶段历史</div>
                <div className="space-y-1">
                  {detail.stages.map((s, i) => {
                    const h = (s.health_json ?? {}) as Record<string,unknown>;
                    const gk = (h.gatekeeper_decision ?? {}) as Record<string,unknown>;
                    const reasons = (gk.decision_reasons ?? []) as string[];
                    return (
                      <details key={s.stage_record_id} className="rounded-lg border border-slate-200 bg-white">
                        <summary className="flex items-center gap-3 px-4 py-2.5 cursor-pointer hover:bg-slate-50">
                          <StatusDot status={s.decision === "PROMOTE" ? "SUCCEEDED" : s.decision === "ROLLBACK" ? "FAILED" : s.decision === "HOLD" ? "HOLD" : "RUNNING"} />
                          <span className="text-sm font-semibold text-slate-700">{s.stage?.replace(/_/g, " ")}</span>
                          <Badge label={s.decision} color={
                            s.decision === "PROMOTE" ? "green" : s.decision === "ROLLBACK" ? "red" : s.decision === "HOLD" ? "amber" : "blue"
                          } />
                          <span className="text-xs text-slate-400 ml-auto">{fmtTs(s.created_at)}</span>
                        </summary>
                        <div className="px-4 pb-4 space-y-2 text-sm">
                          {reasons.length > 0 && (
                            <div>
                              <div className="text-xs font-semibold text-slate-500 mb-1">Gatekeeper 决策理由</div>
                              {reasons.map((r, j) => <div key={j} className="text-xs text-slate-600 font-mono bg-slate-50 rounded px-2 py-1 mb-0.5">{r}</div>)}
                            </div>
                          )}
                          {(() => {
                            const code = gk.selected_strategy_code;
                            if (!code) return null;
                            return (
                              <div>
                                <div className="text-xs font-semibold text-slate-500 mb-1">KG 推荐策略</div>
                                <Badge label={String(code)} color="blue" />
                              </div>
                            );
                          })()}
                          {(() => {
                            const alerts = h.deployment_alerts;
                            if (!alerts || !Array.isArray(alerts) || alerts.length === 0) return null;
                            return (
                              <div>
                                <div className="text-xs font-semibold text-slate-500 mb-1">部署告警 ({alerts.length})</div>
                                {alerts.map((a: Record<string,unknown>, j: number) => (
                                  <div key={j} className="text-xs text-slate-600 font-mono bg-red-50 rounded px-2 py-1 mb-0.5">
                                    {String(a.alert_code)} ({(a.severity as string) ?? "WARNING"})
                                  </div>
                                ))}
                              </div>
                            );
                          })()}
                        </div>
                      </details>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
