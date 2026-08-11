"use client";

import type { FormEvent, ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";
import Shell from "./components/shell";
import Sidebar from "./components/sidebar";
import DashboardOverview from "./components/dashboard-overview";
import DeploymentPanel from "./components/deployment-panel";
import DeploymentKgDecisionPanel from "./components/deployment-kg-decision-panel";
import TaskFourPanel from "./components/task-four-panel";
import KgCalibrationPanel from "./kg-calibration-panel";
import MonitoringPanel from "./components/monitoring-panel";
import { requestJson, Panel, StatTile, StatusDot, Badge, formatValue, Btn, Spinner, Empty } from "./components/shared";

/* ── Types ── */
type Items<T> = { items: T[] };
type LifecycleState = {
  lifecycle_run_id?: string; model_id?: string; champion_version?: string; current_phase?: string;
  monitoring_run_id?: string | null; diagnosis_run_id?: string | null; event_id?: string | null;
  agent_decision_id?: string | null; decision_proposal_id?: string | null; manual_review_id?: string | null;
  training_plan_id?: string | null; training_job_id?: string | null; experiment_id?: string | null;
  qualification_run_id?: string | null; deployment_id?: string | null;
  has_alerts?: boolean | null; alert_count?: number | null; max_alert_severity?: string | null;
  trigger_diagnosis?: boolean | null; decay_degree?: string | null; status_7d?: string | null; status_30d?: string | null;
  diagnosis_status?: string | null; persistence_judgment?: Record<string, unknown> | null;
  recommended_action?: string | null; need_iteration?: boolean | null; requires_manual_review?: boolean;
  agent_confidence?: number | null; primary_root_cause_code?: string | null; primary_root_cause_score?: number | null;
  challenger_version?: string | null; challenger_qualified?: boolean | null; business_round?: number | null;
  iteration_exit_reason?: string | null; training_callback_status?: string | null;
  training_dispatched?: boolean | null; training_dispatch_mode?: string | null;
  deployment_stage?: string | null; deployment_decision?: string | null;
  last_error?: Record<string,unknown> | null; [key: string]: unknown;
};
type LifecycleRun = { lifecycle_run_id: string; current_phase?: string; state?: LifecycleState };
type TrainingPlanDetail = {
  training_plan_id?: string; model_id?: string; algorithm?: string; status?: string; risk_level?: string;
  root_cause_code?: string; champion_version?: string; frozen_champion_version?: string; rollback_target?: string;
  strategy_code?: string; strategy_parameters?: Record<string,unknown>; random_seed?: number;
  business_round?: number; max_business_rounds?: number; experiment_id?: string; iteration_run_id?: string;
  proposal_id?: string; approval_id?: string; diagnosis_run_id?: string; preprocessing_version?: string;
  feature_schema_version?: string; label_versions?: string[]; data_snapshot_ids?: string[];
  data_eligibility_assessment_ids?: string[]; target_metric_codes?: string[]; qualification_rule_version?: string;
  windows?: { baseline_window_id?: string; training_window_ids?: string[]; validation_window_ids?: string[]; oot_window_id?: string; oot_locked?: boolean };
  blocking_reasons?: string[];
};
type DeploymentStageRecord = {
  deployment_id?: string;
  stage?: string;
  decision?: string;
  status?: string;
  health_json?: Record<string,unknown> | string | null;
  result_json?: Record<string,unknown> | string | null;
  created_at?: string;
};
type DeploymentStagesResponse = { deployment_id: string; stages: DeploymentStageRecord[] };
type FeatureTransformItem = { operation: string; source_feature: string; target_feature?: string | null; reason: string; parameters?: Record<string,unknown> };
type FeatureReconstructionPlan = { plan_id: string; target_schema_version: string; transforms: FeatureTransformItem[]; expected_feature_count_before: number; expected_feature_count_after: number };
type DecisionProposalDetail = {
  proposal_id?: string;
  primary_root_cause_code?: string;
  action?: string;
  strategies?: Array<{
    strategy_code?: string;
    parameters?: Record<string, unknown>;
    rationale?: string;
  }>;
  selected_strategy_code?: string | null;
  decision_reasons?: string[];
  requires_manual_review?: boolean;
  confidence?: string;
};

/* ── Constants ── */
const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_MODEL_OPS_API_BASE ?? "http://localhost:8001";
const lifecycleSteps = [
  ["monitoring_run_id","监控"],["diagnosis_run_id","诊断"],["event_id","事件"],["agent_decision_id","Agent"],
  ["decision_proposal_id","决策"],["manual_review_id","复核"],["training_plan_id","训练计划"],
  ["training_job_id","训练任务"],["qualification_run_id","资格验证"],["deployment_id","部署"],
] as const;
const deploymentStages = ["OFFLINE_VALIDATION","OOT_GATE","SHADOW","CANARY_5","CANARY_20","CANARY_50","PRODUCTION"];
const deploymentStageMeta: Record<string,{label:string;desc:string}> = {
  OFFLINE_VALIDATION:{label:"离线验证",desc:"离线数据检查候选模型"},OOT_GATE:{label:"跨期验证",desc:"未来窗口验证稳健性"},
  SHADOW:{label:"影子部署",desc:"旁路打分，不影响业务"},CANARY_5:{label:"灰度 5%",desc:"5% 流量试用"},
  CANARY_20:{label:"灰度 20%",desc:"20% 流量观察"},CANARY_50:{label:"灰度 50%",desc:"半量验证"},
  PRODUCTION:{label:"全量生产",desc:"候选模型成为生产版本"},
};
const terminalPhases = new Set(["EVENT_CLOSED","NO_ALERT","FAILED","COMPLETED","PROMOTED","ROLLED_BACK"]);
const phaseMeta: Record<string,{label:string;desc:string}> = {
  NO_RUN:{label:"尚未启动",desc:"还没有创建生命周期"},CREATED:{label:"已创建",desc:"准备进入监控"},
  MONITORING:{label:"监控中",desc:"检查模型指标和漂移"},MONITORING_COMPLETED:{label:"监控完成",desc:"监控完成"},
  NO_ALERT:{label:"无告警关闭",desc:"无异常，流程结束"},DIAGNOSING:{label:"诊断中",desc:"分析异常根因"},
  DIAGNOSIS_COMPLETED:{label:"诊断完成",desc:"已得根因和建议"},WAITING_AGENT_DECISION:{label:"等待Agent决策",desc:"诊断结果交Agent"},
  AGENT_DECIDING:{label:"Agent决策中",desc:"判断处理方向"},DECISION_PROPOSED:{label:"等待复核",desc:"需人工确认"},
  MANUAL_REVIEW:{label:"人工复核中",desc:"流程暂停等待复核"},ITERATING:{label:"迭代处理中",desc:"生成修复/训练计划"},
  WAITING_TRAINING_CALLBACK:{label:"等待训练回调",desc:"Worker训练中"},CHALLENGER_TRAINED:{label:"候选已训练",desc:"准备验证"},
  OFFLINE_VALIDATING:{label:"离线验证中",desc:"资格验证和质量门禁"},QUALIFICATION_COMPLETED:{label:"资格验证完成",desc:"完成上线前检查"},
  CANARY_RUNNING:{label:"灰度部署中",desc:"逐步推进"},PROMOTED:{label:"已提升生产",desc:"候选成为生产"},
  ROLLED_BACK:{label:"已回滚",desc:"部署失败已回滚"},EVENT_CLOSED:{label:"事件已关闭",desc:"闭环完成"},
  FAILED:{label:"流程失败",desc:"无法自动处理"},
};
phaseMeta.NO_ALERT = { label: "未触发诊断", desc: "B1 判定无需进入诊断，流程结束" };
const actionMeta: Record<string,{label:string;desc:string}> = {
  NO_ACTION:{label:"无需处理",desc:"不用修复"},CONTINUE_OBSERVATION:{label:"继续观察",desc:"继续监控"},
  DATA_REPAIR:{label:"数据修复",desc:"数据异常需修复"},PIPELINE_REPAIR:{label:"管道修复",desc:"ETL异常"},
  CALIBRATION_ADJUSTMENT:{label:"校准调整",desc:"重新校准概率"},THRESHOLD_ADJUSTMENT:{label:"阈值调整",desc:"重新搜索阈值"},
  MODEL_ITERATION:{label:"模型迭代",desc:"训练challenger"},MANUAL_REVIEW:{label:"人工判断",desc:"需人工定夺"},
};
function describePhase(v?: string|null) { const k = v||"NO_RUN"; return phaseMeta[k]??{label:k,desc:""}; }
function describeAction(v?: string|null) { const k = v||""; return actionMeta[k]??{label:k||"暂无",desc:""}; }
function formatRootCause(v?: string) { const m: Record<string,string>={FEATURE_DRIFT:"特征漂移",LABEL_DRIFT:"标签漂移",DATA_QUALITY:"数据质量异常",PERFORMANCE_DROP:"模型效果下降"}; return v?`${m[v]||v} (${v})`:"-"; }
function formatRiskLevel(v?: string) { const m: Record<string,string>={LOW:"低风险",MEDIUM:"中风险",HIGH:"高风险"}; return v?`${m[v]||v} (${v})`:"-"; }
function formatDeploymentDecision(v?: unknown) { const m: Record<string,string>={PROMOTE:"提升生产",ADVANCE_STAGE:"进入下一阶段",HOLD:"暂停观察",ROLLBACK:"回滚",ABORT_DEPLOYMENT:"终止部署"}; return m[String(v||"")]||formatValue(v); }
function joinValues(v?: Array<string|number|boolean>|null) { return v&&v.length>0?v.map(x=>formatValue(x)).join("、"):"-"; }

/* ── Page ── */
export default function Page() {
  const [activeNav, setActiveNav] = useState("overview");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [apiBase, setApiBase] = useState(DEFAULT_API_BASE);

  // ── All existing state (unchanged) ──
  const [busy, setBusy] = useState<string|null>(null);
  const [message, setMessage] = useState<{type:"ok"|"error";text:string}|null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [showManualCallback, setShowManualCallback] = useState(false);
  const [modelId, setModelId] = useState("credit_model_001");
  const [championVersion, setChampionVersion] = useState("champion_v1");
  const [triggerType, setTriggerType] = useState("SCHEDULED_TRIGGER");
  const [runId, setRunId] = useState("");
  const [reviewerId, setReviewerId] = useState("admin");
  const [reviewReason, setReviewReason] = useState("确认本次修复方向，允许进入真实训练链路。");
  const [callbackStatus, setCallbackStatus] = useState("SUCCEEDED");
  const [candidateVersion, setCandidateVersion] = useState("v1_challenger_manual");
  const [experimentId, setExperimentId] = useState("");
  const [lifecycleRun, setLifecycleRun] = useState<LifecycleRun|null>(null);
  const [trainingPlan, setTrainingPlan] = useState<TrainingPlanDetail|null>(null);
  const [deploymentStageRecords, setDeploymentStageRecords] = useState<DeploymentStageRecord[]>([]);
  const [featureReconPlan, setFeatureReconPlan] = useState<FeatureReconstructionPlan | null>(null);

  const state = useMemo(() => lifecycleRun?.state??{}, [lifecycleRun]);
  const currentRunId = runId || lifecycleRun?.lifecycle_run_id || "";
  const currentPhase = lifecycleRun?.current_phase || String(state.current_phase||"NO_RUN");
  const currentPhaseMeta = describePhase(currentPhase);
  const currentAction = String(state.recommended_action||"");
  const currentActionMeta = describeAction(currentAction);
  const currentTrainingPlanId = String(state.training_plan_id||"");
  const trainingJobId = String(state.training_job_id||"");
  const decisionProposalId = String(state.decision_proposal_id||"");
  const currentExperimentId = experimentId || String(state.experiment_id||"");
  const businessRound = Number(state.business_round||1);
  const isTerminal = terminalPhases.has(String(currentPhase));
  const completedSteps = lifecycleSteps.filter(([key])=>Boolean(state[key])).length;
  const progress = Math.round((completedSteps/lifecycleSteps.length)*100);
  const deploymentIndex = Math.max(deploymentStages.findIndex(s=>s===state.deployment_stage), state.deployment_id?0:-1);

  // ── Effects (unchanged) ──
  useEffect(()=>{if(!autoRefresh||!currentRunId||isTerminal)return;const t=setInterval(()=>{loadLifecycleRun(currentRunId,false);},3000);return()=>clearInterval(t);},[autoRefresh,currentRunId,isTerminal]);
  useEffect(()=>{if(!currentTrainingPlanId){setTrainingPlan(null);return;}let c=false;(async()=>{try{const d=await requestJson<TrainingPlanDetail>(apiBase,`/api/iteration/plans/${currentTrainingPlanId}`);if(!c)setTrainingPlan(d);}catch{if(!c)setTrainingPlan(null);}})();return()=>{c=true;};},[apiBase,currentTrainingPlanId]);
  useEffect(()=>{const deploymentId=String(state.deployment_id||"");if(!deploymentId){setDeploymentStageRecords([]);return;}let c=false;(async()=>{try{const d=await requestJson<DeploymentStagesResponse>(apiBase,`/api/iteration/deployments/${deploymentId}/stages`);if(!c)setDeploymentStageRecords(d.stages??[]);}catch{if(!c)setDeploymentStageRecords([]);}})();return()=>{c=true;};},[apiBase,state.deployment_id]);
  useEffect(()=>{const planId=String(state.feature_reconstruction_plan_id||"");if(!planId){setFeatureReconPlan(null);return;}let c=false;(async()=>{try{const d=await requestJson<FeatureReconstructionPlan>(apiBase,`/api/iteration/features/reconstruction-plans/${planId}`);if(!c)setFeatureReconPlan(d);}catch{if(!c)setFeatureReconPlan(null);}})();return()=>{c=true;};},[apiBase,state.feature_reconstruction_plan_id]);

  // ── Handlers (unchanged) ──
  async function runAction<T>(key:string,action:()=>Promise<T>,ok:string,show=true){setBusy(key);if(show)setMessage(null);try{const r=await action();if(show)setMessage({type:"ok",text:ok});return r;}catch(e){setMessage({type:"error",text:e instanceof Error?e.message:"请求失败"});return null;}finally{setBusy(null);}}
  async function testApi(){await runAction("health",()=>requestJson(apiBase,"/health/live"),"后端连接正常。");}
  async function loadLifecycleRun(id=currentRunId,show=true){if(!id){setMessage({type:"error",text:"请先输入 lifecycle_run_id"});return null;}const d=await runAction("load-run",()=>requestJson<LifecycleRun>(apiBase,`/api/lifecycle-runs/${id}`),"生命周期已刷新。",show);if(d){setLifecycleRun(d);setRunId(d.lifecycle_run_id);}return d;}
  async function startLifecycle(e?:FormEvent<HTMLFormElement>){e?.preventDefault();const d=await runAction("start",()=>requestJson<LifecycleRun>(apiBase,"/api/lifecycle-runs?wait=false",{method:"POST",body:JSON.stringify({model_id:modelId,champion_version:championVersion,trigger_type:triggerType})}),"生命周期已启动，后台监控正在运行。");if(d){setLifecycleRun(d);setRunId(d.lifecycle_run_id);setTrainingPlan(null);setActiveNav("workflow");}}
  async function resumeLifecycle(payload:Record<string,unknown>,key:string){if(!currentRunId){setMessage({type:"error",text:"请先启动或加载生命周期"});return null;}const d=await runAction(key,()=>requestJson<LifecycleRun>(apiBase,`/api/lifecycle-runs/${currentRunId}/resume`,{method:"POST",body:JSON.stringify(payload)}),"生命周期已恢复。");if(d){setLifecycleRun(d);setRunId(d.lifecycle_run_id);}return d;}
  async function submitManualReview(decision:"APPROVE"|"REJECT"){if(!currentRunId||!decisionProposalId){setMessage({type:"error",text:"当前无复核建议"});return;}const ok=decision==="APPROVE";const reason=reviewReason.trim()||(ok?"人工确认通过。":"人工确认拒绝。");const report=await runAction(ok?"approve":"reject",()=>requestJson<{review_id:string}>(apiBase,`/api/iteration/decisions/${decisionProposalId}/reviews`,{method:"POST",body:JSON.stringify({proposal_id:decisionProposalId,reviewer_id:reviewerId.trim()||"admin",decision,reason,rejection_reason_codes:ok?[]:["MANUAL_REJECTED"],adjustment_instructions:ok?[]:["请重新生成修复建议"],forbidden_adjustments:[],expected_evidence:[],reviewed_at:new Date().toISOString()})}),ok?"复核已通过":"复核已拒绝");if(!report)return;await resumeLifecycle({decision:ok?"approved":"rejected",manual_review_id:report.review_id,review_id:report.review_id},ok?"approve":"reject");}
  async function submitTrainingCallback(){if(!currentRunId||!trainingJobId){setMessage({type:"error",text:"缺少训练任务ID"});return;}const st=callbackStatus.trim()||"SUCCEEDED";const cv=candidateVersion.trim()||String(state.challenger_version||`${state.champion_version||"champion"}_challenger_v1`);const cb=await runAction("callback",()=>requestJson<{callback_applied:boolean}>(apiBase,`/api/internal/iteration/jobs/${trainingJobId}/callback`,{method:"POST",body:JSON.stringify({training_job_id:trainingJobId,lifecycle_run_id:currentRunId,idempotency_key:`${String(state.iteration_run_id||"iter")}:round-${businessRound}:exp-${currentExperimentId}`,experiment_id:currentExperimentId,status:st,candidate_version:cv,model_artifact_uri:st==="SUCCEEDED"?`s3://riskitem/demo/models/${cv}`:undefined,training_metrics:{auc:0.81,ks:0.43},validation_metrics:{original_drop:0.04,recovered_amount:0.035,recovery_rate:0.875,champion_auc:0.74,challenger_auc:0.775,healthy_lower_bound:0.76,score_psi:0.08,train_valid_gap:0.015,discrimination_passed:true,calibration_passed:true,oot_passed:true},segment_metrics:{segment_governance_passed:true},artifact_checksums:{},environment_manifest:{runtime:"frontend-manual"},technical_retry_count:0})}),"手动回调已提交");if(cb)await loadLifecycleRun(currentRunId,false);}
  const primaryAction = (()=>{const p=currentPhase;if(!p||p==="NO_RUN")return"启动生命周期";if(p==="DECISION_PROPOSED"&&decisionProposalId)return"通过人工复核";if(p==="WAITING_FEATURE_RECONSTRUCTION")return"等待特征重构完成";if(p==="WAITING_TRAINING_CALLBACK")return state.training_callback_status==="SUCCEEDED"?"训练已回调，刷新闭环状态":"Worker 训练中，必要时可手动兜底回调";if(p==="EVENT_CLOSED")return"流程已闭环";return"刷新状态";})();

  // ── Topbar ──
  const topbar = (
    <>
      <span className="inline-flex h-2 w-2 rounded-full bg-emerald-500 animate-pulse" title="后端在线" />
      <span className="text-xs font-semibold text-slate-500 mr-2">API</span>
      <input className="flex-1 max-w-md rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 font-mono text-xs focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition" value={apiBase} onChange={e=>setApiBase(e.target.value)} />
      <label className="flex items-center gap-2 text-xs text-slate-500 cursor-pointer select-none">
        <input type="checkbox" checked={autoRefresh} onChange={e=>setAutoRefresh(e.target.checked)} className="rounded" /> 自动刷新
      </label>
      <Btn onClick={testApi} disabled={busy==="health"}>测试后端</Btn>
      <a className="rounded-lg bg-slate-900 px-3 py-2 text-xs font-semibold text-white hover:bg-slate-800 transition" href={`${apiBase}/docs`} target="_blank">API 文档</a>
      {currentRunId && <Btn onClick={()=>{loadLifecycleRun();setActiveNav("workflow");}} disabled={busy==="load-run"}>刷新流程</Btn>}
    </>
  );

  // ── Sidebar ──
  const sidebar = <Sidebar active={activeNav} onNav={(k)=>{setActiveNav(k);setMessage(null);}} collapsed={sidebarCollapsed} onToggle={()=>setSidebarCollapsed(!sidebarCollapsed)} />;

  return (
    <Shell topbar={topbar} sidebar={sidebar} sidebarCollapsed={sidebarCollapsed}>
      {message && (
        <div className={`fixed top-14 right-5 z-50 rounded-xl border px-4 py-3 text-sm font-medium shadow-lg animate-fade-up ${message.type==="ok"?"border-emerald-200 bg-emerald-50 text-emerald-700":"border-red-200 bg-red-50 text-red-700"}`}>
          {message.text}
          <button className="ml-3 text-xs opacity-50 hover:opacity-100" onClick={()=>setMessage(null)}>✕</button>
        </div>
      )}

      {activeNav==="overview" && <DashboardOverview apiBase={apiBase} onNav={setActiveNav} />}
      {activeNav==="deployment" && <DeploymentPanel apiBase={apiBase} />}
      {activeNav==="task4" && <TaskFourPanel apiBase={apiBase} initialModelId={modelId} />}
      {activeNav==="kg" && <KgCalibrationPanel apiBase={apiBase} />}

      {activeNav==="workflow" && (
        <div className="space-y-5 p-5">
          <div><p className="text-[11px] font-semibold uppercase tracking-[.16em] text-indigo-600">流程控制</p><h1 className="text-2xl font-bold tracking-tight text-slate-900 mt-1">生命周期管理</h1></div>
          {(Array.isArray(state.warnings) && (state.warnings as string[]).length > 0) && (
            <div className="rounded-xl border-2 border-amber-300 bg-amber-50 p-4 shadow-lg animate-fade-up">
              <div className="flex items-start gap-3">
                <span className="text-2xl">&#9888;</span>
                <div className="flex-1">
                  <div className="text-sm font-bold text-amber-800 mb-2">流程告警 — 以下事项需要关注，可能影响迭代质量</div>
                  {(state.warnings as string[]).map((w: string, i: number) => (
                    <div key={i} className="text-xs text-amber-700 font-mono bg-amber-100 rounded px-2 py-1 mb-1">{w}</div>
                  ))}
                </div>
              </div>
            </div>
          )}
          <div className="grid gap-5 lg:grid-cols-[380px_1fr]">
            {/* Left column */}
            <div className="space-y-4">
              <Panel title="一键流程">
                <form className="space-y-3" onSubmit={startLifecycle}>
                  <Input label="模型 ID" value={modelId} onChange={setModelId} />
                  <Input label="Champion 版本" value={championVersion} onChange={setChampionVersion} />
                  <Select label="触发类型" value={triggerType} onChange={setTriggerType} options={["SCHEDULED_TRIGGER","THRESHOLD_TRIGGER","ABNORMAL_TRIGGER","MANUAL_TRIGGER"]} />
                  <Btn primary disabled={busy==="start"}>{currentRunId?"启动新的生命周期":"启动生命周期"}</Btn>
                </form>
                <div className="mt-4 rounded-lg border border-indigo-100 bg-indigo-50 p-3 text-sm text-indigo-800">推荐动作：{primaryAction}</div>
              </Panel>
              <Panel title="智能操作">
                <div className="space-y-3">
                  <Btn primary disabled={!decisionProposalId||!!busy} onClick={()=>submitManualReview("APPROVE")}>通过复核并启动训练</Btn>
                  <Btn disabled={!decisionProposalId||!!busy} onClick={()=>submitManualReview("REJECT")}>拒绝建议</Btn>
                  <Input label="复核人" value={reviewerId} onChange={setReviewerId} />
                  <Input label="复核意见" value={reviewReason} onChange={setReviewReason} />
                  <ReadOnly label="复核 ID" value={String(state.manual_review_id||"")} />
                  <ReadOnly label="决策建议 ID" value={decisionProposalId} />
                </div>
              </Panel>
              <Panel title="真实 Worker">
                <div className="space-y-2 text-sm">
                  <StatusLine label="派发模式" value={formatValue(state.training_dispatch_mode)} />
                  <StatusLine label="训练任务" value={trainingJobId} mono />
                  <StatusLine label="实验 ID" value={String(state.experiment_id||"")} mono />
                  <StatusLine label="回调状态" value={formatValue(state.training_callback_status)} />
                  <Btn onClick={()=>setShowManualCallback(v=>!v)}>{showManualCallback?"收起手动回调":"手动兜底回调"}</Btn>
                  {showManualCallback&&(
                    <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 space-y-3">
                      <Select label="回调状态" value={callbackStatus} onChange={setCallbackStatus} options={["SUCCEEDED","FAILED","CANCELLED"]} />
                      <Input label="候选版本" value={candidateVersion} onChange={setCandidateVersion} />
                      <Input label="实验 ID" value={currentExperimentId} onChange={setExperimentId} />
                      <Btn disabled={!trainingJobId||busy==="callback"} onClick={submitTrainingCallback}>提交手动回调</Btn>
                    </div>
                  )}
                </div>
              </Panel>
            </div>
            {/* Right column */}
            <div className="space-y-4">
              <div className="grid gap-3 grid-cols-2 xl:grid-cols-4">
                <StatTile label="阶段" value={currentPhaseMeta.label} sub={currentPhaseMeta.desc} />
                <StatTile label="修复方向" value={currentActionMeta.label} sub={currentActionMeta.desc} />
                <StatTile label="业务轮次" value={formatValue(state.business_round)} sub="最多 3 轮自动迭代" />
                <StatTile label="Agent 置信度" value={formatValue(state.agent_confidence)} sub="决策把握程度" />
              </div>
              <div className="dash-card px-4 py-3 font-mono text-xs text-slate-500 break-all">lifecycle_run_id: {formatValue(currentRunId)}</div>

              {/* 诊断摘要 — 解释流程为何关闭或停滞 */}
              {(state.diagnosis_run_id || state.trigger_diagnosis) && (
                <Panel title="诊断摘要">
                  <div className="space-y-2 text-sm">
                    <StatusLine label="根因" value={formatRootCause(state.primary_root_cause_code??undefined)} />
                    <StatusLine label="建议动作" value={formatValue(state.recommended_action)} />
                    <StatusLine label="触发诊断" value={state.trigger_diagnosis ? "是" : "否"} />
                    <StatusLine label="衰减程度" value={formatValue(state.decay_degree)} />
                    <StatusLine label="7D 状态" value={formatValue(state.status_7d)} />
                    <StatusLine label="30D 状态" value={formatValue(state.status_30d)} />
                    {state.primary_root_cause_code === "uncertain" && (
                      <div className="mt-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-800">
                        诊断无法确定根因（知识图谱中缺少对应指标的根因映射）。
                        流程已关闭，不会自动进入迭代或训练。
                      </div>
                    )}
                    {state.current_phase === "EVENT_CLOSED" && !state.decision_proposal_id && (
                      <div className="mt-2 rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs text-slate-600">
                        事件已闭环：未生成修复决策，不会触发模型迭代或部署。
                      </div>
                    )}
                    {state.recommended_action === "MANUAL_REVIEW" && !state.decision_proposal_id && state.current_phase !== "EVENT_CLOSED" && (
                      <div className="mt-2 rounded-lg border border-indigo-200 bg-indigo-50 p-3 text-xs text-indigo-700">
                        需要人工复核，请点击下方 刷新状态 按钮推进流程。
                      </div>
                    )}
                  </div>
                </Panel>
              )}

              <Panel title="流程进度">
                <div className="phase-timeline">
                  {lifecycleSteps.map(([key,label],i)=>(
                    <div key={key} className="phase-step">
                      {i<lifecycleSteps.length-1&&<div className={`ph-line${Boolean(state[key])?" done":""}`} />}
                      <div className={`ph-dot${Boolean(state[key])?" done":" active"}`} />
                      <div className="ph-label">{label}</div>
                    </div>
                  ))}
                </div>
                <ProgressBar value={Math.max(progress,currentRunId?5:0)} />
              </Panel>

              <KgDecisionCardV2 state={state} proposalId={decisionProposalId||undefined} apiBase={apiBase} />
              <DeploymentKgDecisionPanel records={deploymentStageRecords} state={state} />

              <Panel title="部署进度">
                <div className="phase-timeline">
                  {deploymentStages.map((stage,i)=>(
                    <div key={stage} className="phase-step">
                      {i<deploymentStages.length-1&&<div className={`ph-line${i<=deploymentIndex?" done":""}`} />}
                      <div className={`ph-dot${i<=deploymentIndex?" done":" active"}`} />
                      <div className="ph-label">{deploymentStageMeta[stage]?.label||stage}</div>
                    </div>
                  ))}
                </div>
                <div className="grid gap-3 grid-cols-3 mt-4">
                  <StatTile label="部署 ID" value={formatValue(state.deployment_id)} />
                  <StatTile label="部署阶段" value={state.deployment_stage?deploymentStageMeta[String(state.deployment_stage)]?.label||formatValue(state.deployment_stage):"-"} sub={state.deployment_stage?deploymentStageMeta[String(state.deployment_stage)]?.desc:""} />
                  <StatTile label="部署决策" value={formatDeploymentDecision(state.deployment_decision)} sub={String(state.deployment_decision||"")} />
                </div>
              </Panel>

              <Panel title="关键结果">
                <div className="grid gap-3 grid-cols-2 xl:grid-cols-4">
                  {[["训练计划",state.training_plan_id],["候选版本",state.challenger_version],["资格验证",state.qualification_run_id],["是否合格",state.challenger_qualified]].map(([l,v])=>(
                    <div key={String(l)} className="rounded-lg bg-slate-50 px-3 py-2"><div className="text-[10px] text-slate-400">{l}</div><div className="text-sm font-semibold text-slate-800 mt-0.5">{formatValue(v)}</div></div>
                  ))}
                </div>
                {currentTrainingPlanId && <TrainingPlanDetail plan={trainingPlan} id={currentTrainingPlanId} />}
              </Panel>

              {featureReconPlan && featureReconPlan.transforms.length > 0 && (
                <FeatureReconstructionView plan={featureReconPlan} />
              )}
            </div>
          </div>
        </div>
      )}

      {activeNav==="monitoring" && <MonitoringPanel apiBase={apiBase} />}

      {activeNav==="state" && (
        <div className="space-y-5 p-5">
          <div><p className="text-[11px] font-semibold uppercase tracking-[.16em] text-indigo-600">系统状态</p><h1 className="text-2xl font-bold tracking-tight text-slate-900 mt-1">完整生命周期 State</h1></div>
          <div className="grid gap-4 grid-cols-1 lg:grid-cols-2">
            {["monitoring","diagnosis","iteration","training","deployment"].map(section=>{
              const keys=Object.keys(state).filter(k=>{const v=String(state[k]||"");return v&&v!=="null"&&v!=="None";});
              const sectionKeys=keys.filter(k=>{
                if(section==="monitoring")return/monitoring|alert|metric/i.test(k);
                if(section==="diagnosis")return/diagnosis|root_cause|drift|event/i.test(k);
                if(section==="iteration")return/decision|proposal|review|agent|plan/i.test(k);
                if(section==="training")return/training|experiment|challenger|qualif/i.test(k);
                if(section==="deployment")return/deploy|gatekeeper/i.test(k);
                return false;
              });
              if(sectionKeys.length===0)return null;
              return (
                <Panel key={section} title={section.charAt(0).toUpperCase()+section.slice(1)}>
                  <div className="space-y-1">
                    {sectionKeys.map(k=>(
                      <div key={k} className="flex items-start gap-3 rounded px-2 py-1.5 hover:bg-slate-50">
                        <span className="text-xs font-mono text-indigo-500 font-semibold shrink-0">{k}</span>
                        <span className="text-xs font-mono text-slate-600 break-all">{formatValue(state[k])}</span>
                      </div>
                    ))}
                  </div>
                </Panel>
              );
            })}
          </div>
          <details>
            <summary className="text-sm text-slate-500 cursor-pointer hover:text-slate-700">完整 JSON</summary>
            <pre className="mt-3 max-h-[500px] overflow-auto rounded-xl bg-slate-950 p-4 text-xs leading-relaxed text-slate-100">{JSON.stringify(state,null,2)}</pre>
          </details>
        </div>
      )}
    </Shell>
  );
}

/* ── Inline UI Components ── */

function Input({label,value,placeholder,onChange}:{label:string;value:string;placeholder?:string;onChange:(v:string)=>void}){return(<label className="grid gap-1 text-sm"><span className="text-xs font-medium text-slate-500">{label}</span><input className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition" placeholder={placeholder} value={value} onChange={e=>onChange(e.target.value)}/></label>);}
function ReadOnly({label,value}:{label:string;value:string}){return(<label className="grid gap-1 text-sm"><span className="text-xs font-medium text-slate-500">{label}</span><input className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 font-mono text-xs" value={value} readOnly placeholder="由流程生成"/></label>);}
function Select({label,value,onChange,options}:{label:string;value:string;onChange:(v:string)=>void;options:string[]}){return(<label className="grid gap-1 text-sm"><span className="text-xs font-medium text-slate-500">{label}</span><select className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm focus:outline-none focus:border-indigo-400" value={value} onChange={e=>onChange(e.target.value)}>{options.map(o=><option key={o}>{o}</option>)}</select></label>);}
function StatusLine({label,value,mono}:{label:string;value:string;mono?:boolean}){return(<div className="grid grid-cols-[90px_1fr] gap-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm"><span className="text-slate-500">{label}</span><span className={mono?"break-all font-mono text-xs text-slate-700":"text-slate-800"}>{value||"-"}</span></div>);}

function ProgressBar({value}:{value:number}){return(<div className="mt-4"><div className="mb-1 flex justify-between text-xs text-slate-400"><span>完成度</span><span>{value}%</span></div><div className="h-2 rounded-full bg-slate-100 overflow-hidden"><div className="h-full rounded-full bg-indigo-500 transition-all duration-500" style={{width:`${Math.min(value,100)}%`}}/></div></div>);}

function KgDecisionCardV2({state,proposalId,apiBase}:{state:Record<string,unknown>;proposalId?:string;apiBase:string}) {
  const [proposal, setProposal] = useState<DecisionProposalDetail | null>(null);
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    if (!proposalId) {
      setProposal(null);
      setLoadError("");
      return;
    }
    let cancelled = false;
    requestJson<DecisionProposalDetail>(apiBase, `/api/iteration/decisions/${proposalId}`)
      .then((data) => {
        if (!cancelled) {
          setProposal(data);
          setLoadError("");
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setProposal(null);
          setLoadError(error instanceof Error ? error.message : "读取决策建议失败");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [apiBase, proposalId]);

  const stateReasons = Array.isArray(state.decision_reasons) ? state.decision_reasons as string[] : [];
  const reasons = proposal?.decision_reasons?.length ? proposal.decision_reasons : stateReasons;
  const strategy = String(proposal?.selected_strategy_code || state.selected_strategy_code || proposal?.strategies?.[0]?.strategy_code || "");
  const kgStrategy = reasons.find((r) => r.startsWith("KG_STRATEGY:"))?.replace("KG_STRATEGY:", "") ?? "";
  const kgEffectiveness = reasons.find((r) => r.startsWith("HISTORICAL_EFFECTIVENESS:"))?.replace("HISTORICAL_EFFECTIVENESS:", "") ?? "";
  const kgCases = reasons.find((r) => r.startsWith("SUPPORT_CASES:"))?.replace("SUPPORT_CASES:", "") ?? "";
  const kgRelation = reasons.find((r) => r.startsWith("RELATION:"))?.replace("RELATION:", "") ?? "";
  const kgDegraded = reasons.includes("KG_RETRIEVAL_DEGRADED");
  const isYamlFallback = reasons.some((r) => r.startsWith("ROOT_CAUSE_RULE_MATCHED:"));
  const hasKG = Boolean(kgStrategy);
  const autoPassed = !(proposal?.requires_manual_review ?? state.requires_manual_review) && hasKG;
  const proposalUrl = proposalId ? `${apiBase.replace(/\/+$/,"")}/api/iteration/decisions/${proposalId}` : "";

  return (
    <div className="dash-card">
      <div className="dash-card-header flex items-center justify-between">
        <span>任务三策略决策</span>
        {proposalId ? <a href={proposalUrl} target="_blank" className="text-xs text-indigo-500 hover:underline">查看 Proposal</a> : null}
      </div>
      <div className="dash-card-body">
        {!proposalId ? (
          <p className="text-xs text-slate-400 py-4 text-center">启动生命周期并生成 decision_proposal_id 后显示策略决策。</p>
        ) : loadError ? (
          <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-700">
            已生成决策建议，但详情读取失败：{loadError}
          </div>
        ) : isYamlFallback ? (
          <div className="space-y-3">
            <div className="rounded-lg bg-slate-50 border border-slate-100 p-3 text-xs text-slate-600">
              <div className="font-semibold text-slate-800">当前没有命中 KG 策略，已使用本地规则兜底</div>
              <div className="mt-1">根因：{formatValue(proposal?.primary_root_cause_code || state.primary_root_cause_code)}</div>
              <div>采用策略：{strategy || "-"}</div>
              <div>动作：{formatValue(proposal?.action || state.recommended_action)}</div>
            </div>
            {proposal?.strategies?.[0]?.rationale ? (
              <div className="rounded-lg border border-slate-200 bg-white p-3 text-xs text-slate-600">
                策略说明：{proposal.strategies[0].rationale}
              </div>
            ) : null}
          </div>
        ) : kgDegraded ? (
          <div className="rounded-lg bg-amber-50 border border-amber-100 p-3 text-xs text-amber-700">
            KG 检索降级：Neo4j 不可用或查询失败，流程会转人工复核或规则兜底。
          </div>
        ) : hasKG ? (
          <div className="space-y-2">
            <div className="rounded-lg bg-indigo-50 border border-indigo-100 p-3">
              <div className="text-[10px] text-indigo-400 font-mono mb-1">MATCH (RootCause)-[:RECOMMENDS]-&gt;(Strategy)-[:MITIGATES]-&gt;(RootCause)</div>
              <div className="text-xs text-indigo-600 font-mono">RootCause: {kgRelation.split("|")[0] || formatValue(proposal?.primary_root_cause_code || state.primary_root_cause_code)}</div>
              <div className="grid grid-cols-2 gap-2 mt-2">
                {[
                  ["策略", kgStrategy],
                  ["历史效果", kgEffectiveness ? String(Number(kgEffectiveness).toFixed(3)) : "?"],
                  ["支持案例", kgCases || "?"],
                  ["决策方式", autoPassed ? "自动通过" : "人工复核"],
                ].map(([label, value]) => (
                  <div key={label} className="bg-white/60 rounded p-2">
                    <div className="text-[10px] text-indigo-400">{label}</div>
                    <div className="text-sm font-bold text-indigo-800">{value}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <p className="text-xs text-slate-400 py-4 text-center">决策建议正在读取；如果长期为空，请检查 proposal 接口和 Neo4j 状态。</p>
        )}
        {reasons.length > 0 && (
          <details className="mt-3">
            <summary className="text-[10px] text-slate-400 cursor-pointer hover:text-slate-600">全部决策理由 ({reasons.length})</summary>
            <div className="mt-1 space-y-0.5 max-h-32 overflow-auto">
              {reasons.map((reason, index) => (
                <div key={index} className="text-[10px] font-mono text-slate-500 bg-slate-50 rounded px-1.5 py-0.5">{reason}</div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}

function KgDecisionCard({state,proposalId}:{state:Record<string,unknown>;proposalId?:string}){const reasons: string[]=Array.isArray(state.decision_reasons)?state.decision_reasons as string[]:[];const strategy=(state.selected_strategy_code??"")as string;const kgStrategy=reasons.find(r=>r.startsWith("KG_STRATEGY:"))?.replace("KG_STRATEGY:","")??"";const kgEffectiveness=reasons.find(r=>r.startsWith("HISTORICAL_EFFECTIVENESS:"))?.replace("HISTORICAL_EFFECTIVENESS:","")??"";const kgCases=reasons.find(r=>r.startsWith("SUPPORT_CASES:"))?.replace("SUPPORT_CASES:","")??"";const kgRelation=reasons.find(r=>r.startsWith("RELATION:"))?.replace("RELATION:","")??"";const kgDegraded=reasons.includes("KG_RETRIEVAL_DEGRADED");const isYamlFallback=reasons.some(r=>r.startsWith("ROOT_CAUSE_RULE_MATCHED:"));const hasKG=Boolean(kgStrategy);const autoPassed=!state.requires_manual_review&&hasKG;return(<div className="dash-card"><div className="dash-card-header flex items-center justify-between">KG Strategy Decision{proposalId?<a href={`${DEFAULT_API_BASE}/api/iteration/decisions/${proposalId}`} target="_blank" className="text-xs text-indigo-500 hover:underline">Proposal</a>:null}</div><div className="dash-card-body">{isYamlFallback?<div className="rounded-lg bg-slate-50 border border-slate-100 p-3 text-xs text-slate-600">YAML 规则匹配 (无 KG 命中)<br/>策略: {strategy||"—"}</div>:kgDegraded?<div className="rounded-lg bg-amber-50 border border-amber-100 p-3 text-xs text-amber-700">KG 检索降级 — Neo4j 不可用，需人工复核</div>:hasKG?<div className="space-y-2"><div className="rounded-lg bg-indigo-50 border border-indigo-100 p-3"><div className="text-[10px] text-indigo-400 font-mono mb-1">MATCH (rc)→[:RECOMMENDS]→(s)→[:MITIGATES]→(rc)</div><div className="text-xs text-indigo-600 font-mono">RootCause: {kgRelation.split("|")[0]||"?"}</div><div className="grid grid-cols-2 gap-2 mt-2">{[["Strategy",kgStrategy],["Effectiveness",kgEffectiveness?String(Number(kgEffectiveness).toFixed(3)):"?"],["Support Cases",kgCases||"?"],["Decision",autoPassed?"AUTO":"REVIEW"]].map(([l,v])=>(<div key={l} className="bg-white/60 rounded p-2"><div className="text-[10px] text-indigo-400">{l}</div><div className={`text-sm font-bold ${l==="Decision"?(autoPassed?"text-emerald-700":"text-amber-700"):"text-indigo-800"}`}>{v}</div></div>))}</div></div></div>:<p className="text-xs text-slate-400 py-4 text-center">暂无 KG 决策数据。启动 lifecycle 后显示。</p>}{reasons.length>0&&(<details className="mt-3"><summary className="text-[10px] text-slate-400 cursor-pointer hover:text-slate-600">全部理由 ({reasons.length})</summary><div className="mt-1 space-y-0.5 max-h-32 overflow-auto">{reasons.map((r,i)=><div key={i} className="text-[10px] font-mono text-slate-500 bg-slate-50 rounded px-1.5 py-0.5">{r}</div>)}</div></details>)}</div></div>);}

function FeatureReconstructionView({ plan }: { plan: FeatureReconstructionPlan }) {
  const opLabel: Record<string, string> = {
    DROP: "删除", LOG_TRANSFORM: "对数变换", INTERACTION: "特征交互",
    IMPUTE: "缺失插补", STANDARDIZE: "标准化",
  };
  const opBadge: Record<string, string> = {
    DROP: "text-red-600 bg-red-100",
    LOG_TRANSFORM: "text-sky-600 bg-sky-100",
    INTERACTION: "text-indigo-600 bg-indigo-100",
    IMPUTE: "text-amber-600 bg-amber-100",
    STANDARDIZE: "text-emerald-600 bg-emerald-100",
  };
  const core = plan.transforms.filter(t => t.reason.includes("核心特征"));
  const normal = plan.transforms.filter(t => !t.reason.includes("核心特征"));
  return (
    <Panel title={`特征重构计划 (schema: ${plan.target_schema_version})`}>
      <div className="mb-3 grid grid-cols-2 gap-2 text-xs text-slate-500">
        <div>变换前: <b className="text-slate-800">{plan.expected_feature_count_before}</b> 个特征</div>
        <div>变换后: <b className="text-slate-800">{plan.expected_feature_count_after}</b> 个特征</div>
        <div>总变换: <b className="text-slate-800">{plan.transforms.length}</b> 条</div>
        <div>核心保护: <b className="text-amber-700">{core.length}</b> 条</div>
      </div>
      {core.length > 0 && (
        <details className="mb-3" open>
          <summary className="text-xs font-semibold text-amber-700 cursor-pointer hover:text-amber-800">
            ⚠ 核心特征保护 ({core.length} 个特征被拦截，改为修改而非删除)
          </summary>
          <div className="mt-2 space-y-1.5 max-h-64 overflow-auto">
            {core.map((t, i) => (
              <div key={i} className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs">
                <span className="font-semibold text-amber-800">{t.source_feature}</span>
                <span className="ml-2 text-amber-600">{opLabel[t.operation] || t.operation}</span>
                <div className="mt-0.5 text-amber-500 font-mono text-[10px]">{t.reason}</div>
              </div>
            ))}
          </div>
        </details>
      )}
      {normal.length > 0 && (
        <details open>
          <summary className="text-xs font-semibold text-slate-600 cursor-pointer hover:text-slate-700">
            常规变换 ({normal.length} 条)
          </summary>
          <div className="mt-2 space-y-1.5 max-h-64 overflow-auto">
            {normal.map((t, i) => (
              <div key={i} className={`rounded-lg border px-3 py-2 text-xs ${
                t.operation === "DROP"
                  ? "border-red-200 bg-red-50"
                  : "border-slate-200 bg-slate-50"
              }`}>
                <span className="font-semibold text-slate-800">
                  {t.source_feature}
                  {t.target_feature ? <span className="text-slate-400"> → {t.target_feature}</span> : null}
                </span>
                <span className={`ml-2 rounded-full px-1.5 py-0.5 text-[10px] font-semibold ${opBadge[t.operation] || "text-slate-600 bg-slate-100"}`}>
                  {opLabel[t.operation] || t.operation}
                </span>
                <div className="mt-0.5 text-slate-400 font-mono text-[10px]">{t.reason}</div>
              </div>
            ))}
          </div>
        </details>
      )}
    </Panel>
  );
}

function TrainingPlanDetail({plan,id}:{plan:TrainingPlanDetail|null;id:string}){if(!id)return<Empty text="还没有训练计划"/>;if(!plan)return<Empty text="正在读取训练计划…"/>;const w=plan.windows??{};return(<div className="mt-4 space-y-3"><div className="rounded-lg border border-indigo-100 bg-indigo-50 px-4 py-3"><div className="flex justify-between items-center"><div><p className="text-sm font-semibold text-indigo-900">训练计划详情</p><p className="text-xs text-indigo-700 mt-1">Worker 按此计划的数据窗口、算法和验收规则训练候选模型</p></div><span className="font-mono text-[11px] font-semibold text-indigo-800 bg-white rounded-lg px-3 py-1.5">{id}</span></div></div><div className="grid gap-3 lg:grid-cols-2">{[{title:"训练对象",rows:[["模型",plan.model_id],["算法",plan.algorithm],["当前版本",plan.frozen_champion_version||plan.champion_version],["根因",formatRootCause(plan.root_cause_code)],["风险等级",formatRiskLevel(plan.risk_level)]]},{title:"数据窗口",rows:[["基线",w.baseline_window_id],["训练窗口",joinValues(w.training_window_ids)],["验证窗口",joinValues(w.validation_window_ids)],["OOT",`${formatValue(w.oot_window_id)} / ${w.oot_locked?"已锁定":"未锁定"}`],["标签版本",joinValues(plan.label_versions)]]},{title:"训练配置",rows:[["策略",plan.strategy_code],["业务轮次",plan.business_round||plan.max_business_rounds?`第 ${formatValue(plan.business_round)} / ${formatValue(plan.max_business_rounds)} 轮`:"-"],["特征版本",plan.feature_schema_version],["随机种子",plan.random_seed],["策略参数",JSON.stringify(plan.strategy_parameters??{})]]},{title:"验收规则",rows:[["目标指标",joinValues(plan.target_metric_codes)],["资格规则版本",plan.qualification_rule_version],["状态",plan.status],["阻塞原因",joinValues(plan.blocking_reasons)]]}].map(g=>(<div key={g.title} className="rounded-lg border border-slate-200 bg-white p-3"><p className="text-sm font-semibold text-slate-800">{g.title}</p><div className="mt-2 space-y-1.5">{g.rows.map(([l,v])=>(<div key={l} className="grid grid-cols-[90px_1fr] gap-2 rounded bg-slate-50 px-2 py-1.5"><span className="text-xs text-slate-500">{l}</span><span className="text-xs font-semibold text-slate-700 break-words">{v||"-"}</span></div>))}</div></div>))}</div></div>);}
