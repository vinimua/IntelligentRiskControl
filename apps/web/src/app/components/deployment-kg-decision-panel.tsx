"use client";

import { Empty, Panel, StatTile, formatValue } from "./shared";

type DeploymentStageRecord = {
  deployment_id?: string;
  stage?: string;
  decision?: string;
  status?: string;
  health_json?: Record<string, unknown> | string | null;
  result_json?: Record<string, unknown> | string | null;
  created_at?: string;
};

function asObj(value: unknown): Record<string, unknown> {
  if (!value) return {};
  if (typeof value === "string") {
    try {
      return JSON.parse(value) as Record<string, unknown>;
    } catch {
      return {};
    }
  }
  return typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function asList(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value)
    ? value.filter((item) => item && typeof item === "object").map((item) => item as Record<string, unknown>)
    : [];
}

function formatDeploymentDecision(value?: unknown) {
  const map: Record<string, string> = {
    PROMOTE: "提升生产",
    ADVANCE_STAGE: "进入下一阶段",
    HOLD: "暂停观察",
    ROLLBACK: "回滚",
    ABORT_DEPLOYMENT: "终止部署",
  };
  return map[String(value || "")] || formatValue(value);
}

function alertLabel(code: unknown) {
  const map: Record<string, string> = {
    BAD_RATE_DRIFT_HIGH: "坏账率漂移过高",
    HIGH_DEPLOYMENT_SCORE_PSI: "部署分数分布漂移",
    CHALLENGER_AUC_REGRESSION: "候选模型 AUC 下降",
    CHALLENGER_KS_REGRESSION: "候选模型 KS 下降",
    TRAIN_VALID_GAP_LARGE: "训练/验证差距过大",
    RECOVERY_RATE_LOW: "恢复率不足",
    OOT_DEPLOYMENT_RISK: "跨期验证风险",
    DISCRIMINATION_GATE_FAILED: "区分能力门禁失败",
    CALIBRATION_GATE_FAILED: "校准门禁失败",
  };
  return map[String(code || "")] || formatValue(code);
}

function riskLabel(code: unknown) {
  const map: Record<string, string> = {
    BUSINESS_BAD_RATE_RISK: "业务坏账率漂移风险",
    ONLINE_SCORE_DISTRIBUTION_RISK: "线上分数分布风险",
    MODEL_PERFORMANCE_REGRESSION_RISK: "模型效果回退风险",
    OVERFITTING_GENERALIZATION_RISK: "过拟合/泛化风险",
    RECOVERY_INSUFFICIENT_RISK: "修复恢复不足风险",
    OOT_STABILITY_RISK: "跨期稳定性风险",
    MODEL_VALIDATION_GATE_RISK: "模型验证门禁风险",
  };
  return map[String(code || "")] || formatValue(code);
}

function strategyLabel(code: unknown) {
  const map: Record<string, string> = {
    rollback_to_stable: "回滚到稳定版本",
    pause_canary_and_review: "暂停灰度并人工复核",
    reduce_to_previous_canary: "降回上一灰度阶段",
    hold_for_oot_investigation: "暂停并检查跨期稳定性",
    advance_with_close_monitoring: "推进并加强观察",
  };
  return map[String(code || "")] || formatValue(code);
}

export default function DeploymentKgDecisionPanel({
  records,
  state,
}: {
  records: DeploymentStageRecord[];
  state: Record<string, unknown>;
}) {
  const latest = records[records.length - 1];
  const health = asObj(latest?.health_json);
  const alerts = asList(health.deployment_alerts);
  const context = asObj(health.deployment_context);
  const risks = asList(context.deployment_risks);
  const gate = asObj(health.gatekeeper_decision);
  const action = asObj(health.deployment_action);
  const strategies = risks.flatMap((risk) => asList(risk.strategy_candidates));
  const reasons = Array.isArray(gate.decision_reasons)
    ? gate.decision_reasons
    : Array.isArray(state.gatekeeper_reasons)
      ? state.gatekeeper_reasons
      : [];
  const degraded = Boolean(context.retrieval_degraded);
  const hasData = records.length > 0 || alerts.length > 0 || risks.length > 0 || Object.keys(gate).length > 0;

  return (
    <Panel title="部署 KG 决策依据">
      {!hasData ? (
        <Empty text="暂无部署 KG 数据。模型进入部署阶段后自动显示。" />
      ) : (
        <div className="space-y-3">
          <div className="grid gap-3 md:grid-cols-4">
            <StatTile label="阶段记录" value={formatValue(records.length)} sub={latest?.stage ? `最新 ${formatValue(latest.stage)}` : "等待部署"} />
            <StatTile label="KG 状态" value={degraded ? "检索降级" : "正常"} sub={formatValue(context.degradation_reason)} color={degraded ? "#d97706" : "#047857"} />
            <StatTile label="最终决策" value={formatDeploymentDecision(gate.decision || state.gatekeeper_decision || state.deployment_decision)} sub={formatValue(gate.selected_strategy_code || state.selected_deployment_strategy)} />
            <StatTile label="执行结果" value={action.action_failed ? "执行失败" : "已记录"} sub={formatValue(action.action_error)} color={action.action_failed ? "#dc2626" : "#0f766e"} />
          </div>

          <div className="grid gap-3 lg:grid-cols-3">
            <DecisionColumn title="1. 部署告警" tone="red">
              {alerts.length === 0 ? (
                <p className="text-xs text-red-400">当前阶段没有生成 KG 告警。</p>
              ) : (
                alerts.slice(0, 4).map((alert, index) => (
                  <InfoBlock key={index} title={alertLabel(alert.alert_code)} code={alert.alert_code}>
                    当前值 {formatValue(alert.value)} / 阈值 {formatValue(alert.threshold)}
                  </InfoBlock>
                ))
              )}
            </DecisionColumn>

            <DecisionColumn title="2. 识别风险" tone="amber">
              {risks.length === 0 ? (
                <p className="text-xs text-amber-500">暂无 KG 风险命中。</p>
              ) : (
                risks.slice(0, 3).map((risk, index) => (
                  <InfoBlock key={index} title={riskLabel(risk.risk_code)} code={risk.risk_code}>
                    权重 {formatValue(risk.effective_weight_snapshot)} / 置信下界 {formatValue(risk.confidence_lower_bound_snapshot)}
                  </InfoBlock>
                ))
              )}
            </DecisionColumn>

            <DecisionColumn title="3. 推荐策略" tone="indigo">
              {strategies.length === 0 ? (
                <p className="text-xs text-indigo-400">暂无 KG 策略候选。</p>
              ) : (
                strategies.slice(0, 3).map((strategy, index) => (
                  <InfoBlock key={index} title={strategyLabel(strategy.strategy_code)} code={`${formatValue(strategy.strategy_code)} / ${formatValue(strategy.action_type)}`}>
                    权重 {formatValue(strategy.effective_weight_snapshot)} / 案例 {formatValue(strategy.support_case_count)}
                  </InfoBlock>
                ))
              )}
            </DecisionColumn>
          </div>

          <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <p className="text-xs font-semibold text-slate-700">4. Gatekeeper 最终说明</p>
            {reasons.length === 0 ? (
              <p className="mt-2 text-xs text-slate-400">暂无决策理由。</p>
            ) : (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {reasons.map((reason, index) => (
                  <span key={index} className="rounded-md border border-slate-200 bg-white px-2 py-1 font-mono text-[11px] text-slate-600">
                    {formatValue(reason)}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </Panel>
  );
}

function DecisionColumn({ title, tone, children }: { title: string; tone: "red" | "amber" | "indigo"; children: React.ReactNode }) {
  const tones = {
    red: "border-red-100 bg-red-50/60 text-red-700",
    amber: "border-amber-100 bg-amber-50/60 text-amber-700",
    indigo: "border-indigo-100 bg-indigo-50/60 text-indigo-700",
  };
  return (
    <div className={`rounded-lg border p-3 ${tones[tone]}`}>
      <p className="text-xs font-semibold">{title}</p>
      <div className="mt-2 space-y-2">{children}</div>
    </div>
  );
}

function InfoBlock({ title, code, children }: { title: string; code: unknown; children: React.ReactNode }) {
  return (
    <div className="rounded bg-white/80 p-2">
      <div className="text-sm font-semibold text-slate-800">{title}</div>
      <div className="mt-1 break-all font-mono text-[11px] text-slate-500">{formatValue(code)}</div>
      <div className="mt-1 text-xs text-slate-600">{children}</div>
    </div>
  );
}
