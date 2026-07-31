"""监控仪表盘 — 按设计规范重写。"""

import json
from zoneinfo import ZoneInfo
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..repositories.diagnosis_repo import DiagnosisRepo
from ..repositories.monitoring_repo import MonitoringRepo

router = APIRouter(tags=["dashboard"])


def _dashboard_window_key(metric_detail: dict) -> str:
    """Map formal monitoring horizons to the dashboard's three display slots."""
    window_days = metric_detail.get("window_days")
    if window_days == 7:
        return "W2"
    if window_days == 30:
        return "W3"
    return str(metric_detail.get("window_id") or "?")


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    repo = MonitoringRepo(db)
    diagnosis_repo = DiagnosisRepo(db)
    runs = await repo.list_runs(limit=50)

    # 构建嵌入数据——每个模型带指标摘要
    models_data = []
    for r in runs:
        rid = str(r.get("monitoring_run_id", ""))
        active_event = await diagnosis_repo.get_active_event(
            str(r.get("model_id", "")),
            str(r.get("champion_version", "")),
        )
        diagnosis_time = None
        lifecycle_status = None
        if active_event:
            lifecycle_status = str(active_event["status"])
            event_time = active_event.get("event_time")
            if event_time is not None:
                if getattr(event_time, "tzinfo", None) is not None:
                    event_time = event_time.astimezone(ZoneInfo("Asia/Shanghai"))
                diagnosis_time = str(event_time)[:19]
        # 加载该 run 的核心指标摘要
        run_metrics = await repo.get_metrics(rid)
        summary = {}
        for m in run_metrics:
            code = m.get("metric_code", "")
            detail = m.get("metric_detail") or {}
            wid = _dashboard_window_key(detail)
            if code in ("AUC","KS","BAD_RATE","PREDICTION_MEAN","SCORE_PSI","FEATURE_PSI","SAMPLE_SIZE"):
                summary.setdefault(wid, {})[code] = m.get("current_value")
                base_val = m.get("baseline_value")
                if base_val is not None:
                    summary.setdefault(wid, {})[code + "_BASE"] = base_val
                    summary.setdefault("W1", {}).setdefault(code, base_val)
                    summary.setdefault("W1", {}).setdefault(code + "_BASE", base_val)
            if code == "SAMPLE_SIZE":
                summary.setdefault(wid, {})["BAD_COUNT"] = detail.get("bad_count", 0)

        models_data.append({
            "run_id": rid,
            "model_id": str(r.get("model_id", "")),
            "champion_version": str(r.get("champion_version", "")),
            "status": str(r.get("overall_status", "")),
            "lifecycle_status": lifecycle_status,
            "diagnosis_event_id": (
                str(active_event["event_id"]) if active_event else None
            ),
            "diagnosis_time": diagnosis_time,
            "alert_count": r.get("alert_count", 0),
            "max_severity": str(r.get("max_alert_severity") or ""),
            "started_at": str(r.get("started_at") or "")[:19],
            "monitor_window_id": str(r.get("monitor_window_id") or r.get("current_window_id", "")),
            "metrics": summary,
        })

    # 取最新 run 的详细指标（展开面板用）
    metrics_items = []
    latest_run_id = models_data[0]["run_id"] if models_data else None
    if latest_run_id:
        metrics_items = await repo.get_metrics(latest_run_id)

    # 按窗口 + 指标整理
    window_metrics = {}
    for m in metrics_items:
        detail = m.get("metric_detail") or {}
        wid = _dashboard_window_key(detail)
        code = m.get("metric_code", "")
        if wid not in window_metrics:
            window_metrics[wid] = {}
        if code not in window_metrics[wid]:
            window_metrics[wid][code] = {
                "current": m.get("current_value"),
                "baseline": m.get("baseline_value"),
                "delta": m.get("delta"),
                "score_type": (m.get("metric_detail") or {}).get("score_type", ""),
                "category": (m.get("metric_detail") or {}).get("category", ""),
                "feature_name": (m.get("metric_detail") or {}).get("feature_name", ""),
            }

        # 特殊处理 MONITOR_STATUS
        if code == "MONITOR_STATUS":
            window_metrics[wid][code]["status"] = (m.get("metric_detail") or {}).get("status", "")

        # SAMPLE_SIZE 的 bad_count
        if code == "SAMPLE_SIZE":
            window_metrics[wid][code]["bad_count"] = (m.get("metric_detail") or {}).get("bad_count", 0)

        # FEATURE_PSI 的 n_features/max_psi
        if code == "FEATURE_PSI":
            window_metrics[wid][code]["max_psi"] = (m.get("metric_detail") or {}).get("max_psi")
            window_metrics[wid][code]["n_features"] = (m.get("metric_detail") or {}).get("n_features")

    # 特征中文说明
    feature_labels = {
        "credit_query_times": "征信查询次数", "multi_loan_count": "多头借贷数量",
        "overdue_history": "逾期历史", "credit_utilization": "信用额度使用率",
        "credit_length_months": "信用时长(月)", "max_overdue_days": "最大逾期天数",
        "social_score": "社交评分", "telecom_score": "电信评分",
        "ecomm_risk_score": "电商风险评分", "judicial_risk_score": "司法风险评分",
        "blacklist_hit": "黑名单命中", "app_duration": "APP使用时长",
        "click_frequency": "点击频率", "page_depth": "页面深度",
        "session_count": "会话次数", "night_activity_ratio": "夜间活跃占比",
        "login_fail_count": "登录失败次数", "reg_to_apply_days": "注册到申请天数",
        "device_risk_score": "设备风险评分", "ip_change_freq": "IP变更频率",
        "gps_anomaly": "GPS异常", "device_type": "设备类型",
        "emulator_flag": "模拟器标识", "age": "年龄",
        "income_level": "收入水平", "consumption_level": "消费水平",
        "education_level": "教育程度", "job_stability": "工作稳定性",
        "marital_status": "婚姻状况", "gender": "性别",
        "city_tier": "城市等级", "debt_income_ratio": "负债收入比",
        "loan_amount_request": "申请贷款金额", "repayment_period": "还款周期",
    }

    # Drift per-feature detail: drift_detail[window_id][feature_name] = {psi, js, ks, ...}
    drift_detail = {}
    quality_detail = {}
    for m in metrics_items:
        md = m.get("metric_detail") or {}
        cat = md.get("category", "")
        wid = _dashboard_window_key(md)
        fn = md.get("feature_name", "")
        if not fn:
            continue
        code = m.get("metric_code", "")
        cur = m.get("current_value")
        if cat == "drift":
            drift_detail.setdefault(wid, {}).setdefault(fn, {})[code] = cur
        elif cat == "quality":
            if code == "Q_DQ_FLAG":
                quality_detail.setdefault(wid, {}).setdefault(fn, {})["FLAG"] = md.get("value_str", "-")
            else:
                quality_detail.setdefault(wid, {}).setdefault(fn, {})[code] = cur

    # Drift TOP-15 per window
    drift_top = {}
    for wid in ["W1", "W2", "W3"]:
        wd = drift_detail.get(wid, {})
        top = []
        for fn, metrics in wd.items():
            psi = metrics.get("D_PSI")
            if psi is not None:
                top.append({
                    "feature": fn,
                    "label": feature_labels.get(fn, ""),
                    "psi": psi,
                    "js": metrics.get("D_JS_DIVERGENCE"),
                    "ks": metrics.get("D_KS_STATISTIC"),
                    "wasserstein": metrics.get("D_WASSERSTEIN_DISTANCE"),
                    "ks_p": metrics.get("D_KS_P_VALUE"),
                    "ks_q": metrics.get("D_KS_Q_VALUE"),
                })
        top.sort(key=lambda x: x["psi"] or 0, reverse=True)
        drift_top[wid] = top[:100]

    # Quality TOP-15 per window
    quality_top = {}
    for wid in ["W1", "W2", "W3"]:
        wq = quality_detail.get(wid, {})
        top = []
        for fn, metrics in wq.items():
            dq = metrics.get("Q_DQ_SCORE")
            if dq is not None:
                top.append({
                    "feature": fn,
                    "label": feature_labels.get(fn, ""),
                    "dq_score": dq,
                    "missing": metrics.get("Q_MISSING_RATE"),
                    "outlier": metrics.get("Q_OUTLIER_RATE"),
                    "default": metrics.get("Q_DEFAULT_VALUE_RATE"),
                    "flag": metrics.get("FLAG", "-"),
                })
        top.sort(key=lambda x: x["dq_score"] or 0)
        quality_top[wid] = top[:100]

    # 管道步骤数据（从 run context 推算）
    pipeline_steps = []
    if latest_run_id:
        total_metrics = len(metrics_items)
        win_counts = {}
        for m in metrics_items:
            w = (m.get("metric_detail") or {}).get("window_id", "?")
            win_counts[w] = win_counts.get(w, 0) + 1
        pipeline_steps = [
            {"step": "WP02 基线构建", "status": "OK", "output": "MonitoringBaseline", "rows": "1 个基线包"},
            {"step": "WP03 窗口加载", "status": "OK", "output": "W0/W1/W2/W3 Parquet", "rows": "4 个窗口"},
            {"step": "WP04 模型预测", "status": "OK", "output": "risk_score + y_pred_proba", "rows": "Champion V1 + IsotonicCalibrator"},
            {"step": "WP05 漂移检测", "status": "OK", "output": "PSI/JS/KS/Wasserstein + BH", "rows": f"{sum(1 for m in metrics_items if (m.get('metric_detail') or {}).get('category')=='drift')} 条"},
            {"step": "WP05 数据质量", "status": "OK", "output": "feature_quality × 34 特征", "rows": f"{sum(1 for m in metrics_items if (m.get('metric_detail') or {}).get('category')=='quality')} 条"},
            {"step": "WP06 性能评估", "status": "OK", "output": "AUC/KS/PR_AUC/BRIER/ECE/BAD_RECALL", "rows": f"{sum(1 for m in metrics_items if (m.get('metric_detail') or {}).get('category')=='core')} 条"},
            {"step": "WP07 检测器", "status": "PENDING", "output": "ADWIN/PageHinkley/KSWIN/RobustZ", "rows": "待接入"},
            {"step": "WP08 告警 + 持久化", "status": "OK", "output": "monitoring_metrics + monitoring_alerts", "rows": f"总计 {total_metrics} 条入库"},
        ]

    # 窗口时间线
    window_timeline = []
    for w_id in ["W0", "W1", "W2", "W3"]:
        w_info = window_metrics.get(w_id, {})
        sample = (w_info.get("SAMPLE_SIZE") or {}).get("current") if w_info else None
        bad = (w_info.get("SAMPLE_SIZE") or {}).get("bad_count") if w_info else None
        window_timeline.append({
            "id": w_id,
            "role": "FIXED_REFERENCE" if w_id == "W0" else f"MONITOR_WINDOW",
            "locked": w_id == "W0",
            "sample_count": int(sample) if sample else None,
            "bad_count": int(bad) if bad else None,
        })

    embedded = {
        "models": models_data,
        "latest_run_id": latest_run_id,
        "window_metrics": {str(k): v for k, v in window_metrics.items()},
        "drift_top": drift_top,
        "quality_top": quality_top,
        "pipeline_steps": pipeline_steps,
        "window_timeline": window_timeline,
        "total_metrics": len(metrics_items),
    }

    return HTMLResponse(
        build_html(json.dumps(embedded, ensure_ascii=False, default=str)),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    )


def build_html(embedded_json: str) -> str:
    html_template = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>WP02-WP08 Champion V1 持续监测</title>
<style>
:root{
--bg:#f9f9f7;--card:#fcfcfb;--text:#1a1a1a;--muted:#6b6b6b;--border:rgba(0,0,0,.08);
--blue:#1a73e8;--green:#0ca30c;--yellow:#fab219;--red:#d03b3b;--purple:#7c3aed;
--mono:'Cascadia Code','SF Mono',Consolas,monospace;--sans:system-ui,-apple-system,sans-serif
}
.dark{
--bg:#0d0d0d;--card:#1a1a19;--text:#e0e0e0;--muted:#888;--border:rgba(255,255,255,.08)
}
*{margin:0;padding:0;box-sizing:border-box}
body{font:13px/1.5 var(--sans);background:var(--bg);color:var(--text);transition:.2s}
.topbar{display:flex;justify-content:space-between;align-items:center;padding:12px 24px;background:var(--card);border-bottom:1px solid var(--border)}
.topbar h1{font-size:15px;font-weight:600}
.topbar .sub{font-size:11px;color:var(--muted);margin-left:12px}
.container{max-width:1500px;margin:0 auto;padding:20px}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:14px}
.stat-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.stat{text-align:center;padding:16px 8px;background:var(--card);border:1px solid var(--border);border-radius:8px}
.stat .num{font-size:28px;font-weight:700;font-family:var(--mono)}
.stat .lbl{font-size:11px;color:var(--muted);margin-top:4px}
.tabs{display:flex;gap:0;margin-bottom:0;border-bottom:2px solid var(--border)}
.tab-btn{padding:8px 20px;border:none;background:none;cursor:pointer;font-size:13px;color:var(--muted);border-bottom:2px solid transparent;margin-bottom:-2px;transition:.15s}
.tab-btn.active{color:var(--blue);border-bottom-color:var(--blue);font-weight:600}
.tab-content{display:none}
.tab-content.active{display:block}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{padding:6px 8px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}
th{color:var(--muted);font-weight:500;font-size:11px;position:sticky;top:0;background:var(--card)}
td{font-family:var(--mono)}
td.wide{max-width:140px;overflow:hidden;text-overflow:ellipsis}
tr:hover{background:rgba(26,115,232,.05)}
tr.expanded{background:rgba(26,115,232,.08)}
.green{color:var(--green)}.red{color:var(--red)}.yellow{color:var(--yellow)}.blue{color:var(--blue)}.purple{color:var(--purple)}.muted{color:var(--muted)}
.tag{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600}
.tag-ok{background:rgba(12,163,12,.12);color:var(--green)}
.tag-warn{background:rgba(250,178,25,.12);color:var(--yellow)}
.tag-err{background:rgba(208,59,59,.12);color:var(--red)}
.btn{padding:4px 12px;border:1px solid var(--border);border-radius:5px;background:var(--card);color:var(--text);cursor:pointer;font-size:12px;transition:.15s}
.btn:hover{border-color:var(--blue);color:var(--blue)}
.btn.active{background:var(--blue);color:#fff;border-color:var(--blue)}
input{padding:6px 10px;border:1px solid var(--border);border-radius:5px;background:var(--card);color:var(--text);font-size:12px;outline:none}
input:focus{border-color:var(--blue)}
.operate{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;gap:12px;flex-wrap:wrap}
.expand-arrow{cursor:pointer;display:inline-block;transition:transform .2s;font-size:12px}
.expand-arrow.rotated{transform:rotate(90deg)}
.detail-row td{padding:0;border:none}
.detail-panel{display:flex;flex-direction:column;gap:12px;padding:12px 8px}
.detail-card{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:10px}
.detail-card h5{font-size:12px;margin-bottom:6px}
.detail-card table{font-size:11px}
.detail-card td{font-family:var(--mono);font-size:11px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}
.dot-g{background:var(--green)}.dot-y{background:var(--yellow)}.dot-r{background:var(--red)}
.footer{text-align:center;color:var(--muted);font-size:11px;padding:20px}
.num{font-variant-numeric:tabular-nums;font-family:var(--mono)}
.sort-btn{font-size:11px;padding:3px 8px}
@media(max-width:1000px){.stat-row{grid-template-columns:repeat(2,1fr)}.detail-panel{grid-template-columns:1fr}}
.diag-hero{display:flex;align-items:center;gap:16px;padding:16px;border-radius:8px;margin-bottom:12px}
.diag-hero .score{font-size:48px;font-weight:700;font-family:var(--mono);line-height:1}
.diag-hero .meta{font-size:13px;line-height:1.6}
.diag-hero .meta .rc{font-size:16px;font-weight:600}
.ev-badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:600;margin:0 2px}
.ev-D{background:rgba(42,120,214,.12);color:var(--blue)}
.ev-R{background:rgba(235,104,52,.12);color:#eb6834}
.ev-C{background:rgba(27,175,122,.12);color:var(--green)}
.ev-T{background:rgba(74,58,167,.12);color:#4a3aa7}
.ev-I{background:rgba(232,123,164,.12);color:#e87ba4}
.ev-bar{display:flex;gap:4px;align-items:center;margin:4px 0}
.ev-seg{height:6px;border-radius:2px;flex-shrink:0}
.ev-seg.support{background:var(--green)}.ev-seg.against{background:var(--red)}.ev-seg.neutral{background:#d5d4d0}
.rc-row{cursor:pointer}.rc-row:hover{background:rgba(26,115,232,.05)}
.action-tag{padding:4px 12px;border-radius:6px;font-size:12px;font-weight:600;display:inline-block}
.action-iterate{background:rgba(26,115,232,.12);color:var(--blue)}
.action-repair{background:rgba(250,178,25,.15);color:var(--yellow)}
.action-observe{background:rgba(12,163,12,.12);color:var(--green)}
.action-manual{background:rgba(208,59,59,.12);color:var(--red)}
.model-selector{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
.model-selector select{padding:6px 12px;border:1px solid var(--border);border-radius:5px;background:var(--card);color:var(--text);font-size:13px;min-width:200px}
</style></head>
<body>
<div style="background:var(--blue);color:#fff;text-align:center;padding:4px;font-size:11px" id="loadBanner">PAGE LOADED</div>
<div class="topbar">
  <div><h1>模型监测</h1><span class="sub" id="subtitle">周期 —</span></div>
  <button class="btn" onclick="toggleTheme()" id="themeBtn">🌙 暗色模式</button>
</div>
<div class="container">
  <div class="stat-row">
    <div class="stat"><div class="num green" id="stat-pass">-</div><div class="lbl">全部通过</div></div>
    <div class="stat"><div class="num" id="stat-time">-</div><div class="lbl">墙钟时间</div></div>
    <div class="stat"><div class="num" id="stat-mem">-</div><div class="lbl">入库指标</div></div>
    <div class="stat"><div class="num" id="stat-cc">-</div><div class="lbl">监控窗口</div></div>
  </div>

  <div class="card" style="margin-top:14px">
    <div class="tabs" id="main-tabs">
      <button class="tab-btn active" data-tab="tab-runs">运行列表 &amp; 指标详情</button>
      <button class="tab-btn" data-tab="tab-outputs">监测产出物</button>
      <button class="tab-btn" data-tab="tab-timeline">窗口时间线</button>
      <button class="tab-btn" data-tab="tab-diagnosis">任务二 诊断</button>
    </div>

    <!-- Tab 1: 运行列表 -->
    <div class="tab-content active" id="tab-runs" style="padding-top:12px">
      <div class="operate">
        <input type="text" id="searchBox" placeholder="🔍 搜索模型 ID..." oninput="filterRuns()" style="width:240px">
        <div style="display:flex;gap:6px;align-items:center">
          <span class="muted" style="font-size:11px">排序:</span>
          <button class="btn sort-btn active" data-sort="default">默认</button>
          <button class="btn sort-btn" data-sort="auc">AUC ↓</button>
          <button class="btn sort-btn" data-sort="ks">KS ↓</button>
          <button class="btn sort-btn" data-sort="time">耗时 ↓</button>
          <span class="muted" style="font-size:11px;margin-left:8px" id="modelCount">- 个模型</span>
        </div>
      </div>
      <div style="overflow-x:auto">
        <table id="runsTable"><thead><tr>
          <th></th><th>Run ID</th><th>Model</th><th>Version</th><th>Status</th><th>Alerts</th><th>Severity</th><th>Time</th>
          <th>AUC(Base)</th><th>AUC(7D)</th><th>AUC(30D)</th><th>ΔAUC(30D)</th>
          <th>KS(Base)</th><th>KS(7D)</th><th>KS(30D)</th><th>ΔKS(30D)</th>
          <th>BAD_RATE</th><th>Pred_Mean</th><th>PRED_PSI</th>
        </tr></thead><tbody id="runsBody"></tbody></table>
      </div>
    </div>

    <!-- Tab 2: 监测产出物 -->
    <div class="tab-content" id="tab-outputs" style="padding-top:12px">
      <table><thead><tr><th>产出物</th><th>路径 / 描述</th><th>行数 / 说明</th></tr></thead><tbody id="outputsBody"></tbody></table>
    </div>

    <!-- Tab 3: 窗口时间线 -->
    <div class="tab-content" id="tab-timeline" style="padding-top:12px">
      <div class="stat-row" id="timelineCards"></div>
      <div class="muted" style="text-align:center;margin-top:8px;font-size:11px">时区 Asia/Shanghai · 评估日 2025-12-31</div>
    </div>

    <!-- Tab 4: 任务二 诊断 -->
    <div class="tab-content" id="tab-diagnosis" style="padding-top:12px">
      <div class="model-selector">
        <span style="font-size:13px;font-weight:600">选择模型:</span>
        <select id="diagModelSelect" onchange="loadDiagnosis()">
          <option value="">— 请选择模型 —</option>
        </select>
        <span class="muted" style="font-size:11px" id="diagStatus"></span>
      </div>
      <div id="diagContent"></div>
    </div>

  </div>

  <div class="footer">任务一 WP03-WP08 Champion V1 持续监测 V1.1 · 验收日期 2026-07-18</div>
</div>

<script id="dash-data" type="application/json">__JSON__</script>
<script>
document.getElementById('loadBanner').textContent = 'JS STARTED';
var DATA = JSON.parse(document.getElementById('dash-data').textContent);
var MODELS = DATA.models || [];
var WM = DATA.window_metrics || {};
var DRIFT = DATA.drift_top || {};
var QTOP = DATA.quality_top || {};
var STEPS = DATA.pipeline_steps || [];
var TIMELINE = DATA.window_timeline || [];
var SORT = 'default';
var EXPANDED = {};

function $(id){return document.getElementById(id)}

// ── 统计卡片 ──
(function(){
  var n = MODELS.length;
  var ok = MODELS.filter(function(m){return m.status==='COMPLETED'}).length;
  $('stat-pass').textContent = ok+'/'+n;
  $('stat-pass').className = 'num ' + (ok===n?'green':'yellow');
  $('stat-time').textContent = '-';
  $('stat-mem').textContent = DATA.total_metrics||0;
  $('stat-cc').textContent = Object.keys(WM).filter(function(k){return k!=='W0'&&WM[k]&&Object.keys(WM[k]).length>0}).length;
  if(MODELS.length>0){
    $('subtitle').textContent = '周期 '+MODELS[0].started_at+' · '+MODELS.length+' 个模型';
  }
})();

// ── Tab 切换 ──
document.getElementById('main-tabs').addEventListener('click',function(e){
  if(!e.target.classList.contains('tab-btn'))return;
  var btns=document.querySelectorAll('.tab-btn');
  for(var i=0;i<btns.length;i++)btns[i].classList.remove('active');
  e.target.classList.add('active');
  var tid=e.target.dataset.tab;
  var contents=document.querySelectorAll('.tab-content');
  for(var j=0;j<contents.length;j++)contents[j].classList.remove('active');
  $(tid).classList.add('active');
});

// ── 主题切换 ──
function toggleTheme(){
  var dark=document.body.classList.toggle('dark');
  $('themeBtn').textContent=dark?'☀️ 亮色模式':'🌙 暗色模式';
}

// ── 运行列表 ──
function renderRuns(){
  var filtered=filterModels();
  $('modelCount').textContent=filtered.length+' 个模型';
  var tbody=$('runsBody');tbody.innerHTML='';
  for(var i=0;i<filtered.length;i++){
    var r=filtered[i],rid=r.run_id||'';
    var sm=r.metrics||{};var w1=sm['W1']||{},w2=sm['W2']||{},w3=sm['W3']||{};
    var auc1=v(w1,'AUC'),auc2=v(w2,'AUC'),auc3=v(w3,'AUC');
    var ks1=v(w1,'KS'),ks2=v(w2,'KS'),ks3=v(w3,'KS');
    var auc0=v(w3,'AUC_BASE')||v(w2,'AUC_BASE')||v(w1,'AUC_BASE');
    var dAUC=auc0!=null&&auc3!=null?auc3-auc0:null;
    var ks0=v(w3,'KS_BASE')||v(w2,'KS_BASE')||v(w1,'KS_BASE');var dKS=ks0!=null&&ks3!=null?ks3-ks0:null;
    var badRate=v(w3,'BAD_RATE'),predMean=v(w3,'PREDICTION_MEAN');
    var psi=w3['SCORE_PSI'];
    var sev=r.max_severity||'';var sc=sev==='CRITICAL'||sev==='HIGH'?'tag-err':sev==='WARNING'?'tag-warn':'tag-ok';
    var phase=r.lifecycle_status||r.status||'';
    var phaseLabel=phase==='WAITING_AGENT_DECISION'?'等待Agent决策':phase==='OPEN'?'诊断中':phase;
    var phaseClass=r.lifecycle_status?'tag-warn':'tag-ok';
    var displayTime=r.lifecycle_status?(r.diagnosis_time||'—'):r.started_at;

    var exp=EXPANDED[rid];
    tbody.innerHTML+='<tr class="'+(exp?'expanded':'')+'" onclick="toggleRow(\\''+rid+'\\',this)" style="cursor:pointer">'+
      '<td><span class="expand-arrow'+(exp?' rotated':'')+'">▶</span></td>'+
      '<td class="wide muted">'+h(rid)+'</td>'+
      '<td><b>'+h(r.model_id)+'</b></td>'+
      '<td class="muted">'+h(r.champion_version)+'</td>'+
      '<td><span class="tag '+phaseClass+'">'+h(phaseLabel)+'</span></td>'+
      '<td class="'+(r.alert_count>0?'red':'')+'">'+r.alert_count+'</td>'+
      '<td><span class="tag '+sc+'">'+(sev||'NORMAL')+'</span></td>'+
      '<td class="muted">'+h(displayTime)+(r.lifecycle_status?'（诊断时间点）':'')+'</td>'+
      '<td class="num">'+f(auc1)+'</td><td class="num">'+f(auc2)+'</td><td class="num blue">'+f(auc3)+'</td>'+
      '<td class="num '+(dAUC!=null?(dAUC<0?'red':'green'):'')+'">'+(dAUC!=null?(dAUC>0?'+':'')+dAUC.toFixed(4):'-')+'</td>'+
      '<td class="num">'+f(ks1)+'</td><td class="num">'+f(ks2)+'</td><td class="num blue">'+f(ks3)+'</td>'+
      '<td class="num '+(dKS!=null?(dKS<0?'red':'green'):'')+'">'+(dKS!=null?(dKS>0?'+':'')+dKS.toFixed(4):'-')+'</td>'+
      '<td class="num">'+f(badRate)+'</td><td class="num">'+f(predMean)+'</td>'+
      '<td class="num '+(psi!=null&&psi>0.1?'red':'green')+'">'+f6(psi)+'</td>'+
      '</tr>';
    if(exp)renderDetail(rid, tbody);
  }
}

function toggleRow(rid,tr){
  if(EXPANDED[rid]){delete EXPANDED[rid]}else{EXPANDED[rid]=true}
  renderRuns();
}

var METRICS_CACHE={};

function renderDetail(rid,tr){
  var model=null;
  for(var i=0;i<MODELS.length;i++){if(MODELS[i].run_id===rid){model=MODELS[i];break}}
  if(!model)return;

  // 如果有缓存就直接渲染，否则从 API 加载
  if(METRICS_CACHE[rid]){
    _renderDetailContent(rid, METRICS_CACHE[rid]);
    return;
  }

  // 懒加载
  document.getElementById('runsBody').insertAdjacentHTML('beforeend','<tr class=\"detail-row\"><td colspan=\"19\"><div class=\"detail-panel\"><div class=\"loading\">Loading metrics for '+h(model.model_id)+'...</div></div></td></tr>');

  var xhr=new XMLHttpRequest();
  xhr.open('GET','http://localhost:8000/api/monitoring/runs/'+rid+'/metrics');
  xhr.onload=function(){
    if(xhr.status===200){
      var data=JSON.parse(xhr.responseText);
      var items=data.data?data.data.items||[]:[];
      // Build window_metrics from the raw items
      var wm={};
      for(var i=0;i<items.length;i++){
        var m=items[i];
        var md=m.metric_detail||{};
        var wid=md.window_days===7?'W2':md.window_days===30?'W3':(md.window_id||'?');
        var code=m.metric_code;
        if(!wm[wid])wm[wid]={};
        if(!wm[wid][code])wm[wid][code]={};
        wm[wid][code]={
          current:m.current_value, baseline:m.baseline_value,
          delta:m.delta,
          score_type:md.score_type||'',
          status:md.status||'',
        };
        if(m.baseline_value!=null){
          if(!wm.W1)wm.W1={};
          if(!wm.W1[code]){
            wm.W1[code]={
              current:m.baseline_value,
              baseline:m.baseline_value,
              delta:0,
              score_type:md.score_type||'',
              status:'',
            };
          }
        }
      }
      METRICS_CACHE[rid]=wm;
      // Remove the loading row and re-render
      var detailRows=document.querySelectorAll('.detail-row');
      for(var i=0;i<detailRows.length;i++)detailRows[i].remove();
      EXPANDED[rid]=true;
      _renderDetailContent(rid, wm);
    }
  };
  xhr.onerror=function(){};
  xhr.send();
  return;
}

function _renderDetailContent(rid, wm){
  var model=null;
  for(var i=0;i<MODELS.length;i++){if(MODELS[i].run_id===rid){model=MODELS[i];break}}
  if(!model)return;
  var w1=wm['W1']||{},w2=wm['W2']||{},w3=wm['W3']||{};

  var html='<tr class="detail-row"><td colspan="19"><div class="detail-panel">';

  var metricsByCat={
    core:[
      {code:'AUC',lbl:'AUC',hi:true},{code:'KS',lbl:'KS',hi:true},
      {code:'PR_AUC',lbl:'PR_AUC',hi:true},{code:'BAD_RECALL',lbl:'BAD_RECALL',hi:true},
      {code:'BRIER',lbl:'BRIER'},{code:'ECE',lbl:'ECE'},
      {code:'BAD_RATE',lbl:'BAD_RATE'},{code:'BAD_RATE_DELTA',lbl:'BAD_RATE_DELTA'},
      {code:'PERFORMANCE_DROP_MAX',lbl:'PERFORMANCE_DROP_MAX'},
      {code:'MONITOR_STATUS',lbl:'MONITOR_STATUS',isStatus:true}
    ],
    distribution:[
      {code:'PREDICTION_STD',lbl:'PREDICTION_STD'},{code:'PREDICTION_MIN',lbl:'PREDICTION_MIN'},
      {code:'PREDICTION_MAX',lbl:'PREDICTION_MAX'},{code:'SCORE_PSI',lbl:'SCORE_PSI'},
      {code:'PREDICTION_MEAN',lbl:'PREDICTION_MEAN'}
    ],
    aggregate:[{code:'FEATURE_PSI',lbl:'FEATURE_PSI'}]
  };

  for(var cat in metricsByCat){
    var catItems=metricsByCat[cat];
    var catNames={core:'📊 Core 指标 — 模型级排序+校准+标签',distribution:'📈 Distribution — 预测分数分布形态',aggregate:'📐 Aggregate — 所有特征 PSI 汇总为 1 个模型级指标'};
    html+='<div class="detail-card"><h5>'+catNames[cat]+'</h5>';
    html+='<table><tr><th>Metric</th><th>Baseline</th><th>7D</th><th>30D</th><th>Reference</th><th>Δ(30D)</th><th>Type</th></tr>';
    for(var ai=0;ai<catItems.length;ai++){
      var m=catItems[ai];
      var av=v(w1,m.code),bv=v(w2,m.code),cv=v(w3,m.code),base=(w3[m.code]||{}).baseline||(w2[m.code]||{}).baseline||(w1[m.code]||{}).baseline;if(base==null)base=bl(m.code);
      var d=cv!=null&&base!=null?cv-base:null;
      var hi=m.hi===true,dc='';
      if(d!=null){dc=hi?(d<0?'red':'green'):(d>0?'red':'green')}
      var st=vst(w3,m.code)||'-';
      if(m.isStatus){
        var ms=(w3[m.code]||{}).status||(w2[m.code]||{}).status||(w1[m.code]||{}).status||'-';
        html+='<tr><td><b>'+m.lbl+'</b></td><td colspan="3" class="num">'+ms+'</td><td class="num muted">-</td><td>-</td><td style="font-size:10px;color:#8b949e">'+st+'</td></tr>';
      }else{
        html+='<tr><td><b>'+m.lbl+'</b></td>'+
          '<td class="num">'+f(av)+'</td><td class="num">'+f(bv)+'</td><td class="num blue">'+f(cv)+'</td>'+
          '<td class="num muted">'+f(base)+'</td>'+
          '<td class="num '+dc+'">'+(d!=null?(d>0?'+':'')+d.toFixed(4):'-')+'</td>'+
          '<td style="font-size:10px;color:#8b949e">'+st+'</td></tr>';
      }
    }
    html+='</table></div>';
  }

  // Meta card
  html+='<div class="detail-card"><h5>📋 Meta <span style="font-weight:400;font-size:11px;color:var(--muted)">— 非算法产出：行数统计 + 列名校验</span></h5>';
  html+='<table><tr><th>Metric</th><th>Baseline</th><th>7D</th><th>30D</th></tr>';
  [{code:'SAMPLE_SIZE',lbl:'SAMPLE_SIZE',isInt:true},{code:'SCHEMA_CONSISTENCY',lbl:'SCHEMA_CONSISTENCY',isSchema:true}].forEach(function(m){
    var av=v(w1,m.code),bv=v(w2,m.code),cv=v(w3,m.code);
    html+='<tr><td><b>'+m.lbl+'</b></td>'+
      '<td class="num">'+(m.isInt?(av!=null?av.toFixed(0):'-'):m.isSchema?(av==1?'OK':'-'):f(av))+'</td>'+
      '<td class="num">'+(m.isInt?(bv!=null?bv.toFixed(0):'-'):m.isSchema?(bv==1?'OK':'-'):f(bv))+'</td>'+
      '<td class="num blue">'+(m.isInt?(cv!=null?cv.toFixed(0):'-'):m.isSchema?(cv==1?'OK':'-'):f(cv))+'</td></tr>';
  });
  html+='</table></div>';

  // Drift 全部 (per window)
  for(var wi=0;wi<3;wi++){
    var wid=['W1','W2','W3'][wi];
    var topD=(DRIFT[wid]||[]).slice(0,100);
    if(!topD.length)continue;
    html+='<div class="detail-card"><h5>🔬 Drift 全部 — '+wid+' <span style="font-weight:400;font-size:11px;color:var(--muted)">— 特征分布漂移（PSI/JS/KS/Wasserstein），按 PSI 降序</span></h5>';
    html+='<table><tr><th>#</th><th>Feature</th><th>说明</th><th>PSI</th><th>JS</th><th>KS</th><th>W-dist</th></tr>';
    for(var di=0;di<topD.length;di++){
      var dm=topD[di];
      var pc=dm.psi>0.25?'red':dm.psi>0.1?'yellow':'green';
      html+='<tr><td>'+(di+1)+'</td><td>'+dm.feature+'</td><td style="font-size:10px;color:var(--muted)">'+(dm.label||'')+'</td>'+
        '<td class="num '+pc+'">'+f(dm.psi)+'</td>'+
        '<td class="num">'+f(dm.js)+'</td><td class="num">'+f(dm.ks)+'</td>'+
        '<td class="num">'+f(dm.wasserstein)+'</td></tr>';
    }
    html+='</table></div>';
  }

  // Quality 全部 (per window)
  for(var wi=0;wi<3;wi++){
    var wid=['W1','W2','W3'][wi];
    var topQ=(QTOP[wid]||[]).slice(0,100);
    if(!topQ.length)continue;
    html+='<div class="detail-card"><h5>✅ Quality 全部 — '+wid+' <span style="font-weight:400;font-size:11px;color:var(--muted)">— 数据质量检查（缺失率/离群率/默认值率/DQ分），按 DQ 分降序</span></h5>';
    html+='<table><tr><th>#</th><th>Feature</th><th>说明</th><th>DQ</th><th>Missing</th><th>Outlier</th><th>Default</th><th>Flag</th></tr>';
    for(var qi=0;qi<topQ.length;qi++){
      var qm=topQ[qi];
      var fc=qm.flag==='ALERT'?'tag-err':qm.flag==='WARN'?'tag-warn':'tag-ok';
      html+='<tr><td>'+(qi+1)+'</td><td>'+qm.feature+'</td><td style="font-size:10px;color:var(--muted)">'+(qm.label||'')+'</td>'+
        '<td class="num">'+f(qm.dq_score)+'</td>'+
        '<td class="num">'+f(qm.missing)+'</td><td class="num">'+f(qm.outlier)+'</td>'+
        '<td class="num">'+f(qm.default)+'</td>'+
        '<td><span class="tag '+fc+'">'+qm.flag+'</span></td></tr>';
    }
    html+='</table></div>';
  }

  html+='</div></td></tr>';
  document.getElementById('runsBody').insertAdjacentHTML('beforeend',html);
}

function v(win,code){var m=win&&win[code];if(m==null)return null;if(typeof m==="number")return m;return m.current!=null?m.current:null}
function bl(code){for(var w in WM){var b=(WM[w][code]||{}).baseline;if(b!=null)return b}return null}
function vst(win,code){return (win&&win[code]||{}).score_type||''}
function f(v){return v!=null?v.toFixed(4):'-'}
function f6(v){return v!=null?v.toFixed(6):'-'}
function h(s){return s||'-'}

// ── 排序 / 过滤 ──
function filterModels(){
  var q=($('searchBox').value||'').toLowerCase();
  var list=MODELS.filter(function(m){return !q||(m.model_id||'').toLowerCase().indexOf(q)>=0});
  if(SORT==='auc')list.sort(function(a,b){var av=v(((a.metrics||{}).W3)||{},'AUC'),bv=v(((b.metrics||{}).W3)||{},'AUC');return(bv||0)-(av||0)});
  else if(SORT==='ks')list.sort(function(a,b){var av=v(((a.metrics||{}).W3)||{},'KS'),bv=v(((b.metrics||{}).W3)||{},'KS');return(bv||0)-(av||0)});
  return list;
}
function filterRuns(){renderRuns()}
document.querySelectorAll('.sort-btn').forEach(function(b){
  b.addEventListener('click',function(){
    document.querySelectorAll('.sort-btn').forEach(function(x){x.classList.remove('active')});
    this.classList.add('active');
    SORT=this.dataset.sort;renderRuns();
  });
});

// ── Tab 2: 产出物 ──
(function(){
  var tbody=$('outputsBody'),h='';
  for(var i=0;i<STEPS.length;i++){
    var s=STEPS[i];
    h+='<tr><td>'+s.step+'</td><td style="font-family:var(--mono)">'+s.output+'</td><td class="num">'+s.rows+'</td></tr>';
  }
  tbody.innerHTML=h;
})();

// ── Tab 3: 窗口时间线 ──
(function(){
  var cards=$('timelineCards'),h='';
  for(var i=0;i<TIMELINE.length;i++){
    var t=TIMELINE[i],lc=t.locked?'🔒 LOCKED':'unlocked';
    var color=t.id==='W0'?'purple':'blue';
    h+='<div class="stat"><div class="num '+color+'">'+t.id+'</div>'+
      '<div class="lbl">'+t.role+'</div>'+
      '<div class="lbl muted">Samples: '+(t.sample_count||'-')+' / Bad: '+(t.bad_count||'-')+'</div>'+
      '<div class="lbl '+(t.locked?'red':'muted')+'">'+lc+'</div></div>';
  }
  cards.innerHTML=h;
})();

// ── Tab 4: 任务二 诊断 ──
var DIAG_DATA=null;
var DIAG_EXPANDED={};

// 填充模型下拉框
function populateDiagModels(){
  var sel=$('diagModelSelect');
  var current=sel.value;
  sel.innerHTML='<option value="">— 请选择模型 —</option>';
  for(var i=0;i<MODELS.length;i++){
    var m=MODELS[i];
    var phase=m.lifecycle_status?(' · '+m.lifecycle_status+' @ '+(m.diagnosis_time||'—')):'';
    var label=m.model_id+' ('+m.champion_version+') ['+h(m.started_at)+']'+phase;
    sel.innerHTML+='<option value="'+h(m.run_id)+'">'+label+'</option>';
  }
  if(current) sel.value=current;
}

// 加载诊断数据
function loadDiagnosis(){
  var runId=$('diagModelSelect').value;
  if(!runId){$('diagContent').innerHTML='';$('diagStatus').textContent='';return}
  $('diagStatus').textContent='加载中...';
  $('diagContent').innerHTML='<div class="loading" style="padding:20px;text-align:center;color:var(--muted)">正在查询诊断数据...</div>';

  var xhr=new XMLHttpRequest();
  xhr.open('GET','/api/diagnosis/runs/by-monitoring/'+encodeURIComponent(runId));
  xhr.timeout=15000;
  xhr.onload=function(){
    if(xhr.status===200){
      try{
        var resp=JSON.parse(xhr.responseText);
        DIAG_DATA=resp.data||resp;
        renderDiagnosis(DIAG_DATA);
        $('diagStatus').textContent='';
      }catch(e){
        $('diagContent').innerHTML='<div class="card"><p class="red">解析诊断数据失败: '+e.message+'</p></div>';
        $('diagStatus').textContent='';
      }
    }else if(xhr.status===404){
      $('diagContent').innerHTML='<div class="card" style="text-align:center;padding:32px"><p>该模型尚未执行诊断</p><button class="btn" onclick="triggerDiagnosis()" style="margin-top:8px">🔬 触发诊断</button></div>';
      $('diagStatus').textContent='未诊断';
    }else{
      $('diagContent').innerHTML='<div class="card"><p class="red">请求失败: HTTP '+xhr.status+'</p></div>';
      $('diagStatus').textContent='';
    }
  };
  xhr.onerror=function(){
    $('diagContent').innerHTML='<div class="card"><p class="red">网络请求失败，请确认 API 服务正在运行</p></div>';
    $('diagStatus').textContent='';
  };
  xhr.send();
}

// 触发诊断
function triggerDiagnosis(){
  var runId=$('diagModelSelect').value;
  if(!runId)return;
  $('diagStatus').textContent='触发中...';
  var xhr=new XMLHttpRequest();
  xhr.open('POST','/api/diagnosis/trigger');
  xhr.setRequestHeader('Content-Type','application/json');
  xhr.timeout=30000;
  xhr.onload=function(){
    if(xhr.status===200){
      $('diagStatus').textContent='诊断完成，重新加载...';
      setTimeout(loadDiagnosis,500);
    }else{
      $('diagContent').innerHTML='<div class="card"><p class="red">触发失败: HTTP '+xhr.status+'</p><pre style="font-size:11px">'+h(xhr.responseText)+'</pre></div>';
      $('diagStatus').textContent='';
    }
  };
  xhr.onerror=function(){$('diagStatus').textContent='触发失败';};
  xhr.send(JSON.stringify({monitoring_run_id:runId}));
}

// 渲染诊断面板
function renderDiagnosis(d){
  if(!d||!d.run){$('diagContent').innerHTML='<div class="card"><p class="muted">无诊断数据</p></div>';return}

  var run=d.run,candidates=d.candidates||[],evidence=d.evidence||[],source=d.source||{},event=d.event||{},handoff=d.agent_handoff||{};

  // 按 rank_no 排序候选
  candidates.sort(function(a,b){return (a.rank_no||99)-(b.rank_no||99)});

  // 按 hypothesis_code (对应 root_cause_code) 分组证据
  var evByCand={};
  for(var i=0;i<evidence.length;i++){
    var e=evidence[i];
    // 优先用 candidate_id，回退到 hypothesis_code，再回退到 method_code
    var cid=e.candidate_id||e.hypothesis_code||'';
    if(!evByCand[cid])evByCand[cid]=[];
    evByCand[cid].push(e);
  }

  var primary=candidates.length>0?candidates[0]:null;
  var primaryEvidence=[];
  if(primary){
    primaryEvidence=evByCand[primary.candidate_id||'']||[];
    if(primaryEvidence.length===0){
      primaryEvidence=evByCand[primary.root_cause_code||'']||[];
    }
  }
  var primaryHasSupport=primaryEvidence.some(function(ev){
    return ev.applicable!==false && ev.direction==='SUPPORT';
  });

  // 动作样式
  var action=run.recommended_action||'';
  var actionClass='';
  if(action==='MODEL_ITERATION')actionClass='action-iterate';
  else if(action==='DATA_REPAIR'||action==='PIPELINE_REPAIR')actionClass='action-repair';
  else if(action==='CONTINUE_OBSERVATION'||action==='NO_ACTION')actionClass='action-observe';
  else actionClass='action-manual';

  var statusLabel=run.status||'';
  var statusClass=statusLabel==='COMPLETED'?'tag-ok':statusLabel==='RUNNING'?'tag-warn':'tag-err';

  var dimName={FEATURE:'特征维度',MODEL:'模型维度',DATA:'数据维度',BUSINESS:'业务维度'};
  var dim=run.primary_root_cause_dimension||'';
  var dimLabel=dimName[dim]||dim||'—';

  var html='';

  // ── Hero Card ──
  html+='<div class="diag-hero" style="background:'+(primary?'var(--blue-bg)':'var(--bg)')+'">';
  if(primary){
    html+='<div class="score blue">'+(run.primary_root_cause_score!=null?run.primary_root_cause_score.toFixed(2):'—')+'</div>';
    html+='<div class="meta">';
    html+='<div class="rc">首位根因候选: <span class="blue">'+h(run.primary_root_cause_code)+'</span> <span class="muted">('+dimLabel+')</span></div>';
    html+='<div style="margin-top:4px">';
    html+='<span class="action-tag '+actionClass+'">'+h(action)+'</span> ';
    html+='<span class="tag '+statusClass+'">'+h(statusLabel)+'</span> ';
    if(!primaryHasSupport){
      html+='<span class="tag tag-warn">证据不足，仅为候选</span> ';
    }
    html+='<span class="muted">need_iteration: '+(run.need_iteration!=null?run.need_iteration:'—')+'</span>';
    html+='</div>';
    if(source.model_id){
      var srcCodes=(source.diagnosis_alert_codes||[]).join(', ')||'—';
      var largestDrop=source.largest_drop!=null?Number(source.largest_drop).toFixed(4):'—';
      html+='<div style="margin-top:5px;font-size:11px"><b>诊断输入:</b> '+
        h(source.model_id)+' / '+h(source.model_version||'')+
        ' · 触发 '+h(srcCodes)+' × '+h(source.alert_count||0)+
        ' · 最大降幅 '+h(largestDrop)+'</div>';
    }
    if(event.event_id){
      var eventLabel=event.status==='WAITING_AGENT_DECISION'?'等待Agent决策':(event.status||'—');
      html+='<div style="margin-top:5px;font-size:12px"><b>诊断时间点:</b> '+
        h(event.event_time||'—')+' · <b>当前阶段:</b> <span class="tag tag-warn">'+
        h(eventLabel)+'</span> · <b>Agent:</b> '+
        (handoff.agent_connected?'已接入':'未接入，本流程在此停止')+'</div>';
    }
    html+='<div class="muted" style="margin-top:4px;font-size:11px">Diagnosis Run: '+h(run.diagnosis_run_id)+'</div>';
    html+='</div>';
  }else{
    html+='<div class="meta" style="padding:8px">';
    html+='<div class="rc">状态: <span class="'+(statusClass==='tag-ok'?'green':'red')+'">'+h(statusLabel)+'</span></div>';
    html+='<div class="muted" style="font-size:11px">Diagnosis Run: '+h(run.diagnosis_run_id)+'</div>';
    html+='</div>';
  }
  html+='</div>';

  if(!candidates.length){
    html+='<div class="card" style="text-align:center;padding:24px"><p class="muted">无候选根因 — 告警数可能为 0</p></div>';
    $('diagContent').innerHTML=html;
    return;
  }

  // ── 根因排序表 ──
  html+='<div class="detail-card"><h5>根因排序 (PathRanker 融合: KG权重 × 0.6 + 证据均值 × 0.4)</h5>';
  html+='<table><thead><tr><th>#</th><th>根因</th><th>维度</th><th>KG权重</th><th>排序分</th><th>证据条数</th><th>证据分布</th><th>主要</th></tr></thead><tbody>';

  for(var i=0;i<candidates.length;i++){
    var c=candidates[i];
    var cid=c.candidate_id||c.root_cause_code||'';
    var evs=evByCand[cid]||[];
    // 如果 candidate_id 匹配失败，尝试用 root_cause_code 匹配（兼容旧数据）
    if(evs.length===0 && c.root_cause_code){
      evs=evByCand[c.root_cause_code]||[];
    }

    // 按类型统计证据
    var typeCount={D:0,R:0,C:0,T:0,I:0};
    var supportCount=0,againstCount=0,neutralCount=0,naCount=0;
    for(var j=0;j<evs.length;j++){
      var t=evs[j].evidence_type||'';
      if(typeCount[t]!==undefined)typeCount[t]++;
      var dir=evs[j].direction||'';
      if(evs[j].applicable===false)naCount++;
      else if(dir==='SUPPORT')supportCount++;
      else if(dir==='AGAINST')againstCount++;
      else neutralCount++;
    }

    var isPrimary=c.is_primary||c.rank_no===1;
    var rowId='rc-'+i;
    var expanded=DIAG_EXPANDED[rowId];

    html+='<tr class="rc-row'+(isPrimary?' expanded':'')+'" onclick="toggleDiagRC(\\''+rowId+'\\',this)" style="'+(isPrimary?'background:rgba(26,115,232,.04)':'')+'">';
    html+='<td><span class="expand-arrow'+(expanded?' rotated':'')+'">▶</span> '+(c.rank_no||'?')+'</td>';
    html+='<td><b>'+(isPrimary?'★ ':'')+h(c.root_cause_code)+'</b></td>';
    html+='<td><span class="tag tag-ok">'+h(c.dimension_code)+'</span></td>';
    html+='<td class="num">'+(c.effective_weight_snapshot!=null?c.effective_weight_snapshot.toFixed(3):'—')+'</td>';
    html+='<td class="num'+(isPrimary?' blue':'')+'" style="font-weight:'+(isPrimary?'700':'400')+'">'+(c.ranked_score!=null?c.ranked_score.toFixed(4):'—')+'</td>';
    html+='<td class="num">'+evs.length+'</td>';
    html+='<td>';
    // 证据条形图
    if(evs.length>0){
      html+='<div class="ev-bar">';
      var tkeys=['D','R','C','T','I'];
      for(var ti=0;ti<tkeys.length;ti++){
        if(typeCount[tkeys[ti]]>0){
          var pct=typeCount[tkeys[ti]]/evs.length*100;
          html+='<span class="ev-seg support" style="width:'+Math.max(pct*2,4)+'px"'+
            ' title="'+tkeys[ti]+': '+typeCount[tkeys[ti]]+'/'+evs.length+'"></span>';
        }
      }
      html+=' <span style="font-size:10px;color:var(--muted)">S:'+supportCount+' A:'+againstCount+' N:'+neutralCount+' N/A:'+naCount+'</span>';
      html+='</div>';
    }
    html+='</td>';
    html+='<td>'+(isPrimary?'<span class="tag tag-ok">★ PRIMARY</span>':'')+'</td>';
    html+='</tr>';

    // 展开行 — 证据详情
    if(expanded||isPrimary){
      html+='<tr class="detail-row"><td colspan="8"><div class="detail-panel" style="padding:8px 16px">';
      if(evs.length===0){
        html+='<p class="muted">无验证器产出证据</p>';
      }else{
        // 按 D/R/C/T/I 顺序展示
        var typeOrder=['D','R','C','T','I'];
        var typeNames={D:'数据/分布',R:'反事实修复',C:'关联/回归',T:'时序优先',I:'重要性依赖'};
        for(var ti=0;ti<typeOrder.length;ti++){
          var tt=typeOrder[ti];
          var tevs=evs.filter(function(e){return e.evidence_type===tt});
          if(tevs.length===0)continue;
          html+='<div class="detail-card"><h5><span class="ev-badge ev-'+tt+'">'+tt+'</span> '+typeNames[tt]+' ('+tevs.length+' 条证据)</h5>';
          html+='<table style="font-size:11px"><thead><tr><th>方法</th><th>适用?</th><th>方向</th><th>得分</th><th>置信度</th><th>详情</th></tr></thead><tbody>';
          for(var ei=0;ei<tevs.length;ei++){
            var ev=tevs[ei];
            var dirIcon=ev.direction==='SUPPORT'?'✅':ev.direction==='AGAINST'?'❌':'➖';
            var dirClass=ev.direction==='SUPPORT'?'green':ev.direction==='AGAINST'?'red':'muted';
            var detJson='';
            try{
              var det=typeof ev.evidence_detail_json==='string'?JSON.parse(ev.evidence_detail_json):ev.evidence_detail_json;
              var detParts=[];
              if(det&&det.message)detParts.push(h(det.message));
              if(det&&det.alert_metric){
                detParts.push('<b>触发告警:</b> '+h(det.alert_metric));
              }
              if(det&&det.target_metric_code){
                detParts.push('<b>绑定指标:</b> '+h(det.target_metric_code));
              }
              if(det&&det.per_window_delta){
                var deltaText=Object.keys(det.per_window_delta).sort().map(function(w){
                  return w+': '+det.per_window_delta[w];
                }).join('；');
                detParts.push('<b>窗口Δ:</b> '+h(deltaText));
              }
              if(det&&det.per_window_degradation){
                var degradationText=Object.keys(det.per_window_degradation).sort().map(function(w){
                  return w+': '+det.per_window_degradation[w];
                }).join('；');
                detParts.push('<b>退化量:</b> '+h(degradationText));
              }
              detJson=detParts.length?detParts.join('<br>'):h(JSON.stringify(det||{}).substring(0,160));
            }catch(ex){detJson=h(ev.evidence_detail_json).substring(0,120)}
            html+='<tr>'+
              '<td class="val">'+h(ev.method_code)+'</td>'+
              '<td>'+(ev.applicable?'<span class="tag tag-ok">✓</span>':'<span class="tag tag-warn">N/A</span>')+'</td>'+
              '<td class="'+dirClass+'">'+dirIcon+' '+h(ev.direction)+'</td>'+
              '<td class="num">'+(ev.normalized_score!=null?ev.normalized_score.toFixed(4):'—')+'</td>'+
              '<td><span class="tag tag-ok">'+h(ev.confidence_level)+'</span></td>'+
              '<td style="max-width:300px;font-size:10px;white-space:normal;word-break:break-all">'+detJson+'</td>'+
              '</tr>';
          }
          html+='</tbody></table></div>';
        }
      }
      html+='</div></td></tr>';
    }
  }
  html+='</tbody></table></div>';

  $('diagContent').innerHTML=html;
}

function toggleDiagRC(rowId,tr){
  if(DIAG_EXPANDED[rowId]){delete DIAG_EXPANDED[rowId]}else{DIAG_EXPANDED[rowId]=true}
  renderDiagnosis(DIAG_DATA);
}

// Tab 切换时填充模型列表
var origTabClick=document.getElementById('main-tabs').onclick;
document.getElementById('main-tabs').addEventListener('click',function(e){
  if(!e.target.classList.contains('tab-btn'))return;
  if(e.target.dataset.tab==='tab-diagnosis'){
    setTimeout(function(){
      if($('diagModelSelect').options.length<=1)populateDiagModels();
    },100);
  }
});

try{
  renderRuns();
  populateDiagModels();
  document.getElementById('loadBanner').textContent='RENDER OK';
}catch(e){
  document.getElementById('loadBanner').textContent='JS ERROR: '+e.message;
  document.getElementById('loadBanner').style.background='var(--red)';
}
</script>
</body></html>"""
    return html_template.replace("__JSON__", embedded_json)
