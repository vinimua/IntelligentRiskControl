"""监控仪表盘 — 按设计规范重写。"""

import json
from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["dashboard"])


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request):
    from ..database import get_db
    from ..repositories.monitoring_repo import MonitoringRepo

    db_gen = get_db()
    db = await db_gen.__anext__()
    try:
        repo = MonitoringRepo(db)
        runs = await repo.list_runs(limit=50)
    finally:
        await db_gen.aclose()

    # 构建嵌入数据——每个模型带指标摘要
    models_data = []
    for r in runs:
        rid = str(r.get("monitoring_run_id", ""))
        # 加载该 run 的核心指标摘要
        run_metrics = await repo.get_metrics(rid)
        summary = {}
        for m in run_metrics:
            code = m.get("metric_code", "")
            wid = (m.get("metric_detail") or {}).get("window_id", "?")
            if code in ("AUC","KS","BAD_RATE","PREDICTION_MEAN","SCORE_PSI","FEATURE_PSI","SAMPLE_SIZE"):
                summary.setdefault(wid, {})[code] = m.get("current_value")
                base_val = m.get("baseline_value")
                if base_val is not None:
                    summary.setdefault(wid, {})[code + "_BASE"] = base_val
            if code == "SAMPLE_SIZE":
                summary.setdefault(wid, {})["BAD_COUNT"] = (m.get("metric_detail") or {}).get("bad_count", 0)

        models_data.append({
            "run_id": rid,
            "model_id": str(r.get("model_id", "")),
            "champion_version": str(r.get("champion_version", "")),
            "status": str(r.get("overall_status", "")),
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
        wid = (m.get("metric_detail") or {}).get("window_id", "?")
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
        wid = md.get("window_id", "?")
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
          <th>AUC(W1)</th><th>AUC(W2)</th><th>AUC(W3)</th><th>ΔAUC</th>
          <th>KS(W1)</th><th>KS(W2)</th><th>KS(W3)</th><th>ΔKS</th>
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

    var exp=EXPANDED[rid];
    tbody.innerHTML+='<tr class="'+(exp?'expanded':'')+'" onclick="toggleRow(\\''+rid+'\\',this)" style="cursor:pointer">'+
      '<td><span class="expand-arrow'+(exp?' rotated':'')+'">▶</span></td>'+
      '<td class="wide muted">'+h(rid)+'</td>'+
      '<td><b>'+h(r.model_id)+'</b></td>'+
      '<td class="muted">'+h(r.champion_version)+'</td>'+
      '<td><span class="tag tag-ok">'+h(r.status)+'</span></td>'+
      '<td class="'+(r.alert_count>0?'red':'')+'">'+r.alert_count+'</td>'+
      '<td><span class="tag '+sc+'">'+(sev||'NORMAL')+'</span></td>'+
      '<td class="muted">'+h(r.started_at)+'</td>'+
      '<td class="num">'+f(auc1)+'</td><td class="num">'+f(auc2)+'</td><td class="num blue">'+f(auc3)+'</td>'+
      '<td class="num '+(dAUC!=null&&dAUC<0?'green':'red')+'">'+(dAUC!=null?(dAUC>0?'+':'')+dAUC.toFixed(4):'-')+'</td>'+
      '<td class="num">'+f(ks1)+'</td><td class="num">'+f(ks2)+'</td><td class="num blue">'+f(ks3)+'</td>'+
      '<td class="num '+(dKS!=null&&dKS<0?'green':'red')+'">'+(dKS!=null?(dKS>0?'+':'')+dKS.toFixed(4):'-')+'</td>'+
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
        var wid=(m.metric_detail&&m.metric_detail.window_id)||'?';
        var code=m.metric_code;
        if(!wm[wid])wm[wid]={};
        if(!wm[wid][code])wm[wid][code]={};
        wm[wid][code]={
          current:m.current_value, baseline:m.baseline_value,
          delta:m.delta,
          score_type:(m.metric_detail||{}).score_type||'',
          status:(m.metric_detail||{}).status||'',
        };
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
    html+='<table><tr><th>Metric</th><th>W1</th><th>W2</th><th>W3</th><th>Baseline</th><th>Delta</th><th>Type</th></tr>';
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
  html+='<table><tr><th>Metric</th><th>W1</th><th>W2</th><th>W3</th></tr>';
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
  if(SORT==='auc')list.sort(function(a,b){var av=v(WM['W3']||{},'AUC'),bv=v(WM['W3']||{},'AUC');return(bv||0)-(av||0)});
  else if(SORT==='ks')list.sort(function(a,b){var av=v(WM['W3']||{},'KS'),bv=v(WM['W3']||{},'KS');return(bv||0)-(av||0)});
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

try{
  renderRuns();
  document.getElementById('loadBanner').textContent='RENDER OK';
}catch(e){
  document.getElementById('loadBanner').textContent='JS ERROR: '+e.message;
  document.getElementById('loadBanner').style.background='var(--red)';
}
</script>
</body></html>"""
    return html_template.replace("__JSON__", embedded_json)
