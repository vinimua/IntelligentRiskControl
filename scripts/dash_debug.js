
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
    tbody.innerHTML+='<tr class="'+(exp?'expanded':'')+'" onclick="toggleRow(\''+rid+'\',this)" style="cursor:pointer">'+
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
  document.getElementById('runsBody').insertAdjacentHTML('beforeend','<tr class="detail-row"><td colspan="19"><div class="detail-panel"><div class="loading">Loading metrics for '+h(model.model_id)+'...</div></div></td></tr>');

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
    var label=m.model_id+' ('+m.champion_version+') ['+h(m.started_at)+']';
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
  xhr.open('GET','http://localhost:8000/api/diagnosis/runs/by-monitoring/'+encodeURIComponent(runId));
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
  xhr.open('POST','http://localhost:8000/api/diagnosis/trigger');
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

  var run=d.run,candidates=d.candidates||[],evidence=d.evidence||[];

  // 按 rank_no 排序候选
  candidates.sort(function(a,b){return (a.rank_no||99)-(b.rank_no||99)});

  // 按 candidate_id 分组证据
  var evByCand={};
  for(var i=0;i<evidence.length;i++){
    var e=evidence[i],cid=e.candidate_id||'';
    if(!evByCand[cid])evByCand[cid]=[];
    evByCand[cid].push(e);
  }

  var primary=candidates.length>0?candidates[0]:null;

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
    html+='<div class="rc">主要根因: <span class="blue">'+h(run.primary_root_cause_code)+'</span> <span class="muted">('+dimLabel+')</span></div>';
    html+='<div style="margin-top:4px">';
    html+='<span class="action-tag '+actionClass+'">'+h(action)+'</span> ';
    html+='<span class="tag '+statusClass+'">'+h(statusLabel)+'</span> ';
    html+='<span class="muted">need_iteration: '+(run.need_iteration!=null?run.need_iteration:'—')+'</span>';
    html+='</div>';
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
    var cid=c.candidate_id||'';
    var evs=evByCand[cid]||[];

    // 按类型统计证据
    var typeCount={D:0,R:0,C:0,T:0,I:0};
    var supportCount=0,againstCount=0,neutralCount=0;
    for(var j=0;j<evs.length;j++){
      var t=evs[j].evidence_type||'';
      if(typeCount[t]!==undefined)typeCount[t]++;
      var dir=evs[j].direction||'';
      if(dir==='SUPPORT')supportCount++;
      else if(dir==='AGAINST')againstCount++;
      else neutralCount++;
    }

    var isPrimary=c.is_primary||c.rank_no===1;
    var rowId='rc-'+i;
    var expanded=DIAG_EXPANDED[rowId];

    html+='<tr class="rc-row'+(isPrimary?' expanded':'')+'" onclick="toggleDiagRC(\''+rowId+'\',this)" style="'+(isPrimary?'background:rgba(26,115,232,.04)':'')+'">';
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
      html+=' <span style="font-size:10px;color:var(--muted)">S:'+supportCount+' A:'+againstCount+' N:'+neutralCount+'</span>';
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
              if(det&&det.message)detJson=det.message;
              else detJson=JSON.stringify(det||{}).substring(0,120);
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
