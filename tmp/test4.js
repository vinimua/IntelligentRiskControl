
document.getElementById('loadBanner').textContent = 'JS STARTED';
var DATA = JSON.parse(document.getElementById('dash-data').textContent);
var MODELS = DATA.models || [];
var WM = DATA.window_metrics || {};
var DRIFT = DATA.drift_top || [];
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
    var w1=WM['W1']||{},w2=WM['W2']||{},w3=WM['W3']||{};
    var auc1=v(w1,'AUC'),auc2=v(w2,'AUC'),auc3=v(w3,'AUC');
    var ks1=v(w1,'KS'),ks2=v(w2,'KS'),ks3=v(w3,'KS');
    var auc0=bl('AUC');
    var dAUC=auc0!=null&&auc3!=null?auc3-auc0:null;
    var dKS=bl('KS')!=null&&ks3!=null?ks3-bl('KS'):null;
    var badRate=v(w3,'BAD_RATE'),predMean=v(w3,'PREDICTION_MEAN');
    var psi=(w3['SCORE_PSI']||{}).current;
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

function renderDetail(rid,tr){
  var model=null;
  for(var i=0;i<MODELS.length;i++){if(MODELS[i].run_id===rid){model=MODELS[i];break}}
  if(!model)return;
  var w1=WM['W1']||{},w2=WM['W2']||{},w3=WM['W3']||{};

  var html='<tr class="detail-row"><td colspan="19"><div class="detail-panel">';

  // Card 1: 排序能力 + 校准（所有模型级指标）
  html+='<div class="detail-card"><h5>📊 模型级指标 — '+h(model.model_id)+' / '+h(model.champion_version)+'</h5>';
  html+='<table><tr><th>指标</th><th>分数类型</th><th>W1</th><th>W2</th><th>W3</th><th>W0基线</th><th>Δ(W3−W0)</th></tr>';

  var allMetrics=[
    {code:'AUC',lbl:'AUC',hi:true},{code:'KS',lbl:'KS',hi:true},
    {code:'PR_AUC',lbl:'PR_AUC',hi:true},{code:'BAD_RECALL',lbl:'BAD_RECALL',hi:true},
    {code:'BRIER',lbl:'BRIER'},{code:'ECE',lbl:'ECE'},
    {code:'BAD_RATE',lbl:'BAD_RATE'},{code:'BAD_RATE_DELTA',lbl:'BAD_RATE_DELTA'},
    {code:'PERFORMANCE_DROP_MAX',lbl:'PERF_DROP_MAX'},
    {code:'PREDICTION_MEAN',lbl:'PRED_MEAN'},{code:'PREDICTION_STD',lbl:'PRED_STD'},
    {code:'PREDICTION_MIN',lbl:'PRED_MIN'},{code:'PREDICTION_MAX',lbl:'PRED_MAX'},
    {code:'SCORE_PSI',lbl:'SCORE_PSI'},{code:'FEATURE_PSI',lbl:'FEATURE_PSI'},
    {code:'SAMPLE_SIZE',lbl:'SAMPLE_SIZE',isInt:true},{code:'SCHEMA_CONSISTENCY',lbl:'SCHEMA',isSchema:true}
  ];

  for(var ai=0;ai<allMetrics.length;ai++){
    var m=allMetrics[ai];
    var av=v(w1,m.code),bv=v(w2,m.code),cv=v(w3,m.code),base=bl(m.code);
    var d=cv!=null&&base!=null?cv-base:null;
    var hi=m.hi===true;
    var dc='';
    if(d!=null&&m.code!=='SCHEMA_CONSISTENCY'){
      dc=hi?(d<0?'red':'green'):(d>0?'red':'green');
    }
    var st=vst(w3,m.code)||((w3[m.code]||{}).score_type)||'-';
    if(!st||st==='-')st=(w3[m.code]||{}).score_type||'-';

    html+='<tr><td><b>'+m.lbl+'</b></td><td style="font-size:10px;color:#8b949e">'+st+'</td>'+
      '<td class="num">'+(m.isInt?(av!=null?av.toFixed(0):'-'):m.isSchema?(av==1?'OK':'-'):f(av))+'</td>'+
      '<td class="num">'+(m.isInt?(bv!=null?bv.toFixed(0):'-'):m.isSchema?(bv==1?'OK':'-'):f(bv))+'</td>'+
      '<td class="num blue">'+(m.isInt?(cv!=null?cv.toFixed(0):'-'):m.isSchema?(cv==1?'OK':'-'):f(cv))+'</td>'+
      '<td class="num muted">'+(m.isInt?(base!=null?base.toFixed(0):'-'):m.isSchema?'-':f(base))+'</td>'+
      '<td class="num '+dc+'">'+(d!=null?(d>0?'+':'')+d.toFixed(4):'-')+'</td></tr>';
  }
  html+='</table></div>';

  // Card 2: 特征漂移 TOP-10 + 数据质量 TOP-10
  html+='<div class="detail-card"><h5>🔬 特征漂移 TOP-10（W3, 按 PSI 排序）</h5>';
  html+='<table><tr><th>#</th><th>特征</th><th>PSI</th><th>JS</th><th>KS</th><th>W-dist</th></tr>';
  var driftAll=[];
  for(var code in w3){
    var dm=(w3[code]||{});
    if(code.indexOf('D_PSI')===0&&dm.current!=null){driftAll.push({feature:code.replace('D_PSI',''),psi:dm.current})}
  }
  // Actually use the drift data from the metric_detail
  // The WM stores drift metrics as D_PSI, D_JS_DIVERGENCE etc with feature_name in metric_detail
  // But our window_metrics structure groups by metric_code, not feature_name.
  // So D_PSI's value is stored per-feature? No - it's stored as separate rows.
  // window_metrics[wid][code] aggregates same code for same window, so only the LAST feature's value is kept!
  // This is a data structure issue. Use DRIFT data instead.
  var topDrift=DATA.drift_top||[];
  if(topDrift.length>0){
    for(var di=0;di<Math.min(topDrift.length,10);di++){
      var dm=topDrift[di];
      var pc=dm.psi>0.25?'red':dm.psi>0.1?'yellow':'green';
      html+='<tr><td>'+(di+1)+'</td><td>'+dm.feature+'</td>'+
        '<td class="num '+pc+'">'+f(dm.psi)+'</td>'+
        '<td class="num">-</td><td class="num">-</td><td class="num">-</td></tr>';
    }
  }else{
    html+='<tr><td colspan="6" class="muted">点击展开以加载完整漂移数据</td></tr>';
  }
  html+='</table></div>';

  html+='</div></td></tr>';
  document.getElementById('runsBody').insertAdjacentHTML('beforeend',html);
}

function v(win,code){var m=win&&win[code];return m&&m.current!=null?m.current:null}
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
