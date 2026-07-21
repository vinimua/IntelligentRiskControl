
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
    var catNames={core:'📊 Core 指标',distribution:'📈 Distribution',aggregate:'📐 Aggregate'};
    html+='<div class="detail-card"><h5>'+catNames[cat]+'</h5>';
    html+='<table><tr><th>Metric</th><th>W1</th><th>W2</th><th>W3</th><th>Baseline</th><th>Delta</th><th>Type</th></tr>';
    for(var ai=0;ai<catItems.length;ai++){
      var m=catItems[ai];
      var av=v(w1,m.code),bv=v(w2,m.code),cv=v(w3,m.code),base=bl(m.code);
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
  html+='<div class="detail-card"><h5>📋 Meta</h5>';
  html+='<table><tr><th>Metric</th><th>W1</th><th>W2</th><th>W3</th></tr>';
  [{code:'SAMPLE_SIZE',lbl:'SAMPLE_SIZE',isInt:true},{code:'SCHEMA_CONSISTENCY',lbl:'SCHEMA_CONSISTENCY',isSchema:true}].forEach(function(m){
    var av=v(w1,m.code),bv=v(w2,m.code),cv=v(w3,m.code);
    html+='<tr><td><b>'+m.lbl+'</b></td>'+
      '<td class="num">'+(m.isInt?(av!=null?av.toFixed(0):'-'):m.isSchema?(av==1?'OK':'-'):f(av))+'</td>'+
      '<td class="num">'+(m.isInt?(bv!=null?bv.toFixed(0):'-'):m.isSchema?(bv==1?'OK':'-'):f(bv))+'</td>'+
      '<td class="num blue">'+(m.isInt?(cv!=null?cv.toFixed(0):'-'):m.isSchema?(cv==1?'OK':'-'):f(cv))+'</td></tr>';
  });
  html+='</table></div>';

  // Drift TOP-10 (per window)
  for(var wi=0;wi<3;wi++){
    var wid=['W1','W2','W3'][wi];
    var topD=(DRIFT[wid]||[]).slice(0,10);
    if(!topD.length)continue;
    html+='<div class="detail-card"><h5>🔬 Drift TOP-10 — '+wid+'</h5>';
    html+='<table><tr><th>#</th><th>Feature</th><th>PSI</th><th>JS</th><th>KS</th><th>W-dist</th></tr>';
    for(var di=0;di<topD.length;di++){
      var dm=topD[di];
      var pc=dm.psi>0.25?'red':dm.psi>0.1?'yellow':'green';
      html+='<tr><td>'+(di+1)+'</td><td>'+dm.feature+'</td>'+
        '<td class="num '+pc+'">'+f(dm.psi)+'</td>'+
        '<td class="num">'+f(dm.js)+'</td><td class="num">'+f(dm.ks)+'</td>'+
        '<td class="num">'+f(dm.wasserstein)+'</td></tr>';
    }
    html+='</table></div>';
  }

  // Quality TOP-10 (per window)
  for(var wi=0;wi<3;wi++){
    var wid=['W1','W2','W3'][wi];
    var topQ=(QTOP[wid]||[]).slice(0,10);
    if(!topQ.length)continue;
    html+='<div class="detail-card"><h5>✅ Quality TOP-10 — '+wid+'</h5>';
    html+='<table><tr><th>#</th><th>Feature</th><th>DQ</th><th>Missing</th><th>Outlier</th><th>Default</th><th>Flag</th></tr>';
    for(var qi=0;qi<topQ.length;qi++){
      var qm=topQ[qi];
      var fc=qm.flag==='ALERT'?'tag-err':qm.flag==='WARN'?'tag-warn':'tag-ok';
      html+='<tr><td>'+(qi+1)+'</td><td>'+qm.feature+'</td>'+
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
