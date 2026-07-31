
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

  var html='<tr class="detail-row"><td colspan="19"><div class="detail-panel">';

  // Card 1: 三窗口性能对比
  html+='<div class="detail-card"><h5>📊 性能对比 — '+h(model.model_id)+'</h5>';
  html+='<table><tr><th>指标</th><th>W1</th><th>W2</th><th>W3</th><th>W0基线</th><th>Δ(W3-W0)</th></tr>';
  ['AUC','KS','PR_AUC','BAD_RECALL','BRIER','ECE'].forEach(function(code){
    var w1=WM['W1']||{},w2=WM['W2']||{},w3=WM['W3']||{};
    var a=v(w1,code),b=v(w2,code),c=v(w3,code),base=bl(code);
    var d=c!=null&&base!=null?c-base:null;
    var dc=d!=null?(code==='BRIER'||code==='ECE'?(d>0?'red':'green'):(d<0?'red':'green')):'';
    html+='<tr><td><b>'+code+'</b></td>'+
      '<td class="num">'+f(a)+'</td><td class="num">'+f(b)+'</td><td class="num blue">'+f(c)+'</td>'+
      '<td class="num muted">'+f(base)+'</td>'+
      '<td class="num '+dc+'">'+(d!=null?(d>0?'+':'')+d.toFixed(4):'-')+'</td></tr>';
  });
  html+='</table></div>';

  // Card 2: 运行时指标三窗口对比
  html+='<div class="detail-card"><h5>🔬 监测指标</h5>';
  html+='<table><tr><th>指标</th><th>W1</th><th>W2</th><th>W3</th><th>W0基线</th></tr>';
  [{code:'BAD_RATE',lbl:'BAD_RATE'},{code:'SAMPLE_SIZE',lbl:'SAMPLE_SIZE',isInt:true},
   {code:'PREDICTION_MEAN',lbl:'PRED_MEAN'},{code:'SCORE_PSI',lbl:'SCORE_PSI'},
   {code:'FEATURE_PSI',lbl:'FEATURE_PSI'}].forEach(function(x){
    var w1=WM['W1']||{},w2=WM['W2']||{},w3=WM['W3']||{};
    var a=v(w1,x.code),b=v(w2,x.code),c=v(w3,x.code),base=bl(x.code);
    html+='<tr><td><b>'+x.lbl+'</b></td>'+
      '<td class="num">'+(x.isInt?(a!=null?a.toFixed(0):'-'):f(a))+'</td>'+
      '<td class="num">'+(x.isInt?(b!=null?b.toFixed(0):'-'):f(b))+'</td>'+
      '<td class="num blue">'+(x.isInt?(c!=null?c.toFixed(0):'-'):f(c))+'</td>'+
      '<td class="num muted">'+(x.isInt?(base!=null?base.toFixed(0):'-'):f(base))+'</td></tr>';
  });
  html+='</table></div>';

  // Card 3: 模型信息
  html+='<div class="detail-card"><h5>📋 模型信息</h5>';
  html+='<table>';
  [{k:'Run ID',v:model.run_id},{k:'Model',v:model.model_id},
   {k:'Version',v:model.champion_version},{k:'Status',v:model.status},
   {k:'Alerts',v:model.alert_count},{k:'Windows',v:'W1 / W2 / W3'}].forEach(function(rr){
    html+='<tr><td class="muted">'+rr.k+'</td><td>'+h(rr.v)+'</td></tr>';
  });
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
