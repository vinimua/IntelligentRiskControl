
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
    var r=filtered[i],rid=r.run_id||'',wid=(r.monitor_window_id||'W3').replace('W','');
    var w0=WM['W0']||{},ww=WM['W3']||{};
    var auc0=v(w0,'AUC'),auc3=v(ww,'AUC'),ks0=v(w0,'KS'),ks3=v(ww,'KS');
    var dAUC=auc0!=null&&auc3!=null?auc3-auc0:null;
    var dKS=ks0!=null&&ks3!=null?ks3-ks0:null;
    var badRate=v(ww,'BAD_RATE'),predMean=v(ww,'PREDICTION_MEAN');
    var psi=(ww['SCORE_PSI']||ww['PREDICTION_PSI']||{}).current;
    var sev=r.max_severity||'';var sc=sev==='CRITICAL'||sev==='HIGH'?'tag-err':sev==='WARNING'?'tag-warn':'tag-ok';

    var exp=EXPANDED[rid];
    tbody.innerHTML+='<tr class="'+(exp?'expanded':'')+'" onclick="toggleRow(''+rid+'',this)" style="cursor:pointer">'+
      '<td><span class="expand-arrow'+(exp?' rotated':'')+'">▶</span></td>'+
      '<td class="wide muted">'+h(rid)+'</td>'+
      '<td><b>'+h(r.model_id)+'</b></td>'+
      '<td class="muted">'+h(r.champion_version)+'</td>'+
      '<td><span class="tag tag-ok">'+h(r.status)+'</span></td>'+
      '<td class="'+(r.alert_count>0?'red':'')+'">'+r.alert_count+'</td>'+
      '<td><span class="tag '+sc+'">'+(sev||'NORMAL')+'</span></td>'+
      '<td class="muted">'+h(r.started_at)+'</td>'+
      '<td class="num">'+f(auc0)+'</td><td class="num blue">'+f(auc3)+'</td>'+
      '<td class="num '+(dAUC!=null&&dAUC<0?'green':'red')+'">'+(dAUC!=null?(dAUC>0?'+':'')+dAUC.toFixed(4):'-')+'</td>'+
      '<td class="num">'+f(ks0)+'</td><td class="num blue">'+f(ks3)+'</td>'+
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
  var exp=EXPANDED[rid];
  // Find the model
  var model=null;
  for(var i=0;i<MODELS.length;i++){if(MODELS[i].run_id===rid){model=MODELS[i];break}}
  if(!model)return;
  var w3=WM['W3']||{},w0=WM['W0']||{};

  var html='<tr class="detail-row"><td colspan="18"><div class="detail-panel">';

  // Card 1: 性能指标
  html+='<div class="detail-card"><h5>📊 性能指标 — '+h(model.model_id)+' / '+h(model.champion_version)+'</h5>';
  html+='<table><tr><th>指标</th><th>当前值</th><th>基线值</th><th>变化</th><th>状态</th></tr>';
  ['AUC','KS','PR_AUC','BAD_RECALL','BRIER','ECE'].forEach(function(code){
    var c=v(w3,code),b=v(w0,code),d=c!=null&&b!=null?c-b:null,st=vst(w3,code);
    html+='<tr><td><b>'+code+'</b></td><td class="num">'+f(c)+'</td><td class="num muted">'+f(b)+'</td>'+
      '<td class="num '+(d!=null?(code==='BRIER'||code==='ECE'?(d>0?'red':'green'):(d<0?'red':'green')):'')+'">'+(d!=null?(d>0?'+':'')+d.toFixed(4):'-')+'</td>'+
      '<td><span class="tag '+(c!=null?'tag-ok':'tag-warn')+'">'+(c!=null?'COMPUTED':'BASELINE')+'</span></td></tr>';
  });
  html+='</table></div>';

  // Card 2: 运行时监测指标
  html+='<div class="detail-card"><h5>🔬 运行时监测指标（W3 窗口）</h5>';
  html+='<table><tr><th>指标</th><th>値</th><th>基线</th><th>变化</th><th>状态</th></tr>';
  [{code:'BAD_RATE',lbl:'BAD_RATE'},{code:'SAMPLE_SIZE',lbl:'SAMPLE_SIZE'},
   {code:'PREDICTION_MEAN',lbl:'PREDICTION_MEAN'},{code:'SCORE_PSI',lbl:'SCORE_PSI'},
   {code:'FEATURE_PSI',lbl:'FEATURE_PSI'},{code:'SCHEMA_CONSISTENCY',lbl:'SCHEMA_CONSISTENCY'}].forEach(function(x){
    var c=v(w3,x.code),b=v(w0,x.code),d=c!=null&&b!=null?c-b:null;
    var cur=c!=null?(x.code==='SCHEMA_CONSISTENCY'?(c===1?'✅':'⚠️'):c.toFixed(4)):'-';
    html+='<tr><td><b>'+x.lbl+'</b></td><td class="num">'+cur+'</td><td class="num muted">'+f(b)+'</td>'+
      '<td class="num">'+(d!=null?(d>0?'+':'')+d.toFixed(4):'-')+'</td>'+
      '<td><span class="tag '+(c!=null?'tag-ok':'tag-warn')+'">'+(c!=null?'COMPUTED':'RUNTIME')+'</span></td></tr>';
  });
  html+='</table></div>';

  // Card 3: 模型信息
  html+='<div class="detail-card"><h5>📋 模型信息</h5>';
  html+='<table>';
  [{k:'Run ID',v:h(model.run_id)},{k:'Model',v:h(model.model_id)},
   {k:'Version',v:h(model.champion_version)},{k:'Status',v:h(model.status)},
   {k:'Alert Count',v:model.alert_count},{k:'Windows',v:model.monitor_window_id||'W1_W2_W3'}].forEach(function(r){
    html+='<tr><td class="muted">'+r.k+'</td><td>'+r.v+'</td></tr>';
  });
  html+='</table></div>';

  html+='</div></td></tr>';
  document.getElementById('runsBody').insertAdjacentHTML('beforeend',html);
}

function v(win,code){return (win[code]||{}).current!=null?(win[code]||{}).current:null}
function vst(win,code){return (win[code]||{}).score_type||''}
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
