import {runSim} from 'teardown-kit/runtime.js';
import {controls,solve,pinchGap,PRESETS,FINGERS} from './kin.js';
import {createStage} from './stage.js';
import {tour,duration} from './film.js';
const modes=['solid','tendon','skeleton','exploded'];
const names={solid:'Assembly',tendon:'Tendons',skeleton:'Skeleton',exploded:'Exploded'};
const pretty=s=>s[0].toUpperCase()+s.slice(1);
const $=(el,s)=>el.querySelector(s);
const $$=(el,s)=>Array.from(el.querySelectorAll(s));
const button=(group,value,label)=>'<button type="button" data-'+group+'="'+value+'">'+label+'</button>';
const slider=(id,label,min,max,step)=>'<label class="th-slider"><span>'+label+'<b id="'+id+'-value"></b></span><input type="range" id="'+id+'" min="'+min+'" max="'+max+'" step="'+step+'"></label>';
export function run(film=false){
 runSim({duration,fps:60,theme:'light',accent:'#007e83',hud:{enabled:false,title:'TENDON / 01',subtitle:'Robotic hand'},director:{enabled:false}},async({canvas,capture})=>{
  const app=document.getElementById('app');
  const root=document.createElement('div');root.id='th';
  root.innerHTML='<header class="th-header"><div class="th-brand"><i></i><div><div class="th-eyebrow">MECHANISM LAB / SERIES 01</div><h1>TENDON<span> / </span>01</h1></div></div><div class="th-sub">ANTHROPOMORPHIC HAND<br><strong>5 DIGITS · 20 JOINT AXES · 5 FLEXOR CABLES</strong></div><div class="th-status"><i></i><span id="status">AUTO TOUR</span></div></header>'+
  '<main class="th-main"><aside class="th-panel th-left"><div class="th-kicker">01 / COMMAND</div><h2>Actuation</h2><p>Set cable-driven posture and thumb opposition.</p><div class="th-group"><h3>GRASP PRESET</h3><div class="th-buttons">'+Object.keys(PRESETS).map(v=>button('preset',v,pretty(v))).join('')+'</div></div><div class="th-group">'+slider('grip','Grip',0,1,.01)+slider('spread','Finger spread',0,1,.01)+slider('oppose','Thumb opposition',0,1,.01)+'</div><div class="th-group th-trims"><h3>INDIVIDUAL CURL TRIM</h3>'+FINGERS.map((f,i)=>slider('trim'+i,f.name,-1,1,.01)).join('')+'</div><div class="th-group"><h3>INSPECT DIGIT</h3><div class="th-buttons">'+button('focus','all','All')+FINGERS.map(f=>button('focus',f.id,f.name)).join('')+'</div></div></aside>'+
  '<section class="th-viewport" id="viewport"><div class="th-viewtag"><span>LIVE MECHANISM / MM</span><b id="viewtag">ASSEMBLY</b></div><div class="th-orbit">DRAG ROTATE &nbsp; / &nbsp; SCROLL ZOOM</div><div class="th-scale"><i></i> SCHEMATIC MODEL</div><div class="th-cameras">'+button('camera','front','Front')+button('camera','three','3/4')+button('camera','back','Back')+button('camera','detail','Detail')+'</div></section>'+
  '<aside class="th-panel th-right"><div class="th-kicker">02 / TELEMETRY</div><h2>Flexor state</h2><p>Angles in degrees · take-up in mm · estimated tension in N.</p><div id="telemetry"></div><div class="th-gap"><span>INDEX ↔ THUMB TIP CENTERS</span><strong id="gap">—</strong><small>Distance only · no contact or collision solve</small></div><div class="th-equation"><b>ΔL = Σ rᵢ θᵢ</b><span>T = 1.2 + 0.38 ΔL</span><small>Illustrative effective spring / not calibrated</small></div></aside></main>'+
  '<footer class="th-footer"><div class="th-mode"><div class="th-kicker">03 / DISASSEMBLY</div><div class="th-buttons">'+modes.map(v=>button('view',v,names[v])).join('')+'</div><div class="th-legend"><i></i> FLEXOR &nbsp;<i></i> RETURN &nbsp;<i></i> FRAME</div></div><div class="th-explain"><div class="th-kicker" id="chapter"></div><strong id="explain"></strong><div class="th-progress"><i id="progress"></i></div></div><div class="th-tour">'+button('action','tour','Auto tour')+button('action','reset','Reset view')+'<small>Offline · kinematics V1.0</small></div></footer>';
  app.appendChild(root);const viewport=$(root,'#viewport');viewport.prepend(canvas);
  const stage=createStage(canvas,viewport,capture);
  let c=controls('open'),view='solid',focus='all',preset='open',camera='front',auto=true,last=performance.now(),lastTime=0,activeChapter=-1,lastUI=-Infinity;
  const rows=FINGERS.map(f=>'<article class="th-readout" data-digit="'+f.id+'"><div class="th-readhead"><b>'+f.name.toUpperCase()+'</b><span class="th-force" id="force-'+f.id+'">0 N</span></div><div class="th-bar"><i id="bar-'+f.id+'"></i></div><div class="th-angles" id="angles-'+f.id+'"></div><div class="th-stroke"><span>Take-up</span><b id="stroke-'+f.id+'"></b></div></article>').join('');
  $(root,'#telemetry').innerHTML=rows;
  const select=(group,val)=>$$ (root,'[data-'+group+']').forEach(b=>{b.classList.toggle('selected',b.dataset[group]===val);b.setAttribute('aria-pressed',String(b.dataset[group]===val));});
  function takeover(){if(auto){auto=false;activeChapter=-1;}last=performance.now();}
  function chooseCamera(v){camera=v;stage.setView(v);}
  function bind(g,fn){$$ (root,'[data-'+g+']').forEach(b=>b.addEventListener('click',()=>{if(!film&&!capture){takeover();fn(b.dataset[g]);}}));}
  bind('preset',v=>{preset=v;c=controls(v);});bind('view',v=>{view=v;});bind('focus',v=>{focus=v;if(v!=='all')chooseCamera('detail');else chooseCamera('front');});bind('camera',chooseCamera);
  bind('action',v=>{if(v==='tour'){auto=true;activeChapter=-1;last=performance.now();lastTime=0;}else{chooseCamera('front');focus='all';view='solid';preset='open';c=controls();}});
  for(const id of ['grip','spread','oppose',...FINGERS.map((_,i)=>'trim'+i)])$(root,'#'+id).addEventListener('input',e=>{if(film||capture)return;takeover();const v=Number(e.target.value);if(id.startsWith('trim'))c.trim[Number(id.slice(4))]=v;else c[id]=v;preset='custom';});
  ['pointerdown','wheel'].forEach(type=>canvas.addEventListener(type,()=>{if(!film&&!capture)takeover();},{passive:true}));
  if(film||capture){stage.orbit.enabled=false;root.classList.add('film');}
  function updateUI(state,beat,progress){
    select('view',view);select('preset',preset);select('focus',focus);select('camera',camera);
    for(const id of ['grip','spread','oppose',...FINGERS.map((_,i)=>'trim'+i)]){const v=id.startsWith('trim')?c.trim[Number(id.slice(4))]:c[id];const el=$(root,'#'+id);if(document.activeElement!==el||auto)el.value=v;$(root,'#'+id+'-value').textContent=Math.round(v*100)+'%';}
    for(const f of state){$(root,'#force-'+f.id).textContent=f.tension.toFixed(1)+' N';$(root,'#bar-'+f.id).style.width=Math.min(100,f.tension/12*100)+'%';$(root,'#angles-'+f.id).textContent=(f.id==='thumb'?'BASE ':'MCP ')+f.angles[0].toFixed(0)+'° / '+(f.id==='thumb'?'MCP ':'PIP ')+f.angles[1].toFixed(0)+'° / '+(f.id==='thumb'?'IP ':'DIP ')+f.angles[2].toFixed(0)+'° / '+(f.id==='thumb'?'Y ':'S ')+f.spread.toFixed(0)+'°';$(root,'#stroke-'+f.id).textContent=f.stroke.toFixed(1)+' mm';$(root,'[data-digit="'+f.id+'"]').classList.toggle('dimmed',focus!=='all'&&focus!==f.id);}
    $(root,'#gap').textContent=pinchGap(state).toFixed(1)+' mm';
    $(root,'#status').textContent=auto?'AUTO TOUR':'MANUAL';
    $(root,'[data-action="tour"]').classList.toggle('selected',auto);
    $(root,'#viewtag').textContent=names[view].toUpperCase()+' / '+focus.toUpperCase();
    $(root,'#chapter').textContent=beat?beat.title:'MANUAL / EXPLORER';
    $(root,'#explain').textContent=beat?beat.note:'Select a grasp, inspect a digit or reveal its tendons. Readouts and geometry share the same joint state.';
    $(root,'#progress').style.width=(progress*100)+'%';
  }
  function render(t){
    let beat=null,progress=0;
    if(auto||film||capture){let clock=film||capture?t:(performance.now()-last)/1000;beat=tour(clock);progress=beat.progress;c=beat.c;view=beat.view;focus=beat.focus;preset=beat.preset;
      if(activeChapter!==beat.index){chooseCamera(beat.camera);activeChapter=beat.index;}
    }else if(performance.now()-last>25000){auto=true;last=performance.now();activeChapter=-1;}
    const state=solve(c);stage.draw(state,view,focus,preset);
    if(capture||film||performance.now()-lastUI>80){updateUI(state,beat,progress);lastUI=performance.now();}
    window.HAND={state,view,focus,preset,controls:{...c,trim:[...c.trim]},pinchGap:pinchGap(state)};
    return {};
  }
  return {render,resize:stage.resize};
 });
}
