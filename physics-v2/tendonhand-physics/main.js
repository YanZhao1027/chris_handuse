import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import recordings from './recordings.json';
import dims from '../dimensions.json';
const app=document.getElementById('app');
const fields=[['pull','Four-finger pull','N',0,25,.1],['thumb','Thumb pull','N',0,25,.1],['mass','Ball mass','g',15,120,1],['friction','Friction μ','',.2,2,.01],['oppose','Thumb base yaw','°',-95,65,1]];
app.innerHTML=`<div id="lab"><header><div class="brand"><i></i><div><small>MECHANISM LAB / SERIES 01</small><h1>TENDON <em>/</em> 02</h1></div></div><div class="headcopy">DYNAMIC GRASP LAB<br><strong>MUJOCO 3.8.1 · FIXED TENDONS · CONTACT SOLVER</strong></div><span id="engine">RECORDED SOLVER DATA</span></header>
<main><aside><small>01 / EXPERIMENT</small><h2>Grasp controls</h2><p>Choose a real simulated trial, or change the parameters and run MuJoCo again on this workstation.</p><div class="buttons"><button data-case="grasp">Grasp</button><button data-case="zero">Zero force</button><button data-case="release">Release</button></div><hr><div id="inputs">${fields.map(([id,label,unit,min,max,step])=>`<label><span>${label}<b id="${id}-v"></b></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}"></label>`).join('')}</div><button id="solve">RUN PHYSICS ↗</button><p id="message">Saved trials are available offline. A local solver service enables custom runs.</p><hr><small>MODEL CONTRACT</small><p>20 joint coordinates, five ideal fixed tendons, five yaw servos, passive return springs. Contacts with ball and floor enabled. Hand self-collision omitted.</p><p>Wrist is a prescribed moving base. Sphere has a free joint; the temporary positioning weld is disabled at 1.0 s.</p></aside>
<section id="view"><canvas id="sim"></canvas><div id="stage-label"><small>LIVE SOLVER TRAJECTORY</small><strong id="phase">POSITION / WELD ON</strong></div><div id="viewbuttons"><button id="front">3/4 view</button><button id="side">Side view</button><button id="contact">Contacts</button></div><div class="legend"><i></i> ACTUAL CONTACT POINTS &nbsp; <i></i> FREE BODY</div></section>
<aside><small>02 / TELEMETRY</small><h2>Contact state</h2><div class="metric"><span>Fixture constraint</span><strong id="weld"></strong></div><div class="metric"><span>Object height / floor</span><strong id="height"></strong></div><div class="metric"><span>Total normal force</span><strong id="force"></strong></div><div class="metric"><span>Contacting digits</span><strong id="digits"></strong></div><hr><small>ACTUATOR FORCE / TAKE-UP</small><div id="readouts">${dims.map(f=>`<div class="readout"><b>${f.name}</b><span id="t-${f.id}"></span><div class="bar"><i id="b-${f.id}"></i></div><small id="q-${f.id}"></small></div>`).join('')}</div><hr><small>FINAL VALIDATION</small><div id="verdict"></div></aside></main>
<footer><div class="transport"><button id="play">Pause</button><button id="restart">Restart</button><b id="time"></b></div><input type="range" id="timeline" min="0" max="6" step=".02" value="0"><div class="footnote"><span id="event">Close → unweld → lift → hold</span><span>1 ms integration · 50 Hz recorded samples</span></div></footer></div>`;
const $=id=>document.getElementById(id);
let trial=recordings.grasp,active='grasp',playing=true,t=0,last=performance.now(),markersOn=true,server=false,busy=false;
const scene=new T.Scene();scene.background=new T.Color(0xe8efee);scene.fog=new T.Fog(0xe8efee,800,1800);
const renderer=new T.WebGLRenderer({canvas:$('sim'),antialias:true,powerPreference:'high-performance'});
renderer.outputColorSpace=T.SRGBColorSpace;renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.2;
const camera=new T.PerspectiveCamera(42,1,1,3000);camera.up.set(0,0,1);
const orbit=new OrbitControls(camera,$('sim'));orbit.minDistance=160;orbit.maxDistance=1300;
function setCamera(side=false){camera.position.set(...(side?[420,-50,225]:[350,330,165]));orbit.target.set(0,10,170);orbit.update();}
setCamera();
scene.add(new T.HemisphereLight(0xffffff,0x8fa1a2,2.5));
for(const [p,intensity] of [[[250,-250,450],3.2],[[-200,100,300],1.6]]){const l=new T.DirectionalLight(0xffffff,intensity);l.position.set(...p);scene.add(l);}
const floor=new T.Mesh(new T.PlaneGeometry(1800,1800),new T.MeshStandardMaterial({color:0xdce5e3,roughness:.95}));scene.add(floor);
const grid=new T.GridHelper(1000,50,0xa0b1b0,0xc3d1cf);grid.rotation.x=Math.PI/2;grid.position.z=.3;scene.add(grid);
const material=(color,metalness=.2)=>new T.MeshStandardMaterial({color,metalness,roughness:.47});
const shell=material(0xa8c1c4,.3),dark=material(0x31494f,.38),steel=material(0x6d8991,.6),accent=material(0x007e83),orange=material(0xe67b24,.05);
const wrist=new T.Group();wrist.rotation.y=Math.PI;scene.add(wrist);
function mesh(parent,g,m,x=0,y=0,z=0){const o=new T.Mesh(g,m);o.position.set(x,y,z);parent.add(o);return o;}
mesh(wrist,new T.BoxGeometry(79,76,20),dark,0,-9,0);mesh(wrist,new T.BoxGeometry(75,42,24),steel,0,-69,0);
const digits=dims.map(f=>{const yaw=new T.Group();yaw.position.set(f.x,f.y,0);wrist.add(yaw);const pivots=[];let p=yaw;
for(let j=0;j<3;j++){const pivot=new T.Group();if(j)pivot.position.y=f.lengths[j-1];p.add(pivot);const r=f.w*(.43-.045*j),len=f.lengths[j];mesh(pivot,new T.CapsuleGeometry(r,len,6,14),j===2?dark:shell,0,len/2,0);const pin=mesh(pivot,new T.CylinderGeometry(r*.70,r*.70,r*2+.5,16),steel);pin.rotation.z=Math.PI/2;pivots.push(pivot);p=pivot;}return {yaw,pivots};});
const ball=mesh(scene,new T.SphereGeometry(1,40,24),orange);
const contactMeshes=Array.from({length:36},()=>{const o=mesh(scene,new T.SphereGeometry(1,10,8),accent);o.visible=false;return o;});
const pointer=mesh(scene,new T.TorusGeometry(1,.055,6,32),steel);pointer.rotation.x=0;pointer.visible=false;
function resize(){const w=$('view').clientWidth,h=$('view').clientHeight;renderer.setPixelRatio(Math.min(devicePixelRatio,1.6));renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}
new ResizeObserver(resize).observe($('view'));resize();
const rad=Math.PI/180;
function display(f){
 wrist.position.set(...f.wrist.map(v=>1000*v));
 digits.forEach((d,i)=>{d.yaw.rotation.z=f.yaw[i]*rad;d.pivots.forEach((p,j)=>p.rotation.x=f.angles[i][j]*rad);});
 ball.position.set(...f.object.map(v=>v*1000));ball.scale.setScalar(trial.summary.radius*1000);
 const q=f.objectQuat;ball.quaternion.set(q[1],q[2],q[3],q[0]);
 contactMeshes.forEach((o,i)=>{const c=f.contacts?.[i];o.visible=markersOn&&!!c;if(c){o.position.set(...c.p.map(v=>1000*v));o.scale.setScalar(1.3+Math.min(3,Math.sqrt(c.n)));}});
 pointer.visible=f.fixture;pointer.position.copy(ball.position);pointer.scale.setScalar(trial.summary.radius*1180);
 $('phase').textContent=f.fixture?'POSITION / WELD ON':f.t<1.10?'RELEASED / UNWELDED':f.t<1.95?'LIFT / FREE OBJECT':trial.summary.releaseAt!=null&&f.t>=trial.summary.releaseAt?'RELEASE / ZERO PULL':'HOLD / FREE OBJECT';
 $('weld').textContent=f.fixture?'ACTIVE':'OFF';$('weld').classList.toggle('off',!f.fixture);
 $('height').textContent=(f.object[2]*1000).toFixed(1)+' mm';$('force').textContent=f.normalForce.toFixed(2)+' N';$('digits').textContent=f.digits.length?f.digits.join(' · '):'None';
 dims.forEach((d,i)=>{$('t-'+d.id).textContent=f.tendonForce[i].toFixed(2)+' N / '+f.tendonLength[i].toFixed(1)+' mm';$('b-'+d.id).style.width=Math.min(100,f.tendonForce[i]/25*100)+'%';$('q-'+d.id).textContent=f.angles[i].map(q=>q.toFixed(0)+'°').join(' / ');});
 $('time').textContent=t.toFixed(2)+' / '+trial.summary.duration.toFixed(1)+' s';$('timeline').value=t;
}
function updateInputs(){for(const [id,label,unit] of fields){let v=trial.summary[id]*(id==='mass'?1000:1);$(id).value=v;$(id+'-v').textContent=Number($(id).value).toFixed(id==='friction'?2:id==='mass'||id==='oppose'?0:1)+' '+unit;}}
function choose(name,data){active=name;trial=data;t=0;playing=true;$('play').textContent='Pause';$('timeline').max=trial.summary.duration;document.querySelectorAll('[data-case]').forEach(b=>b.classList.toggle('selected',b.dataset.case===name));updateInputs();
 const s=trial.summary;$('verdict').innerHTML='<b class="'+(s.success?'pass':'fail')+'">'+(s.releaseAt!=null&&s.finalHeight<.10?'RELEASED AS COMMANDED':s.success?'GRASP RETAINED':'GRASP NOT RETAINED')+'</b><p>Final height '+(s.finalHeight*1000).toFixed(1)+' mm<br>Max hold drift '+(s.maxHoldDrift*1000).toFixed(1)+' mm<br>Fixture off at end: '+(!s.fixture?'yes':'no')+'</p>';display(trial.frames[0]);}
document.querySelectorAll('[data-case]').forEach(b=>b.onclick=()=>choose(b.dataset.case,recordings[b.dataset.case]));
for(const [id,label,unit] of fields)$(id).oninput=()=>{$(id+'-v').textContent=$(id).value+' '+unit;};
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pause':'Play';};
$('restart').onclick=()=>{t=0;playing=true;$('play').textContent='Pause';};
$('timeline').oninput=()=>{playing=false;$('play').textContent='Play';t=Number($('timeline').value);};
$('front').onclick=()=>setCamera();$('side').onclick=()=>setCamera(true);$('contact').onclick=()=>{markersOn=!markersOn;$('contact').classList.toggle('selected',markersOn);};$('contact').classList.add('selected');
async function connect(){if(!/^https?:$/.test(location.protocol))return;try{const res=await fetch('/api/status');if(res.ok){const data=await res.json();server=data.engine==='mujoco';if(server){$('engine').textContent='LOCAL MUJOCO SOLVER / ONLINE';$('message').textContent='Change settings and run a fresh 6-second rigid-body simulation.';}}}catch(e){}}
$('solve').onclick=async()=>{if(busy)return;if(!server){$('message').textContent='Open the local solver URL to run custom experiments. Offline playback remains available.';return;}busy=true;$('solve').disabled=true;$('message').textContent='Solving…';try{const cfg=Object.fromEntries(fields.map(([id])=>[id,Number($(id).value)/(id==='mass'?1000:1)]));const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)});if(!r.ok)throw Error('HTTP '+r.status);const data=await r.json();choose('custom',data);$('message').textContent='New MuJoCo trajectory computed and loaded.';}catch(e){$('message').textContent='Solver error: '+e.message;}finally{busy=false;$('solve').disabled=false;}};
choose('grasp',recordings.grasp);connect();
function animate(now){const dt=Math.min(.1,(now-last)/1000);last=now;if(playing){t+=dt;if(t>=trial.summary.duration){t=trial.summary.duration;playing=false;$('play').textContent='Play';}}let idx=Math.min(trial.frames.length-1,Math.round(t*50));display(trial.frames[idx]);orbit.update();renderer.render(scene,camera);window.PHYSICS={ready:true,scenario:active,t,frame:trial.frames[idx],summary:trial.summary,server};requestAnimationFrame(animate);}
requestAnimationFrame(animate);
