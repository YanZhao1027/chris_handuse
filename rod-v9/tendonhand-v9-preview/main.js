import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import records from './recordings.json';
import dims from './dimensions.json';
const app=document.getElementById('app'),$=id=>document.getElementById(id);
const seed=records.rotated.summary.seed;
app.innerHTML=`<div id="lab"><header><div class="brand"><i></i><div><small>MECHANISM LAB / SERIES 09</small><h1>TENDON <em>/</em> CONTACT AUDIT</h1></div></div><div class="headcopy">IN-HAND REORIENTATION / 110 × 26 MM<br><strong>MUJOCO 3.8.1 · SELF-COLLISION · GOAL CONTROL</strong></div><span id="engine">PAIRED PHYSICS TRIAL</span></header>
<main><aside><small>01 / EXPERIMENT</small><h2>Long-rod rotation</h2><p>The same elongated object, initial pose, mass, friction and grasp. Compare ordinary hold against a differential finger-yaw command.</p><div class="buttons"><button data-case="fixed">−15° hold</button><button data-case="rotated">−15° candidate</button><button data-case="positiveHold">+15° hold</button><button data-case="positive">+15° candidate</button></div><hr><small>OBJECT SPECIFICATION</small><div class="metric"><span>Overall length</span><strong id="objlength">${(records.rotated.summary.params.half_length*2000).toFixed(1)} mm</strong></div><div class="metric"><span>Maximum diameter</span><strong id="objdiam">${(records.rotated.summary.params.radius*2000).toFixed(1)} mm</strong></div><div class="metric"><span>Mass</span><strong id="objmass">${(records.rotated.summary.params.mass*1000).toFixed(1)} g</strong></div><div class="metric"><span>Friction coefficient</span><strong id="objmu">${records.rotated.summary.params.friction.toFixed(2)}</strong></div><hr><small>CONTROL COMMAND</small><p>Hold the grasp prior, then apply goal-conditioned differential commands to the finger actuators. The fixture is disabled before lifting and rotation.</p><div class="command"><code>10-AXIS RESIDUAL COMMAND</code><small>PRETRAINING GEOMETRY CHECK · searched fixed command</small></div></aside>
<section id="view"><canvas id="sim"></canvas><div id="stage-label"><small>SELF-COLLISION ENABLED / RECORDED PHYSICS</small><strong id="phase"></strong></div><div id="viewbuttons"><button id="front">3/4 view</button><button id="side">Side view</button><button id="top">Top view</button><button id="contact">Contacts</button></div><div class="legend"><i></i> CONTACT POINTS &nbsp; <i></i> GRID SURFACE</div></section>
<aside><small>02 / TELEMETRY</small><h2>Manipulation state</h2><div class="metric"><span>Rotation component about Z</span><strong id="rotation">—</strong></div><div class="metric"><span>Target orientation error</span><strong id="error">—</strong></div><div class="metric"><span>Angular speed</span><strong id="omega">—</strong></div><div class="metric"><span>Object height</span><strong id="height">—</strong></div><div class="metric"><span>Normal contact force</span><strong id="force">—</strong></div><div class="metric"><span>Peak self-contact penetration</span><strong id="penetration">—</strong></div><div class="metric"><span>Contacting digits</span><strong id="digits">—</strong></div><hr><small>ACTUAL JOINT STATE</small><div id="readouts">${dims.map(f=>`<div class="readout"><b>${f.name}</b><span id="t-${f.id}"></span><div class="bar"><i id="b-${f.id}"></i></div><small id="q-${f.id}"></small></div>`).join('')}</div><hr><small>TRIAL OUTCOME</small><div id="verdict"></div></aside></main>
<footer><div class="transport"><button id="play">Pause</button><button id="restart">Restart</button><b id="time"></b></div><input type="range" id="timeline" min="0" max="6" step=".04" value="0"><div class="footnote"><span>Grasp → release fixture → lift → rotate → hold</span><span>CPU reference replay · 1 ms physics step · 25 Hz control</span></div></footer></div>`;
let trial=records.rotated,active='rotated',playing=true,t=0,last=performance.now(),markersOn=true;
const scene=new T.Scene();scene.background=new T.Color(0xe8efee);scene.fog=new T.Fog(0xe8efee,800,1800);
const renderer=new T.WebGLRenderer({canvas:$('sim'),antialias:true,powerPreference:'high-performance'});
renderer.outputColorSpace=T.SRGBColorSpace;renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.2;
const camera=new T.PerspectiveCamera(42,1,1,3000);camera.up.set(0,0,1);
const orbit=new OrbitControls(camera,$('sim'));orbit.minDistance=160;orbit.maxDistance=1300;
function setCamera(mode='three'){
 if(mode==='side')camera.position.set(420,-50,225);
 else if(mode==='top')camera.position.set(10,10,720);
 else camera.position.set(350,330,165);
 orbit.target.set(0,10,170);orbit.update();
}setCamera();
scene.add(new T.HemisphereLight(0xffffff,0x8fa1a2,2.5));
for(const [p,intensity] of [[[250,-250,450],3.2],[[-200,100,300],1.6]]){const l=new T.DirectionalLight(0xffffff,intensity);l.position.set(...p);scene.add(l);}
const floor=new T.Mesh(new T.PlaneGeometry(1800,1800),new T.MeshStandardMaterial({color:0xdce5e3,roughness:.95}));scene.add(floor);
const grid=new T.GridHelper(1000,50,0xa0b1b0,0xc3d1cf);grid.rotation.x=Math.PI/2;grid.position.z=.3;scene.add(grid);
const material=(color,metalness=.2)=>new T.MeshStandardMaterial({color,metalness,roughness:.47});
const shell=material(0xa8c1c4,.3),dark=material(0x31494f,.38),steel=material(0x6d8991,.6),accent=material(0x007e83),orange=material(0xd87a32,.28),stripe=material(0x163f45,.25);
const wrist=new T.Group();wrist.rotation.y=Math.PI;scene.add(wrist);
function mesh(parent,g,m,x=0,y=0,z=0){const o=new T.Mesh(g,m);o.position.set(x,y,z);parent.add(o);return o;}
mesh(wrist,new T.BoxGeometry(79,76,20),dark,0,-9,0);mesh(wrist,new T.BoxGeometry(75,42,24),steel,0,-69,0);
const digits=dims.map(f=>{const yaw=new T.Group();yaw.position.set(f.x,f.y,f.z||0);wrist.add(yaw);const pivots=[];let p=yaw;
for(let j=0;j<3;j++){const pivot=new T.Group();if(j)pivot.position.y=f.lengths[j-1];p.add(pivot);const r=f.w*(.43-.045*j),len=f.lengths[j];mesh(pivot,new T.CapsuleGeometry(r,len,6,14),j===2?dark:shell,0,len/2,0);pivots.push(pivot);p=pivot;}return {yaw,pivots};});
const rod=new T.Group();scene.add(rod);
const shape=records.rotated.summary.params;
const rr=shape.radius*1000,hh=shape.cylinder_half_length*1000;
// Grid texture on the cylinder and caps. Flat end faces match MuJoCo cylinder geometry.
function gridTexture(cols,rows){
 const canvas=document.createElement('canvas');canvas.width=1024;canvas.height=1024;
 const ctx=canvas.getContext('2d');ctx.fillStyle='#9cd3f1';ctx.fillRect(0,0,1024,1024);
 ctx.strokeStyle='#4d87ad';ctx.lineWidth=2.2;ctx.beginPath();
 for(let i=0;i<=cols;i++){let x=i*1024/cols;ctx.moveTo(x,0);ctx.lineTo(x,1024);}
 for(let i=0;i<=rows;i++){let y=i*1024/rows;ctx.moveTo(0,y);ctx.lineTo(1024,y);}
 ctx.stroke();const tex=new T.CanvasTexture(canvas);tex.colorSpace=T.SRGBColorSpace;tex.anisotropy=8;return tex;
}
const gridMat=new T.MeshStandardMaterial({map:gridTexture(20,28),roughness:.59,metalness:.15});
const capMat=new T.MeshStandardMaterial({map:gridTexture(9,9),roughness:.59,metalness:.15});
const rodBody=mesh(rod,new T.CylinderGeometry(rr,rr,2*hh,64,1,false),[gridMat,capMat,capMat]);
rodBody.rotation.z=Math.PI/2;
const contacts=Array.from({length:36},()=>{const o=mesh(scene,new T.SphereGeometry(1,10,8),accent);o.visible=false;return o;});
function resize(){const w=$('view').clientWidth,h=$('view').clientHeight;renderer.setPixelRatio(Math.min(devicePixelRatio,1.6));renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}
new ResizeObserver(resize).observe($('view'));resize();
const rad=Math.PI/180,identity=new T.Quaternion();
function quat(a){return new T.Quaternion(a[1],a[2],a[3],a[0]);}
function display(f){
 wrist.position.set(...f.wrist.map(v=>1000*v));
 digits.forEach((d,i)=>{d.yaw.rotation.z=f.yaw[i]*rad;d.pivots.forEach((p,j)=>p.rotation.x=f.angles[i][j]*rad);});
 rod.position.set(...f.object.map(v=>1000*v));rod.quaternion.copy(quat(f.objectQuat));
 const rp=trial.summary.params;rodBody.scale.set(rp.radius*1000/rr,rp.cylinder_half_length*1000/hh,rp.radius*1000/rr);
 
 contacts.forEach((o,i)=>{const c=f.contacts?.[i];o.visible=markersOn&&!!c;if(c){o.position.set(...c.p.map(v=>1000*v));o.scale.setScalar(1.3+Math.min(3,Math.sqrt(c.n)));}});
 const start=quat(trial.frames[Math.min(49,trial.frames.length-1)].objectQuat),now=quat(f.objectQuat),delta=now.clone().multiply(start.clone().invert());
 if(delta.w<0)delta.set(-delta.x,-delta.y,-delta.z,-delta.w);
 const axis=new T.Vector3(delta.x,delta.y,delta.z).normalize(),angle=delta.angleTo(identity)/rad;
 const signed=f.t>=2?angle*axis.z:0;
 const target=quat(f.targetQuat),err=target.clone().multiply(now.clone().invert()).angleTo(identity)/rad;
 $('phase').textContent=f.fixture?'POSITION / FIXTURE ON':f.t<1.1?'FIXTURE RELEASED':f.t<2?'LIFT / FREE ROD':f.streak>=10?'TARGET HOLD':'FREE MOTION';
 $('rotation').textContent=signed.toFixed(1)+'°';
 $('error').textContent=f.t>2?err.toFixed(1)+'°':'—';
 $('omega').textContent=f.angularSpeed.toFixed(2)+' rad/s';
 $('height').textContent=(f.object[2]*1000).toFixed(1)+' mm';
 $('force').textContent=f.normalForce.toFixed(2)+' N';
 $('penetration').textContent=Number(f.selfPenetrationMM||0).toFixed(3)+' mm';
 $('digits').textContent=f.digits.length?f.digits.join(' · '):'None';
 dims.forEach((d,i)=>{$('t-'+d.id).textContent=f.tendonForce[i].toFixed(2)+' N';$('b-'+d.id).style.width=Math.min(100,f.tendonForce[i]/25*100)+'%';$('q-'+d.id).textContent='Yaw '+f.yaw[i].toFixed(0)+'° · '+f.angles[i].map(q=>q.toFixed(0)+'°').join(' / ');});
 $('time').textContent=f.t.toFixed(2)+' / '+trial.frames.at(-1).t.toFixed(2)+' s';$('timeline').value=t;
}
function choose(name){active=name;trial=records[name];t=0;playing=true;$('play').textContent='Pause';$('timeline').max=trial.frames.at(-1).t;document.querySelectorAll('[data-case]').forEach(b=>b.classList.toggle('selected',b.dataset.case===name));const s=trial.summary,p=s.params;$('engine').textContent='PAIRED TEST · SEED '+s.seed;$('objlength').textContent=(p.half_length*2000).toFixed(1)+' mm';$('objdiam').textContent=(p.radius*2000).toFixed(1)+' mm';$('objmass').textContent=(p.mass*1000).toFixed(1)+' g';$('objmu').textContent=p.friction.toFixed(2);$('verdict').innerHTML='<b class="'+(s.success?'pass':'fail')+'">'+(s.success?'TARGET HELD':'TARGET NOT REACHED')+'</b><p>Target ' + s.targetDegrees.toFixed(0) + '° about world Z<br>Rotation component about Z '+s.actualRotation.zComponentDegrees.toFixed(2)+'°<br>Rotation axis Z component '+s.actualRotation.axis[2].toFixed(3)+'<br>Final error '+s.errorDegrees.toFixed(2)+'°<br>Fixture off: '+(!s.fixture?'yes':'no')+'</p>';display(trial.frames[0]);}
document.querySelectorAll('[data-case]').forEach(b=>b.onclick=()=>choose(b.dataset.case));
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pause':'Play';};
$('restart').onclick=()=>{t=0;playing=true;$('play').textContent='Pause';};
$('timeline').oninput=()=>{playing=false;$('play').textContent='Play';t=Number($('timeline').value);};
$('front').onclick=()=>setCamera('three');$('side').onclick=()=>setCamera('side');$('top').onclick=()=>setCamera('top');$('contact').onclick=()=>{markersOn=!markersOn;$('contact').classList.toggle('selected',markersOn);};$('contact').classList.add('selected');
choose('rotated');
function animate(now){const dt=Math.min(.1,(now-last)/1000);last=now;let end=trial.frames.at(-1).t;if(playing){t+=dt;if(t>=end){t=end;playing=false;$('play').textContent='Play';}}let idx=Math.max(0,Math.min(trial.frames.length-1,Math.round(t*25)-1));display(trial.frames[idx]);orbit.update();renderer.render(scene,camera);window.ROD={ready:true,active,frame:trial.frames[idx],summary:trial.summary};requestAnimationFrame(animate);}
requestAnimationFrame(animate);
