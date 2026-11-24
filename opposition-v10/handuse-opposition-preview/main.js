import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import rec from './recordings.json';
const app=document.getElementById('app'),$=id=>document.getElementById(id),p=rec.params,s=rec.summary;
app.innerHTML=`<div id="lab"><header><div><small>CHRIS HANDUSE / SERIES 10</small><h1>OPPOSITION <em>/</em> LAB</h1></div><div class="meta">EXACT MUJOCO COLLISION GEOMETRY<br>PRETRAINING GRASP VALIDATION · SEED ${rec.seed}</div></header>
<main><aside><small>01 / MECHANISM</small><h2>Opposed thumb</h2><p>The thumb rises from the proximal side and curls toward the shaft. Its base frame is rotated relative to the four fingers.</p><div class="metric"><span>Rod length / diameter</span><strong>${(p.half_length*2000).toFixed(1)} / ${(p.radius*2000).toFixed(1)} mm</strong></div><div class="metric"><span>Thumb base tilt</span><strong>${rec.mount.tilt.toFixed(1)}°</strong></div><div class="metric"><span>Thumb base twist</span><strong>${rec.mount.twist.toFixed(1)}°</strong></div><hr><small>DISPLAY</small><button id="front">3/4 view</button><button id="rear">Thumb side</button><button id="end">Rod end</button><button id="bottom">Palm underside</button><button id="contact">Contacts on/off</button><p>Each mesh is transformed using recorded geom_xpos / geom_xmat. Surface grid and colors are visual only.</p></aside>
<section id="view"><canvas id="sim"></canvas><div class="overlay"><small>REAL PHYSICS REPLAY</small><strong id="phase"></strong></div><div class="legend"><i></i> Thumb &nbsp; <i></i> Loaded contacts</div></section>
<aside><small>02 / OPPOSITION AUDIT</small><h2>Loaded contact</h2><div class="metric"><span>Opposed grasp now</span><strong id="opposed">—</strong></div><div class="metric"><span>Radial normal separation</span><strong id="angle">—</strong></div><div class="metric"><span>Thumb side load</span><strong id="thumb">—</strong></div><div class="metric"><span>Four-finger side load</span><strong id="fingers">—</strong></div><div class="metric"><span>Self-contact peak, sampled</span><strong id="self">—</strong></div><div class="metric"><span>Rod penetration peak, sampled</span><strong id="rod">—</strong></div><hr><small>FINAL GEOMETRY AUDIT</small><div class="metric"><span>Peak self-overlap</span><strong>${s.geometryAudit.maxSelfPenetrationMM.toFixed(3)} mm</strong></div><div class="metric"><span>Opposed fraction after lift</span><strong>${(100*s.opposedFraction).toFixed(0)}%</strong></div><p>Grasp validation only; no learned rotation claimed.</p></aside></main>
<footer><div class="transport"><button id="play">Pause</button><button id="restart">Restart</button><b id="time"></b></div><input id="timeline" type="range" min="0" max="4" value="0" step=".04"><div class="footnote">Close → release fixture → lift → stable opposed grasp <span>1 ms integrator / 25 Hz telemetry</span></div></footer></div>`;
const scene=new T.Scene();scene.background=new T.Color(0xe8efee);
const renderer=new T.WebGLRenderer({canvas:$('sim'),antialias:true,powerPreference:'high-performance'});
renderer.outputColorSpace=T.SRGBColorSpace;renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.15;
const camera=new T.PerspectiveCamera(35,1,1,2000);camera.up.set(0,0,1);
const orbit=new OrbitControls(camera,$('sim'));orbit.minDistance=120;orbit.maxDistance=1000;
function cam(mode){
 const targets={front:[310,290,195],rear:[270,-290,190],end:[400,15,225],bottom:[20,5,-90]};
 camera.position.set(...targets[mode]);orbit.target.set(0,12,270);orbit.update();
}
cam('front');
scene.add(new T.HemisphereLight(0xffffff,0x8ca0a0,2.5));
for(const [v,intensity] of [[[250,-250,450],3.2],[[-200,100,300],1.6]]){const light=new T.DirectionalLight(0xffffff,intensity);light.position.set(...v);scene.add(light);}
function mat(color){return new T.MeshStandardMaterial({color,metalness:.25,roughness:.5});}
function gridTexture(cols,rows){const canvas=document.createElement('canvas');canvas.width=1024;canvas.height=1024;const c=canvas.getContext('2d');c.fillStyle='#9cd3f1';c.fillRect(0,0,1024,1024);c.strokeStyle='#4d87ad';c.lineWidth=2.2;c.beginPath();for(let i=0;i<=cols;i++){const x=i*1024/cols;c.moveTo(x,0);c.lineTo(x,1024);}for(let i=0;i<=rows;i++){const y=i*1024/rows;c.moveTo(0,y);c.lineTo(1024,y);}c.stroke();const tex=new T.CanvasTexture(canvas);tex.colorSpace=T.SRGBColorSpace;tex.anisotropy=8;return tex;}
const shaft=new T.MeshStandardMaterial({map:gridTexture(20,28),roughness:.59,metalness:.15}),cap=new T.MeshStandardMaterial({map:gridTexture(9,9),roughness:.59,metalness:.15});
const materials={palm:mat(0x344d55),finger:mat(0xa8c1c4),thumb:mat(0x397f88),distal:mat(0x56727b)};
const geoms=rec.geoms.map(g=>{const group=new T.Group();scene.add(group);let geo,m;
 if(g.type==='box'){geo=new T.BoxGeometry(...g.size.map(v=>2*v));m=materials.palm;}
 else if(g.type==='capsule'){geo=new T.CapsuleGeometry(g.size[0],2*g.size[1],10,24);m=g.name.startsWith('thumb')?materials.thumb:g.name.endsWith('2')?materials.distal:materials.finger;}
 else {geo=new T.CylinderGeometry(g.size[0],g.size[0],2*g.size[1],64,1,false);m=[shaft,cap,cap];}
 const mesh=new T.Mesh(geo,m);if(g.type!=='box')mesh.rotation.x=Math.PI/2;group.add(mesh);return group;
});
const contacts=Array.from({length:64},()=>{const m=new T.Mesh(new T.SphereGeometry(1,10,8),mat(0xf0aa37));scene.add(m);m.visible=false;return m;});
let markers=true;
function resize(){const w=$('view').clientWidth,h=$('view').clientHeight;renderer.setPixelRatio(Math.min(devicePixelRatio,1.6));renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}
new ResizeObserver(resize).observe($('view'));resize();
let t=0,playing=true,last=performance.now();
for(const key of ['front','rear','end','bottom'])$(key).onclick=()=>cam(key);
$('contact').onclick=()=>markers=!markers;
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pause':'Play';};
$('restart').onclick=()=>{t=0;playing=true;$('play').textContent='Pause';};
$('timeline').oninput=()=>{playing=false;$('play').textContent='Play';t=Number($('timeline').value);};
function show(f){
 geoms.forEach((g,i)=>{g.position.set(...f.poses[i].p);g.quaternion.set(...f.poses[i].q);});
 contacts.forEach((g,i)=>{const c=f.contacts[i];g.visible=markers&&!!c&&c.n>.005;if(c){g.position.set(...c.p.map(v=>v*1000));g.scale.setScalar(1.2+Math.min(2,Math.sqrt(c.n)));}});
 const o=f.opposition;$('phase').textContent=f.fixture?'CLOSING / FIXTURE ON':f.t<2?'FREE BODY / LIFT':'FREE BODY / HOLD';
 $('opposed').textContent=o.opposed?'YES':'NO';$('angle').textContent=(Math.acos(Math.max(-1,Math.min(1,o.cos)))*180/Math.PI).toFixed(1)+'°';
 $('thumb').textContent=o.thumbForce.toFixed(2)+' N';$('fingers').textContent=o.fingerForce.toFixed(2)+' N';$('self').textContent=f.selfPenetrationMM.toFixed(3)+' mm';$('rod').textContent=f.objectPenetrationMM.toFixed(3)+' mm';$('time').textContent=f.t.toFixed(2)+' / 4.00 s';$('timeline').value=t;
 window.HANDUSE={ready:true,frame:f,summary:rec.summary,geometry:rec.geoms};
}
function animate(now){const dt=Math.min(.1,(now-last)/1000);last=now;if(playing){t+=dt;if(t>=4){t=4;playing=false;$('play').textContent='Play';}}const i=Math.max(0,Math.min(rec.frames.length-1,Math.round(t*25)-1));show(rec.frames[i]);orbit.update();renderer.render(scene,camera);requestAnimationFrame(animate);}
requestAnimationFrame(animate);
