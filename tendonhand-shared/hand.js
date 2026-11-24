import * as T from 'three';
import {RoundedBoxGeometry} from 'three/addons/geometries/RoundedBoxGeometry.js';
import {FINGERS,RAD,forward} from './kin.js';
const mat=(color,metalness=.25,roughness=.47)=>new T.MeshStandardMaterial({color,metalness,roughness});
const M={shell:mat(0xd5dfe1,.3,.43),rail:mat(0x647b86,.65,.37),dark:mat(0x26343d,.48,.48),steel:mat(0x9facb5,.8,.32),cable:mat(0x007e83,.35,.40),return:mat(0x78858d,.42,.5),tip:mat(0x293d43,.1,.8),amber:mat(0x007e83,.35,.5),base:mat(0xadbfc3,.3,.55)};
const unitCylinder=new T.CylinderGeometry(1,1,1,20);
const unitSphere=new T.SphereGeometry(1,18,12);
const Z=new T.Vector3(0,1,0);
const V=(x,y,z)=>new T.Vector3(x,y,z);
function mesh(parent,geo,material,x=0,y=0,z=0){const o=new T.Mesh(geo,material);o.position.set(x,y,z);o.castShadow=true;o.receiveShadow=true;parent.add(o);return o;}
function box(p,w,l,d,m,x=0,y=0,z=0,r=1.8){return mesh(p,new RoundedBoxGeometry(w,l,d,2,Math.min(r,w/3,l/3,d/3)),m,x,y,z);}
function cylinder(p,r,len,m,x=0,y=0,z=0,axis='x'){const o=mesh(p,unitCylinder,m,x,y,z);o.scale.set(r,len,r);if(axis==='x')o.rotation.z=Math.PI/2;if(axis==='z')o.rotation.x=Math.PI/2;return o;}
function screw(p,x,y,z,r=1.7){cylinder(p,r,.95,M.steel,x,y,z,'z');box(p,r*1.25,.45,.35,M.dark,x,y,z+.6,.1);}
function link(p,w,L,idx,isThumb){const cover=new T.Group();p.add(cover);const metal=new T.Group();p.add(metal);const width=w*(1-idx*.12);
 // Pair of structural cheeks, open central channel and actual hinge bearings.
 for(const s of [-1,1]){
  box(metal,2.7,L-4.5,8.6,M.rail,s*(width/2-1.4),L/2,0,1);
  box(cover,2.2,L-7,11.2,M.shell,s*(width/2+.15),L/2,-.2,1.5);
 }
 box(metal,width-5,L-10,4,M.dark,0,L/2,-1.3,1);
 box(cover,width-2,L-7,4.8,M.shell,0,L/2,-5.2,1.8);
 box(cover,width-4,L-10,2.5,M.shell,0,L/2,5.3,1.3);
 for(const y of [6,L-6])for(const s of [-1,1])screw(cover,s*(width/2-3),y,-8,1.2);
 if(idx===2){box(cover,width-1,9,11.5,M.tip,0,L-3.8,.4,3);for(let j=0;j<3;j++)box(cover,width-3,.8,.7,M.steel,0,L-6+j*2.2,6.1,.25);}
 return {cover,metal};
}
function rope(p,count,m,r){
 const g=new T.Group();p.add(g);
 const seg=Array.from({length:count-1},()=>mesh(g,unitCylinder,m));
 const nodes=Array.from({length:count},()=>mesh(g,unitSphere,m));
 const direction=new T.Vector3();
 return {g,update(points){nodes.forEach((o,i)=>{o.visible=i<points.length;if(o.visible){o.position.copy(points[i]);o.scale.setScalar(r);}});
 seg.forEach((o,i)=>{o.visible=i+1<points.length;if(!o.visible)return;direction.subVectors(points[i+1],points[i]);let len=direction.length();o.visible=len>.01;if(o.visible){o.position.copy(points[i]).addScaledVector(direction,.5);o.quaternion.setFromUnitVectors(Z,direction.normalize());o.scale.set(r,len,r);}});}};
}
function flexRoute(f,root,sign){
 const pts=[V(0,-8,sign*f.r[0])];let a=0,p=V(0,0,0);
 for(let j=0;j<3;j++){
  const end=a+f.angles[j]*RAD;const r=f.r[j];
  for(let k=0;k<=6;k++){const h=a+(end-a)*k/6;pts.push(p.clone().add(V(0,-sign*r*Math.sin(h),sign*r*Math.cos(h))));}
  p.add(V(0,f.lengths[j]*Math.cos(end),f.lengths[j]*Math.sin(end)));a=end;
 }
 pts.push(p.clone().add(V(0,-sign*2*Math.sin(a),sign*2*Math.cos(a))));
 return pts;
}
export function createHand(scene){
 const group=new T.Group();scene.add(group);
 const palm=new T.Group();group.add(palm);
 const palmShell=new T.Group();palm.add(palmShell);
 // Palmar frame, dorsal cover, and detachable front plate.
 box(palm,79,76,15,M.dark,0,-9,-1,6);
 box(palm,73,68,6,M.base,0,-9,3,4);
 for(const s of [-1,1]){box(palm,5,67,17,M.rail,s*39,-9,0,2);box(palm,5,14,18,M.rail,s*32,21,0,2);}
 box(palmShell,79,73,5.5,M.shell,0,-9,-12,6);
 const front=new T.Group();palmShell.add(front);
 box(front,77,67,4.3,M.shell,0,-10,11.8,5);
 box(front,55,38,1.5,M.base,0,-10,14.4,4);
 for(let j=0;j<7;j++)box(front,17,.9,.55,M.rail,-11,-17+j*3.5,15.45,.3);
 for(let j=0;j<4;j++)box(front,3,3,.9,M.rail,17,-15+j*4.3,15.5,.5);
 // Raised guide rails and bushings underneath the removable face.
 for(let i=0;i<5;i++){
  const f=FINGERS[i];const x=-28+i*14;
  box(palm,3.5,48,2.2,M.rail,x,-14,7.1,.8);
  cylinder(palm,3.5,2.5,M.steel,x,-37,8,'z');
  cylinder(palm,1.4,3.2,M.dark,x,-37,8,'z');
  cylinder(palm,3.4,2.5,M.steel,f.x,f.y-10,8,'z');
 }
 for(const x of [-32,32])for(const y of [-37,17])screw(front,x,y,14.5,2.4);
 // Exposed palm channels and five capstan motors in the proximal block.
 const drive=new T.Group();group.add(drive);
 box(drive,75,42,5,M.dark,0,-69,-13,3); for(const s of [-1,1])box(drive,4,40,22,M.rail,s*35,-69,0,2);
 box(drive,76,5,28,M.rail,0,-48,0,2);
 box(drive,62,8,22,M.rail,0,-94,0,2);
 const driveCover=new T.Group();drive.add(driveCover);box(driveCover,73,39,4,M.shell,0,-69,15,4);
 for(const x of [-29,29])for(const y of [-84,-55])screw(driveCover,x,y,17.4,1.8);
 const digits=FINGERS.map((f,i)=>{
   const root=new T.Group();group.add(root);root.position.set(f.x,f.y,0);
   const joints=[],parts=[];let parent=root;
   for(let j=0;j<3;j++){
    const pivot=new T.Group();parent.add(pivot);if(j>0)pivot.position.y=f.lengths[j-1];
    const width=f.w*(1-j*.12);
    cylinder(parent,6.2-j*.65,width+2,M.dark,0,pivot.position.y,0);
    cylinder(parent,3.1-j*.25,width+5,M.steel,0,pivot.position.y,0);
    for(const s of [-1,1])cylinder(parent,4-j*.3,1.5,M.rail,s*(width/2+2.2),pivot.position.y,0);
    // The cable guide is centered on the hinge's modeled moment arm.
    cylinder(parent,f.r[j],4.1,M.steel,0,pivot.position.y,0);
    parts.push(link(pivot,f.w,f.lengths[j],j,i===4));joints.push(pivot);parent=pivot;
   }
   const flex=rope(root,23,M.cable,.85),ext=rope(root,23,M.return,.60);
   const spool=new T.Group();drive.add(spool);spool.position.set(-28+i*14,-69,4);
   cylinder(spool,7,8,M.steel,0,0,0,'z');
   cylinder(spool,8,1.7,M.rail,0,0,5,'z');
   cylinder(spool,8,1.7,M.rail,0,0,-5,'z');
   box(spool,2,9,2,M.cable,0,3,6);
   cylinder(drive,5.5,14,M.dark,-28+i*14,-69,-8,'z');
   const channel=rope(group,4,M.cable,.85);
   const returnChannel=rope(group,4,M.return,.60);
   return {root,joints,parts,flex,ext,spool,channel,returnChannel};
 });
 const ring=new T.Group();group.add(ring);
 const markerMat=mat(0x007e83,.15,.6);
 const tipMarkers=[mesh(ring,unitSphere,markerMat),mesh(ring,unitSphere,markerMat)];
 tipMarkers.forEach(o=>o.scale.setScalar(2.4));const gap=rope(ring,2,markerMat,.45);
 function update(state,view,selected,grasp){
   const ex=view==='exploded';const shell=view==='solid'||ex;
   front.position.set(ex?-18:0,ex?3:0,ex?39:0);palmShell.visible=shell;driveCover.visible=shell;driveCover.position.set(ex?-18:0,ex?-7:0,ex?38:0);
   for(let i=0;i<5;i++){
    const f=state[i],d=digits[i];const show=selected==='all'||selected===f.id;
    d.root.visible=show;d.spool.visible=show;d.channel.g.visible=show&&(view==='tendon'||ex);d.returnChannel.g.visible=show&&(view==='tendon'||ex);
    d.root.position.set(f.x,f.y,0);d.root.rotation.z=f.spread*RAD;
    d.joints.forEach((j,k)=>{j.rotation.x=f.angles[k]*RAD;const p=d.parts[k];p.cover.visible=shell;p.cover.position.z=ex?16+k*4:0;});
    d.spool.rotation.z=f.spool;
    d.flex.g.visible=view!=='skeleton';d.ext.g.visible=view==='tendon'||ex;
    d.flex.update(flexRoute(f,d.root,1));d.ext.update(flexRoute(f,d.root,-1));
    // Fixed proximal guide path, rendered as schematic routing to the basal pivot.
    d.channel.update([V(-28+i*14,-62,10),V(-28+i*14,-45,9),V(f.x,f.y-10,7),V(f.x,f.y, f.r[0])]);
    d.returnChannel.update([V(-28+i*14,-62,-10),V(-28+i*14,-45,-9),V(f.x,f.y-10,-7),V(f.x,f.y,-f.r[0])]);
   }
   ring.visible=grasp==='pinch'||grasp==='tripod';
   if(ring.visible){const p=forward(state[0])[3],q=forward(state[4])[3];const v=[V(...p),V(...q)];tipMarkers.forEach((o,i)=>o.position.copy(v[i]));gap.update(v);}
 }
 return {group,update};
}
