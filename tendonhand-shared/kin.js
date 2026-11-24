export const RAD = Math.PI/180;
export const clamp = (v,a=0,b=1)=>Math.max(a,Math.min(b,v));
export const FINGERS = [
 {id:'index',name:'Index',x:-29,y:27,w:15,lengths:[43,26,20],max:[65,95,65],r:[5.2,4.2,3.2],spread:13},
 {id:'middle',name:'Middle',x:-9,y:32,w:16,lengths:[47,29,21],max:[70,100,68],r:[5.5,4.4,3.4],spread:4},
 {id:'ring',name:'Ring',x:11,y:28,w:15,lengths:[44,27,19],max:[72,100,68],r:[5.2,4.2,3.2],spread:-5},
 {id:'little',name:'Little',x:29,y:18,w:12.5,lengths:[34,21,16],max:[75,95,62],r:[4.6,3.5,2.8],spread:-16},
 {id:'thumb',name:'Thumb',x:-40,y:-21,w:19,lengths:[34,27,22],max:[36,60,65],r:[6,5,3.8],spread:0},
];
export const PRESETS = {
 open:{grip:0,spread:.65,oppose:0,trim:[0,0,0,0,0]},
 power:{grip:.86,spread:.12,oppose:.65,trim:[-.02,.02,.05,.08,-.20]},
 pinch:{grip:0,spread:.10,oppose:.78,trim:[.70,0,0,0,.60]},
 tripod:{grip:0,spread:.12,oppose:.92,trim:[.58,.55,.05,.05,.54]},
 point:{grip:.82,spread:.1,oppose:.55,trim:[-.82,0,0,0,-.25]},
};
export function controls(p='open'){const c=PRESETS[p]||PRESETS.open;return {...c,trim:[...c.trim]};}
export function mix(a,b,t){return {grip:a.grip+(b.grip-a.grip)*t,spread:a.spread+(b.spread-a.spread)*t,oppose:a.oppose+(b.oppose-a.oppose)*t,trim:a.trim.map((x,i)=>x+(b.trim[i]-x)*t)};}
export function solve(c){
 return FINGERS.map((f,i)=>{
   const u=clamp(c.grip+c.trim[i]); const isThumb=i===4;
   // One cable drives a prescribed coupled posture. No dynamics or contact solver.
   const angles=f.max.map((m,j)=>m*Math.pow(u,[.96,1.05,1.12][j]));
   const spread=isThumb?62-100*clamp(c.oppose):f.spread*clamp(c.spread);
   const stroke=angles.reduce((s,x,j)=>s+x*RAD*f.r[j],0);
   // Effective illustrative series elastic spring, not a measured force.
   const tension=1.2+0.38*stroke;
   return {...f,u,angles,spread,stroke,tension,spool:stroke/8};
 });
}
export function forward(f){
 let phi=0,p=[f.x,f.y,0]; const points=[p.slice()]; const a=f.spread*RAD;
 for(let i=0;i<3;i++){
   phi+=f.angles[i]*RAD; const l=f.lengths[i];
   p=[p[0]-Math.sin(a)*Math.cos(phi)*l,p[1]+Math.cos(a)*Math.cos(phi)*l,p[2]+Math.sin(phi)*l]; points.push(p);
 }
 return points;
}
export function pinchGap(state){const a=forward(state[0])[3],b=forward(state[4])[3];return Math.hypot(...a.map((v,i)=>v-b[i]));}
