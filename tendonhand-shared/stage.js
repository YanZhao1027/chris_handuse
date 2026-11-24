import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {createHand} from './hand.js';
export const VIEWS={
 front:{pos:[125,55,365],target:[0,13,0],zoom:1},
 three:{pos:[225,95,310],target:[0,13,4],zoom:1},
 detail:{pos:[100,55,330],target:[-20,35,24],zoom:1.17},
 back:{pos:[-105,50,-365],target:[0,13,0],zoom:1},
};
export function createStage(canvas,stage,capture){
 const renderer=new T.WebGLRenderer({canvas,antialias:true,preserveDrawingBuffer:capture,powerPreference:'high-performance'});
 renderer.setClearColor(0xeaf0ef);renderer.outputColorSpace=T.SRGBColorSpace;renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.28;
 const scene=new T.Scene();scene.background=new T.Color(0xeaf0ef);
 const camera=new T.PerspectiveCamera(39,1,1,2200);
 const orbit=new OrbitControls(camera,canvas);orbit.enableDamping=false;orbit.enablePan=true;orbit.minDistance=160;orbit.maxDistance=800;
 scene.add(new T.HemisphereLight(0xf9ffff,0x72818c,2.65));
 const key=new T.DirectionalLight(0xfff6e8,3.35);key.position.set(-130,210,240);scene.add(key);
 const rim=new T.DirectionalLight(0xd5f0ff,1.9);rim.position.set(160,85,-130);scene.add(rim);
 const fill=new T.DirectionalLight(0xffffff,.7);fill.position.set(100,-150,80);scene.add(fill);
 const grid=new T.GridHelper(420,21,0xadbfc1,0xd4dddd);grid.rotation.x=Math.PI/2;grid.position.set(0,0,-78);grid.material.transparent=true;grid.material.opacity=.30;scene.add(grid);
 const hand=createHand(scene);
 function setView(name){let v=VIEWS[name]||VIEWS.three;camera.position.set(...v.pos);orbit.target.set(...v.target);camera.zoom=v.zoom;camera.updateProjectionMatrix();orbit.update();}
 function resize(){let w=Math.max(1,stage.clientWidth),h=Math.max(1,stage.clientHeight);renderer.setPixelRatio(capture?Math.min(devicePixelRatio,1.35):Math.min(devicePixelRatio,1.6));renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}
 function draw(state,view,focus,preset){hand.update(state,view,focus,preset);orbit.update();renderer.render(scene,camera);}
 setView('front');resize();
 return {renderer,camera,orbit,resize,draw,setView,hand};
}
