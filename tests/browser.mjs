import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import fs from 'node:fs';
import assert from 'node:assert/strict';
const req=createRequire(process.env.HOME+'/.agents/skills/teardown/package.json');
const {chromium}=req('playwright');
const browser=await chromium.launch({args:['--ignore-gpu-blocklist','--enable-webgl','--force_high_performance_gpu']});
const errors=[],net=[];
const page=await browser.newPage({viewport:{width:1600,height:900},deviceScaleFactor:1});
page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))net.push(r.url());});
const html=name=>pathToFileURL(path.resolve('teardown-out/dist/'+name+'.html')).href;
fs.mkdirSync('teardown-out/review',{recursive:true});
try{
 await page.goto(html('tendonhand-live'));await page.waitForFunction(()=>window.SIM?.ready&&window.HAND?.state?.length===5);
 await page.locator('[data-preset=power]').click();await page.waitForFunction(()=>window.HAND.preset==='power'&&window.HAND.state[0].stroke>10);
 assert.equal(await page.locator('#status').textContent(),'MANUAL');
 await page.locator('[data-view=tendon]').click();await page.locator('[data-focus=index]').click();await page.waitForFunction(()=>window.HAND.view==='tendon'&&window.HAND.focus==='index');
 await page.locator('[data-preset=pinch]').click();await page.locator('[data-focus=all]').click();await page.waitForFunction(()=>window.HAND.preset==='pinch'&&window.HAND.focus==='all');
 const gap=await page.evaluate(()=>window.HAND.pinchGap);assert.ok(gap>5&&gap<15);console.log('pinch center gap',gap);
 await page.locator('#grip').fill('0.25');await page.waitForFunction(()=>window.HAND.preset==='custom'&&window.HAND.controls.grip===.25);
 await page.locator('[data-view=exploded]').click();await page.waitForFunction(()=>window.HAND.view==='exploded');
 await page.screenshot({path:'teardown-out/review/live-interaction.png'});
 await page.locator('[data-action=tour]').click();await page.waitForFunction(()=>window.HAND.preset==='open');
 console.log('PASS live presets, sliders, views, digit isolation, resume tour');
 const capture=await browser.newPage({viewport:{width:1600,height:900},deviceScaleFactor:1});capture.on('pageerror',e=>errors.push(e.message));capture.on('request',r=>{if(/^https?:/.test(r.url()))net.push(r.url());});
 await capture.addInitScript(()=>window.__CAPTURE=true);await capture.goto(html('tendonhand-film'));await capture.waitForFunction(()=>window.SIM?.ready&&window.HAND);
 for(const [i,t] of [1.8,6.2,10.1,14.1,18.1,22.1].entries()){
   await capture.evaluate(t=>window.SIM.render(t),t);await capture.screenshot({path:'teardown-out/review/beat-'+i+'.png'});console.log('frame',t,await capture.evaluate(()=>[window.HAND.view,window.HAND.focus,window.HAND.preset]));
 }
 const a=await capture.evaluate(()=>{window.SIM.render(14.1);return JSON.stringify(window.HAND);});
 const b=await capture.evaluate(()=>{window.SIM.render(2);window.SIM.render(14.1);return JSON.stringify(window.HAND);});
 assert.equal(a,b);assert.deepEqual(errors,[]);assert.deepEqual(net,[]);console.log('PASS deterministic film state, no page errors, no network requests');
 const gpu=await capture.evaluate(()=>{const gl=document.createElement('canvas').getContext('webgl2');const e=gl.getExtension('WEBGL_debug_renderer_info');return gl.getParameter(e.UNMASKED_RENDERER_WEBGL);});console.log('GPU',gpu);
}finally{await browser.close();}
