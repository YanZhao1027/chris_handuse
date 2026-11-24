import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';
const {chromium}=createRequire(process.env.HOME+'/.agents/skills/teardown/package.json')('playwright');
const browser=await chromium.launch({args:['--enable-webgl','--ignore-gpu-blocklist']});
try{
 const p=await browser.newPage({viewport:{width:1600,height:900},deviceScaleFactor:1});
 const errors=[],external=[];p.on('pageerror',e=>errors.push(e.message));p.on('request',r=>{if(/^https?:/.test(r.url()))external.push(r.url());});
 await p.goto(pathToFileURL(path.resolve('teardown-out/dist/handuse-longshaft-preview.html')).href);
 await p.waitForFunction(()=>window.HANDUSE?.ready);
 await p.locator('#timeline').fill('4');
 await p.waitForFunction(()=>window.HANDUSE.frame.t>=3.96);
 const state=await p.evaluate(()=>({opposition:window.HANDUSE.frame.opposition,fixture:window.HANDUSE.frame.fixture,shaft:window.HANDUSE.geometry.find(g=>g.name==='ball'),summary:window.HANDUSE.summary}));
 assert.equal(state.fixture,false);assert.equal(state.opposition.opposed,true);
 assert.equal(state.shaft.type,'cylinder');assert.ok(state.shaft.size[1]*2>210&&state.shaft.size[1]*2<230);
 assert.ok(state.summary.geometryAudit.maxSelfPenetrationMM<1);assert.ok(state.summary.geometryAudit.maxRodPenetrationMM<2);
 await p.locator('#front').click();await p.screenshot({path:'runs/preview-front.png'});
 await p.locator('#rear').click();await p.screenshot({path:'runs/preview-thumb-side.png'});
 assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
 console.log('PASS long shaft preview',state.shaft,state.opposition);
}finally{await browser.close();}
