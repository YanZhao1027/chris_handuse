"""Run official Menagerie E3M5 scene without changing solver or actuators."""
import json,time
from pathlib import Path
import mujoco as mj,mujoco_warp as mw,warp as wp,numpy as np,torch
m=mj.MjModel.from_xml_path('.assets/shadow_hand/scene_right.xml');d=mj.MjData(m)
mj.mj_forward(m,d)
wp.init();wp.set_device('cuda:0')
n=256;gm=mw.put_model(m);gd=mw.put_data(m,d,nworld=n,nconmax=128,njmax=512)
q=wp.to_torch(gd.qpos);v=wp.to_torch(gd.qvel);ctrl=wp.to_torch(gd.ctrl)
lo=torch.tensor(m.actuator_ctrlrange[:,0],device='cuda',dtype=torch.float32);hi=torch.tensor(m.actuator_ctrlrange[:,1],device='cuda',dtype=torch.float32)
ctrl.copy_(torch.zeros_like(ctrl).clamp(lo,hi))
for _ in range(2):mw.step(gm,gd)
wp.synchronize()
with wp.ScopedCapture() as cap:mw.step(gm,gd)
start=time.perf_counter()
for k in range(500):
    wp.capture_launch(cap.graph)
wp.synchronize();sec=time.perf_counter()-start
finite=bool(torch.isfinite(q).all()&torch.isfinite(v).all())
assert finite
r=dict(worlds=n,steps=500,physicsSteps=n*500,seconds=sec,physicsStepsPerSecond=n*500/sec,finite=finite,actuators=m.nu,dt=m.opt.timestep,model='unmodified Menagerie E3M5 scene_right.xml; no grasp or policy claim')
Path('runs/shadow-gpu-probe.json').write_text(json.dumps(r,indent=2))
print('SHADOW_GPU_PROBE',r,flush=True)
