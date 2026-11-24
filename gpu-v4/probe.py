import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'rl-v3'))
from env import Hand,CFG
import mujoco_warp as mw,warp as wp,mujoco as mj,torch,numpy as np
wp.init()
wp.set_device('cuda:0')
print('CUDA',torch.cuda.get_device_name(0),torch.version.cuda,flush=True)
h=Hand();h.reset(0,0)
m=mw.put_model(h.m)
for k in ['body_mass','body_inertia','body_invweight0','body_subtreemass','dof_invweight0','geom_size','geom_aabb','geom_rbound','geom_friction','qpos0','body_pos','actuator_acc0','tendon_invweight0']:
 a=getattr(m,k);print(k,a.shape,a.dtype,flush=True)
d=mw.put_data(h.m,h.d,nworld=32,nconmax=32,njmax=128)
print('data',d.qpos.shape,d.eq_active.shape,d.mocap_pos.shape,d.contact.geom.shape,d.efc.force.shape,flush=True)
print('start warmup',flush=True)
t=time.perf_counter();mw.step(m,d);wp.synchronize();print('first step',time.perf_counter()-t,'qpos finite',np.isfinite(d.qpos.numpy()).all(),flush=True)
print('start graph capture',flush=True)
with wp.ScopedCapture() as cap:
 mw.step(m,d)
g=cap.graph
t=time.perf_counter()
for _ in range(100):wp.capture_launch(g)
wp.synchronize()
print('graph 32 worlds/100 steps',time.perf_counter()-t, 'sim time',d.time.numpy()[:3],flush=True)
print('PROBE_OK',flush=True)
