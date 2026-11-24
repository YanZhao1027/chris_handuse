import sys,math,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'rl-v3'))
from env import BASE,SCALE,START_YAW,DT,SUB
from rod_shape import RodHand
from train import Policy
import torch,numpy as np,mujoco as mj
torch.set_num_threads(1)
p=Policy();p.load_state_dict(torch.load('../gpu-v4/runs/ppo-gpu01/best.pt',map_location='cpu',weights_only=True)['policy']);p.eval()
for seed in range(3):
 h=RodHand();h.reset(seed,0)
 d=h.d;m=h.m
 print('PARAMS',h.params,'geomsize',m.geom_size[h.ball],'aabb',m.geom_aabb[h.ball],'quat',m.geom_quat[h.ball], 'inertia',m.body_inertia[h.oid])
 last=np.zeros(6)
 for k in range(150):
  with torch.no_grad():raw=p.actor(torch.tensor(h.obs()[None])).tanh().numpy()[0]
  last=.65*last+.35*raw
  u=BASE+SCALE*last
  if k==25:h.release_center=h.relative().copy();d.eq_active[h.eq]=0
  for sub in range(SUB):
   t=(k*SUB+sub)*.001;a=min(1,t/.7);a=a*a*(3-2*a);l=np.clip((t-1.10)/.85,0,1);l=l*l*(3-2*l)
   d.ctrl[h.yaws]=(1-a)*START_YAW+a*np.radians([2,1,-1,-3,u[5]])
   d.ctrl[h.pulls]=a*u[:5];d.mocap_pos[h.wrist,2]=.25+.065*l
   mj.mj_step(m,d)
  f,n,tot,pts=h.contacts(True)
  if k in [0,5,10,15,20,24,25,26,30,35,40,49,50,60,75,100,149]:
   print(seed,k,'z',round(d.xpos[h.oid,2],4),'pos',np.round(d.xpos[h.oid],3),'n',n,'digits',[(i,round(x,2)) for i,x in enumerate(f)],'total',round(tot,2),'ctrl',np.round(u,1),'q',np.round(np.degrees(d.qpos[h.qj[:,1:]]),0).astype(int).tolist())
  if k>30 and d.xpos[h.oid,2]<.09:break
