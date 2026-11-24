import numpy as np,torch,mujoco as mj
from manipulate_rod import RodManipHand,MAN_ACT
from gpu_rod_base import BASE,SCALE,START_YAW,DT,SUB
torch.set_num_threads(1)
e=RodManipHand(1,seed=606,difficulty=0,variants=1,goal_min=15,goal_max=15)
h=e.cpu;d=h.d;m=h.m;last=np.zeros(6)
zero=torch.zeros((1,MAN_ACT),device='cuda')
for k in range(60):
 with torch.no_grad():raw=e.policy.actor(torch.tensor(h.obs()[None],device='cuda')).tanh().cpu().numpy()[0]
 last=.65*last+.35*raw;u=BASE+SCALE*last
 if k==25:h.release_center=h.relative().copy();d.eq_active[h.eq]=0
 for sub in range(SUB):
  t=(k*SUB+sub)*.001;a=np.clip(t/.7,0,1);a=a*a*(3-2*a);l=np.clip((t-1.10)/.85,0,1);l=l*l*(3-2*l)
  d.ctrl[h.yaws]=(1-a)*START_YAW+a*np.radians([2,1,-1,-3,u[5]])
  d.ctrl[h.pulls]=a*u[:5];d.mocap_pos[h.wrist,2]=.25+.065*l;mj.mj_step(m,d)
 _,_,done,info=e.step(zero,autoreset=False)
 if k in [0,5,10,14,15,19,20,24,25,26,30,34,35,40,49,50,59]:
  cp=d.qpos[h.obj_q:h.obj_q+7];gp=e.q[0,e.obj_q:e.obj_q+7].cpu().numpy()
  cf=h.contacts()[0];gf=e.force_t[0].cpu().numpy()
  print('k',k,'qerr',np.round(cp-gp,5),'cpu_z',round(d.qpos[h.obj_q+2],5),'gpu_z',round(float(e.q[0,e.obj_q+2]),5),'force',np.round(cf,2),np.round(gf,2),'last',np.round(last,3),flush=True)
