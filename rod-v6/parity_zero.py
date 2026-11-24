import numpy as np,torch,mujoco as mj
from manipulate_rod import RodManipHand,MAN_ACT
from gpu_rod_base import BASE,SCALE,START_YAW,DT,SUB
torch.set_num_threads(1)
e=RodManipHand(1,seed=606,difficulty=0,variants=1,goal_min=15,goal_max=15)
h=e.cpu;d=h.d;m=h.m;last=np.zeros(6)
for par in e.policy.actor.parameters():par.data.zero_()
zero=torch.zeros((1,MAN_ACT),device='cuda')
for k in range(60):
 _,_,cd,ci=h.step(np.zeros(6))
 _,_,done,info=e.step(zero,autoreset=False)
 if k in [0,5,10,14,15,19,20,24,25,26,30,34,35,40,49,50,59]:
  cp=d.qpos[h.obj_q:h.obj_q+7];gp=e.q[0,e.obj_q:e.obj_q+7].cpu().numpy()
  cf=h.contacts()[0];gf=e.force_t[0].cpu().numpy()
  print('k',k,'qerr',np.round(cp-gp,5),'cpu_z',round(d.qpos[h.obj_q+2],5),'gpu_z',round(float(e.q[0,e.obj_q+2]),5),'force',np.round(cf,2),np.round(gf,2),'last',np.round(h.last,3),flush=True)
