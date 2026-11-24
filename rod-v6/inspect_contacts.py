import torch,numpy as np,mujoco as mj,warp as wp
from manipulate_rod import RodManipHand,MAN_ACT
e=RodManipHand(1,seed=606,difficulty=0,variants=1,goal_min=15,goal_max=15)
h=e.cpu
print('ball',e.ball,'floor',e.floor,'map',h.geom_digit)
for k in range(h.m.ngeom):
 print(k,mj.mj_id2name(h.m,mj.mjtObj.mjOBJ_GEOM,k),int(e.mapping.numpy()[k]))
zero=torch.zeros((1,MAN_ACT),device='cuda')
for i in range(16):
 e.step(zero,autoreset=False)
 if i in [13,14,15]:
  n=int(e.d.nacon.numpy()[0]);print('\nSTEP',i+1,'nacon',n)
  ge=e.d.contact.geom.numpy();wo=e.d.contact.worldid.numpy();fo=e.contact_force.numpy()
  for j in range(n):
   a,b=ge[j]
   if wo[j]==0:
    print(j,a,mj.mj_id2name(h.m,mj.mjtObj.mjOBJ_GEOM,int(a)),b,mj.mj_id2name(h.m,mj.mjtObj.mjOBJ_GEOM,int(b)),'map',e.mapping.numpy()[int(a)],e.mapping.numpy()[int(b)],'f',fo[j])
  print('reduced',e.force_t[0].cpu().numpy())
