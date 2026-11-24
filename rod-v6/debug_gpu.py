import torch,numpy as np
from manipulate_rod import RodManipHand,MAN_ACT
e=RodManipHand(8,seed=606,difficulty=0,variants=8,goal_min=15,goal_max=15,horizon=150)
b=e.ball
for n in ['geom_type','geom_size','geom_aabb','geom_rbound','geom_quat','geom_pos','geom_contype','geom_conaffinity']:
 a=getattr(e.m,n);arr=a.numpy();print(n,a.shape,arr.shape,arr[b] if arr.ndim==1 else arr[0,b],flush=True)
print('cpu model',e.cpu.m.geom_type[b],e.cpu.m.geom_size[b],e.cpu.m.geom_aabb[b],e.cpu.m.geom_rbound[b],e.cpu.m.geom_quat[b],e.cpu.m.geom_pos[b])
print('init q',e.q[0,e.obj_q:e.obj_q+7].cpu().numpy(),'wrist',e.mocap[0,e.wrist].cpu().numpy())
zero=torch.zeros((8,MAN_ACT),device='cuda')
for k in range(50):
 o,r,d,i=e.step(zero,autoreset=False)
 if k in [14,19,24,25,26,30,34,40,49]:
  print('k',k,'z',e.q[:,e.obj_q+2].mean().item(),'pos',e.q[0,e.obj_q:e.obj_q+3].cpu().numpy(),'force',e.force_t.mean(0).cpu().numpy(),'total',e.total_t.mean().item(),'nacon',e.d.nacon.numpy()[0],'contacts',e.d.contact.geom.numpy()[:12],'world',e.d.contact.worldid.numpy()[:12],flush=True)
