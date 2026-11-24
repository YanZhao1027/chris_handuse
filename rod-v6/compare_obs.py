import torch,numpy as np
from manipulate_rod import RodManipHand
e=RodManipHand(1,seed=606,difficulty=0,variants=1,goal_min=15,goal_max=15)
cpu=e.cpu
go=e.obs()[0,:68].cpu().numpy();co=cpu.obs()
print('obs max abs',np.max(np.abs(go-co)),'different',[(i,round(float(go[i]),4),round(float(co[i]),4)) for i in np.where(np.abs(go-co)>1e-4)[0][:40]])
with torch.no_grad():
 ga=e.policy.actor(torch.tensor(go[None],device='cuda')).tanh().cpu().numpy()[0]
 ca=e.policy.actor(torch.tensor(co[None],device='cuda')).tanh().cpu().numpy()[0]
print('GPU actor',ga,'CPU actor',ca,'diff',ga-ca)
print('qpos',e.q[0].cpu().numpy()[:7],cpu.d.qpos[:7])
print('qvel',e.v[0].cpu().numpy()[:7],cpu.d.qvel[:7])
print('ten',e.ten[0].cpu().numpy(),cpu.d.ten_length)
print('mass',e.m.body_mass.numpy()[0,e.oid],cpu.m.body_mass[cpu.oid])
print('inertia',e.m.body_inertia.numpy()[0,e.oid],cpu.m.body_inertia[cpu.oid])
print('size',e.m.geom_size.numpy()[0,e.ball],cpu.m.geom_size[cpu.ball])
print('aabb',e.m.geom_aabb.numpy()[0,e.ball],cpu.m.geom_aabb[cpu.ball])
print('rbound',e.m.geom_rbound.numpy()[0,e.ball],cpu.m.geom_rbound[cpu.ball])
print('quat',e.m.geom_quat.numpy()[0,e.ball],cpu.m.geom_quat[cpu.ball])
