import math,json
from pathlib import Path
import numpy as np,torch,mujoco as mj
from cpu_eval_rod import CPURod
from rod_shape import audit_pairs,THUMB_X,THUMB_Y,THUMB_Z
from train import Policy
from manipulate_rod import PRETRAINED
torch.set_num_threads(1)
p=Policy();p.load_state_dict(torch.load(PRETRAINED,map_location='cpu',weights_only=True)['policy']);p.eval()
teacher=json.loads(Path('../rod-v8/runs/teachers.json').read_text())
rows=[]
for seed in range(913000,913008):
 goal=-15 if seed%2==0 else 15
 h=CPURod(seed,goal,.35,2.)
 m=h.h.m;d=h.d;pairs=audit_pairs(m)
 def audit():
  return min(mj.mj_geomDistance(m,d,a,b,.2,None) for a,b in pairs)
 initial=audit();worst=initial;peak=0
 while not h.done:
  with torch.no_grad():a=p.actor(torch.tensor(h.baseobs()[None])).tanh().numpy()[0]
  h.step(a,np.array(teacher[str(goal)][0]['action']))
  mj.mj_forward(m,d)
  worst=min(worst,audit());peak=max(peak,d.ncon)
 rows.append(dict(goal=goal,initial=initial,worst=worst,maxContacts=peak,**h.final))
 print('AUDIT',seed,goal,'initial mm',round(initial*1000,3),'worst mm',round(worst*1000,3),'grasp',h.final['grasped'],'retained',h.final['retained'],'success',h.final['success'],'error',round(h.final['errorDegrees'],1),'contacts',peak,flush=True)
print('SUMMARY',THUMB_X,THUMB_Y,THUMB_Z,'retained',sum(r['retained'] for r in rows),'success',sum(r['success'] for r in rows),'worst mm',1000*min(r['worst'] for r in rows),flush=True)
