import json,math,time
from pathlib import Path
import numpy as np,torch
from manipulate import ManipHand,MAN_ACT
search=json.load(open('runs/action-search.json'))['rows']
base=[r['action'] for r in search if r['success']>0][:8]
# Add sign-reversed policies and zero, plus low-amplitude versions.
policies=[]
for a in base:
 policies.append(np.asarray(a,np.float32))
 policies.append(-np.asarray(a,np.float32))
policies.append(np.zeros(10,np.float32))
P=len(policies);rep=64;n=P*rep
goals=np.array([(-1 if j%2 else 1)*[15,25,40,60][(j//2)%4] for j in range(rep)])
acts=np.repeat(np.stack(policies),rep,axis=0)
for shape in ['sphere','ellipsoid']:
 e=ManipHand(n,seed=606,variants=1024,shape=shape,goal_min=15,goal_max=60,horizon=150)
 e.fixed_goals=torch.tensor(np.tile(goals,P),device='cuda')*math.pi/180
 e.reset(torch.ones(n,dtype=torch.bool,device='cuda'),initial=True)
 a=torch.tensor(acts,device='cuda')
 t=time.perf_counter()
 for k in range(150):_,_,_,info=e.step(a,autoreset=False)
 res={k:v.detach().cpu().numpy() for k,v in info.items() if isinstance(v,torch.Tensor)}
 rows=[]
 for i,policy in enumerate(policies):
  sl=slice(i*rep,(i+1)*rep)
  rows.append(dict(shape=shape,id=i,action=policy.tolist(),success=float(res['success'][sl].mean()),retained=float(res['retained'][sl].mean()),grasped=float(res['grasped'][sl].mean()),meanError=float(res['error'][sl].mean()*180/math.pi),bestError=float(res['bestError'][sl].mean()*180/math.pi),meanDrift=float(res['drift'][sl].mean()*1000)))
 rows.sort(key=lambda r:(r['success'],r['retained'],-r['meanError']),reverse=True)
 print('TEACHER',shape,'elapsed',time.perf_counter()-t,json.dumps(rows,indent=2),flush=True)
 Path('runs/teacher-'+shape+'.json').write_text(json.dumps(rows,indent=2))
