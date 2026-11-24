import torch,json
from pathlib import Path
from gpu_env import Env
from policy import load
from common import *
e=Env(256,140100,64)
p,_=load('../shadow-v13/runs/ppo01/best.pt');p=p.cuda()
e.goal_max=25.;e.multi_fraction=.5
e.reset(torch.ones(e.n,device='cuda',dtype=torch.bool))
assert e.obs().shape==(256,OBS)
assert 0<int(e.plane.sum())<256
for k in range(HORIZON):
 with torch.no_grad():a=p.command(e.obs())
 _,r,d,i=e.step(a,autoreset=False)
 assert torch.isfinite(r).all() and not bool(i['bad'].any())
cap=e.capacity();assert cap['finite'] and cap['contacts']<cap['contactCapacity'] and cap['constraints']<cap['constraintCapacity']
report=dict(n=e.n,retained=int(i['retained'].sum()),success=int(i['success'].sum()),multiWorlds=int(i['plane'].sum()),recontacts=int(i['recontacts'].sum()),maxSelfMM=float(i['selfMM'].max()),maxObjectMM=float(i['objMM'].max()),capacity=cap)
print('SMOKE',report,flush=True)
Path('runs/smoke-report.json').write_text(json.dumps(report,indent=2))
assert report['retained']>=.9*e.n
