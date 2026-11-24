"""Compare learned feedback against a time-only distilled open-loop schedule.
The schedule uses independent calibration seeds, never held-out test trajectories.
"""
import json
from pathlib import Path
import numpy as np,torch
from assess import load_policy
from env import Hand,ACT,DT
from train import evaluate,writejson
p=Path('runs/ppo01')
policy,ck=load_policy(p/'best.pt')
calibration=list(range(300000,300032))
_,_,traj=evaluate(policy,calibration,1.,150,True)
a=[]
for k in range(150):
    rows=[e[k]['action'] for e in traj if len(e)>k]
    a.append(np.mean(rows,axis=0) if rows else a[-1])
schedule=np.stack(a).astype('float32')
class OpenLoop:
    def act(self,obs,deterministic=True):
        # Phase observation = min(t/2,1.5). After 3s, use fixed final residual.
        k=np.clip(np.rint(obs[:,-3].numpy()*2/DT).astype(int),0,len(schedule)-1)
        result=torch.as_tensor(schedule[k]);return None,result,None,None
# Match the policy's phase cap: for t>=3s, use 3s mean rather than unavailable time.
report={}
for difficulty in [1.,1.5]:
    seeds=list(range(20270000,20270096))
    score,rows,_=evaluate(OpenLoop(),seeds,difficulty,150)
    report[str(difficulty)]=dict(openLoop=score,trials=rows)
    print('OPEN_LOOP',difficulty,score,flush=True)
writejson(p/'ablation.json',dict(checkpointUpdate=ck.get('update'),calibrationSeeds=calibration,results=report))
