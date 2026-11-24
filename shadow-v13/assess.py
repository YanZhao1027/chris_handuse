"""Final untouched seed sets, including held-out longer angles and stronger domain perturbation."""
import json,torch
from pathlib import Path
from policy import load
from reference import cases,evaluate
from search_grasp import write
torch.set_num_threads(1)
out=Path('runs/ppo01')
p,c=load(out/'best.pt');w,wc=load(out/'best-wide.pt')
report=dict(bestUpdate=c['update'],wideUpdate=wc['update'],tests=[])
for seed,n,angles,diff in [(136000,96,(-15,15),1.),(137000,96,(-30,30),1.),(138000,96,(-15,15,-30,30,-45,45),1.5)]:
    cs=cases(seed,n,angles,diff);r=dict(seed=seed,angles=angles,difficulty=diff,results={})
    for name,pol in [('hold',None),('PPO',p),('PPO-wide',w)]:
        s,tr=evaluate(pol,cs);r['results'][name]=dict(summary=s,trials=tr);print('BLIND',seed,name,json.dumps(s),flush=True)
    report['tests'].append(r);write(out/'blind.json',report)
print('ASSESS_COMPLETE',flush=True)
