"""Final untouched seed sets, including held-out longer angles and stronger domain perturbation."""
import json,torch
from pathlib import Path
from policy import load
from reference import cases,evaluate
from search_grasp import write
torch.set_num_threads(1)
out=Path('runs/ppo01')
p,c=load(out/'best.pt');w,wc=load(out/'best-wide.pt');old,oldc=load('../shadow-v13/runs/ppo01/best.pt')
report=dict(bestUpdate=c['update'],wideUpdate=wc['update'],tests=[])
for seed,n,angles,diff in [(142000,96,(-15,15),1.),(143000,96,(-30,30),1.),(144000,96,(-45,45),1.),(145000,96,((-15,1),(15,1),(-25,1),(25,1)),1.),(146000,96,((-15,0),(15,0),(-30,0),(30,0),(-45,0),(45,0),(-15,1),(15,1),(-25,1),(25,1)),1.5)]:
    cs=cases(seed,n,angles,diff);r=dict(seed=seed,angles=angles,difficulty=diff,results={})
    for name,pol in [('inherited-V13-under-V14-controls',old),('PPO',p),('PPO-wide',w)]:
        s,tr=evaluate(pol,cs);r['results'][name]=dict(summary=s,trials=tr);print('BLIND',seed,name,json.dumps(s),flush=True)
    report['tests'].append(r);write(out/'blind.json',report)
    for name,pol in [('PPO',p),('PPO-wide',w)]:
        rows=r['results'][name]['trials']
        for sign in [-1,1]:
            idx=next((i for i,v in enumerate(rows) if v['success'] and v['recontacts']>0 and v['angle']*sign>0),None)
            if idx is None:idx=next((i for i,v in enumerate(rows) if v['success'] and v['angle']*sign>0),None)
            if idx is not None:
                score,tr=evaluate(pol,[cs[idx]],record=True)
                write(out/f'replay-{seed}-{name}-{sign}.json',dict(summary=tr[0],frames=cs[idx].frames,selection='representative successful blind case; recontact preferred'))

print('ASSESS_COMPLETE',flush=True)
