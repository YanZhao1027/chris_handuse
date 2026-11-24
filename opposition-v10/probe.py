import json,math
from pathlib import Path
import torch,numpy as np
from manipulate_rod import RodManipHand,MAN_ACT
from cpu_eval_rod import evaluate
n=64;seeds=list(range(1004000,1004000+n));angles=[0.]*n
e=RodManipHand(n,seed=1004000,exact_seeds=seeds,difficulty=.35,goal_min=0.,goal_max=0.)
a=torch.zeros((n,MAN_ACT),device='cuda')
for k in range(150):
    _,_,_,i=e.step(a,autoreset=False)
    if k in [24,25,49,50,99,149]:
        print('GPU_PROBE',k+1,'opposed',int(i['opposed'].sum()),'cos',float(i['oppositionCos'].mean()),'loads',e.load_t.mean(0).cpu().tolist(),'retained',int(i['retained'].sum()),'success',int(i['success'].sum()),'depth',float(i['maxSelfPen'].max())*1000,flush=True)
    if k==24:assert bool(e.eq.all())
    if k==25:assert not bool(e.eq.any())
cap=e.capacities()
assert not bool(i['bad'].any()) and cap['finite'] and cap['maxConstraints']<cap['constraintCapacity'] and cap['contacts']<cap['contactCapacity']
cpu,rows,_=evaluate(None,seeds,angles,difficulty=.35,scale=2.)
r=dict(gpuSuccess=int(i['success'].sum()),gpuRetained=int(i['retained'].sum()),gpuMeanOpposition=float(i['oppositionFraction'].mean()),gpuMaxSelfMM=float(i['maxSelfPen'].max())*1000,gpuMaxObjectMM=float(i['maxObjectPen'].max())*1000,cpu=cpu,capacity=cap,n=n)
print('PARITY',json.dumps(r),flush=True)
Path('runs/gpu-probe.json').write_text(json.dumps(r,indent=2))
assert r['gpuRetained']>=.80*n and r['gpuMeanOpposition']>=.80 and cpu['retained']>=.80 and cpu['meanOppositionFraction']>=.80
