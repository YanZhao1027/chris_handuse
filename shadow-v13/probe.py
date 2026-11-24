import argparse,json,time,math
from pathlib import Path
import torch
from gpu_env import Env
from common import ACT,OBS,HORIZON
p=argparse.ArgumentParser();p.add_argument('--worlds',type=int,default=256);a=p.parse_args()
e=Env(a.worlds,seed=134000)
assert e.obs().shape==(a.worlds,OBS)
u=torch.zeros((a.worlds,ACT),device='cuda')
start=time.perf_counter()
for k in range(HORIZON):
 o,r,d,i=e.step(u,autoreset=False)
 if (k+1)%50==0:
  cap=e.capacity();assert cap['finite'] and cap['contacts']<cap['contactCapacity'] and cap['constraints']<cap['constraintCapacity']
  print('PROBE',k+1,'opposed',int(i['opposed'].sum()),'dropped',int(i['dropped'].sum()),'maxselfmm',float(i['selfMM'].max()),'maxobjmm',float(i['objMM'].max()),'capacity',cap,flush=True)
sec=time.perf_counter()-start
rep=dict(worlds=a.worlds,steps=HORIZON,retained=int(i['retained'].sum()),meanOpposition=float(i['opposition'].mean()),maxSelfMM=float(i['selfMM'].max()),maxObjectMM=float(i['objMM'].max()),seconds=sec,stepsPerSecond=a.worlds*HORIZON/sec,capacity=e.capacity())
Path(f'runs/probe-{a.worlds}.json').write_text(json.dumps(rep,indent=2))
print('PROBE_COMPLETE',rep,flush=True)
assert rep['retained']>=.85*a.worlds and rep['meanOpposition']>.8
