import argparse,time,json
from pathlib import Path
import torch,warp as wp,numpy as np
from gpu_env import GPUHand,ACT,Hand,domain
p=argparse.ArgumentParser();p.add_argument('--worlds',type=int,default=512);p.add_argument('--steps',type=int,default=110);p.add_argument('--iterations',type=int,default=80);p.add_argument('--exact',action='store_true');a=p.parse_args()
seeds=list(range(100000,100000+a.worlds)) if a.exact else None
e=GPUHand(a.worlds,seed=314159,variants=256,horizon=100,exact_seeds=seeds,iterations=a.iterations)
actions=torch.zeros((a.worlds,ACT),device='cuda')
done_count=success_count=bad_count=0; maxcon=0; maxefc=0
alive=torch.ones(a.worlds,dtype=torch.bool,device='cuda');success=torch.zeros_like(alive)
torch.cuda.synchronize();t=time.perf_counter()
for k in range(a.steps):
 o,r,d,i=e.step(actions,autoreset=not a.exact)
 if a.exact:
  fresh=alive&d;success|=fresh&i['success'];alive&=~d
  if not bool(alive.any()):break
 else:done_count+=int(d.sum());success_count+=int(i['success'].sum())
 bad_count+=int(i['bad'].sum())
 if k%10==0:
  c=e.capacities();maxcon=max(maxcon,c['contacts']);maxefc=max(maxefc,c['maxConstraints'])
  print('progress',k+1,'contacts',c['contacts'],'constraints',c['maxConstraints'],'finite',c['finite'],flush=True)
torch.cuda.synchronize();secs=time.perf_counter()-t
if a.exact:
 done_count=int((~alive).sum());success_count=int(success.sum())
result=dict(worlds=a.worlds,steps=k+1,seconds=secs,policy_steps_per_second=a.worlds*(k+1)/secs,physics_steps_per_second=a.worlds*(k+1)*40/secs,done=done_count,success=success_count,bad=bad_count,maxContacts=maxcon,maxConstraints=maxefc,memFreeGiB=torch.cuda.mem_get_info()[0]/2**30)
print('BENCH',json.dumps(result),flush=True)
assert bad_count==0 and maxefc<e.d.njmax and maxcon<e.d.naconmax,result
if a.exact:
 h=Hand();cpu=[]
 for seed in seeds:
  h.reset(seed)
  for _ in range(100):
   _,_,d,i=h.step(np.zeros(ACT))
   if d:break
  cpu.append(i['success'])
 gpu=success.cpu().numpy().tolist()
 report=dict(result,cpuSuccess=sum(cpu),gpuSuccess=sum(gpu),agreement=sum(x==y for x,y in zip(cpu,gpu)),n=len(cpu),cpu=cpu,gpu=gpu)
 print('CPU_GPU',json.dumps({k:v for k,v in report.items() if k not in ('cpu','gpu')}),flush=True)
 Path('runs/parity.json').write_text(json.dumps(report,indent=2))
else:
 Path(f'runs/bench-{a.worlds}-{a.iterations}.json').write_text(json.dumps(result,indent=2))
