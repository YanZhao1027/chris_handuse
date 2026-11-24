import argparse,math,json,time
import numpy as np,torch
from manipulate import ManipHand,MAN_ACT,qerror,quat_mul
p=argparse.ArgumentParser();p.add_argument('--shape',default='sphere');p.add_argument('--worlds',type=int,default=256);a=p.parse_args()
e=ManipHand(a.worlds,shape=a.shape,exact_seeds=list(range(510000,510000+a.worlds)))
e.fixed_goals=torch.where(torch.arange(a.worlds,device='cuda')%2==0,math.pi/6,-math.pi/6)
e.reset(torch.ones(a.worlds,dtype=torch.bool,device='cuda'),initial=True)
alive=torch.ones(a.worlds,dtype=torch.bool,device='cuda')
res={k:0 for k in ('success','retained','grasped','bad')}
errors=[];ts=time.perf_counter()
for k in range(e.horizon):
 o,r,d,i=e.step(torch.zeros((a.worlds,MAN_ACT),device='cuda'),autoreset=False,active_mask=alive)
 fresh=alive&d
 for key in res:res[key]+=int((i[key]&fresh).sum())
 errors+=i['error'][fresh&i['retained']].cpu().tolist()
 alive&=~d
 if k in (0,24,25,49,50,75,100,149):
  print('STEP',k+1,'alive',int(alive.sum()),'grasped',int(i['grasped'].sum()),'mean error deg',float(i['error'][alive].mean())*180/math.pi if alive.any() else None,'fixture',int(i['fixture'].sum()),'capacity',e.capacities(),flush=True)
 if k==24:assert bool(e.eq.all())
 if k==25:assert not bool(e.eq.any())
 if k==49:assert not bool(e.started.any())
 if k==50:
  assert bool(e.started[alive].all())
 if not bool(alive.any()):break
print('MANIP_PROBE',json.dumps(dict(shape=a.shape,worlds=a.worlds,elapsed=time.perf_counter()-ts,meanRetainedError=float(np.mean(errors))*180/math.pi if errors else None,**res)),flush=True)
assert res['bad']==0 and res['grasped']>a.worlds*.35
