import argparse,math,json,time
import numpy as np,torch
from manipulate_rod import RodManipHand,MAN_ACT
p=argparse.ArgumentParser();p.add_argument('--worlds',type=int,default=256);p.add_argument('--angle',type=float,default=15.);p.add_argument('--difficulty',type=float,default=1.);a=p.parse_args()
e=RodManipHand(a.worlds,seed=606,variants=256,difficulty=a.difficulty,goal_min=a.angle,goal_max=a.angle,horizon=150)
e.fixed_goals=torch.where(torch.arange(a.worlds,device='cuda')%2==0,1.,-1.)*math.radians(a.angle)
e.reset(torch.ones(a.worlds,dtype=torch.bool,device='cuda'),initial=True)
zero=torch.zeros((a.worlds,MAN_ACT),device='cuda')
for k in range(e.horizon):
 o,r,d,i=e.step(zero,autoreset=False)
 if k in [0,24,25,49,50,75,100,149]:
  print('STEP',k+1,'alive',int((~d).sum()),'grasped',int(i['grasped'].sum()),'retained',int(i['retained'].sum()),'error',float(i['error'].mean()*180/math.pi),'z',float(i['height'].mean()),'contacts',int(i['holdContacts'].sum()),'capacity',e.capacities(),flush=True)
 if k==24:assert bool(e.eq.all())
 if k==25:assert not bool(e.eq.any())
 if k==50:assert bool(e.started.all())
 if k==149:
  print('SUMMARY',json.dumps({k:float(v.float().mean()) if v.dtype==torch.bool else float(v.float().mean()) for k,v in i.items() if isinstance(v,torch.Tensor)},indent=2),flush=True)
assert int(i['bad'].sum())==0
