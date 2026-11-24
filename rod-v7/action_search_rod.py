"""GPU parallel fixed-action screening for controllable long-rod rotation."""
import math,json,time,argparse
from pathlib import Path
import numpy as np,torch
from manipulate_rod import RodManipHand,MAN_ACT

def candidates(n,seed=707):
 rng=np.random.default_rng(seed);out=[np.zeros(10)]
 levels=[-.75,-.5,-.25,.25,.5,.75]
 for i in range(10):
  for a in levels: 
   v=np.zeros(10);v[i]=a;out.append(v)
 for i in range(10):
  for j in range(i+1,10):
   for a in [-.65,.65]:
    v=np.zeros(10);v[i]=a;v[j]=-a;out.append(v)
 # Differential tendon pulls across opposite sides of the palm.
 patterns=[
  [-.6,.6,.6,-.6,0,0,0,0,0,0],
  [.6,-.6,-.6,.6,0,0,0,0,0,0],
  [-.5,.5,0,0,0,0,0,0,0,0],
  [0,0,-.5,.5,0,0,0,0,0,0],
  [0,-.6,.6,0,0,0,0,0,0,0],
  [0,.6,-.6,0,0,0,0,0,0,0],
  [0,0,0,0,0,-.6,.6,.6,-.6,0],
  [0,0,0,0,0,.6,-.6,-.6,.6,0],
  [0,0,0,0,0,-.5,.5,0,0,0],
  [0,0,0,0,0,0,0,-.5,.5,0],
 ]
 out.extend(np.asarray(patterns,dtype=np.float32))
 while len(out)<n:
  v=np.zeros(10);k=int(rng.integers(1,6));ids=rng.choice(10,k,replace=False)
  v[ids]=rng.choice(levels,k);out.append(v)
 return np.asarray(out[:n],dtype=np.float32)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scale',type=float,default=1.);ap.add_argument('--angle',type=float,default=10.);ap.add_argument('--sign',type=float,default=-1.);a=ap.parse_args()
 nc=1024;rep=8;n=nc*rep
 cand=candidates(nc);acts=torch.tensor(np.repeat(cand,rep,axis=0),device='cuda')
 e=RodManipHand(n,seed=707,difficulty=.35,variants=nc,goal_min=a.angle,goal_max=a.angle,horizon=150)
 e.mscale.mul_(a.scale)
 e.fixed_goals=torch.full((n,),a.sign*math.radians(a.angle),device='cuda')
 e.reset(torch.ones(n,dtype=torch.bool,device='cuda'),initial=True)
 t=time.perf_counter()
 for k in range(150):
  _,_,_,info=e.step(acts,autoreset=False)
  if k in [49,50,75,100,149]:
   print('step',k+1,'grasped',int(info['grasped'].sum()),'retained',int(info['retained'].sum()),'besterr',float(info['bestError'].mean()*180/math.pi),'err',float(info['error'].mean()*180/math.pi),'drift',float(info['drift'].mean()*1000),'contacts',float(info['holdContacts'].float().mean()),flush=True)
 torch.cuda.synchronize()
 result={k:v.detach().cpu().numpy() for k,v in info.items() if isinstance(v,torch.Tensor)}
 rows=[]
 for i,c in enumerate(cand):
  sl=slice(i*rep,(i+1)*rep)
  rows.append(dict(candidate=i,action=c.tolist(),success=float(result['success'][sl].mean()),retained=float(result['retained'][sl].mean()),grasped=float(result['grasped'][sl].mean()),meanBestError=float(result['bestError'][sl].mean()*180/math.pi),meanFinalError=float(result['error'][sl].mean()*180/math.pi),meanDrift=float(result['drift'][sl].mean()*1000),meanContacts=float(result['holdContacts'][sl].mean())))
 rows.sort(key=lambda r:(r['success'],r['retained'],-r['meanBestError']),reverse=True)
 Path(f'runs/action-search-{a.sign:+g}-{a.angle:g}deg-scale{a.scale:g}.json').write_text(json.dumps(dict(worlds=n,candidates=nc,replicates=rep,seconds=time.perf_counter()-t,rows=rows),indent=2))
 print('BEST',json.dumps(rows[:20],indent=2),flush=True)
 print('SEARCH_DONE',time.perf_counter()-t,flush=True)
if __name__=='__main__':main()
