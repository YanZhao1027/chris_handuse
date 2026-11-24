"""GPU parallel constant residual search to test whether hand/object can be rotated."""
import math,json,time
from pathlib import Path
import torch,numpy as np
from manipulate import ManipHand,MAN_ACT
def candidates(n,seed=404):
    rng=np.random.default_rng(seed);out=[np.zeros(10)]
    # Axis sweeps at several amplitudes.
    for i in range(10):
        for a in [-1.,-.75,-.5,-.25,.25,.5,.75,1.]:
            v=np.zeros(10);v[i]=a;out.append(v)
    # Paired differential cable/yaw motions.
    for i in range(10):
        for j in range(i+1,10):
            for a in [-.7,.7]:
                v=np.zeros(10);v[i]=a;v[j]=-a;out.append(v)
    while len(out)<n:
        v=np.zeros(10)
        k=int(rng.integers(1,5));ids=rng.choice(10,k,replace=False)
        v[ids]=rng.choice([-1.,-.75,-.5,.5,.75,1.],k)
        out.append(v)
    return np.asarray(out[:n],dtype=np.float32)
def main():
    nc=512;rep=8;n=nc*rep
    cand=candidates(nc);acts=torch.tensor(np.repeat(cand,rep,axis=0),device='cuda')
    e=ManipHand(n,seed=404,variants=nc,shape='sphere',goal_min=25,goal_max=25,horizon=150)
    signs=torch.where(torch.arange(n,device='cuda')%2==0,1.,-1.)
    e.fixed_goals=signs*math.radians(25)
    e.reset(torch.ones(n,dtype=torch.bool,device='cuda'),initial=True)
    alive=torch.ones(n,dtype=torch.bool,device='cuda')
    t=time.perf_counter()
    for k in range(150):
        _,_,done,info=e.step(acts,autoreset=False)
        if k%25==0:
            print('step',k+1,'alive',int(alive.sum()),'retained',int(info['retained'].sum()),'besterr',float(info['bestError'].mean()*180/math.pi),'bad',int(info['bad'].sum()),flush=True)
    torch.cuda.synchronize()
    result={k:v.detach().cpu().numpy() for k,v in info.items() if isinstance(v,torch.Tensor)}
    rows=[]
    for i,c in enumerate(cand):
        sl=slice(i*rep,(i+1)*rep)
        row=dict(candidate=i,action=c.tolist(),success=float(result['success'][sl].mean()),retained=float(result['retained'][sl].mean()),grasped=float(result['grasped'][sl].mean()),meanBestError=float(result['bestError'][sl].mean()*180/math.pi),meanFinalError=float(result['error'][sl].mean()*180/math.pi),meanDrift=float(result['drift'][sl].mean()*1000),meanContacts=float(result['holdContacts'][sl].mean()))
        rows.append(row)
    rows.sort(key=lambda r:(r['success'],r['retained'],-r['meanBestError']),reverse=True)
    p=Path('runs/action-search.json');p.write_text(json.dumps(dict(worlds=n,candidates=nc,replicates=rep,seconds=time.perf_counter()-t,rows=rows),indent=2))
    print('BEST',json.dumps(rows[:15],indent=2),flush=True)
    print('SEARCH_DONE',time.perf_counter()-t,flush=True)
if __name__=='__main__':main()
