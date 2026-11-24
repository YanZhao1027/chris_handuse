"""Two-phase regrasp candidate screening. Common physics variants, independent calibration."""
import json,math,time
from pathlib import Path
import numpy as np,torch
from manipulate_rod import RodManipHand,quat_mul,qerror
from action_search_rod import candidates
from train import writejson
GOALS=[-30,-15,15,30]
def run(rows,rep,seed):
    n=len(rows)*rep
    e=RodManipHand(n,seed=seed,difficulty=.35,variants=rep,goal_min=15.,goal_max=15.)
    first=torch.tensor(np.repeat(np.array([x['action'] for x in rows],np.float32),rep,axis=0),device='cuda')
    second=torch.tensor(np.repeat(np.array([x['action2'] for x in rows],np.float32),rep,axis=0),device='cuda')
    switch=torch.tensor(np.repeat([x['switch'] for x in rows],rep),device='cuda')
    streak=torch.zeros((4,n),device='cuda',dtype=torch.int32)
    errors=torch.zeros((4,n),device='cuda');targets=None;t=time.perf_counter()
    for k in range(150):
        if k==50:
            q=e.quat().clone();targets=[]
            for g in GOALS:
                dq=torch.zeros_like(q);dq[:,0]=math.cos(math.radians(g)/2);dq[:,3]=math.sin(math.radians(g)/2)
                targets.append(quat_mul(dq,q))
        act=torch.where((k<switch)[:,None],first,second)
        _,_,_,i=e.step(act,autoreset=False)
        if targets is not None:
            for j,target in enumerate(targets):
                _,err=qerror(target,e.quat());errors[j]=err
                near=(err<math.radians(8))&(i['omega']<1.2)&(i['height']>.25-e.center[:,2]+.035)&~i['fixture']
                streak[j]=torch.where(near,streak[j]+1,0)
        if k in [49,99,149]:print('SEQUENCE',seed,k+1,'retained',int(i['retained'].sum()),'capacity',e.capacities(),flush=True)
    cap=e.capacities()
    assert not bool(i['bad'].any()) and cap['finite'] and cap['maxConstraints']<cap['constraintCapacity'] and cap['contacts']<cap['contactCapacity']
    safe=i['retained']&(i['maxSelfPen']<.001)&(i['maxObjectPen']<.002)
    out=[]
    for j,g in enumerate(GOALS):
        suc=((streak[j]>=10)&safe).reshape(len(rows),rep)
        er=torch.where(safe,errors[j],math.pi).reshape(len(rows),rep)
        table=torch.stack([suc.float().mean(1),safe.reshape(len(rows),rep).float().mean(1),er.mean(1)*180/math.pi],1).cpu().numpy()
        for idx,v in enumerate(table):
            out.append(dict(**rows[idx],goal=g,id=idx,success=float(v[0]),retained=float(v[1]),cappedError=float(v[2])))
    print('SEQUENCE_COMPLETE',seed,n,time.perf_counter()-t,flush=True)
    return out
def top(rows,g,n):
    return sorted((r for r in rows if r['goal']==g),key=lambda r:(r['success'],r['retained'],-r['cappedError']),reverse=True)[:n]
def main():
    rng=np.random.default_rng(909)
    teachers=json.loads(Path('runs/teachers.json').read_text());cal=json.loads(Path('runs/calibration.json').read_text())
    # Include safe fixed commands, even for the harder direction, and controlled release pulses.
    fixed=[]
    for g in GOALS:
        fixed.extend([r['action'] for r in teachers[str(g)]])
        fixed.extend([r['action'] for r in sorted([r for r in cal if r['goal']==g and r['retained']>=.95],key=lambda x:x['cappedError'])[:6]])
    base=np.unique(np.asarray(fixed,np.float32),axis=0);rows=[]
    def add(a,b,k):rows.append(dict(action=np.asarray(a).tolist(),action2=np.asarray(b).tolist(),switch=int(k)))
    for x in base:add(x,x,75)
    zero=np.zeros(10)
    for b in base:
        for j in range(5):
            a=b.copy();a[j]=-1.
            add(a,b,70)
    while len(rows)<512:
        b=base[rng.integers(len(base))].copy()
        a=b.copy()
        if rng.random()<.45:
            a=base[rng.integers(len(base))].copy()
        else:
            ids=rng.choice(10,int(rng.integers(1,4)),replace=False)
            a[ids]=rng.choice([-1.,-.75,-.25,.25,.75,1.],len(ids))
        if rng.random()<.3:
            ids=rng.choice(10,int(rng.integers(1,3)),replace=False);b[ids]=rng.choice([-.75,-.25,.25,.75],len(ids))
        add(a,b,rng.choice([62,70,80,95,110]))
    raw=run(rows[:512],8,919000)
    writejson(Path('runs/sequence-search.json'),raw)
    ids=set();chosen=[]
    for g in GOALS:
        ranks=top(raw,g,8)
        ranks+=sorted([r for r in raw if r['goal']==g and r['retained']>=.875],key=lambda x:x['cappedError'])[:4]
        for r in ranks:
            if r['id'] not in ids:
                ids.add(r['id']);chosen.append({k:r[k] for k in ['action','action2','switch']})
    fine=run(chosen,64,920000)
    writejson(Path('runs/sequence-calibration.json'),fine)
    t={str(g):top(fine,g,4) for g in GOALS}
    writejson(Path('runs/sequence-teachers.json'),t)
    for g,r in t.items():print('SEQUENCE_TEACHER',g,json.dumps(r[0]),flush=True)
if __name__=='__main__':main()
