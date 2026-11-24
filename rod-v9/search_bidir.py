"""Fair common-random-number search, full quaternion goals, fixed commands."""
import json,math,time
from pathlib import Path
import numpy as np,torch
from manipulate_rod import RodManipHand,MAN_ACT,quat_mul,qerror
from action_search_rod import candidates
from train import writejson
def run_search(cand,rep,seed,difficulty=.35):
    n=len(cand)*rep
    e=RodManipHand(n,seed=seed,difficulty=difficulty,variants=rep,goal_min=15,goal_max=15)
    action=torch.tensor(np.repeat(cand,rep,axis=0),device='cuda',dtype=torch.float32)
    goals=[-30,-15,15,30];targets=None
    streak=torch.zeros((len(goals),n),device='cuda',dtype=torch.int32)
    errors=torch.zeros((len(goals),n),device='cuda')
    t=time.perf_counter()
    for k in range(150):
        if k==50:
            q=e.quat().clone()
            targets=[]
            for g in goals:
                dq=torch.zeros_like(q);dq[:,0]=math.cos(math.radians(g)/2);dq[:,3]=math.sin(math.radians(g)/2)
                targets.append(quat_mul(dq,q))
        _,_,_,i=e.step(action,autoreset=False)
        if targets is not None:
            for j,target in enumerate(targets):
                _,err=qerror(target,e.quat());errors[j]=err
                near=(err<math.radians(8))&(i['omega']<1.2)&(i['height']>.25-e.center[:,2]+.035)&~i['fixture']
                streak[j]=torch.where(near,streak[j]+1,0)
        if k in [49,99,149]:print('SEARCH_STEP',seed,k+1,'grasped',int(i['grasped'].sum()),'retained',int(i['retained'].sum()),'capacity',e.capacities(),flush=True)
    cap=e.capacities()
    assert not bool(i['bad'].any()) and cap['finite'] and cap['maxConstraints']<cap['constraintCapacity'] and cap['contacts']<cap['contactCapacity']
    retain=i['retained'].reshape(len(cand),rep)
    rows=[]
    for j,g in enumerate(goals):
        suc=((streak[j]>=10)&i['retained']&(i["maxSelfPen"]<.001)&(i["maxObjectPen"]<.002)).reshape(len(cand),rep)
        er=torch.where(i['retained'],errors[j],math.pi).reshape(len(cand),rep)
        table=torch.stack([suc.float().mean(1),retain.float().mean(1),er.mean(1)*180/math.pi],1).cpu().numpy()
        for idx,values in enumerate(table):
            rows.append(dict(goal=g,id=idx,action=cand[idx].tolist(),success=float(values[0]),retained=float(values[1]),cappedError=float(values[2])))
    print('SEARCH_COMPLETE',seed,n,'seconds',time.perf_counter()-t,flush=True)
    return rows
def top(rows,g,n):
    return sorted((r for r in rows if r['goal']==g),key=lambda r:(r['success'],r['retained'],-r['cappedError']),reverse=True)[:n]
if __name__=='__main__':
    # Both signs use the same candidate set and same 16 physical replicas.
    cand=candidates(512,seed=808)
    prev0=json.loads(Path('../rod-v8/runs/teachers.json').read_text());prev=[r for g in ['-15','15'] for r in prev0[g]]
    cand[-16:]=np.array([r['action'] for r in prev[:8]]+[[ -x for x in r['action']] for r in prev[:8]])
    rows=run_search(cand,8,914000)
    writejson(Path('runs/search.json'),rows)
    chosen=[];ids=set()
    for g in [-30,-15,15,30]:
        for r in top(rows,g,8):
            if r['id'] not in ids:ids.add(r['id']);chosen.append(r['action'])
    # Independent calibration seeds, 64 replicates per finalist.
    fine=run_search(np.asarray(chosen,np.float32),64,915000)
    writejson(Path('runs/calibration.json'),fine)
    teachers={str(g):top(fine,g,4) for g in [-30,-15,15,30]}
    writejson(Path('runs/teachers.json'),teachers)
    for g in teachers:print('TEACHERS',g,json.dumps(teachers[g]),flush=True)
