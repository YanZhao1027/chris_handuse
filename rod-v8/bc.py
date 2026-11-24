"""Behavior clone successful physical trials on disjoint calibration variants."""
import math,json
from pathlib import Path
import numpy as np,torch
from manipulate_rod import RodManipHand,MAN_ACT,MAN_OBS
from policy_v8 import ManipPolicy
from train import save,writejson
torch.set_num_threads(1)
def main():
    torch.manual_seed(808)
    teachers=json.loads(Path('runs/teachers.json').read_text())
    rep=256
    candidates=[(g,np.array(r['action'],np.float32)) for g in [-15,15] for r in teachers[str(g)]]
    n=rep*len(candidates)
    e=RodManipHand(n,seed=810000,difficulty=.35,variants=rep,goal_min=15,goal_max=15)
    a=torch.tensor(np.repeat(np.stack([c[1] for c in candidates]),rep,axis=0),device='cuda')
    e.fixed_goals=torch.tensor(np.repeat([math.radians(c[0]) for c in candidates],rep),device='cuda',dtype=torch.float32)
    e.reset(torch.ones(n,dtype=torch.bool,device='cuda'),initial=True)
    obs=[];o=e.obs()
    for k in range(150):
        if k>=50:obs.append(o.clone())
        o,_,_,info=e.step(a,autoreset=False)
    cap=e.capacities()
    assert not bool(info['bad'].any()) and cap['finite'] and cap['maxConstraints']<cap['constraintCapacity'] and cap['contacts']<cap['contactCapacity']
    good=info['success']
    report={'worlds':n,'success':int(good.sum()),'byGoal':{},'teachers':[]}
    # choose priors using only search-calibration ranking, not blind evaluation
    prior=[teachers[str(g)][0]['action'] for g in [-15,15]]
    p=ManipPolicy(prior).cuda();opt=torch.optim.Adam(p.actor.parameters(),lr=3e-4)
    X=[];Y=[]
    stack=torch.stack(obs)
    for j,(g,cmd) in enumerate(candidates):
        sl=slice(j*rep,(j+1)*rep)
        report['teachers'].append({'goal':g,'action':cmd.tolist(),'successfulTrials':int(good[sl].sum()),'trials':rep})
    for g in [-15,15]:
        idx=torch.where(good&(e.fixed_goals.sign()==np.sign(g)))[0]
        report['byGoal'][str(g)]=int(len(idx))
        if len(idx):
            X.append(stack[:,idx].reshape(-1,MAN_OBS));Y.append(a[idx].repeat(len(obs),1))
    if X:
        # Equal sampling per available direction rather than weighting by easy direction.
        for step in range(360):
            xs=[];ys=[]
            for x,y in zip(X,Y):
                ids=torch.randint(len(x),(2048,),device='cuda')
                xs.append(x[ids]);ys.append(y[ids])
            x=torch.cat(xs);y=torch.cat(ys)
            loss=(p.command(x)-y).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(p.actor.parameters(),1.);opt.step()
            if step%60==0:print('BC_STEP',step,'mse',float(loss.detach()),flush=True)
        report['finalLoss']=float(loss.detach())
    else:report['warning']='No successful demonstrations; only searched-action prior initialized.'
    report['prior']=prior;report['samples']=sum(len(x) for x in X)
    writejson(Path('runs/bc.json'),report)
    save(Path('runs/bc.pt'),{'policy':{k:v.detach().cpu() for k,v in p.state_dict().items()},'report':report})
    print('BC_COMPLETE',json.dumps(report),flush=True)
if __name__=='__main__':main()
