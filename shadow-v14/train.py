"""Large bounded GPU PPO. Select checkpoints using independent CPU validation, not rollout score."""
import argparse,json,math,time
from pathlib import Path
import numpy as np,torch
from gpu_env import Env
from common import *
from policy import Policy,cpu,load
from reference import cases,evaluate
from search_grasp import write
torch.set_num_threads(1)
def save(path,x):
    p=Path(path);tmp=p.with_suffix('.tmp');torch.save(x,tmp);tmp.replace(p)
def train(a):
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.set_float32_matmul_precision('high')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);assert not (out/'history.json').exists()
    assert a.val_n>=48 and a.val_n%12==0
    e=Env(a.worlds,a.seed,a.variants);p=Policy().cuda()
    if a.warm:p.load_state_dict(torch.load(a.warm,map_location='cpu',weights_only=True)['policy'])
    # Retain learned actor, but reduce random action noise; the new curriculum
    # and larger motion range need a fresh optimizer.
    with torch.no_grad():p.logstd.clamp_(-2.7,-2.1)
    opt=torch.optim.Adam(p.parameters(),lr=a.lr,eps=1e-5)
    # Freeze fresh validation starts; blind cases use separate seeds after training.
    specs=((-15,0),(15,0),(-25,0),(25,0),(-30,0),(30,0),(-45,0),(45,0),(-15,1),(15,1),(-25,1),(25,1))
    val=cases(141000,a.val_n,specs);base,br=evaluate(None,val)
    before,_=evaluate(cpu(p),val)
    def key(r):
        g=r['byPlaneAngle'];anchor=min(g['0:-15']['success'],g['0:15']['success'])
        frontier=np.mean([g[f'0:{i}']['success'] for i in [-25,25,-30,30,-45,45]])
        bal=min(np.mean([g[f'0:{i}']['success'] for i in [-25,-30,-45]]),np.mean([g[f'0:{i}']['success'] for i in [25,30,45]]))
        return (anchor>=.875,bal,frontier,anchor,r['retained'],-r['meanError'])
    def broad(r):
        g=r['byPlaneAngle'];anchor=min(g['0:-15']['success'],g['0:15']['success'])
        return (anchor>=.875,min(r['negative'],r['positive']),r['success'],r['retained'],-r['meanError'])
    best=key(before);wide=broad(before)
    save(out/'best.pt',dict(policy=cpu(p).state_dict(),update=0,validation=before))
    save(out/'best-wide.pt',dict(policy=cpu(p).state_dict(),update=0,validation=before))
    write(out/'config.json',dict(**vars(a),baseline=base,initial=before,physicsStepsPerControl=SUB,dt=.002,horizon=HORIZON,obs=OBS,act=ACT,goal='directed long axis in palm frame; rotations about palm Y and Z; fixed wrist targets; extended full legal actuator range; final 5 deg/20 steps, opposition >=75%, retention and depth limits'))
    print('BASELINE',base,'INITIAL',before,flush=True)
    N=a.worlds;T=a.rollout;dev='cuda';o=e.obs();assert o.shape==(N,OBS)
    ob=torch.empty((T,N,OBS),device=dev);z=torch.empty((T,N,ACT),device=dev);lp=torch.empty((T,N),device=dev);v=torch.empty_like(lp);r=torch.empty_like(lp);done=torch.empty_like(lp);adv=torch.empty_like(lp)
    hist=[];total=0;start=time.perf_counter();rolling=[];maxgoal=20.
    for update in range(1,a.updates+1):
        # Explicit staged curriculum: do not get stuck indefinitely at +/-15.
        # 25% anchors preserve the learned motion while the frontier expands.
        maxgoal=20.+25.*min(1.,(update-1)/max(1.,a.updates*.75))
        e.goal_min=15.;e.goal_max=maxgoal
        e.multi_fraction=.25*min(1.,max(0.,(update-a.updates*.20)/max(1.,a.updates*.50)))
        t0=time.perf_counter();counts=torch.zeros(18,device=dev);sums=torch.zeros(4,device=dev)
        for k in range(T):
            ob[k].copy_(o);raw,action,l,valp=p.act(o);z[k].copy_(raw);lp[k].copy_(l);v[k].copy_(valp)
            o,rr,d,i=e.step(action);r[k].copy_(rr);done[k].copy_(d)
            neg=i['angle']<0;pos=~neg
            counts+=torch.stack([d.sum(),i['success'].sum(),i['retained'].sum(),i['bad'].sum(),(d&neg).sum(),(i['success']&neg).sum(),(d&pos).sum(),(i['success']&pos).sum(),(d&(i['selfMM']>=1.5)).sum(),(d&(i['objMM']>=3)).sum(),i['dropped'].sum(),(d&(i['error']<math.radians(5))).sum(),(d&(i['plane']==1)).sum(),(i['success']&(i['plane']==1)).sum(),(d&(i['recontacts']>0)).sum(),(i['success']&(i['recontacts']>0)).sum(),(d&(i['plane']==0)&(i['angle'].abs()>math.radians(20))).sum(),(i['success']&(i['plane']==0)&(i['angle'].abs()>math.radians(20))).sum()])
            sums+=torch.stack([torch.where(d,i['opposition'],0.).sum(),torch.where(d,i['error'],0.).sum(),i['selfMM'].max(),i['objMM'].max()])
        with torch.no_grad():
            nxt=p.value(o);last=torch.zeros(N,device=dev)
            for k in reversed(range(T)):
                delta=r[k]+.995*(1-done[k])*(nxt if k==T-1 else v[k+1])-v[k]
                last=delta+.995*.95*(1-done[k])*last;adv[k].copy_(last)
            ret=(adv+v).flatten();A=adv.flatten();A=(A-A.mean())/A.std().clamp(min=1e-6)
        x=ob.flatten(0,1);zz=z.flatten(0,1);old=lp.flatten();oldv=v.flatten()
        kl=[];losses=[]
        for ep in range(a.epochs):
            for idx in torch.randperm(len(x),device=dev).split(a.minibatch):
                dist=p.dist(x[idx]);l=dist.log_prob(zz[idx]).sum(1);ratio=(l-old[idx]).exp()
                pg=-torch.minimum(ratio*A[idx],ratio.clamp(.8,1.2)*A[idx]).mean()
                pred=p.value(x[idx]);clip=oldv[idx]+(pred-oldv[idx]).clamp(-.2,.2);vl=.5*torch.maximum((pred-ret[idx])**2,(clip-ret[idx])**2).mean()
                loss=pg+.5*vl-.0002*dist.entropy().sum(1).mean()
                assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(p.parameters(),.5);opt.step()
                kl.append((old[idx]-l).mean().detach());losses.append(loss.detach())
            if float(torch.stack(kl).mean())>.03:break
        torch.cuda.synchronize();total+=T*N
        ep,s,hold,bad,n,ns,po,ps,sp,op,drops,near,multi,ms,rc,rs,front,fs=map(int,counts.cpu().tolist());cap=e.capacity()
        row=dict(multiFraction=e.multi_fraction,multiSuccess=ms/multi if multi else None,frontierSuccess=fs/front if front else None,recontactEpisodes=rc,successfulRecontactEpisodes=rs,update=update,steps=total,physicsSteps=SUB*total,seconds=time.perf_counter()-start,stepsPerSecond=T*N/(time.perf_counter()-t0),goalMin=e.goal_min,goalMax=e.goal_max,episodes=ep,success=s/ep if ep else None,negative=ns/n if n else None,positive=ps/po if po else None,retained=hold/ep if ep else None,opposition=float(sums[0])/ep if ep else None,bad=bad,drops=drops,penetrationSelfEpisodes=sp,penetrationObjectEpisodes=op,capacity=cap,kl=float(torch.stack(kl).mean()),loss=float(torch.stack(losses).mean()))
        if bad or not cap['finite'] or cap['contacts']>=cap['contactCapacity'] or cap['constraints']>=cap['constraintCapacity']:
            write(out/'failure.json',row);raise RuntimeError('Invalid physics or capacity overflow')
        if update==1 or update%a.eval_every==0 or update==a.updates:
            cp=cpu(p);score,_=evaluate(cp,val);row['validation']=score;state=dict(policy=cp.state_dict(),update=update,steps=total,validation=score)
            if key(score)>best:best=key(score);save(out/'best.pt',state)
            if broad(score)>wide:wide=broad(score);save(out/'best-wide.pt',state)
            save(out/'latest.pt',dict(**state,optimizer=opt.state_dict(),goalMax=maxgoal))
        hist.append(row);write(out/'history.json',hist);print('PPO',json.dumps(row,separators=(',',':')),flush=True)
    print('TRAIN_COMPLETE',total,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worlds',type=int,default=8192);p.add_argument('--variants',type=int,default=1024);p.add_argument('--rollout',type=int,default=32);p.add_argument('--updates',type=int,default=900);p.add_argument('--epochs',type=int,default=3);p.add_argument('--minibatch',type=int,default=8192);p.add_argument('--lr',type=float,default=5e-5);p.add_argument('--eval-every',type=int,default=50);p.add_argument('--seed',type=int,default=2026112414);p.add_argument('--out',default='runs/ppo01');p.add_argument('--val-n',type=int,default=48);p.add_argument('--warm',default='../shadow-v13/runs/ppo01/best.pt');train(p.parse_args())
