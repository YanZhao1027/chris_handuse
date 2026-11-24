"""GPU goal-conditioned PPO, searched opposed-grasp prior, bounded 10-axis residual."""
import argparse,time,json,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from manipulate_rod import RodManipHand,MAN_OBS,MAN_ACT,START_MAN
from policy_v8 import ManipPolicy
from train import save,writejson
from cpu_eval_rod import evaluate as cpu_evaluate

def evaluate(policy,shape,seeds):
    angles=[(-1 if i%2==0 else 1)*15. for i in range(len(seeds))]
    score,rows,frames=cpu_evaluate(None if policy is None else policy.command,seeds,angles,difficulty=.35,scale=2.)
    score["negative"]=float(np.mean([r["success"] for r in rows if r["targetDegrees"]<0]))
    score["positive"]=float(np.mean([r["success"] for r in rows if r["targetDegrees"]>0]))
    score["byGoal"]={str(g):float(np.mean([r["success"] for r in rows if abs(r["targetDegrees"]-g)<1e-6])) for g in [-15,15]}
    return score,rows,frames
torch.set_num_threads(1)
def cpu(policy):
    p=ManipPolicy();p.load_state_dict({k:v.detach().cpu() for k,v in policy.state_dict().items()});p.eval();return p

def train(a):
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.set_float32_matmul_precision('high')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'history.json').exists(),'new run required'
    env=RodManipHand(a.worlds,seed=a.seed,variants=a.variants,difficulty=.35,goal_min=15.,goal_max=15.)
    p=ManipPolicy().cuda()
    if a.warm:
        ck=torch.load(a.warm,map_location='cpu',weights_only=True);p.load_state_dict(ck['policy'])
    opt=torch.optim.Adam(p.parameters(),lr=a.lr,eps=1e-5)
    val_seeds=list(range(1008000,1008000+a.val_n))
    zero,_,_=evaluate(None,a.shape,val_seeds)
    before,_,_=evaluate(cpu(p),a.shape,val_seeds)
    key=lambda r:(min(r['negative'],r['positive']),r['success'],r['meanOppositionFraction'],r['retained'],-r['meanCappedError'])
    best=key(before);history=[];total=0
    save(out/'initial.pt',dict(policy=cpu(p).state_dict(),shape=a.shape,validation=before))
    save(out/'best.pt',dict(policy=cpu(p).state_dict(),shape=a.shape,update=0,validation=before))
    writejson(out/'config.json',dict(**vars(a),obs=MAN_OBS,act=MAN_ACT,horizon=env.horizon,baseline=zero,initial=before,goal='continuous signed 12-15 deg with opposed thumb about world Z, full quaternion error <8 deg for last 0.4s, self/object penetration limits; CPU balanced validation',physics='mujoco-warp 3.8.1'))
    print('REFERENCE_BASELINE',a.shape,json.dumps(zero),flush=True)
    print('REFERENCE_INITIAL',a.shape,json.dumps(before),flush=True)
    obs=env.obs();T=a.rollout;N=a.worlds;dev='cuda'
    ob=torch.empty((T,N,MAN_OBS),device=dev);ac=torch.empty((T,N,MAN_ACT),device=dev);lp=torch.empty((T,N),device=dev);v=torch.empty_like(lp);r=torch.empty_like(lp);done=torch.empty_like(lp);mask=torch.empty_like(lp);adv=torch.empty_like(lp)
    start=time.perf_counter()
    for update in range(1,a.updates+1):
        # Ramp to the final task distribution; initial targets are easier but nonzero.
        c=min(1.,update/60.)
        env.goal_min=12.;env.goal_max=15.
        t0=time.perf_counter()
        counts=torch.zeros(12,device=dev)
        errs=torch.zeros((),device=dev)
        for k in range(T):
            ob[k].copy_(obs);raw,act,logp,val=p.act(obs);ac[k].copy_(raw);lp[k].copy_(logp);v[k].copy_(val)
            mask[k].copy_((env.k>=START_MAN).float())
            obs,reward,d,i=env.step(act)
            r[k].copy_(reward);done[k].copy_(d)
            counts+=torch.stack([d.sum(),i['success'].sum(),i['retained'].sum(),(i['grasped']&d).sum(),i['bad'].sum(),(i['bestError']<math.radians(8.)).logical_and(d).sum(),(d&(i['targetAngle']<0)).sum(),(i['success']&(i['targetAngle']<0)).sum(),(d&(i['targetAngle']>0)).sum(),(i['success']&(i['targetAngle']>0)).sum(),(d&(i['maxSelfPen']>=.001)).sum(),(d&(i['maxObjectPen']>=.002)).sum()])
            errs+=torch.where(i['retained'],i['error'],0.).sum()
        with torch.no_grad():
            nxt=p.value(obs);last=torch.zeros(N,device=dev)
            for k in reversed(range(T)):
                delta=r[k]+.99*(1-done[k])*(nxt if k==T-1 else v[k+1])-v[k]
                last=delta+.99*.95*(1-done[k])*last;adv[k].copy_(last)
            ret=(adv+v).flatten();A=adv.flatten();active=mask.flatten()
            # No actor gradient from the frozen approach, only from learned control actions.
            denom=active.sum().clamp(min=1)
            avg=(A*active).sum()/denom
            std=(((A-avg)**2*active).sum()/denom).sqrt().clamp(min=1e-6)
            A=(A-avg)/std
        x=ob.flatten(0,1);old=lp.flatten();oldv=v.flatten();actions=ac.flatten(0,1)
        losses=[];kls=[]
        for epoch in range(a.epochs):
            for ids in torch.randperm(len(x),device=dev).split(a.minibatch):
                sel=active[ids];w=sel.sum().clamp(min=1)
                dist=p.dist(x[ids]);l=dist.log_prob(actions[ids]).sum(-1);ratio=(l-old[ids]).exp()
                pg=-(torch.minimum(ratio*A[ids],ratio.clamp(.8,1.2)*A[ids])*sel).sum()/w
                pred=p.value(x[ids]);vc=oldv[ids]+(pred-oldv[ids]).clamp(-.2,.2)
                vl=.5*torch.maximum((pred-ret[ids])**2,(vc-ret[ids])**2).mean()
                ent=(dist.entropy().sum(-1)*sel).sum()/w;loss=pg+.5*vl-.001*ent
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(p.parameters(),.5);opt.step()
                losses.append(loss.detach());kls.append(((old[ids]-l)*sel).sum().detach()/w)
            if float(torch.stack(kls).mean())>.03:break
        torch.cuda.synchronize()
        total+=T*N
        ep,s,hold,g,b,near,neg,neg_s,pos,pos_s,selfbad,objbad=map(int,counts.cpu().tolist())
        cap=env.capacities()
        row=dict(selfPenLimitEpisodes=selfbad,objectPenLimitEpisodes=objbad,trainNegative=neg_s/neg if neg else None,trainPositive=pos_s/pos if pos else None,update=update,steps=total,physicsSteps=40*total,seconds=time.perf_counter()-start,stepsPerSecond=T*N/(time.perf_counter()-t0),worlds=N,episodes=ep,trainSuccess=s/ep if ep else None,trainRetained=hold/ep if ep else None,trainGrasped=g/ep if ep else None,everNear=near/ep if ep else None,retainedMeanError=float(errs)*180/math.pi/hold if hold else None,activeFraction=float(active.mean()),goalMin=env.goal_min,goalMax=env.goal_max,bad=b,capacity=cap,loss=float(torch.stack(losses).mean()),kl=float(torch.stack(kls).mean()),freeGiB=torch.cuda.mem_get_info()[0]/2**30)
        if b or not cap['finite'] or cap['contacts']>=cap['contactCapacity'] or cap['maxConstraints']>=cap['constraintCapacity']:
            writejson(out/'failure.json',row);raise RuntimeError('physics invalid')
        if update==1 or update%a.eval_every==0 or update==a.updates:
            cp=cpu(p);score,_,_=evaluate(cp,a.shape,val_seeds)
            row['validation']=score
            if key(score)>best:
                best=key(score);save(out/'best.pt',dict(policy=cp.state_dict(),shape=a.shape,update=update,validation=score))
            save(out/'latest.pt',dict(policy=cp.state_dict(),optimizer=opt.state_dict(),shape=a.shape,update=update,steps=total,validation=score))
        history.append(row);writejson(out/'history.json',history)
        print(json.dumps(row,separators=(',',':')),flush=True)
    print('MANIP_TRAINING_COMPLETE',a.shape,json.dumps(dict(steps=total,best=best)),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worlds',type=int,default=4096);p.add_argument('--variants',type=int,default=1024);p.add_argument('--shape',choices=['cylinder'],default='cylinder');p.add_argument('--updates',type=int,default=120);p.add_argument('--rollout',type=int,default=32);p.add_argument('--minibatch',type=int,default=4096);p.add_argument('--epochs',type=int,default=3);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=2026112410);p.add_argument('--eval-every',type=int,default=10);p.add_argument('--val-n',type=int,default=64);p.add_argument('--out',default='runs/opposition01');p.add_argument('--warm',default='runs/bc.pt');train(p.parse_args())
