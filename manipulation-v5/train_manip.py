"""GPU goal-conditioned PPO, frozen V4 grasp prior, bounded 10-axis residual."""
import argparse,time,json,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from manipulate import ManipHand,MAN_OBS,MAN_ACT,START_MAN
from train import save,writejson
from cpu_eval import evaluate
torch.set_num_threads(1)
class ManipPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(MAN_OBS,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,MAN_ACT))
        self.critic=nn.Sequential(nn.Linear(MAN_OBS,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
        for net in (self.actor,self.critic):
            for l in net:
                if isinstance(l,nn.Linear):nn.init.orthogonal_(l.weight,2**.5);nn.init.zeros_(l.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01);nn.init.orthogonal_(self.critic[-1].weight,1.)
        self.logstd=nn.Parameter(torch.full((MAN_ACT,),-1.15))
    def dist(self,x):return Normal(self.actor(x),self.logstd.clamp(-2.7,-.3).exp())
    def value(self,x):return self.critic(x).squeeze(-1)
    @torch.no_grad()
    def act(self,x):
        d=self.dist(x);r=d.sample()
        return r,r.tanh(),d.log_prob(r).sum(-1),self.value(x)
def cpu(policy):
    p=ManipPolicy();p.load_state_dict({k:v.detach().cpu() for k,v in policy.state_dict().items()});p.eval();return p

def train(a):
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.set_float32_matmul_precision('high')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'history.json').exists(),'new run required'
    env=ManipHand(a.worlds,seed=a.seed,variants=a.variants,shape=a.shape,difficulty=1.,goal_min=15.,goal_max=30.)
    p=ManipPolicy().cuda()
    if a.warm:
        ck=torch.load(a.warm,map_location='cpu',weights_only=True);p.load_state_dict(ck['policy'])
    opt=torch.optim.Adam(p.parameters(),lr=a.lr,eps=1e-5)
    val_seeds=list(range(520000,520000+a.val_n))
    zero,_,_=evaluate(None,a.shape,val_seeds)
    before,_,_=evaluate(cpu(p),a.shape,val_seeds)
    key=lambda r:(r['success'],r['retained'],-r['meanCappedError'])
    best=key(before);history=[];total=0
    save(out/'initial.pt',dict(policy=cpu(p).state_dict(),shape=a.shape,validation=before))
    save(out/'best.pt',dict(policy=cpu(p).state_dict(),shape=a.shape,update=0,validation=before))
    writejson(out/'config.json',dict(**vars(a),obs=MAN_OBS,act=MAN_ACT,horizon=env.horizon,baseline=zero,initial=before,goal='world z, full quaternion error, stable 0.4s within 12 degrees, 25-60 degrees validation',physics='mujoco-warp 3.8.1'))
    print('REFERENCE_BASELINE',a.shape,json.dumps(zero),flush=True)
    print('REFERENCE_INITIAL',a.shape,json.dumps(before),flush=True)
    obs=env.obs();T=a.rollout;N=a.worlds;dev='cuda'
    ob=torch.empty((T,N,MAN_OBS),device=dev);ac=torch.empty((T,N,MAN_ACT),device=dev);lp=torch.empty((T,N),device=dev);v=torch.empty_like(lp);r=torch.empty_like(lp);done=torch.empty_like(lp);mask=torch.empty_like(lp);adv=torch.empty_like(lp)
    start=time.perf_counter()
    for update in range(1,a.updates+1):
        # Ramp to the final task distribution; initial targets are easier but nonzero.
        c=min(1.,update/60.)
        env.goal_min=15.+10.*c;env.goal_max=30.+30.*c
        t0=time.perf_counter()
        counts=torch.zeros(6,device=dev)
        errs=torch.zeros((),device=dev)
        for k in range(T):
            ob[k].copy_(obs);raw,act,logp,val=p.act(obs);ac[k].copy_(raw);lp[k].copy_(logp);v[k].copy_(val)
            mask[k].copy_((env.k>=START_MAN).float())
            obs,reward,d,i=env.step(act)
            r[k].copy_(reward);done[k].copy_(d)
            counts+=torch.stack([d.sum(),i['success'].sum(),i['retained'].sum(),(i['grasped']&d).sum(),i['bad'].sum(),(i['bestError']<math.radians(12.)).logical_and(d).sum()])
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
                ent=dist.entropy().sum(-1).mean();loss=pg+.5*vl-.002*ent
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(p.parameters(),.5);opt.step()
                losses.append(loss.detach());kls.append(((old[ids]-l)*sel).sum().detach()/w)
            if float(torch.stack(kls).mean())>.03:break
        torch.cuda.synchronize()
        total+=T*N
        ep,s,hold,g,b,near=map(int,counts.cpu().tolist())
        cap=env.capacities()
        row=dict(update=update,steps=total,physicsSteps=40*total,seconds=time.perf_counter()-start,stepsPerSecond=T*N/(time.perf_counter()-t0),worlds=N,episodes=ep,trainSuccess=s/ep if ep else None,trainRetained=hold/ep if ep else None,trainGrasped=g/ep if ep else None,everNear=near/ep if ep else None,retainedMeanError=float(errs)*180/math.pi/hold if hold else None,activeFraction=float(active.mean()),goalMin=env.goal_min,goalMax=env.goal_max,bad=b,capacity=cap,loss=float(torch.stack(losses).mean()),kl=float(torch.stack(kls).mean()),freeGiB=torch.cuda.mem_get_info()[0]/2**30)
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
 p=argparse.ArgumentParser();p.add_argument('--worlds',type=int,default=8192);p.add_argument('--variants',type=int,default=1024);p.add_argument('--shape',choices=['sphere','ellipsoid'],default='sphere');p.add_argument('--updates',type=int,default=180);p.add_argument('--rollout',type=int,default=32);p.add_argument('--minibatch',type=int,default=8192);p.add_argument('--epochs',type=int,default=3);p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--seed',type=int,default=2026112505);p.add_argument('--eval-every',type=int,default=10);p.add_argument('--val-n',type=int,default=48);p.add_argument('--out',default='runs/sphere01');p.add_argument('--warm');train(p.parse_args())
