"""PPO with residual actions, GAE, curriculum, deterministic held-out validation."""
import argparse,json,time,os,random,multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from env import Hand,OBS,ACT,HORIZON,BASE,SCALE,DT

torch.set_num_threads(1)
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(OBS,96),nn.Tanh(),nn.Linear(96,96),nn.Tanh(),nn.Linear(96,ACT))
        self.critic=nn.Sequential(nn.Linear(OBS,96),nn.Tanh(),nn.Linear(96,96),nn.Tanh(),nn.Linear(96,1))
        for block in [self.actor,self.critic]:
            for layer in block:
                if isinstance(layer,nn.Linear):nn.init.orthogonal_(layer.weight,2**.5);nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01)
        nn.init.orthogonal_(self.critic[-1].weight,1.)
        self.logstd=nn.Parameter(torch.full((ACT,),-.95))
    def dist(self,x):return Normal(self.actor(x),self.logstd.clamp(-2.7,.0).exp())
    def value(self,x):return self.critic(x).squeeze(-1)
    @torch.no_grad()
    def act(self,x,deterministic=False):
        d=self.dist(x);raw=d.mean if deterministic else d.sample()
        return raw,raw.tanh(),d.log_prob(raw).sum(-1),self.value(x)

def worker(pipe,n,worker_id,seed):
    envs=[Hand() for _ in range(n)];rng=np.random.default_rng(seed+worker_id*104729)
    def reset(h,difficulty):return h.reset(int(rng.integers(0,2**31-1)),difficulty)
    try:
        while True:
            op,inputs=pipe.recv()
            if op=='reset':pipe.send(np.stack([reset(e,inputs) for e in envs]))
            elif op=='step':
                acts,difficulty=inputs;obs=[];rewards=[];dones=[];infos=[]
                for e,a in zip(envs,acts):
                    o,r,d,i=e.step(a);obs.append(reset(e,difficulty) if d else o);rewards.append(r);dones.append(d)
                    if d:infos.append(i)
                pipe.send((np.stack(obs),np.array(rewards,dtype=np.float32),np.array(dones,dtype=np.float32),infos))
            elif op=='close':break
    finally:pipe.close()

class Vec:
    def __init__(self,n,workers,seed):
        assert n%workers==0
        self.n=n;self.per=n//workers;self.pipes=[];self.procs=[]
        ctx=mp.get_context('fork')
        for i in range(workers):
            a,b=ctx.Pipe();p=ctx.Process(target=worker,args=(b,self.per,i,seed),daemon=True);p.start();b.close();self.pipes.append(a);self.procs.append(p)
    def reset(self,difficulty):
        for p in self.pipes:p.send(('reset',difficulty))
        return np.concatenate([p.recv() for p in self.pipes])
    def step(self,actions,difficulty):
        for i,p in enumerate(self.pipes):p.send(('step',(actions[i*self.per:(i+1)*self.per],difficulty)))
        r=[p.recv() for p in self.pipes]
        return np.concatenate([x[0] for x in r]),np.concatenate([x[1] for x in r]),np.concatenate([x[2] for x in r]),[i for x in r for i in x[3]]
    def close(self):
        for p in self.pipes:
            try:p.send(('close',None))
            except (BrokenPipeError,EOFError):pass
        for p in self.procs:p.join(3)
        for p in self.pipes:p.close()

@torch.no_grad()
def evaluate(policy,seeds,difficulty=1.,horizon=HORIZON,record=False):
    # Batch policy inference; independent simulations.
    hs=[Hand() for _ in seeds];obs=np.stack([h.reset(s,difficulty,horizon=horizon) for h,s in zip(hs,seeds)])
    done=np.zeros(len(hs),dtype=bool);result=[None]*len(hs);frames=[[] for h in hs]
    for _ in range(horizon):
        if policy is None:acts=np.zeros((len(hs),ACT))
        else:acts=policy.act(torch.as_tensor(obs),True)[1].numpy()
        for i,h in enumerate(hs):
            if done[i]:continue
            o,r,d,info=h.step(acts[i],record=record);obs[i]=o
            if record:frames[i].append(info.pop('frame'))
            if d:done[i]=True;result[i]=dict(info,seed=int(seeds[i]),params=h.params)
        if done.all():break
    scores=dict(success=float(np.mean([r['success'] for r in result])),meanReturn=float(np.mean([r['return_'] for r in result])),n=len(result))
    return scores,result,frames

def save(path,state):
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,path)
def writejson(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,path)
def train(args):
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    policy=Policy();opt=torch.optim.Adam(policy.parameters(),lr=3e-4,eps=1e-5)
    val_seeds=list(range(100000,100000+args.val_n))
    baseline,_,_=evaluate(None,val_seeds)
    print('VALIDATION_BASELINE',json.dumps(baseline),flush=True)
    history=[];start=0;best=(-1.,-1e9);total=0
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=False);policy.load_state_dict(ck['policy']);opt.load_state_dict(ck['optimizer']);start=ck['update'];total=ck['steps'];history=ck['history'];best=tuple(ck['best'])
    else:
        # Zero residual policy, before learning. Preserve for exact comparison.
        save(out/'initial.pt',{'policy':policy.state_dict(),'obs':OBS,'act':ACT})
    vec=Vec(args.envs,args.workers,args.seed+start*101);obs=vec.reset(.20 if start<5 else .65)
    ts=time.perf_counter()
    try:
        for update in range(start+1,args.updates+1):
            difficulty=min(1.,.20+update*.035)
            if update<=5:difficulty=.20
            ob=[];raw=[];old=[];values=[];rew=[];done=[];episodes=[]
            for k in range(args.rollout):
                x=torch.as_tensor(obs);a,b,lp,v=policy.act(x)
                nxt,r,d,info=vec.step(b.numpy(),difficulty)
                ob.append(x);raw.append(a);old.append(lp);values.append(v);rew.append(torch.from_numpy(r));done.append(torch.from_numpy(d));episodes+=info;obs=nxt
            with torch.no_grad():nextv=policy.value(torch.as_tensor(obs));v=torch.stack(values);r=torch.stack(rew);d=torch.stack(done)
            adv=torch.zeros_like(r);last=torch.zeros(args.envs)
            for k in reversed(range(args.rollout)):
                delta=r[k]+.99*(1-d[k])*(nextv if k==args.rollout-1 else v[k+1])-v[k]
                last=delta+.99*.95*(1-d[k])*last;adv[k]=last
            ret=(adv+v).flatten();adv=adv.flatten();adv=(adv-adv.mean())/(adv.std()+1e-8)
            x=torch.cat(ob);actions=torch.cat(raw);oldlp=torch.cat(old);oldv=v.flatten()
            n=len(x);losses=[];kls=[]
            for epoch in range(4):
                for ids in torch.randperm(n).split(512):
                    dist=policy.dist(x[ids]);lp=dist.log_prob(actions[ids]).sum(-1);ratio=(lp-oldlp[ids]).exp()
                    pg=-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()
                    vp=policy.value(x[ids]);vc=oldv[ids]+(vp-oldv[ids]).clamp(-.2,.2)
                    vl=.5*torch.maximum((vp-ret[ids])**2,(vc-ret[ids])**2).mean()
                    ent=dist.entropy().sum(-1).mean();loss=pg+.5*vl-.003*ent
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),.5);opt.step()
                    losses.append(float(loss.detach()));kls.append(float((oldlp[ids]-lp).mean().detach()))
            total+=n
            record={'update':update,'steps':total,'difficulty':difficulty,'episodes':len(episodes),'trainSuccess':float(np.mean([i['success'] for i in episodes])) if episodes else None,'trainReturn':float(np.mean([i['return_'] for i in episodes])) if episodes else None,'loss':float(np.mean(losses)),'approxKL':float(np.mean(kls)),'seconds':time.perf_counter()-ts}
            if update%args.eval_every==0 or update==args.updates or update==1:
                metrics,_,_=evaluate(policy,val_seeds);record['validation']=metrics
                key=(metrics['success'],metrics['meanReturn'])
                if key>best:
                    best=key;save(out/'best.pt',{'policy':policy.state_dict(),'obs':OBS,'act':ACT,'update':update,'validation':metrics})
            history.append(record)
            print(json.dumps(record,separators=(',',':')),flush=True)
            writejson(out/'history.json',history)
            if update%args.eval_every==0 or update==args.updates or update==1:
                save(out/'latest.pt',{'policy':policy.state_dict(),'optimizer':opt.state_dict(),'update':update,'steps':total,'history':history,'best':best,'obs':OBS,'act':ACT})
    finally:vec.close()
    print('TRAINING_COMPLETE',json.dumps({'steps':total,'best':best}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--updates',type=int,default=100);p.add_argument('--envs',type=int,default=48);p.add_argument('--workers',type=int,default=6);p.add_argument('--rollout',type=int,default=64);p.add_argument('--seed',type=int,default=20261124);p.add_argument('--eval-every',type=int,default=5);p.add_argument('--val-n',type=int,default=24);p.add_argument('--out',default='runs/ppo01');p.add_argument('--resume');train(p.parse_args())
