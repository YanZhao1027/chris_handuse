"""Standard Adroit CPU physics in 12 workers, CUDA PPO initialized from expert BC.
The time limit is the finite-horizon learning terminal. No NoSlip changes.
"""
import argparse,multiprocessing as mp,time,math,traceback
from pathlib import Path
import numpy as np,torch,gymnasium as gym
from torch import nn
from torch.distributions import Normal
from benchmark import Policy,load,save,write,evaluate
def worker(conn,n,seed,horizon):
    try:
        envs=[gym.make('AdroitHandPen-v1',max_episode_steps=horizon) for _ in range(n)]
        rng=np.random.default_rng(seed)
        def reset(e):return e.reset(seed=int(rng.integers(10000000,2000000000)))[0]
        obs=np.stack([reset(e) for e in envs]).astype('float32')
        returns=np.zeros(n);counts=np.zeros(n,int);streak=np.zeros(n,int);best=np.zeros(n,int);steps=np.zeros(n,int)
        conn.send(('ok',obs))
        while True:
            cmd,a=conn.recv()
            if cmd=='close':break
            reward=np.zeros(n,np.float32);done=np.zeros(n,bool);eps=[]
            for i,e in enumerate(envs):
                o,r,term,trunc,info=e.step(a[i]);s=bool(info['success'])
                assert np.isfinite(o).all() and np.isfinite(r)
                reward[i]=r;returns[i]+=r;counts[i]+=s;streak[i]=streak[i]+1 if s else 0;best[i]=max(best[i],streak[i]);steps[i]+=1
                done[i]=term or trunc
                if done[i]:
                    eps.append([float(returns[i]),int(counts[i]>=20),int(best[i]>=20),int(s)])
                    o=reset(e);returns[i]=0;counts[i]=0;streak[i]=0;best[i]=0;steps[i]=0
                obs[i]=o
            conn.send(('ok',(obs,reward,done,eps)))
        for e in envs:e.close()
    except Exception:
        conn.send(('error',traceback.format_exc()))
    finally:conn.close()
class Pool:
    def __init__(self,workers,per,seed,horizon):
        self.ctx=mp.get_context('spawn');self.ps=[];self.conns=[];self.per=per
        for j in range(workers):
            p,c=self.ctx.Pipe();proc=self.ctx.Process(target=worker,args=(c,per,seed+1009*j,horizon),daemon=True)
            proc.start();c.close();self.ps.append(proc);self.conns.append(p)
        self.obs=np.concatenate([self.recv(c) for c in self.conns])
    @staticmethod
    def recv(c):
        if not c.poll(60):raise TimeoutError('CPU physics worker stopped responding')
        status,v=c.recv()
        if status!='ok':raise RuntimeError(v)
        return v
    def step(self,a):
        for j,c in enumerate(self.conns):c.send(('step',a[j*self.per:(j+1)*self.per]))
        rows=[self.recv(c) for c in self.conns]
        return np.concatenate([r[0] for r in rows]),np.concatenate([r[1] for r in rows]),np.concatenate([r[2] for r in rows]),[e for r in rows for e in r[3]]
    def close(self):
        for c in self.conns:
            try:c.send(('close',None))
            except (BrokenPipeError,EOFError):pass
        for p in self.ps:
            p.join(5)
            if p.is_alive():p.terminate();p.join()
class AC(nn.Module):
    def __init__(self,bc):
        super().__init__();self.actor=bc
        self.critic=nn.Sequential(nn.Linear(45,256),nn.Tanh(),nn.Linear(256,128),nn.Tanh(),nn.Linear(128,1))
        self.logstd=nn.Parameter(torch.full((24,),-2.3))
    def dist(self,x):
        loc=torch.atanh(self.actor(x).clamp(-.999,.999));return Normal(loc,self.logstd.clamp(-3.5,-1.3).exp())
    def v(self,x):return self.critic(((x-self.actor.mean)/self.actor.std).clamp(-20,20)).squeeze(-1)
def cpu(p):
    q=Policy();q.load_state_dict({k:v.detach().cpu() for k,v in p.actor.state_dict().items()});q.eval();return q
def main(a):
    torch.set_num_threads(1);torch.manual_seed(1212);torch.cuda.manual_seed_all(1212)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'history.json').exists()
    bc,bcinfo=load('runs/bc01/best.pt');prior,_=load('runs/bc01/best.pt');prior=prior.cuda().eval()
    for v in prior.parameters():v.requires_grad_(False)
    ac=AC(bc).cuda()
    opt=torch.optim.Adam([dict(params=ac.actor.parameters(),lr=1e-5),dict(params=ac.critic.parameters(),lr=3e-4),dict(params=[ac.logstd],lr=1e-5)],eps=1e-5)
    seeds=list(range(1204000,1204032));base,_=evaluate(cpu(ac),seeds)
    key=lambda v:(v['stable20'],v['success20'],v['finalSuccess'],v['meanReturn'])
    best=key(base)
    save(out/'best.pt',dict(policy=cpu(ac).state_dict(),update=0,validation=base))
    write(out/'config.json',dict(**vars(a),baseline=base,worlds=a.workers*a.per,physics='original CPU MuJoCo 3.8.1 with NoSlip 20',bcEpoch=bcinfo['epoch'],rewardScale=60.))
    print('PPO_BASELINE',base,flush=True)
    pool=Pool(a.workers,a.per,1212000,200)
    n=a.workers*a.per;T=a.rollout;dev='cuda';hist=[];total=0;start=time.perf_counter()
    obs=torch.tensor(pool.obs,device=dev)
    ob=torch.empty((T,n,45),device=dev);raw=torch.empty((T,n,24),device=dev);lp=torch.empty((T,n),device=dev);v=torch.empty_like(lp);rew=torch.empty_like(lp);done=torch.empty_like(lp);adv=torch.empty_like(lp)
    try:
        for update in range(1,a.updates+1):
            t0=time.perf_counter();eps=[]
            with torch.no_grad():
                for k in range(T):
                    ob[k].copy_(obs);d=ac.dist(obs);z=d.sample();raw[k].copy_(z);lp[k].copy_(d.log_prob(z).sum(1));v[k].copy_(ac.v(obs))
                    o,r,b,e=pool.step(z.tanh().cpu().numpy());eps.extend(e)
                    obs=torch.tensor(o,device=dev);rew[k].copy_(torch.tensor(r,device=dev)/60.);done[k].copy_(torch.tensor(b,device=dev))
                nextv=ac.v(obs);last=torch.zeros(n,device=dev)
                for k in reversed(range(T)):
                    delta=rew[k]+.99*(1-done[k])*(nextv if k==T-1 else v[k+1])-v[k]
                    last=delta+.99*.95*(1-done[k])*last;adv[k].copy_(last)
                target=(adv+v).flatten();A=adv.flatten();A=(A-A.mean())/A.std().clamp(min=1e-6)
            x=ob.flatten(0,1);z=raw.flatten(0,1);old=lp.flatten();oldv=v.flatten()
            kl=[];losses=[]
            for ep in range(4):
                for idx in torch.randperm(len(x),device=dev).split(2048):
                    d=ac.dist(x[idx]);l=d.log_prob(z[idx]).sum(1);ratio=(l-old[idx]).exp()
                    pg=-torch.minimum(ratio*A[idx],ratio.clamp(.8,1.2)*A[idx]).mean()
                    val=ac.v(x[idx]);clip=oldv[idx]+(val-oldv[idx]).clamp(-.2,.2);vl=.5*torch.maximum((val-target[idx])**2,(clip-target[idx])**2).mean()
                    # Small behavior-anchor regularizer, decayed over training.
                    anchor=(ac.actor(x[idx])-prior(x[idx])).square().mean()
                    loss=vl if update<=5 else pg+.5*vl+max(.2,1-update/a.updates)*anchor-.0005*d.entropy().sum(1).mean()
                    assert torch.isfinite(loss)
                    opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(ac.parameters(),.5);opt.step()
                    kl.append((old[idx]-l).mean().detach());losses.append(loss.detach())
                if float(torch.stack(kl).mean())>.025:break
            total+=T*n
            row=dict(update=update,steps=total,physicsSteps=5*total,seconds=time.perf_counter()-start,stepsPerSecond=T*n/(time.perf_counter()-t0),episodes=len(eps),trainStable20=float(np.mean(np.asarray(eps)[:,2])) if eps else None,trainSuccess20=float(np.mean(np.asarray(eps)[:,1])) if eps else None,kl=float(torch.stack(kl).mean()),loss=float(torch.stack(losses).mean()))
            if update==1 or update%10==0 or update==a.updates:
                q=cpu(ac);val,_=evaluate(q,seeds);row['validation']=val;ck=dict(policy=q.state_dict(),update=update,validation=val)
                if key(val)>best:best=key(val);save(out/'best.pt',ck)
                save(out/'latest.pt',dict(**ck,critic=ac.critic.state_dict(),optimizer=opt.state_dict(),logstd=ac.logstd.detach().cpu()))
            hist.append(row);write(out/'history.json',hist);print('PPO',row,flush=True)
    finally:pool.close()
    p,ck=load(out/'best.pt');base,_=load('runs/bc01/best.pt');results={}
    for name,pol in [('BC',base),('PPO',p)]:
        s,r=evaluate(pol,list(range(1205000,1205096)));results[name]=dict(summary=s,trials=r);print('PPO_BLIND',name,s,flush=True);write(out/'blind.json',dict(bestUpdate=ck['update'],results=results))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=12);p.add_argument('--per',type=int,default=8);p.add_argument('--rollout',type=int,default=64);p.add_argument('--updates',type=int,default=100);p.add_argument('--out',default='runs/ppo01');main(p.parse_args())
