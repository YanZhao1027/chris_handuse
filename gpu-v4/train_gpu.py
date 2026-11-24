"""Large batched PPO. GPU physics, actions, GAE and gradients; CPU reference validation."""
import argparse,json,time,os,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from gpu_env import GPUHand,V3,OBS,ACT
from train import Policy,evaluate,save,writejson

def cpu_policy(policy):
    # Checkpoint copies explicitly synchronize the training stream.
    p=Policy()
    p.load_state_dict({k:v.detach().cpu() for k,v in policy.state_dict().items()})
    p.eval()
    return p
def train(args):
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision('high')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'history.json').exists() and not args.resume:
        raise RuntimeError('run already exists; use --resume')
    env=GPUHand(args.worlds,seed=args.seed,difficulty=args.difficulty,variants=args.variants,iterations=args.iterations)
    # Initialize after Warp owns the current CUDA stream.
    policy=Policy().cuda()
    warm=torch.load(V3/'runs/ppo01/best.pt',map_location='cpu',weights_only=True)
    policy.load_state_dict(warm['policy'])
    opt=torch.optim.Adam(policy.parameters(),lr=args.lr,eps=1e-5)
    start=0;total=0;history=[];best=(-1.,-1e9)
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=False)
        policy.load_state_dict(ck['policy']);opt.load_state_dict(ck['optimizer'])
        history=ck['history'];start=ck['update'];total=ck['steps'];best=tuple(ck['best'])
    val_seeds=list(range(100000,100000+args.val_n))
    if start==0:
        initial,_,_=evaluate(cpu_policy(policy),val_seeds,1.)
        best=(initial['success'],initial['meanReturn'])
        save(out/'initial.pt',{'policy':cpu_policy(policy).state_dict(),'obs':OBS,'act':ACT,'validation':initial})
        save(out/'best.pt',{'policy':cpu_policy(policy).state_dict(),'obs':OBS,'act':ACT,'update':0,'validation':initial})
        print('V3_REFERENCE_VALIDATION',json.dumps(initial),flush=True)
    config=vars(args).copy()
    config.update({'observation_dim':OBS,'action_dim':ACT,'physics':'mujoco-warp','source':str(V3/'runs/ppo01/best.pt'),'physics_dt':.001,'policy_dt':.04,'model_randomization':'fixed mass/radius/friction per variant; reset position resampled independently','seed_note':'validation seeds 100000+, new blind test seeds 40270000+'})
    writejson(out/'config.json',config)
    obs=env.obs()
    n=args.worlds;T=args.rollout
    obs_buf=torch.empty((T,n,OBS),device='cuda')
    act_buf=torch.empty((T,n,ACT),device='cuda')
    lp_buf=torch.empty((T,n),device='cuda')
    v_buf=torch.empty((T,n),device='cuda')
    r_buf=torch.empty((T,n),device='cuda')
    done_buf=torch.empty((T,n),device='cuda')
    adv=torch.empty_like(r_buf)
    best_run=best
    ts=time.perf_counter()
    for update in range(start+1,args.updates+1):
        u0=time.perf_counter()
        completed=torch.zeros((),device='cuda');successes=torch.zeros((),device='cuda');returns=torch.zeros((),device='cuda');bad=torch.zeros((),device='cuda')
        for k in range(T):
            obs_buf[k].copy_(obs)
            raw,action,lp,v=policy.act(obs)
            act_buf[k].copy_(raw);lp_buf[k].copy_(lp);v_buf[k].copy_(v)
            obs,reward,done,info=env.step(action)
            r_buf[k].copy_(reward);done_buf[k].copy_(done)
            completed+=done.sum();successes+=info['success'].sum()
            returns+=(info['return']*done).sum();bad+=info['bad'].sum()
        # GPU buffers are shared with Warp. No per-world transfer in rollout.
        with torch.no_grad():
            nextv=policy.value(obs)
            last=torch.zeros(n,device='cuda')
            for k in reversed(range(T)):
                delta=r_buf[k]+.99*(1-done_buf[k])*(nextv if k==T-1 else v_buf[k+1])-v_buf[k]
                last=delta+.99*.95*(1-done_buf[k])*last;adv[k].copy_(last)
            ret=(adv+v_buf).flatten()
            advantage=adv.flatten();advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
        x=obs_buf.flatten(0,1);actions=act_buf.flatten(0,1);oldlp=lp_buf.flatten();oldv=v_buf.flatten()
        size=len(x);losses=[];kls=[];epochs=0
        for epoch in range(args.epochs):
            for ids in torch.randperm(size,device='cuda').split(args.minibatch):
                dist=policy.dist(x[ids]);lp=dist.log_prob(actions[ids]).sum(-1);ratio=(lp-oldlp[ids]).exp()
                pg=-torch.minimum(ratio*advantage[ids],ratio.clamp(.8,1.2)*advantage[ids]).mean()
                vp=policy.value(x[ids]);vc=oldv[ids]+(vp-oldv[ids]).clamp(-.2,.2)
                vl=.5*torch.maximum((vp-ret[ids])**2,(vc-ret[ids])**2).mean()
                ent=dist.entropy().sum(-1).mean();loss=pg+.5*vl-.003*ent
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),.5);opt.step()
                losses.append(loss.detach());kls.append((oldlp[ids]-lp).mean().detach())
            epochs+=1
            if float(torch.stack(kls).mean())>.025:break
        torch.cuda.synchronize()
        total+=size;sec=time.perf_counter()-u0
        ep=int(completed.item());b=int(bad.item());s=int(successes.item())
        capacity=env.capacities()
        r={'update':update,'steps':total,'physicsSteps':total*40,'worlds':n,'seconds':time.perf_counter()-ts,'updateSeconds':sec,'stepsPerSecond':size/sec,'episodes':ep,'trainSuccess':s/ep if ep else None,'trainReturn':float(returns.item())/ep if ep else None,'loss':float(torch.stack(losses).mean()),'approxKL':float(torch.stack(kls).mean()),'epochs':epochs,'bad':b,'capacity':capacity,'vramFreeGiB':torch.cuda.mem_get_info()[0]/2**30}
        if b or not capacity['finite'] or capacity['contacts']>=capacity['contactCapacity'] or capacity['maxConstraints']>=capacity['constraintCapacity']:
            writejson(out/'failure.json',r);raise RuntimeError('Nonfinite state or exceeded solver buffers')
        # CPU validation uses the same reference dynamics but independent parameters.
        if update==start+1 or update%args.eval_every==0 or update==args.updates:
            cp=cpu_policy(policy)
            val,_,_=evaluate(cp,val_seeds,1.)
            r['validation']=val
            key=(val['success'],val['meanReturn'])
            if key>best:
                best=key
                save(out/'best.pt',{'policy':cp.state_dict(),'obs':OBS,'act':ACT,'update':update,'validation':val})
        history.append(r);writejson(out/'history.json',history)
        if update==start+1 or update%args.eval_every==0 or update==args.updates:
            cp=cpu_policy(policy)
            save(out/'latest.pt',{'policy':cp.state_dict(),'optimizer':opt.state_dict(),'update':update,'steps':total,'history':history,'best':best,'obs':OBS,'act':ACT})
        print(json.dumps(r,separators=(',',':')),flush=True)
    print('GPU_TRAINING_COMPLETE',json.dumps({'steps':total,'physicsSteps':total*40,'best':best}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--worlds',type=int,default=4096);p.add_argument('--rollout',type=int,default=32)
    p.add_argument('--updates',type=int,default=120);p.add_argument('--variants',type=int,default=512)
    p.add_argument('--iterations',type=int,default=80);p.add_argument('--difficulty',type=float,default=1.)
    p.add_argument('--seed',type=int,default=2026112404);p.add_argument('--lr',type=float,default=1.0e-4)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--minibatch',type=int,default=8192)
    p.add_argument('--eval-every',type=int,default=5);p.add_argument('--val-n',type=int,default=32)
    p.add_argument('--out',default='runs/ppo-gpu01');p.add_argument('--resume')
    train(p.parse_args())
