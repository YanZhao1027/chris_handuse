"""Unmodified Adroit task and held-out seeds, no hidden state or success relabeling."""
import json
from pathlib import Path
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from torch import nn
gym.register_envs(gymnasium_robotics)
OBS,ACT=45,24
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean',torch.zeros(OBS))
        self.register_buffer('std',torch.ones(OBS))
        self.net=nn.Sequential(nn.Linear(OBS,256),nn.LayerNorm(256),nn.SiLU(),nn.Linear(256,256),nn.LayerNorm(256),nn.SiLU(),nn.Linear(256,ACT),nn.Tanh())
    def forward(self,x):
        return self.net(((x-self.mean)/self.std).clamp(-20,20))
def save(path,obj):
    p=Path(path);q=p.with_suffix('.tmp');torch.save(obj,q);q.replace(p)
def write(path,obj):
    p=Path(path);q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2));q.replace(p)
def evaluate(policy,seeds,kind='policy',horizon=200):
    envs=[gym.make('AdroitHandPen-v1',max_episode_steps=horizon) for _ in seeds]
    obs=np.stack([e.reset(seed=int(s))[0] for e,s in zip(envs,seeds)])
    n=len(envs);ret=np.zeros(n);count=np.zeros(n,int);streak=np.zeros(n,int);best=np.zeros(n,int);drop=np.zeros(n,bool)
    rng=np.random.default_rng(12001)
    final=np.zeros(n,bool)
    with torch.inference_mode():
        for _ in range(horizon):
            if kind=='zero':a=np.zeros((n,ACT),np.float32)
            elif kind=='random':a=rng.uniform(-1,1,(n,ACT)).astype('float32')
            else:a=policy(torch.tensor(obs,dtype=torch.float32)).numpy()
            for i,e in enumerate(envs):
                obs[i],r,term,trunc,info=e.step(a[i]);ret[i]+=r
                final[i]=bool(info['success']);count[i]+=final[i];streak[i]=streak[i]+1 if final[i] else 0;best[i]=max(best[i],streak[i]);drop[i]|=obs[i,26]<.075
                assert np.isfinite(obs[i]).all() and np.isfinite(r)
    rows=[dict(seed=int(s),return_=float(ret[i]),successSteps=int(count[i]),longestSuccessStreak=int(best[i]),finalSuccess=bool(final[i]),success20=bool(count[i]>=20),stable20=bool(best[i]>=20),everBelowHeight=bool(drop[i])) for i,s in enumerate(seeds)]
    for e in envs:e.close()
    summary=dict(n=n,horizon=horizon,meanReturn=float(ret.mean()),success20=float((count>=20).mean()),stable20=float((best>=20).mean()),finalSuccess=float(final.mean()),everSuccess=float((count>0).mean()),everBelowHeight=float(drop.mean()),meanSuccessSteps=float(count.mean()))
    return summary,rows
def load(path):
    c=torch.load(path,map_location='cpu',weights_only=True);p=Policy();p.load_state_dict(c['policy']);p.eval();return p,c
