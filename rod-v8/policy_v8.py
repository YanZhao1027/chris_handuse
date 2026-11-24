"""Goal-dependent searched-action prior plus learned observation feedback."""
import torch
from torch import nn
from torch.distributions import Normal
from manipulate_rod import MAN_OBS,MAN_ACT
class Actor(nn.Module):
    def __init__(self,priors=None):
        super().__init__()
        if priors is None:priors=torch.zeros(2,MAN_ACT)
        self.register_buffer('prior',torch.atanh(torch.as_tensor(priors,dtype=torch.float32).clamp(-.95,.95)))
        self.net=nn.Sequential(nn.Linear(MAN_OBS,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,MAN_ACT))
        for l in self.net:
            if isinstance(l,nn.Linear):nn.init.orthogonal_(l.weight,2**.5);nn.init.zeros_(l.bias)
        nn.init.orthogonal_(self.net[-1].weight,.005)
    def forward(self,x):
        sign=(x[:,-1]>=0).long()
        # Amplify small orientation-error and goal components for learning.
        z=x.clone()
        z[:,69:72]*=4.;z[:,-1]*=8.
        return self.prior[sign]+self.net(z)
class ManipPolicy(nn.Module):
    def __init__(self,priors=None):
        super().__init__()
        self.actor=Actor(priors)
        self.critic=nn.Sequential(nn.Linear(MAN_OBS,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
        for l in self.critic:
            if isinstance(l,nn.Linear):nn.init.orthogonal_(l.weight,2**.5);nn.init.zeros_(l.bias)
        nn.init.orthogonal_(self.critic[-1].weight,1.)
        self.logstd=nn.Parameter(torch.full((MAN_ACT,),-1.9))
    def dist(self,x):return Normal(self.actor(x),self.logstd.clamp(-3.5,-.8).exp())
    def value(self,x):return self.critic(x).squeeze(-1)
    @torch.no_grad()
    def act(self,x):
        d=self.dist(x);r=d.sample()
        return r,r.tanh(),d.log_prob(r).sum(-1),self.value(x)
    def command(self,x):return self.actor(x).tanh()
def load_policy(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    p=ManipPolicy();p.load_state_dict(ck['policy']);p.eval()
    return p,ck
