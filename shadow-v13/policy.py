import torch
from torch import nn
from torch.distributions import Normal
from common import OBS,ACT
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(OBS,256),nn.ELU(),nn.Linear(256,256),nn.ELU(),nn.Linear(256,ACT))
        self.critic=nn.Sequential(nn.Linear(OBS,256),nn.ELU(),nn.Linear(256,256),nn.ELU(),nn.Linear(256,1))
        for net in (self.actor,self.critic):
            for layer in net:
                if isinstance(layer,nn.Linear):nn.init.orthogonal_(layer.weight,2**.5);nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01);nn.init.orthogonal_(self.critic[-1].weight,1.)
        self.logstd=nn.Parameter(torch.full((ACT,),-1.6))
    def dist(self,x):return Normal(self.actor(x).clamp(-4,4),self.logstd.clamp(-3,-.5).exp())
    def value(self,x):return self.critic(x).squeeze(-1)
    def command(self,x):return self.actor(x).tanh()
    @torch.no_grad()
    def act(self,x):
        d=self.dist(x);z=d.sample()
        return z,z.tanh(),d.log_prob(z).sum(1),self.value(x)
def cpu(p):
    q=Policy();q.load_state_dict({k:v.detach().cpu() for k,v in p.state_dict().items()});q.eval();return q
def load(path):
    c=torch.load(path,map_location='cpu',weights_only=True);p=Policy();p.load_state_dict(c['policy']);p.eval();return p,c
