"""Shared observation, actuator and orientation conventions for CPU and GPU."""
import math
import numpy as np,torch
ACT=18;OBS=90;SUB=10;HORIZON=200
SCALE=np.array([.65,.65,.19,.55,.9,.24,.9,1.55,.24,.9,1.55,.24,.9,1.55,.45,.24,.9,1.55],np.float32)
def obs(q,v,rel,axis,goal,last,force,cos,k,mass,friction):
    return torch.cat((q[:,:24]/1.5,(v[:,:24]/12).clamp(-4,4),rel/.1,axis,(v[:,24:]/torch.as_tensor([.6,.6,.6,10,10,10],device=q.device)).clamp(-4,4),goal,last,(force/5).clamp(0,5),cos[:,None],(k/HORIZON)[:,None],(mass/.055)[:,None],friction[:,None]),1).clamp(-10,10)
def relative(R,P,O,A):
    return torch.bmm(R.transpose(1,2),(O-P)[:,:,None]).squeeze(-1),torch.bmm(R.transpose(1,2),A[:,:,None]).squeeze(-1)
def target(axis,angle):
    c=torch.cos(angle);s=torch.sin(angle)
    x,y,z=axis.unbind(1);return torch.stack([c*x+s*z,y,-s*x+c*z],1)
def error(a,b):
    # Directed long axis, marked ends, no demand on spin around the axis.
    return torch.acos((a*b).sum(1).clamp(-1.,1.))
def controls(base,last,lo,hi):
    cmd=base.clone()
    cmd[:,2:]=(base[:,2:]+torch.as_tensor(SCALE,device=base.device)*last).clamp(lo[2:],hi[2:])
    return cmd
