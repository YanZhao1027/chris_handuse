"""Shared observation, actuator and orientation conventions for CPU and GPU."""
import math
import numpy as np,torch
ACT=18;OBS=90;SUB=10;HORIZON=300
SCALE=np.array([.65,.65,.19,.55,.9,.24,.9,1.55,.24,.9,1.55,.24,.9,1.55,.45,.24,.9,1.55],np.float32)
def obs(q,v,rel,axis,goal,last,force,cos,k,mass,friction):
    return torch.cat((q[:,:24]/1.5,(v[:,:24]/12).clamp(-4,4),rel/.1,axis,(v[:,24:]/torch.as_tensor([.6,.6,.6,10,10,10],device=q.device)).clamp(-4,4),goal,last,(force/5).clamp(0,5),cos[:,None],(k/HORIZON)[:,None],(mass/.055)[:,None],friction[:,None]),1).clamp(-10,10)
def relative(R,P,O,A):
    return torch.bmm(R.transpose(1,2),(O-P)[:,:,None]).squeeze(-1),torch.bmm(R.transpose(1,2),A[:,:,None]).squeeze(-1)
def target(axis,angle,plane=None):
    # 0: palm Y (V13); 1: palm Z (second independent rotation plane).
    c=torch.cos(angle);s=torch.sin(angle);x,y,z=axis.unbind(1)
    a=torch.stack([c*x+s*z,y,-s*x+c*z],1)
    b=torch.stack([c*x-s*y,s*x+c*y,z],1)
    return a if plane is None else torch.where(plane[:,None].bool(),b,a)
def error(a,b):
    # Directed long axis, marked ends, no demand on spin around the axis.
    return torch.acos((a*b).sum(1).clamp(-1.,1.))
def controls(base,last,lo,hi):
    # Exact V13 response through 75% of normalized action; outer quarter
    # unlocks the entire legal servo range, including full finger opening.
    cmd=base.clone();a=last.clamp(-1,1);b=base[:,2:];scale=torch.as_tensor(SCALE,device=base.device)
    inner=(b+scale*a.clamp(-.75,.75)).clamp(lo[2:],hi[2:])
    endpoint=torch.where(a>=0,hi[2:],lo[2:])
    blend=((a.abs()-.75)/.25).clamp(0,1)
    cmd[:,2:]=inner+(endpoint-inner)*blend
    return cmd
def controls_np(base,last,lo,hi):
    a=np.clip(last,-1,1);b=base[2:];inner=np.clip(b+SCALE*np.clip(a,-.75,.75),lo[2:],hi[2:])
    endpoint=np.where(a>=0,hi[2:],lo[2:]);blend=np.clip((np.abs(a)-.75)/.25,0,1)
    cmd=base.copy();cmd[2:]=inner+(endpoint-inner)*blend
    return cmd
