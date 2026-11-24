"""Independent CPU reference for the elongated-rod manipulation task."""
import math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"rl-v3"))
import numpy as np,torch,mujoco as mj
from env import Hand,BASE,SCALE,START_YAW,DT,SUB,OBS
from rod_shape import RodHand
from manipulate_rod import MAN_OBS,MAN_ACT,START_MAN,HORIZON,PRETRAINED
from train import Policy
MSCALE=np.array([4.,4.,4.,4.,3.,16.,16.,16.,16.,26.])
SIGN=np.array([-1.,1.,-1.])
BASE_YAW=np.array([2.,1.,-1.,-3.,0.])
def mul(a,b):
 return np.r_[a[0]*b[0]-np.dot(a[1:],b[1:]),a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:],b[1:])]
def err(goal,q):
 q=q/max(1e-8,np.linalg.norm(q));e=mul(goal,np.r_[q[0],-q[1:]])
 if e[0]<0:e=-e
 return e,2*math.atan2(np.linalg.norm(e[1:]),max(1e-8,abs(e[0])))
class CPURod:
 def __init__(self,seed,angle=10.,difficulty=.35,scale=2.):
  self.h=RodHand();self.h.reset(seed,difficulty,horizon=HORIZON)
  self.seed=seed;self.angle=math.radians(angle);self.scale=scale
  self.axes=np.array(self.h.params['axes']);self.radius=self.h.params['radius']
  self.last=np.zeros(6);self.mlast=np.zeros(10);self.goal=np.array([1.,0.,0.,0.])
  self.dq=np.array([math.cos(self.angle/2),0.,0.,math.sin(self.angle/2)])
  self.k=0;self.grasped=False;self.started=False;self.minheight=1.;self.maxdrift=0.;self.mincontacts=5;self.streak=0;self.best_error=math.pi
  self.center=self.h.center.copy();self.release_center=self.center.copy();self.initial=.25-self.center[2]
  self.forces=np.zeros(5);self.total=0.;self.done=False;self.final=None;self.frames=[]
 @property
 def d(self):return self.h.d
 def q(self):return self.d.qpos[self.h.obj_q+3:self.h.obj_q+7]
 def z(self):return float(self.d.qpos[self.h.obj_q+2])
 def rel(self):return (self.d.qpos[self.h.obj_q:self.h.obj_q+3]-self.d.mocap_pos[self.h.wrist])*SIGN
 def baseobs(self):
  h=self.h;d=self.d
  return np.clip(np.r_[d.qpos[h.qj].ravel()/1.5,np.clip(d.qvel[h.vj].ravel()/15,-3,3),
   self.rel()/.065,np.clip(d.qvel[h.obj_v:h.obj_v+3]*SIGN/.5,-3,3),self.center/.065,
   np.clip(self.forces/4,0,5),d.ten_length/.025,self.last,min(self.k*DT/2,1.5),float(d.eq_active[h.eq]),self.radius/.025],-10,10).astype('float32')
 def obs(self):
  e,_=err(self.goal,self.q()) if self.started else (self.dq,abs(self.angle))
  omega=self.d.qvel[self.h.obj_v+3:self.h.obj_v+6]
  return np.clip(np.r_[self.baseobs(),e,np.clip(omega/8,-3,3),self.mlast,self.axes/.05,np.clip((self.k-START_MAN)*DT/4,0,1),self.angle/math.pi],-10,10).astype('float32')
 def step(self,base_action,action,record=False):
  h=self.h;d=self.d;m=h.m
  if self.k==START_MAN:
   self.goal=mul(self.dq,self.q().copy());self.started=True
   self.grasped=bool(self.z()>self.initial+.035 and (self.forces>.005).sum()>=2 and not d.eq_active[h.eq])
  self.last=.65*self.last+.35*base_action
  u=BASE+SCALE*self.last;ty=BASE_YAW.copy();ty[-1]=u[5];u=np.r_[u[:5],ty]
  self.mlast=.65*self.mlast+.35*(action if self.started else np.zeros(MAN_ACT))
  u+=MSCALE*self.scale*self.mlast
  u[:5]=np.clip(u[:5],0,25);u[5:9]=np.clip(u[5:9],-22,22);u[9]=np.clip(u[9],-95,65)
  if self.k==25:self.release_center=self.rel().copy();d.eq_active[h.eq]=0
  if self.k<50:
   for sub in range(SUB):
    t=(self.k*SUB+sub)*.001;a=np.clip(t/.7,0,1);a=a*a*(3-2*a);l=np.clip((t-1.10)/.85,0,1);l=l*l*(3-2*l)
    d.ctrl[h.yaws]=(1-a)*START_YAW+a*np.radians(u[5:]);d.ctrl[h.pulls]=a*u[:5];d.mocap_pos[h.wrist,2]=.25+.065*l
    mj.mj_step(m,d)
  else:
   d.ctrl[h.yaws]=np.radians(u[5:]);d.ctrl[h.pulls]=u[:5];d.mocap_pos[h.wrist,2]=.315;mj.mj_step(m,d,nstep=SUB)
  self.k+=1;self.forces,count,self.total,points=h.contacts(record)
  n=int((self.forces>.005).sum());z=self.z();dist=float(np.linalg.norm(self.rel()-self.release_center))
  _,error=err(self.goal,self.q());omega=float(np.linalg.norm(d.qvel[h.obj_v+3:h.obj_v+6]))
  bad=not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all();dropped=self.k>30 and (z<.09 or dist>.12)
  if self.started:
   self.best_error=min(self.best_error,error);self.minheight=min(self.minheight,z);self.maxdrift=max(self.maxdrift,dist);self.mincontacts=min(self.mincontacts,n)
   near=error<math.radians(8.) and omega<1.2 and z>self.initial+.035 and n>=2 and dist<.025 and not d.eq_active[h.eq]
   self.streak=self.streak+1 if near else 0
  self.done=bool(bad or dropped or self.k>=HORIZON)
  retained=bool(self.grasped and not bad and not dropped and self.k>=HORIZON and self.minheight>self.initial+.035 and self.maxdrift<.025 and self.mincontacts>=2 and not d.eq_active[h.eq])
  if record:self.frames.append(dict(t=self.k*DT,wrist=d.mocap_pos[h.wrist].tolist(),object=d.qpos[h.obj_q:h.obj_q+3].tolist(),objectQuat=self.q().tolist(),targetQuat=self.goal.tolist(),errorDegrees=math.degrees(error),targetDegrees=math.degrees(self.angle),angularSpeed=omega,angles=np.degrees(d.qpos[h.qj[:,1:]]).tolist(),yaw=np.degrees(d.qpos[h.qj[:,0]]).tolist(),tendonForce=d.actuator_force[h.pulls].tolist(),tendonLength=(1000*d.ten_length).tolist(),contactCount=count,digits=[name for name,f in zip(['index','middle','ring','little','thumb'],self.forces) if f>.005],contacts=points,normalForce=self.total,fixture=bool(d.eq_active[h.eq]),action=self.mlast.tolist(),streak=self.streak))
  if self.done:
   qq=self.q().copy();yaw=math.degrees(2*math.atan2(qq[3],qq[0]))
   self.final=dict(seed=self.seed,params=h.params,targetDegrees=math.degrees(self.angle),objectYawDegrees=yaw,objectQuat=qq.tolist(),success=retained and self.streak>=10,retained=retained,grasped=self.grasped,bad=bool(bad),dropped=bool(dropped),errorDegrees=math.degrees(error),bestErrorDegrees=math.degrees(self.best_error),angularSpeed=omega,streak=self.streak,finalHeight=z,maxHoldDrift=self.maxdrift,fixture=bool(d.eq_active[h.eq]),steps=self.k)
def evaluate(policy,seeds,angles=None,difficulty=.35,scale=2.,fixed_action=None,record=False,sign_by_goal=False):
 if angles is None:angles=[(1 if i%2==0 else -1)*10 for i in range(len(seeds))]
 base=Policy();base.load_state_dict(torch.load(PRETRAINED,map_location='cpu',weights_only=True)['policy']);base.eval()
 hs=[CPURod(s,a,difficulty,scale) for s,a in zip(seeds,angles)]
 for _ in range(HORIZON):
  live=[h for h in hs if not h.done]
  if not live:break
  with torch.no_grad():
   b=base.actor(torch.tensor(np.stack([h.baseobs() for h in live]))).tanh().numpy()
   if policy is not None:acts=policy(torch.tensor(np.stack([h.obs() for h in live]))).detach().numpy()
   elif fixed_action is not None:
    v=np.asarray(fixed_action,dtype=np.float32)
    acts=np.stack([v*(1. if (not sign_by_goal or h.angle>=0) else -1.) for h in live])
   else:acts=np.zeros((len(live),MAN_ACT),np.float32)
  for h,ba,ma in zip(live,b,acts):h.step(ba,ma,record)
 rows=[h.final for h in hs];ret=[r for r in rows if r['retained']]
 score=dict(n=len(rows),grasped=float(np.mean([r['grasped'] for r in rows])),retained=float(np.mean([r['retained'] for r in rows])),success=float(np.mean([r['success'] for r in rows])),meanErrorRetained=float(np.mean([r['errorDegrees'] for r in ret])) if ret else None,meanCappedError=float(np.mean([r['errorDegrees'] if r['retained'] else 180. for r in rows])),bad=sum(r['bad'] for r in rows))
 return score,rows,[h.frames for h in hs]
