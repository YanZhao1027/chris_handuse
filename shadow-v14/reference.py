"""Independent CPU MuJoCo trials on fresh grasp seeds and independently randomized physics."""
import json,math
from pathlib import Path
import numpy as np,torch,mujoco as mj
from model import make_model,ids,digitmap,metrics
from search_grasp import trial,write
from common import *
class Case:
    def __init__(self,seed,angle,difficulty=1.,plane=0):
        self.seed=seed;self.angle=angle;self.plane=plane
        cfg=json.loads(Path('grasp.json').read_text())
        r=trial(cfg,seed,difficulty,record=True)
        self.valid=r['success'];self.init=r
        self.m=make_model();self.d=mj.MjData(self.m);self.j=ids(self.m);self.mp=digitmap(self.m)
        j=self.j;m=self.m;rng=np.random.default_rng(seed+9813)
        self.mass=.055*(1+rng.uniform(-.20,.20)*difficulty);self.friction=1.075+rng.uniform(-.225,.225)*difficulty
        m.body_inertia[j['rod']]*=self.mass/m.body_mass[j['rod']];m.body_mass[j['rod']]=self.mass;m.geom_friction[j['shaft'],0]=self.friction
        mj.mj_setConst(m,self.d)
    def rel(self):
        j=self.j;d=self.d;R=d.xmat[j['palm']].reshape(3,3)
        return R.T@(d.xpos[j['rod']]-d.xpos[j['palm']]),R.T@d.xmat[j['rod']].reshape(3,3)[:,2]
    def reset(self):
        self.done=not self.valid;self.k=0;self.streak=0;self.opp=0;self.sd=0.;self.od=0.;self.drift=0.;self.drop=False;self.bad=False;self.err=math.pi;self.frames=[]
        if not self.valid:return
        m,d=self.m,self.d;mj.mj_resetData(m,d)
        d.qpos[:]=self.init['qpos'];d.qvel[:]=self.init['qvel'];d.ctrl[:]=self.init['ctrl'];self.base=d.ctrl.copy();self.last=np.zeros(ACT,np.float32)
        mj.mj_forward(m,d);self.metric=metrics(m,d,self.j,self.mp);self.start,a=self.rel()
        theta=math.radians(self.angle);c,s=math.cos(theta),math.sin(theta);self.goal=np.array([c*a[0]+s*a[2],a[1],-s*a[0]+c*a[2]]) if not self.plane else np.array([c*a[0]-s*a[1],s*a[0]+c*a[1],a[2]])
        self.off=np.zeros(5,int);self.on=np.zeros(5,int);self.ever_loaded=np.asarray(self.metric['forces'])>.10;self.released=np.zeros(5,bool);self.recontacts=np.zeros(5,int)
    def observation(self):
        p,a=self.rel();d=self.d;m=self.metric
        return np.clip(np.r_[d.qpos[:24]/1.5,np.clip(d.qvel[:24]/12,-4,4),p/.1,a,np.clip(d.qvel[24:]/[.6,.6,.6,10,10,10],-4,4),self.goal,self.last,np.clip(np.asarray(m['forces'])/5,0,5),m['cos'],self.k/HORIZON,self.mass/.055,self.friction],-10,10).astype(np.float32)
    def step(self,a,record=False):
        m,d,j=self.m,self.d,self.j
        self.last=.55*self.last+.45*np.clip(a,-1,1)
        d.ctrl[:]=controls_np(self.base,self.last,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
        for _ in range(SUB):
            mj.mj_step(m,d)
            for c in d.contact:
                aa,bb=int(c.geom1),int(c.geom2)
                if j['floor'] in (aa,bb):continue
                dep=max(0.,-float(c.dist))
                if j['shaft'] in (aa,bb):self.od=max(self.od,dep)
                else:self.sd=max(self.sd,dep)
        self.k+=1;self.metric=metrics(m,d,j,self.mp)
        f=np.asarray(self.metric['forces']);low=f<.05;high=f>.10
        self.off=np.where(low,self.off+1,0);self.ever_loaded|=high;self.released|=(self.off>=5)&self.ever_loaded
        self.on=np.where(high,self.on+1,0);back=self.released&(self.on>=5);self.recontacts+=back;self.released&=~back
        p,axis=self.rel();self.err=math.acos(float(np.clip(np.dot(axis,self.goal),-1,1)));dist=float(np.linalg.norm(p-self.start));self.drift=max(self.drift,dist)
        omega=np.linalg.norm(d.qvel[27:30]);self.opp+=self.metric['opposed']
        near=self.err<math.radians(5) and dist<.035 and omega<1.4 and self.metric['opposed']
        self.streak=self.streak+1 if near else 0
        self.bad=not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all()
        self.drop=d.qpos[26]<.10 or dist>.085
        self.done=self.bad or self.drop or self.k>=HORIZON
        if record:self.frames.append(dict(t=self.k*.02,recontacts=self.recontacts.tolist(),qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),ctrl=d.ctrl.tolist(),goal=self.goal.tolist(),axis=axis.tolist(),errorDegrees=math.degrees(self.err),metrics=self.metric,peakSelfMM=self.sd*1000,peakObjectMM=self.od*1000,streak=self.streak))
    def result(self):
        retained=bool(self.valid and not self.bad and not self.drop and self.k==HORIZON and self.drift<.04)
        success=bool(retained and self.streak>=20 and self.opp/max(1,self.k)>=.75 and self.sd<.0015 and self.od<.003)
        return dict(seed=self.seed,angle=self.angle,plane=self.plane,recontacts=int(getattr(self,'recontacts',np.zeros(5)).sum()),recontactDigits=int((getattr(self,'recontacts',np.zeros(5))>0).sum()),mass=self.mass,friction=self.friction,validInitial=bool(self.valid),success=success,retained=retained,errorDegrees=math.degrees(self.err),opposition=self.opp/max(1,self.k),selfMM=self.sd*1000,objMM=self.od*1000,driftMM=self.drift*1000,streak=self.streak,bad=bool(self.bad),drop=bool(self.drop))
def cases(seed,n,angles=(-15,15,-30,30),difficulty=1.):
    return [Case(seed+i, spec[0] if isinstance(spec,tuple) else spec,difficulty, spec[1] if isinstance(spec,tuple) else 0) for i,spec in enumerate([angles[j%len(angles)] for j in range(n)])]
def evaluate(policy,cs,record=False):
    for c in cs:c.reset()
    with torch.inference_mode():
        while True:
            live=[c for c in cs if not c.done]
            if not live:break
            x=torch.tensor(np.stack([c.observation() for c in live]))
            a=np.zeros((len(live),ACT),np.float32) if policy is None else policy.command(x).numpy()
            for c,u in zip(live,a):c.step(u,record)
    rows=[c.result() for c in cs]
    def group(rs):
        return dict(n=len(rs),success=float(np.mean([x['success'] for x in rs])),retained=float(np.mean([x['retained'] for x in rs])),meanError=float(np.mean([x['errorDegrees'] if x['retained'] else 180 for x in rs])),meanOpposition=float(np.mean([x['opposition'] for x in rs])),invalidInitial=sum(not x['validInitial'] for x in rs),bad=sum(x['bad'] for x in rs),meanRecontacts=float(np.mean([x['recontacts'] for x in rs])),successWithRecontact=float(np.mean([x['success'] and x['recontacts']>0 for x in rs])))
    s=group(rows);s['byAngle']={str(g):group([r for r in rows if r['angle']==g and r['plane']==0]) for g in sorted(set(c.angle for c in cs if c.plane==0))};s['byPlaneAngle']={f'{pl}:{g}':group([r for r in rows if r['plane']==pl and r['angle']==g]) for pl,g in sorted(set((c.plane,c.angle) for c in cs))}
    s['negative']=group([r for r in rows if r['angle']<0])['success'];s['positive']=group([r for r in rows if r['angle']>0])['success']
    return s,rows
