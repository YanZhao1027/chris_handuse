"""Closed-loop residual RL over V2 MuJoCo. SI units, no object attachment after 1s."""
import sys,json,math
from pathlib import Path
import numpy as np
import mujoco as mj
V2=Path(__file__).resolve().parent.parent/'physics-v2'
sys.path.insert(0,str(V2))
from model import load,NAMES
CFG=json.loads((V2/'results/config.json').read_text())
DT=.04
SUB=40
HORIZON=100
ACT=6
OBS=20+20+3+3+3+5+5+6+3
BASE=np.array([CFG['pull']]*4+[CFG['thumb'],CFG['oppose']])
SCALE=np.array([2.5]*4+[1.8,18.])
START_YAW=np.radians([8,3,-3,-10,62])
def domain(seed,difficulty=1.):
    rng=np.random.default_rng(seed);s=difficulty
    return dict(center=(np.array(CFG['center'])+rng.uniform(-1,1,3)*np.array([.0035,.0035,.003])*s).tolist(),
        radius=CFG['radius']+rng.uniform(-.0015,.0015)*s,
        mass=CFG['mass']*(1+rng.uniform(-.35,.55)*s),
        friction=CFG['friction']*(1+rng.uniform(-.28,.48)*s))
class Hand:
    def __init__(self):
        self.m=load(**{k:CFG[k] for k in ['center','radius','mass','friction']})
        m=self.m; self.d=mj.MjData(m)
        self.oid=m.body('object').id;self.wid=m.body('wrist').id;self.wrist=m.body_mocapid[self.wid]
        self.fix=m.body_mocapid[m.body('fixture').id];self.eq=m.equality('fixture_hold').id
        self.ball=m.geom('ball').id;self.obj_q=int(m.joint('object_free').qposadr[0])
        self.obj_v=int(m.joint('object_free').dofadr[0])
        self.yaws=np.array([m.actuator(n+'_yaw_servo').id for n in NAMES])
        self.pulls=np.array([m.actuator(n+'_pull').id for n in NAMES])
        self.qj=np.array([[int(m.joint(n+'_yaw').qposadr[0])]+[int(m.joint(n+'_flex'+str(j)).qposadr[0]) for j in range(3)] for n in NAMES])
        self.vj=np.array([[int(m.joint(n+'_yaw').dofadr[0])]+[int(m.joint(n+'_flex'+str(j)).dofadr[0]) for j in range(3)] for n in NAMES])
        self.geom_digit={m.geom(n+'_segment'+str(j)).id:i for i,n in enumerate(NAMES) for j in range(3)}
        self.floor=m.geom('floor').id
        self.force_buf=np.zeros(6)
    def reset(self,seed,difficulty=1.,horizon=HORIZON,params=None):
        self.params=domain(seed,difficulty) if params is None else params
        m=self.m;d=self.d;p=self.params
        # Update geometry/mass before computing constants; no XML reload or hidden object constraints.
        m.geom_size[self.ball,0]=p['radius']
        m.geom_rbound[self.ball]=p['radius']
        m.body_mass[self.oid]=p['mass']
        m.body_inertia[self.oid]=.4*p['mass']*p['radius']**2
        m.geom_friction[:,0]=p['friction']
        # Sphere broad-phase bounding box follows the changed radius.
        m.geom_aabb[self.ball,3:]=p['radius']
        mj.mj_setConst(m,d);mj.mj_resetData(m,d)
        wp=np.array([0,0,.25])+np.array([-p['center'][0],p['center'][1],-p['center'][2]])
        d.qpos[self.obj_q:self.obj_q+3]=wp;d.qpos[self.obj_q+3:self.obj_q+7]=[1,0,0,0]
        d.mocap_pos[self.fix]=wp
        d.qpos[self.qj[:,0]]=START_YAW;d.ctrl[self.yaws]=START_YAW
        mj.mj_forward(m,d)
        self.k=0;self.horizon=horizon;self.last=np.zeros(ACT);self.center=np.array(p['center'])
        self.initial_height=.25-self.center[2];self.release_center=self.center.copy()
        self.minheight=1.;self.maxdrift=0.;self.mindigits=5;self.ep_return=0.;self.hold_steps=0
        self.prev_rel=self.relative()
        return self.obs()
    def relative(self):
        d=self.d
        return d.xmat[self.wid].reshape(3,3).T@(d.xpos[self.oid]-d.xpos[self.wid])
    def contacts(self,record=False):
        d=self.d;forces=np.zeros(5);count=0;total=0.;points=[]
        for k in range(d.ncon):
            con=d.contact[k];a=int(con.geom1);b=int(con.geom2)
            if a!=self.ball and b!=self.ball:continue
            other=b if a==self.ball else a
            if other==self.floor:continue
            mj.mj_contactForce(self.m,d,k,self.force_buf)
            force=max(0.,float(self.force_buf[0]));count+=1;total+=force
            i=self.geom_digit.get(other)
            if i is not None:forces[i]+=force
            if record:points.append({'p':con.pos.tolist(),'n':force,'part':mj.mj_id2name(self.m,mj.mjtObj.mjOBJ_GEOM,other)})
        return forces,count,total,points
    def obs(self,contact=None):
        d=self.d;rel=self.relative();force=self.contacts()[0] if contact is None else contact
        worldvel=d.qvel[self.obj_v:self.obj_v+3]
        localvel=d.xmat[self.wid].reshape(3,3).T@worldvel
        t=self.k*DT
        out=np.concatenate([d.qpos[self.qj].ravel()/1.5,np.clip(d.qvel[self.vj].ravel()/15,-3,3),
          rel/.065,np.clip(localvel/.5,-3,3),self.center/.065,np.clip(force/4,0,5),
          d.ten_length/.025,self.last,[min(t/2,1.5),float(d.eq_active[self.eq]),self.params['radius']/.025]])
        assert out.shape==(OBS,),out.shape
        return np.clip(out,-10,10).astype(np.float32)
    def step(self,action,record=False):
        d=self.d;m=self.m
        raw=np.clip(np.asarray(action),-1,1)
        # Smooth actuation, 25 Hz policy; nominal residual zero exactly recovers reference schedule.
        self.last=.65*self.last+.35*raw
        u=BASE+SCALE*self.last;u[:5]=np.clip(u[:5],0,25);u[5]=np.clip(u[5],-95,65)
        t=self.k*DT
        if self.k==25:
            self.release_center=self.relative().copy();d.eq_active[self.eq]=0
        # Exact 1ms reference ramp while closing/lifting; fast nstep otherwise.
        if self.k<50:
            for sub in range(SUB):
                ts=t+sub*.001
                a=min(1.,ts/.70);a=a*a*(3-2*a)
                d.ctrl[self.yaws]=(1-a)*START_YAW+a*np.radians([2,1,-1,-3,u[5]])
                d.ctrl[self.pulls]=a*u[:5]
                lift=np.clip((ts-1.10)/.85,0,1);lift=lift*lift*(3-2*lift)
                d.mocap_pos[self.wrist,2]=.25+.065*lift
                mj.mj_step(m,d)
        else:
            d.ctrl[self.yaws]=np.radians([2,1,-1,-3,u[5]])
            d.ctrl[self.pulls]=u[:5]
            d.mocap_pos[self.wrist,2]=.315
            mj.mj_step(m,d,nstep=SUB)
        self.k+=1;rel=self.relative();forces,count,total,points=self.contacts(record)
        z=float(d.xpos[self.oid,2]);n=int(np.sum(forces>.005));dist=float(np.linalg.norm(rel-self.release_center))
        if self.k>=50:
            self.minheight=min(self.minheight,z);self.maxdrift=max(self.maxdrift,dist);self.mindigits=min(self.mindigits,n);self.hold_steps+=1
        # Dense shaping only after object is free. Before release favor gentle multi-digit contact.
        if self.k<=25:
            reward=.025*n+.025*float(forces[-1]>.005)-.003*max(0,total-12)
        else:
            height=float(np.clip((z-(self.initial_height-.045))/.045,0,1))
            stable=math.exp(-dist/.020)
            reward=.45*height+.55*stable+.10*min(n,4)/4+.07*float(forces[-1]>.005)-.008*max(0,total-12)
        reward-=.012*float(np.mean(raw**2))
        bad=not np.isfinite(d.qpos).all()
        dropped=(self.k>30 and (z<.09 or dist>.15))
        done=bool(bad or dropped or self.k>=self.horizon)
        success=bool(not bad and not dropped and self.k>=self.horizon and self.mindigits>=2 and self.maxdrift<.025 and self.minheight>self.initial_height+.035 and not d.eq_active[self.eq])
        if done:reward+=8 if success else -5
        self.ep_return+=reward
        info=dict(success=success,return_=self.ep_return,steps=self.k,dropped=bool(dropped),bad=bool(bad),minHoldDigits=self.mindigits if self.hold_steps else 0,maxHoldDrift=self.maxdrift,minHoldHeight=self.minheight if self.hold_steps else z,finalHeight=z,fixture=bool(d.eq_active[self.eq]))
        if record:
            info['frame']=dict(t=self.k*DT,wrist=d.xpos[self.wid].tolist(),object=d.xpos[self.oid].tolist(),objectLocal=rel.tolist(),objectQuat=d.xquat[self.oid].tolist(),angles=np.degrees(d.qpos[self.qj[:,1:]]).tolist(),yaw=np.degrees(d.qpos[self.qj[:,0]]).tolist(),tendonLength=(1000*d.ten_length).tolist(),tendonForce=d.actuator_force[self.pulls].tolist(),contactCount=count,digits=[NAMES[i] for i in range(5) if forces[i]>.005],normalForce=total,contacts=points,fixture=bool(d.eq_active[self.eq]),action=raw.tolist())
        return self.obs(forces),float(reward),done,info
