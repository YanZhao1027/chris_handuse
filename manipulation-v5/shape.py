"""Same hand topology, mildly non-spherical rigid bodies. No new constraints."""
import numpy as np
import mujoco as mj
from env import Hand,CFG,START_YAW,domain
from model import make_model

class ShapeHand(Hand):
    def __init__(self):
        super().__init__()
        xml=make_model(**{k:CFG[k] for k in ['center','radius','mass','friction']})
        from xml.etree.ElementTree import fromstring,tostring
        root=fromstring(xml);g=root.find(".//geom[@name='ball']")
        g.set('type','ellipsoid');g.set('size',' '.join([str(CFG['radius'])]*3))
        self.m=mj.MjModel.from_xml_string(tostring(root,encoding='unicode'))
        self.d=mj.MjData(self.m)
    def reset(self,seed,difficulty=1.,horizon=100,params=None):
        p=domain(seed,difficulty) if params is None else dict(params)
        rng=np.random.default_rng(seed+7231)
        aspect=1.+rng.uniform(.06,.18)*difficulty
        axes=np.array(p.get('axes',[p['radius']*aspect,p['radius'],p['radius']/aspect]))
        p['axes']=axes.tolist()
        self.params=p
        m=self.m;d=self.d
        m.geom_size[self.ball]=axes
        m.geom_rbound[self.ball]=max(axes)
        m.geom_aabb[self.ball,:3]=0
        m.geom_aabb[self.ball,3:]=axes
        m.body_mass[self.oid]=p['mass']
        m.body_inertia[self.oid]=.2*p['mass']*np.array([axes[1]**2+axes[2]**2,axes[0]**2+axes[2]**2,axes[0]**2+axes[1]**2])
        m.geom_friction[:,0]=p['friction']
        mj.mj_setConst(m,d);mj.mj_resetData(m,d)
        pos=np.array([0,0,.25])+np.array([-p['center'][0],p['center'][1],-p['center'][2]])
        d.qpos[self.obj_q:self.obj_q+3]=pos
        d.qpos[self.obj_q+3:self.obj_q+7]=[1,0,0,0]
        d.mocap_pos[self.fix]=pos
        d.qpos[self.qj[:,0]]=START_YAW;d.ctrl[self.yaws]=START_YAW
        mj.mj_forward(m,d)
        self.k=0;self.horizon=horizon;self.last=np.zeros(6);self.center=np.array(p['center'])
        self.initial_height=.25-self.center[2];self.release_center=self.center.copy()
        self.minheight=1.;self.maxdrift=0.;self.mindigits=5;self.ep_return=0.;self.hold_steps=0
        self.prev_rel=self.relative()
        return self.obs()
