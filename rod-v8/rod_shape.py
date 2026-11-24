"""Flat-ended cylinder, long axis in local X."""
import os
from xml.etree.ElementTree import fromstring,tostring,SubElement
import numpy as np
import mujoco as mj
from env import Hand,CFG,START_YAW,domain
from model import make_model
R=float(os.getenv('ROD_R','.013'))
H=float(os.getenv('ROD_H','.055'))
def inertia(m,r,h):
    return np.array([.5*m*r*r,m*(3*r*r+4*h*h)/12,m*(3*r*r+4*h*h)/12])
def rod_params(seed,difficulty=1.):
    p=domain(seed,difficulty)
    rng=np.random.default_rng(seed+17761)
    r=R+rng.uniform(-.001,.001)*difficulty
    h=H+rng.uniform(-.003,.003)*difficulty
    p.update(radius=r,cylinder_half_length=h,half_length=h,axes=[h,r,r],mass=.055*(1+rng.uniform(-.25,.35)*difficulty),friction=1.0*(1+rng.uniform(-.22,.30)*difficulty))
    return p
class RodHand(Hand):
    def __init__(self):
        super().__init__()
        root=fromstring(make_model(**{k:CFG[k] for k in ['center','radius','mass','friction']}))
        b=root.find(".//body[@name='object']")
        g=b.find("geom[@name='ball']")
        g.set('type','cylinder');g.set('size',f'{R} {H}');g.set('quat','0.7071067811865476 0 0.7071067811865476 0');g.set('mass','.055');g.set('rgba','.48 .73 .92 1')
        SubElement(b,'inertial',pos='0 0 0',quat='1 0 0 0',mass='.055',diaginertia=' '.join(map(str,inertia(.055,R,H))))
        self.m=mj.MjModel.from_xml_string(tostring(root,encoding='unicode'))
        self.d=mj.MjData(self.m)
    def reset(self,seed,difficulty=1.,horizon=100,params=None):
        p=rod_params(seed,difficulty) if params is None else dict(params)
        self.params=p
        m=self.m;d=self.d;r=p['radius'];h=p['cylinder_half_length']
        m.geom_size[self.ball]=[r,h,0]
        m.geom_rbound[self.ball]=np.hypot(h,r)
        # AABB lives in GEOM frame, whose local Z is the long axis.
        m.geom_aabb[self.ball]=[0,0,0,r,r,h]
        m.body_mass[self.oid]=p['mass']
        m.body_inertia[self.oid]=inertia(p['mass'],r,h)
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
