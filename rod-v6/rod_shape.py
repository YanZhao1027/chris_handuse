"""Prolate ellipsoid rod across the finger span; smooth contacts and exact inertia."""
from pathlib import Path
from xml.etree.ElementTree import fromstring,tostring
import numpy as np
import mujoco as mj
from env import Hand,CFG,START_YAW,domain
from model import make_model

def rod_params(seed,difficulty=1.):
    p=domain(seed,difficulty)
    rng=np.random.default_rng(seed+17761)
    p['radius']=.018+rng.uniform(-.001,.001)*difficulty
    p['half_length']=.040+rng.uniform(-.003,.003)*difficulty
    p['axes']=[p['half_length'],p['radius'],p['radius']]
    p['mass']=.055*(1+rng.uniform(-.25,.35)*difficulty)
    p['friction']=1.0*(1+rng.uniform(-.22,.30)*difficulty)
    return p

def ellipsoid_inertia(mass,a,b):
    return mass/5*np.array([2*b*b,a*a+b*b,a*a+b*b])

class RodHand(Hand):
    def __init__(self):
        super().__init__()
        xml=make_model(**{k:CFG[k] for k in ['center','radius','mass','friction']})
        root=fromstring(xml)
        g=root.find(".//geom[@name='ball']")
        g.set('type','ellipsoid')
        g.set('size',f"{.040} {.018} {.018}")
        g.set('mass',str(.055))
        g.set('rgba','.89 .46 .12 1')
        self.m=mj.MjModel.from_xml_string(tostring(root,encoding='unicode'))
        self.d=mj.MjData(self.m)
    def reset(self,seed,difficulty=1.,horizon=100,params=None):
        p=rod_params(seed,difficulty) if params is None else dict(params)
        self.params=p
        m=self.m;d=self.d;a=p['half_length'];b=p['radius']
        m.geom_size[self.ball]=[a,b,b]
        m.geom_rbound[self.ball]=a
        m.geom_aabb[self.ball]=[0,0,0,a,b,b]
        m.body_mass[self.oid]=p['mass']
        m.body_inertia[self.oid]=ellipsoid_inertia(p['mass'],a,b)
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
