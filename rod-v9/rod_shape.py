"""V9 cylinder and collision-aware five-digit hand."""
import os
from xml.etree.ElementTree import fromstring,tostring,SubElement
import numpy as np
import mujoco as mj
from env import Hand,CFG,START_YAW,domain
from model import make_model
THUMB_X=float(os.getenv('THUMB_X','-.049'))
THUMB_Y=float(os.getenv('THUMB_Y','-.021'))
THUMB_Z=float(os.getenv('THUMB_Z','.008'))
NAMES=['index','middle','ring','little','thumb']
def collision_hand(root):
    palm=root.find(".//body[@name='wrist']")
    palm.find("body[@name='thumb_yaw']").set('pos',f'{THUMB_X} {THUMB_Y} {THUMB_Z}')
    bits=[8,16,32,64,128]
    allbits=sum(bits)
    for n,bit in zip(NAMES,bits):
        for j in range(3):
            g=root.find(f".//geom[@name='{n}_segment{j}']")
            g.set('contype',str(1|bit))
            g.set('conaffinity',str(2|(allbits^bit)|(256 if j>0 or n=="thumb" else 0)))
            g.set('solref','.0025 1');g.set('solimp','.97 .995 .001')
    for n in ['palm','drive']:
        g=root.find(f".//geom[@name='{n}']")
        g.set('contype','257');g.set('conaffinity','2')
        g.set('solref','.0025 1');g.set('solimp','.97 .995 .001')
    c=SubElement(root,'contact')
    for n in NAMES:
        SubElement(c,'pair',geom1=f'{n}_segment0',geom2=f'{n}_segment2',condim='3',friction='.5 .5 .008 .0005 .0005',solref='.0025 1',solimp='.97 .995 .001')
    return root

def audit_pairs(model):
    pairs=[]
    for i,n in enumerate(NAMES):
        for m in NAMES[i+1:]:
            for j in range(3):
                for k in range(3):pairs.append((f'{n}_segment{j}',f'{m}_segment{k}'))
        pairs.append((f'{n}_segment0',f'{n}_segment2'))
        for j in ([0,1,2] if n=="thumb" else [1,2]):
            for p in ['palm','drive']:pairs.append((f'{n}_segment{j}',p))
    return [(model.geom(a).id,model.geom(b).id) for a,b in pairs]

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
        root=collision_hand(root)
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
