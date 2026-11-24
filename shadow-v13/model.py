"""Pinned E3M5 hand, unchanged joint/actuator assets; fixed palm-down forearm, free 220mm rod."""
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np,mujoco as mj
from scipy.spatial.transform import Rotation
ASSETS=Path(__file__).resolve().parent.parent/'shadow-v12/.assets/shadow_hand'
R=.013;H=.110;MASS=.055
def make_model():
    root=ET.parse(ASSETS/'right_hand.xml').getroot()
    root.find('compiler').set('meshdir',str(ASSETS/'assets'))
    world=root.find('worldbody');fore=world.find("body[@name='rh_forearm']")
    old=np.array([float(x) for x in fore.get('quat').split()]);old/=np.linalg.norm(old)
    rot=Rotation.from_rotvec([np.pi,0,0])*Rotation.from_quat(old[[1,2,3,0]])
    fore.set('quat',' '.join(map(str,rot.as_quat()[[3,0,1,2]])));fore.set('pos','0 0 .35')
    ET.SubElement(world,'geom',name='floor',type='plane',size='1 1 .05',rgba='.8 .85 .85 1')
    rod=ET.SubElement(world,'body',name='rod',pos='.3 0 .2')
    ET.SubElement(rod,'freejoint',name='rod_free')
    ET.SubElement(rod,'geom',name='shaft',type='cylinder',size=f'{R} {H}',mass=str(MASS),rgba='.55 .78 .94 1',condim='4',friction='1.1 .004 .0002',solref='.005 1',solimp='.90 .98 .001')
    m=mj.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    return m
def ids(m):
    return dict(palm=m.body('rh_palm').id,rod=m.body('rod').id,shaft=m.geom('shaft').id,floor=m.geom('floor').id,q=int(m.jnt_qposadr[m.joint('rod_free').id]),v=int(m.jnt_dofadr[m.joint('rod_free').id]))
def digitmap(m):
    a=np.full(m.ngeom,-1,np.int32)
    for i in range(m.ngeom):
        name=m.body(int(m.geom_bodyid[i])).name
        for j,s in enumerate(['rh_ff','rh_mf','rh_rf','rh_lf','rh_th']):
            if name.startswith(s):a[i]=j
    return a
def pose(m,d,center,tilt=0.):
    j=ids(m);mj.mj_forward(m,d);P=d.xpos[j['palm']].copy();A=d.xmat[j['palm']].reshape(3,3).copy()
    # Local rod z -> palm x; tilt rotates in palm x-z plane.
    Q=Rotation.from_matrix(A)*Rotation.from_euler('y',np.pi/2+tilt)
    return np.r_[P+A@center,Q.as_quat()[[3,0,1,2]]]
def metrics(m,d,j,mapping):
    ns=np.zeros((2,3));loads=np.zeros(2);digitforces=np.zeros(5);digits=set();sd=od=0.;buf=np.zeros(6);axis=d.xmat[j['rod']].reshape(3,3)[:,2]
    for k in range(d.ncon):
        c=d.contact[k];a,b=int(c.geom1),int(c.geom2)
        if j['floor'] in (a,b):continue
        dep=max(0.,-float(c.dist))
        if j['shaft'] not in (a,b):sd=max(sd,dep);continue
        od=max(od,dep);other=b if a==j['shaft'] else a;g=mapping[other]
        if g<0:continue
        mj.mj_contactForce(m,d,k,buf);f=max(0.,float(buf[0]))
        digitforces[g]+=f
        if f<.005:continue
        n=c.frame[:3]*(1 if a==j['shaft'] else -1);n=n-axis*np.dot(n,axis)
        group=int(g==4);ns[group]+=f*n;loads[group]+=f*np.linalg.norm(n)
        if g<4:digits.add(int(g))
    lens=np.linalg.norm(ns,axis=1);cos=float(np.dot(ns[0],ns[1])/max(1e-9,np.prod(lens))) if min(lens)>.02 else 1.
    return dict(opposed=bool(cos<-.35 and min(loads)>.1 and len(digits)>=2),cos=cos,thumb=float(loads[1]),fingers=float(loads[0]),digits=len(digits),forces=digitforces.tolist(),selfPen=sd,objPen=od)
