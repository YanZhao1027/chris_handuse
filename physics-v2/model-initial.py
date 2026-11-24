"""TENDON/01 V2: original dimensions, fixed-tendon moment arms, rigid bodies."""
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, indent, tostring
import json, math
import numpy as np
import mujoco as mj

# Source dimensions are exported directly from V1 kin.js, in mm / degrees.
PARAMS=json.loads((Path(__file__).parent/'dimensions.json').read_text())
NAMES=[p['id'] for p in PARAMS]

def E(parent,tag,**attrs):
    return SubElement(parent,tag,{k:str(v) for k,v in attrs.items()})
def fmt(v):
    return ' '.join(f'{x:.8g}' for x in v)

def make_model(center=(0,.056,.049),radius=.032,mass=.045,friction=1.15):
    root=Element('mujoco',model='TENDON_01_physics')
    E(root,'compiler',angle='degree',autolimits='true')
    E(root,'option',timestep='.001',gravity='0 0 -9.81',integrator='implicitfast',solver='Newton',iterations='80',cone='elliptic')
    E(root,'size',njmax='2000',nconmax='600')
    visual=E(root,'visual'); E(visual,'global',offwidth='1280',offheight='720'); E(visual,'quality',shadowsize='2048')
    d=E(root,'default')
    E(d,'joint',damping='.002',armature='.000015',limited='true')
    E(d,'geom',friction=f'{friction} .008 .0005',condim='4',solref='.006 1',solimp='.92 .98 .002',density='1100',contype='1',conaffinity='2',rgba='.64 .72 .75 1')
    world=E(root,'worldbody')
    E(world,'light',pos='0 -.35 .6',dir='0 .5 -1',diffuse='.9 .9 .9',castshadow='true')
    E(world,'light',pos='.4 .4 .5',dir='-1 -1 -1',diffuse='.6 .65 .7')
    E(world,'geom',name='floor',type='plane',size='1 1 .02',contype='4',conaffinity='2',rgba='.18 .23 .26 1')
    # Mocap base acts as a prescribed robot wrist. Palm faces down, so gravity pulls the object OUT.
    palm=E(world,'body',name='wrist',mocap='true',pos='0 0 .25',quat='0 0 1 0')
    E(palm,'geom',name='palm',type='box',size='.0395 .038 .010',pos='0 -.009 0',rgba='.35 .43 .49 1')
    E(palm,'geom',name='drive',type='box',size='.0375 .021 .012',pos='0 -.069 0',rgba='.22 .29 .33 1')
    E(palm,'site',name='palm_frame',pos='0 0 0',size='.002',rgba='1 1 1 0')
    ten=E(root,'tendon'); act=E(root,'actuator')
    for f in PARAMS:
        name=f['id']; thumb=name=='thumb'
        yaw=E(palm,'body',name=name+'_yaw',pos=fmt([f['x']/1000,f['y']/1000,0]))
        E(yaw,'joint',name=name+'_yaw',axis='0 0 1',range='-40 65' if thumb else '-22 22',stiffness='0',damping='.004')
        E(yaw,'geom',name=name+'_base',type='sphere',size='.005',mass='.008',contype='0',conaffinity='0')
        E(act,'position',name=name+'_yaw_servo',joint=name+'_yaw',kp='.18',kv='.012',ctrlrange='-.698132 1.134464' if thumb else '-.383972 .383972',forcerange='-.12 .12')
        tendon=E(ten,'fixed',name=name+'_flexor',limited='false')
        # Positive fixed-tendon coordinate is cable take-up, not actual physical routing.
        parent=yaw
        for j,(L,r,limit) in enumerate(zip(f['lengths'],f['r'],f['max'])):
            b=E(parent,'body',name=f'{name}_link{j}',pos='0 0 0' if j==0 else f'0 {f["lengths"][j-1]/1000} 0')
            # Spring return and damping are physical modeled joint torques, not prescribed angles.
            k=8*(r/1000)/math.radians(limit)
            E(b,'joint',name=f'{name}_flex{j}',axis='1 0 0',range=f'-2 {limit}',stiffness=f'{k:.8f}',springref='0')
            rad=f['w']/1000*(.43-.045*j)
            E(b,'geom',name=f'{name}_segment{j}',type='capsule',fromto=fmt([0,0,0,0,L/1000,0]),size=f'{rad:.7f}',rgba='.73 .81 .82 1' if j!=2 else '.29 .43 .44 1')
            E(b,'site',name=f'{name}_joint{j}',size='.0017',rgba='0 .75 .68 1')
            E(tendon,'joint',joint=f'{name}_flex{j}',coef=f'{r/1000:.7f}')
            if j==2:E(b,'site',name=name+'_tip',pos=fmt([0,L/1000,0]),size='.0028',rgba='0 .8 .7 1')
            parent=b
        E(act,'motor',name=name+'_pull',tendon=name+'_flexor',gear='1',ctrlrange='0 25',forcerange='0 25')
    # Initial fixture, removed before lift. No support or weld during evaluation.
    wp=np.array([0,0,.25])+np.array([-center[0],center[1],-center[2]])
    fixture=E(world,'body',name='fixture',mocap='true',pos=fmt(wp))
    obj=E(world,'body',name='object',pos=fmt(wp))
    E(obj,'freejoint',name='object_free')
    E(obj,'geom',name='ball',type='sphere',size=radius,mass=mass,friction=f'{friction} .008 .0005',condim='4',contype='2',conaffinity='5',rgba='.89 .46 .12 1')
    eq=E(root,'equality')
    E(eq,'weld',name='fixture_hold',body1='fixture',body2='object',relpose='0 0 0 1 0 0 0',solref='.003 1')
    indent(root)
    return tostring(root,encoding='unicode')

def load(**kwargs):
    return mj.MjModel.from_xml_string(make_model(**kwargs))
