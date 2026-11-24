"""Headless physical grasp and release/lift test, with a no-drive negative control."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import mujoco as mj
from model import load,make_model,NAMES,PARAMS

def run(center=(0,.056,.049),radius=.032,mass=.045,pull=12,thumb=12,oppose=-22,friction=1.15,record=False,duration=3.,release_at=None):
    m=load(center=center,radius=radius,mass=mass,friction=friction)
    d=mj.MjData(m)
    oid=m.body('object').id; wid=m.body('wrist').id
    wrist=m.body_mocapid[wid]; eq=m.equality('fixture_hold').id
    ball=m.geom('ball').id
    yaws=[m.actuator(n+'_yaw_servo').id for n in NAMES]
    pulls=[m.actuator(n+'_pull').id for n in NAMES]
    jointids=[[m.joint(n+'_flex'+str(j)).qposadr[0] for j in range(3)] for n in NAMES]
    yj=[m.joint(n+'_yaw').qposadr[0] for n in NAMES]
    yawtargets=np.radians([2,1,-1,-3,oppose])
    d.qpos[yj]=np.radians([8,3,-3,-10,62])
    d.ctrl[yaws]=d.qpos[yj]
    mj.mj_forward(m,d)
    def contact_info():
        digits=set(); force=0.; count=0; contacts=[]
        for k in range(d.ncon):
            con=d.contact[k]; ids=[con.geom1,con.geom2]
            if ball not in ids:continue
            other=ids[1] if ids[0]==ball else ids[0]
            name=mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,other)
            if name=='floor':continue
            count+=1
            f=np.zeros(6);mj.mj_contactForce(m,d,k,f)
            force+=max(0.,f[0]);contacts.append({'p':con.pos.tolist(),'n':float(max(0.,f[0])),'part':name})
            if name and name.split('_')[0] in NAMES:digits.add(name.split('_')[0])
        return count,sorted(digits),force,contacts
    frames=[]
    initial=None; min_z=1.; max_drift=0.; min_digits=5; release_time=1.0
    for i in range(int(round(duration*1000))+1):
        t=i*.001
        a=min(1,t/.70);a=a*a*(3-2*a)
        d.ctrl[yaws]=(1-a)*np.radians([8,3,-3,-10,62])+a*yawtargets
        d.ctrl[pulls]=a*np.array([pull]*4+[thumb]) if release_at is None or t<release_at else 0.
        if i==1000:
            initial=(d.xmat[wid].reshape(3,3).T@(d.xpos[oid]-d.xpos[wid])).copy()
            d.eq_active[eq]=0
        lift=np.clip((t-1.10)/.85,0,1);lift=lift*lift*(3-2*lift)
        d.mocap_pos[wrist,2]=.25+.065*lift
        mj.mj_step(m,d)
        if not np.isfinite(d.qpos).all():raise RuntimeError('nonfinite state')
        if i%10==0:
            count,digits,force,contacts=contact_info()
            rel=d.xmat[wid].reshape(3,3).T@(d.xpos[oid]-d.xpos[wid])
            if i>=2000 and (release_at is None or t<release_at):
                min_z=min(min_z,float(d.xpos[oid,2]));max_drift=max(max_drift,float(np.linalg.norm(rel-initial)));min_digits=min(min_digits,len(digits))
            if record:
                frames.append(dict(t=t,wrist=d.xpos[wid].tolist(),object=d.xpos[oid].tolist(),objectLocal=rel.tolist(),objectQuat=d.xquat[oid].tolist(),angles=np.degrees(d.qpos[jointids]).tolist(),yaw=np.degrees(d.qpos[yj]).tolist(),tendonLength=(1000*d.ten_length).tolist(),tendonForce=d.actuator_force[pulls].tolist(),contactCount=count,digits=digits,normalForce=force,contacts=contacts,fixture=bool(d.eq_active[eq])))
    count,digits,force,contacts=contact_info()
    result=dict(center=list(center),radius=radius,mass=mass,pull=pull,thumb=thumb,oppose=oppose,friction=friction,duration=duration,releaseAt=release_at,initialHeight=float(.25-center[2]),finalHeight=float(d.xpos[oid,2]),minHoldHeight=min_z,maxHoldDrift=max_drift,minHoldDigits=min_digits,finalDigits=digits,finalNormalForce=force,fixture=bool(d.eq_active[eq]),success=bool(min_digits>=2 and max_drift<.025 and min_z>(.25-center[2]+.035)))
    return result,frames

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--center',nargs=3,type=float,default=[0,.056,.049]);p.add_argument('--radius',type=float,default=.032);p.add_argument('--pull',type=float,default=12);p.add_argument('--thumb',type=float,default=12);p.add_argument('--oppose',type=float,default=-22);p.add_argument('--friction',type=float,default=1.15);p.add_argument('--record',action='store_true');a=p.parse_args()
    args=vars(a);r,f=run(**args);print(json.dumps(r,indent=2))
    if a.record:
        out=Path('results');out.mkdir(exist_ok=True)
        (out/'grasp.json').write_text(json.dumps(dict(summary=r,frames=f),separators=(',',':')))
        (out/'hand.xml').write_text(make_model(center=a.center,radius=a.radius,friction=a.friction))
