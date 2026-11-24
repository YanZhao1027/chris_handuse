"""Mechanical design search before RL: CPU MuJoCo, free-body grasp, real contacts."""
import argparse,concurrent.futures as cf,json,math,time
from pathlib import Path
import numpy as np,mujoco as mj
from rod_shape import RodHand,DEFAULT_MOUNT,mount_quat,audit_pairs
from opposition import metrics

def trial(c,seed=1104000,difficulty=0.,record=False):
    h=RodHand(c);h.reset(seed,difficulty,horizon=100)
    m=h.m;d=h.d;p=h.params
    start=np.radians([8,3,-3,-10,c['start']]);target=np.radians([2,1,-1,-3,c['yaw']])
    d.qpos[h.qj[:,0]]=start;d.ctrl[h.yaws]=start
    mj.mj_forward(m,d)
    pairs=audit_pairs(m)
    initial=min(mj.mj_geomDistance(m,d,a,b,.2,None) for a,b in pairs)
    if initial<-.001:
        return dict(success=False,retained=False,opposedFraction=0.,maxSelfMM=-initial*1000,maxObjectMM=0.,driftMM=999.,meanCos=1.,thumbForce=0.,fingerForce=0.,initialMM=initial*1000,score=-100.,frames=[])
    pulls=np.array(c['pulls']);z0=float(d.qpos[h.obj_q+2]);startrel=np.array(p['center'])
    count=opp=0;cos=[];th=[];fi=[];depth=0.;objdepth=0.;drift=0.;minz=1.;frames=[];drop=False
    for k in range(100):
        if k==25:d.eq_active[h.eq]=0;startrel=h.relative().copy()
        for j in range(40):
            t=(40*k+j)*.001;a=np.clip(t/.7,0,1);a=a*a*(3-2*a);l=np.clip((t-1.1)/.85,0,1);l=l*l*(3-2*l)
            d.ctrl[h.yaws]=(1-a)*start+a*target;d.ctrl[h.pulls]=a*pulls;d.mocap_pos[h.wrist,2]=.25+.065*l
            mj.mj_step(m,d)
        v=metrics(h);depth=max(depth,v['selfPen']);objdepth=max(objdepth,v['objectPen'])
        z=float(d.qpos[h.obj_q+2])
        rel=(d.qpos[h.obj_q:h.obj_q+3]-d.mocap_pos[h.wrist])*np.array([-1.,1.,-1.])
        delta=float(np.linalg.norm(rel-startrel))
        if k>=50:
            minz=min(minz,z);drift=max(drift,delta);count+=1;opp+=v['opposed'];cos.append(v['cos']);th.append(v['thumbForce']);fi.append(v['fingerForce'])
        if record:
            mj.mj_forward(m,d);f,n,total,contacts=h.contacts(True)
            frames.append(dict(t=(k+1)*.04,wrist=d.mocap_pos[h.wrist].tolist(),object=d.qpos[h.obj_q:h.obj_q+3].tolist(),objectQuat=d.qpos[h.obj_q+3:h.obj_q+7].tolist(),targetQuat=d.qpos[h.obj_q+3:h.obj_q+7].tolist(),errorDegrees=0.,targetDegrees=0.,angularSpeed=float(np.linalg.norm(d.qvel[h.obj_v+3:h.obj_v+6])),angles=np.degrees(d.qpos[h.qj[:,1:]]).tolist(),yaw=np.degrees(d.qpos[h.qj[:,0]]).tolist(),tendonForce=d.actuator_force[h.pulls].tolist(),tendonLength=(1000*d.ten_length).tolist(),contactCount=n,digits=[name for name,force in zip(['index','middle','ring','little','thumb'],f) if force>.005],contacts=contacts,normalForce=total,fixture=bool(d.eq_active[h.eq]),selfPenetrationMM=1000*depth,objectPenetrationMM=1000*objdepth,opposition=v,streak=0))
        if not np.isfinite(d.qpos).all() or (k>30 and (z<.09 or delta>.12)):drop=True;break
    fraction=opp/max(1,count)
    retained=not drop and count==50 and minz>z0+.035 and drift<.03
    safe=depth<.001 and objdepth<.002
    success=bool(retained and safe and fraction>=.90 and v['opposed'])
    return dict(success=success,retained=bool(retained),opposedFraction=float(fraction),maxSelfMM=float(depth*1000),maxObjectMM=float(objdepth*1000),driftMM=1000*drift,meanCos=float(np.mean(cos)) if cos else 1.,thumbForce=float(np.mean(th)) if th else 0.,fingerForce=float(np.mean(fi)) if fi else 0.,initialMM=initial*1000,score=(3*retained+4*fraction+max(0.,-np.mean(cos) if cos else 0.)-4*min(1.,depth/.002)-2*min(1.,objdepth/.004)-min(1.,drift/.05)),frames=frames)

def work(item):
    i,c=item
    r=trial(c);r.pop('frames',None);return dict(id=i,mount=c,**r)
def write(path,x):
    p=Path(path);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=1200);ap.add_argument('--workers',type=int,default=12);a=ap.parse_args()
    rng=np.random.default_rng(1010)
    cands=[json.loads(Path("mount-v10-reference.json").read_text())]
    for i in range(a.n-1):
        c=dict(x=float(rng.uniform(-.065,-.042)),y=float(rng.uniform(-.039,-.010)),z=float(rng.uniform(.004,.023)),tilt=float(rng.uniform(-40,35)),twist=float(rng.uniform(-40,35)),start=float(rng.uniform(-75,-15)),yaw=float(rng.uniform(-15,65)),pulls=[float(rng.uniform(5,14))]*4+[float(rng.uniform(2,14))],center=[float(rng.uniform(-.012,.012)),float(rng.uniform(.010,.033)),float(rng.uniform(.032,.050))])
        cands.append(c)
    rows=[];t=time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(work,enumerate(cands),chunksize=1):
            rows.append(r)
            if len(rows)%40==0:print('MOUNT_SEARCH',len(rows),'successes',sum(x['success'] for x in rows),'seconds',time.perf_counter()-t,flush=True)
    rows.sort(key=lambda r:(r['success'],r['score']),reverse=True)
    write('runs/mount-search.json',rows)
    cal=[]
    for r in rows[:12]:
        trials=[trial(r['mount'],seed=1105000+j,difficulty=.35) for j in range(8)]
        for x in trials:x.pop('frames',None)
        cal.append(dict(id=r['id'],mount=r['mount'],success=sum(x['success'] for x in trials),retained=sum(x['retained'] for x in trials),opposition=float(np.mean([x['opposedFraction'] for x in trials])),maxSelfMM=max(x['maxSelfMM'] for x in trials),maxObjectMM=max(x['maxObjectMM'] for x in trials),trials=trials))
    cal.sort(key=lambda r:(r['success'],r['retained'],r['opposition'],-r['maxSelfMM'],-r['maxObjectMM']),reverse=True)
    write('runs/mount-calibration.json',cal)
    best=cal[0];print('BEST_MOUNT',json.dumps({k:v for k,v in best.items() if k!='trials'}),flush=True)
    if best['success']<6:raise RuntimeError('No adequately opposed grasp; no RL on invalid geometry')
    write('mount.json',best['mount'])
    blind=[trial(best['mount'],seed=1106000+j,difficulty=.35) for j in range(32)]
    for x in blind:x.pop('frames',None)
    write('runs/mount-holdout.json',blind)
    print('MOUNT_HOLDOUT',sum(x['success'] for x in blind),'/',len(blind),'retained',sum(x['retained'] for x in blind),flush=True)
    if sum(x['success'] for x in blind)<24:raise RuntimeError('Opposition holdout gate failed')
if __name__=='__main__':main()
