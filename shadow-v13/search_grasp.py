"""Search free-body grasp. Temporary positioning only during closing; release before scored hold."""
import json,math,concurrent.futures as cf,argparse,time
from pathlib import Path
import numpy as np,mujoco as mj
from model import make_model,ids,digitmap,pose,metrics
M=None;D=None;J=None;MAP=None
def init():
 global M,D,J,MAP
 M=make_model();D=mj.MjData(M);J=ids(M);MAP=digitmap(M)
def trial(c,seed=0,difficulty=0.,record=False):
 if M is None:init()
 m,d,j=M,D,J
 mj.mj_resetData(m,d)
 rng=np.random.default_rng(seed)
 center=np.array(c['center'])+rng.uniform(-.003,.003,3)*difficulty
 pos=pose(m,d,center,c['tilt']+rng.uniform(-.04,.04)*difficulty)
 q,v=j['q'],j['v'];d.qpos[q:q+7]=pos;d.ctrl[:]=0.;mj.mj_forward(m,d)
 target=np.asarray(c['ctrl']);target=np.clip(target,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
 opp=0;num=0;cos=[];sd=od=drift=0.;approachSD=approachOD=0.;frames=[];dropped=False;initial=pos[:3].copy()
 for k in range(850):
    if k<350:
       a=min(1.,k/240);a=a*a*(3-2*a)
       d.ctrl[:]=target*a;d.qpos[q:q+7]=pos;d.qvel[v:v+6]=0.
       mj.mj_forward(m,d)
    mj.mj_step(m,d)
    if k%10==9:
      r=metrics(m,d,j,MAP)
      if k<350:approachSD=max(approachSD,r['selfPen']);approachOD=max(approachOD,r['objPen'])
      else:sd=max(sd,r['selfPen']);od=max(od,r['objPen'])
      if k>=500:
       num+=1;opp+=r['opposed'];cos.append(r['cos'])
       drift=max(drift,float(np.linalg.norm(d.qpos[q:q+3]-initial)))
      if record:frames.append(dict(t=(k+1)*.002,qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),ctrl=d.ctrl.tolist(),metrics=r,fixture=k<350))
    if not np.isfinite(d.qpos).all() or (k>=350 and (np.linalg.norm(d.qpos[q:q+3]-initial)>.10 or d.qpos[q+2]<.1)):
       dropped=True;break
 fraction=opp/max(num,1);retained=not dropped and num==35 and drift<.025
 success=bool(retained and fraction>=.9 and sd<.0015 and od<.003)
 score=4*retained+4*fraction+max(0,-np.mean(cos) if cos else 0)-3*min(sd/.003,1)-2*min(od/.006,1)-min(drift/.05,1)
 result=dict(success=success,retained=bool(retained),opposedFraction=fraction,meanCos=float(np.mean(cos)) if cos else 1.,selfMM=sd*1000,objectMM=od*1000,driftMM=drift*1000,score=float(score),approachSelfMM=1000*approachSD,approachObjectMM=1000*approachOD)
 if record:result.update(qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),ctrl=d.ctrl.tolist(),frames=frames)
 return result
def work(item):
 i,c=item;return dict(id=i,config=c,**trial(c))
def candidates(n):
 rng=np.random.default_rng(13013)
 for i in range(n):
    c=np.zeros(20);c[2:7]=[rng.uniform(-1.,1.),rng.uniform(.1,1.15),rng.uniform(-.18,.18),rng.uniform(-.65,.65),rng.uniform(.05,1.5)]
    for j in [7,10,13,17]:
      c[j]=rng.uniform(-.16,.16);c[j+1]=rng.uniform(.55,1.45);c[j+2]=rng.uniform(.65,2.8)
    c[16]=rng.uniform(0,.45)
    if i%2==0:
      c[[8,11,14,18]]=rng.uniform(.7,1.4);c[[9,12,15,19]]=rng.uniform(.7,2.5)
    yield dict(center=[float(rng.uniform(-.018,.022)),float(rng.uniform(-.060,-.026)),float(rng.uniform(.083,.128))],tilt=float(rng.uniform(-.18,.18)),ctrl=c.tolist())
def write(p,x):
 q=Path(p);tmp=q.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(q)
def main():
 p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=4000);p.add_argument('--workers',type=int,default=12);a=p.parse_args()
 rows=[];t=time.perf_counter()
 with cf.ProcessPoolExecutor(max_workers=a.workers,initializer=init) as ex:
    for r in ex.map(work,enumerate(candidates(a.n)),chunksize=2):
       rows.append(r)
       if len(rows)%100==0:print('SEARCH',len(rows),'success',sum(x['success'] for x in rows),'best',max(x['score'] for x in rows),'sec',time.perf_counter()-t,flush=True)
 rows.sort(key=lambda x:(x['success'],x['score']),reverse=True);write('runs/search.json',rows)
 cal=[]
 for x in rows[:16]:
    rs=[trial(x['config'],seed=131000+i,difficulty=1.) for i in range(8)]
    cal.append(dict(id=x['id'],config=x['config'],success=sum(r['success'] for r in rs),retained=sum(r['retained'] for r in rs),opposition=float(np.mean([r['opposedFraction'] for r in rs])),trials=rs))
 cal.sort(key=lambda x:(x['success'],x['retained'],x['opposition']),reverse=True);write('runs/calibration.json',cal)
 best=cal[0];print('BEST',best,flush=True)
 if best['success']<6:raise RuntimeError('No adequate opposed grasp. Refuse to train invalid initial states.')
 write('grasp.json',best['config'])
 bank=[];blind=[]
 for i in range(96):
    r=trial(best['config'],seed=132000+i,difficulty=1.,record=True);r.pop('frames')
    if r['success']:bank.append(dict(qpos=r['qpos'],qvel=r['qvel'],ctrl=r['ctrl']))
    blind.append({k:v for k,v in r.items() if k not in ('qpos','qvel','ctrl')})
 write('runs/grasp-holdout.json',blind)
 print('HOLDOUT',len(bank),'of',len(blind),flush=True)
 if len(bank)<80:raise RuntimeError('Grasp holdout gate failed')
 # Train bank uses separately generated seeds, never blind states.
 bank=[]
 for i in range(256):
    r=trial(best['config'],seed=133000+i,difficulty=1.,record=True)
    if r['success']:bank.append(dict(qpos=r['qpos'],qvel=r['qvel'],ctrl=r['ctrl']))
 assert len(bank)>=200
 write('runs/reset-bank.json',bank);print('RESET_BANK',len(bank),flush=True)
if __name__=='__main__':main()
