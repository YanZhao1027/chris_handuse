import json,math
from pathlib import Path
import numpy as np,torch,mujoco as mj
from cpu_eval_rod import evaluate,mul,CPURod
from rod_shape import audit_pairs
torch.set_num_threads(1)
teachers=json.loads(Path('runs/teachers.json').read_text())
data={};audit=[]
for g,seed,hkey,ckey in [(-15,921000,'fixed','rotated'),(15,921001,'positiveHold','positive')]:
 for name,a in [(hkey,None),(ckey,teachers[str(g)][0]['action'])]:
  _,r,f=evaluate(None,[seed],[g],difficulty=.35,scale=2.,fixed_action=a,record=True)
  frames=f[0];summary=r[0]
  h=CPURod(seed,g,.35,2.).h;m=h.m;d=h.d
  pairs=audit_pairs(m);worst=0.;rodworst=0.
  for x in frames:
   d.qpos[h.qj[:,0]]=np.radians(x['yaw']);d.qpos[h.qj[:,1:]]=np.radians(x['angles'])
   d.qpos[h.obj_q:h.obj_q+3]=x['object'];d.qpos[h.obj_q+3:h.obj_q+7]=x['objectQuat'];d.mocap_pos[h.wrist]=x['wrist']
   mj.mj_forward(m,d)
   worst=max(worst,-min(mj.mj_geomDistance(m,d,a,b,.2,None) for a,b in pairs))
   for i in list(h.geom_digit)+[m.geom('palm').id,m.geom('drive').id]:
    rodworst=max(rodworst,-mj.mj_geomDistance(m,d,i,h.ball,.2,None))
  q0=np.array(frames[min(49,len(frames)-1)]['objectQuat']);q1=np.array(frames[-1]['objectQuat']);q=mul(q1,np.r_[q0[0],-q0[1:]])
  if q[0]<0:q=-q
  sn=np.linalg.norm(q[1:]);ang=math.degrees(2*math.atan2(sn,abs(q[0])));axis=q[1:]/max(sn,1e-9)
  summary['actualRotation']=dict(angleDegrees=ang,axis=axis.tolist(),zComponentDegrees=ang*float(axis[2]))
  data[name]=dict(summary=summary,frames=frames)
  audit.append(dict(name=name,seed=seed,target=g,selfGeometryPenetrationMM=worst*1000,rodGeometryPenetrationMM=rodworst*1000,retained=summary['retained'],success=summary['success'],error=summary['errorDegrees']))
Path('tendonhand-v9-preview/recordings.json').write_text(json.dumps(data,separators=(',',':')))
Path('runs/preview-audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
