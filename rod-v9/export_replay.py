"""Export recorded held-out CPU trajectories, no fabricated motion."""
import json,math
from pathlib import Path
import numpy as np
from cpu_eval_rod import mul
out=Path('tendonhand-rod-v9-live')
data={}
for key,fn in [('fixed','-15-hold'),('rotated','-15-PPO'),('positiveHold','+15-hold'),('positive','+15-PPO')]:
 d=json.loads(Path('runs/assessment',fn+'.json').read_text())
 f=d['frames'];q0=np.array(f[min(49,len(f)-1)]['objectQuat']);q1=np.array(f[-1]['objectQuat'])
 q=mul(q1,np.r_[q0[0],-q0[1:]])
 if q[0]<0:q=-q
 sn=float(np.linalg.norm(q[1:]));angle=2*math.atan2(sn,abs(q[0]))*180/math.pi;axis=q[1:]/max(sn,1e-9)
 d['summary']['actualRotation']=dict(angleDegrees=angle,axis=axis.tolist(),zComponentDegrees=angle*float(axis[2]))
 data[key]=d
(out/'recordings.json').write_text(json.dumps(data,separators=(',',':')))
print('REPLAY_DATA_READY',flush=True)
