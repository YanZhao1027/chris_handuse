import json,math
from pathlib import Path
p=Path('runs/demo')
def mul(a,b):
 return [a[0]*b[0]-sum(x*y for x,y in zip(a[1:],b[1:])),
 a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
 a[0]*b[2]+a[2]*b[0]+a[3]*b[1]-a[1]*b[3],
 a[0]*b[3]+a[3]*b[0]+a[1]*b[2]-a[2]*b[1]]
def calc(f):
 q0=f[49]['objectQuat'];q1=f[-1]['objectQuat']
 d=mul(q1,[q0[0],-q0[1],-q0[2],-q0[3]])
 if d[0]<0:d=[-v for v in d]
 sn=math.sqrt(sum(v*v for v in d[1:]))
 angle=2*math.atan2(sn,abs(d[0]))
 axis=[v/max(sn,1e-9) for v in d[1:]]
 return dict(angleDegrees=math.degrees(angle),axis=axis,zComponentDegrees=math.degrees(angle)*axis[2],quaternion=d)
summary=json.loads((p/'summary.json').read_text())
rec={}
for name in ['fixed','rotated']:
 d=json.loads((p/(name+'.json')).read_text())
 d['summary']['actualRotation']=calc(d['frames'])
 rec[name]=d
 (p/(name+'.json')).write_text(json.dumps(d,separators=(',',':')))
summary['fixedRotation']=rec['fixed']['summary']['actualRotation']
summary['controlledRotation']=rec['rotated']['summary']['actualRotation']
(p/'summary.json').write_text(json.dumps(summary,indent=2))
(p/'recordings.json').write_text(json.dumps(rec,separators=(',',':')))
print(json.dumps({k:summary[k] for k in ['seed','baseline','controller','fixedRotation','controlledRotation']},indent=2))
