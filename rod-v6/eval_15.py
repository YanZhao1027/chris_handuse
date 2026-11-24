import json
from pathlib import Path
from cpu_eval_rod import evaluate
d=json.load(open('runs/action-search--1-15deg-scale2.json'))
row=next(r for r in d['rows'] if r['candidate']==861);a=row['action']
seeds=list(range(885000,885096));angles=[-15.]*len(seeds)
s,r,_=evaluate(None,seeds,angles,difficulty=.35,scale=2.,fixed_action=a)
print('CPU BLIND ROD -15 DEG',json.dumps(s),flush=True)
print('yaw mean',sum(x['objectYawDegrees'] for x in r)/len(r),'target',-15,'mean error',sum(x['errorDegrees'] for x in r)/len(r))
print('success',sum(x['success'] for x in r),'/',len(r),'retained',sum(x['retained'] for x in r),'/',len(r))
Path('runs/cpu-rod15-blind.json').write_text(json.dumps(dict(summary=s,trials=r,action=a,scale=2.,target=-15),indent=2))
