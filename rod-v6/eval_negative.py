import json
from pathlib import Path
from cpu_eval_rod import evaluate
action=json.load(open('runs/action-search-10deg-scale2.json'))['rows'][0]['action']
seeds=list(range(883000,883096));angles=[-10.]*len(seeds)
s,r,_=evaluate(None,seeds,angles,difficulty=.35,scale=2.,fixed_action=action)
print('CPU BLIND NEGATIVE 10 DEG',json.dumps(s),flush=True)
print('per-trial successes',sum(x['success'] for x in r),'/',len(r),'retained',sum(x['retained'] for x in r))
print('mean final rod yaw',sum(x['objectYawDegrees'] for x in r)/len(r))
Path('runs/cpu-negative10-blind.json').write_text(json.dumps(dict(summary=s,trials=r,action=action,scale=2.),indent=2))
