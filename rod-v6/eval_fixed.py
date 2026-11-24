import json
from pathlib import Path
from cpu_eval_rod import evaluate
d=json.load(open('runs/action-search-10deg-scale2.json'))
a=d['rows'][0]['action'];print('candidate',d['rows'][0],flush=True)
seeds=list(range(880000,880096))
for name,act in [('no_action',None),('teacher_363',a)]:
 s,r,_=evaluate(None,seeds,fixed_action=act,scale=2.)
 print(name,json.dumps(s),flush=True)
 Path('runs/cpu-'+name+'.json').write_text(json.dumps(dict(summary=s,trials=r),indent=2))
