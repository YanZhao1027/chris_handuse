import json
from pathlib import Path
from cpu_eval_rod import evaluate
d=json.load(open('runs/action-search-10deg-scale2.json'));a=d['rows'][0]['action']
seeds=list(range(881000,881096))
for name,sign in [('same',False),('sign_by_goal',True)]:
 s,r,_=evaluate(None,seeds,fixed_action=a,sign_by_goal=sign,scale=2.)
 print(name,json.dumps(s),flush=True)
 for goal in [-10,10]:
  x=[v for v in r if v['targetDegrees']==goal]
  print(' ',goal,'success',sum(v['success'] for v in x),'/',len(x),'retained',sum(v['retained'] for v in x),'bestErr',sum(v['bestErrorDegrees'] for v in x)/len(x))
 Path('runs/cpu-'+name+'.json').write_text(json.dumps(dict(summary=s,trials=r),indent=2))
