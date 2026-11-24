import json
from pathlib import Path
from cpu_eval_rod import evaluate
a=json.load(open('runs/action-search-10deg-scale2.json'))['rows'][0]['action']
seeds=list(range(882000,882048))
s,r,_=evaluate(None,seeds,fixed_action=a,sign_by_goal=True,scale=2.)
print('signed',json.dumps(s),flush=True)
for g in [-10,10]:
 x=[v for v in r if v['targetDegrees']==g]
 print(g,'success',sum(v['success'] for v in x),'/',len(x),'retained',sum(v['retained'] for v in x),'meanFinalErr',sum(v['errorDegrees'] for v in x)/len(x),flush=True)
Path('runs/cpu-sign_by_goal.json').write_text(json.dumps(dict(summary=s,trials=r),indent=2))
