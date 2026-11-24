import json
from pathlib import Path
from cpu_eval_rod import evaluate
d=json.load(open('runs/action-search--1-15deg-scale2.json'))
row=d['rows'][0];action=row['action'];print('SEARCH BEST',row,flush=True)
seeds=list(range(887000,887048));angles=[-15.]*len(seeds)
base,br,bf=evaluate(None,seeds,angles,difficulty=.35,scale=2.,record=True)
ctrl,cr,cf=evaluate(None,seeds,angles,difficulty=.35,scale=2.,fixed_action=action,record=True)
pairs=[i for i,(a,b) in enumerate(zip(br,cr)) if b['success'] and not a['success']]
if not pairs:pairs=[i for i,b in enumerate(cr) if b['success']]
if not pairs:raise RuntimeError('no successful representative pair')
i=pairs[0]
out=Path('runs/demo');out.mkdir(exist_ok=True)
for name,summary,frames in [('fixed',br[i],bf[i]),('rotated',cr[i],cf[i])]:
 (out/(name+'.json')).write_text(json.dumps(dict(summary=summary,frames=frames),separators=(',',':')))
result=dict(seed=seeds[i],index=i,baseline=base,controller=ctrl,chosenFixed=br[i],chosenRotated=cr[i],action=action,scale=2.,target=-15.,pairCount=len(pairs))
(out/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('chosenFixed','chosenRotated')},indent=2))
print('FIXED',json.dumps(br[i]));print('ROTATED',json.dumps(cr[i]))
