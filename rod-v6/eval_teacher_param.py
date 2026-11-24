import argparse,json
from pathlib import Path
from cpu_eval_rod import evaluate
p=argparse.ArgumentParser();p.add_argument('--scale',type=float,default=2.);p.add_argument('--angle',type=float,default=-10.);p.add_argument('--n',type=int,default=96);p.add_argument('--seed',type=int,default=884000);a=p.parse_args()
action=json.load(open('runs/action-search-10deg-scale2.json'))['rows'][0]['action']
seeds=list(range(a.seed,a.seed+a.n));angles=[a.angle]*len(seeds)
s,r,_=evaluate(None,seeds,angles,difficulty=.35,scale=a.scale,fixed_action=action)
print('RESULT',json.dumps(dict(scale=a.scale,angle=a.angle,summary=s,meanYaw=sum(x['objectYawDegrees'] for x in r)/len(r),meanError=sum(x['errorDegrees'] for x in r)/len(r),success=sum(x['success'] for x in r),retained=sum(x['retained'] for x in r),n=len(r))),flush=True)
Path(f'runs/cpu-teacher-scale{a.scale:g}-angle{a.angle:g}.json').write_text(json.dumps(dict(summary=s,trials=r,action=action,scale=a.scale,angle=a.angle),indent=2))
