import json
from pathlib import Path
root=Path(__file__).parent/'runs'
for stage in ('sphere01','ellipsoid01'):
 p=root/stage;f=p/'history.json'
 print('\n'+stage)
 if not f.exists(): print('Not started / no completed update');continue
 h=json.loads(f.read_text());c=json.loads((p/'config.json').read_text());r=h[-1]
 print(f"Update {r['update']}/{c['updates']} | {r['steps']:,}/{c['worlds']*c['rollout']*c['updates']:,} transitions | {r['stepsPerSecond']:,.0f}/s")
 print(f"Last rollout: grasped={r['trainGrasped']} retained={r['trainRetained']} goal success={r['trainSuccess']} error={r['retainedMeanError']}deg")
 print('Frozen V4 baseline',c['baseline'])
 for x in h:
  if 'validation' in x:
   v=x['validation'];print(f"  update {x['update']:3d}  goal {v['success']:6.1%}  retained {v['retained']:6.1%}  error {v['meanErrorRetained']}deg")
f=root/'assessment/blind.json'
if f.exists():
 print('\nBlind test')
 for x in json.loads(f.read_text())['results']:
  print(x['shape'],x['difficulty'],{k:round(v['success'],4) for k,v in x['score'].items()})
